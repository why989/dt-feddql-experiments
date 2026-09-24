from __future__ import annotations

import os
import numpy as np
import pandas as pd
from collections import deque
import random

from algorithm_utils import (
    ACTION_ID,
    ACTION_MAP,
    DEFAULT_REWARD_WEIGHTS,
    calculate_metrics,
    calculate_reward,
    load_task_set,
    prepare_task_dataframe,
    run_policy_on_tasks,
    save_records,
)
from deployment_model import (
    EDGE_ACTION_IDS,
    NUM_ACTIONS,
    NetworkQueueingModel,
)
from digital_twin import evaluate_runtime_trust
from models.fuzzy_classifier import FuzzyTaskClassifier

# Global-norm clip for the SGD step; keeps a single atypical transition from
# throwing the whole network.
GRAD_CLIP_NORM: float = 10.0

# How often the bootstrapping target network is refreshed from the online
# network, counted in environment steps.  The manuscript states that
# FedAvg-on-DQN instability is mitigated "using target networks", which is what
# this implements.
TARGET_SYNC_STEPS: int = 128


class NeuralNetwork:
    """Simple neural network implementation using numpy."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        """Initialize neural network.

        The hidden layers use He initialisation for ReLU units.  The previous
        release scaled *every* weight by 0.01, which drove the hidden
        activations to ~1e-5: the resulting gradients are ~1e-5 as well, so the
        hidden layers never moved and the network degenerated into five output
        biases -- a constant Q-value per action, independent of the state.  That
        is the mechanism behind the released policy collapsing onto a single
        execution location.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        rng = np.random.default_rng(1234)
        self.weights1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = rng.standard_normal((hidden_dim, hidden_dim)) * np.sqrt(2.0 / hidden_dim)
        self.bias2 = np.zeros((1, hidden_dim))
        # Small output layer: the Q-values start near zero and the two hidden
        # layers keep a usable gradient.
        self.weights3 = rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(1.0 / hidden_dim)
        self.bias3 = np.zeros((1, output_dim))
        
        # Learning rate
        self.learning_rate = 0.02
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return np.where(x > 0, 1, 0)
    
    def forward(self, x):
        """Forward pass."""
        # First hidden layer
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        
        # Second hidden layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)
        
        # Output layer
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        return self.z3
    
    def backward(self, x, y, output):
        """Backward pass over a minibatch.

        The gradients are averaged over the batch and a single update is
        applied.  Updating once per sample inside the loop would make the
        effective step size ``batch_size`` times larger than the configured
        learning rate, which is what made the released training oscillate
        instead of settling on a state-dependent policy.
        """
        batch = max(int(np.asarray(x).shape[0]), 1)

        # Calculate error
        error = (output - y) / batch
        
        # Output layer gradients
        d_z3 = error
        d_weights3 = np.dot(self.a2.T, d_z3)
        d_bias3 = np.sum(d_z3, axis=0, keepdims=True)
        
        # Second hidden layer gradients
        d_a2 = np.dot(d_z3, self.weights3.T)
        d_z2 = d_a2 * self.relu_derivative(self.z2)
        d_weights2 = np.dot(self.a1.T, d_z2)
        d_bias2 = np.sum(d_z2, axis=0, keepdims=True)
        
        # First hidden layer gradients
        d_a1 = np.dot(d_z2, self.weights2.T)
        d_z1 = d_a1 * self.relu_derivative(self.z1)
        d_weights1 = np.dot(x.T, d_z1)
        d_bias1 = np.sum(d_z1, axis=0, keepdims=True)

        grads = [d_weights1, d_bias1, d_weights2, d_bias2, d_weights3, d_bias3]
        self.apply_gradients(grads)

    def apply_gradients(self, grads, learning_rate: float | None = None):
        """Apply one (mean) gradient step with global-norm clipping."""

        lr = self.learning_rate if learning_rate is None else float(learning_rate)
        norm = float(np.sqrt(sum(float(np.sum(np.square(g))) for g in grads)))
        scale = 1.0 if norm <= GRAD_CLIP_NORM else GRAD_CLIP_NORM / (norm + 1e-12)
        for param, grad in zip(
            (self.weights1, self.bias1, self.weights2, self.bias2, self.weights3, self.bias3),
            grads,
        ):
            param -= lr * scale * grad
    
    def get_weights(self):
        """Get model weights."""
        return [self.weights1, self.bias1, self.weights2, self.bias2, self.weights3, self.bias3]
    
    def set_weights(self, weights):
        """Set model weights."""
        self.weights1, self.bias1, self.weights2, self.bias2, self.weights3, self.bias3 = weights


class LocalAgent:
    """Local agent for federated learning."""
    
    def __init__(self, state_dim: int, action_dim: int, agent_id: int):
        """Initialize local agent."""
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = NeuralNetwork(state_dim, action_dim)
        # Bootstrapping target network (see TARGET_SYNC_STEPS).
        self.target_model = NeuralNetwork(state_dim, action_dim)
        self.target_model.set_weights(self.model.get_weights())
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.replay_interval = 5
        self.learning_rate = 0.02
        # The optimizer lives on the network; without this the agent-level rate
        # is silently ignored and the network keeps its 1e-3 default.
        self.model.learning_rate = self.learning_rate
        self.target_model.learning_rate = self.learning_rate
        self.gamma = 0.95
        self.epsilon = 1.0
        # Decayed once per federated round.  The schedule has to actually reach
        # an exploitation regime within the 20 aggregation rounds reported in
        # the paper, otherwise the replay buffer stays dominated by random
        # actions and the learned Q-values never separate the execution nodes.
        self.epsilon_decay = 0.88
        self.epsilon_min = 0.05
        # Optional reward-weight override used by the reward-sensitivity sweep.
        self.reward_weights = None
        # 任务队列管理
        from algorithm_utils import TaskQueue
        self.task_queue = TaskQueue(max_size=100)
        # 记录训练过程中的loss和reward
        self.rewards = []
        self.losses = []
        self.delays = []
        # 记录任务卸载分布（节点级动作空间 A = {L, e1, e2, e3, C}）
        self.offloading_distribution = {a: 0 for a in range(self.action_dim)}
        # 记录不同网络负载下的卸载分布
        self.network_load_offloading = {
            level: {a: 0 for a in range(self.action_dim)}
            for level in ('low', 'medium', 'high')
        }
        # 固定随机种子
        self.rng = np.random.default_rng(42 + agent_id)
        # Dedicated, seeded Python RNG for experience replay so that training is
        # reproducible (``random.sample`` on the global RNG is not).
        self.py_rng = random.Random(42 + agent_id)
        
    def update_model(self, global_weights: list):
        """Update local model with global weights."""
        self.model.set_weights(global_weights)
        self.target_model.set_weights(global_weights)
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience."""
        self.memory.append((state, action, reward, next_state, done))

    def sync_target(self):
        """Copy the online network into the bootstrapping target network."""

        self.target_model.set_weights(self.model.get_weights())
    
    def train(self, tasks=None, epochs=3):
        """Train local model with more detailed process."""
        if tasks is not None:
            print(f"代理 {self.agent_id} 开始本地训练...")
            
            task_rows = list(tasks.iterrows())
            task_count = len(task_rows)

            # The environment is a *continuing* task, not a set of independent
            # episodes: the finite service capacity is kept alive across the
            # local epochs of a federated round, so the congestion created by
            # the agent's own (increasingly greedy) policy is visible to the
            # next epoch instead of being erased at every epoch boundary.
            queue_model = getattr(self, "queue_model", None)
            if queue_model is None:
                queue_model = NetworkQueueingModel()
                self.queue_model = queue_model
                self.env_step = 0
            base_step = int(getattr(self, "env_step", 0))

            for epoch in range(epochs):
                epoch_reward = 0
                epoch_loss = 0
                epoch_delay = 0
                batch_count = 0

                # 收集经验数据
                for j, (_, task) in enumerate(task_rows):
                    step_index = base_step + j
                    loads = queue_model.context(task, step_index)
                    state = get_state(task, loads)

                    # epsilon-greedy策略
                    if self.rng.random() <= self.epsilon:
                        action = int(self.rng.integers(self.action_dim))
                    else:
                        q_values = self.model.forward(state.reshape(1, -1))[0]
                        action = int(np.argmax(q_values))

                    # 记录卸载分布
                    self.offloading_distribution[action] += 1

                    # 按当前边缘层积压划分负载级别
                    backlog = float(loads.get('queue_backlog_mean', 0.0))
                    if backlog < 0.33:
                        network_load = 'low'
                    elif backlog < 0.66:
                        network_load = 'medium'
                    else:
                        network_load = 'high'
                    self.network_load_offloading[network_load][action] += 1

                    # 第二步决策：CPU核心数和内存配额分配
                    cpu_cores = None
                    memory_quota = None
                    if action == ACTION_ID['terminal']:  # 本地执行
                        from algorithm_utils import CPU_CORES, MEMORY_QUOTAS
                        cpu_cores = CPU_CORES[self.rng.integers(len(CPU_CORES))]
                        memory_quota = MEMORY_QUOTAS[self.rng.integers(len(MEMORY_QUOTAS))]

                    from algorithm_utils import simulate_task_execution
                    exec_result = simulate_task_execution(
                        task,
                        action,
                        self.rng,
                        cpu_cores,
                        memory_quota,
                        priority_qos=True,
                    )

                    # 有限服务容量：排队等待直接计入执行时延
                    queue_wait = queue_model.step(task, action, exec_result['exec_delay_ms'], step_index)
                    if queue_wait > 0.0:
                        exec_result['exec_delay_ms'] = float(
                            np.clip(exec_result['exec_delay_ms'] + queue_wait, 3.0, 300.0)
                        )
                        exec_result['deadline_met'] = int(
                            exec_result['deadline_met'] == 1
                            and exec_result['exec_delay_ms'] <= float(task['deadline_ms'])
                        )

                    exec_result.update(
                        evaluate_runtime_trust(
                            task,
                            action,
                            action,
                            exec_result,
                            trust_threshold=0.80,
                        )
                    )
                    reward = calculate_reward(task, action, exec_result, self.reward_weights)
                    epoch_delay += float(exec_result.get('exec_delay_ms', 0.0))

                    # 获取下一个状态
                    if j < task_count - 1:
                        next_state = get_state(
                            task_rows[j + 1][1],
                            queue_model.context(task_rows[j + 1][1], step_index + 1),
                        )
                        done = False
                    else:
                        next_state = state
                        done = True
                    
                    # 存储经验
                    self.remember(state, action, reward, next_state, done)
                    epoch_reward += reward
                    
                    # 经验回放
                    if len(self.memory) > self.batch_size and ((j + 1) % self.replay_interval == 0 or j == task_count - 1):
                        loss = self._replay()
                        epoch_loss += loss
                        batch_count += 1

                    if (step_index + 1) % TARGET_SYNC_STEPS == 0:
                        self.sync_target()
                
                self.env_step = base_step + task_count
                base_step = self.env_step

                avg_reward = epoch_reward / task_count
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                avg_delay = epoch_delay / task_count if task_count > 0 else 0
                self.rewards.append(avg_reward)
                self.losses.append(avg_loss)
                self.delays.append(avg_delay)
                print(f"  代理 {self.agent_id} 第 {epoch + 1} 轮: 平均奖励 = {avg_reward:.4f}, 平均loss = {avg_loss:.4f}, 平均延迟 = {avg_delay:.2f}ms, 经验回放次数 = {batch_count}")
            
            # 衰减epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            print(f"  代理 {self.agent_id} 训练完成，最终epsilon = {self.epsilon:.4f}")
        else:
            # 仅进行经验回放
            if len(self.memory) >= self.batch_size:
                self._replay()
    
    def _replay(self):
        """经验回放：以目标网络做自助采样，分批平均梯度更新。"""
        minibatch = self.py_rng.sample(self.memory, self.batch_size)

        states = np.vstack([sample[0].reshape(1, -1) for sample in minibatch])
        actions = np.array([int(sample[1]) for sample in minibatch])
        rewards = np.array([float(sample[2]) for sample in minibatch])
        next_states = np.vstack([sample[3].reshape(1, -1) for sample in minibatch])
        dones = np.array([bool(sample[4]) for sample in minibatch])

        # Bootstrapping is done with the frozen target network, which is what
        # keeps the FedAvg-averaged online network from chasing its own output.
        next_q = self.target_model.forward(next_states)
        targets = rewards + self.gamma * np.max(next_q, axis=1) * (~dones)

        current_q = self.model.forward(states)
        target_q = current_q.copy()
        target_q[np.arange(len(actions)), actions] = targets

        loss = float(np.mean(np.square(target_q - current_q)))
        self.model.backward(states, target_q, current_q)
        return loss
    
    def save_offloading_distribution(self, output_dir="results"):
        """保存任务卸载分布数据"""
        import os
        import pandas as pd
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存总体卸载分布
        distribution_data = []
        for action, count in self.offloading_distribution.items():
            action_name = ACTION_MAP.get(action, str(action))
            distribution_data.append({
                'agent_id': self.agent_id,
                'action': action_name,
                'count': count,
                'total': sum(self.offloading_distribution.values()),
                'percentage': (count / sum(self.offloading_distribution.values())) * 100
            })
        
        distribution_df = pd.DataFrame(distribution_data)
        distribution_path = os.path.join(output_dir, f"offloading_distribution_agent_{self.agent_id}.csv")
        distribution_df.to_csv(distribution_path, index=False)
        print(f"卸载分布数据已保存至: {distribution_path}")
        
        # 保存不同网络负载下的卸载分布
        network_load_data = []
        for load, distribution in self.network_load_offloading.items():
            total = sum(distribution.values())
            for action, count in distribution.items():
                action_name = ACTION_MAP.get(action, str(action))
                network_load_data.append({
                    'agent_id': self.agent_id,
                    'network_load': load,
                    'action': action_name,
                    'count': count,
                    'total': total,
                    'percentage': (count / total * 100) if total > 0 else 0
                })
        
        network_load_df = pd.DataFrame(network_load_data)
        network_load_path = os.path.join(output_dir, f"network_load_offloading_agent_{self.agent_id}.csv")
        network_load_df.to_csv(network_load_path, index=False)
        print(f"网络负载卸载分布数据已保存至: {network_load_path}")


class FederatedServer:
    """Federated learning server."""
    
    def __init__(self, state_dim: int, action_dim: int):
        """Initialize federated server."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_model = NeuralNetwork(state_dim, action_dim)
        self.local_agents = []
        # 记录所有代理的训练数据
        self.agent_rewards = {}
        self.agent_losses = {}
        self.agent_delays = {}
        self.agent_log_offsets = {}
        # 记录验证数据
        self.val_rewards = []
        self.val_losses = []
        self.val_delays = []
        self.val_epochs = []
    
    def add_local_agent(self, agent: LocalAgent):
        """Add local agent."""
        self.local_agents.append(agent)
    
    def federated_averaging(self):
        """Perform federated averaging."""
        if not self.local_agents:
            return
        
        print("执行联邦平均算法...")
        
        # Get global weights as template
        global_weights = self.global_model.get_weights()
        avg_weights = [np.zeros_like(w) for w in global_weights]
        
        # Sum weights from all agents
        for i, agent in enumerate(self.local_agents):
            agent_weights = agent.model.get_weights()
            # 模拟加密上传（这里简化处理）
            print(f"  接收代理 {i} 的模型参数")
            for j in range(len(avg_weights)):
                avg_weights[j] += agent_weights[j]
            
            # 收集代理的训练数据
            if i not in self.agent_rewards:
                self.agent_rewards[i] = []
                self.agent_losses[i] = []
                self.agent_delays[i] = []
                self.agent_log_offsets[i] = 0
            # 只追加本轮新增的训练日志，避免重复写入历史epoch
            start_idx = self.agent_log_offsets.get(i, 0)
            self.agent_rewards[i].extend(agent.rewards[start_idx:])
            self.agent_losses[i].extend(agent.losses[start_idx:])
            self.agent_delays[i].extend(agent.delays[start_idx:])
            self.agent_log_offsets[i] = len(agent.rewards)
        
        # Average weights
        num_agents = len(self.local_agents)
        for i in range(len(avg_weights)):
            avg_weights[i] /= num_agents
        
        # Update global model
        self.global_model.set_weights(avg_weights)
        print("  全局模型已更新")
        
        # Distribute global weights to local agents
        for i, agent in enumerate(self.local_agents):
            agent.update_model(avg_weights)
            # 模拟模型融合（这里简化处理）
            print(f"  下发全局模型到代理 {i}")
    
    def plot_training_metrics(self, output_dir="results"):
        """生成训练过程中的loss和reward图表"""
        import matplotlib.pyplot as plt
        import os
        import pandas as pd
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成训练和验证的reward图表
        plt.figure(figsize=(12, 6))
        # 绘制训练reward
        for agent_id, rewards in self.agent_rewards.items():
            plt.plot(rewards, label=f"Agent {agent_id} (Train)")
        # 绘制验证reward
        if self.val_rewards:
            # 验证是每5轮进行一次，需要调整x轴坐标
            val_x = self.val_epochs if self.val_epochs else [i * 5 for i in range(len(self.val_rewards))]
            plt.plot(val_x, self.val_rewards, 'r--', label="Validation", linewidth=2)
        plt.title("Training and Validation Rewards")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.grid(True)
        reward_path = os.path.join(output_dir, "training_validation_rewards.png")
        plt.savefig(reward_path)
        plt.close()
        print(f"训练和验证奖励图表已保存至: {reward_path}")
        
        # 保存详细的训练日志
        self.save_training_logs(output_dir)
        
        return reward_path
    
    def save_training_logs(self, output_dir="results"):
        """保存详细的训练日志"""
        import pandas as pd
        import os
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存训练奖励日志
        training_logs = []
        for agent_id, rewards in self.agent_rewards.items():
            for epoch, reward in enumerate(rewards):
                avg_delay = None
                if agent_id in self.agent_delays and epoch < len(self.agent_delays[agent_id]):
                    avg_delay = self.agent_delays[agent_id][epoch]
                training_logs.append({
                    'agent_id': agent_id,
                    'epoch': epoch,
                    'reward': reward,
                    'avg_delay_ms': avg_delay,
                    'type': 'training'
                })
        
        # 保存验证奖励日志
        if self.val_rewards:
            for epoch, reward in enumerate(self.val_rewards):
                avg_delay = self.val_delays[epoch] if epoch < len(self.val_delays) else None
                epoch_index = self.val_epochs[epoch] if epoch < len(self.val_epochs) else epoch * 5
                training_logs.append({
                    'agent_id': -1,  # 验证是全局的
                    'epoch': epoch_index,
                    'reward': reward,
                    'avg_delay_ms': avg_delay,
                    'type': 'validation'
                })
        
        # 保存验证平均延迟日志，便于图3直接绘制时延收敛曲线
        if self.val_delays:
            for epoch, delay in enumerate(self.val_delays):
                epoch_index = self.val_epochs[epoch] if epoch < len(self.val_epochs) else epoch * 5
                training_logs.append({
                    'agent_id': -1,
                    'epoch': epoch_index,
                    'avg_delay_ms': delay,
                    'type': 'validation_delay'
                })
        
        # 保存训练损失日志
        for agent_id, losses in self.agent_losses.items():
            for epoch, loss in enumerate(losses):
                training_logs.append({
                    'agent_id': agent_id,
                    'epoch': epoch,
                    'loss': loss,
                    'avg_delay_ms': None,
                    'type': 'training_loss'
                })
        
        # 保存验证损失日志
        if self.val_losses:
            for epoch, loss in enumerate(self.val_losses):
                epoch_index = self.val_epochs[epoch] if epoch < len(self.val_epochs) else epoch * 5
                training_logs.append({
                    'agent_id': -1,  # 验证是全局的
                    'epoch': epoch_index,
                    'loss': loss,
                    'avg_delay_ms': None,
                    'type': 'validation_loss'
                })
        
        # 创建DataFrame并保存
        logs_df = pd.DataFrame(training_logs)
        logs_path = os.path.join(output_dir, "training_logs.csv")
        logs_df.to_csv(logs_path, index=False)
        print(f"训练详细日志已保存至: {logs_path}")
    
    def get_global_model(self) -> NeuralNetwork:
        """Get global model."""
        return self.global_model
    
    def save_all_offloading_distributions(self, output_dir="results"):
        """保存所有代理的卸载分布数据"""
        for agent in self.local_agents:
            agent.save_offloading_distribution(output_dir)
        
        # 汇总所有代理的卸载分布
        import os
        import pandas as pd
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        all_distribution_data = []
        all_network_load_data = []
        
        for agent in self.local_agents:
            # 汇总总体卸载分布
            for action, count in agent.offloading_distribution.items():
                action_name = ACTION_MAP.get(action, str(action))
                all_distribution_data.append({
                    'agent_id': agent.agent_id,
                    'action': action_name,
                    'count': count
                })
            
            # 汇总网络负载卸载分布
            for load, distribution in agent.network_load_offloading.items():
                for action, count in distribution.items():
                    action_name = ACTION_MAP.get(action, str(action))
                    all_network_load_data.append({
                        'agent_id': agent.agent_id,
                        'network_load': load,
                        'action': action_name,
                        'count': count
                    })
        
        # 保存汇总数据
        if all_distribution_data:
            distribution_df = pd.DataFrame(all_distribution_data)
            distribution_summary = distribution_df.groupby('action')['count'].sum().reset_index()
            total = distribution_summary['count'].sum()
            distribution_summary['percentage'] = (distribution_summary['count'] / total) * 100
            summary_path = os.path.join(output_dir, "offloading_distribution_summary.csv")
            distribution_summary.to_csv(summary_path, index=False)
            print(f"卸载分布汇总数据已保存至: {summary_path}")
        
        if all_network_load_data:
            network_load_df = pd.DataFrame(all_network_load_data)
            network_load_summary = network_load_df.groupby(['network_load', 'action'])['count'].sum().reset_index()
            summary_path = os.path.join(output_dir, "network_load_offloading_summary.csv")
            network_load_summary.to_csv(summary_path, index=False)
            print(f"网络负载卸载分布汇总数据已保存至: {summary_path}")


def get_state(task: pd.Series, loads: dict | None = None) -> np.ndarray:
    """Extract the 15-dimensional state used by the DQN.

    The last six entries are the *live* load of the execution locations, taken
    from the queueing model.  They are what makes the decision non-degenerate:
    a policy can only balance a heterogeneous edge tier if it can observe how
    congested each node currently is.
    """
    loads = loads or {}

    def _load(key: str) -> float:
        try:
            return float(min(max(float(loads.get(key, 0.0)), 0.0), 1.0))
        except (TypeError, ValueError):
            return 0.0

    return np.array([
        float(task['delay_norm']),
        float(task['compute_norm']),
        float(task['data_size_norm']),
        float(task['priority_score']),
        float(task.get('memory_norm', 0)),
        float(task.get('dependency_count', 0)) / 5.0,  # 归一化依赖计数
        float(task.get('in_degree', 0)) / 3.0,  # 归一化入度
        float(task.get('out_degree', 0)) / 3.0,  # 归一化出度
        float(task.get('high_priority', 0)),  # 高优先级标记
        _load('queue_backlog_mean'),  # 边缘层平均积压
        _load('queue_backlog_max'),  # 边缘层最大积压
        _load('load_edge_1'),  # 各执行位置负载
        _load('load_edge_2'),
        _load('load_edge_3'),
        _load('load_cloud'),
    ])


def train_federated_dql(tasks: pd.DataFrame, num_agents: int = 3, rounds: int = 100, local_epochs: int = 5) -> FederatedServer:
    """Train FedDQL with federated learning."""
    state_dim = 15  # 9 个任务特征 + 6 个执行位置负载特征
    action_dim = NUM_ACTIONS  # 节点级动作空间 A = {L, e1, e2, e3, C}
    
    # 创建联邦服务器
    server = FederatedServer(state_dim, action_dim)
    
    # 创建本地代理
    for i in range(num_agents):
        agent = LocalAgent(state_dim, action_dim, i)
        server.add_local_agent(agent)
    
    # 分配任务给本地代理
    task_chunks = np.array_split(tasks, num_agents)
    
    for round_num in range(rounds):
        print(f"\n=== Round {round_num + 1}/{rounds} ===")
        
        # 本地训练
        for i, (agent, task_chunk) in enumerate(zip(server.local_agents, task_chunks)):
            # 使用新的train方法，传入tasks参数
            agent.train(task_chunk, local_epochs)
        
        # 联邦平均
        print("Performing federated averaging...")
        server.federated_averaging()
        print("Global model updated.")
    
    return server

def fed_dql_policy_with_federated_learning(
    task: pd.Series,
    server: FederatedServer,
    loads: dict | None = None,
) -> int:
    """FedDQL decision rule with federated learning.

    High-priority tasks are pinned to the least-loaded edge node so that the
    priority QoS reservation is not defeated by queue backlog.  Every other task
    follows the learned Q-values, which now observe the live node loads.
    """
    loads = loads or {}
    if bool(task.get("high_priority", 0)):
        return int(min(EDGE_ACTION_IDS, key=lambda a: float(loads.get(f"load_edge_{a}", 0.0))))
    state = get_state(task, loads)
    q_values = server.global_model.forward(state.reshape(1, -1))[0]
    return int(np.argmax(q_values))

def main_federated(task_set: str = "real", train: bool = True) -> str:
    if task_set != "real":
        raise ValueError("FedDQL only supports real SAC dataset.")

    raw_tasks = load_task_set("real")
    tasks = prepare_task_dataframe(raw_tasks, target_tasks=3000, seed=42)

    fuzzy = FuzzyTaskClassifier()
    fuzzy_eval = fuzzy.evaluate(tasks)

    if train:
        # 训练联邦学习模型
        server = train_federated_dql(tasks)
        
        # 使用训练好的模型进行推理
        records = run_policy_on_tasks(
            tasks,
            lambda task, loads: fed_dql_policy_with_federated_learning(task, server, loads),
            algorithm_name="FedDQL-Federated",
            seed=42,
            dependency_aware=True,
            priority_qos=True,
        )
    else:
        # 使用规则-based策略（兼容现有代码）
        from models.fed_dql_algorithm import fed_dql_policy
        records = run_policy_on_tasks(
            tasks,
            fed_dql_policy,
            algorithm_name="FedDQL",
            seed=42,
            dependency_aware=True,
            priority_qos=True,
        )

    records["fuzzy_pred_action"] = fuzzy_eval["y_pred"]

    metrics = calculate_metrics(records, fuzzy_accuracy=fuzzy_eval["accuracy"])
    for key, value in metrics.items():
        records[key] = value

    result_path = save_records(records, "fed_dql_federated_result.csv")

    # Save one-line metrics for reporting.
    summary_path = os.path.join(os.path.dirname(result_path), "fed_dql_federated_summary.csv")
    pd.DataFrame([{"algorithm": "FedDQL-Federated", **metrics}]).to_csv(summary_path, index=False)

    return result_path


if __name__ == "__main__":
    path = main_federated("real", train=True)
    print(f"FedDQL-Federated result saved: {path}")
