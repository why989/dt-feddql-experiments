from __future__ import annotations

import inspect
import os
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from deployment_model import (
    ACTION_ID,
    ACTION_MAP,
    CLOUD_ACTION_ID,
    EDGE_ACTION_IDS,
    EDGE_NODE_BY_ACTION,
    NUM_ACTIONS,
    TERMINAL_ACTION_ID,
    NetworkQueueingModel,
    edge_node_speedup,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_REAL_DATA_PATH = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "..", "data", "real_microseismic_task_set.csv")
)
PROJECT_REAL_DATA_PATH = os.path.join(DATA_DIR, "real_sac_task_set_3000.csv")

# CPU核心数选项
CPU_CORES = [1, 2, 4, 8]

# 内存配额选项（MB）
MEMORY_QUOTAS = [256, 512, 1024, 2048]

class TaskQueue:
    """任务队列管理"""
    
    def __init__(self, max_size=100):
        """初始化任务队列"""
        self.queue = []
        self.max_size = max_size
        self.processing_time = 0.0
    
    def add_task(self, task, processing_time):
        """添加任务到队列"""
        if len(self.queue) < self.max_size:
            self.queue.append((task, processing_time))
            return True
        return False
    
    def get_queue_length(self):
        """获取队列长度"""
        return len(self.queue)
    
    def get_queue_delay(self):
        """获取队列延迟"""
        total_delay = 0.0
        for _, processing_time in self.queue:
            total_delay += processing_time
        return total_delay
    
    def process_task(self):
        """处理队列中的一个任务"""
        if self.queue:
            return self.queue.pop(0)
        return None, 0.0
    
    def is_empty(self):
        """检查队列是否为空"""
        return len(self.queue) == 0


def _parse_dependency_ids(raw) -> list[int]:
    if pd.isna(raw):
        return []

    text = str(raw).strip()
    if not text:
        return []

    normalized = text.replace("|", ",").replace(";", ",").replace(" ", ",")
    ids = []
    for token in normalized.split(","):
        t = token.strip()
        if not t:
            continue
        try:
            ids.append(int(float(t)))
        except ValueError:
            continue
    return ids


def _priority_to_score(value) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(np.clip(value, 0.0, 1.0))

    s = str(value).strip().lower()
    if s == "high":
        return 1.0
    if s == "medium":
        return 0.65
    if s == "low":
        return 0.3
    return 0.5


def load_task_set(task_set: str = "real") -> pd.DataFrame:
    """Load only real SAC-derived tasks (no simulation fallback)."""
    if task_set != "real":
        raise ValueError("Only 'real' task_set is supported in this project.")

    if os.path.exists(PROJECT_REAL_DATA_PATH):
        return pd.read_csv(PROJECT_REAL_DATA_PATH)

    if os.path.exists(DEFAULT_REAL_DATA_PATH):
        return pd.read_csv(DEFAULT_REAL_DATA_PATH)

    raise FileNotFoundError(
        "Real SAC task CSV not found. Run data_preprocess.py first. "
        f"Expected one of: {PROJECT_REAL_DATA_PATH} or {DEFAULT_REAL_DATA_PATH}"
    )


def prepare_task_dataframe(df: pd.DataFrame, target_tasks: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Normalize task fields and enforce a unified 3000-task real dataset."""
    if df.empty:
        raise ValueError("Input task dataframe is empty.")

    frame = df.copy()

    required_cols = [
        "data_size_byte",
        "compute_density_FLOPs_per_byte",
        "delay_sensitivity",
        "deadline_second",
    ]
    
    # 添加内存占用字段（如果不存在）
    if "memory_usage_mb" not in frame.columns:
        # 根据数据大小和计算密度估算内存占用
        frame["memory_usage_mb"] = frame["data_size_byte"] / (1024 * 1024) * (1 + frame["compute_density_FLOPs_per_byte"] / 1e9)
    for col in required_cols:
        if col not in frame.columns:
            raise KeyError(f"Missing required column: {col}")

    frame["priority_score"] = frame.get("priority", 0.5).apply(_priority_to_score)

    for col in ["data_size_byte", "compute_density_FLOPs_per_byte", "delay_sensitivity", "deadline_second"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["data_size_byte", "compute_density_FLOPs_per_byte", "delay_sensitivity", "deadline_second"])

    # Keep only real-data-derived records, then resample from real records when needed.
    if len(frame) < target_tasks:
        extra = frame.sample(n=target_tasks - len(frame), replace=True, random_state=seed)
        frame = pd.concat([frame, extra], ignore_index=True)
    elif len(frame) > target_tasks:
        frame = frame.sample(n=target_tasks, random_state=seed).reset_index(drop=True)

    frame = frame.reset_index(drop=True)
    if "global_task_id" in frame.columns:
        source = pd.to_numeric(frame["global_task_id"], errors="coerce")
        fallback = pd.Series(frame.index, index=frame.index, dtype="float64")
        frame["source_task_id"] = source.fillna(fallback).astype(int)
    else:
        frame["source_task_id"] = frame.index.astype(int)
    frame["task_id"] = np.arange(len(frame))

    # Feature normalization
    frame["data_size_mb"] = frame["data_size_byte"] / (1024 * 1024)
    frame["data_size_norm"] = (frame["data_size_mb"] - frame["data_size_mb"].min()) / (
        frame["data_size_mb"].max() - frame["data_size_mb"].min() + 1e-8
    )
    frame["compute_norm"] = (frame["compute_density_FLOPs_per_byte"] - frame["compute_density_FLOPs_per_byte"].min()) / (
        frame["compute_density_FLOPs_per_byte"].max() - frame["compute_density_FLOPs_per_byte"].min() + 1e-8
    )
    frame["delay_norm"] = (frame["delay_sensitivity"] - frame["delay_sensitivity"].min()) / (
        frame["delay_sensitivity"].max() - frame["delay_sensitivity"].min() + 1e-8
    )
    # 内存占用归一化
    frame["memory_norm"] = (frame["memory_usage_mb"] - frame["memory_usage_mb"].min()) / (
        frame["memory_usage_mb"].max() - frame["memory_usage_mb"].min() + 1e-8
    )

    # Build realistic deadline in ms for scheduling evaluation
    # 调整截止时间计算：高优先级任务截止时间更严格，低优先级任务更宽松
    frame["deadline_ms"] = (
        10  # 基础时间进一步减少，使整体截止时间更严格
        + 30 * (1 - frame["delay_norm"])  # 延迟敏感度权重调整
        - 15 * frame["priority_score"]  # 高优先级任务截止时间更严格
        + 4 * frame["data_size_norm"]  # 数据大小权重调整
    ).clip(lower=8, upper=45)  # 进一步调整范围，使截止时间更符合实际

    frame["high_priority"] = frame["priority_score"] >= 0.8

    source_id_to_task = {}
    for _, row in frame.iterrows():
        source_id_to_task.setdefault(int(row["source_task_id"]), int(row["task_id"]))

    dependency_col = "dependency_task_id" if "dependency_task_id" in frame.columns else None
    if dependency_col is not None:
        deps = []
        dep_counts = []
        in_degree = []
        out_degree = []
        
        # 首先构建依赖关系映射
        dependency_map = {}
        for _, row in frame.iterrows():
            task_id = int(row["task_id"])
            source_deps = _parse_dependency_ids(row[dependency_col])
            mapped = [source_id_to_task[d] for d in source_deps if d in source_id_to_task]
            mapped = [m for m in mapped if m != task_id]
            uniq = sorted(set(mapped))
            dependency_map[task_id] = uniq
        
        # 计算入度和出度
        in_degree_map = {i: 0 for i in range(len(frame))}
        out_degree_map = {i: 0 for i in range(len(frame))}
        
        for task_id, deps_list in dependency_map.items():
            out_degree_map[task_id] = len(deps_list)
            for dep_id in deps_list:
                if dep_id in in_degree_map:
                    in_degree_map[dep_id] += 1
        
        # 填充依赖信息
        for _, row in frame.iterrows():
            task_id = int(row["task_id"])
            task_deps = dependency_map.get(task_id, [])
            deps.append(task_deps)
            dep_counts.append(len(task_deps))
            in_degree.append(in_degree_map.get(task_id, 0))
            out_degree.append(out_degree_map.get(task_id, 0))
        
        frame["dependency_ids"] = deps
        frame["dependency_count"] = dep_counts
        frame["in_degree"] = in_degree
        frame["out_degree"] = out_degree
        
        # 检测并处理循环依赖
        def has_cycle(task_id, visited, rec_stack):
            visited[task_id] = True
            rec_stack[task_id] = True
            
            for dep_id in dependency_map.get(task_id, []):
                if not visited.get(dep_id, False):
                    if has_cycle(dep_id, visited, rec_stack):
                        return True
                elif rec_stack.get(dep_id, False):
                    return True
            
            rec_stack[task_id] = False
            return False
        
        visited = {}
        rec_stack = {}
        for task_id in range(len(frame)):
            if not visited.get(task_id, False):
                if has_cycle(task_id, visited, rec_stack):
                    # 如果发现循环依赖，移除循环
                    frame.at[task_id, "dependency_ids"] = []
                    frame.at[task_id, "dependency_count"] = 0
                    frame.at[task_id, "in_degree"] = 0
                    frame.at[task_id, "out_degree"] = 0
    else:
        frame["dependency_ids"] = [[] for _ in range(len(frame))]
        frame["dependency_count"] = 0
        frame["in_degree"] = [0 for _ in range(len(frame))]
        frame["out_degree"] = [0 for _ in range(len(frame))]

    # 添加边缘节点累计能耗和平均延迟特征
    # 基于任务特征估算边缘节点累计能耗
    frame["edge_energy_kj"] = (0.026 + 0.017 * frame["compute_norm"] + 0.008 * frame["data_size_norm"]) * 10
    # 基于任务特征估算平均延迟
    frame["avg_delay_ms"] = 14 + 18 * frame["compute_norm"] + 10 * frame["data_size_norm"]

    frame["reference_action"] = frame.apply(reference_action_rule, axis=1)

    return frame


def reference_action_rule(task: pd.Series) -> int:
    """Reference heuristic used for fuzzy-prior consistency evaluation.

    This label is not an external expert annotation. It is used only to check
    whether the fuzzy-prior module agrees with a deterministic reference
    offloading heuristic.
    """
    delay_urgent = float(task["delay_norm"]) > 0.72
    high_priority = float(task["priority_score"]) >= 0.8
    compute_high = float(task["compute_norm"]) > 0.62
    data_large = float(task["data_size_norm"]) > 0.68

    if (delay_urgent and high_priority) or (delay_urgent and compute_high):
        return ACTION_ID["cloud"]
    if compute_high or data_large or high_priority:
        return ACTION_ID["edge"]
    return ACTION_ID["terminal"]


def simulate_task_execution(
    task: pd.Series,
    action: int,
    rng: np.random.Generator,
    cpu_cores=None,
    memory_quota=None,
    bandwidth_simulator=None,
    priority_qos: bool = False,
) -> Dict[str, float]:
    c = float(task["compute_norm"])
    d = float(task["data_size_norm"])
    memory = float(task.get("memory_norm", 0))
    priority = float(task.get("priority_score", 0.5))
    high_priority = bool(task.get("high_priority", priority >= 0.8))
    action = int(action)
    edge_spec = EDGE_NODE_BY_ACTION.get(action)
    priority_qos_enabled = bool(priority_qos and high_priority and edge_spec is not None)

    # Priority-aware QoS: high-priority edge tasks receive reserved resources
    # and a more stable uplink. Metrics are still computed from execution logs.
    if priority_qos_enabled:
        cpu_cores = max(int(cpu_cores or 0), 4)
        memory_quota = max(int(memory_quota or 0), 2048)
    
    # 架构约束参数（符合论文要求）
    if bandwidth_simulator:
        B_up = bandwidth_simulator()  # 使用自定义带宽模拟器
    elif edge_spec is not None:
        # Heterogeneous edge uplink: every edge node has its own bandwidth range.
        B_up = rng.uniform(*edge_spec["uplink_mbps"])
    else:
        B_up = rng.uniform(10, 100)  # 终端-边缘上行带宽 (Mbps)，10-100 Mbps
    if priority_qos_enabled:
        B_up = max(float(B_up), 80.0)
    B_down = rng.uniform(10, 100)  # 边缘-云下行带宽 (Mbps)，10-100 Mbps
    E_edge_max = 80.0  # 边缘节点最大能量 (KJ/h)
    
    # 网络延迟参数（符合论文要求）
    edge_edge_delay = rng.uniform(5, 20)  # 边缘节点间延迟 (ms)
    edge_cloud_delay = rng.uniform(50, 100)  # 边缘到云的延迟 (ms)
    
    # 数据通信量 (MB)
    data_in = float(task.get("data_size_mb", d * 100))
    data_out = data_in * 0.7  # 假设输出数据量为输入的70%
    
    # CPU核心数和内存配额的影响
    cpu_speedup = 1.0
    memory_factor = 1.0
    
    if edge_spec is not None:
        # Node clock frequency, with the reserved-core boost under priority QoS.
        cpu_speedup = edge_node_speedup(action, priority_qos=priority_qos_enabled)
    elif cpu_cores is not None:
        # CPU核心数的加速效果（非线性）
        cpu_speedup = min(cpu_cores, 8) ** 0.7
    
    if memory_quota is not None:
        # 内存配额的影响
        memory_factor = min(memory_quota / 1024, 2.0)  # 以1024MB为基准
    
    if action == TERMINAL_ACTION_ID:
        # 本地执行，无通信延迟和能耗
        base_delay = 36 + 30 * c + 18 * d
        # 增加随机性和资源竞争影响
        delay_ms = base_delay / cpu_speedup + rng.normal(0, 2.5) + rng.uniform(0, 5)
        compute_energy = (0.055 + 0.028 * c + 0.014 * d) * (1 / cpu_speedup)
        comm_energy = 0.0
        cpu = 0.54 + 0.08 * c
        bw = 0.47 + 0.05 * d
    elif edge_spec is not None:
        # 边缘执行，需要终端到边缘的通信
        # 增加通信延迟的计算，考虑网络拥塞
        congestion_factor = rng.uniform(1.0, 1.5)
        if priority_qos_enabled:
            congestion_factor = min(float(congestion_factor), 1.08)
        comm_delay = (data_in * 8) / (B_up * 1000) * 1000 * congestion_factor  # 转换为ms，增加网络拥塞影响
        base_exec_delay = 14 + 18 * c + 10 * d
        # 增加执行延迟的随机性和资源竞争影响
        if priority_qos_enabled:
            exec_delay = base_exec_delay / cpu_speedup + rng.normal(0, 0.6) + rng.uniform(0, 2)
        else:
            exec_delay = base_exec_delay / cpu_speedup + rng.normal(0, 1.5) + rng.uniform(0, 8)
        delay_ms = comm_delay + exec_delay
        
        compute_energy = (0.026 + 0.017 * c + 0.008 * d) * (1 / cpu_speedup)
        comm_energy = (data_in * 8) / (B_up) * 0.01  # 简化的通信能耗模型
        
        if priority_qos_enabled:
            cpu = 0.78 + 0.10 * c
            bw = 0.76 + 0.10 * d
        else:
            cpu = 0.74 + 0.12 * c
            bw = 0.73 + 0.12 * d
    else:  # cloud
        # 云执行，需要终端到边缘再到云的通信
        # 增加通信延迟的计算，考虑网络拥塞
        edge_comm_delay = (data_in * 8) / (B_up * 1000) * 1000 * rng.uniform(1.0, 1.5)
        cloud_comm_delay = (data_in * 8) / (B_down * 1000) * 1000 * rng.uniform(1.0, 1.5)
        base_exec_delay = 17 + 11 * c + 15 * d
        # 增加执行延迟的随机性和资源竞争影响
        exec_delay = base_exec_delay / cpu_speedup + rng.normal(0, 2.0) + rng.uniform(0, 10)
        # 添加边缘到云的固定延迟，增加随机性
        edge_cloud_delay = rng.uniform(60, 120)  # 增加边缘到云的延迟范围
        delay_ms = edge_comm_delay + cloud_comm_delay + exec_delay + edge_cloud_delay
        
        compute_energy = (0.038 + 0.022 * c + 0.016 * d) * (1 / cpu_speedup)
        comm_energy = (data_in * 8) / (B_up) * 0.01 + (data_in * 8) / (B_down) * 0.005
        
        cpu = 0.46 + 0.10 * c
        bw = 0.62 + 0.20 * d

    # 总能耗
    energy_kj = compute_energy + comm_energy
    
    # 应用约束
    delay_ms = float(np.clip(delay_ms, 3.0, 120.0))
    energy_kj = float(np.clip(energy_kj, 0.008, 0.20))
    cpu = float(np.clip(cpu, 0.30, 0.95))
    bw = float(np.clip(bw, 0.30, 0.95))

    deadline_ms = float(task["deadline_ms"])
    deadline_met = delay_ms <= deadline_ms
    
    # 增加故障注入机制：模拟系统故障或资源竞争导致的任务超时
    # 为所有任务添加一定的失败概率，高优先级任务失败概率较低
    # 失败概率与优先级成反比，高优先级任务失败概率低，低优先级任务失败概率高
    failure_probability = 0.05 + (1 - priority) * 0.15  # 失败概率范围：0.05-0.2
    if priority_qos_enabled:
        # Reserved edge resources plus one lightweight retry reduce random
        # deadline failure for high-priority tasks.
        failure_probability = max(0.001, failure_probability * 0.08)
    if rng.random() < failure_probability:
        # 模拟任务失败，设置为超时
        deadline_met = False

    return {
        "exec_delay_ms": delay_ms,
        "exec_energy_kj": energy_kj,
        "exec_cpu_util": cpu,
        "bandwidth_utilization": bw,
        "deadline_met": int(deadline_met),
        "compute_energy_kj": compute_energy,
        "comm_energy_kj": comm_energy,
        "data_in_mb": data_in,
        "data_out_mb": data_out,
        "cpu_cores": cpu_cores,
        "memory_quota": memory_quota,
        "priority_qos": int(priority_qos_enabled),
        "failure_probability": float(failure_probability),
    }


def _policy_wants_context(policy_fn: Callable) -> bool:
    """Detect whether a policy callable accepts the load-observation argument."""

    try:
        params = inspect.signature(policy_fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
        return True
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def run_policy_on_tasks(
    tasks: pd.DataFrame,
    policy_fn: Callable[[pd.Series], int],
    algorithm_name: str,
    seed: int = 42,
    dependency_aware: bool = True,
    bandwidth_simulator: Callable[[], float] = None,
    priority_qos: bool = False,
    digital_twin_orchestration: bool = False,
    trust_threshold: float = 0.80,
    queueing: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt_orchestrator = None
    if digital_twin_orchestration:
        from digital_twin import DigitalTwinTrustOrchestrator, evaluate_runtime_trust

        dt_orchestrator = DigitalTwinTrustOrchestrator(
            trust_threshold=trust_threshold,
            priority_qos=priority_qos,
        )
    else:
        from digital_twin import evaluate_runtime_trust

    # Finite service capacity: every execution location is an FCFS server, so
    # the realised routing decisions determine the queueing delay (Eq. 11).
    queue_model = NetworkQueueingModel() if queueing else None
    wants_context = _policy_wants_context(policy_fn)

    completed = set()
    records = []
    for index, (_, task) in enumerate(tasks.iterrows()):
        context = queue_model.context(task, index) if queue_model is not None else {}
        original_action = int(policy_fn(task, context) if wants_context else policy_fn(task))
        action = original_action
        dt_decision = None
        if dt_orchestrator is not None:
            action, dt_decision = dt_orchestrator.decide(task, original_action, context)

        runtime = simulate_task_execution(
            task,
            action,
            rng,
            bandwidth_simulator=bandwidth_simulator,
            priority_qos=priority_qos,
        )

        queue_wait_ms = 0.0
        if queue_model is not None:
            queue_wait_ms = queue_model.step(task, action, runtime["exec_delay_ms"], index)
            if queue_wait_ms > 0.0:
                runtime["exec_delay_ms"] = float(
                    np.clip(runtime["exec_delay_ms"] + queue_wait_ms, 3.0, 300.0)
                )
                runtime["deadline_met"] = int(
                    runtime["deadline_met"] == 1
                    and runtime["exec_delay_ms"] <= float(task["deadline_ms"])
                )
        runtime["queue_wait_ms"] = float(queue_wait_ms)

        deps = task.get("dependency_ids", [])
        if not isinstance(deps, list):
            deps = []
        dep_met = all(int(dep) in completed for dep in deps) if deps else True
        if dependency_aware and not dep_met:
            runtime["exec_delay_ms"] = float(np.clip(runtime["exec_delay_ms"] + 8.0, 3.0, 300.0))
            # Waiting for unfinished predecessors can only add delay. A task
            # that already failed (injected failure) or already missed its
            # deadline must therefore stay unfinished; recomputing the flag
            # from the delay alone would silently erase that failure.
            runtime["deadline_met"] = int(
                runtime["deadline_met"] == 1
                and runtime["exec_delay_ms"] <= float(task["deadline_ms"])
            )

        if dt_orchestrator is not None:
            dt_runtime = dt_orchestrator.observe(task, original_action, action, runtime, dt_decision or {})
        else:
            dt_runtime = evaluate_runtime_trust(
                task,
                original_action,
                action,
                runtime,
                trust_threshold=trust_threshold,
            )

        records.append(
            {
                "task_id": int(task["task_id"]),
                "algorithm": algorithm_name,
                "action_id": action,
                "action": ACTION_MAP[action],
                "digital_twin_orchestration": int(digital_twin_orchestration),
                "reference_action": int(task["reference_action"]),
                "priority_score": float(task["priority_score"]),
                "high_priority": int(task["high_priority"]),
                "dependency_count": int(task.get("dependency_count", 0)),
                "dependency_met": int(dep_met),
                "deadline_ms": float(task["deadline_ms"]),
                **runtime,
                **dt_runtime,
            }
        )
        if runtime["deadline_met"] == 1:
            completed.add(int(task["task_id"]))

    records_df = pd.DataFrame(records)
    if queue_model is not None:
        for key, value in queue_model.summary().items():
            records_df[key] = value
    return records_df


def _edge_hourly_energy_from_util(edge_cpu_bw_util: float) -> float:
    # Low-power edge node model: P = 5.5 + 15.0 * utilization (W)
    power_w = 5.5 + 15.0 * edge_cpu_bw_util
    return power_w * 3600 / 1000.0


def calculate_metrics(records: pd.DataFrame, fuzzy_accuracy: float | None = None) -> Dict[str, float]:
    if records.empty:
        raise ValueError("records dataframe is empty")

    avg_delay = float(records["exec_delay_ms"].mean())
    avg_energy = float(records["exec_energy_kj"].mean())
    avg_queue_wait = (
        float(records["queue_wait_ms"].mean()) if "queue_wait_ms" in records.columns else 0.0
    )

    high = records[records["high_priority"] == 1]
    if len(high) == 0:
        high_completion = 1.0
    else:
        # 计算实际完成率
        high_completion = float(high["deadline_met"].mean())

    edge_records = records[records["action_id"].isin(EDGE_ACTION_IDS)]
    if len(edge_records) == 0:
        edge_cpu = 0.0
        edge_bw = 0.0
        edge_cpu_bw = 0.0
        edge_hourly_energy = 0.0
    else:
        edge_cpu = float(edge_records["exec_cpu_util"].mean())
        edge_bw = float(edge_records["bandwidth_utilization"].mean())
        edge_cpu_bw = float(((edge_records["exec_cpu_util"] + edge_records["bandwidth_utilization"]) / 2).mean())
        edge_hourly_energy = _edge_hourly_energy_from_util(edge_cpu_bw)

    dep_sensitive = records[records["dependency_count"] > 0]
    if len(dep_sensitive) == 0:
        dependency_success = 1.0
    else:
        dependency_success = float(dep_sensitive["dependency_met"].mean())
    
    # 计算总体成功率
    success_rate = float(records["deadline_met"].mean())

    if "dtt_score" in records.columns:
        avg_dtt = float(records["dtt_score"].mean())
        trust_violation_rate = float(records["trust_violation"].mean()) * 100.0
        if "reorchestrated" in records.columns:
            reorchestration_count = int(records["reorchestrated"].sum())
            reorchestration_rate = float(records["reorchestrated"].mean()) * 100.0
        else:
            reorchestration_count = 0
            reorchestration_rate = 0.0
        if "dt_recovered" in records.columns:
            dt_recovered_count = int(records["dt_recovered"].sum())
            dt_recovered_rate = float(records["dt_recovered"].mean()) * 100.0
        else:
            dt_recovered_count = 0
            dt_recovered_rate = 0.0
        avg_recovery_time = float(records.get("dt_recovery_time_ms", pd.Series([0.0])).mean())
        if len(high) == 0:
            high_trusted_completion = 1.0
        else:
            high_trusted_completion = float(((high["deadline_met"] == 1) & (high["trust_violation"] == 0)).mean())
    else:
        avg_dtt = np.nan
        trust_violation_rate = np.nan
        reorchestration_count = 0
        reorchestration_rate = 0.0
        dt_recovered_count = 0
        dt_recovered_rate = 0.0
        avg_recovery_time = 0.0
        high_trusted_completion = high_completion

    fuzzy_score = round((fuzzy_accuracy if fuzzy_accuracy is not None else np.nan) * 100, 4) if fuzzy_accuracy is not None else np.nan

    return {
        "task_count": int(len(records)),
        "avg_delay_ms": round(avg_delay, 4),
        "avg_queue_wait_ms": round(avg_queue_wait, 4),
        "avg_energy_kj": round(avg_energy, 6),
        "success_rate": round(success_rate, 4),
        "advanced_task_priority_rate": round(high_completion * 100, 4),
        "high_priority_completion_rate": round(high_completion * 100, 4),
        "dependency_aware_success_rate": round(dependency_success * 100, 4),
        "fuzzy_rule_consistency": fuzzy_score,
        # Backward-compatible alias for older CSVs and plotting scripts.
        "fuzzy_accuracy": fuzzy_score,
        "edge_hourly_energy_kj": round(edge_hourly_energy, 4),
        "edge_cpu_utilization": round(edge_cpu * 100, 4),
        "edge_bandwidth_utilization": round(edge_bw * 100, 4),
        "edge_cpu_bw_utilization": round(edge_cpu_bw * 100, 4),
        "avg_dtt_score": round(avg_dtt, 4) if not pd.isna(avg_dtt) else np.nan,
        "trust_violation_rate": round(trust_violation_rate, 4) if not pd.isna(trust_violation_rate) else np.nan,
        "high_priority_trusted_completion_rate": round(high_trusted_completion * 100, 4),
        "reorchestration_count": reorchestration_count,
        "reorchestration_rate": round(reorchestration_rate, 4),
        "dt_recovered_count": dt_recovered_count,
        "dt_recovered_rate": round(dt_recovered_rate, 4),
        "avg_recovery_time_ms": round(avg_recovery_time, 4),
    }


def compare_with_baselines(summary_df: pd.DataFrame, target_algo: str = "FedDQL") -> pd.DataFrame:
    row = summary_df[summary_df["algorithm"] == target_algo]
    if row.empty:
        raise ValueError(f"{target_algo} not found in summary dataframe")

    fed_delay = float(row.iloc[0]["avg_delay_ms"])
    fed_energy = float(row.iloc[0]["avg_energy_kj"])

    baseline_df = summary_df[summary_df["algorithm"] != target_algo].copy()
    baseline_df["delay_improvement_pct"] = (
        (baseline_df["avg_delay_ms"] - fed_delay) / baseline_df["avg_delay_ms"] * 100
    )
    baseline_df["energy_improvement_pct"] = (
        (baseline_df["avg_energy_kj"] - fed_energy) / baseline_df["avg_energy_kj"] * 100
    )
    return baseline_df


def save_records(records: pd.DataFrame, filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    records.to_csv(path, index=False)
    return path


class EnergyMonitor:
    """边缘节点能耗监测器"""
    
    def __init__(self):
        """初始化能耗监测器"""
        self.energy_logs = []
        self.start_time = None
    
    def start_monitoring(self):
        """开始监测"""
        import time
        self.start_time = time.time()
    
    def log_energy(self, energy_kj, timestamp=None):
        """记录能耗数据"""
        if timestamp is None:
            import time
            timestamp = time.time() - (self.start_time if self.start_time else 0)
        
        self.energy_logs.append({
            'timestamp': timestamp,  # 小时
            'energy_kj': energy_kj
        })
    
    def save_logs(self, output_dir="results"):
        """保存能耗监测日志"""
        import os
        import pandas as pd
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建DataFrame并保存
        logs_df = pd.DataFrame(self.energy_logs)
        logs_path = os.path.join(output_dir, "energy_monitoring_logs.csv")
        logs_df.to_csv(logs_path, index=False)
        print(f"能耗监测日志已保存至: {logs_path}")
        
        # 生成能耗监测图表
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(logs_df['timestamp'], logs_df['energy_kj'], 'b-', linewidth=2)
        plt.axhline(y=80, color='r', linestyle='--', label='80KJ Target')
        plt.title('Edge Node Energy Consumption Over Time')
        plt.xlabel('Time (hours)')
        plt.ylabel('Energy Consumption (KJ)')
        plt.legend()
        plt.grid(True)
        
        chart_path = os.path.join(output_dir, "energy_consumption_over_time.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"能耗监测图表已保存至: {chart_path}")


def split_dataset(
    tasks: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into training, validation, and test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    total = len(tasks)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    shuffled = tasks.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_set = shuffled.iloc[:train_size].copy()
    val_set = shuffled.iloc[train_size:train_size + val_size].copy()
    test_set = shuffled.iloc[train_size + val_size:].copy()

    print(f"数据集划分完成：训练集 {len(train_set)} 任务 ({train_ratio*100:.0f}%), "
          f"验证集 {len(val_set)} 任务 ({val_ratio*100:.0f}%), "
          f"测试集 {len(test_set)} 任务 ({test_ratio*100:.0f}%)")

    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Multi-objective reward (manuscript, Section II-E / Eq. "optimization objective")
# ---------------------------------------------------------------------------
#
#   r_i = clip( w3 * C_i^hp + w4 * DTT_i
#               - w1 * T~_i - w2 * E~_i - w5 * P_i^vio , r_min, r_max )
#
# Latency is normalised by the *task deadline* rather than by a fixed reference
# latency.  The deadline is the quantity the objective actually constrains, and
# the released workload works on a 8--39 ms deadline scale, so normalising by an
# unrelated 100 ms constant compresses the whole operating range into a few
# hundredths of a reward unit: the learner then cannot tell a 26 ms execution
# from a 200 ms one, which is exactly the degeneracy the release suffered from.
# With T~ = T/tau a task that finishes exactly at its deadline has unit cost and
# finishing late is penalised in proportion to how late it is.
ENERGY_REFERENCE_KJ: float = 0.10
REWARD_TRUST_THRESHOLD: float = 0.80
REWARD_CLIP: Tuple[float, float] = (-5.0, 5.0)

DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "latency": 0.25,
    "energy": 0.15,
    "completion": 0.25,
    "trust": 0.20,
    "violation": 0.15,
}


def calculate_reward(
    task: pd.Series,
    action: int,
    exec_result: Dict[str, float],
    weights: Dict[str, float] | None = None,
) -> float:
    """Five-term multi-objective reward of the manuscript.

    ``weights`` overrides ``DEFAULT_REWARD_WEIGHTS``; this is what the
    reward-weight sensitivity sweep varies.  ``action`` is accepted for
    interface stability and is not used: the reward is a function of the
    realised outcome, not of the identity of the chosen location.
    """

    w = dict(DEFAULT_REWARD_WEIGHTS)
    if weights:
        w.update(weights)

    delay = float(exec_result.get("exec_delay_ms", 0.0))
    deadline = max(float(task.get("deadline_ms", 100.0)), 1e-6)
    energy = float(exec_result.get("exec_energy_kj", 0.0))
    deadline_met = int(exec_result.get("deadline_met", 0))
    high_priority = bool(task.get("high_priority", 0))
    dtt = float(exec_result.get("dtt_score", 0.0))

    t_norm = delay / deadline
    e_norm = energy / ENERGY_REFERENCE_KJ
    c_hp = 1.0 if (high_priority and deadline_met) else 0.0
    p_vio = 1.0 if delay > deadline else 0.0
    if "dtt_score" in exec_result and dtt < REWARD_TRUST_THRESHOLD:
        p_vio += 1.0

    total = (
        w["completion"] * c_hp
        + w["trust"] * dtt
        - w["latency"] * t_norm
        - w["energy"] * e_norm
        - w["violation"] * p_vio
    )
    return float(np.clip(total, *REWARD_CLIP))

