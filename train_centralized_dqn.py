"""Train the centralized single-DQN baseline used in the manuscript.

The manuscript's "Centralized DQN" baseline is a *learning* baseline: one DQN
agent with a centralized state observation, trained on the full training
workload with the same architecture, reward, and environment as the federated
agents.  Only two ingredients are removed relative to DT-FedDQL:

* no federated aggregation (a single agent sees the whole training set), and
* no digital-twin trustworthy re-orchestration (evaluation uses the raw
  argmax policy).

The state vector is identical to the federated agents' state and therefore
carries the live load of *every* execution location -- this is the
"centralized state observation" the paper describes.

Run from the repository root, after ``train_released_model.py``:

    python train_centralized_dqn.py [--rounds 40] [--local-epochs 2]

Produces ``results/centralized_dqn_weights.pkl`` and prints the argmax action
distribution plus the headline evaluation metrics so the non-degeneracy of
the baseline can be checked directly.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from algorithm_utils import (  # noqa: E402
    ACTION_MAP,
    calculate_metrics,
    load_task_set,
    prepare_task_dataframe,
    run_policy_on_tasks,
    simulate_task_execution,
    split_dataset,
)
from deployment_model import NUM_ACTIONS, NetworkQueueingModel  # noqa: E402
from models.federated_learning_numpy import LocalAgent, get_state  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_centralized(tasks: pd.DataFrame, rounds: int, local_epochs: int,
                      verbose: bool = False) -> LocalAgent:
    """Train one DQN agent on the *entire* training workload.

    The per-round budget matches the federated configuration: each federated
    agent runs ``local_epochs`` passes over its shard per round, so the
    centralized agent runs the same number of passes over the full training
    set per round.  Epsilon decays once per round on the same schedule.
    """

    agent = LocalAgent(15, NUM_ACTIONS, 0)
    for round_index in range(rounds):
        if verbose:
            agent.train(tasks, local_epochs)
            print(f"round {round_index + 1}/{rounds} done, epsilon={agent.epsilon:.4f}")
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                agent.train(tasks, local_epochs)
    return agent


def argmax_distribution(model, tasks: pd.DataFrame) -> dict:
    """Argmax action histogram under the same capacity pressure as evaluation."""

    rng = np.random.default_rng(42)
    queue_model = NetworkQueueingModel()
    counts = {ACTION_MAP[a]: 0 for a in ACTION_MAP}
    for index, (_, task) in enumerate(tasks.iterrows()):
        state = get_state(task, queue_model.context(task, index))
        action = int(np.argmax(model.forward(state.reshape(1, -1))[0]))
        counts[ACTION_MAP[action]] += 1
        exec_result = simulate_task_execution(task, action, rng, priority_qos=False)
        queue_model.step(task, action, exec_result["exec_delay_ms"], index)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--out", type=str,
                        default=os.path.join(RESULTS_DIR, "centralized_dqn_weights.pkl"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    raw = load_task_set("real")
    train_raw, _, _ = split_dataset(raw, 0.7, 0.2, 0.1, 42)
    train_set = prepare_task_dataframe(train_raw, target_tasks=len(train_raw), seed=42)
    full = prepare_task_dataframe(raw, target_tasks=3000, seed=42)

    print(f"train tasks={len(train_set)}  evaluation trace={len(full)}")
    started = time.perf_counter()
    agent = train_centralized(train_set, args.rounds, args.local_epochs,
                              verbose=args.verbose)
    elapsed = time.perf_counter() - started
    print(f"training wall clock: {elapsed:.1f} s "
          f"({args.rounds} rounds x {args.local_epochs} epochs, single agent)")

    weights = agent.model.get_weights()
    with open(args.out, "wb") as handle:
        pickle.dump(weights, handle)
    print(f"weights -> {args.out}")

    dist = argmax_distribution(agent.model, full)
    total = sum(dist.values())
    print("centralized-DQN argmax distribution:",
          {k: f"{100.0 * v / total:.2f}%" for k, v in dist.items()})

    def policy(task, loads):
        loads = loads or {}
        if bool(task.get("high_priority", 0)):
            from deployment_model import EDGE_ACTION_IDS
            return int(min(EDGE_ACTION_IDS, key=lambda a: float(loads.get(f"load_edge_{a}", 0.0))))
        state = get_state(task, loads)
        return int(np.argmax(agent.model.forward(state.reshape(1, -1))[0]))

    records = run_policy_on_tasks(
        full, policy, algorithm_name="Centralized-DQN", seed=42,
        dependency_aware=True, priority_qos=True,
        digital_twin_orchestration=False,
    )
    metrics = calculate_metrics(records)
    share = (records["action"].value_counts(normalize=True) * 100).round(2).to_dict()
    print(f"Centralized DQN   delay={metrics['avg_delay_ms']:.3f} "
          f"qwait={metrics['avg_queue_wait_ms']:.3f} "
          f"energy={metrics['avg_energy_kj']:.5f} "
          f"HPC={metrics['high_priority_completion_rate']:.2f} "
          f"DTT={metrics['avg_dtt_score']:.4f} "
          f"vio={metrics['trust_violation_rate']:.2f} dist={share}")


if __name__ == "__main__":
    main()
