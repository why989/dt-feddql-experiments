"""Train and release the federated global model.

This is the single entry point that produces
``results/fed_dql_federated_model_weights.pkl`` -- the weights used by every
other script in this repository and by ``results/dt_feddql_result.csv``.

Run from the repository root:

    python train_released_model.py [--rounds 40] [--local-epochs 2]

The configuration matches the "Learning configuration" row of the manuscript.
The learner trains on the capacity-constrained, heterogeneous deployment
defined in ``deployment_model.py``: the three edge nodes differ in clock
frequency and uplink bandwidth, each execution location is a finite server, and
the state vector carries the live utilisation of every location.
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
from models.federated_learning_numpy import (  # noqa: E402
    FederatedServer,
    LocalAgent,
    fed_dql_policy_with_federated_learning,
    get_state,
)

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train(tasks: pd.DataFrame, rounds: int, local_epochs: int, agents: int = 3,
          verbose: bool = False) -> FederatedServer:
    server = FederatedServer(15, NUM_ACTIONS)
    for index in range(agents):
        server.add_local_agent(LocalAgent(15, NUM_ACTIONS, index))

    chunks = np.array_split(tasks, agents)
    for round_index in range(rounds):
        for agent, chunk in zip(server.local_agents, chunks):
            if verbose:
                agent.train(chunk, local_epochs)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    agent.train(chunk, local_epochs)
        with contextlib.redirect_stdout(io.StringIO()):
            server.federated_averaging()
        if verbose:
            print(f"round {round_index + 1}/{rounds} done")
    return server


def argmax_distribution(model, tasks: pd.DataFrame) -> dict:
    """Argmax action histogram of the released network on the evaluation trace.

    The virtual queue clocks are driven with the *actual* simulated service time
    of the chosen action, so the histogram is produced under the same capacity
    pressure as the evaluation rollouts rather than under a fixed constant.
    """

    rng = np.random.default_rng(42)
    queue_model = NetworkQueueingModel()
    counts = {ACTION_MAP[a]: 0 for a in ACTION_MAP}
    for index, (_, task) in enumerate(tasks.iterrows()):
        state = get_state(task, queue_model.context(task, index))
        action = int(np.argmax(model.forward(state.reshape(1, -1))[0]))
        counts[ACTION_MAP[action]] += 1
        exec_result = simulate_task_execution(task, action, rng, priority_qos=True)
        queue_model.step(task, action, exec_result["exec_delay_ms"], index)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--out", type=str,
                        default=os.path.join(RESULTS_DIR, "fed_dql_federated_model_weights.pkl"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    raw = load_task_set("real")
    train_raw, _, _ = split_dataset(raw, 0.7, 0.2, 0.1, 42)
    train_set = prepare_task_dataframe(train_raw, target_tasks=len(train_raw), seed=42)
    full = prepare_task_dataframe(raw, target_tasks=3000, seed=42)

    print(f"train tasks={len(train_set)}  evaluation trace={len(full)}")
    started = time.perf_counter()
    server = train(train_set, args.rounds, args.local_epochs, verbose=args.verbose)
    elapsed = time.perf_counter() - started
    print(f"training wall clock: {elapsed:.1f} s ({args.rounds} rounds x {args.local_epochs} epochs)")

    weights = server.global_model.get_weights()
    with open(args.out, "wb") as handle:
        pickle.dump(weights, handle)
    print(f"weights -> {args.out}")

    # Persist the per-round reward/loss/delay trace so that the convergence
    # figure (fig01) and the training-overhead diagnostics are reproducible
    # from the released repository alone.
    with contextlib.redirect_stdout(io.StringIO()):
        server.save_training_logs(RESULTS_DIR)
    print(f"training logs -> {os.path.join(RESULTS_DIR, 'training_logs.csv')}")

    dist = argmax_distribution(server.global_model, full)
    total = sum(dist.values())
    print("released-model argmax distribution:",
          {k: f"{100.0 * v / total:.2f}%" for k, v in dist.items()})

    def policy(task, loads):
        return fed_dql_policy_with_federated_learning(task, server, loads)

    rows = []
    for name, kwargs in (
        ("FedDQL-Federated", dict(digital_twin_orchestration=False)),
        ("DT-FedDQL", dict(digital_twin_orchestration=True, trust_threshold=0.80)),
    ):
        records = run_policy_on_tasks(
            full, policy, algorithm_name=name, seed=42,
            dependency_aware=True, priority_qos=True, **kwargs,
        )
        metrics = calculate_metrics(records)
        share = (records["action"].value_counts(normalize=True) * 100).round(2).to_dict()
        rows.append({"algorithm": name, **metrics})
        print(f"{name:18s} delay={metrics['avg_delay_ms']:.3f} qwait={metrics['avg_queue_wait_ms']:.3f} "
              f"energy={metrics['avg_energy_kj']:.5f} HPC={metrics['high_priority_completion_rate']:.2f} "
              f"DTT={metrics['avg_dtt_score']:.4f} vio={metrics['trust_violation_rate']:.2f} dist={share}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "train_release_summary.csv"), index=False)


if __name__ == "__main__":
    main()
