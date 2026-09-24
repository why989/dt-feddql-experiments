"""Regenerate every result file that the manuscript tables are built from.

This script is the bridge between the released artifacts and the ``results/``
and ``outputs/`` directories:

* ``results/dt_feddql_result.csv`` and ``results/fed_dql_federated_result.csv``
  are re-derived from ``results/fed_dql_federated_model_weights.pkl`` alone, so
  the released trace is reproducible rather than transcribed;
* every baseline in Table III is re-run through its own implementation in
  ``scripts/baseline_policies.py`` and stored in ``results/<baseline>.csv``;
* ``results/algorithm_metrics_comparison.csv`` is rebuilt from those traces, so
  the aggregated table cannot drift away from the per-task traces.

Run from the repository root, after ``train_released_model.py``:

    python build_release_results.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import dag_utils as du  # noqa: E402
from algorithm_utils import (  # noqa: E402
    ACTION_ID,
    calculate_metrics,
    load_task_set,
    prepare_task_dataframe,
    run_policy_on_tasks,
)
from baseline_policies import (  # noqa: E402
    FedServPolicy,
    GreedyPolicy,
    action_histogram,
    centralized_dqn_policy,
    make_fuzzy_dql_policy,
    milp_policy,
)
from models.federated_learning_numpy import (  # noqa: E402
    FederatedServer,
    LocalAgent,
    fed_dql_policy_with_federated_learning,
)
from models.fuzzy_classifier import FuzzyTaskClassifier  # noqa: E402
from deployment_model import EDGE_ACTION_IDS  # noqa: E402

RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"
SEED = 42


def restore_server(weights_path: Path) -> FederatedServer:
    """Rebuild a FederatedServer around the released global weights."""

    import pickle

    from deployment_model import NUM_ACTIONS

    server = FederatedServer(15, NUM_ACTIONS)
    server.add_local_agent(LocalAgent(15, NUM_ACTIONS, 0))
    with open(weights_path, "rb") as handle:
        weights = pickle.load(handle)
    server.global_model.set_weights(weights)
    return server


def paired_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    n = diff.size
    sd = float(diff.std(ddof=1))
    if sd == 0:
        return float("inf"), 0.0
    t = float(diff.mean()) / (sd / np.sqrt(n))
    try:
        from scipy import stats  # type: ignore

        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        p = float("nan")
    return t, p


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    weights_path = RESULTS / "fed_dql_federated_model_weights.pkl"
    if not weights_path.exists():
        raise SystemExit("run train_released_model.py first")

    raw = load_task_set("real")
    tasks = prepare_task_dataframe(raw, target_tasks=3000, seed=SEED)
    server = restore_server(weights_path)

    def policy(task, loads):
        return fed_dql_policy_with_federated_learning(task, server, loads)

    fuzzy = FuzzyTaskClassifier()
    fuzzy_eval = fuzzy.evaluate(tasks)
    fuzzy_accuracy = float(fuzzy_eval["accuracy"])

    # ------------------------------------------------------------------ our method
    traces = {}
    for name, kwargs, stem in (
        ("FedDQL-Federated", dict(digital_twin_orchestration=False), "fed_dql_federated_result"),
        ("DT-FedDQL", dict(digital_twin_orchestration=True, trust_threshold=0.80), "dt_feddql_result"),
    ):
        records = run_policy_on_tasks(
            tasks, policy, algorithm_name=name, seed=SEED,
            dependency_aware=True, priority_qos=True, **kwargs,
        )
        records["fuzzy_pred_action"] = fuzzy_eval["y_pred"]
        metrics = calculate_metrics(records, fuzzy_accuracy=fuzzy_accuracy)
        for key, value in metrics.items():
            records[key] = value
        records.to_csv(RESULTS / f"{stem}.csv", index=False)
        traces[name] = records
        print(f"{name:18s} delay={metrics['avg_delay_ms']:8.4f} energy={metrics['avg_energy_kj']:.6f} "
              f"HPC={metrics['high_priority_completion_rate']:7.4f} DTT={metrics['avg_dtt_score']:.4f} "
              f"vio={metrics['trust_violation_rate']:7.4f} {action_histogram(records)}")

    pd.DataFrame(
        [
            {"algorithm": "FedDQL-Federated", **calculate_metrics(traces["FedDQL-Federated"], fuzzy_accuracy=fuzzy_accuracy)},
            {"algorithm": "DT-FedDQL", **calculate_metrics(traces["DT-FedDQL"], fuzzy_accuracy=fuzzy_accuracy)},
        ]
    ).to_csv(RESULTS / "fed_dql_federated_summary.csv", index=False)

    # ------------------------------------------------------------------ baselines
    rng = np.random.default_rng(SEED)
    rr_cycle = itertools.cycle(EDGE_ACTION_IDS)
    runs = [
        ("Local Only", run_policy_on_tasks(tasks, lambda t: ACTION_ID["terminal"], "Local-Only"), None,
         "local_only_results"),
        ("Random Offloading",
         run_policy_on_tasks(tasks, lambda t: int(rng.choice(list(ACTION_ID.values())[:5])), "Random"),
         None, "random_offloading_result"),
        ("Greedy", run_policy_on_tasks(tasks, GreedyPolicy(), "Greedy"), None, "greedy_results"),
        ("FedServ", run_policy_on_tasks(tasks, FedServPolicy(), "FedServ"), None, "fedserv_result"),
        ("MILP", run_policy_on_tasks(tasks, milp_policy, "MILP"), None, "milp_results"),
        ("Centralized DQN", run_policy_on_tasks(tasks, centralized_dqn_policy, "Centralized-DQN",
                                                priority_qos=True), None,
         "centralized_dqn_results"),
        ("Fuzzy DQL", run_policy_on_tasks(tasks, make_fuzzy_dql_policy(), "Fuzzy-DQL"), fuzzy_accuracy,
         "fuzzy_dql_result"),
        ("Round-Robin", run_policy_on_tasks(
            tasks, lambda t, ctx=None: next(rr_cycle), "Round-Robin"), None,
         "round_robin_results"),
    ]

    dt = traces["DT-FedDQL"].sort_values("task_id").reset_index(drop=True)
    rows, tests = [], []
    for name, frame, fuzz, stem in runs:
        metrics = calculate_metrics(frame, fuzzy_accuracy=fuzz)
        rows.append({"algorithm": name, "task_count": int(len(frame)), **metrics})
        rec = frame.sort_values("task_id").reset_index(drop=True)
        t, p = paired_ttest(rec["exec_delay_ms"].to_numpy(float), dt["exec_delay_ms"].to_numpy(float))
        tests.append({"algorithm": name, "t_stat": round(t, 3), "p_value": p, "paired_n": int(len(rec))})
        rec.to_csv(RESULTS / f"{stem}.csv", index=False)
        print(f"{name:18s} delay={metrics['avg_delay_ms']:8.4f} energy={metrics['avg_energy_kj']:.6f} "
              f"HPC={metrics['high_priority_completion_rate']:7.4f} DTT={metrics['avg_dtt_score']:.4f} "
              f"vio={metrics['trust_violation_rate']:7.4f} {action_histogram(frame)}")

    headline = pd.DataFrame(
        [
            {"algorithm": name, "task_count": int(len(traces[name])),
             **calculate_metrics(traces[name], fuzzy_accuracy=fuzzy_accuracy)}
            for name in ("DT-FedDQL", "FedDQL-Federated")
        ]
    )
    combined = pd.concat([headline, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(RESULTS / "algorithm_metrics_comparison.csv", index=False)
    pd.DataFrame(tests).to_csv(OUTPUTS / "exp19_baseline_paired_ttests.csv", index=False)
    print("saved results/algorithm_metrics_comparison.csv")

    # ------------------------------------------------- node-level action summary
    # NOTE: ``scripts/18_...py`` owns ``exp18_node_level_action_distribution.csv``
    # (one row per node for DT-FedDQL).  The richer all-algorithm histogram is
    # written under a separate name so the two scripts cannot overwrite each
    # other.
    node_rows = []
    for name, frame in [("DT-FedDQL", traces["DT-FedDQL"]), ("FedDQL-Federated", traces["FedDQL-Federated"])] + [
        (row[0], row[1]) for row in runs
    ]:
        histogram = action_histogram(frame)
        node_rows.append({"algorithm": name, **{k: histogram.get(k, 0.0) for k in
                                                ("terminal", "edge_1", "edge_2", "edge_3", "cloud")},
                          "edge_utilisation_mean_percent": round(
                              float(frame["edge_utilisation_mean"].iloc[0]) * 100, 4),
                          "edge_rho_mean_percent": round(float(frame["edge_rho_mean"].iloc[0]) * 100, 4)})
    pd.DataFrame(node_rows).to_csv(OUTPUTS / "node_level_action_distribution_all_algorithms.csv", index=False)
    print("saved outputs/node_level_action_distribution_all_algorithms.csv")


if __name__ == "__main__":
    main()
