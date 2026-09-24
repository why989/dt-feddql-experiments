"""Consistency checks for the released trace and the released policy.

Three independent checks that a reviewer is likely to run:

1. Does the released task set plus the released federated weights reproduce the
   released DT-FedDQL trace, metric by metric?

2. Is the released policy distinguishable from a trivial "offload everything to
   the strongest edge node" policy?  Under a capacity-constrained, heterogeneous
   edge tier the two must differ, otherwise the reported numbers could be
   obtained without any learning at all.

3. Does the learned policy actually condition its choice on the live node load?
   The check regresses the action-selection margin ``Q(e2) - Q(e1)`` on the
   measured load margin ``rho(e1) - rho(e2)``: a load-aware policy has a
   positive association, a degenerate one has none.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import dag_utils as du  # noqa: E402
from algorithm_utils import ACTION_ID, calculate_metrics, run_policy_on_tasks  # noqa: E402
from deployment_model import ACTION_MAP, EDGE_ACTION_IDS, NetworkQueueingModel  # noqa: E402
from models.federated_learning_numpy import get_state  # noqa: E402

OUT_DIR = ROOT / "outputs"
RELEASED = ROOT / "results" / "dt_feddql_result.csv"

METRIC_KEYS = (
    "avg_delay_ms",
    "avg_energy_kj",
    "high_priority_completion_rate",
    "avg_dtt_score",
    "trust_violation_rate",
)


def policy_margin(trace: pd.DataFrame, model) -> pd.DataFrame:
    """Live-load trace of the greedy margin between edge_2 and edge_1."""

    queue = NetworkQueueingModel()
    rows = []
    for index, (_, task) in enumerate(trace.iterrows()):
        context = queue.context(task, index)
        q = model.forward(get_state(task, context).reshape(1, -1))[0]
        action = int(np.argmax(q))
        rows.append(
            {
                "action": action,
                "q_edge_1": float(q[EDGE_ACTION_IDS[0]]),
                "q_edge_2": float(q[EDGE_ACTION_IDS[1]]),
                "load_edge_1": float(context["load_edge_1"]),
                "load_edge_2": float(context["load_edge_2"]),
            }
        )
        queue.step(task, action, 22.0, index)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trace = du.load_base_trace()
    model = du.load_trained_model()
    released = pd.read_csv(RELEASED)

    records, metrics = du.evaluate_policy(trace, model, seed=42)
    released_metrics = {key: float(released[key].iloc[0]) for key in METRIC_KEYS}

    # -- 2. learned policy versus the strongest-edge heuristic ---------------
    cycle = itertools.cycle(EDGE_ACTION_IDS)
    strongest = run_policy_on_tasks(
        trace,
        lambda task: ACTION_ID["edge_1"],
        "all-strongest-edge",
        seed=42,
        dependency_aware=True,
        priority_qos=True,
        digital_twin_orchestration=True,
        trust_threshold=0.80,
    )
    # A round-robin reference separates "load awareness" from "any spread at all".
    cycle = itertools.cycle(EDGE_ACTION_IDS)
    round_robin = run_policy_on_tasks(
        trace,
        lambda task, ctx=None: next(cycle),
        "round-robin-edge",
        seed=42,
        dependency_aware=True,
        priority_qos=True,
        digital_twin_orchestration=True,
        trust_threshold=0.80,
    )
    metrics_strongest = calculate_metrics(strongest)
    metrics_rr = calculate_metrics(round_robin)

    # -- 3. is the decision load-conditioned? --------------------------------
    margins = policy_margin(trace, model)
    q_margin = margins["q_edge_2"] - margins["q_edge_1"]
    load_margin = margins["load_edge_1"] - margins["load_edge_2"]
    association = float(np.corrcoef(q_margin, load_margin)[0, 1])

    rows = [
        {
            "check": f"released {key}",
            "released_value": released_metrics[key],
            "recomputed_value": float(metrics[key]),
            "match": bool(abs(released_metrics[key] - float(metrics[key])) < 1e-6),
        }
        for key in METRIC_KEYS
    ]
    rows.append(
        {
            "check": "per-task delay identical to all-strongest-edge policy",
            "released_value": np.nan,
            "recomputed_value": float(
                np.max(np.abs(records["exec_delay_ms"].to_numpy() - strongest["exec_delay_ms"].to_numpy()))
            ),
            "match": bool(np.allclose(records["exec_delay_ms"], strongest["exec_delay_ms"])),
        }
    )
    rows.append(
        {
            "check": "trust decisions identical to all-strongest-edge policy",
            "released_value": np.nan,
            "recomputed_value": float((records["trust_violation"] != strongest["trust_violation"]).sum()),
            "match": bool((records["trust_violation"] == strongest["trust_violation"]).all()),
        }
    )
    rows.append(
        {
            "check": "corr(Q(e2)-Q(e1), rho(e1)-rho(e2))",
            "released_value": np.nan,
            "recomputed_value": association,
            "match": bool(association > 0.1),
        }
    )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "exp22_release_consistency_check.csv", index=False)

    argmax_counts = margins["action"].value_counts().to_dict()
    summary = pd.DataFrame(
        [
            {
                "quantity": "global model argmax action counts",
                "value": str({ACTION_MAP[a]: int(argmax_counts.get(a, 0)) for a in range(len(ACTION_MAP))}),
            },
            {
                "quantity": "tasks routed by the learned network to edge_1",
                "value": f"{int(argmax_counts.get(ACTION_ID['edge_1'], 0))}/{len(margins)}",
            },
            {
                "quantity": "DT-FedDQL latency (ms)",
                "value": metrics["avg_delay_ms"],
            },
            {
                "quantity": "all-strongest-edge latency (ms)",
                "value": metrics_strongest["avg_delay_ms"],
            },
            {
                "quantity": "round-robin-edge latency (ms)",
                "value": metrics_rr["avg_delay_ms"],
            },
            {
                "quantity": "DT-FedDQL high-priority completion (%)",
                "value": metrics["high_priority_completion_rate"],
            },
            {
                "quantity": "all-strongest-edge high-priority completion (%)",
                "value": metrics_strongest["high_priority_completion_rate"],
            },
        ]
    )
    summary.to_csv(OUT_DIR / "exp22_policy_degeneracy_summary.csv", index=False)

    print(table.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print(f"Saved: {OUT_DIR / 'exp22_release_consistency_check.csv'}")
    print(f"Saved: {OUT_DIR / 'exp22_policy_degeneracy_summary.csv'}")


if __name__ == "__main__":
    main()
