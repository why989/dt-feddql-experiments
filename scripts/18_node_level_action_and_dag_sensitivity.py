"""Node-level action distribution and DAG-width sensitivity.

Two different jobs:

1. ``exp18_node_level_action_distribution.csv`` reports the **actual** node-level
   routing of the released DT-FedDQL trace (``results/dt_feddql_result.csv``).
   The released action space is already node-level
   (``A = {L, e1, e2, e3, C}``), so the per-node histogram is a direct count of
   the recorded ``action`` labels -- no post-hoc rebalancing is applied.
2. ``exp18_dag_width_sensitivity.csv`` is **re-computed from scratch**: the
   dependency structure of the released task trace is rebuilt as a layered DAG
   with a controlled branch width and the released DT-FedDQL policy is re-run
   on the re-wired tasks.  No value in that table is transcribed or
   interpolated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import dag_utils as du  # noqa: E402

RESULT_PATH = ROOT / "results" / "dt_feddql_result.csv"
OUT_DIR = ROOT / "outputs"

WIDTHS = (1, 3, 5)
BASE_DEPENDENT_RATIO = 0.093
SEEDS = (42, 43, 44, 45, 46)

NODE_LABEL_MAP = {
    "terminal": "terminal-local",
    "edge_1": "edge-1",
    "edge_2": "edge-2",
    "edge_3": "edge-3",
    "cloud": "cloud",
}


def build_node_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count the released node-level actions per execution node."""

    node = (
        df["action"]
        .astype(str)
        .str.lower()
        .map(NODE_LABEL_MAP)
        .fillna(df["action"].astype(str))
    )
    work = df.copy()
    work["execution_node"] = node

    order = ["terminal-local", "edge-1", "edge-2", "edge-3", "cloud"]
    rows = []
    for name in order:
        sub = work[work["execution_node"] == name]
        if sub.empty:
            rows.append(
                {
                    "execution_node": name,
                    "task_count": 0,
                    "ratio_percent": 0.0,
                    "avg_delay_ms": np.nan,
                    "avg_energy_kj": np.nan,
                    "avg_cpu_util_percent": np.nan,
                    "avg_bandwidth_util_percent": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "execution_node": name,
                    "task_count": int(len(sub)),
                    "ratio_percent": round(len(sub) / len(work) * 100.0, 2),
                    "avg_delay_ms": round(float(sub["exec_delay_ms"].mean()), 4),
                    "avg_energy_kj": round(float(sub["exec_energy_kj"].mean()), 6),
                    "avg_cpu_util_percent": round(float(sub["exec_cpu_util"].mean() * 100.0), 2),
                    "avg_bandwidth_util_percent": round(float(sub["bandwidth_utilization"].mean() * 100.0), 2),
                }
            )
    return pd.DataFrame(rows)


def _run_once(tasks: pd.DataFrame, model, seed: int) -> dict:
    records, metrics = du.evaluate_policy(tasks, model, seed=seed)
    high = records[records["high_priority"] == 1]
    return {
        "avg_delay_ms": metrics["avg_delay_ms"],
        "avg_energy_kj": metrics["avg_energy_kj"],
        "high_priority_completion_rate": metrics["high_priority_completion_rate"],
        "avg_dtt_score": metrics["avg_dtt_score"],
        "trust_violation_rate": metrics["trust_violation_rate"],
        "high_priority_latency_ms": round(float(high["exec_delay_ms"].mean()), 4) if len(high) else np.nan,
        "tasks_with_unmet_predecessor": int((records["dependency_met"] == 0).sum()),
    }


def build_dag_width_scan(model, base_trace: pd.DataFrame) -> pd.DataFrame:
    """Re-run the DT-FedDQL policy on re-wired DAGs of increasing branch width.

    The dependent-task ratio is held at the level of the released workload so
    that only the branch width changes.  Each width is evaluated over
    ``SEEDS`` independent simulator seeds.
    """

    rows = []
    for width in WIDTHS:
        rewired = du.rewire_dependencies(base_trace, BASE_DEPENDENT_RATIO, width)
        stats = du.graph_statistics(rewired)

        runs = [_run_once(rewired, model, seed) for seed in SEEDS]
        frame = pd.DataFrame(runs)

        row = {
            "scan_type": "dag_width",
            "scale_value": width,
            "dependent_task_ratio_percent": stats["dependent_task_ratio_percent"],
            "dependent_task_count": stats["dependent_task_count"],
            "mean_predecessors_per_dependent_task": stats["mean_predecessors"],
            "max_predecessors": stats["max_predecessors"],
            "seeds": len(SEEDS),
        }
        for column in frame.columns:
            row[column] = round(float(frame[column].mean()), 4)
            row[f"{column}_std"] = round(float(frame[column].std(ddof=1)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_baseline_row(model, base_trace: pd.DataFrame) -> dict:
    """Released dependency graph, kept as the reference point of the scan."""

    stats = du.graph_statistics(base_trace)
    frame = pd.DataFrame([_run_once(base_trace, model, seed) for seed in SEEDS])
    row = {
        "scan_type": "released_graph",
        "scale_value": stats["max_predecessors"],
        "dependent_task_ratio_percent": stats["dependent_task_ratio_percent"],
        "dependent_task_count": stats["dependent_task_count"],
        "mean_predecessors_per_dependent_task": stats["mean_predecessors"],
        "max_predecessors": stats["max_predecessors"],
        "seeds": len(SEEDS),
    }
    for column in frame.columns:
        row[column] = round(float(frame[column].mean()), 4)
        row[f"{column}_std"] = round(float(frame[column].std(ddof=1)), 4)
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULT_PATH)

    node_distribution = build_node_distribution(df)
    node_path = OUT_DIR / "exp18_node_level_action_distribution.csv"
    node_distribution.to_csv(node_path, index=False)

    model = du.load_trained_model()
    base_trace = du.load_base_trace()
    width_scan = pd.DataFrame(
        [build_baseline_row(model, base_trace)]
        + build_dag_width_scan(model, base_trace).to_dict("records")
    )
    width_path = OUT_DIR / "exp18_dag_width_sensitivity.csv"
    width_scan.to_csv(width_path, index=False)

    print(node_distribution.to_string(index=False))
    print()
    print(width_scan.to_string(index=False))
    print(f"Saved: {node_path}")
    print(f"Saved: {width_path}")


if __name__ == "__main__":
    main()
