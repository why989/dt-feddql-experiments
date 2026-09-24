"""Dependency-ratio sensitivity of the DT-FedDQL policy.

The released seismic workload contains dependent tasks only in single-predecessor
chains.  To test whether the dependency-aware design stays stable when a larger
share of the workload is chained, this script **rebuilds** the dependency graph
of the released trace at several dependent-task ratios and **re-runs** the
released DT-FedDQL policy on the re-wired tasks.

Nothing here is extrapolated from the released metrics: every reported value is
produced by the simulator on a re-wired task set, and the branch width is held
at one predecessor per dependent task so that only the *ratio* changes.
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

OUT_DIR = ROOT / "outputs"

RATIOS = (0.02, 0.10, 0.30, 0.50)
WIDTH = 1
SEEDS = (42, 43, 44, 45, 46)


def _run_once(tasks: pd.DataFrame, model, seed: int) -> dict:
    records, metrics = du.evaluate_policy(tasks, model, seed=seed)
    high = records[records["high_priority"] == 1]
    return {
        "avg_latency_ms": metrics["avg_delay_ms"],
        "avg_energy_kj": metrics["avg_energy_kj"],
        "high_priority_completion_percent": metrics["high_priority_completion_rate"],
        "avg_dtt_score": metrics["avg_dtt_score"],
        "trust_violation_percent": metrics["trust_violation_rate"],
        "high_priority_latency_ms": round(float(high["exec_delay_ms"].mean()), 4) if len(high) else np.nan,
        "tasks_with_unmet_predecessor": int((records["dependency_met"] == 0).sum()),
    }


def _summarise(tasks: pd.DataFrame, model, ratio_label: dict) -> dict:
    stats = du.graph_statistics(tasks)
    frame = pd.DataFrame([_run_once(tasks, model, seed) for seed in SEEDS])
    row = {
        **ratio_label,
        "dependent_task_ratio_percent": stats["dependent_task_ratio_percent"],
        "dependent_task_count": stats["dependent_task_count"],
        "seeds": len(SEEDS),
    }
    for column in frame.columns:
        row[column] = round(float(frame[column].mean()), 4)
        row[f"{column}_std"] = round(float(frame[column].std(ddof=1)), 4)
    return row


def run_dependency_ratio_scan(model, base_trace: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _summarise(
            base_trace,
            model,
            {"scan_type": "released_graph", "target_ratio_percent": np.nan},
        )
    ]
    for ratio in RATIOS:
        rewired = du.rewire_dependencies(base_trace, ratio, WIDTH)
        rows.append(
            _summarise(
                rewired,
                model,
                {
                    "scan_type": "dependency_ratio",
                    "target_ratio_percent": round(100.0 * ratio, 1),
                },
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = du.load_trained_model()
    base_trace = du.load_base_trace()

    out = run_dependency_ratio_scan(model, base_trace)
    out_path = OUT_DIR / "exp20_dependency_ratio_sensitivity.csv"
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
