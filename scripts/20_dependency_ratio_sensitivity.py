from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "dt_feddql_result.csv"
OUT_DIR = ROOT / "outputs"


def run_dependency_ratio_scan(df: pd.DataFrame) -> pd.DataFrame:
    """Stress-test the dependency-aware policy under denser DAG relations.

    The real seismic workload contains a small fraction of dependent tasks.
    To check whether the DAG-aware modeling remains stable when more tasks are
    chained, this script increases the dependent-task ratio while preserving
    the original delay, energy, priority, and trust traces. A deterministic
    dependency waiting term is added to the selected tasks to emulate additional
    predecessor-release latency.
    """

    rng = np.random.default_rng(42)
    rows = []
    base_delay = df["exec_delay_ms"].to_numpy(float)
    base_energy = df["exec_energy_kj"].to_numpy(float)
    deadline = df["deadline_ms"].to_numpy(float)
    data_size = df["data_in_mb"].to_numpy(float)
    base_dtt = df["dtt_score"].to_numpy(float)
    base_violation = df["trust_violation"].to_numpy(bool)
    high_priority = df["high_priority"].to_numpy(bool)

    for ratio in [0.02, 0.10, 0.30, 0.50]:
        n_tasks = len(df)
        n_dependent = int(round(n_tasks * ratio))
        selected = rng.choice(n_tasks, n_dependent, replace=False)

        wait = np.zeros(n_tasks)
        wait[selected] = 2.0 + 0.06 * deadline[selected] + 0.35 * data_size[selected] + 2.0 * ratio

        delay = base_delay + wait
        energy = base_energy + 0.0002 * (wait > 0)
        dtt = np.clip(base_dtt - 0.003 * wait, 0.0, 1.0)
        extra_violation = (delay > deadline) | (dtt < 0.80)
        violation = base_violation | extra_violation

        rows.append(
            {
                "dependent_task_ratio_percent": round(100.0 * ratio, 1),
                "dependent_task_count": n_dependent,
                "avg_latency_ms": round(float(delay.mean()), 2),
                "avg_energy_kj": round(float(energy.mean()), 4),
                "high_priority_completion_percent": round(float((~violation[high_priority]).mean() * 100.0), 2),
                "avg_dtt_score": round(float(dtt.mean()), 3),
                "trust_violation_percent": round(float(violation.mean() * 100.0), 2),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULT_PATH)
    out = run_dependency_ratio_scan(df)
    out_path = OUT_DIR / "exp20_dependency_ratio_sensitivity.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

