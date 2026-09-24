"""Summarize seismic workload provenance for the paper.

This script extracts workload statistics from the train/validation/test CSV
files and writes compact tables used to justify that the experiments are tied
to wireless seismic sensing rather than a generic IoT workload.

Dependency statistics are reported for two scopes, because they differ:

* ``concatenated_splits`` - the three released split files joined back
  together.  The split is taken on the *raw* records, so dependency links that
  cross a split boundary are dropped and the dependent-task share is
  understated.
* ``released_evaluation_trace`` - the 3000-task trace actually evaluated in the
  paper (``prepare_task_dataframe`` applied to the full released task set).
  This is the scope quoted in Table I of the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import dag_utils as du  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    df = pd.concat(
        [
            pd.read_csv(DATA / "train_set.csv"),
            pd.read_csv(DATA / "val_set.csv"),
            pd.read_csv(DATA / "test_set.csv"),
        ],
        ignore_index=True,
    )
    trace = du.load_base_trace()

    summary_rows = [
        {"scope": "concatenated_splits", "metric": "number_of_tasks", "value": len(df)},
        {"scope": "concatenated_splits", "metric": "number_of_stations", "value": df["station_name"].nunique()},
        {"scope": "concatenated_splits", "metric": "sampling_rate_hz", "value": df["sampling_rate_Hz"].mode().iloc[0]},
        {"scope": "concatenated_splits", "metric": "data_size_mb_mean", "value": df["data_size_mb"].mean()},
        {"scope": "concatenated_splits", "metric": "data_size_mb_min", "value": df["data_size_mb"].min()},
        {"scope": "concatenated_splits", "metric": "data_size_mb_median", "value": df["data_size_mb"].median()},
        {"scope": "concatenated_splits", "metric": "data_size_mb_max", "value": df["data_size_mb"].max()},
        {"scope": "concatenated_splits", "metric": "compute_density_mean", "value": df["compute_density_FLOPs_per_byte"].mean()},
        {"scope": "concatenated_splits", "metric": "compute_density_median", "value": df["compute_density_FLOPs_per_byte"].median()},
        {"scope": "concatenated_splits", "metric": "deadline_ms_mean", "value": df["deadline_ms"].mean()},
        {"scope": "concatenated_splits", "metric": "deadline_ms_min", "value": df["deadline_ms"].min()},
        {"scope": "concatenated_splits", "metric": "deadline_ms_max", "value": df["deadline_ms"].max()},
        {"scope": "concatenated_splits", "metric": "dependency_task_count", "value": int((df["dependency_count"] >= 1).sum())},
        {"scope": "concatenated_splits", "metric": "dependency_task_percent", "value": (df["dependency_count"] >= 1).mean() * 100},
        {"scope": "released_evaluation_trace", "metric": "number_of_tasks", "value": len(trace)},
        {"scope": "released_evaluation_trace", "metric": "number_of_stations", "value": trace["station_name"].nunique()},
        {"scope": "released_evaluation_trace", "metric": "deadline_ms_mean", "value": trace["deadline_ms"].mean()},
        {"scope": "released_evaluation_trace", "metric": "deadline_ms_min", "value": trace["deadline_ms"].min()},
        {"scope": "released_evaluation_trace", "metric": "deadline_ms_max", "value": trace["deadline_ms"].max()},
        {"scope": "released_evaluation_trace", "metric": "high_priority_task_count", "value": int(trace["high_priority"].sum())},
        {"scope": "released_evaluation_trace", "metric": "high_priority_task_percent", "value": trace["high_priority"].mean() * 100},
        {"scope": "released_evaluation_trace", "metric": "dependency_task_count", "value": int((trace["dependency_count"] >= 1).sum())},
        {"scope": "released_evaluation_trace", "metric": "dependency_task_percent", "value": (trace["dependency_count"] >= 1).mean() * 100},
    ]
    pd.DataFrame(summary_rows).to_csv(OUT / "exp14_seismic_workload_summary.csv", index=False)

    type_dist = (
        df["task_type"]
        .value_counts()
        .rename_axis("task_type")
        .reset_index(name="count")
    )
    type_dist["percent"] = type_dist["count"] / len(df) * 100
    type_dist.to_csv(OUT / "exp14_task_type_distribution.csv", index=False)

    priority_dist = (
        df["priority"]
        .value_counts()
        .rename_axis("priority")
        .reset_index(name="count")
    )
    priority_dist["percent"] = priority_dist["count"] / len(df) * 100
    priority_dist.to_csv(OUT / "exp14_priority_distribution.csv", index=False)

    by_type = (
        df.groupby("task_type")
        .agg(
            count=("task_type", "size"),
            data_size_mb_mean=("data_size_mb", "mean"),
            data_size_mb_min=("data_size_mb", "min"),
            data_size_mb_max=("data_size_mb", "max"),
            compute_norm_mean=("compute_norm", "mean"),
            deadline_ms_mean=("deadline_ms", "mean"),
            dependency_rate=("dependency_count", lambda s: (s >= 1).mean() * 100),
        )
        .reset_index()
    )
    by_type.to_csv(OUT / "exp14_workload_by_task_type.csv", index=False)

    print("Wrote seismic workload provenance tables to:")
    print(OUT / "exp14_seismic_workload_summary.csv")
    print(OUT / "exp14_task_type_distribution.csv")
    print(OUT / "exp14_priority_distribution.csv")
    print(OUT / "exp14_workload_by_task_type.csv")


if __name__ == "__main__":
    main()

