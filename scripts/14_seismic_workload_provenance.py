"""Summarize seismic workload provenance for the paper.

This script extracts workload statistics from the train/validation/test CSV
files and writes compact tables used to justify that the experiments are tied
to wireless seismic sensing rather than a generic IoT workload.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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

    summary_rows = [
        {"metric": "number_of_tasks", "value": len(df)},
        {"metric": "number_of_stations", "value": df["station_name"].nunique()},
        {"metric": "sampling_rate_hz", "value": df["sampling_rate_Hz"].mode().iloc[0]},
        {"metric": "data_size_mb_mean", "value": df["data_size_mb"].mean()},
        {"metric": "data_size_mb_min", "value": df["data_size_mb"].min()},
        {"metric": "data_size_mb_median", "value": df["data_size_mb"].median()},
        {"metric": "data_size_mb_max", "value": df["data_size_mb"].max()},
        {"metric": "compute_density_mean", "value": df["compute_density_FLOPs_per_byte"].mean()},
        {"metric": "compute_density_median", "value": df["compute_density_FLOPs_per_byte"].median()},
        {"metric": "deadline_ms_mean", "value": df["deadline_ms"].mean()},
        {"metric": "deadline_ms_min", "value": df["deadline_ms"].min()},
        {"metric": "deadline_ms_max", "value": df["deadline_ms"].max()},
        {"metric": "dependency_task_count", "value": int((df["dependency_count"] >= 1).sum())},
        {"metric": "dependency_task_percent", "value": (df["dependency_count"] >= 1).mean() * 100},
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

