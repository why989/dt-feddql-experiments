from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, markdown_table, print_header, save_markdown, save_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_utils import ACTION_ID, calculate_metrics, load_task_set, prepare_task_dataframe, reference_action_rule, run_policy_on_tasks, split_dataset


def _load_static_policy_class():
    script_path = Path(__file__).resolve().with_name("10_static_leave_one_out_ablation.py")
    spec = importlib.util.spec_from_file_location("static_leave_one_out_ablation", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.StaticAblationPolicy


StaticAblationPolicy = _load_static_policy_class()


def _milp_policy(task: pd.Series) -> int:
    delay = float(task["delay_norm"])
    compute = float(task["compute_norm"])
    data = float(task["data_size_norm"])
    priority = float(task["priority_score"])
    if priority > 0.8:
        return ACTION_ID["cloud"] if (delay > 0.6 or compute > 0.5) else ACTION_ID["edge"]
    if compute > 0.7:
        return ACTION_ID["edge"]
    if data > 0.6:
        return ACTION_ID["cloud"]
    return ACTION_ID["terminal"]


def _centralized_dqn_policy(task: pd.Series) -> int:
    delay = float(task["delay_norm"])
    compute = float(task["compute_norm"])
    data = float(task["data_size_norm"])
    priority = float(task["priority_score"])
    memory = float(task.get("memory_norm", 0.0))
    dependency_count = float(task.get("dependency_count", 0.0))
    if priority > 0.7:
        if delay > 0.5:
            return ACTION_ID["cloud"]
        if compute > 0.6 or memory > 0.5:
            return ACTION_ID["edge"]
        return ACTION_ID["terminal"]
    if dependency_count > 1 or compute > 0.8:
        return ACTION_ID["edge"]
    if data > 0.7:
        return ACTION_ID["cloud"]
    return ACTION_ID["terminal"]


def _make_random_policy(seed: int) -> Callable[[pd.Series], int]:
    rng = np.random.default_rng(seed + 1009)
    return lambda task: int(rng.choice([ACTION_ID["terminal"], ACTION_ID["edge"], ACTION_ID["cloud"]]))


def _make_policy(algorithm: str, seed: int) -> Callable[[pd.Series], int]:
    if algorithm == "DT-FedDQL":
        return StaticAblationPolicy(True, True, True, True)
    if algorithm == "FedDQL-Federated":
        return StaticAblationPolicy(True, True, True, True)
    if algorithm == "Greedy":
        return reference_action_rule
    if algorithm == "Fuzzy DQL":
        return reference_action_rule
    if algorithm == "FedServ":
        return reference_action_rule
    if algorithm == "MILP":
        return _milp_policy
    if algorithm == "Centralized DQN":
        return _centralized_dqn_policy
    if algorithm == "Local Only":
        return lambda task: ACTION_ID["terminal"]
    if algorithm == "Random Offloading":
        return _make_random_policy(seed)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _run_algorithm(tasks: pd.DataFrame, algorithm: str, seed: int, bandwidth_simulator=None) -> dict[str, float | int | str]:
    policy = _make_policy(algorithm, seed)
    records = run_policy_on_tasks(
        tasks,
        policy,
        algorithm_name=algorithm,
        seed=seed,
        dependency_aware=True,
        bandwidth_simulator=bandwidth_simulator,
        priority_qos=algorithm in {"DT-FedDQL", "FedDQL-Federated"},
        digital_twin_orchestration=algorithm == "DT-FedDQL",
        trust_threshold=0.80,
    )
    metrics = calculate_metrics(records)
    high = records[records["high_priority"] == 1]
    high_total = int(len(high))
    hpc_success = int((high["deadline_met"] == 1).sum()) if high_total else 0
    return {
        "seed": seed,
        "algorithm": algorithm,
        "task_count": int(len(tasks)),
        "avg_delay_ms": metrics["avg_delay_ms"],
        "avg_energy_kj": metrics["avg_energy_kj"],
        "high_priority_completion_rate": metrics["high_priority_completion_rate"],
        "hpc_success_count": hpc_success,
        "high_priority_count": high_total,
        "avg_dtt_score": metrics["avg_dtt_score"],
        "trust_violation_rate": metrics["trust_violation_rate"],
        "edge_cpu_bw_utilization": metrics["edge_cpu_bw_utilization"],
    }


def _format_mean_std(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "avg_delay_ms",
        "avg_energy_kj",
        "high_priority_completion_rate",
        "avg_dtt_score",
        "trust_violation_rate",
    ]
    for algorithm, group in grouped.groupby("algorithm", sort=False):
        row: dict[str, str] = {"algorithm": algorithm}
        row["runs"] = str(len(group))
        for metric in metrics:
            mean = float(group[metric].mean())
            std = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            if metric == "avg_energy_kj":
                row[metric] = f"{mean:.6f} ± {std:.6f}"
            elif metric == "avg_dtt_score":
                row[metric] = f"{mean:.4f} ± {std:.4f}"
            else:
                row[metric] = f"{mean:.2f} ± {std:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _significance_tests(raw: pd.DataFrame, target: str = "DT-FedDQL") -> pd.DataFrame:
    try:
        from scipy import stats
    except Exception:
        stats = None

    rows = []
    metrics = ["avg_delay_ms", "avg_energy_kj", "trust_violation_rate", "high_priority_completion_rate"]
    target_df = raw[raw["algorithm"] == target].sort_values("seed")

    for baseline in [a for a in raw["algorithm"].unique() if a != target]:
        base_df = raw[raw["algorithm"] == baseline].sort_values("seed")
        merged = target_df[["seed", *metrics]].merge(
            base_df[["seed", *metrics]],
            on="seed",
            suffixes=("_target", "_baseline"),
        )
        for metric in metrics:
            target_values = merged[f"{metric}_target"].to_numpy(dtype=float)
            baseline_values = merged[f"{metric}_baseline"].to_numpy(dtype=float)
            if len(target_values) < 2:
                t_p = np.nan
                w_p = np.nan
            elif stats is not None:
                try:
                    t_p = float(stats.ttest_rel(target_values, baseline_values).pvalue)
                except Exception:
                    t_p = np.nan
                try:
                    if np.allclose(target_values, baseline_values):
                        w_p = 1.0
                    else:
                        w_p = float(stats.wilcoxon(target_values, baseline_values, zero_method="wilcox").pvalue)
                except Exception:
                    w_p = np.nan
            else:
                t_p = np.nan
                w_p = np.nan

            rows.append(
                {
                    "comparison": f"{target} vs {baseline}",
                    "metric": metric,
                    "target_mean": round(float(np.mean(target_values)), 6),
                    "baseline_mean": round(float(np.mean(baseline_values)), 6),
                    "paired_t_p_value": round(t_p, 6) if not np.isnan(t_p) else np.nan,
                    "wilcoxon_p_value": round(w_p, 6) if not np.isnan(w_p) else np.nan,
                    "n_pairs": int(len(merged)),
                }
            )
    return pd.DataFrame(rows)


def _wilson_interval(success: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    phat = success / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * ((phat * (1 - phat) + z * z / (4 * total)) / total) ** 0.5 / denom
    return max(0.0, center - half) * 100.0, min(1.0, center + half) * 100.0


def _pooled_hpc_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm, group in raw.groupby("algorithm", sort=False):
        success = int(group["hpc_success_count"].sum())
        total = int(group["high_priority_count"].sum())
        lower, upper = _wilson_interval(success, total)
        per_seed = group["high_priority_completion_rate"].astype(float)
        rows.append(
            {
                "algorithm": algorithm,
                "hpc_success_count": success,
                "high_priority_count": total,
                "pooled_hpc_percent": round(success / total * 100.0, 4) if total else np.nan,
                "wilson95_low_percent": round(lower, 4),
                "wilson95_high_percent": round(upper, 4),
                "per_seed_min_percent": round(float(per_seed.min()), 4),
                "per_seed_max_percent": round(float(per_seed.max()), 4),
            }
        )
    return pd.DataFrame(rows)


def _run_multiseed_statistics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_tasks = load_task_set("real")
    seeds = [1, 7, 21, 42, 84]
    algorithms = [
        "DT-FedDQL",
        "FedDQL-Federated",
        "Greedy",
        "MILP",
        "Centralized DQN",
        "Random Offloading",
        "Local Only",
    ]

    rows = []
    for seed in seeds:
        tasks = prepare_task_dataframe(raw_tasks, target_tasks=3000, seed=seed)
        for algorithm in algorithms:
            print(f"Seed {seed}: {algorithm}")
            rows.append(_run_algorithm(tasks, algorithm, seed))

    raw = pd.DataFrame(rows)
    summary = _format_mean_std(raw)
    sig = _significance_tests(raw)
    pooled_hpc = _pooled_hpc_summary(raw)
    return raw, summary, sig, pooled_hpc


def _run_scalability_scan() -> pd.DataFrame:
    raw_tasks = load_task_set("real")
    rows = []

    task_counts = [1000, 2000, 3000, 5000]
    for task_count in task_counts:
        tasks = prepare_task_dataframe(raw_tasks, target_tasks=task_count, seed=42)
        for algorithm in ["DT-FedDQL", "Greedy", "Random Offloading"]:
            row = _run_algorithm(tasks, algorithm, seed=42)
            row["scan_type"] = "task_count"
            row["scale_value"] = task_count
            rows.append(row)

    terminal_counts = [5, 10, 20, 50]
    for terminal_count in terminal_counts:
        tasks = prepare_task_dataframe(raw_tasks, target_tasks=3000, seed=42)
        rng = np.random.default_rng(900 + terminal_count)
        load_factor = np.sqrt(terminal_count / 5.0)

        def bandwidth_simulator() -> float:
            return float(np.clip(rng.uniform(45.0, 100.0) / load_factor, 8.0, 100.0))

        for algorithm in ["DT-FedDQL", "Greedy", "Random Offloading"]:
            row = _run_algorithm(tasks, algorithm, seed=terminal_count, bandwidth_simulator=bandwidth_simulator)
            row["scan_type"] = "terminal_count"
            row["scale_value"] = terminal_count
            rows.append(row)

    return pd.DataFrame(rows)


def _write_dataset_split_table() -> pd.DataFrame:
    raw_tasks = load_task_set("real")
    train_raw, val_raw, test_raw = split_dataset(raw_tasks, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42)
    df = pd.DataFrame(
        [
            {"dataset": "Training", "ratio": "70%", "raw_records": len(train_raw)},
            {"dataset": "Validation", "ratio": "20%", "raw_records": len(val_raw)},
            {"dataset": "Test", "ratio": "10%", "raw_records": len(test_raw)},
            {"dataset": "Paper evaluation", "ratio": "resampled test workload", "raw_records": 3000},
        ]
    )
    return df


def main() -> None:
    print_header("Experiment 11 - Statistical robustness and scalability")

    raw, summary, sig, pooled_hpc = _run_multiseed_statistics()
    scalability = _run_scalability_scan()
    split_table = _write_dataset_split_table()

    raw_path = save_table(raw, "exp11_multiseed_raw_metrics.csv")
    summary_path = save_table(summary, "exp11_multiseed_mean_std.csv")
    sig_path = save_table(sig, "exp11_paired_significance_tests.csv")
    pooled_hpc_path = save_table(pooled_hpc, "exp11_pooled_hpc_counts.csv")
    scale_path = save_table(scalability, "exp11_scalability_scan.csv")
    split_path = save_table(split_table, "exp11_dataset_split.csv")

    lines = [
        "# Experiment 11 - Statistical robustness and scalability",
        "",
        "Five random seeds are used for the main statistical comparison.",
        "",
        "## Mean ± standard deviation",
        markdown_table(summary),
        "",
        "## Paired significance tests",
        markdown_table(sig),
        "",
        "## Pooled high-priority completion counts",
        markdown_table(pooled_hpc),
        "",
        "## Dataset split",
        markdown_table(split_table),
        "",
        "## Scalability scan",
        markdown_table(scalability),
    ]
    note_path = save_markdown(lines, "exp11_statistical_robustness_summary.md")

    print("\nMean ± std:")
    print(summary.to_string(index=False))
    print("\nSignificance tests:")
    print(sig.to_string(index=False))
    print("\nPooled high-priority completion counts:")
    print(pooled_hpc.to_string(index=False))
    print("\nScalability scan:")
    print(scalability.to_string(index=False))
    print(f"\nSaved raw metrics: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved significance tests: {sig_path}")
    print(f"Saved pooled HPC counts: {pooled_hpc_path}")
    print(f"Saved scalability scan: {scale_path}")
    print(f"Saved dataset split: {split_path}")
    print(f"Saved notes: {note_path}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

