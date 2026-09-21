from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "algorithm_metrics_comparison.csv"
OUT_DIR = ROOT / "outputs"


ORDER = [
    "DT-FedDQL",
    "FedDQL-Federated",
    "Greedy",
    "Fuzzy DQL",
    "FedServ",
    "MILP",
    "Centralized DQN",
    "Local Only",
    "Random Offloading",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULT_PATH)
    df["algorithm"] = df["algorithm"].astype(str)
    df["_order"] = df["algorithm"].map({name: idx for idx, name in enumerate(ORDER)})
    df = df.sort_values("_order").drop(columns=["_order"])

    comparison = df[
        [
            "algorithm",
            "avg_delay_ms",
            "avg_energy_kj",
            "high_priority_completion_rate",
            "avg_dtt_score",
            "trust_violation_rate",
            "edge_cpu_bw_utilization",
        ]
    ].copy()
    comparison = comparison.rename(
        columns={
            "algorithm": "Algorithm",
            "avg_delay_ms": "Delay_ms",
            "avg_energy_kj": "Energy_kJ_per_task",
            "high_priority_completion_rate": "HPC_percent",
            "avg_dtt_score": "DTT_score",
            "trust_violation_rate": "Violation_percent",
            "edge_cpu_bw_utilization": "CPU_BW_util_percent",
        }
    )
    comparison["Algorithm"] = comparison["Algorithm"].replace({"MILP": "MILP-inspired"})

    baseline_sources = pd.DataFrame(
        [
            ["Local Only", "Deterministic baseline", "All tasks are executed on the source terminal."],
            ["Random Offloading", "Stochastic baseline", "Randomly selects one feasible execution layer."],
            ["Greedy", "Self-implemented heuristic", "Uses latency urgency, workload size, and resource availability."],
            ["MILP-inspired", "Optimization-inspired baseline", "Uses deterministic assignment under the same objective components; no exact MILP optimality gap is claimed."],
            ["Centralized DQN", "Self-implemented learning baseline", "Single DQN policy with centralized state observation."],
            ["Fuzzy DQL", "Self-implemented fuzzy-guided DQL baseline", "Adds fuzzy-prior guidance but no federated aggregation or DT re-orchestration."],
            ["FedServ", "Literature-inspired federated service baseline", "Federated service-offloading style policy under the same simulator."],
            ["FedDQL-Federated", "Proposed-family ablation", "Federated DQL without DT trustworthy re-orchestration."],
            ["DT-FedDQL", "Proposed method", "Fuzzy prior + FedDQL + DT trustworthy orchestration."],
        ],
        columns=["Algorithm", "Implementation_type", "Description"],
    )

    comparison_path = OUT_DIR / "exp19_complete_baseline_comparison.csv"
    source_path = OUT_DIR / "exp19_baseline_implementation_sources.csv"
    comparison.to_csv(comparison_path, index=False)
    baseline_sources.to_csv(source_path, index=False)

    print(comparison.to_string(index=False))
    print(baseline_sources.to_string(index=False))
    print(f"Saved: {comparison_path}")
    print(f"Saved: {source_path}")


if __name__ == "__main__":
    main()

