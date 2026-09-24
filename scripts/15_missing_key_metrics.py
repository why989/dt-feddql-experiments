"""Generate missing key metrics requested by the review checklist.

The script summarizes action distribution, completion/violation rates,
DT gating diagnostics, DTT components, load-balance fairness, and federated
communication/training metadata from existing experiment outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def jain_index(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    denom = len(values) * np.sum(values**2)
    return float((np.sum(values) ** 2) / denom) if denom > 0 else float("nan")


def main() -> None:
    task = pd.read_csv(RESULTS / "dt_feddql_result.csv")
    gamma = float(task["trust_threshold"].iloc[0]) if "trust_threshold" in task else 0.8

    # Action distribution and completion metrics.  The released action space is
    # node-level, A = {L, e1, e2, e3, C}, so the histogram is taken over the
    # three edge nodes individually and merged only for the edge-tier row.
    action_pct = task["action"].astype(str).str.lower().value_counts(normalize=True).mul(100)
    # Merge local/terminal naming for paper reporting.
    terminal_pct = float(action_pct.get("local", 0.0) + action_pct.get("terminal", 0.0))
    edge_1_pct = float(action_pct.get("edge_1", 0.0))
    edge_2_pct = float(action_pct.get("edge_2", 0.0))
    edge_3_pct = float(action_pct.get("edge_3", 0.0))
    edge_pct = edge_1_pct + edge_2_pct + edge_3_pct
    cloud_pct = float(action_pct.get("cloud", 0.0))

    core_metrics = pd.DataFrame(
        [
            {"metric": "terminal_action_percent", "value": terminal_pct},
            {"metric": "edge_action_percent", "value": edge_pct},
            {"metric": "edge_1_action_percent", "value": edge_1_pct},
            {"metric": "edge_2_action_percent", "value": edge_2_pct},
            {"metric": "edge_3_action_percent", "value": edge_3_pct},
            {"metric": "cloud_action_percent", "value": cloud_pct},
            {"metric": "overall_deadline_completion_percent", "value": float(task["deadline_met"].mean() * 100)},
            {"metric": "deadline_violation_percent", "value": float((1 - task["deadline_met"].mean()) * 100)},
            {"metric": "high_priority_completion_percent", "value": float(task["high_priority_completion_rate"].iloc[0])},
            {"metric": "avg_timeliness", "value": float(task["dt_timeliness"].mean())},
            {"metric": "avg_reliability", "value": float(task["dt_reliability"].mean())},
            {"metric": "avg_availability", "value": float(task["dt_availability"].mean())},
            {"metric": "avg_dtt_score", "value": float(task["dtt_score"].mean())},
        ]
    )
    core_metrics.to_csv(OUT / "exp15_action_completion_dtt_metrics.csv", index=False)

    # DT gating diagnostics. Strict false triggering needs counterfactual execution;
    # here we report prediction-validation proxies.
    selected_pre = task["dt_selected_predicted_dtt"].astype(float)
    measured = task["dtt_score"].astype(float)
    gating = pd.DataFrame(
        [
            {"metric": "static_reorchestration_trigger_percent", "value": float(task["reorchestrated"].mean() * 100)},
            {"metric": "static_reorchestration_count", "value": int(task["reorchestrated"].sum())},
            {"metric": "avg_dt_recovery_time_ms", "value": float(task["dt_recovery_time_ms"].mean())},
            {
                "metric": "false_alarm_proxy_percent_pre_below_gamma_measured_safe",
                "value": float(((selected_pre < gamma) & (measured >= gamma)).mean() * 100),
            },
            {
                "metric": "missed_risk_proxy_percent_pre_safe_measured_below_gamma",
                "value": float(((selected_pre >= gamma) & (measured < gamma)).mean() * 100),
            },
        ]
    )
    gating.to_csv(OUT / "exp15_dt_gating_diagnostics.csv", index=False)

    # Dynamic DT re-orchestration and DTT decomposition by phase.
    phase = pd.read_csv(RESULTS / "dynamic_trust_phase_summary.csv")
    phase[
        [
            "epoch",
            "timeliness",
            "reliability",
            "availability",
            "dtt_score",
            "trust_violation_rate",
            "reorchestration_rate",
        ]
    ].to_csv(OUT / "exp15_dynamic_dtt_decomposition.csv", index=False)

    # Load balance / Jain fairness from node-status snapshot.
    node = pd.read_csv(RESULTS / "node_status_snapshot.csv")
    rows = []
    for tier, sub in node.groupby("tier"):
        vals = sub.groupby("node")["cpu_utilization"].mean().to_numpy(dtype=float)
        rows.append(
            {
                "scope": tier,
                "node_count": len(vals),
                "jain_cpu_fairness": jain_index(vals),
                "cpu_utilization_variance": float(np.var(vals)),
            }
        )
    vals = node.groupby("node")["cpu_utilization"].mean().to_numpy(dtype=float)
    rows.append(
        {
            "scope": "All monitored nodes",
            "node_count": len(vals),
            "jain_cpu_fairness": jain_index(vals),
            "cpu_utilization_variance": float(np.var(vals)),
        }
    )
    pd.DataFrame(rows).to_csv(OUT / "exp15_load_balance_fairness.csv", index=False)

    # Federated training metadata and communication overhead.
    comm = pd.read_csv(OUT / "exp12_federated_communication_overhead.csv")
    training = pd.read_csv(RESULTS / "training_logs.csv")
    fed = pd.DataFrame(
        [
            {"metric": "federated_rounds", "value": 40},
            {"metric": "local_epochs_per_round", "value": 2},
            {"metric": "convergence_epoch_recorded", "value": int(training["epoch"].max())},
            {
                "metric": "parameter_upload_download_mb",
                "value": float(comm.loc[comm["scheme"].eq("Federated upload/download"), "traffic_mb"].iloc[0]),
            },
            {
                "metric": "raw_training_backhaul_mb",
                "value": float(comm.loc[comm["scheme"].eq("Centralized training"), "traffic_mb"].iloc[0]),
            },
            {"metric": "training_wall_clock_s", "value": "not_recorded_in_previous_run"},
        ]
    )
    fed.to_csv(OUT / "exp15_federated_training_overhead.csv", index=False)

    print("Wrote P1-8 metric tables:")
    for name in [
        "exp15_action_completion_dtt_metrics.csv",
        "exp15_dt_gating_diagnostics.csv",
        "exp15_dynamic_dtt_decomposition.csv",
        "exp15_load_balance_fairness.csv",
        "exp15_federated_training_overhead.csv",
    ]:
        print(OUT / name)


if __name__ == "__main__":
    main()

