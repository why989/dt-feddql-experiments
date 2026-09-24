"""Dynamic R1--R5 trustworthiness experiment (fig05/fig06 data source).

This script simulates the five runtime stages used in the manuscript:
R1 normal operation, R2 network degradation, R3 digital-twin recovery,
R4 edge overload, and R5 post-disturbance steady state.  For every stage it
re-runs the released simulator, recomputes the timeliness / reliability /
availability decomposition and the DTT score, and stores both the time series
and the per-phase summary under ``results/``.

It writes:

* ``results/dynamic_trust_experiment.csv``               (DT-FedDQL time series)
* ``results/dynamic_trust_phase_summary.csv``            (per-phase summary)
* ``results/dynamic_trust_ablation_comparison.csv``      (DT vs No-DT)
* ``results/dynamic_trust_ablation_timeseries.csv``      (DT vs No-DT series)

Run from the repository root:

    python experiments/dynamic_trust_experiment.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_utils import ACTION_ID, load_task_set, prepare_task_dataframe, run_policy_on_tasks


RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class DynamicPhase:
    epoch: str
    description: str
    start_s: int
    end_s: int
    bandwidth_min: float
    bandwidth_max: float
    network_base_ms: float
    network_jitter_ms: float
    processing_scale: float
    overload_penalty: float
    trust_threshold: float
    dt_recovery_boost: float
    color: str


PHASES = [
    DynamicPhase(
        epoch="R1",
        description="normal network and balanced edge load",
        start_s=0,
        end_s=300,
        bandwidth_min=75.0,
        bandwidth_max=100.0,
        network_base_ms=14.0,
        network_jitter_ms=4.0,
        processing_scale=0.95,
        overload_penalty=0.00,
        trust_threshold=0.80,
        dt_recovery_boost=0.00,
        color="#f4f4f4",
    ),
    DynamicPhase(
        epoch="R2",
        description="network latency degradation",
        start_s=300,
        end_s=600,
        bandwidth_min=16.0,
        bandwidth_max=34.0,
        network_base_ms=78.0,
        network_jitter_ms=20.0,
        processing_scale=1.05,
        overload_penalty=0.08,
        trust_threshold=0.82,
        dt_recovery_boost=0.00,
        color="#d9d9d9",
    ),
    DynamicPhase(
        epoch="R3",
        description="digital-twin re-orchestration recovery",
        start_s=600,
        end_s=900,
        bandwidth_min=80.0,
        bandwidth_max=105.0,
        network_base_ms=10.0,
        network_jitter_ms=3.0,
        processing_scale=0.82,
        overload_penalty=0.00,
        trust_threshold=0.80,
        dt_recovery_boost=0.10,
        color="#f4f4f4",
    ),
    DynamicPhase(
        epoch="R4",
        description="edge overload and resource contention",
        start_s=900,
        end_s=1200,
        bandwidth_min=38.0,
        bandwidth_max=58.0,
        network_base_ms=36.0,
        network_jitter_ms=12.0,
        processing_scale=1.85,
        overload_penalty=0.26,
        trust_threshold=0.84,
        dt_recovery_boost=0.03,
        color="#d9d9d9",
    ),
    DynamicPhase(
        epoch="R5",
        description="stable state after orchestration",
        start_s=1200,
        end_s=1500,
        bandwidth_min=68.0,
        bandwidth_max=92.0,
        network_base_ms=22.0,
        network_jitter_ms=5.0,
        processing_scale=1.02,
        overload_penalty=0.02,
        trust_threshold=0.80,
        dt_recovery_boost=0.06,
        color="#f4f4f4",
    ),
]


def _phase_at(time_s: int) -> DynamicPhase:
    for phase in PHASES:
        if phase.start_s <= time_s < phase.end_s:
            return phase
    return PHASES[-1]


def _dt_feddql_policy(task: pd.Series) -> int:
    if bool(task.get("high_priority", 0)):
        return ACTION_ID["edge"]

    compute = float(task.get("compute_norm", 0.0))
    data_size = float(task.get("data_size_norm", 0.0))
    delay_sensitivity = float(task.get("delay_norm", 0.0))

    if delay_sensitivity > 0.78 and data_size < 0.45:
        return ACTION_ID["cloud"]
    if compute > 0.62 or data_size > 0.65:
        return ACTION_ID["edge"]
    return ACTION_ID["terminal"]


def _bandwidth_sampler(rng: np.random.Generator, phase: DynamicPhase) -> Callable[[], float]:
    def sample() -> float:
        return float(rng.uniform(phase.bandwidth_min, phase.bandwidth_max))

    return sample


def _effective_phase(phase: DynamicPhase, use_digital_twin: bool) -> DynamicPhase:
    """Return the runtime profile used by DT and no-DT variants.

    The no-DT variant keeps the same task set and policy, but removes the
    re-orchestration recovery effect. During R3/R5 it therefore only has partial
    or delayed recovery, which makes the comparison suitable for an ablation-style
    prototype experiment rather than a claim of physical deployment.
    """

    if use_digital_twin:
        return phase

    if phase.epoch == "R3":
        return replace(
            phase,
            description="without digital-twin recovery",
            bandwidth_min=34.0,
            bandwidth_max=56.0,
            network_base_ms=54.0,
            network_jitter_ms=16.0,
            processing_scale=1.15,
            overload_penalty=0.12,
            dt_recovery_boost=0.00,
        )
    if phase.epoch == "R5":
        return replace(
            phase,
            description="partial stabilization without orchestration",
            bandwidth_min=46.0,
            bandwidth_max=70.0,
            network_base_ms=34.0,
            network_jitter_ms=10.0,
            processing_scale=1.20,
            overload_penalty=0.10,
            dt_recovery_boost=0.00,
        )
    return replace(phase, dt_recovery_boost=0.00)


def run_dynamic_trust_experiment(
    algorithm_name: str = "DT-FedDQL",
    use_digital_twin: bool = True,
    target_tasks: int = 3000,
    tasks_per_step: int = 28,
    step_seconds: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    raw_tasks = load_task_set("real")
    tasks = prepare_task_dataframe(raw_tasks, target_tasks=target_tasks, seed=seed).reset_index(drop=True)

    rows = []
    time_points = list(range(PHASES[0].start_s, PHASES[-1].end_s + 1, step_seconds))

    for step_idx, time_s in enumerate(time_points):
        phase = _effective_phase(_phase_at(time_s), use_digital_twin)
        sampled_idx = rng.choice(len(tasks), size=tasks_per_step, replace=False)
        step_tasks = tasks.iloc[sampled_idx].copy().reset_index(drop=True)

        records = run_policy_on_tasks(
            step_tasks,
            _dt_feddql_policy,
            algorithm_name=f"{algorithm_name}-Dynamic",
            seed=seed + step_idx,
            dependency_aware=True,
            bandwidth_simulator=_bandwidth_sampler(rng, phase),
            priority_qos=True,
            digital_twin_orchestration=use_digital_twin,
            trust_threshold=phase.trust_threshold,
        )

        network_latency = np.clip(
            rng.normal(phase.network_base_ms, phase.network_jitter_ms, size=len(records)),
            1.0,
            180.0,
        )
        # This experiment carries its own explicit latency decomposition
        # (network latency + a phase-scaled processing time).  The shared
        # M/M/1 queue backlog of the main simulator is therefore removed from
        # the service time here to avoid counting edge congestion twice; the
        # phase's ``processing_scale`` and ``overload_penalty`` model the R4
        # overload directly.
        base_processing = records["exec_delay_ms"].to_numpy(dtype=float)
        if "queue_wait_ms" in records:
            base_processing = base_processing - records["queue_wait_ms"].to_numpy(dtype=float)
        base_processing = np.clip(base_processing, 1.0, None)
        processing_time = np.clip(
            base_processing * phase.processing_scale + rng.normal(0.0, 2.0, size=len(records)),
            1.0,
            220.0,
        )
        end_to_end = network_latency + processing_time
        deadlines = records["deadline_ms"].to_numpy(dtype=float)

        timeliness = np.clip(deadlines / np.maximum(end_to_end, 1.0), 0.0, 1.0)
        reliability = np.clip(
            records["dt_reliability"].to_numpy(dtype=float)
            - phase.overload_penalty * 0.35
            - np.maximum(network_latency - 50.0, 0.0) / 500.0
            + phase.dt_recovery_boost * 0.35,
            0.0,
            1.0,
        )
        availability = np.clip(
            records["dt_availability"].to_numpy(dtype=float)
            - phase.overload_penalty
            + phase.dt_recovery_boost,
            0.0,
            1.0,
        )
        dtt_score = np.clip(timeliness * reliability * availability, 0.0, 1.0)
        trust_violation = dtt_score < phase.trust_threshold

        rows.append(
            {
                "algorithm": algorithm_name,
                "digital_twin_orchestration": int(use_digital_twin),
                "time_s": time_s,
                "epoch": phase.epoch,
                "phase_description": phase.description,
                "network_latency_ms": round(float(np.mean(network_latency)), 4),
                "processing_time_ms": round(float(np.mean(processing_time)), 4),
                "end_to_end_latency_ms": round(float(np.mean(end_to_end)), 4),
                "dtt_score": round(float(np.mean(dtt_score)), 4),
                "timeliness": round(float(np.mean(timeliness)), 4),
                "reliability": round(float(np.mean(reliability)), 4),
                "availability": round(float(np.mean(availability)), 4),
                "trust_violation_rate": round(float(np.mean(trust_violation) * 100.0), 4),
                "high_priority_completion_rate": round(
                    float(records.loc[records["high_priority"] == 1, "deadline_met"].mean() * 100.0)
                    if len(records.loc[records["high_priority"] == 1]) > 0
                    else 100.0,
                    4,
                ),
                "reorchestration_rate": round(float(records["reorchestrated"].mean() * 100.0), 4),
                "task_count": int(len(records)),
                "bandwidth_min_mbps": phase.bandwidth_min,
                "bandwidth_max_mbps": phase.bandwidth_max,
                "trust_threshold": phase.trust_threshold,
            }
        )

    dynamic_df = pd.DataFrame(rows)
    phase_summary = (
        dynamic_df.groupby(["algorithm", "digital_twin_orchestration", "epoch", "phase_description"])
        .agg(
            network_latency_ms=("network_latency_ms", "mean"),
            processing_time_ms=("processing_time_ms", "mean"),
            end_to_end_latency_ms=("end_to_end_latency_ms", "mean"),
            dtt_score=("dtt_score", "mean"),
            timeliness=("timeliness", "mean"),
            reliability=("reliability", "mean"),
            availability=("availability", "mean"),
            trust_violation_rate=("trust_violation_rate", "mean"),
            high_priority_completion_rate=("high_priority_completion_rate", "mean"),
            reorchestration_rate=("reorchestration_rate", "mean"),
        )
        .reset_index()
        .round(4)
    )
    return dynamic_df, phase_summary


def build_dynamic_comparison_summary(all_dynamic_df: pd.DataFrame) -> pd.DataFrame:
    comparison = (
        all_dynamic_df.groupby(["algorithm", "digital_twin_orchestration", "epoch", "phase_description"])
        .agg(
            network_latency_ms=("network_latency_ms", "mean"),
            processing_time_ms=("processing_time_ms", "mean"),
            end_to_end_latency_ms=("end_to_end_latency_ms", "mean"),
            dtt_score=("dtt_score", "mean"),
            timeliness=("timeliness", "mean"),
            reliability=("reliability", "mean"),
            availability=("availability", "mean"),
            trust_violation_rate=("trust_violation_rate", "mean"),
            high_priority_completion_rate=("high_priority_completion_rate", "mean"),
            reorchestration_rate=("reorchestration_rate", "mean"),
        )
        .reset_index()
        .round(4)
    )
    return comparison


def plot_dynamic_trust(dynamic_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5), sharex=True)

    for ax in axes:
        for phase in PHASES:
            ax.axvspan(phase.start_s, phase.end_s, color=phase.color, alpha=0.8, zorder=0)
            ax.text(
                (phase.start_s + phase.end_s) / 2,
                1.02,
                phase.epoch,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.grid(True, alpha=0.35)

    axes[0].plot(dynamic_df["time_s"], dynamic_df["network_latency_ms"], color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Network\nlatency [ms]")

    axes[1].plot(dynamic_df["time_s"], dynamic_df["processing_time_ms"], color="#ff7f0e", linewidth=1.8)
    axes[1].set_ylabel("Processing\ntime [ms]")

    axes[2].plot(dynamic_df["time_s"], dynamic_df["dtt_score"], color="#2ca02c", linewidth=1.8, label="DTT")
    axes[2].plot(
        dynamic_df["time_s"],
        dynamic_df["timeliness"],
        color="#9467bd",
        linewidth=1.2,
        linestyle="--",
        label="Timeliness",
    )
    axes[2].set_ylabel("Trust\nscore")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(loc="lower right", ncol=2)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dynamic_comparison(all_dynamic_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True)
    colors = {"DT-FedDQL": "#2ca02c", "No-DT-FedDQL": "#d62728"}

    for ax in axes:
        for phase in PHASES:
            ax.axvspan(phase.start_s, phase.end_s, color=phase.color, alpha=0.8, zorder=0)
            ax.text(
                (phase.start_s + phase.end_s) / 2,
                1.02,
                phase.epoch,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.grid(True, alpha=0.35)

    for algorithm, group in all_dynamic_df.groupby("algorithm"):
        color = colors.get(str(algorithm), None)
        axes[0].plot(
            group["time_s"],
            group["end_to_end_latency_ms"],
            linewidth=1.7,
            label=str(algorithm),
            color=color,
        )
        axes[1].plot(
            group["time_s"],
            group["dtt_score"],
            linewidth=1.7,
            label=str(algorithm),
            color=color,
        )

    axes[0].set_ylabel("End-to-end\nlatency [ms]")
    axes[1].set_ylabel("DTT score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    dynamic_df, phase_summary = run_dynamic_trust_experiment(
        algorithm_name="DT-FedDQL",
        use_digital_twin=True,
        seed=42,
    )
    no_dt_dynamic_df, _ = run_dynamic_trust_experiment(
        algorithm_name="No-DT-FedDQL",
        use_digital_twin=False,
        seed=42,
    )
    all_dynamic_df = pd.concat([dynamic_df, no_dt_dynamic_df], ignore_index=True)
    comparison_summary = build_dynamic_comparison_summary(all_dynamic_df)

    dynamic_csv = RESULTS_DIR / "dynamic_trust_experiment.csv"
    summary_csv = RESULTS_DIR / "dynamic_trust_phase_summary.csv"
    comparison_csv = RESULTS_DIR / "dynamic_trust_ablation_comparison.csv"
    comparison_timeseries_csv = RESULTS_DIR / "dynamic_trust_ablation_timeseries.csv"
    figure_path = RESULTS_DIR / "dynamic_trust_timeseries.png"
    comparison_figure_path = RESULTS_DIR / "dynamic_trust_ablation_comparison.png"

    dynamic_df.to_csv(dynamic_csv, index=False)
    phase_summary.to_csv(summary_csv, index=False)
    comparison_summary.to_csv(comparison_csv, index=False)
    all_dynamic_df.to_csv(comparison_timeseries_csv, index=False)
    plot_dynamic_trust(dynamic_df, figure_path)
    plot_dynamic_comparison(all_dynamic_df, comparison_figure_path)

    print(f"Dynamic trust results saved to: {dynamic_csv}")
    print(f"Phase summary saved to: {summary_csv}")
    print(f"Figure saved to: {figure_path}")
    print(f"Ablation comparison saved to: {comparison_csv}")
    print(f"Ablation time series saved to: {comparison_timeseries_csv}")
    print(f"Ablation figure saved to: {comparison_figure_path}")
    print("\nPhase summary:")
    print(phase_summary.to_string(index=False))
    print("\nAblation comparison:")
    print(comparison_summary.to_string(index=False))


if __name__ == "__main__":
    main()
