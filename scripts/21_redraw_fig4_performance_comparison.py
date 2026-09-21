from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_DIR = Path(
    r"D:\研究生资料\基于联邦深度Q学习的微动勘探数据处理任务卸载系统设计\paiban\paiban"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "publication_quality_figures"


ORDER = [
    "DT-FedDQL",
    "Greedy",
    "Fuzzy DQL",
    "FedServ",
    "MILP",
    "Centralized DQN",
    "Local Only",
    "Random Offloading",
]

LABELS = {
    "DT-FedDQL": "DT-FDQL",
    "Greedy": "Greedy",
    "Fuzzy DQL": "Fuzzy",
    "FedServ": "FServ",
    "MILP": "MILP",
    "Centralized DQN": "C-DQN",
    "Local Only": "Local",
    "Random Offloading": "Rand.",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.7,
        }
    )


def read_data() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "algorithm_metrics_comparison.csv")
    df = df[df["algorithm"].isin(ORDER)].copy()
    df["algorithm"] = pd.Categorical(df["algorithm"], categories=ORDER, ordered=True)
    df = df.sort_values("algorithm").reset_index(drop=True)
    df["label"] = df["algorithm"].map(LABELS)
    return df


def add_value_labels(ax, bars, fmt, dy=1.0, fontsize=6.2, color="#333333") -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )


def soften_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)


def main() -> None:
    setup_style()
    df = read_data()
    labels = df["label"].tolist()
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45), constrained_layout=True)

    # (a) Latency and energy. Energy is scaled by 1000 to share the y-axis.
    ax = axes[0]
    width = 0.34
    latency = df["avg_delay_ms"].to_numpy(dtype=float)
    energy_scaled = df["avg_energy_kj"].to_numpy(dtype=float) * 1000.0
    b1 = ax.bar(
        x - width / 2,
        latency,
        width,
        label="Latency (ms)",
        color="#2f6f9f",
        edgecolor="#1f4e70",
        linewidth=0.35,
    )
    b2 = ax.bar(
        x + width / 2,
        energy_scaled,
        width,
        label="Energy (kJ) $\\times 10^3$",
        color="#d95f02",
        edgecolor="#9a4200",
        linewidth=0.35,
        alpha=0.88,
    )
    ax.set_title("(a) Latency and energy")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, max(latency.max(), energy_scaled.max()) * 1.18)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, borderpad=0.35)
    soften_axis(ax)

    # (b) QoS and trust metrics.
    ax = axes[1]
    width = 0.24
    hpcr = df["high_priority_completion_rate"].to_numpy(dtype=float)
    dtt = df["avg_dtt_score"].to_numpy(dtype=float) * 100.0
    violation = df["trust_violation_rate"].to_numpy(dtype=float)
    bars_hpcr = ax.bar(
        x - width,
        hpcr,
        width,
        label="HPC (%)",
        color="#1b9e77",
        edgecolor="#117554",
        linewidth=0.3,
    )
    bars_dtt = ax.bar(
        x,
        dtt,
        width,
        label="DTT $\\times 100$",
        color="#e6ab02",
        edgecolor="#a57900",
        linewidth=0.3,
        alpha=0.9,
    )
    bars_viol = ax.bar(
        x + width,
        violation,
        width,
        label="Violation (%)",
        color="#7570b3",
        edgecolor="#514c84",
        linewidth=0.3,
        alpha=0.9,
    )
    ax.set_title("(b) QoS and trust metrics")
    ax.set_ylabel("Percentage / normalized score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, ncol=1, borderpad=0.35)
    soften_axis(ax)

    for out_dir in (PAPER_DIR, OUTPUT_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "delay_constraint_energy.png", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(out_dir / "delay_constraint_energy.pdf", bbox_inches="tight", pad_inches=0.03)

    print(PAPER_DIR / "delay_constraint_energy.png")


if __name__ == "__main__":
    main()

