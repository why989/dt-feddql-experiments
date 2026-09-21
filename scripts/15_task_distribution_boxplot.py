from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def load_metric(filename: str, metric: str) -> pd.Series:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, usecols=[metric])
    return data[metric].dropna()


def color_boxplot(bp, colors) -> None:
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#2b2b2b")
        patch.set_linewidth(0.8)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.0)
    for item in bp["whiskers"] + bp["caps"]:
        item.set_color("#444444")
        item.set_linewidth(0.75)
    for mean in bp["means"]:
        mean.set_marker("D")
        mean.set_markerfacecolor("white")
        mean.set_markeredgecolor("#222222")
        mean.set_markersize(3.2)


def main() -> None:
    setup_style()

    algorithms = [
        ("DT-FedDQL", "dt_feddql_result.csv"),
        ("Greedy", "greedy_results.csv"),
        ("MILP", "milp_results.csv"),
        ("C-DQN", "dqn_results.csv"),
        ("Local", "local_only_results.csv"),
        ("Random", "random_offloading_result.csv"),
    ]
    labels = [name for name, _ in algorithms]
    colors = ["#1f77b4", "#9ecae1", "#bdbdbd", "#c7c7c7", "#d9d9d9", "#f2c14e"]

    latency = [load_metric(filename, "exec_delay_ms") for _, filename in algorithms]
    energy = [load_metric(filename, "exec_energy_kj") for _, filename in algorithms]

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.85), sharex=True)

    bp_delay = axes[0].boxplot(
        latency,
        patch_artist=True,
        widths=0.55,
        whis=(5, 95),
        showfliers=False,
        showmeans=True,
    )
    color_boxplot(bp_delay, colors)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("(a) Task latency distribution")
    axes[0].grid(axis="y", linestyle="-", alpha=0.22)

    bp_energy = axes[1].boxplot(
        energy,
        patch_artist=True,
        widths=0.55,
        whis=(5, 95),
        showfliers=False,
        showmeans=True,
    )
    color_boxplot(bp_energy, colors)
    axes[1].set_ylabel("Energy (kJ/task)")
    axes[1].set_title("(b) Task energy distribution")
    axes[1].grid(axis="y", linestyle="-", alpha=0.22)
    axes[1].set_xticks(range(1, len(labels) + 1))
    axes[1].set_xticklabels(labels, rotation=0)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5, h_pad=0.75)

    for directory in [PROJECT_ROOT / "outputs" / "publication_quality_figures", FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "task_distribution_boxplot.png", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(directory / "task_distribution_boxplot.pdf", bbox_inches="tight", pad_inches=0.02)

    print(FIGURES_DIR / "task_distribution_boxplot.png")


if __name__ == "__main__":
    main()

