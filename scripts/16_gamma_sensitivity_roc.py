"""Gamma-threshold sensitivity and DTT discrimination diagnostics.

This script clarifies that the reported trust-violation rate is a
constraint-level metric: a task violates trust constraints if it either misses
its deadline or has DTT below the selected threshold. It also evaluates how
the violation rate changes with Gamma and how well DTT discriminates
deadline satisfaction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

FILES = {
    "DT-FedDQL": "dt_feddql_result.csv",
    "FedDQL-Federated": "fed_dql_federated_result.csv",
    "Greedy": "greedy_results.csv",
    "MILP": "milp_results.csv",
    "Centralized DQN": "centralized_dqn_results.csv",
    "Local Only": "local_only_results.csv",
    "Random Offloading": "random_offloading_result.csv",
}


def roc_pr_auc(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-score)
    y = y_true[order].astype(int)
    s = score[order]
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan"), float("nan")

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / pos
    fpr = fp / neg
    # prepend origin for ROC.
    roc_auc = float(np.trapz(np.r_[0, tpr], np.r_[0, fpr]))

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    # Average precision with step-wise interpolation.
    recall_prev = np.r_[0, recall[:-1]]
    pr_auc = float(np.sum((recall - recall_prev) * precision))
    return roc_auc, pr_auc


def roc_pr_points(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-score)
    y = y_true[order].astype(int)
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return pd.DataFrame()
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    return pd.DataFrame(
        {
            "fpr": fp / neg,
            "tpr": tp / pos,
            "recall": recall,
            "precision": precision,
            "threshold_score": score[order],
        }
    )


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.35,
            "lines.linewidth": 1.2,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def main() -> None:
    gammas = [0.3, 0.5, 0.7, 0.8, 0.9]
    rows = []
    roc_rows = []
    curve_rows = []
    cause_rows = []

    for algo, filename in FILES.items():
        df = pd.read_csv(RESULTS / filename)
        dtt = df["dtt_score"].astype(float)
        deadline_met = df["deadline_met"].astype(int)

        for gamma in gammas:
            dtt_only = (dtt < gamma)
            composite = (deadline_met == 0) | dtt_only
            rows.append(
                {
                    "algorithm": algo,
                    "gamma": gamma,
                    "dtt_below_gamma_rate_percent": dtt_only.mean() * 100,
                    "constraint_violation_rate_percent": composite.mean() * 100,
                    "deadline_violation_rate_percent": (deadline_met == 0).mean() * 100,
                }
            )

        # DTT as a score for predicting safe deadline completion.
        roc_auc, pr_auc = roc_pr_auc(deadline_met.to_numpy(), dtt.to_numpy())
        roc_rows.append(
            {
                "algorithm": algo,
                "positive_label": "deadline_met",
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )
        curve = roc_pr_points(deadline_met.to_numpy(), dtt.to_numpy())
        if not curve.empty:
            curve.insert(0, "algorithm", algo)
            curve_rows.append(curve)

        if algo == "DT-FedDQL":
            gamma = 0.8
            dtt_low = dtt < gamma
            deadline_low = deadline_met == 0
            predicted = df["dt_selected_predicted_dtt"].astype(float)
            cause_rows.extend(
                [
                    {"cause": "deadline_miss_total", "count": int(deadline_low.sum()), "percent": deadline_low.mean() * 100},
                    {"cause": "dtt_below_gamma_total", "count": int(dtt_low.sum()), "percent": dtt_low.mean() * 100},
                    {
                        "cause": "deadline_miss_and_dtt_below_gamma",
                        "count": int((deadline_low & dtt_low).sum()),
                        "percent": (deadline_low & dtt_low).mean() * 100,
                    },
                    {
                        "cause": "deadline_miss_but_dtt_not_below_gamma",
                        "count": int((deadline_low & ~dtt_low).sum()),
                        "percent": (deadline_low & ~dtt_low).mean() * 100,
                    },
                    {
                        "cause": "pre_dtt_safe_but_post_constraint_violation",
                        "count": int(((predicted >= gamma) & deadline_low).sum()),
                        "percent": ((predicted >= gamma) & deadline_low).mean() * 100,
                    },
                    {
                        "cause": "high_priority_deadline_miss",
                        "count": int((deadline_low & df["high_priority"].astype(bool)).sum()),
                        "percent": (deadline_low & df["high_priority"].astype(bool)).mean() * 100,
                    },
                ]
            )

    pd.DataFrame(rows).to_csv(OUT / "exp16_gamma_sensitivity.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(OUT / "exp16_dtt_roc_pr.csv", index=False)
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(OUT / "exp16_dtt_roc_pr_curve_points.csv", index=False)
    pd.DataFrame(cause_rows).to_csv(OUT / "exp16_dt_violation_cause.csv", index=False)

    set_ieee_style()
    gamma_df = pd.DataFrame(rows)
    gamma_focus = gamma_df[gamma_df["algorithm"].isin(["DT-FedDQL", "Greedy", "MILP", "Random Offloading"])]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for algo, group in gamma_focus.groupby("algorithm"):
        ax.plot(group["gamma"], group["constraint_violation_rate_percent"], marker="o", markersize=3, label=algo)
    ax.plot(
        gamma_df[gamma_df["algorithm"].eq("DT-FedDQL")]["gamma"],
        gamma_df[gamma_df["algorithm"].eq("DT-FedDQL")]["dtt_below_gamma_rate_percent"],
        marker="s",
        markersize=3,
        linestyle="--",
        label="DT-FedDQL DTT below $\\Gamma$",
    )
    ax.set_xlabel("Trust threshold $\\Gamma$")
    ax.set_ylabel("Violation rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.4)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "exp16_gamma_sensitivity_curve.pdf", bbox_inches="tight")
    fig.savefig(OUT / "exp16_gamma_sensitivity_curve.png", bbox_inches="tight")
    plt.close(fig)

    if curve_rows:
        curve_all = pd.concat(curve_rows, ignore_index=True)
        plot_algorithms = ["DT-FedDQL", "Greedy", "MILP", "Random Offloading"]
        fig, ax = plt.subplots(figsize=(3.45, 2.35))
        for algo in plot_algorithms:
            group = curve_all[curve_all["algorithm"].eq(algo)]
            ax.plot(group["fpr"], group["tpr"], label=algo)
        ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.4)
        ax.legend(frameon=True, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / "exp16_dtt_roc_curve.pdf", bbox_inches="tight")
        fig.savefig(OUT / "exp16_dtt_roc_curve.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3.45, 2.35))
        for algo in plot_algorithms:
            group = curve_all[curve_all["algorithm"].eq(algo)]
            ax.plot(group["recall"], group["precision"], label=algo)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.4)
        ax.legend(frameon=True, loc="lower left")
        fig.tight_layout()
        fig.savefig(OUT / "exp16_dtt_pr_curve.pdf", bbox_inches="tight")
        fig.savefig(OUT / "exp16_dtt_pr_curve.png", bbox_inches="tight")
        plt.close(fig)

    print("Gamma sensitivity:")
    print(pd.DataFrame(rows).query("algorithm == 'DT-FedDQL'").round(4).to_string(index=False))
    print("ROC/PR:")
    print(pd.DataFrame(roc_rows).round(4).to_string(index=False))
    print("Causes:")
    print(pd.DataFrame(cause_rows).round(4).to_string(index=False))
    print("Wrote figures:")
    print(OUT / "exp16_gamma_sensitivity_curve.pdf")
    print(OUT / "exp16_dtt_roc_curve.pdf")
    print(OUT / "exp16_dtt_pr_curve.pdf")


if __name__ == "__main__":
    main()

