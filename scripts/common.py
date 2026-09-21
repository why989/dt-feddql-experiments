from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_result_csv(filename: str) -> pd.DataFrame:
    """Read one CSV file from the project results directory."""
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"结果文件不存在: {path}")
    df = pd.read_csv(path)
    return auto_numeric(df)


def auto_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric when possible without changing text columns."""
    converted = df.copy()
    for column in converted.columns:
        converted[column] = pd.to_numeric(converted[column], errors="ignore")
    return converted


def save_table(df: pd.DataFrame, filename: str) -> Path:
    """Save a table to outputs and return the path."""
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_markdown(lines: Iterable[str], filename: str) -> Path:
    path = OUTPUT_DIR / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def markdown_table(df: pd.DataFrame) -> str:
    """Return a markdown-friendly table without requiring optional dependencies."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def highlight_color(labels: Iterable[str], target: str = "DT-FedDQL") -> list[str]:
    return ["#1f77b4" if str(label) == target else "#b8c2cc" for label in labels]


def safe_percent_change(new_value: float, baseline_value: float, higher_is_better: bool) -> float:
    """Return improvement percentage against baseline.

    For lower-is-better metrics such as latency, improvement is
    (baseline - new) / baseline. For higher-is-better metrics, it is
    (new - baseline) / baseline.
    """
    if baseline_value == 0 or np.isnan(baseline_value):
        return np.nan
    if higher_is_better:
        return (new_value - baseline_value) / baseline_value * 100.0
    return (baseline_value - new_value) / baseline_value * 100.0

