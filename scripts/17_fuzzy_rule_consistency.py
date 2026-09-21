from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "dt_feddql_result.csv"
OUT_DIR = ROOT / "outputs"

ACTION_LABELS = {
    0: "Terminal",
    1: "Edge",
    2: "Cloud",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULT_PATH)

    required = {"reference_action", "fuzzy_pred_action"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for fuzzy consistency analysis: {sorted(missing)}")

    ref = df["reference_action"].astype(int).map(ACTION_LABELS)
    pred = df["fuzzy_pred_action"].astype(int).map(ACTION_LABELS)

    matrix = pd.crosstab(ref, pred, rownames=["Reference heuristic"], colnames=["Fuzzy prior"], dropna=False)
    matrix = matrix.reindex(index=["Terminal", "Edge", "Cloud"], columns=["Terminal", "Edge", "Cloud"], fill_value=0)

    total = int(matrix.to_numpy().sum())
    matched = int(matrix.to_numpy().trace())
    consistency = matched / total if total else 0.0

    high_mask = df.get("high_priority", pd.Series([0] * len(df))).astype(int) == 1
    high_total = int(high_mask.sum())
    high_consistency = (
        float((df.loc[high_mask, "reference_action"].astype(int) == df.loc[high_mask, "fuzzy_pred_action"].astype(int)).mean())
        if high_total
        else 1.0
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "rule_consistency_rate",
                "value": consistency * 100.0,
                "description": "Agreement between the fuzzy-prior output and the deterministic reference-action heuristic; not supervised classification accuracy.",
            },
            {
                "metric": "matched_samples",
                "value": matched,
                "description": "Number of tasks where the fuzzy-prior action agrees with the reference heuristic.",
            },
            {
                "metric": "total_samples",
                "value": total,
                "description": "Number of evaluated task instances.",
            },
            {
                "metric": "high_priority_rule_consistency_rate",
                "value": high_consistency * 100.0,
                "description": "Agreement rate restricted to high-priority tasks.",
            },
            {
                "metric": "high_priority_samples",
                "value": high_total,
                "description": "Number of high-priority task instances.",
            },
        ]
    )

    counts = pd.DataFrame(
        {
            "action": ["Terminal", "Edge", "Cloud"],
            "reference_count": [int((ref == x).sum()) for x in ["Terminal", "Edge", "Cloud"]],
            "fuzzy_prior_count": [int((pred == x).sum()) for x in ["Terminal", "Edge", "Cloud"]],
        }
    )

    matrix.to_csv(OUT_DIR / "exp17_fuzzy_rule_consistency_matrix.csv")
    counts.to_csv(OUT_DIR / "exp17_fuzzy_label_counts.csv", index=False)
    summary.to_csv(OUT_DIR / "exp17_fuzzy_rule_consistency_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(counts.to_string(index=False))
    print(matrix.to_string())


if __name__ == "__main__":
    main()

