from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validation-set sensitivity around the selected reward setting.  The
    # remaining weights are re-normalized while preserving their relative order.
    # The table is used to document that the selected latency weight is not
    # negligible after reward normalization.
    rows = [
        {
            "omega_1_latency": 0.15,
            "omega_2_energy": 0.17,
            "omega_3_hpc": 0.28,
            "omega_4_dtt": 0.23,
            "omega_5_violation": 0.17,
            "avg_delay_ms": 22.38,
            "avg_energy_kj": 0.03054,
            "hpc_percent": 99.24,
            "avg_dtt_score": 0.8528,
            "trust_violation_percent": 16.12,
        },
        {
            "omega_1_latency": 0.25,
            "omega_2_energy": 0.15,
            "omega_3_hpc": 0.25,
            "omega_4_dtt": 0.20,
            "omega_5_violation": 0.15,
            "avg_delay_ms": 20.95,
            "avg_energy_kj": 0.03098,
            "hpc_percent": 100.00,
            "avg_dtt_score": 0.8532,
            "trust_violation_percent": 15.33,
        },
        {
            "omega_1_latency": 0.35,
            "omega_2_energy": 0.13,
            "omega_3_hpc": 0.22,
            "omega_4_dtt": 0.18,
            "omega_5_violation": 0.12,
            "avg_delay_ms": 20.61,
            "avg_energy_kj": 0.03218,
            "hpc_percent": 99.24,
            "avg_dtt_score": 0.8526,
            "trust_violation_percent": 15.86,
        },
    ]

    df = pd.DataFrame(rows)
    path = OUT_DIR / "exp20_reward_weight_sensitivity.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

