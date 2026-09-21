from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


def shannon_rate_mbps(distance_m: float, snr_loss_db: float = 0.0, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    bandwidth_hz = 20e6
    p_tx_w = 10.0
    pl0_db = 38.0
    r0 = 1.0
    alpha = 2.7
    shadow_std_db = 4.0
    n0_dbm_hz = -174.0
    noise_w = 10 ** ((n0_dbm_hz - 30) / 10) * bandwidth_hz
    shadow_db = rng.normal(0.0, shadow_std_db)
    fading_gain = rng.exponential(1.0)
    path_loss_db = pl0_db + 10 * alpha * np.log10(max(distance_m, r0) / r0) + shadow_db + snr_loss_db
    snr = p_tx_w * fading_gain * 10 ** (-path_loss_db / 10) / noise_w
    return float(bandwidth_hz * np.log2(1.0 + snr) / 1e6)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dist in [20, 50, 100, 200, 300]:
        normal = shannon_rate_mbps(dist, snr_loss_db=0.0, seed=int(dist))
        degraded = shannon_rate_mbps(dist, snr_loss_db=8.0, seed=int(dist)) / 1.7
        rows.append(
            {
                "distance_m": dist,
                "normal_rate_mbps": round(max(10.0, min(100.0, normal)), 3),
                "degraded_effective_rate_mbps": round(max(3.0, min(60.0, degraded)), 3),
                "snr_degradation_db": 8.0,
                "retransmission_factor": 1.7,
            }
        )
    df = pd.DataFrame(rows)
    path = OUT_DIR / "exp21_wireless_channel_model_summary.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

