"""Build the terminal-edge-cloud node-status snapshot (fig07 data source).

The manuscript's prototype-monitoring figure (fig07) visualises the runtime
state of the terminal, edge, and cloud nodes.  In the development prototype
these numbers were exposed by a small Flask/Prometheus service; the released
repository instead regenerates the same snapshot directly from the dynamic
trust experiment, so the figure is reproducible without running the monitoring
stack.

The snapshot is a *simulation-side* status table derived from
``results/dynamic_trust_phase_summary.csv`` (or the ablation comparison if it
is present).  It is intended for prototype-monitoring visualisation, not for
claiming real hardware telemetry.

Run from the repository root, after
``experiments/dynamic_trust_experiment.py``:

    python scripts/07_terminal_edge_cloud_node_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
PHASE_LABEL_MAP = {"E1": "R1", "E2": "R2", "E3": "R3", "E4": "R4", "E5": "R5"}


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _node_status_label(code: int) -> str:
    if code >= 2:
        return "healthy"
    if code == 1:
        return "degraded"
    return "overloaded"


def _normalize_epoch_label(epoch: Any) -> str:
    label = str(epoch).upper()
    return PHASE_LABEL_MAP.get(label, label)


def _read_dynamic_summary_table() -> pd.DataFrame:
    csv_path = RESULTS_DIR / "dynamic_trust_ablation_comparison.csv"
    if not csv_path.exists():
        csv_path = RESULTS_DIR / "dynamic_trust_phase_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dynamic trust result file: {csv_path}")
    df = pd.read_csv(csv_path)
    if "epoch" in df.columns:
        df = df.copy()
        df["epoch"] = df["epoch"].map(_normalize_epoch_label)
    return df


def build_node_status(algorithm: str = "DT-FedDQL", epoch: str = "R5") -> pd.DataFrame:
    """Build a simulation-side status snapshot for terminal/edge/cloud nodes."""

    dynamic_df = _read_dynamic_summary_table()
    normalized_epoch = _normalize_epoch_label(epoch)
    filtered = dynamic_df[
        (dynamic_df["algorithm"].astype(str).str.lower() == algorithm.lower())
        & (dynamic_df["epoch"].map(_normalize_epoch_label) == normalized_epoch)
    ]
    if filtered.empty:
        raise ValueError(
            f"Node status source not found for algorithm={algorithm}, epoch={normalized_epoch}"
        )

    base = filtered.iloc[0]
    processing_ms = float(base.get("processing_time_ms", 25.0))
    network_ms = float(base.get("network_latency_ms", 20.0))
    base_timeliness = float(base.get("timeliness", 0.75))
    base_dtt = float(base.get("dtt_score", 0.70))
    base_availability = float(base.get("availability", 0.95))

    node_profiles = [
        {"tier": "Terminal", "node": "terminal-1", "load": 0.58, "compute": 1.18, "network": 0.22, "trust": 0.96},
        {"tier": "Terminal", "node": "terminal-2", "load": 0.66, "compute": 1.34, "network": 0.30, "trust": 0.92},
        {"tier": "Terminal", "node": "terminal-3", "load": 0.62, "compute": 1.25, "network": 0.26, "trust": 0.94},
        {"tier": "Terminal", "node": "terminal-4", "load": 0.70, "compute": 1.42, "network": 0.34, "trust": 0.90},
        {"tier": "Terminal", "node": "terminal-5", "load": 0.54, "compute": 1.10, "network": 0.20, "trust": 0.98},
        {"tier": "Cloud", "node": "cloud-1", "load": 0.78, "compute": 1.12, "network": 1.28, "trust": 0.96},
        {"tier": "Edge", "node": "edge-1", "load": 0.68, "compute": 0.82, "network": 0.76, "trust": 1.06},
        {"tier": "Edge", "node": "edge-2", "load": 0.74, "compute": 0.94, "network": 0.84, "trust": 1.02},
        {"tier": "Edge", "node": "edge-3", "load": 0.86, "compute": 1.08, "network": 0.92, "trust": 0.92},
        {"tier": "On-premises", "node": "on-prem-1", "load": 0.92, "compute": 1.48, "network": 0.60, "trust": 0.84},
        {"tier": "Continuum", "node": "continuum-1", "load": 0.72, "compute": 0.88, "network": 0.82, "trust": 1.04},
    ]
    workloads = [
        {"workload": "heavy", "compute": 1.42, "timeliness": 0.82, "trust": 0.88},
        {"workload": "stringent", "compute": 0.26, "timeliness": 1.08, "trust": 1.08},
    ]
    dt_counts = [25, 50, 75]

    rows: list[Dict[str, Any]] = []
    for profile in node_profiles:
        node_cpu_values: list[float] = []
        node_bw_values: list[float] = []
        node_dtt_values: list[float] = []
        node_timeliness_values: list[float] = []

        for workload in workloads:
            for dt_count in dt_counts:
                scale = dt_count / 50.0
                compute_time = (
                    processing_ms * profile["compute"] * workload["compute"] * (0.78 + 0.44 * scale)
                    + network_ms * profile["network"] * 0.18
                )
                timeliness = _clip(
                    base_timeliness
                    * workload["timeliness"]
                    * profile["trust"]
                    - max(0.0, scale - 1.0) * 0.08
                    - max(0.0, profile["load"] - 0.82) * 0.20,
                    0.0,
                    1.0,
                )
                availability = _clip(
                    base_availability - max(0.0, profile["load"] - 0.80) * 0.45 - max(0.0, scale - 1.0) * 0.04,
                    0.0,
                    1.0,
                )
                dtt = _clip(base_dtt * profile["trust"] * workload["trust"] * availability, 0.0, 1.0)
                cpu_util = _clip((profile["load"] * 72.0) + scale * 12.0 + workload["compute"] * 4.5, 5.0, 99.0)
                bandwidth_util = _clip((network_ms / 110.0) * 100.0 * profile["network"] + scale * 9.0, 5.0, 99.0)

                node_cpu_values.append(cpu_util)
                node_bw_values.append(bandwidth_util)
                node_dtt_values.append(dtt)
                node_timeliness_values.append(timeliness)
                rows.append(
                    {
                        "algorithm": algorithm,
                        "epoch": epoch.upper(),
                        "tier": profile["tier"],
                        "node": profile["node"],
                        "workload": workload["workload"],
                        "dt_count": dt_count,
                        "compute_time_ms": round(float(compute_time), 4),
                        "timeliness": round(float(timeliness), 4),
                        "dtt_score": round(float(dtt), 4),
                        "cpu_utilization": round(float(cpu_util), 4),
                        "bandwidth_utilization": round(float(bandwidth_util), 4),
                        "availability": round(float(availability), 4),
                    }
                )

        avg_dtt = float(np.mean(node_dtt_values))
        avg_timeliness = float(np.mean(node_timeliness_values))
        avg_availability = float(np.mean([row["availability"] for row in rows if row["node"] == profile["node"]]))
        status_code = 2 if avg_dtt >= 0.70 and avg_timeliness >= 0.75 and avg_availability >= 0.82 else 1
        if avg_dtt < 0.45 or avg_availability < 0.65:
            status_code = 0

        for row in rows:
            if row["node"] == profile["node"]:
                row["node_status_code"] = status_code
                row["node_status"] = _node_status_label(status_code)

    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    frames = [
        build_node_status("DT-FedDQL", "R5"),
        build_node_status("No-DT-FedDQL", "R5"),
    ]
    df = pd.concat(frames, ignore_index=True)
    out_path = RESULTS_DIR / "node_status_snapshot.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote node-status snapshot: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
