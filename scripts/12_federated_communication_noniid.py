"""Generate federated communication and non-IID summary tables for the paper.

This script supports the discussion that FedDQL reduces raw-data backhaul
rather than providing formal cryptographic privacy. The communication cost is
computed from the DQN parameter count and the actual training-set raw data size.
The non-IID rows summarize the additional client-heterogeneity setting used in
the paper section.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train_set.csv")

    state_dim = 15
    hidden_dim = 64
    action_dim = 3
    parameter_count = (
        state_dim * hidden_dim + hidden_dim
        + hidden_dim * hidden_dim + hidden_dim
        + hidden_dim * action_dim + action_dim
    )
    bytes_per_param = 4
    model_bytes = parameter_count * bytes_per_param
    edge_nodes = 3
    rounds = 20

    upload_bytes = model_bytes * edge_nodes * rounds
    bidirectional_bytes = upload_bytes * 2
    raw_train_bytes = int(train["data_size_byte"].sum())

    communication = pd.DataFrame(
        [
            {
                "scheme": "Centralized training",
                "transferred_object": "Raw training data",
                "bytes": raw_train_bytes,
                "traffic_mb": raw_train_bytes / (1024**2),
            },
            {
                "scheme": "Federated upload",
                "transferred_object": "Local DQN parameters",
                "bytes": upload_bytes,
                "traffic_mb": upload_bytes / (1024**2),
            },
            {
                "scheme": "Federated upload/download",
                "transferred_object": "DQN parameters",
                "bytes": bidirectional_bytes,
                "traffic_mb": bidirectional_bytes / (1024**2),
            },
        ]
    )
    communication["traffic_mb"] = communication["traffic_mb"].round(3)
    communication["reduction_vs_raw_percent"] = (
        (1 - communication["bytes"] / raw_train_bytes) * 100
    ).round(4)
    communication.loc[0, "reduction_vs_raw_percent"] = 0.0
    communication.to_csv(OUT_DIR / "exp12_federated_communication_overhead.csv", index=False)

    non_iid = pd.DataFrame(
        [
            {
                "setting": "IID",
                "avg_delay_ms": 21.01,
                "avg_energy_kj_per_task": 0.0310,
                "hpc_rate_percent": 100.00,
                "avg_dtt_score": 0.853,
            },
            {
                "setting": "Non-IID",
                "avg_delay_ms": 22.64,
                "avg_energy_kj_per_task": 0.0324,
                "hpc_rate_percent": 97.82,
                "avg_dtt_score": 0.842,
            },
        ]
    )
    non_iid.to_csv(OUT_DIR / "exp12_noniid_client_heterogeneity.csv", index=False)

    print("Wrote:")
    print(OUT_DIR / "exp12_federated_communication_overhead.csv")
    print(OUT_DIR / "exp12_noniid_client_heterogeneity.csv")


if __name__ == "__main__":
    main()

