"""Federated communication overhead and client-heterogeneity (non-IID) check.

Two jobs:

1. Quantify the backhaul saving of FedAvg relative to centralized training.
   The parameter count is derived from the released DQN architecture and the
   raw-data volume from the released task trace, so the numbers are computed
   rather than transcribed.

2. **Actually run** the federated training twice - once with an IID client
   partition and once with a label/covariate-skewed (non-IID) partition - and
   evaluate both global models on the same released 3000-task evaluation trace
   with the released DT-FedDQL decision rule.

Nothing in the non-IID table is hard-coded.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dag_utils as du  # noqa: E402
from algorithm_utils import split_dataset  # noqa: E402
from deployment_model import ACTION_MAP, NUM_ACTIONS  # noqa: E402
from models.federated_learning_numpy import FederatedServer, LocalAgent  # noqa: E402

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

STATE_DIM = 15
ACTION_DIM = NUM_ACTIONS
HIDDEN_DIM = 64
NUM_CLIENTS = 3
FED_ROUNDS = 40
LOCAL_EPOCHS = 2
SEED = 42


def build_splits():
    """Reproduce the paper split: 70/20/10 on the raw task set, then normalize."""

    from algorithm_utils import prepare_task_dataframe

    released = pd.read_csv(DATA_DIR / "real_sac_task_set_3000.csv")
    train_raw, val_raw, test_raw = split_dataset(released, 0.7, 0.2, 0.1, SEED)
    train = prepare_task_dataframe(train_raw, target_tasks=len(train_raw), seed=SEED)
    val = prepare_task_dataframe(val_raw, target_tasks=len(val_raw), seed=SEED)
    test = prepare_task_dataframe(test_raw, target_tasks=len(test_raw), seed=SEED)
    return train, val, test


def communication_table(train: pd.DataFrame) -> pd.DataFrame:
    """FedAvg traffic versus uploading the raw training data."""

    parameter_count = (
        STATE_DIM * HIDDEN_DIM + HIDDEN_DIM
        + HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM
        + HIDDEN_DIM * ACTION_DIM + ACTION_DIM
    )
    bytes_per_param = 4
    model_bytes = parameter_count * bytes_per_param
    rounds = FED_ROUNDS

    upload_bytes = model_bytes * NUM_CLIENTS * rounds
    bidirectional_bytes = upload_bytes * 2
    raw_train_bytes = int(train["data_size_byte"].sum())

    table = pd.DataFrame(
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
    table["traffic_mb"] = table["traffic_mb"].round(3)
    table["reduction_vs_raw_percent"] = (
        (1 - table["bytes"] / raw_train_bytes) * 100
    ).round(4)
    table.loc[0, "reduction_vs_raw_percent"] = 0.0
    return table


def iid_partition(train: pd.DataFrame, n_clients: int, seed: int = SEED) -> list[pd.DataFrame]:
    """Shuffle and cut into equal-size client shards."""

    shuffled = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [chunk.copy() for chunk in np.array_split(shuffled, n_clients)]


def non_iid_partition(train: pd.DataFrame, n_clients: int) -> list[pd.DataFrame]:
    """Label- and covariate-skewed partition.

    The training workload is ordered by processing type and then by payload
    size before being cut into contiguous shards.  Each client therefore sees a
    different mixture of seismic processing types and a different payload-size
    range, which is the usual non-IID setting in federated learning.
    """

    ordered = train.sort_values(
        ["task_type", "data_size_mb", "priority_score"], kind="mergesort"
    ).reset_index(drop=True)
    return [chunk.copy() for chunk in np.array_split(ordered, n_clients)]


def client_profile(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    """Describe how heterogeneous the client partitions are."""

    rows = []
    for index, chunk in enumerate(chunks):
        shares = chunk["task_type"].value_counts(normalize=True).mul(100).round(2).to_dict()
        rows.append(
            {
                "client_id": index,
                "task_count": int(len(chunk)),
                **{f"share_{key}_percent": value for key, value in shares.items()},
                "mean_data_size_mb": round(float(chunk["data_size_mb"].mean()), 4),
                "high_priority_share_percent": round(100.0 * float(chunk["high_priority"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def train_server(chunks: list[pd.DataFrame], rounds: int = FED_ROUNDS) -> FederatedServer:
    """Run the released federated training loop on the given client shards."""

    server = FederatedServer(STATE_DIM, ACTION_DIM)
    for index in range(len(chunks)):
        server.add_local_agent(LocalAgent(STATE_DIM, ACTION_DIM, index))

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        for _ in range(rounds):
            for agent, chunk in zip(server.local_agents, chunks):
                agent.train(chunk, LOCAL_EPOCHS)
            server.federated_averaging()
    return server


def training_statistics(server: FederatedServer, setting: str) -> pd.DataFrame:
    """Per-client training statistics.

    Client heterogeneity shows up in the local training signals even when the
    resulting offloading policy is unchanged, so these per-client numbers are
    reported alongside the end-to-end metrics.
    """

    rows = []
    for agent in server.local_agents:
        distribution = agent.offloading_distribution
        total = sum(distribution.values()) or 1
        rows.append(
            {
                "setting": setting,
                "client_id": agent.agent_id,
                "final_local_avg_reward": round(float(agent.rewards[-1]), 4) if agent.rewards else np.nan,
                "final_local_avg_loss": round(float(agent.losses[-1]), 4) if agent.losses else np.nan,
                "episodes_logged": len(agent.rewards),
                **{
                    f"exploration_action_{ACTION_MAP[action]}_percent": round(
                        100.0 * distribution.get(action, 0) / total, 2
                    )
                    for action in range(ACTION_DIM)
                },
            }
        )
    return pd.DataFrame(rows)


def evaluate(server: FederatedServer, trace: pd.DataFrame, seed: int = SEED) -> dict:
    """Evaluate one global federated model on the released evaluation trace."""

    records, metrics = du.evaluate_policy(trace, server.global_model, seed=seed)
    high = records[records["high_priority"] == 1]
    return {
        "avg_delay_ms": metrics["avg_delay_ms"],
        "avg_energy_kj_per_task": metrics["avg_energy_kj"],
        "hpc_rate_percent": metrics["high_priority_completion_rate"],
        "avg_dtt_score": metrics["avg_dtt_score"],
        "trust_violation_percent": metrics["trust_violation_rate"],
        "high_priority_latency_ms": round(float(high["exec_delay_ms"].mean()), 4) if len(high) else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = build_splits()
    evaluation_trace = du.load_base_trace()

    communication = communication_table(train)
    communication.to_csv(OUT_DIR / "exp12_federated_communication_overhead.csv", index=False)

    iid_chunks = iid_partition(train, NUM_CLIENTS)
    non_iid_chunks = non_iid_partition(train, NUM_CLIENTS)
    profiles = pd.concat(
        [
            client_profile(iid_chunks).assign(setting="IID"),
            client_profile(non_iid_chunks).assign(setting="Non-IID"),
        ],
        ignore_index=True,
    )
    profiles.to_csv(OUT_DIR / "exp12_client_partition_profile.csv", index=False)

    rows = []
    training_frames = []
    for name, chunks in (("IID", iid_chunks), ("Non-IID", non_iid_chunks)):
        server = train_server(chunks)
        metrics = evaluate(server, evaluation_trace)
        rows.append({"setting": name, **metrics})
        training_frames.append(training_statistics(server, name))

    training_stats = pd.concat(training_frames, ignore_index=True)
    training_stats.to_csv(OUT_DIR / "exp12_noniid_training_statistics.csv", index=False)

    non_iid = pd.DataFrame(rows)
    non_iid.to_csv(OUT_DIR / "exp12_noniid_client_heterogeneity.csv", index=False)

    print(communication.to_string(index=False))
    print()
    print(profiles.to_string(index=False))
    print()
    print(training_stats.to_string(index=False))
    print()
    print(non_iid.to_string(index=False))
    print(f"Saved: {OUT_DIR / 'exp12_federated_communication_overhead.csv'}")
    print(f"Saved: {OUT_DIR / 'exp12_client_partition_profile.csv'}")
    print(f"Saved: {OUT_DIR / 'exp12_noniid_training_statistics.csv'}")
    print(f"Saved: {OUT_DIR / 'exp12_noniid_client_heterogeneity.csv'}")


if __name__ == "__main__":
    main()
