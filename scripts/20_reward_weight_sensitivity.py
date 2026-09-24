"""Reward-weight sensitivity, measured by retraining.

The reward trade-off reported in the paper is selected on the validation set.
This script verifies that claim instead of asserting it: the latency weight of
the implemented multi-objective reward is swept around the selected value, the
remaining weights are rescaled so the weight vector still sums to one, and a
federated model is trained from scratch for every setting.  All reported
metrics are then measured on the same released evaluation trace.

Set ``QUICK = True`` to reduce the federated rounds for a smoke test only.
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
from algorithm_utils import prepare_task_dataframe, split_dataset  # noqa: E402
from deployment_model import NUM_ACTIONS  # noqa: E402
from models.federated_learning_numpy import (  # noqa: E402
    DEFAULT_REWARD_WEIGHTS,
    FederatedServer,
    LocalAgent,
)

OUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

STATE_DIM = 15
ACTION_DIM = NUM_ACTIONS
NUM_CLIENTS = 3
FED_ROUNDS = 40
LOCAL_EPOCHS = 2
SEED = 42
QUICK = False

# The latency weight omega_1 is swept around the selected value 0.25.
LATENCY_WEIGHTS = (0.15, 0.25, 0.35)


def renormalised_weights(latency_weight: float) -> dict:
    """Set the latency weight and rescale the rest so the vector sums to one."""

    weights = dict(DEFAULT_REWARD_WEIGHTS)
    weights["latency"] = latency_weight
    others = [key for key in weights if key != "latency"]
    remaining = 1.0 - latency_weight
    current = sum(weights[key] for key in others)
    for key in others:
        weights[key] = weights[key] / current * remaining
    return weights


def build_train_set() -> pd.DataFrame:
    released = pd.read_csv(DATA_DIR / "real_sac_task_set_3000.csv")
    train_raw, _, _ = split_dataset(released, 0.7, 0.2, 0.1, SEED)
    return prepare_task_dataframe(train_raw, target_tasks=len(train_raw), seed=SEED)


def train_and_evaluate(train: pd.DataFrame, trace: pd.DataFrame, weights: dict) -> dict:
    rounds = 2 if QUICK else FED_ROUNDS
    server = FederatedServer(STATE_DIM, ACTION_DIM)
    for index in range(NUM_CLIENTS):
        agent = LocalAgent(STATE_DIM, ACTION_DIM, index)
        agent.reward_weights = weights
        server.add_local_agent(agent)

    chunks = [chunk.copy() for chunk in np.array_split(train, NUM_CLIENTS)]
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        for _ in range(rounds):
            for agent, chunk in zip(server.local_agents, chunks):
                agent.train(chunk, LOCAL_EPOCHS)
            server.federated_averaging()

    records, metrics = du.evaluate_policy(trace, server.global_model, seed=SEED)
    high = records[records["high_priority"] == 1]
    return {
        "avg_delay_ms": metrics["avg_delay_ms"],
        "avg_energy_kj": metrics["avg_energy_kj"],
        "hpc_percent": metrics["high_priority_completion_rate"],
        "avg_dtt_score": metrics["avg_dtt_score"],
        "trust_violation_percent": metrics["trust_violation_rate"],
        "high_priority_latency_ms": round(float(high["exec_delay_ms"].mean()), 4) if len(high) else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = build_train_set()
    trace = du.load_base_trace()

    rows = []
    for latency_weight in LATENCY_WEIGHTS:
        weights = renormalised_weights(latency_weight)
        result = train_and_evaluate(train, trace, weights)
        rows.append(
            {
                "omega_1_latency": round(latency_weight, 4),
                "omega_2_energy": round(weights["energy"], 4),
                "omega_3_hpc": round(weights["completion"], 4),
                "omega_4_dtt": round(weights["trust"], 4),
                "omega_5_violation": round(weights["violation"], 4),
                "fed_rounds": 2 if QUICK else FED_ROUNDS,
                **result,
            }
        )

    df = pd.DataFrame(rows)
    path = OUT_DIR / "exp20_reward_weight_sensitivity.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
