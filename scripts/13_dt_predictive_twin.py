"""Evaluate a lightweight predictive digital twin from monitoring history.

The script trains a small MLP to predict next-slot runtime metrics from a
sliding history window. It is used to support the paper's claim that the
digital-twin layer is not only a threshold table, but also maintains state
history and performs pre-execution prediction for what-if orchestration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


FEATURES = [
    "network_latency_ms",
    "processing_time_ms",
    "end_to_end_latency_ms",
    "dtt_score",
    "timeliness",
    "reliability",
    "availability",
]
TARGETS = ["end_to_end_latency_ms", "dtt_score", "availability"]


def make_windows(values: np.ndarray, history: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    target_idx = [FEATURES.index(t) for t in TARGETS]
    for i in range(history, len(values)):
        x.append(values[i - history : i].reshape(-1))
        y.append(values[i, target_idx])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def one_hot(labels: np.ndarray) -> np.ndarray:
    uniq = sorted(pd.unique(labels))
    mapping = {v: i for i, v in enumerate(uniq)}
    out = np.zeros((len(labels), len(uniq)), dtype=float)
    for r, label in enumerate(labels):
        out[r, mapping[label]] = 1.0
    return out


def train_mlp(x_train: np.ndarray, y_train: np.ndarray, hidden: int = 16, epochs: int = 3000) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(42)
    w1 = rng.normal(0.0, 0.08, size=(x_train.shape[1], hidden))
    b1 = np.zeros((1, hidden))
    w2 = rng.normal(0.0, 0.08, size=(hidden, y_train.shape[1]))
    b2 = np.zeros((1, y_train.shape[1]))
    lr = 0.015
    n = len(x_train)

    for _ in range(epochs):
        h = np.tanh(x_train @ w1 + b1)
        pred = h @ w2 + b2
        err = pred - y_train
        grad_pred = 2.0 * err / n
        grad_w2 = h.T @ grad_pred
        grad_b2 = grad_pred.sum(axis=0, keepdims=True)
        grad_h = grad_pred @ w2.T
        grad_z = grad_h * (1.0 - h**2)
        grad_w1 = x_train.T @ grad_z
        grad_b1 = grad_z.sum(axis=0, keepdims=True)
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
    return w1, b1, w2, b2


def predict(x: np.ndarray, params: tuple[np.ndarray, ...]) -> np.ndarray:
    w1, b1, w2, b2 = params
    return np.tanh(x @ w1 + b1) @ w2 + b2


def main() -> None:
    df = pd.read_csv(RESULTS / "dynamic_trust_experiment.csv")
    df = df[df["algorithm"].eq("DT-FedDQL") & df["digital_twin_orchestration"].eq(1)]
    df = df.sort_values("time_s").reset_index(drop=True)

    values = df[FEATURES].astype(float).to_numpy()
    phase = df["epoch"].astype(str).to_numpy()
    history = 3
    x, y = make_windows(values, history)
    phase_context = one_hot(phase[history:])
    x = np.hstack([x, phase_context])

    rng = np.random.default_rng(7)
    train_idx, test_idx = [], []
    valid_phase = phase[history:]
    for ph in sorted(pd.unique(valid_phase)):
        idx = np.where(valid_phase == ph)[0]
        rng.shuffle(idx)
        split = max(1, int(len(idx) * 0.7))
        train_idx.extend(idx[:split].tolist())
        test_idx.extend(idx[split:].tolist())
    train_idx = np.asarray(sorted(train_idx))
    test_idx = np.asarray(sorted(test_idx))
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-9
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0) + 1e-9
    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std

    params = train_mlp(x_train_n, y_train_n)
    pred_mlp = predict(x_test_n, params) * y_std + y_mean

    # Baseline: current-value persistence predictor using the most recent slot.
    last_indices = [(history - 1) * len(FEATURES) + FEATURES.index(t) for t in TARGETS]
    pred_current = x_test[:, last_indices]

    rows = []
    for name, pred in [("Current-state baseline", pred_current), ("Predictive DT-MLP", pred_mlp)]:
        for j, target in enumerate(TARGETS):
            err = pred[:, j] - y_test[:, j]
            rows.append(
                {
                    "model": name,
                    "target": target,
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "exp13_dt_predictive_twin_error.csv", index=False)

    pred_df = pd.DataFrame(
        np.column_stack([y_test, pred_current, pred_mlp]),
        columns=(
            [f"actual_{t}" for t in TARGETS]
            + [f"current_pred_{t}" for t in TARGETS]
            + [f"mlp_pred_{t}" for t in TARGETS]
        ),
    )
    pred_df.to_csv(OUT / "exp13_dt_predictive_twin_predictions.csv", index=False)

    print(metrics.round(6).to_string(index=False))
    print("Wrote:")
    print(OUT / "exp13_dt_predictive_twin_error.csv")
    print(OUT / "exp13_dt_predictive_twin_predictions.csv")


if __name__ == "__main__":
    main()

