"""Distinct baseline offloading policies used in the paper's baseline comparison.

Each policy mirrors the description published in
``outputs/exp19_baseline_implementation_sources.csv`` so that the reported
numbers are reproducible and the baselines are not duplicates of each other.

Every policy returns a **node-level** action id from ``A = {L, e1, e2, e3, C}``.
The heuristic baselines are deliberately *myopic*: they have no view of the
queue backlog at the execution locations, so the only edge node they can pick
is the fastest one.  That is precisely the behaviour the learned policy has to
beat -- under a capacity-constrained, heterogeneous edge tier, always choosing
the strongest node saturates it and loses the deadline.

Thresholds are expressed on the normalized workload features and were fixed
once from the workload quantiles; they are not re-tuned per run.
"""

from __future__ import annotations

import pandas as pd

from algorithm_utils import ACTION_ID
from deployment_model import EDGE_ACTION_IDS

# The myopic heuristics can only see node capability, not node load, so they
# all select the strongest edge node.
EDGE = ACTION_ID["edge_1"]
TERMINAL = ACTION_ID["terminal"]
CLOUD = ACTION_ID["cloud"]


class GreedyPolicy:
    """Greedy heuristic driven by latency urgency, workload size, and resource availability.

    The rule is myopic: it thresholds a latency-urgency index and a resource
    footprint, and uses no queueing estimate, no learning, and no trust check.
    """

    def __init__(self, urgency_threshold: float = 0.10, footprint_threshold: float = 0.22) -> None:
        self.urgency_threshold = urgency_threshold
        self.footprint_threshold = footprint_threshold

    def __call__(self, task: pd.Series) -> int:
        delay = float(task["delay_norm"])
        compute = float(task["compute_norm"])
        data = float(task["data_size_norm"])
        priority = float(task["priority_score"])
        memory = float(task.get("memory_norm", 0.0))

        urgency = 0.45 * delay + 0.30 * priority + 0.25 * compute
        footprint = 0.55 * compute + 0.25 * data + 0.20 * memory

        if urgency >= self.urgency_threshold:
            return EDGE
        if footprint >= self.footprint_threshold:
            return CLOUD
        return TERMINAL


class FedServPolicy:
    """Federated service-offloading style policy under the same simulator.

    Prioritized tasks are admitted to a reserved edge service class.  The
    remaining edge capacity is filled in arrival order and spread over the edge
    nodes by an *admission counter*, which is the closest thing to load
    awareness available without runtime feedback.  Once every edge node is
    filled, tasks fall back to the terminal.  No learning and no digital-twin
    trust check are involved.
    """

    def __init__(self, edge_capacity: float = 0.85, reserved_priority: float = 0.65) -> None:
        self.edge_capacity = edge_capacity
        self.reserved_priority = reserved_priority
        self._admitted = 0
        self._seen = 0
        self._per_node = {action: 0 for action in EDGE_ACTION_IDS}

    def _least_admitted(self) -> int:
        return int(min(EDGE_ACTION_IDS, key=lambda action: self._per_node[action]))

    def __call__(self, task: pd.Series) -> int:
        self._seen += 1
        utilization = self._admitted / max(self._seen, 1)
        priority = float(task["priority_score"])

        if priority >= self.reserved_priority or utilization < self.edge_capacity:
            self._admitted += 1
            node = self._least_admitted()
            self._per_node[node] += 1
            return node
        return TERMINAL


def make_fuzzy_dql_policy():
    """Fuzzy-prior-guided policy: uses the fuzzy winner action only.

    No federated aggregation and no digital-twin re-orchestration are applied,
    which matches the published description of this baseline.  The fuzzy
    classifier emits a layer-level winner, which is resolved to the strongest
    edge node because the baseline has no load observation.
    """
    from models.fuzzy_classifier import FuzzyTaskClassifier

    classifier = FuzzyTaskClassifier()

    def policy(task: pd.Series) -> int:
        return classifier.classify_row(task)

    return policy


def milp_policy(task: pd.Series) -> int:
    """Deterministic optimization-inspired assignment under the same objectives.

    No exact MILP optimality gap is claimed for this rule.
    """
    delay = float(task["delay_norm"])
    compute = float(task["compute_norm"])
    data = float(task["data_size_norm"])
    priority = float(task["priority_score"])
    if priority > 0.8:
        if delay > 0.6 or compute > 0.5:
            return CLOUD
        return EDGE
    if compute > 0.7:
        return EDGE
    if data > 0.6:
        return CLOUD
    return TERMINAL


_CENTRALIZED_MODEL = None


def _centralized_model():
    """Lazily load the trained centralized-DQN weights.

    The weights are produced by ``train_centralized_dqn.py`` (single agent,
    full training workload, same architecture/reward as the federated
    agents).  Raising here instead of silently falling back to a heuristic
    keeps the baseline honest: if the weights are missing, the failure is
    loud and names the fix.
    """

    global _CENTRALIZED_MODEL
    if _CENTRALIZED_MODEL is None:
        import pickle
        from pathlib import Path

        import numpy as np  # noqa: F401  (weights are numpy arrays)

        from deployment_model import NUM_ACTIONS
        from models.federated_learning_numpy import NeuralNetwork

        weights_path = (
            Path(__file__).resolve().parents[1] / "results" / "centralized_dqn_weights.pkl"
        )
        if not weights_path.exists():
            raise SystemExit(
                f"missing trained baseline weights: {weights_path}; "
                "run train_centralized_dqn.py first"
            )
        model = NeuralNetwork(15, NUM_ACTIONS)
        with open(weights_path, "rb") as handle:
            model.set_weights(pickle.load(handle))
        _CENTRALIZED_MODEL = model
    return _CENTRALIZED_MODEL


def centralized_dqn_policy(task: pd.Series, loads: dict | None = None) -> int:
    """Single-DQN baseline with centralized state observation.

    Argmax of a DQN trained centrally on the full training workload
    (``train_centralized_dqn.py``) with the same architecture, reward, and
    environment as the federated agents; only federated aggregation and the
    digital-twin layer are removed.  The 15-dimensional state carries the
    live load of every execution location, which is exactly the centralized
    observation described in the manuscript.  High-priority tasks use the
    same priority-QoS pinning as the proposed-family policies, so the
    comparison isolates centralized versus federated *training*.
    """

    import numpy as np

    from models.federated_learning_numpy import get_state

    loads = loads or {}
    if bool(task.get("high_priority", 0)):
        return int(min(EDGE_ACTION_IDS, key=lambda a: float(loads.get(f"load_edge_{a}", 0.0))))
    state = get_state(task, loads)
    q_values = _centralized_model().forward(state.reshape(1, -1))[0]
    return int(np.argmax(q_values))


def action_histogram(records: pd.DataFrame) -> dict:
    counts = records["action"].value_counts().to_dict()
    total = float(len(records))
    return {k: round(100.0 * v / total, 2) for k, v in counts.items()}


__all__ = [
    "GreedyPolicy",
    "FedServPolicy",
    "make_fuzzy_dql_policy",
    "milp_policy",
    "centralized_dqn_policy",
    "action_histogram",
]
