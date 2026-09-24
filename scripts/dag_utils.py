"""Shared utilities for DAG-structure stress tests.

The released seismic workload is a set of mostly independent tasks with a small
number of single-predecessor chains.  To study how the policy behaves under
deeper sequential dependencies and wider parallel branches we do **not**
post-process the recorded metrics.  Instead we rebuild the dependency
structure of the released task trace and re-run the actual policy on the
re-wired tasks, so all reported metrics come from the same simulator,
the same trust evaluation, and the same released model weights.

Rebuilding only touches the dependency columns (``dependency_ids``,
``dependency_count``, ``in_degree``, ``out_degree``).  These three degrees are
part of the 15-dimensional state vector, so re-wiring changes the policy input
and the execution order constraints rather than merely shifting numbers.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_utils import (  # noqa: E402
    ACTION_ID,
    calculate_metrics,
    load_task_set,
    prepare_task_dataframe,
    run_policy_on_tasks,
)
from deployment_model import EDGE_ACTION_IDS, NUM_ACTIONS  # noqa: E402
from models.federated_learning_numpy import NeuralNetwork, get_state  # noqa: E402

STATE_DIM = 15
ACTION_DIM = NUM_ACTIONS
WEIGHTS_NAME = "fed_dql_federated_model_weights.pkl"


def load_base_trace(target_tasks: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Load the released task trace with the full normalized feature set."""

    return prepare_task_dataframe(load_task_set("real"), target_tasks=target_tasks, seed=seed)


def load_trained_model(results_dir: Path | None = None) -> NeuralNetwork:
    """Load the released federated global model.

    Only the aggregated global weights are released (about 43 KB).  They are
    the exact model that produced ``results/dt_feddql_result.csv``.
    """

    results_dir = results_dir or (PROJECT_ROOT / "results")
    with open(results_dir / WEIGHTS_NAME, "rb") as handle:
        weights = pickle.load(handle)
    model = NeuralNetwork(STATE_DIM, ACTION_DIM)
    model.set_weights(weights)
    return model


def make_feddql_policy(model: NeuralNetwork):
    """The decision rule used for the reported DT-FedDQL trace.

    High-priority tasks are pinned to the least-loaded edge node so that the
    priority reservation is not defeated by queue backlog; every other task
    takes the greedy action of the federated global Q-network, whose input
    includes the live load of each execution location.
    """

    def policy(task: pd.Series, loads: dict | None = None) -> int:
        loads = loads or {}
        if bool(task.get("high_priority", 0)):
            return int(
                min(EDGE_ACTION_IDS, key=lambda a: float(loads.get(f"load_edge_{a}", 0.0)))
            )
        state = get_state(task, loads)
        q_values = model.forward(state.reshape(1, -1))[0]
        return int(np.argmax(q_values))

    return policy


def rewire_dependencies(
    tasks: pd.DataFrame,
    dependent_ratio: float,
    width: int = 1,
) -> pd.DataFrame:
    """Rebuild the dependency graph as a layered DAG.

    ``dependent_ratio`` is the target fraction of tasks that receive at least
    one predecessor.  ``width`` is the number of parallel predecessors each
    task in a level depends on, i.e. the branch width of the DAG.  Tasks are
    laid out on evenly spaced positions so that a predecessor always occupies
    an earlier row, which keeps the reconstructed DAG executable in row order.

    Returns a copy of ``tasks`` with only the dependency columns replaced.
    """

    frame = tasks.copy()
    n_tasks = len(frame)
    frame["dependency_ids"] = [[] for _ in range(n_tasks)]
    frame["dependency_count"] = 0
    frame["in_degree"] = 0
    frame["out_degree"] = 0

    width = max(1, int(width))
    if dependent_ratio <= 0 or n_tasks < 3:
        return frame

    n_dependent = int(round(n_tasks * float(dependent_ratio)))
    n_dependent = max(0, min(n_dependent, n_tasks - width - 1))
    if n_dependent == 0:
        return frame

    # Level 0 of the DAG carries no predecessor and only serves as the source
    # of the widest level, hence the extra ``width`` positions.
    positions = np.unique(
        np.round(np.linspace(1, n_tasks - 1, n_dependent + width)).astype(int)
    )

    dependency_ids: dict[int, list[int]] = {}
    in_degree = np.zeros(n_tasks, dtype=int)
    out_degree = np.zeros(n_tasks, dtype=int)

    for position, task_index in enumerate(positions):
        task_index = int(task_index)
        level = position // width
        if level == 0:
            dependency_ids[task_index] = []
            continue
        first = (level - 1) * width
        predecessors = [int(positions[q]) for q in range(first, min(first + width, len(positions)))]
        predecessors = sorted({p for p in predecessors if p < task_index})
        dependency_ids[task_index] = predecessors
        for predecessor in predecessors:
            out_degree[predecessor] += 1
            in_degree[task_index] += 1

    frame["dependency_ids"] = [dependency_ids.get(i, []) for i in range(n_tasks)]
    frame["dependency_count"] = [len(d) for d in frame["dependency_ids"]]
    frame["in_degree"] = in_degree
    frame["out_degree"] = out_degree
    return frame


def graph_statistics(tasks: pd.DataFrame) -> dict:
    """Describe the dependency graph actually present in ``tasks``."""

    counts = tasks["dependency_count"].to_numpy(int)
    in_deg = tasks["in_degree"].to_numpy(int) if "in_degree" in tasks else np.zeros_like(counts)
    out_deg = tasks["out_degree"].to_numpy(int) if "out_degree" in tasks else np.zeros_like(counts)
    dependent = int((counts > 0).sum())
    return {
        "task_count": int(len(tasks)),
        "dependent_task_count": dependent,
        "dependent_task_ratio_percent": round(100.0 * dependent / max(len(tasks), 1), 2),
        "mean_predecessors": round(float(counts[counts > 0].mean()) if dependent else 0.0, 3),
        "max_predecessors": int(counts.max()) if len(counts) else 0,
        "max_in_degree": int(in_deg.max()) if len(in_deg) else 0,
        "max_out_degree": int(out_deg.max()) if len(out_deg) else 0,
    }


def evaluate_policy(
    tasks: pd.DataFrame,
    model: NeuralNetwork,
    seed: int = 42,
    orchestration: bool = True,
    trust_threshold: float = 0.80,
) -> tuple[pd.DataFrame, dict]:
    """Run the DT-FedDQL policy on ``tasks`` and return records plus metrics."""

    records = run_policy_on_tasks(
        tasks,
        make_feddql_policy(model),
        "DT-FedDQL",
        seed=seed,
        dependency_aware=True,
        priority_qos=True,
        digital_twin_orchestration=orchestration,
        trust_threshold=trust_threshold,
    )
    return records, calculate_metrics(records)


def latency_only_metrics(records: pd.DataFrame, baseline_delay: float) -> dict:
    """Extra diagnostics that need the per-task records rather than the summary."""

    dependency_wait = int((records["dependency_met"] == 0).sum())
    return {
        "tasks_with_unmet_predecessor": dependency_wait,
        "tasks_with_unmet_predecessor_percent": round(
            100.0 * dependency_wait / max(len(records), 1), 3
        ),
        "delay_delta_vs_base_ms": round(float(records["exec_delay_ms"].mean()) - baseline_delay, 4),
    }
