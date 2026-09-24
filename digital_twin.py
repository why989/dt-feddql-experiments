from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from deployment_model import (
    ACTION_ID,
    ACTION_MAP,
    EDGE_ACTION_IDS,
    EDGE_NODE_BY_ACTION,
    edge_node_speedup,
)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _task_float(task: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        return float(task.get(name, default))
    except (TypeError, ValueError):
        return default


def _priority_qos_enabled(task: pd.Series, action: int, priority_qos: bool) -> bool:
    priority = _task_float(task, "priority_score", 0.5)
    high_priority = bool(task.get("high_priority", priority >= 0.8))
    return bool(priority_qos and high_priority and int(action) in EDGE_ACTION_IDS)


def estimate_action_trust(
    task: pd.Series,
    action: int,
    load_hint: Tuple[float, float] | None = None,
    priority_qos: bool = False,
    queue_wait_ms: float = 0.0,
) -> Dict[str, float]:
    """Estimate Digital-Twin trustworthiness before executing an offloading action.

    The score is a lightweight simulation-side digital twin, not a measured value
    from real Docker/Kubernetes hardware. It combines three runtime invariants:
    timeliness, reliability, and resource availability.  ``queue_wait_ms`` is the
    FCFS backlog currently held by the candidate execution location, which the
    twin must account for before it can certify an action.
    """

    c = _task_float(task, "compute_norm", 0.5)
    d = _task_float(task, "data_size_norm", 0.5)
    priority = _task_float(task, "priority_score", 0.5)
    deadline_ms = max(_task_float(task, "deadline_ms", 50.0), 1.0)
    data_in = _task_float(task, "data_size_mb", d * 100.0)
    qos = _priority_qos_enabled(task, action, priority_qos)
    action = int(action)
    edge_spec = EDGE_NODE_BY_ACTION.get(action)

    if action == ACTION_ID["terminal"]:
        delay_ms = 36.0 + 30.0 * c + 18.0 * d
        cpu = 0.54 + 0.08 * c
        bandwidth = 0.47 + 0.05 * d
    elif edge_spec is not None:
        bandwidth_mbps = float(np.mean(edge_spec["uplink_mbps"]))
        if qos:
            bandwidth_mbps = max(bandwidth_mbps, 80.0)
        comm_delay = (data_in * 8.0) / bandwidth_mbps
        exec_delay = (14.0 + 18.0 * c + 10.0 * d) / edge_node_speedup(action, priority_qos=qos)
        delay_ms = comm_delay + exec_delay
        if qos:
            cpu = 0.78 + 0.10 * c
            bandwidth = 0.76 + 0.10 * d
        else:
            cpu = 0.74 + 0.12 * c
            bandwidth = 0.73 + 0.12 * d
    else:
        edge_uplink = (data_in * 8.0) / 65.0
        cloud_link = (data_in * 8.0) / 70.0
        delay_ms = edge_uplink + cloud_link + 70.0 + 17.0 + 11.0 * c + 15.0 * d
        cpu = 0.46 + 0.10 * c
        bandwidth = 0.62 + 0.20 * d

    delay_ms += max(float(queue_wait_ms), 0.0)

    if load_hint is not None:
        hinted_cpu, hinted_bw = load_hint
        cpu = 0.65 * cpu + 0.35 * float(hinted_cpu)
        bandwidth = 0.65 * bandwidth + 0.35 * float(hinted_bw)

    failure_probability = 0.05 + (1.0 - priority) * 0.15
    if qos:
        failure_probability = max(0.001, failure_probability * 0.08)

    timeliness = _clip01(deadline_ms / max(delay_ms, 1.0))
    reliability = _clip01(1.0 - failure_probability)
    overload = max(0.0, cpu - 0.88, bandwidth - 0.88)
    availability = _clip01(1.0 - overload / 0.12)
    dtt_score = _clip01(timeliness * reliability * availability)

    return {
        "dt_predicted_delay_ms": float(delay_ms),
        "dt_predicted_cpu_util": _clip01(cpu),
        "dt_predicted_bandwidth_util": _clip01(bandwidth),
        "dt_predicted_failure_probability": float(failure_probability),
        "dt_predicted_timeliness": timeliness,
        "dt_predicted_reliability": reliability,
        "dt_predicted_availability": availability,
        "dt_predicted_dtt": dtt_score,
    }


def evaluate_runtime_trust(
    task: pd.Series,
    original_action: int,
    final_action: int,
    runtime: Dict[str, float],
    trust_threshold: float = 0.80,
    decision: Dict[str, float] | None = None,
    recovered: bool = False,
    recovery_time_ms: float = 0.0,
) -> Dict[str, float | int | str]:
    """Evaluate trustworthiness from actual simulated runtime logs."""

    deadline_ms = max(_task_float(task, "deadline_ms", 50.0), 1.0)
    delay_ms = float(runtime.get("exec_delay_ms", deadline_ms * 10.0))
    failure_probability = float(runtime.get("failure_probability", 0.0))
    cpu = float(runtime.get("exec_cpu_util", 0.0))
    bandwidth = float(runtime.get("bandwidth_utilization", 0.0))

    timeliness = _clip01(deadline_ms / max(delay_ms, 1.0))
    reliability = _clip01(1.0 - failure_probability)
    overload = max(0.0, cpu - 0.90, bandwidth - 0.90)
    availability = _clip01(1.0 - overload / 0.10)
    dtt_score = _clip01(timeliness * reliability * availability)
    deadline_met = int(runtime.get("deadline_met", 0))
    trust_violation = int(dtt_score < trust_threshold or deadline_met == 0)

    decision = decision or {}
    return {
        "original_action_id": int(original_action),
        "original_action": ACTION_MAP.get(int(original_action), str(original_action)),
        "reorchestrated": int(original_action != final_action),
        "dt_recovered": int(recovered),
        "dt_recovery_time_ms": float(recovery_time_ms),
        "trust_threshold": float(trust_threshold),
        "dt_timeliness": timeliness,
        "dt_reliability": reliability,
        "dt_availability": availability,
        "dtt_score": dtt_score,
        "trust_violation": trust_violation,
        "dt_original_predicted_dtt": float(decision.get("original_predicted_dtt", np.nan)),
        "dt_selected_predicted_dtt": float(decision.get("selected_predicted_dtt", np.nan)),
    }


@dataclass
class DigitalTwinTrustOrchestrator:
    """Digital-twin trustworthiness orchestrator for the simulation loop."""

    trust_threshold: float = 0.80
    priority_qos: bool = False
    alpha: float = 0.20
    load_hints: Dict[int, Tuple[float, float]] = field(
        default_factory=lambda: {
            ACTION_ID["terminal"]: (0.56, 0.48),
            ACTION_ID["edge_1"]: (0.70, 0.72),
            ACTION_ID["edge_2"]: (0.68, 0.68),
            ACTION_ID["edge_3"]: (0.66, 0.64),
            ACTION_ID["cloud"]: (0.55, 0.70),
        }
    )

    # Map the five node-level actions onto the raw FCFS backlog of each
    # destination, so the twin can price queueing before certifying an action.
    LOAD_KEYS: Dict[int, str] = field(
        default_factory=lambda: {
            ACTION_ID["terminal"]: "backlog_ms_terminal",
            ACTION_ID["edge_1"]: "backlog_ms_edge_1",
            ACTION_ID["edge_2"]: "backlog_ms_edge_2",
            ACTION_ID["edge_3"]: "backlog_ms_edge_3",
            ACTION_ID["cloud"]: "backlog_ms_cloud",
        }
    )

    def decide(
        self,
        task: pd.Series,
        preferred_action: int,
        context: Dict[str, float] | None = None,
    ) -> tuple[int, Dict[str, float]]:
        preferred_action = int(preferred_action)
        context = context or {}
        estimates = {
            action: estimate_action_trust(
                task,
                action,
                load_hint=self.load_hints.get(action),
                priority_qos=self.priority_qos,
                queue_wait_ms=float(context.get(self.LOAD_KEYS.get(action, ""), 0.0)),
            )
            for action in ACTION_MAP
        }

        original_score = estimates[preferred_action]["dt_predicted_dtt"]
        selected_action = preferred_action
        if original_score < self.trust_threshold:
            high_priority = bool(task.get("high_priority", 0))

            def rank(action: int) -> float:
                score = estimates[action]["dt_predicted_dtt"]
                if high_priority and action in EDGE_ACTION_IDS:
                    score += 0.03
                return score

            selected_action = max(estimates, key=rank)

        return selected_action, {
            "original_predicted_dtt": estimates[preferred_action]["dt_predicted_dtt"],
            "selected_predicted_dtt": estimates[selected_action]["dt_predicted_dtt"],
            "selected_predicted_delay_ms": estimates[selected_action]["dt_predicted_delay_ms"],
        }

    def maybe_recover(self, task: pd.Series, action: int, runtime: Dict[str, float]) -> tuple[bool, float]:
        """Model one lightweight re-orchestration/retry when deadline slack allows it."""

        if int(runtime.get("deadline_met", 0)) == 1:
            return False, 0.0
        if int(action) not in EDGE_ACTION_IDS or not bool(task.get("high_priority", 0)):
            return False, 0.0

        data_in = float(runtime.get("data_in_mb", _task_float(task, "data_size_mb", 0.0)))
        recovery_time_ms = 2.0 + 0.015 * data_in
        recovered_delay = float(runtime.get("exec_delay_ms", 0.0)) + recovery_time_ms
        if recovered_delay <= _task_float(task, "deadline_ms", 0.0):
            runtime["exec_delay_ms"] = recovered_delay
            runtime["exec_energy_kj"] = float(runtime.get("exec_energy_kj", 0.0)) + 0.0015
            runtime["deadline_met"] = 1
            return True, recovery_time_ms
        return False, 0.0

    def observe(
        self,
        task: pd.Series,
        original_action: int,
        final_action: int,
        runtime: Dict[str, float],
        decision: Dict[str, float],
    ) -> Dict[str, float | int | str]:
        cpu = float(runtime.get("exec_cpu_util", self.load_hints[final_action][0]))
        bandwidth = float(runtime.get("bandwidth_utilization", self.load_hints[final_action][1]))
        old_cpu, old_bandwidth = self.load_hints[final_action]
        self.load_hints[final_action] = (
            (1.0 - self.alpha) * old_cpu + self.alpha * cpu,
            (1.0 - self.alpha) * old_bandwidth + self.alpha * bandwidth,
        )

        recovered, recovery_time_ms = self.maybe_recover(task, final_action, runtime)
        return evaluate_runtime_trust(
            task,
            original_action,
            final_action,
            runtime,
            trust_threshold=self.trust_threshold,
            decision=decision,
            recovered=recovered,
            recovery_time_ms=recovery_time_ms,
        )
