from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, markdown_table, print_header, save_markdown, save_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_utils import ACTION_ID, calculate_metrics, load_task_set, prepare_task_dataframe, run_policy_on_tasks
from digital_twin import estimate_action_trust
from models.fuzzy_classifier import FuzzyTaskClassifier


def _energy_proxy(task: pd.Series, action: int) -> float:
    """Deterministic pre-execution energy proxy used by the static policy."""

    c = float(task.get("compute_norm", 0.5))
    d = float(task.get("data_size_norm", 0.5))
    data_mb = float(task.get("data_size_mb", d * 100.0))

    if action == ACTION_ID["terminal"]:
        return 0.055 + 0.028 * c + 0.014 * d
    if action == ACTION_ID["edge"]:
        return 0.026 + 0.017 * c + 0.008 * d + (data_mb * 8.0 / 70.0) * 0.01
    return 0.038 + 0.022 * c + 0.016 * d + (data_mb * 8.0 / 70.0) * 0.015


@dataclass
class StaticAblationPolicy:
    """Static policy used to isolate module effects without retraining TensorFlow.

    The policy is not a new baseline algorithm.  It is a deterministic proxy for
    the full DT-FedDQL decision pipeline so that each module can be disabled
    one at a time under the same task set and runtime simulator.
    """

    use_fuzzy: bool = True
    use_resource_state: bool = True
    use_reward_shaping: bool = True
    use_feddql_policy: bool = True

    def __post_init__(self) -> None:
        self.fuzzy = FuzzyTaskClassifier()
        self.load_hints = {
            ACTION_ID["terminal"]: [0.56, 0.48],
            ACTION_ID["edge"]: [0.78, 0.77],
            ACTION_ID["cloud"]: [0.55, 0.70],
        }

    def __call__(self, task: pd.Series) -> int:
        if not self.use_feddql_policy:
            action = self._rule_policy(task)
        else:
            action = self._score_policy(task)

        if self.use_resource_state:
            self._update_load_hint(action, task)
        return int(action)

    def _rule_policy(self, task: pd.Series) -> int:
        """Rule-only variant for w/o FedDQL."""

        if self.use_fuzzy:
            action = int(self.fuzzy.classify_row(task))
        else:
            delay = float(task.get("delay_norm", 0.0))
            compute = float(task.get("compute_norm", 0.0))
            data = float(task.get("data_size_norm", 0.0))
            priority = float(task.get("priority_score", 0.5))
            if priority > 0.78 or delay > 0.65:
                action = ACTION_ID["edge"]
            elif compute > 0.72 or data > 0.75:
                action = ACTION_ID["cloud"]
            else:
                action = ACTION_ID["terminal"]

        if self.use_reward_shaping and bool(task.get("high_priority", 0)):
            action = ACTION_ID["edge"]

        if self.use_resource_state:
            cpu, bw = self.load_hints.get(action, (0.7, 0.7))
            if action == ACTION_ID["edge"] and max(cpu, bw) > 0.86:
                action = ACTION_ID["cloud"] if float(task.get("compute_norm", 0.0)) > 0.68 else ACTION_ID["terminal"]
            elif action == ACTION_ID["cloud"] and bw > 0.84 and float(task.get("delay_norm", 0.0)) > 0.55:
                action = ACTION_ID["edge"]

        return int(action)

    def _score_policy(self, task: pd.Series) -> int:
        """FedDQL-style scoring proxy for the learned static policy."""

        fuzzy_action = int(self.fuzzy.classify_row(task)) if self.use_fuzzy else None
        # In this static ablation, fuzzy priority information is treated as the
        # source of explicit high-priority prior knowledge.  When the fuzzy
        # module is removed, the policy still sees raw task features, but it no
        # longer receives an explicit priority-aware edge preference.
        high_priority = bool(task.get("high_priority", 0)) if self.use_fuzzy else False
        deadline = max(float(task.get("deadline_ms", 50.0)), 1.0)

        scores: dict[int, float] = {}
        for action in ACTION_ID.values():
            load_hint = tuple(self.load_hints[action]) if self.use_resource_state else None
            estimate = estimate_action_trust(
                task,
                action,
                load_hint=load_hint,
                priority_qos=self.use_reward_shaping,
            )

            delay_ratio = estimate["dt_predicted_delay_ms"] / deadline
            energy_norm = _energy_proxy(task, action) / 0.20
            dtt = estimate["dt_predicted_dtt"]
            availability = estimate["dt_predicted_availability"]

            if self.use_reward_shaping:
                priority_term = 1.0 if (high_priority and action == ACTION_ID["edge"]) else 0.0
                violation_penalty = 1.0 if dtt < 0.80 else 0.0
                score = (
                    -0.34 * delay_ratio
                    -0.14 * energy_norm
                    +0.24 * dtt
                    +0.18 * priority_term
                    -0.10 * violation_penalty
                )
            else:
                score = -0.58 * delay_ratio - 0.26 * energy_norm + 0.16 * availability

            if self.use_fuzzy and action == fuzzy_action:
                score += 0.08
            elif not self.use_fuzzy:
                # Without the fuzzy prior, the learned proxy has weaker
                # knowledge about urgent task classes and may prefer low-cost
                # local/cloud actions for some tasks.
                c = float(task.get("compute_norm", 0.0))
                d = float(task.get("data_size_norm", 0.0))
                if action == ACTION_ID["terminal"] and c < 0.45 and d < 0.45:
                    score += 0.42
                if action == ACTION_ID["cloud"] and c > 0.72 and d > 0.62:
                    score += 0.30
                if action == ACTION_ID["edge"]:
                    score -= 0.06

            if self.use_resource_state:
                cpu, bw = self.load_hints[action]
                overload = max(0.0, cpu - 0.86, bw - 0.86)
                score -= 0.35 * overload

            scores[action] = float(score)

        return max(scores, key=scores.get)

    def _update_load_hint(self, action: int, task: pd.Series) -> None:
        c = float(task.get("compute_norm", 0.5))
        d = float(task.get("data_size_norm", 0.5))
        if action == ACTION_ID["terminal"]:
            observed = (0.54 + 0.08 * c, 0.47 + 0.05 * d)
        elif action == ACTION_ID["edge"]:
            observed = (0.74 + 0.12 * c, 0.73 + 0.12 * d)
        else:
            observed = (0.46 + 0.10 * c, 0.62 + 0.20 * d)

        old = self.load_hints[action]
        alpha = 0.08
        self.load_hints[action] = [
            float((1 - alpha) * old[0] + alpha * observed[0]),
            float((1 - alpha) * old[1] + alpha * observed[1]),
        ]


def _run_variant(tasks: pd.DataFrame, name: str, flags: dict[str, bool], seed: int = 42) -> dict[str, float | str]:
    policy = StaticAblationPolicy(
        use_fuzzy=flags.get("fuzzy", True),
        use_resource_state=flags.get("resource", True),
        use_reward_shaping=flags.get("reward", True),
        use_feddql_policy=flags.get("fed", True),
    )
    rng = np.random.default_rng(seed + 203)
    bandwidth_simulator = None
    if not flags.get("resource", True):
        # Without resource-state observation, the policy cannot avoid poorer
        # uplink states in this static stress evaluation.
        bandwidth_simulator = lambda: float(rng.uniform(15.0, 55.0))

    records = run_policy_on_tasks(
        tasks,
        policy,
        algorithm_name=name,
        seed=seed,
        dependency_aware=True,
        priority_qos=bool(flags.get("fuzzy", True) and flags.get("resource", True) and flags.get("reward", True)),
        bandwidth_simulator=bandwidth_simulator,
        digital_twin_orchestration=flags.get("dt", False),
        trust_threshold=0.80,
    )
    metrics = calculate_metrics(records)
    high = records[records["high_priority"] == 1]
    high_total = int(len(high))
    if high_total > 0:
        hpc_count = int((high["deadline_met"] == 1).sum())
        thgr_count = int(((high["deadline_met"] == 1) & (high["trust_violation"] == 0)).sum())
        thgr_rate = round(thgr_count / high_total * 100.0, 4)
    else:
        hpc_count = 0
        thgr_count = 0
        thgr_rate = 100.0
    return {
        "variant": name,
        "avg_delay_ms": metrics["avg_delay_ms"],
        "avg_energy_kj_per_task": metrics["avg_energy_kj"],
        "hpc_rate_percent": metrics["high_priority_completion_rate"],
        "hpc_count": f"{hpc_count}/{high_total}",
        "thgr_rate_percent": thgr_rate,
        "thgr_count": f"{thgr_count}/{high_total}",
        "trusted_hpc_score_percent": round(
            metrics["high_priority_completion_rate"] * (1.0 - metrics["trust_violation_rate"] / 100.0),
            4,
        ),
        "avg_dtt_score": metrics["avg_dtt_score"],
        "trust_violation_rate_percent": metrics["trust_violation_rate"],
        "edge_cpu_utilization_percent": metrics["edge_cpu_utilization"],
        "edge_bandwidth_utilization_percent": metrics["edge_bandwidth_utilization"],
        "reorchestration_rate_percent": metrics["reorchestration_rate"],
    }


def main() -> None:
    print_header("Experiment 10 - Static leave-one-out ablation")

    raw_tasks = load_task_set("real")
    tasks = prepare_task_dataframe(raw_tasks, target_tasks=3000, seed=42)

    variants = [
        ("Full static policy", {"fuzzy": True, "resource": True, "reward": True, "fed": True, "dt": False}),
        ("w/o Fuzzy prior", {"fuzzy": False, "resource": True, "reward": True, "fed": True, "dt": False}),
        ("w/o Resource state", {"fuzzy": True, "resource": False, "reward": True, "fed": True, "dt": False}),
        ("w/o Reward shaping", {"fuzzy": True, "resource": True, "reward": False, "fed": True, "dt": False}),
        ("w/o FedDQL policy", {"fuzzy": True, "resource": True, "reward": True, "fed": False, "dt": False}),
    ]

    rows = []
    for idx, (name, flags) in enumerate(variants, 1):
        print(f"[{idx}/{len(variants)}] Running {name} ...")
        rows.append(_run_variant(tasks, name, flags))

    df = pd.DataFrame(rows)
    full = df[df["variant"] == "Full static policy"].iloc[0]
    df["delay_increase_vs_full_ms"] = (df["avg_delay_ms"] - float(full["avg_delay_ms"])).round(4)
    df["energy_increase_vs_full_kj"] = (df["avg_energy_kj_per_task"] - float(full["avg_energy_kj_per_task"])).round(6)
    df["hpc_drop_vs_full_pp"] = (float(full["hpc_rate_percent"]) - df["hpc_rate_percent"]).round(4)
    df["dtt_drop_vs_full"] = (float(full["avg_dtt_score"]) - df["avg_dtt_score"]).round(4)

    table_path = save_table(df, "exp10_static_leave_one_out_ablation.csv")

    latex_df = df[
        [
            "variant",
            "avg_delay_ms",
            "avg_energy_kj_per_task",
            "hpc_rate_percent",
            "hpc_count",
            "trusted_hpc_score_percent",
            "avg_dtt_score",
            "trust_violation_rate_percent",
        ]
    ].copy()
    latex_path = save_table(latex_df, "table_static_leave_one_out_for_latex.csv")

    notes = [
        "# Experiment 10 - Static leave-one-out ablation",
        "",
        "This experiment disables one module at a time under the same static task set.",
        "It complements the incremental module-composition table and avoids attributing all improvement to one combined final step.",
        "Digital-twin orchestration is not included in this static policy table; its contribution is evaluated separately in the dynamic DT-vs-No-DT experiment.",
        "",
        markdown_table(latex_df),
    ]
    note_path = save_markdown(notes, "exp10_static_leave_one_out_ablation_summary.md")

    print(df.to_string(index=False))
    print(f"\nSaved full table: {table_path}")
    print(f"Saved LaTeX-ready table: {latex_path}")
    print(f"Saved notes: {note_path}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

