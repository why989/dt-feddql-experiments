# DT-FedDQL Experiment Code

This repository contains the experiment scripts and result files used for the
paper:

**Digital-Twin-Assisted Federated Deep Q-Learning for Trustworthy Task
Offloading in Wireless Seismic Sensor Networks**

The released files are intended to support reproducibility of the main figures,
tables, ablation studies, and reviewer-requested diagnostic experiments.

## Repository structure

```text
.
├── deployment_model.py       # Heterogeneous edge tier + M/M/1 capacity model (Eq. 11)
├── algorithm_utils.py        # Task model, simulator, reward, metrics, policy runner
├── digital_twin.py           # Digital-twin trust estimator and orchestrator
├── models/                   # Fuzzy classifier and the numpy federated DQL model
├── data/                     # Released task trace (3000 tasks, derived from SAC records)
├── experiments/              # Dynamic R1-R5 trust experiment (fig05/fig06 source)
├── scripts/                  # Experiment and figure-generation scripts
├── outputs/                  # CSV/summary outputs reported in tables and analysis
├── figures/                  # Publication figures used in the manuscript
├── results/                  # Core DT-FedDQL trace + released federated model weights
├── train_released_model.py   # Trains and releases the federated global weights
├── train_centralized_dqn.py  # Trains the centralized single-DQN baseline weights
└── build_release_results.py  # Rebuilds every results/ trace from the released weights
```

## How to run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run an individual experiment, for example:

```bash
python scripts/20_dependency_ratio_sensitivity.py
```

The generated CSV files are written to `outputs/`.  Scripts are executed from
the repository root; each script adds the repository root to `sys.path` so that
`algorithm_utils`, `digital_twin`, and `models` resolve without installation.

## Reproducing the released trace

The full release chain is:

```bash
python train_released_model.py --rounds 40 --local-epochs 2   # -> results/fed_dql_federated_model_weights.pkl
python train_centralized_dqn.py --rounds 40 --local-epochs 2  # -> results/centralized_dqn_weights.pkl
python build_release_results.py                               # -> results/*.csv (all traces)
```

The second step trains the *learning* baseline used in Table III: a single DQN
with centralized state observation (the live load of every execution
location), trained on the full training workload with the same architecture,
reward, and priority-QoS reservation as the federated agents.  Only federated
aggregation and the digital-twin layer are removed, so the row isolates
centralized from federated training.  Its released metrics are
30.8566 ms, 0.032538 kJ/task, 99.24 % HPC, 0.8438 DTT, 23.37 % violation.

`results/dt_feddql_result.csv` is then regenerated from the released artifacts
alone:

```bash
python scripts/22_release_consistency_check.py
```

The check reloads `data/real_sac_task_set_3000.csv`, rebuilds the 3000-task
trace, loads `results/fed_dql_federated_model_weights.pkl`, and re-runs the
released decision rule. All five headline metrics are reproduced exactly
(30.4196 ms, 0.032341 kJ/task, 100.00 % HPC, 0.8472 DTT, 20.0333 % violation),
and the check additionally verifies that the learned policy is *not* the trivial
"always use the strongest edge node" rule (the per-task delays differ, and
`corr(Q(e2)-Q(e1), rho(e1)-rho(e2)) = 0.33 > 0.1`).

## Main experiments

- Complete baseline comparison: `scripts/19_complete_baseline_comparison.py`
- Baseline policy implementations: `scripts/baseline_policies.py`
- Static leave-one-out ablation: `scripts/10_static_leave_one_out_ablation.py`
- Multi-seed robustness and statistical tests: `scripts/11_statistical_robustness.py`
- Federated communication and non-IID checks: `scripts/12_federated_communication_noniid.py`
- DT prediction error: `scripts/13_dt_predictive_twin.py`
- Dynamic R1-R5 trust scenarios: `experiments/dynamic_trust_experiment.py`
- Node-status snapshot for fig07: `scripts/07_terminal_edge_cloud_node_status.py`
- Missing-key-metric diagnostics: `scripts/15_missing_key_metrics.py`
- Gamma sensitivity and ROC/PR analysis: `scripts/16_gamma_sensitivity_roc.py`
- Node-level action distribution and DAG-width sensitivity: `scripts/18_node_level_action_and_dag_sensitivity.py`
- Dependency-ratio sensitivity: `scripts/20_dependency_ratio_sensitivity.py`
- Reward-weight sensitivity (retrained per setting): `scripts/20_reward_weight_sensitivity.py`
- Release consistency and policy-degeneracy checks: `scripts/22_release_consistency_check.py`

Every script runs from the repository root with no additional inputs.  The
workload-provenance script reports dependency statistics for two scopes, because
splitting the raw records first drops dependency links that cross a split
boundary: `concatenated_splits` gives 146/3000 = 4.87 % dependent tasks, while
the evaluated trace gives 279/3000 = 9.30 %.

## DAG stress tests

`exp18_dag_width_sensitivity.csv` and `exp20_dependency_ratio_sensitivity.csv`
are **not** post-processed from the released trace.  `scripts/dag_utils.py`
rebuilds the dependency graph of the released task set as a layered DAG with a
controlled branch width and a controlled dependent-task ratio, and the released
policy is then re-run on the re-wired tasks.  Because the dependency degrees are
part of the 15-dimensional state vector, re-wiring changes both the scheduling
constraints and the policy input.

Both scans hold the other factor fixed (the width scan keeps the dependent-task
ratio at the released level; the ratio scan keeps the branch width at one
predecessor) and average over five simulator seeds.

## Notes

Each baseline in `outputs/exp19_complete_baseline_comparison.csv` is produced by
a distinct policy implemented in `scripts/baseline_policies.py`; the mapping
from table row to implementation is recorded in
`outputs/exp19_baseline_implementation_sources.csv`. Paired *t*-tests of every
baseline against the released DT-FedDQL trace are stored in
`outputs/exp19_baseline_paired_ttests.csv`.

Two further points that a reviewer should be aware of:

1. **Dependency-release penalty.**  ``run_policy_on_tasks`` adds a fixed
   dependency-release wait to tasks whose predecessors are unfinished.  The
   released simulator recomputed ``deadline_met`` from the penalised delay
   alone, which silently cleared any previously injected task failure.  The
   simulator now preserves that failure.  On the released trace this fix is
   neutral: the five headline metrics are unchanged.

2. **Non-degenerate, load-aware routing.**  The released global Q-network
   routes the 3000 evaluation tasks across the edge tier as
   64.40 % / 32.03 % / 3.57 % for e1 / e2 / e3 (the terminal-local and cloud
   actions are not selected for this workload).  The split is driven by the
   live node load: `corr(Q(e2)-Q(e1), rho(e1)-rho(e2)) = 0.33`, and the learned
   policy beats the trivial "always use the strongest edge node" rule
   (30.42 ms / 100.00 % HPC versus 37.22 ms / 89.31 % HPC).  This is possible
   because `deployment_model.py` gives the three edge nodes heterogeneous
   compute/uplink capability and a finite M/M/1 service capacity (manuscript
   Eq. 11), so concentrating the whole stream on one node saturates it.
   Consequently the IID/non-IID comparison
   (`exp12_noniid_client_heterogeneity.csv`) and the reward-weight sweep
   (`exp20_reward_weight_sensitivity.csv`) now differ across settings.  Client
   heterogeneity is additionally reported in
   `exp12_noniid_training_statistics.csv`.

The repository contains the compact experiment release used for the manuscript,
not the full development workspace. Large virtual environments, temporary files,
and duplicated figures were intentionally excluded.
