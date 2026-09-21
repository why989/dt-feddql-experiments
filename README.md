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
├── scripts/      # Experiment and figure-generation scripts used in the paper
├── outputs/      # CSV/summary outputs reported in tables and analysis
├── figures/      # Publication figures used in the manuscript
└── results/      # Core DT-FedDQL trace used by the diagnostic scripts
```

## Main experiments

- Complete baseline comparison: `scripts/19_complete_baseline_comparison.py`
- Static leave-one-out ablation: `scripts/10_static_leave_one_out_ablation.py`
- Multi-seed robustness and statistical tests: `scripts/11_statistical_robustness.py`
- Federated communication and non-IID checks: `scripts/12_federated_communication_noniid.py`
- DT prediction error: `scripts/13_dt_predictive_twin.py`
- Missing-key-metric diagnostics: `scripts/15_missing_key_metrics.py`
- Gamma sensitivity and ROC/PR analysis: `scripts/16_gamma_sensitivity_roc.py`
- Node-level action distribution and DAG sensitivity: `scripts/18_node_level_action_and_dag_sensitivity.py`
- Dependency-ratio sensitivity: `scripts/20_dependency_ratio_sensitivity.py`
- Reward-weight sensitivity: `scripts/20_reward_weight_sensitivity.py`

## How to run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run an individual experiment, for example:

```bash
python scripts/20_dependency_ratio_sensitivity.py
```

The generated CSV files are written to `outputs/`.

## Notes

The repository contains the compact experiment release used for the manuscript,
not the full development workspace. Large virtual environments, temporary files,
and duplicated figures were intentionally excluded.

