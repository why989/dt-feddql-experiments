# Experiment 10 - Static leave-one-out ablation

This experiment disables one module at a time under the same static task set.
It complements the incremental module-composition table and avoids attributing all improvement to one combined final step.
Digital-twin orchestration is not included in this static policy table; its contribution is evaluated separately in the dynamic DT-vs-No-DT experiment.

```text
            variant  avg_delay_ms  avg_energy_kj_per_task  hpc_rate_percent hpc_count  trusted_hpc_score_percent  avg_dtt_score  trust_violation_rate_percent
 Full static policy       20.9464                0.030977           99.2366   130/131                    83.9872         0.8532                       15.3667
    w/o Fuzzy prior       42.4261                0.056697           83.2061   109/131                     7.2112         0.7073                       91.3333
 w/o Resource state       21.6587                0.032706           91.6031   120/131                    77.0993         0.8507                       15.8333
 w/o Reward shaping       21.5526                0.031805           87.7863   115/131                    73.8575         0.8511                       15.8667
  w/o FedDQL policy       43.1172                0.057207          100.0000   131/131                     4.9667         0.6980                       95.0333
```