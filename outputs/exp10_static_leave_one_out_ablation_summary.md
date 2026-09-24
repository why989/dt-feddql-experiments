# Experiment 10 - Static leave-one-out ablation

This experiment disables one module at a time under the same static task set.
It complements the incremental module-composition table and avoids attributing all improvement to one combined final step.
Digital-twin orchestration is not included in this static policy table; its contribution is evaluated separately in the dynamic DT-vs-No-DT experiment.

```text
            variant  avg_delay_ms  avg_energy_kj_per_task  hpc_rate_percent hpc_count  trusted_hpc_score_percent  avg_dtt_score  trust_violation_rate_percent
 Full static policy       38.9920                0.029712           77.0992   101/131                    15.7026         0.7706                       79.6333
    w/o Fuzzy prior       55.8027                0.056462           50.3817    66/131                     2.2672         0.5502                       95.5000
 w/o Resource state       41.7073                0.032706            0.0000     0/131                     0.0000         0.7273                       92.5000
 w/o Reward shaping       40.9300                0.030494            0.0000     0/131                     0.0000         0.7388                       90.7000
  w/o FedDQL policy       56.7773                0.057198          100.0000   131/131                     4.4333         0.5390                       95.5667
```