# Experiment 11 - Statistical robustness and scalability

Five random seeds are used for the main statistical comparison.

## Mean ± standard deviation
```text
         algorithm runs  avg_delay_ms        avg_energy_kj high_priority_completion_rate    avg_dtt_score trust_violation_rate
         DT-FedDQL    5  21.01 ± 0.16  0.030996 ± 0.000049                 100.00 ± 0.00  0.8530 ± 0.0001         14.79 ± 0.66
  FedDQL-Federated    5  21.01 ± 0.16  0.030996 ± 0.000048                  99.24 ± 0.54  0.8530 ± 0.0001         14.82 ± 0.67
            Greedy    5  43.32 ± 0.04  0.057763 ± 0.000016                  84.27 ± 3.27  0.6995 ± 0.0006         94.53 ± 0.08
              MILP    5  44.86 ± 0.05  0.058009 ± 0.000020                  83.82 ± 3.07  0.6897 ± 0.0007         95.63 ± 0.20
   Centralized DQN    5  45.27 ± 0.06  0.059265 ± 0.000013                   0.46 ± 0.42  0.6767 ± 0.0008         99.39 ± 0.13
 Random Offloading    5  59.53 ± 0.30  0.046254 ± 0.000294                  28.24 ± 3.05  0.5999 ± 0.0017         71.36 ± 0.75
        Local Only    5  44.70 ± 0.07  0.059354 ± 0.000000                   0.00 ± 0.00  0.6791 ± 0.0012         99.45 ± 0.07
```

## Paired significance tests
```text
                     comparison                         metric  target_mean  baseline_mean  paired_t_p_value  wilcoxon_p_value  n_pairs
  DT-FedDQL vs FedDQL-Federated                   avg_delay_ms    21.012520      21.011860          0.041411          0.065600        5
  DT-FedDQL vs FedDQL-Federated                  avg_energy_kj     0.030996       0.030996          0.177808          0.157299        5
  DT-FedDQL vs FedDQL-Federated           trust_violation_rate    14.786660      14.820020          0.034079          0.065600        5
  DT-FedDQL vs FedDQL-Federated  high_priority_completion_rate   100.000000      99.236620          0.034105          0.058782        5
            DT-FedDQL vs Greedy                   avg_delay_ms    21.012520      43.315160          0.000000          0.062500        5
            DT-FedDQL vs Greedy                  avg_energy_kj     0.030996       0.057763          0.000000          0.062500        5
            DT-FedDQL vs Greedy           trust_violation_rate    14.786660      94.533320          0.000000          0.062500        5
            DT-FedDQL vs Greedy  high_priority_completion_rate   100.000000      84.274800          0.000422          0.062500        5
              DT-FedDQL vs MILP                   avg_delay_ms    21.012520      44.855380          0.000000          0.062500        5
              DT-FedDQL vs MILP                  avg_energy_kj     0.030996       0.058009          0.000000          0.062500        5
              DT-FedDQL vs MILP           trust_violation_rate    14.786660      95.633320          0.000000          0.062500        5
              DT-FedDQL vs MILP  high_priority_completion_rate   100.000000      83.816800          0.000297          0.062500        5
   DT-FedDQL vs Centralized DQN                   avg_delay_ms    21.012520      45.268060          0.000000          0.062500        5
   DT-FedDQL vs Centralized DQN                  avg_energy_kj     0.030996       0.059265          0.000000          0.062500        5
   DT-FedDQL vs Centralized DQN           trust_violation_rate    14.786660      99.386660          0.000000          0.062500        5
   DT-FedDQL vs Centralized DQN  high_priority_completion_rate   100.000000       0.458040          0.000000          0.062500        5
 DT-FedDQL vs Random Offloading                   avg_delay_ms    21.012520      59.533160          0.000000          0.062500        5
 DT-FedDQL vs Random Offloading                  avg_energy_kj     0.030996       0.046254          0.000000          0.062500        5
 DT-FedDQL vs Random Offloading           trust_violation_rate    14.786660      71.360000          0.000000          0.062500        5
 DT-FedDQL vs Random Offloading  high_priority_completion_rate   100.000000      28.244280          0.000001          0.062500        5
        DT-FedDQL vs Local Only                   avg_delay_ms    21.012520      44.703260          0.000000          0.062500        5
        DT-FedDQL vs Local Only                  avg_energy_kj     0.030996       0.059354          0.000000          0.062500        5
        DT-FedDQL vs Local Only           trust_violation_rate    14.786660      99.446660          0.000000          0.062500        5
        DT-FedDQL vs Local Only  high_priority_completion_rate   100.000000       0.000000          0.000000          0.062500        5
```

## Pooled high-priority completion counts
```text
         algorithm  hpc_success_count  high_priority_count  pooled_hpc_percent  wilson95_low_percent  wilson95_high_percent  per_seed_min_percent  per_seed_max_percent
         DT-FedDQL                655                  655            100.0000               99.4169               100.0000              100.0000              100.0000
  FedDQL-Federated                650                  655             99.2366               98.2256                99.6735               98.4733              100.0000
            Greedy                552                  655             84.2748               81.2880                86.8619               79.3893               88.5496
              MILP                549                  655             83.8168               80.8005                86.4388               80.1527               87.0229
   Centralized DQN                  3                  655              0.4580                0.1559                 1.3379                0.0000                0.7634
 Random Offloading                185                  655             28.2443               24.9312                31.8110               22.9008               30.5344
        Local Only                  0                  655              0.0000                0.0000                 0.5831                0.0000                0.0000
```

## Dataset split
```text
          dataset                    ratio  raw_records
         Training                      70%         2100
       Validation                      20%          600
             Test                      10%          300
 Paper evaluation  resampled test workload         3000
```

## Scalability scan
```text
 seed          algorithm  task_count  avg_delay_ms  avg_energy_kj  high_priority_completion_rate  hpc_success_count  high_priority_count  avg_dtt_score  trust_violation_rate  edge_cpu_bw_utilization       scan_type  scale_value
   42          DT-FedDQL        1000       20.7436       0.030993                       100.0000                 41                   41         0.8531               16.4000                  75.2848      task_count         1000
   42             Greedy        1000       42.5954       0.057528                        95.1220                 39                   41         0.7088               94.0000                  76.3208      task_count         1000
   42  Random Offloading        1000       57.2170       0.045838                        21.9512                  9                   41         0.6158               69.4000                  75.1338      task_count         1000
   42          DT-FedDQL        2000       21.1008       0.031097                       100.0000                 83                   83         0.8532               14.8000                  75.3853      task_count         2000
   42             Greedy        2000       43.0549       0.057781                        92.7711                 77                   83         0.7035               94.4500                  76.2699      task_count         2000
   42  Random Offloading        2000       58.3878       0.046433                        26.5060                 22                   83         0.6090               70.7000                  75.2564      task_count         2000
   42          DT-FedDQL        3000       20.9470       0.030977                       100.0000                131                  131         0.8532               15.3333                  75.3777      task_count         3000
   42             Greedy        3000       43.3380       0.057750                        84.7328                111                  131         0.6991               94.5000                  76.0236      task_count         3000
   42  Random Offloading        3000       59.5072       0.046526                        29.7710                 39                  131         0.6013               71.6333                  75.2156      task_count         3000
   42          DT-FedDQL        5000       20.9503       0.030998                       100.0000                217                  217         0.8527               14.8000                  75.3770      task_count         5000
   42             Greedy        5000       43.4169       0.057766                        84.3318                183                  217         0.6978               94.7000                  75.9472      task_count         5000
   42  Random Offloading        5000       59.1628       0.046331                        28.5714                 62                  217         0.6032               71.1000                  75.2292      task_count         5000
    5          DT-FedDQL        3000       20.8161       0.029620                       100.0000                131                  131         0.8532               14.1667                  75.3777  terminal_count            5
    5             Greedy        3000       43.3867       0.057639                        84.7328                111                  131         0.6983               94.5667                  76.0236  terminal_count            5
    5  Random Offloading        3000       57.4386       0.045415                        26.7176                 35                  131         0.6131               71.3667                  75.1955  terminal_count            5
   10          DT-FedDQL        3000       20.8100       0.030386                       100.0000                131                  131         0.8532               14.1667                  75.3777  terminal_count           10
   10             Greedy        3000       43.3821       0.057701                        86.2595                113                  131         0.6986               94.6667                  76.0236  terminal_count           10
   10  Random Offloading        3000       58.3732       0.045766                        33.5878                 44                  131         0.6077               70.7000                  75.2413  terminal_count           10
   20          DT-FedDQL        3000       21.3633       0.031415                       100.0000                131                  131         0.8529               14.3000                  75.3777  terminal_count           20
   20             Greedy        3000       43.2861       0.057790                        84.7328                111                  131         0.6999               94.6000                  76.0236  terminal_count           20
   20  Random Offloading        3000       59.2313       0.046600                        27.4809                 36                  131         0.6014               71.4000                  75.2194  terminal_count           20
   50          DT-FedDQL        3000       21.5473       0.033571                       100.0000                131                  131         0.8530               15.3667                  75.3777  terminal_count           50
   50             Greedy        3000       43.3786       0.057983                        80.9160                106                  131         0.6991               94.7000                  76.0236  terminal_count           50
   50  Random Offloading        3000       59.1586       0.048170                        29.0076                 38                  131         0.6034               72.0333                  75.2180  terminal_count           50
```