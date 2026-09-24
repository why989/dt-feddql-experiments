# Experiment 11 - Statistical robustness and scalability

Five random seeds are used for the main statistical comparison.

## Mean ± standard deviation
```text
         algorithm runs  avg_delay_ms        avg_energy_kj high_priority_completion_rate    avg_dtt_score trust_violation_rate
         DT-FedDQL    5  30.44 ± 0.20  0.032327 ± 0.000026                 100.00 ± 0.00  0.8460 ± 0.0016         24.09 ± 1.46
  FedDQL-Federated    5  30.89 ± 0.26  0.032860 ± 0.000023                  99.69 ± 0.42  0.8402 ± 0.0015         23.33 ± 1.73
            Greedy    5  49.74 ± 0.10  0.048202 ± 0.000019                  21.07 ± 2.13  0.6319 ± 0.0011         79.64 ± 0.46
              MILP    5  58.23 ± 0.08  0.057951 ± 0.000021                  50.08 ± 3.35  0.5316 ± 0.0010         97.78 ± 0.14
   Centralized DQN    5  30.81 ± 0.23  0.032535 ± 0.000003                  99.85 ± 0.34  0.8429 ± 0.0015         23.80 ± 1.74
 Random Offloading    5  52.73 ± 0.70  0.045490 ± 0.000195                  14.81 ± 2.78  0.6459 ± 0.0049         65.59 ± 0.90
        Local Only    5  58.93 ± 0.13  0.059354 ± 0.000000                   0.00 ± 0.00  0.5142 ± 0.0011         99.99 ± 0.01
```

## Paired significance tests
```text
                     comparison                         metric  target_mean  baseline_mean  paired_t_p_value  wilcoxon_p_value  n_pairs
  DT-FedDQL vs FedDQL-Federated                   avg_delay_ms    30.439980      30.886180          0.000490          0.062500        5
  DT-FedDQL vs FedDQL-Federated                  avg_energy_kj     0.032327       0.032860          0.000005          0.062500        5
  DT-FedDQL vs FedDQL-Federated           trust_violation_rate    24.086640      23.333320          0.133648          0.187500        5
  DT-FedDQL vs FedDQL-Federated  high_priority_completion_rate   100.000000      99.694640          0.177808          0.157299        5
            DT-FedDQL vs Greedy                   avg_delay_ms    30.439980      49.738000          0.000000          0.062500        5
            DT-FedDQL vs Greedy                  avg_energy_kj     0.032327       0.048202          0.000000          0.062500        5
            DT-FedDQL vs Greedy           trust_violation_rate    24.086640      79.640000          0.000000          0.062500        5
            DT-FedDQL vs Greedy  high_priority_completion_rate   100.000000      21.068720          0.000000          0.062500        5
              DT-FedDQL vs MILP                   avg_delay_ms    30.439980      58.225580          0.000000          0.062500        5
              DT-FedDQL vs MILP                  avg_energy_kj     0.032327       0.057951          0.000000          0.062500        5
              DT-FedDQL vs MILP           trust_violation_rate    24.086640      97.780000          0.000000          0.062500        5
              DT-FedDQL vs MILP  high_priority_completion_rate   100.000000      50.076340          0.000005          0.062500        5
   DT-FedDQL vs Centralized DQN                   avg_delay_ms    30.439980      30.814980          0.000166          0.062500        5
   DT-FedDQL vs Centralized DQN                  avg_energy_kj     0.032327       0.032535          0.000065          0.062500        5
   DT-FedDQL vs Centralized DQN           trust_violation_rate    24.086640      23.800000          0.428717          0.715001        5
   DT-FedDQL vs Centralized DQN  high_priority_completion_rate   100.000000      99.847320          0.373901          0.317311        5
 DT-FedDQL vs Random Offloading                   avg_delay_ms    30.439980      52.730340          0.000000          0.062500        5
 DT-FedDQL vs Random Offloading                  avg_energy_kj     0.032327       0.045490          0.000000          0.062500        5
 DT-FedDQL vs Random Offloading           trust_violation_rate    24.086640      65.593340          0.000001          0.062500        5
 DT-FedDQL vs Random Offloading  high_priority_completion_rate   100.000000      14.809160          0.000000          0.062500        5
        DT-FedDQL vs Local Only                   avg_delay_ms    30.439980      58.934580          0.000000          0.062500        5
        DT-FedDQL vs Local Only                  avg_energy_kj     0.032327       0.059354          0.000000          0.062500        5
        DT-FedDQL vs Local Only           trust_violation_rate    24.086640      99.993340          0.000000          0.062500        5
        DT-FedDQL vs Local Only  high_priority_completion_rate   100.000000       0.000000          0.000000          0.062500        5
```

## Pooled high-priority completion counts
```text
         algorithm  hpc_success_count  high_priority_count  pooled_hpc_percent  wilson95_low_percent  wilson95_high_percent  per_seed_min_percent  per_seed_max_percent
         DT-FedDQL                655                  655            100.0000               99.4169               100.0000              100.0000              100.0000
  FedDQL-Federated                653                  655             99.6947               98.8936                99.9162               99.2366              100.0000
            Greedy                138                  655             21.0687               18.1189                24.3558               19.0840               23.6641
              MILP                328                  655             50.0763               46.2580                53.8938               47.3282               55.7252
   Centralized DQN                654                  655             99.8473               99.1403                99.9730               99.2366              100.0000
 Random Offloading                 97                  655             14.8092               12.2944                17.7343               11.4504               17.5573
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
   42          DT-FedDQL        1000       29.7579       0.032397                       100.0000                 41                   41         0.8490               21.9000                  75.2848      task_count         1000
   42             Greedy        1000       47.0385       0.047787                        41.4634                 17                   41         0.6537               71.4000                  75.2189      task_count         1000
   42  Random Offloading        1000       52.0324       0.045290                        12.1951                  5                   41         0.6496               64.4000                  75.1357      task_count         1000
   42          DT-FedDQL        2000       30.3768       0.032476                       100.0000                 83                   83         0.8476               22.7000                  75.3853      task_count         2000
   42             Greedy        2000       47.7431       0.048028                        43.3735                 36                   83         0.6493               71.4500                  75.2961      task_count         2000
   42  Random Offloading        2000       52.7718       0.045622                        13.2530                 11                   83         0.6449               65.5000                  75.2439      task_count         2000
   42          DT-FedDQL        3000       30.4196       0.032341                       100.0000                131                  131         0.8472               20.0333                  75.3777      task_count         3000
   42             Greedy        3000       49.7110       0.048182                        20.6107                 27                  131         0.6314               79.6333                  75.2770      task_count         3000
   42  Random Offloading        3000       53.6615       0.045712                        12.2137                 16                  131         0.6384               67.0000                  75.2473      task_count         3000
   42          DT-FedDQL        5000       30.5085       0.032363                       100.0000                217                  217         0.8465               22.6800                  75.3770      task_count         5000
   42             Greedy        5000       49.0446       0.048076                        29.4931                 64                  217         0.6369               75.8600                  75.2619      task_count         5000
   42  Random Offloading        5000       53.0179       0.045524                        13.8249                 30                  217         0.6435               66.2600                  75.2450      task_count         5000
    5          DT-FedDQL        3000       30.4210       0.032089                       100.0000                131                  131         0.8469               22.8333                  75.3777  terminal_count            5
    5             Greedy        3000       49.6473       0.048076                        23.6641                 31                  131         0.6329               79.3000                  75.2770  terminal_count            5
    5  Random Offloading        3000       51.7864       0.044633                        16.0305                 21                  131         0.6531               63.4667                  75.2105  terminal_count            5
   10          DT-FedDQL        3000       30.4312       0.032858                       100.0000                131                  131         0.8471               21.7667                  75.3777  terminal_count           10
   10             Greedy        3000       49.5826       0.048411                        17.5573                 23                  131         0.6328               78.9333                  75.2770  terminal_count           10
   10  Random Offloading        3000       51.9892       0.045124                        23.6641                 31                  131         0.6534               63.8667                  75.2371  terminal_count           10
   20          DT-FedDQL        3000       31.1237       0.033901                       100.0000                131                  131         0.8429               26.8000                  75.3777  terminal_count           20
   20             Greedy        3000       49.8737       0.048904                        21.3740                 28                  131         0.6307               79.5333                  75.2770  terminal_count           20
   20  Random Offloading        3000       52.8371       0.046218                        12.9771                 17                  131         0.6463               65.9667                  75.2381  terminal_count           20
   50          DT-FedDQL        3000       31.4855       0.036074                       100.0000                131                  131         0.8408               29.1667                  75.3777  terminal_count           50
   50             Greedy        3000       50.0816       0.049847                        12.2137                 16                  131         0.6286               80.8333                  75.2770  terminal_count           50
   50  Random Offloading        3000       53.6106       0.047793                        12.2137                 16                  131         0.6427               66.5000                  75.2247  terminal_count           50
```