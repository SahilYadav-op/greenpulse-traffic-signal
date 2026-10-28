# Data of the GreenPulse project

**There is no external data set.** The junction is simulated, so every number comes from traffic streams that the program
generates from fixed random seeds, and from the results measured on them. Everything is stored here as plain CSV (open it in Excel)
and in one compressed NumPy file, so that every table and figure of the report can be traced back to its data.
`experiments/18_export_data.py` rewrites all of these files from `results/*.json`.

## The traffic that goes into the simulator

| File | Content |
|---|---|
| `demand_scenarios.csv` | the six demand patterns: cars per hour on the North, East, South and West approach for each time segment, and whether the pattern was used in tuning |
| `arrival_streams.npz` | the 140 random arrival streams that were generated: 120 test streams (6 patterns x seeds 1001-1020) and 20 training streams (4 patterns x seeds 101-105). Each stream is a `uint8` array of shape (4, 3600): rows are N, E, S, W, columns are seconds, a 1 means a car arrives in that second |
| `arrival_streams_summary.csv` | the number of cars that arrive on each approach in every stream |

```python
import numpy as np
streams = np.load("arrival_streams.npz")
a = streams["test_balanced_seed1001"]       # shape (4, 3600)
print(a.sum(axis=1))                        # cars per approach in that hour
```

Every controller is run on exactly the same stream for a given pattern and seed (common random numbers).

## Results measured on that traffic

| File | Content |
|---|---|
| `test_runs_per_controller.csv` | all 1,320 held-out runs: 11 controllers x 6 patterns x 20 seeds, with mean delay, 95th-percentile delay, stops per car, throughput, longest queue, unserved cars, delay per approach, phase changes, and how greens ended |
| `paired_tests_wilcoxon.csv` | the paired Wilcoxon signed-rank comparisons: change in percent, p-value and number of wins out of 120 (or 20) pairs |
| `stress_tests.csv` | the robustness experiment: noisy sensors, rain, 20% more demand, and all three together |
| `minimum_green_sensitivity.csv` | what happens when the minimum green of the tuned controllers is forced to 4, 6, 8, 10 and 12 s |
| `controller_parameters.csv` | the 20 genes of every tuned controller, next to the hand-tuned one |
| `pareto_front_main.csv`, `pareto_fronts_practical_test.csv` | the NSGA-II fronts: 80 members each, with training and test objectives (main front) or test objectives and minimum green (practical fronts) |
| `tuning_runs_nsga2.csv`, `tuning_runs_ga_random.csv`, `independent_runs_test.csv` | every tuning run: final hypervolume and knee point, best cost of the GA and of random search, and the test delay of each run |
| `validation_discharge.csv`, `validation_fundamental_diagram.csv`, `validation_webster.csv` | the three checks of the simulator: saturation flow, the flow-density curve, and delay against Webster's formula |

Delays are in seconds per car, "stops" are stops per car, and objectives of the two *robust* tuning variants during tuning are ratios to the
actuated controller (see Appendix C of the report).
