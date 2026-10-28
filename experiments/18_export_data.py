"""Step 18: write every data table behind the report and the slides as plain CSV files in data/.

The junction is simulated, so there is no external data set: the data of this project are the traffic streams
that were generated from fixed seeds and the results measured on them.

Usage:  python -I experiments/18_export_data.py
"""
import csv
import json
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from greenpulse import fuzzy as FZ
from greenpulse import scenarios as SC

RES = ROOT / "results"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
load = lambda name: json.loads((RES / name).read_text())
EV, TU, TR = load("evaluation.json"), load("tuning.json"), load("tuning_robust.json")
TP, TPR, PF, VA = load("tuning_practical.json"), load("tuning_practical_robust.json"), load("practical_front.json"), load("validation.json")
CT = load("controllers.json")
for f in ("controllers_robust.json", "controllers_practical.json", "controllers_practical_robust.json"):
    CT.update(load(f))


def write(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name:34s} {len(rows):6d} rows")


r4 = lambda x: round(float(x), 4)

# 1 --------------------------------------------------------------------------------------------- demand scenarios
rows = []
for s, segs in SC.SCENARIOS.items():
    for a, b, flows in segs:
        rows.append([s, SC.LABELS[s], "tuning" if s in SC.TRAIN else "never used in tuning", a, b, *flows])
rows.append(["random_mix", SC.LABELS["random_mix"], "never used in tuning", 0, 3600, "100-340", "100-340", "100-340", "100-340"])
write("demand_scenarios.csv", ["scenario", "label", "role", "start_s", "end_s", "N_cars_per_h", "E_cars_per_h", "S_cars_per_h", "W_cars_per_h"], rows)

# 2 --------------------------------------------------------------------------------------------- every test run
metrics = list(EV["main"]["Vehicle-actuated"]["light"].keys())
rows = []
for kind, per_sc in EV["main"].items():
    for sc, m in per_sc.items():
        for i, seed in enumerate(EV["seeds"]):
            rows.append([kind, sc, seed] + [r4(m[k]["values"][i]) for k in metrics])
write("test_runs_per_controller.csv", ["controller", "scenario", "seed"] + metrics, rows)

# 3 --------------------------------------------------------------------------------------------- paired tests
rows = [[r.get("focus", "Fuzzy + NSGA-II"), r["ref"], r["scenario"], r["metric"], r4(r["nsga_mean"]), r4(r["ref_mean"]), round(r["change_pct"], 2), f"{r['p_value']:.3e}", r["wins"], r["n"]]
        for r in EV["paired"]]
write("paired_tests_wilcoxon.csv", ["controller", "against", "scenario", "metric", "controller_mean", "against_mean", "change_percent", "p_value", "wins", "pairs"], rows)

# 4 --------------------------------------------------------------------------------------------- stress tests
rows = [[cond, kind, r4(v["mean_delay"]), r4(v["p95_delay"]), r4(v["unserved"]), r4(v["cap_share"])] for cond, d in EV["robustness"].items() for kind, v in d.items()]
write("stress_tests.csv", ["condition", "controller", "mean_delay_s", "p95_delay_s", "unserved_cars", "share_of_greens_ended_by_cap"], rows)

# 5 --------------------------------------------------------------------------------------------- minimum-green sensitivity
rows = [[name, mg, r4(v["mean_delay"]), r4(v["switches"])] for name, d in EV["min_green_sensitivity"].items() if "4" in d for mg, v in d.items()]
write("minimum_green_sensitivity.csv", ["controller", "forced_minimum_green_s", "mean_delay_s", "phase_changes_per_hour"], rows)

# 6 --------------------------------------------------------------------------------------------- tuned parameters
cols = [("hand_tuned", FZ.HAND_TUNED.tolist()), ("nsga2_clean", CT["nsga2_knee"]), ("nsga2_robust", CT["nsga2_robust_knee"]), ("nsga2_practical", CT["nsga2_practical_knee"]),
        ("nsga2_practical_robust", CT["nsga2_practical_robust_knee"]), ("single_objective_ga", CT["ga_scalar"]), ("random_search", CT["random_search"])]
rows = [[i + 1, FZ.GENE_NAMES[i]] + [r4(c[1][i]) for c in cols] for i in range(FZ.N_GENES)]
write("controller_parameters.csv", ["gene", "name"] + [c[0] for c in cols], rows)

# 7 --------------------------------------------------------------------------------------------- Pareto fronts
F = EV["front"]
rows = [[i, *[r4(x) for x in F["train"][i]], *[r4(x) for x in F["test"][i]], int(i == F["knee_index"])] for i in range(F["size"])]
write("pareto_front_main.csv", ["member", "train_mean_delay_s", "train_p95_delay_s", "train_stops", "test_mean_delay_s", "test_p95_delay_s", "test_stops", "is_knee_point"], rows)
rows = []
for tag in ("practical", "practical_robust"):
    for i, (obj, mg) in enumerate(zip(PF[tag]["test"], PF[tag]["min_green"])):
        rows.append([tag, i, *[r4(x) for x in obj], r4(mg), int(i == PF[tag]["knee_index"])])
write("pareto_fronts_practical_test.csv", ["front", "member", "test_mean_delay_s", "test_p95_delay_s", "test_stops", "minimum_green_s", "is_knee_point"], rows)

# 8 --------------------------------------------------------------------------------------------- tuning runs
rows = []
for variant, T_, key in (("NSGA-II clean", TU, "nsga2"), ("NSGA-II robust", TR, "runs"), ("NSGA-II practical", TP, "runs"), ("NSGA-II practical robust", TPR, "runs")):
    for x in T_[key]:
        rows.append([variant, x["run"], r4(x["final_hv"]), *[r4(v) for v in x["knee_objectives"]]])
write("tuning_runs_nsga2.csv", ["variant", "run", "final_hypervolume", "knee_objective_1", "knee_objective_2", "knee_objective_3"], rows)
rows = [[fam, i, r4(x["best_J"])] for fam, key in (("single-objective GA", "ga"), ("random search", "random")) for i, x in enumerate(TU[key])]
write("tuning_runs_ga_random.csv", ["method", "run", "best_cost_J"], rows)
rows = [[fam, x["run"], r4(x["mean_delay"]), r4(x["p95_delay"]), r4(x["mean_stops"])] for fam, runs in EV["independent_runs"].items() for x in runs]
write("independent_runs_test.csv", ["family", "run", "mean_delay_s", "p95_delay_s", "stops"], rows)

# 9 --------------------------------------------------------------------------------------------- simulator validation
rows = [[p, r4(v["saturation_flow_cars_per_h"]), r4(v["headway_s"]), r4(v["first_car_departure_s"])] for p, v in VA["calibration"].items()]
write("validation_discharge.csv", ["dawdling_probability", "saturation_flow_cars_per_h", "headway_s", "first_car_crosses_s"], rows)
fd = VA["fundamental_diagram"]
rows = [[p_, r4(dc), r4(dk), r4(fl)] for p_, flows in fd["flow_veh_per_h"].items() for dc, dk, fl in zip(fd["density_cars_per_cell"], fd["density_veh_per_km"], flows)]
write("validation_fundamental_diagram.csv", ["dawdling_probability", "density_cars_per_cell", "density_cars_per_km", "flow_cars_per_h"], rows)
rows = [[w["flow_per_approach"], w["cycle_s"], r4(w["green_s"]), r4(w["Y"]), r4(w["sim_delay"]), r4(w["sim_delay_sd"]), r4(w["webster_delay"]), r4(w["ratio"])] for w in VA["webster_validation"]]
write("validation_webster.csv", ["cars_per_h_per_approach", "cycle_s", "green_s", "flow_ratio_Y", "simulated_delay_s", "simulated_delay_sd", "webster_delay_s", "ratio"], rows)

# 10 -------------------------------------------------------------------------------------------- the traffic streams themselves
arrays, meta = {}, []
for split, scs, seeds in (("test", SC.TRAIN + SC.UNSEEN, EV["seeds"]), ("train", SC.TRAIN, TU["train_seeds"])):
    for sc in scs:
        for seed in seeds:
            a = np.asarray(SC.make_arrivals(sc, seed))
            arrays[f"{split}_{sc}_seed{seed}"] = a
            meta.append([split, sc, seed, *[int(v) for v in a.sum(axis=1)]])
np.savez_compressed(OUT / "arrival_streams.npz", **arrays)
first = next(iter(arrays.values()))
print(f"arrival_streams.npz                {len(arrays):6d} streams, each an array of shape {first.shape} ({first.dtype}): rows N, E, S, W; columns = seconds")
write("arrival_streams_summary.csv", ["split", "scenario", "seed", "cars_N", "cars_E", "cars_S", "cars_W"], meta)
