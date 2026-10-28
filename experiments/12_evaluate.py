"""Step 12: held-out evaluation of every controller (test seeds 1001..1020, never used in tuning).

Usage:  python -I experiments/12_evaluate.py
"""
import json
import site
import sys
import time
from pathlib import Path

sys.path.append(site.getusersitepackages())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy import stats

from greenpulse import baselines as B
from greenpulse import fuzzy as FZ
from greenpulse import scenarios as SC
from greenpulse import sim
from greenpulse import webster as W

OUT = ROOT / "results"
TEST_SEEDS = list(range(1001, 1021))
TRAIN_SEEDS = [101, 102, 103, 104, 105]
T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:5.0f}s] {m}", flush=True)

CTRL = json.loads((OUT / "controllers.json").read_text())
CTRL.update(json.loads((OUT / "controllers_robust.json").read_text()))
for _f in ("controllers_practical.json", "controllers_practical_robust.json"):
    if (OUT / _f).exists():
        CTRL.update(json.loads((OUT / _f).read_text()))
TUNE = json.loads((OUT / "tuning.json").read_text())
CAL = W.calibrate(0.10)
SC_ALL = SC.TRAIN + SC.UNSEEN
METRICS = ["mean_delay", "p95_delay", "mean_stops", "throughput", "max_queue", "unserved", "delay_N", "delay_E", "delay_S", "delay_W", "switches", "cap_ends", "gap_ends"]

# ------------------------------------------------------------------------------------------ controllers
best_fixed_cache = {}


def best_fixed_for(name):
    if name not in best_fixed_cache:
        _, g, info = B.best_fixed(name, TRAIN_SEEDS)
        best_fixed_cache[name] = (g, info)
    return best_fixed_cache[name][0]


def spec(kind: str, scenario: str, seed: int):
    """-> (controller code, parameters)"""
    if kind == "Fixed equal split":
        return B.equal_plan()
    if kind == "Webster formula":
        if scenario == "random_mix":
            plan = W.webster_plan(SC.mean_demand(scenario, seed=seed), CAL["saturation_flow_cars_per_s"], CAL["startup_lost_time_s"])
            return sim.CTRL_FIXED, plan["greens"]
        return B.webster_fixed(scenario, CAL)
    if kind == "Best fixed plan":
        if scenario == "random_mix":      # no single plan exists; give it Webster's plan for that seed's true demand
            return spec("Webster formula", scenario, seed)
        return sim.CTRL_FIXED, best_fixed_for(scenario)
    if kind == "Vehicle-actuated":
        return B.actuated_plan()
    genes = {"Fuzzy (hand-tuned)": CTRL["hand_tuned"], "Fuzzy + random search": CTRL["random_search"], "Fuzzy + GA": CTRL["ga_scalar"],
             "Fuzzy + NSGA-II": CTRL["nsga2_knee"], "Fuzzy + NSGA-II (robust)": CTRL["nsga2_robust_knee"], "Fuzzy + NSGA-II (practical)": CTRL["nsga2_practical_knee"], "Fuzzy + NSGA-II (practical, robust)": CTRL["nsga2_practical_robust_knee"]}[kind]
    return sim.CTRL_FUZZY, np.array(genes)


KINDS = ["Fixed equal split", "Webster formula", "Best fixed plan", "Vehicle-actuated", "Fuzzy (hand-tuned)", "Fuzzy + random search", "Fuzzy + GA", "Fuzzy + NSGA-II", "Fuzzy + NSGA-II (robust)", "Fuzzy + NSGA-II (practical)", "Fuzzy + NSGA-II (practical, robust)"]


def run_many(kind_or_genes, scenario, seeds, scale=1.0, p_slow=0.10, sigma_q=0.0, dropout=0.0, fixed_kind=None):
    rows = {m: [] for m in METRICS}
    for s in seeds:
        if isinstance(kind_or_genes, str):
            ctrl, par = spec(kind_or_genes, scenario, s)
        else:
            ctrl, par = sim.CTRL_FUZZY, np.asarray(kind_or_genes)
        arr = SC.make_arrivals(scenario, s, scale)
        r = sim.run(ctrl, par, arr, seed=s, p_slow=p_slow, sigma_q=sigma_q, dropout=dropout)
        for m in METRICS:
            rows[m].append(r[m])
    return {m: np.array(v) for m, v in rows.items()}


results = {"seeds": TEST_SEEDS, "scenarios": SC_ALL, "labels": SC.LABELS, "kinds": KINDS, "calibration": {k: v for k, v in CAL.items() if k != "mean_departures"}}
raw = {}
# ------------------------------------------------------------------------------------------ main comparison
main = {}
for kind in KINDS:
    main[kind] = {}
    for sc in SC_ALL:
        raw_ = run_many(kind, sc, TEST_SEEDS)
        raw[(kind, sc)] = raw_
        main[kind][sc] = {m: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "values": v.tolist()} for m, v in raw_.items()}
    log(f"{kind} done")
results["main"] = main
results["best_fixed_plans"] = {k: {"greens": best_fixed_cache[k][0].tolist(), **best_fixed_cache[k][1]} for k in best_fixed_cache}

# paired statistics
stat_rows = []
for focus in ("Fuzzy + NSGA-II", "Fuzzy + NSGA-II (robust)", "Fuzzy + NSGA-II (practical)", "Fuzzy + NSGA-II (practical, robust)"):
    for ref in ("Best fixed plan", "Vehicle-actuated", "Fuzzy (hand-tuned)", "Fuzzy + GA", "Fuzzy + NSGA-II (robust)"):
        if ref == focus:
            continue
        for sc in SC_ALL + ["ALL"]:
            for metric in ("mean_delay", "p95_delay", "mean_stops"):
                if sc == "ALL":
                    a_ = np.concatenate([raw[(focus, s_)][metric] for s_ in SC_ALL])
                    b_ = np.concatenate([raw[(ref, s_)][metric] for s_ in SC_ALL])
                else:
                    a_, b_ = raw[(focus, sc)][metric], raw[(ref, sc)][metric]
                w = stats.wilcoxon(a_, b_)
                stat_rows.append({"focus": focus, "ref": ref, "scenario": sc, "metric": metric, "nsga_mean": float(a_.mean()), "ref_mean": float(b_.mean()),
                                  "change_pct": float(100 * (a_.mean() - b_.mean()) / b_.mean()), "p_value": float(w.pvalue), "wins": int((a_ < b_).sum()), "n": int(len(a_))})
results["paired"] = stat_rows
log("paired statistics done")

# ------------------------------------------------------------------------------------------ the five independent runs
runs = {}
for fam, key, nrun in (("NSGA-II knee", "nsga2_knee_run{}", 5), ("GA", "ga_run{}", 5), ("Random search", "random_run{}", 5), ("NSGA-II robust", "robust_knee_run{}", 3), ("NSGA-II practical", "practical_knee_run{}", 3), ("NSGA-II practical robust", "practical_robust_knee_run{}", 3)):
    runs[fam] = []
    for k in range(nrun):
        g = CTRL[key.format(k)]
        d, p9, st = [], [], []
        for sc in SC.TRAIN:
            r_ = run_many(g, sc, TEST_SEEDS)
            d.append(r_["mean_delay"].mean()); p9.append(r_["p95_delay"].mean()); st.append(r_["mean_stops"].mean())
        runs[fam].append({"run": k, "mean_delay": float(np.mean(d)), "p95_delay": float(np.mean(p9)), "mean_stops": float(np.mean(st))})
results["independent_runs"] = runs
log("independent runs done")

# ------------------------------------------------------------------------------------------ robustness
conds = {
    "Clean": dict(),
    "Noisy sensors (3 cars noise, 20% misses)": dict(sigma_q=3.0, dropout=0.2),
    "Rain (more dawdling)": dict(p_slow=0.25),
    "Demand + 20%": dict(scale=1.2),
    "All three together": dict(sigma_q=3.0, dropout=0.2, p_slow=0.25, scale=1.2),
}
rob = {}
for cname, kw in conds.items():
    rob[cname] = {}
    for kind in ("Best fixed plan", "Vehicle-actuated", "Fuzzy (hand-tuned)", "Fuzzy + GA", "Fuzzy + NSGA-II", "Fuzzy + NSGA-II (robust)", "Fuzzy + NSGA-II (practical)", "Fuzzy + NSGA-II (practical, robust)"):
        d, p9, un, caps, gaps = [], [], [], 0.0, 0.0
        for sc in SC.TRAIN:
            r_ = run_many(kind, sc, TEST_SEEDS, **kw)
            d.append(r_["mean_delay"]); p9.append(r_["p95_delay"]); un.append(r_["unserved"])
            caps += float(r_["cap_ends"].sum()); gaps += float(r_["gap_ends"].sum())
        rob[cname][kind] = {"mean_delay": float(np.mean(d)), "p95_delay": float(np.mean(p9)), "unserved": float(np.mean(un)),
                            "cap_share": caps / (caps + gaps) if caps + gaps else 0.0,
                            "per_scenario": {sc: float(x.mean()) for sc, x in zip(SC.TRAIN, d)}}
    log(f"robustness: {cname}")
results["robustness"] = rob

# ------------------------------------------------------------------------------------------ ablation of the two fuzzy systems
abl = {}
for cname, genes0 in (("NSGA-II (clean-tuned)", CTRL["nsga2_knee"]), ("NSGA-II (robust-tuned)", CTRL["nsga2_robust_knee"])):
    g = np.array(genes0)
    g_noq = g.copy(); g_noq[0:3] = [39.0, 39.5, 40.0]                  # queue never registers -> urgency depends on waiting time only
    g_nogt = g.copy(); g_nogt[15:18] = [29.5, 30.0, 30.5]              # green cap always about 30 s
    variants = {"Full controller": g, "Wait-only urgency": FZ.repair(g_noq), "Fixed 30 s green cap": FZ.repair(g_nogt)}
    abl[cname] = {}
    for cond in ("Clean", "All three together"):
        abl[cname][cond] = {}
        for vname, vg in variants.items():
            d, p9, st = [], [], []
            for sc in SC.TRAIN:
                r_ = run_many(vg, sc, TEST_SEEDS, **conds[cond])
                d.append(r_["mean_delay"].mean()); p9.append(r_["p95_delay"].mean()); st.append(r_["mean_stops"].mean())
            abl[cname][cond][vname] = {"mean_delay": float(np.mean(d)), "p95_delay": float(np.mean(p9)), "mean_stops": float(np.mean(st))}
    abl[cname]["green_cap_centres"] = g[15:18].tolist()
    abl[cname]["min_green"] = float(g[18])
    abl[cname]["gap_out"] = float(g[19])
results["ablation"] = abl
log("ablation done")

# ------------------------------------------------------------------------------------------ what if a real controller must keep green longer?
msens = {}
for cname, genes0 in (("NSGA-II (clean-tuned)", CTRL["nsga2_knee"]), ("NSGA-II (robust-tuned)", CTRL["nsga2_robust_knee"])):
    msens[cname] = {}
    for mg in (4, 6, 8, 10, 12):
        g = np.array(genes0)
        g[18] = mg
        d, sw = [], []
        for sc in SC.TRAIN:
            r_ = run_many(g, sc, TEST_SEEDS)
            d.append(r_["mean_delay"].mean()); sw.append(r_["switches"].mean())
        msens[cname][str(mg)] = {"mean_delay": float(np.mean(d)), "switches": float(np.mean(sw))}
for kind in ("Vehicle-actuated", "Best fixed plan"):
    msens[kind] = {"mean_delay": float(np.mean([main[kind][sc]["mean_delay"]["mean"] for sc in SC.TRAIN])),
                   "switches": float(np.mean([main[kind][sc]["switches"]["mean"] for sc in SC.TRAIN]))}
results["min_green_sensitivity"] = msens
log("minimum-green sensitivity done")

# ------------------------------------------------------------------------------------------ Pareto front on test seeds
front = TUNE["selected_front"]
F_train = np.array(front["objectives"])
F_test = []
for genes in front["genes"]:
    d, p9, st = [], [], []
    for sc in SC.TRAIN:
        r_ = run_many(genes, sc, TEST_SEEDS)
        d.append(r_["mean_delay"].mean()); p9.append(r_["p95_delay"].mean()); st.append(r_["mean_stops"].mean())
    F_test.append([np.mean(d), np.mean(p9), np.mean(st)])
F_test = np.array(F_test)


def mean3(kind):
    d, p9, st = [], [], []
    for sc in SC.TRAIN:
        d.append(main[kind][sc]["mean_delay"]["mean"]); p9.append(main[kind][sc]["p95_delay"]["mean"]); st.append(main[kind][sc]["mean_stops"]["mean"])
    return np.array([np.mean(d), np.mean(p9), np.mean(st)])


act, hand = mean3("Vehicle-actuated"), mean3("Fuzzy (hand-tuned)")
results["front"] = {"train": F_train.tolist(), "test": F_test.tolist(), "knee_index": front["knee_index"],
                    "actuated_test": act.tolist(), "hand_test": hand.tolist(),
                    "members_dominating_actuated": int(np.sum(np.all(F_test < act, axis=1))),
                    "members_dominating_hand": int(np.sum(np.all(F_test < hand, axis=1))),
                    "members_beating_actuated_on_delay_and_p95": int(np.sum((F_test[:, 0] < act[0]) & (F_test[:, 1] < act[1]))),
                    "size": int(len(F_test))}
log("front evaluated")

# ------------------------------------------------------------------------------------------ behaviour traces
traces = {}
arr_s = SC.make_arrivals("surge", 1001)
arr_a = SC.make_arrivals("arterial", 1001)
for tag, arr, sc in (("surge", arr_s, "surge"), ("arterial", arr_a, "arterial")):
    for kind in ("Best fixed plan", "Vehicle-actuated", "Fuzzy + NSGA-II"):
        ctrl, par = spec(kind, sc, 1001)
        r_ = sim.run(ctrl, par, arr, seed=1001)
        traces[f"{tag}|{kind}"] = {"signal": r_["signal"][:3700].tolist(), "queues": r_["queues"][:3700].tolist()}
results["trace_keys"] = list(traces)
beh = {}
for sc in SC.TRAIN:
    beh[sc] = {}
    for kind in ("Best fixed plan", "Vehicle-actuated", "Fuzzy + NSGA-II", "Fuzzy + NSGA-II (robust)", "Fuzzy + NSGA-II (practical)", "Fuzzy + NSGA-II (practical, robust)"):
        ctrl, par = spec(kind, sc, 1001)
        r_ = sim.run(ctrl, par, SC.make_arrivals(sc, 1001), seed=1001)
        sig = r_["signal"][:3600]
        shares = [float(np.mean(sig == a)) for a in range(4)]
        runs_ = []
        cur, ln = -1, 0
        for s_ in sig:
            if s_ == cur and s_ >= 0:
                ln += 1
            else:
                if cur >= 0:
                    runs_.append(ln)
                cur, ln = s_, 1 if s_ >= 0 else 0
        beh[sc][kind] = {"green_share": shares, "mean_green_s": float(np.mean(runs_)), "greens_per_hour": len(runs_),
                         "cap_ends": float(r_["cap_ends"]), "gap_ends": float(r_["gap_ends"])}
results["behaviour"] = beh
(OUT / "evaluation.json").write_text(json.dumps(results), encoding="utf-8")
np.savez_compressed(OUT / "eval_traces.npz", **{k.replace("|", "__").replace(" ", "_"): np.array(v["queues"]) for k, v in traces.items()},
                    **{("sig__" + k.replace("|", "__").replace(" ", "_")): np.array(v["signal"]) for k, v in traces.items()})
log("evaluation finished")

# --------------------------------------------------------------------------------------- console summary
print("\nMean delay (s) on 20 test seeds")
print(f"{'':24s}" + "".join(f"{SC.SHORT[s]:>11s}" for s in SC_ALL))
for kind in KINDS:
    print(f"{kind:24s}" + "".join(f"{main[kind][s]['mean_delay']['mean']:11.1f}" for s in SC_ALL))
print("\nP95 delay (s)")
for kind in KINDS:
    print(f"{kind:24s}" + "".join(f"{main[kind][s]['p95_delay']['mean']:11.1f}" for s in SC_ALL))
print("\nStops per vehicle")
for kind in KINDS:
    print(f"{kind:24s}" + "".join(f"{main[kind][s]['mean_stops']['mean']:11.2f}" for s in SC_ALL))
print("\nUnserved at end")
for kind in KINDS:
    print(f"{kind:24s}" + "".join(f"{main[kind][s]['unserved']['mean']:11.1f}" for s in SC_ALL))
print("\nNSGA-II vs references (ALL scenarios):")
for row in stat_rows:
    if row["scenario"] == "ALL" and row["focus"] == "Fuzzy + NSGA-II":
        print(f"  vs {row['ref']:20s} {row['metric']:11s} {row['nsga_mean']:7.2f} vs {row['ref_mean']:7.2f} ({row['change_pct']:+6.1f}%)  p={row['p_value']:.2e} wins {row['wins']}/{row['n']}")
print("\nrobustness (mean delay over 4 scenarios):")
for c_, d in rob.items():
    print(f"  {c_:42s}" + "".join(f"{k.split(' ')[0]}:{v['mean_delay']:6.1f} " for k, v in d.items()))
print("\nablation (clean, all three), full controller:", {k: (round(v["Clean"]["Full controller"]["mean_delay"], 2), round(v["All three together"]["Full controller"]["mean_delay"], 2)) for k, v in abl.items()})
print("front:", {k: v for k, v in results["front"].items() if k not in ("train", "test")})
print("independent runs:", {fam: [round(x["mean_delay"], 2) for x in v] for fam, v in runs.items()})
