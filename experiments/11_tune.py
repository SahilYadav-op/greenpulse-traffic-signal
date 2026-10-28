"""Step 11: tune the fuzzy controller with NSGA-II, a plain GA and random search (same budget, 5 independent runs each).

Training uses only the TRAIN scenarios with training seeds 101..105. Nothing here touches the test seeds.

Usage:  python -I experiments/11_tune.py [pop] [generations]
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

from greenpulse import fuzzy as FZ
from greenpulse import ga
from greenpulse import scenarios as SC

OUT = ROOT / "results"
POP = int(sys.argv[1]) if len(sys.argv) > 1 else 80
GENS = int(sys.argv[2]) if len(sys.argv) > 2 else 80
N_RUNS = 5
TRAIN_SEEDS = [101, 102, 103, 104, 105]
N_WORKERS = 10


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
    specs = [(name, seed, 1.0) for name in SC.TRAIN for seed in TRAIN_SEEDS]
    ev = ga.Evaluator(specs, n_workers=N_WORKERS)
    ref = ev(FZ.HAND_TUNED[None, :])[0]
    log(f"training set: {len(specs)} runs per candidate; hand-tuned objectives on it = {np.round(ref, 3)}")

    results = {"pop": POP, "gens": GENS, "evals_per_run": POP * (GENS + 1), "train_seeds": TRAIN_SEEDS, "train_scenarios": SC.TRAIN,
               "hand_tuned_train_objectives": ref.tolist(), "nsga2": [], "ga": [], "random": []}
    controllers = {"hand_tuned": FZ.HAND_TUNED.tolist()}

    for run in range(N_RUNS):
        rng = np.random.default_rng(1000 + run)
        r = ga.nsga2(ev, POP, GENS, rng, ref)
        front = r["F"][r["front_idx"]]
        k = ga.knee_point(front)
        gene_front = r["pop"][r["front_idx"]]
        results["nsga2"].append({
            "run": run, "final_hv": r["history"][-1]["hv"], "front_size": int(len(front)),
            "front_objectives": front.tolist(), "front_genes": gene_front.tolist(), "knee_index": k,
            "knee_objectives": front[k].tolist(), "history": [{kk: h[kk] for kk in ("gen", "evals", "hv", "best", "front_size")} for h in r["history"]],
        })
        controllers[f"nsga2_knee_run{run}"] = gene_front[k].tolist()
        log(f"NSGA-II run {run}: HV {r['history'][-1]['hv']:.4f}, front {len(front)} points, knee objectives {np.round(front[k], 3)} (hand-tuned = {np.round(ref,3)})")

        rng = np.random.default_rng(2000 + run)
        g = ga.simple_ga(ev, POP, GENS, rng, ref)
        results["ga"].append({"run": run, "best_J": g["best_J"], "best_objectives": g["best_F"].tolist(), "best_genes": g["best"].tolist(),
                              "history": g["history"]})
        controllers[f"ga_run{run}"] = g["best"].tolist()
        log(f"GA run {run}: best J {g['best_J']:.4f}, objectives {np.round(g['best_F'], 3)}")

        rng = np.random.default_rng(3000 + run)
        rs = ga.random_search(ev, POP * (GENS + 1), rng, ref)
        results["random"].append({"run": run, "best_J": rs["best_J"], "best_objectives": rs["best_F"].tolist(), "best_genes": rs["best"].tolist(),
                                  "history": rs["history"]})
        controllers[f"random_run{run}"] = rs["best"].tolist()
        log(f"random search run {run}: best J {rs['best_J']:.4f}")

    # choose final controllers using TRAINING results only
    best_ga = min(results["ga"], key=lambda r: r["best_J"])
    best_nsga = max(results["nsga2"], key=lambda r: r["final_hv"])
    best_rs = min(results["random"], key=lambda r: r["best_J"])
    controllers["ga_scalar"] = best_ga["best_genes"]
    controllers["nsga2_knee"] = controllers[f"nsga2_knee_run{best_nsga['run']}"]
    controllers["random_search"] = best_rs["best_genes"]
    results["selected"] = {"ga_run": best_ga["run"], "nsga2_run": best_nsga["run"], "random_run": best_rs["run"]}
    results["selected_front"] = {"objectives": best_nsga["front_objectives"], "genes": best_nsga["front_genes"], "knee_index": best_nsga["knee_index"]}
    (OUT / "tuning.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (OUT / "controllers.json").write_text(json.dumps(controllers, indent=1), encoding="utf-8")
    ev.close()
    log("tuning finished")


if __name__ == "__main__":
    main()
