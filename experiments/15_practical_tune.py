"""Step 15: practical robust tuning. Same robust NSGA-II as step 13, but the minimum green may not go below 8 s (a 4 s green is not something a real
signal can show). Result: controllers_practical_robust.json.

Usage:  python -I experiments/15_practical_tune.py [pop] [generations] [runs]
"""
import os
import site
import sys

os.environ["GREENPULSE_MIN_GREEN_LO"] = "8"        # must be set before greenpulse is imported (and is inherited by the GA workers)
sys.path.append(site.getusersitepackages())
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from greenpulse import fuzzy as FZ
from greenpulse import ga
from greenpulse import scenarios as SC

assert FZ.GENE_LO[18] == 8.0
OUT = ROOT / "results"
POP = int(sys.argv[1]) if len(sys.argv) > 1 else 80
GENS = int(sys.argv[2]) if len(sys.argv) > 2 else 80
N_RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
TRAIN_SEEDS = [101, 102]
CONDITIONS = {"clean": (1.0, 0.10, 0.0, 0.0), "noisy sensors": (1.0, 0.10, 3.0, 0.2), "rain": (1.0, 0.25, 0.0, 0.0),
              "demand +20%": (1.2, 0.10, 0.0, 0.0), "all three": (1.2, 0.25, 3.0, 0.2)}


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
    specs = [(name, seed, sc, ps, sq, dr) for name in SC.TRAIN for (sc, ps, sq, dr) in CONDITIONS.values() for seed in TRAIN_SEEDS]
    ev = ga.Evaluator(specs, n_workers=10, cfg={"relative": True})
    ref = ev(np.clip(FZ.HAND_TUNED, FZ.GENE_LO, FZ.GENE_HI)[None, :])[0]
    log(f"{len(specs)} runs per candidate; reference objectives {np.round(ref, 3)}")
    out, controllers = {"pop": POP, "gens": GENS, "min_green_lower_bound": 8.0, "runs": []}, {}
    for run in range(N_RUNS):
        rng = np.random.default_rng(7000 + run)
        r = ga.nsga2(ev, POP, GENS, rng, ref)
        front = r["F"][r["front_idx"]]
        k = ga.knee_point(front)
        genes = r["pop"][r["front_idx"]]
        out["runs"].append({"run": run, "final_hv": r["history"][-1]["hv"], "front_size": int(len(front)), "knee_index": k, "knee_objectives": front[k].tolist(),
                            "front_objectives": front.tolist(), "front_genes": genes.tolist()})
        controllers[f"practical_robust_knee_run{run}"] = genes[k].tolist()
        log(f"practical NSGA-II run {run}: HV {r['history'][-1]['hv']:.4f}, knee {np.round(front[k], 3)}, min green {genes[k][18]:.1f}")
    best = max(out["runs"], key=lambda r: r["final_hv"])
    controllers["nsga2_practical_robust_knee"] = controllers[f"practical_knee_run{best['run']}"]
    out["selected_run"] = best["run"]
    (OUT / "tuning_practical_robust.json").write_text(json.dumps(out, indent=1))
    (OUT / "controllers_practical_robust.json").write_text(json.dumps(controllers, indent=1))
    ev.close()
    log("practical robust tuning finished")


if __name__ == "__main__":
    main()
