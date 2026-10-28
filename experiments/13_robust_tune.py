"""Step 13: robust tuning. Same NSGA-II, but the training runs now include noisy sensors, rain, +20% demand and all three at once,
and every objective is expressed relative to the vehicle-actuated controller on the same run.

Why: the clean-tuned controller of step 11 lost most of its advantage with noisy sensors and collapsed under the combined stress.

Usage:  python -I experiments/13_robust_tune.py [pop] [generations] [runs]
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
N_RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
TRAIN_SEEDS = [101, 102]
CONDITIONS = {                       # scale, p_slow, sigma_q, dropout
    "clean": (1.0, 0.10, 0.0, 0.0),
    "noisy sensors": (1.0, 0.10, 3.0, 0.2),
    "rain": (1.0, 0.25, 0.0, 0.0),
    "demand +20%": (1.2, 0.10, 0.0, 0.0),
    "all three": (1.2, 0.25, 3.0, 0.2),
}


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
    specs = [(name, seed, sc, ps, sq, dr) for name in SC.TRAIN for (sc, ps, sq, dr) in CONDITIONS.values() for seed in TRAIN_SEEDS]
    ev = ga.Evaluator(specs, n_workers=10, cfg={"relative": True})
    ref = ev(FZ.HAND_TUNED[None, :])[0]
    log(f"{len(specs)} training runs per candidate; hand-tuned objectives relative to actuated = {np.round(ref, 3)}")
    out = {"pop": POP, "gens": GENS, "conditions": {k: list(v) for k, v in CONDITIONS.items()}, "train_seeds": TRAIN_SEEDS, "hand_tuned_relative": ref.tolist(), "runs": []}
    controllers = {}
    for run in range(N_RUNS):
        rng = np.random.default_rng(5000 + run)
        r = ga.nsga2(ev, POP, GENS, rng, ref)
        front = r["F"][r["front_idx"]]
        k = ga.knee_point(front)
        genes = r["pop"][r["front_idx"]]
        out["runs"].append({"run": run, "final_hv": r["history"][-1]["hv"], "front_size": int(len(front)), "front_objectives": front.tolist(),
                            "front_genes": genes.tolist(), "knee_index": k, "knee_objectives": front[k].tolist(),
                            "history": [{kk: h[kk] for kk in ("gen", "evals", "hv", "best", "front_size")} for h in r["history"]]})
        controllers[f"robust_knee_run{run}"] = genes[k].tolist()
        log(f"robust NSGA-II run {run}: HV {r['history'][-1]['hv']:.4f}, knee relative objectives {np.round(front[k], 3)}")
    best = max(out["runs"], key=lambda r: r["final_hv"])
    controllers["nsga2_robust_knee"] = controllers[f"robust_knee_run{best['run']}"]
    out["selected_run"] = best["run"]
    (OUT / "tuning_robust.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT / "controllers_robust.json").write_text(json.dumps(controllers, indent=1), encoding="utf-8")
    ev.close()
    log("robust tuning finished")


if __name__ == "__main__":
    main()
