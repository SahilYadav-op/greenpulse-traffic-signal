"""Step 17: the two realistic-green fronts (minimum green >= 8 s) evaluated on the held-out test seeds.

Answers one question: with a realistic minimum green, does any member of the Pareto front beat the actuated controller on
mean delay and on 95th-percentile delay at the same time?  Needs evaluation.json from step 12.

Usage:  python -I experiments/17_practical_front.py
"""
import json
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from greenpulse import scenarios as SC
from greenpulse import sim

OUT = ROOT / "results"
TEST_SEEDS = list(range(1001, 1021))
EV = json.loads((OUT / "evaluation.json").read_text())
ACT = np.array(EV["front"]["actuated_test"])      # mean delay, p95 delay, stops of the actuated controller on the four tuned patterns
HAND = np.array(EV["front"]["hand_test"])


def test_objectives(genes) -> list:
    """(mean delay, p95 delay, stops per car) averaged over the four tuned demand patterns x 20 test seeds."""
    g = np.asarray(genes, dtype=np.float64)
    d, p9, st = [], [], []
    for sc in SC.TRAIN:
        dd, pp, ss = [], [], []
        for s in TEST_SEEDS:
            r = sim.run(sim.CTRL_FUZZY, g, SC.make_arrivals(sc, s), seed=s)
            dd.append(r["mean_delay"]); pp.append(r["p95_delay"]); ss.append(r["mean_stops"])
        d.append(np.mean(dd)); p9.append(np.mean(pp)); st.append(np.mean(ss))
    return [float(np.mean(d)), float(np.mean(p9)), float(np.mean(st))]


def main():
    out = {"actuated_test": ACT.tolist(), "hand_test": HAND.tolist()}
    for tag, fname in (("practical", "tuning_practical.json"), ("practical_robust", "tuning_practical_robust.json")):
        T = json.loads((OUT / fname).read_text())
        run = T["runs"][T["selected_run"]]
        F = np.array([test_objectives(g) for g in run["front_genes"]])
        out[tag] = {"run": T["selected_run"], "knee_index": run["knee_index"], "size": int(len(F)), "test": F.tolist(),
                    "min_green": [float(g[18]) for g in run["front_genes"]],
                    "dominating_actuated": int(np.sum(np.all(F < ACT, axis=1))),
                    "beating_actuated_delay_and_p95": int(np.sum((F[:, 0] < ACT[0]) & (F[:, 1] < ACT[1]))),
                    "beating_actuated_delay": int(np.sum(F[:, 0] < ACT[0])),
                    "beating_actuated_p95": int(np.sum(F[:, 1] < ACT[1])),
                    "best_delay": float(F[:, 0].min()), "best_p95": float(F[:, 1].min())}
        if out[tag]["beating_actuated_delay_and_p95"]:
            ok = (F[:, 0] < ACT[0]) & (F[:, 1] < ACT[1])
            out[tag]["both_best_delay_member"] = F[ok][np.argmin(F[ok][:, 0])].tolist()
        print(tag, {k: v for k, v in out[tag].items() if k not in ("test", "min_green")})
    print("actuated", np.round(ACT, 2), "hand", np.round(HAND, 2))
    (OUT / "practical_front.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
