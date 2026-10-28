"""Step 10: does the cellular-automaton junction behave like a real junction?

Three independent checks, none of which uses the controllers:
  1. saturation flow and start-up behaviour of a standing queue released on green
  2. fundamental diagram (flow against density) of the same car-following rule on a ring road
  3. average delay of a fixed-time plan against Webster's analytical delay formula

Usage:  python -I experiments/10_validate_sim.py
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
from greenpulse import webster as W

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
res = {}

# ---------------------------------------------------------------- 1. discharge
cal = {}
for p in (0.0, 0.05, 0.10, 0.20, 0.25):
    c = W.calibrate(p)
    cal[str(p)] = {k: v for k, v in c.items() if k != "mean_departures"}
    if p == 0.10:
        res["discharge_mean_departures"] = c["mean_departures"]
res["calibration"] = cal
base = W.calibrate(0.10)
S_FLOW, LOST = base["saturation_flow_cars_per_s"], base["startup_lost_time_s"]
print("p=0.10 saturation flow", round(3600 * S_FLOW), "veh/h; fitted start-up lost time", round(base["startup_lost_time_fit_s"], 2), "s; first car leaves after", round(base["first_car_departure_s"], 2), "s")

# ---------------------------------------------------------------- 2. fundamental diagram
dens = [0.02, 0.05, 0.08, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
fd = {}
for p in (0.0, 0.10, 0.20):
    flows = []
    for d in dens:
        flows.append(float(np.mean([sim.ring_flow(d, p, s, 200, 5000) for s in range(6)]) * 3600))
    fd[str(p)] = flows
res["fundamental_diagram"] = {"density_cars_per_cell": dens, "density_veh_per_km": [d / (sim.CELL_M / 1000) for d in dens], "flow_veh_per_h": fd}
f01 = np.array(fd["0.1"])
res["fd_summary"] = {"max_flow_veh_per_h": float(f01.max()), "density_at_max_veh_per_km": float(dens[int(f01.argmax())] / (sim.CELL_M / 1000)),
                     "free_flow_speed_kmh": float(f01[0] / (dens[0] / (sim.CELL_M / 1000)))}
print("fundamental diagram:", res["fd_summary"])

# ---------------------------------------------------------------- 3. Webster delay
rows = []
SEEDS = list(range(1, 21))
for q in (100, 150, 200, 250, 300, 350, 400):
    plan = W.webster_plan([q] * 4, S_FLOW, LOST)
    sim_d, thr = [], []
    for s in SEEDS:
        r = sim.run(sim.CTRL_FIXED, plan["greens"], SC.custom_arrivals([q] * 4, s), seed=s)
        sim_d.append(r["mean_delay"])
        thr.append(r["throughput"])
    ana = float(np.mean([W.webster_delay(q, g, plan["cycle_s"], S_FLOW, LOST) for g in plan["greens"]]))
    rows.append({"flow_per_approach": q, "cycle_s": plan["cycle_s"], "green_s": float(plan["greens"][0]), "Y": plan["Y"],
                 "sim_delay": float(np.mean(sim_d)), "sim_delay_sd": float(np.std(sim_d)), "webster_delay": ana,
                 "ratio": float(np.mean(sim_d) / ana) if ana == ana else None})
    print(rows[-1])
res["webster_validation"] = rows
valid = [r for r in rows if r["ratio"] is not None]
res["webster_summary"] = {"mean_abs_pct_error": float(np.mean([abs(r["sim_delay"] - r["webster_delay"]) / r["webster_delay"] for r in valid]) * 100),
                          "correlation": float(np.corrcoef([r["sim_delay"] for r in valid], [r["webster_delay"] for r in valid])[0, 1])}
print(res["webster_summary"])
(OUT / "validation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("wrote", OUT / "validation.json")
