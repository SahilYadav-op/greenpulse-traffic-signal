"""Webster's method for fixed-time plans and his delay formula (used as the classic baseline and for validation).

F. V. Webster, "Traffic signal settings", Road Research Technical Paper No. 39, HMSO, London, 1958.
"""
from __future__ import annotations

import numpy as np

from . import sim


def calibrate(p_slow: float = 0.10, reps: int = 40, queue_len: int = 30):
    """Measure saturation flow s (cars/s) and start-up lost time (s) of the cellular-automaton lane.

    A standing queue is released at t = 0; departure time of car i is modelled as lost + i / s.
    """
    deps = []
    for r in range(reps):
        d = sim.discharge_test(p_slow, 1000 + r, queue_len, 200)
        deps.append(d)
    dep = np.array(deps, dtype=float)               # (reps, queue_len)
    idx = np.arange(1, queue_len + 1)
    sel = slice(4, queue_len)                       # ignore the first cars (start-up), fit the steady part
    mean_dep = dep.mean(axis=0)
    slope, intercept = np.polyfit(idx[sel], mean_dep[sel], 1)
    s = 1.0 / slope
    fit = float(intercept)
    # The first cars of a standing queue leave slightly faster than the steady discharge rate, so the straight-line
    # fit gives a negative intercept. A lane cannot gain time, so the plan calculations use max(fit, 0).
    return {"saturation_flow_cars_per_s": float(s), "saturation_flow_cars_per_h": float(3600 * s),
            "headway_s": float(slope), "startup_lost_time_fit_s": fit, "startup_lost_time_s": max(fit, 0.0),
            "first_car_departure_s": float(mean_dep[0]), "mean_departures": mean_dep.tolist()}


def webster_plan(demand_vph, s: float, lost_start: float, intergreen: int = sim.INTERGREEN, min_green: float = 6.0,
                 max_cycle: float = 150.0):
    """Cycle length and displayed green times (one approach per phase).

    y_i = q_i / s;  Y = sum y_i;  L = total lost time per cycle;  C_o = (1.5 L + 5) / (1 - Y)
    effective green_i = (C_o - L) * y_i / Y ;  displayed green_i = effective green_i + start-up lost time
    """
    q = np.asarray(demand_vph, float) / 3600.0
    y = q / s
    Y = float(y.sum())
    n = len(q)
    L = n * (lost_start + intergreen)
    if Y >= 0.95:
        Co = max_cycle
    else:
        Co = float(np.clip((1.5 * L + 5.0) / (1.0 - Y), 40.0, max_cycle))
    g_eff = (Co - L) * (y / max(Y, 1e-9))
    greens = np.maximum(g_eff + lost_start, min_green)
    return {"cycle_s": float(greens.sum() + n * intergreen), "greens": greens, "Y": Y, "L": float(L)}


def webster_delay(q_vph: float, green_s: float, cycle_s: float, s: float, lost_start: float) -> float:
    """Webster's average delay per vehicle (s) for one approach: uniform + random - correction term."""
    q = q_vph / 3600.0
    g_eff = max(green_s - lost_start, 1e-6)
    lam = g_eff / cycle_s
    x = q / (s * lam)
    if x >= 0.98:
        return float("nan")
    d1 = cycle_s * (1 - lam) ** 2 / (2 * (1 - lam * x))
    d2 = x ** 2 / (2 * q * (1 - x))
    corr = 0.65 * (cycle_s / q ** 2) ** (1 / 3) * x ** (2 + 5 * lam)
    return float(d1 + d2 - corr)
