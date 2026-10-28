"""Reference controllers that the fuzzy controllers are compared with."""
from __future__ import annotations

import numpy as np

from . import scenarios as SC
from . import sim
from . import webster as W

EQUAL_GREEN = 22.0              # four phases of 22 s + 3 s all-red = a 100 s cycle: the 'nobody planned this' signal
ACTUATED = (7.0, 45.0, 3.0)     # min green, max green, gap-out time (seconds)


def equal_plan() -> tuple[int, np.ndarray]:
    return sim.CTRL_FIXED, np.full(4, EQUAL_GREEN)


def actuated_plan() -> tuple[int, np.ndarray]:
    return sim.CTRL_ACTUATED, np.array(ACTUATED)


def webster_fixed(scenario: str, cal: dict, scale: float = 1.0) -> tuple[int, np.ndarray]:
    plan = W.webster_plan(SC.mean_demand(scenario, scale), cal["saturation_flow_cars_per_s"], cal["startup_lost_time_s"])
    return sim.CTRL_FIXED, plan["greens"]


def split_for_cycle(demand_vph, cycle_s: float, intergreen: int = sim.INTERGREEN, min_green: float = 5.0) -> np.ndarray:
    """Green times that share a given cycle length in proportion to demand (Webster's split rule)."""
    q = np.asarray(demand_vph, float)
    avail = cycle_s - 4 * intergreen
    g = avail * (q / max(q.sum(), 1e-9))
    g = np.maximum(g, min_green)
    return g * (avail / g.sum()) if g.sum() > avail else g


def best_fixed(scenario: str, train_seeds, scale: float = 1.0, cycles=None):
    """Grid search over cycle lengths on the TRAINING seeds: the strongest fixed-time plan we can make for a scenario
    whose average demand is known in advance (an optimistic baseline for any fixed plan)."""
    cycles = cycles if cycles is not None else list(range(40, 161, 10))
    demand = SC.mean_demand(scenario, scale)
    arrs = [SC.make_arrivals(scenario, s, scale) for s in train_seeds]
    best = None
    for c in cycles:
        g = split_for_cycle(demand, c)
        d = np.mean([sim.run(sim.CTRL_FIXED, g, a, seed=s)["mean_delay"] for a, s in zip(arrs, train_seeds)])
        if best is None or d < best[0]:
            best = (float(d), float(c), g)
    return sim.CTRL_FIXED, best[2], {"cycle_s": best[1], "train_delay": best[0]}
