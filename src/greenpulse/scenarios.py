"""Demand scenarios (cars per hour on each approach N, E, S, W) and the random arrival streams.

Every (scenario, seed) pair generates ONE arrival stream that is given to all controllers
(common random numbers), so differences between controllers are never luck of the draw.

TRAIN scenarios are what the genetic algorithm sees (with training seeds only).
UNSEEN scenarios are never used in tuning: they test whether the controller adapts to a pattern it
was not tuned for.
"""
from __future__ import annotations

import numpy as np

from . import sim

# (start s, end s, [N, E, S, W] veh/h)
SCENARIOS: dict[str, list[tuple[int, int, list[float]]]] = {
    "light": [(0, 3600, [150, 150, 150, 150])],
    "balanced": [(0, 3600, [300, 300, 300, 300])],
    "arterial": [(0, 3600, [560, 110, 300, 110])],
    "surge": [(0, 1200, [220, 220, 220, 220]), (1200, 2400, [420, 180, 420, 180]), (2400, 3600, [250, 250, 250, 250])],
    "ew_rush": [(0, 3600, [150, 420, 150, 420])],
}
TRAIN = ["light", "balanced", "arterial", "surge"]
UNSEEN = ["ew_rush", "random_mix"]
LABELS = {
    "light": "Light off-peak (150 each)",
    "balanced": "Balanced heavy (300 each)",
    "arterial": "Dominant arterial (N 560)",
    "surge": "Peak surge (N-S rush)",
    "ew_rush": "East-West rush (unseen)",
    "random_mix": "Random demand (unseen)",
}
SHORT = {"light": "Light", "balanced": "Balanced", "arterial": "Arterial", "surge": "Surge",
         "ew_rush": "E-W rush", "random_mix": "Random mix"}
_ID = {n: i for i, n in enumerate(list(SCENARIOS) + ["random_mix"])}
ORDER = TRAIN


def mix_demand(seed: int) -> np.ndarray:
    """Per-approach demand of the 'random_mix' scenario for a seed (veh/h)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4242]))
    return rng.uniform(100, 340, sim.NA)


def rates_per_second(name: str, scale: float = 1.0, T: int = sim.T_ARR, seed: int = 0) -> np.ndarray:
    """(4, T) arrival probability per second for the scenario."""
    r = np.zeros((sim.NA, T))
    if name == "random_mix":
        r[:, :] = (mix_demand(seed) * scale / 3600.0)[:, None]
        return r
    for t0, t1, q in SCENARIOS[name]:
        r[:, t0:min(t1, T)] = (np.asarray(q, float) * scale / 3600.0)[:, None]
    return r


def mean_demand(name: str, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Time-averaged veh/h per approach (what a traffic engineer would count)."""
    return rates_per_second(name, scale, seed=seed).mean(axis=1) * 3600.0


def make_arrivals(name: str, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, _ID[name]]))
    return (rng.random((sim.NA, sim.T_ARR)) < rates_per_second(name, scale, seed=seed)).astype(np.uint8)


def custom_arrivals(veh_per_hour, seed: int, T: int = sim.T_ARR) -> np.ndarray:
    """Constant-rate stream for the app's 'your own junction' mode."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    r = (np.asarray(veh_per_hour, float) / 3600.0)[:, None] * np.ones((1, T))
    return (rng.random((sim.NA, T)) < r).astype(np.uint8)
