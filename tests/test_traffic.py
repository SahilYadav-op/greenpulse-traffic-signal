"""Tests for the junction model, the fuzzy controller and the GA machinery.

Run:  python -I tests/test_traffic.py        (or: pytest tests)
"""
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from greenpulse import fuzzy as FZ
from greenpulse import ga
from greenpulse import scenarios as SC
from greenpulse import sim
from greenpulse import webster as W


def test_conservation_every_arrival_is_served_or_counted():
    for ctrl, p in ((sim.CTRL_FIXED, [22] * 4), (sim.CTRL_ACTUATED, [7, 45, 3]), (sim.CTRL_FUZZY, FZ.HAND_TUNED)):
        r = sim.run(ctrl, p, SC.make_arrivals("balanced", 3), seed=3)
        assert r["arrived"] == r["served"] + r["unserved"]
        assert len(r["delays"]) == r["arrived"]
        assert r["delays"].min() >= 0


def test_deterministic_for_same_seed_and_different_for_other_seed():
    arr = SC.make_arrivals("surge", 4)
    a = sim.run(sim.CTRL_FUZZY, FZ.HAND_TUNED, arr, seed=9)["mean_delay"]
    b = sim.run(sim.CTRL_FUZZY, FZ.HAND_TUNED, arr, seed=9)["mean_delay"]
    c = sim.run(sim.CTRL_FUZZY, FZ.HAND_TUNED, arr, seed=10)["mean_delay"]
    assert a == b and a != c


def test_only_one_approach_has_green_and_intergreen_is_respected():
    r = sim.run(sim.CTRL_FUZZY, FZ.HAND_TUNED, SC.make_arrivals("balanced", 2), seed=2)
    sig = r["signal"]
    # between two different greens there must be at least INTERGREEN red seconds
    last_green, last_t = None, None
    for t, s in enumerate(sig):
        if s >= 0:
            if last_green is not None and s != last_green:
                assert t - last_t - 1 >= sim.INTERGREEN
            last_green, last_t = s, t


def test_cars_never_enter_the_junction_on_red():
    # with all greens 0 s the first approach still gets its minimum; make demand only on N and give E green first:
    arr = np.zeros((4, 600), np.uint8)
    arr[0, ::5] = 1                                  # cars only on the north approach
    r = sim.run(sim.CTRL_FIXED, [1, 30, 1, 1], arr, seed=1, drain=200)
    sig = r["signal"]
    # served count can never exceed what green seconds allow: 1-s greens for N
    assert r["served"] <= int(np.sum(sig == 0)) + 1


def test_saturation_flow_is_realistic():
    cal = W.calibrate(0.10)
    assert 1700 < cal["saturation_flow_cars_per_h"] < 2200


def test_fundamental_diagram_has_one_peak():
    dens = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    flow = [np.mean([sim.ring_flow(d, 0.1, s, 200, 3000) for s in range(2)]) for d in dens]
    assert int(np.argmax(flow)) in (3, 4)            # peak at moderate density
    assert flow[0] < flow[3] and flow[-1] < flow[3]


def test_fuzzy_urgency_grows_with_queue_and_wait():
    g = FZ.HAND_TUNED
    assert FZ.urgency(0, 0, g) < FZ.urgency(10, 40, g) < FZ.urgency(25, 150, g)
    assert FZ.urgency(0, 200, g) > FZ.urgency(0, 0, g)       # an empty but long-neglected approach gains urgency


def test_fuzzy_green_time_grows_with_queue_and_shrinks_with_pressure():
    g = FZ.HAND_TUNED
    assert FZ.green_time(2, 5, g) < FZ.green_time(10, 5, g) < FZ.green_time(25, 0, g)
    assert FZ.green_time(25, 0, g) > FZ.green_time(25, 20, g)
    assert FZ.G_LO <= FZ.green_time(0, 0, g) and FZ.green_time(40, 0, g) <= FZ.G_HI


def test_repair_keeps_bounds_and_order():
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = FZ.repair(rng.normal(20, 80, FZ.N_GENES))
        assert np.all(g >= FZ.GENE_LO) and np.all(g <= FZ.GENE_HI)
        for a, b in FZ.SORTED_TRIPLES:
            assert np.all(np.diff(g[a:b]) > 0)


def test_non_dominated_sort_and_crowding():
    F = np.array([[1, 5, 5], [2, 2, 2], [3, 3, 3], [5, 1, 5], [4, 4, 4]], float)
    fronts = ga.fast_non_dominated_sort(F)
    assert set(fronts[0]) == {0, 1, 3}
    assert fronts[1] == [2] and fronts[2] == [4]
    cd = ga.crowding_distance(F, fronts[0])
    assert np.isinf(cd).sum() >= 2


def test_hypervolume_of_simple_sets():
    ref = np.array([2.0, 2.0, 2.0])
    assert abs(ga.hypervolume3d(np.array([[1.0, 1.0, 1.0]]), ref) - 1.0) < 1e-9
    two = np.array([[1.0, 0.5, 1.0], [0.5, 1.0, 1.0]])
    assert abs(ga.hypervolume3d(two, ref) - (1.5 * 1.0 + 1.0 * 0.5) * 1.0) < 1e-9


def test_webster_delay_against_simulation_is_in_the_right_range():
    cal = W.calibrate(0.10)
    s, lost = cal["saturation_flow_cars_per_s"], cal["startup_lost_time_s"]
    plan = W.webster_plan([250] * 4, s, lost)
    sim_d = np.mean([sim.run(sim.CTRL_FIXED, plan["greens"], SC.custom_arrivals([250] * 4, k), seed=k)["mean_delay"] for k in range(8)])
    ana = W.webster_delay(250, plan["greens"][0], plan["cycle_s"], s, lost)
    assert 0.6 < sim_d / ana < 1.6, (sim_d, ana)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} tests passed")
