"""Smoke tests for the web application, run with Streamlit's own test harness (no browser needed).

They run the whole app many times with the widgets set to ordinary and to extreme values and fail on any exception.

Run:  python tests/test_app.py        (about a minute: the app is executed a few dozen times)
"""
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def fresh() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=180).run()


def by_label(widgets, label):
    for w in widgets:
        if w.label == label:
            return w
    raise AssertionError(f"widget '{label}' not found; have {[w.label for w in widgets]}")


def no_errors(at: AppTest, context: str = ""):
    assert not at.exception, f"{context}: {[e.value for e in at.exception]}"


def test_app_loads_without_errors():
    at = fresh()
    no_errors(at, "default page")
    assert len(at.tabs) == 5, [t.label for t in at.tabs]


def test_every_demand_pattern_runs():
    at = fresh()
    n = len(by_label(at.sidebar.selectbox, "Demand").options)
    assert n == 7, n                         # five fixed patterns, the random mix and the custom flows
    for i in range(n):
        by_label(at.sidebar.selectbox, "Demand").select_index(i).run()
        no_errors(at, f"demand option {i}")


def test_every_controller_runs():
    at = fresh()
    n = len(by_label(at.selectbox, "Controller").options)
    assert n >= 8, n
    for i in range(n):
        by_label(at.selectbox, "Controller").select_index(i).run()
        no_errors(at, f"controller option {i}")


def test_custom_demand_extremes():
    at = fresh()
    by_label(at.sidebar.selectbox, "Demand").select_index(6).run()          # the last option: custom flows
    no_errors(at, "custom demand")
    sliders = [s for s in at.sidebar.slider if s.label.endswith("(cars/h)")]
    assert len(sliders) == 4, [s.label for s in at.sidebar.slider]
    for s in sliders:                       # nobody arrives at all
        s.set_value(0)
    at.run()
    no_errors(at, "zero demand")
    for s in at.sidebar.slider:
        if s.label.endswith("(cars/h)"):
            s.set_value(700)
    by_label(at.sidebar.slider, "Demand multiplier").set_value(1.5)
    at.run()
    no_errors(at, "maximum demand")


def test_stress_settings_at_their_limits():
    at = fresh()
    by_label(at.sidebar.checkbox, "Rain (drivers dawdle more)").check()
    by_label(at.sidebar.slider, "Sensor noise (cars)").set_value(5.0)
    by_label(at.sidebar.slider, "Detector misses (%)").set_value(40)
    by_label(at.sidebar.number_input, "Random seed").set_value(9999)
    at.run()
    no_errors(at, "all stresses, seed 9999")
    by_label(at.sidebar.number_input, "Random seed").set_value(1)
    at.run()
    no_errors(at, "seed 1")


def test_compare_tab_with_one_and_twelve_streams():
    at = fresh()
    for n in (1, 12):
        by_label(at.slider, "Number of random traffic streams").set_value(n)
        at.run()
        no_errors(at, f"{n} streams")


def test_inspector_every_controller_and_slider_extremes():
    at = fresh()
    for opt in ("hand_tuned", "nsga2_knee", "nsga2_robust_knee", "nsga2_practical_knee", "nsga2_practical_robust_knee"):
        by_label(at.radio, "Controller whose rules to inspect").set_value(opt).run()
        no_errors(at, f"inspector '{opt}'")
        for label, lo_hi in (("Queue on the approach (cars)", (0, 40)), ("Seconds since this approach last had green", (0, 240)),
                             ("Queue of the chosen approach (cars)", (0, 40)), ("Pressure: mean queue of the other three", (0, 40))):
            for v in lo_hi:
                by_label(at.slider, label).set_value(v)
                at.run()
                no_errors(at, f"inspector '{opt}' {label}={v}")


def test_pareto_lab_member_can_be_used_as_controller():
    at = fresh()
    label = "Move along the front (left = lowest mean delay)"
    for v in (0, by_label(at.slider, label).max):
        by_label(at.slider, label).set_value(v)
        at.run()
        no_errors(at, f"front member {v}")
    next(b for b in at.button if b.label == "Use this controller in the Live junction tab").click().run()
    no_errors(at, "after saving a front member")
    options = list(by_label(at.selectbox, "Controller").options)
    pareto = [i for i, o in enumerate(options) if "Pareto" in o]
    assert pareto, options
    by_label(at.selectbox, "Controller").select_index(pareto[0]).run()
    no_errors(at, "Pareto member as controller")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} tests passed")
