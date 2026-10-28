"""GreenPulse: a fuzzy traffic signal tuned by NSGA-II, on a car-by-car junction model (web application).

Run from the project folder:   streamlit run app/app.py

Everything shown here is simulated; nothing is sent anywhere and no file outside this project is read.
"""
from __future__ import annotations

import html
import inspect
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from greenpulse import baselines as B
from greenpulse import fuzzy as FZ
from greenpulse import scenarios as SC
from greenpulse import sim
from greenpulse import webster as W

st.set_page_config(page_title="GreenPulse", page_icon="\U0001F6A6", layout="wide", initial_sidebar_state="auto")

RES = ROOT / "results"
INK, MUTED, GREEN, DEEP = "#22303c", "#66768a", "#1f7a5c", "#0f2d2a"
GRID = "#e6eeeb"
APPROACH_COL = ["#1f5fa5", "#e4572e", "#2a9d8f", "#e9b44c"]
TERM_COL = ["#2a9d8f", "#f2a93b", "#d64545"]
FONT = 'Inter, "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif'

# one entry per controller: key, label in the drop-downs, short name, colour
CONTROLLER_DEFS = [
    ("equal", "Fixed, equal split (22 s)", "Equal split", "#c3ccd5"),
    ("webster", "Fixed, Webster plan for this demand", "Webster plan", "#9aa5b1"),
    ("actuated", "Vehicle-actuated", "Actuated", "#e9b44c"),
    ("hand_tuned", "Fuzzy, hand-tuned", "Fuzzy hand-tuned", "#9db8d6"),
    ("nsga2_knee", "Fuzzy, NSGA-II tuned", "NSGA-II", "#1f5fa5"),
    ("nsga2_robust_knee", "Fuzzy, NSGA-II robust", "NSGA-II robust", "#d1495b"),
    ("nsga2_practical_knee", "Fuzzy, NSGA-II practical (green >= 8 s)", "NSGA-II practical", "#7b3fa0"),
    ("nsga2_practical_robust_knee", "Fuzzy, NSGA-II practical robust (green >= 8 s)", "NSGA-II practical robust", "#c8702a"),
]
PARETO_LABEL = "Fuzzy, Pareto member from the lab"


# --------------------------------------------------------------------------------------------- small helpers
def esc(x) -> str:
    return html.escape(str(x), quote=True)


def lst(a):
    return np.asarray(a).tolist()


def rgba(hex_colour: str, alpha: float) -> str:
    """Plotly wants rgba(...) for transparent fills, not 8-digit hex."""
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _stretch(fn) -> dict:
    """Streamlit renamed use_container_width to width="stretch"; use whichever the installed version knows."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


STRETCH_CHART, STRETCH_BTN = _stretch(st.plotly_chart), _stretch(st.button)
CHART_CONFIG = {"config": {"displayModeBar": False}} if "config" in inspect.signature(st.plotly_chart).parameters else {}


def show(fig, key=None):
    st.plotly_chart(fig, key=key, **STRETCH_CHART, **CHART_CONFIG)


def card():
    try:
        return st.container(border=True)
    except TypeError:                           # very old Streamlit
        return st.container()


def html_block(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def embed(page: str):
    """Show the junction animation. Newer Streamlit sizes the frame to its content (st.iframe); older versions need a fixed height."""
    if hasattr(st, "iframe"):
        st.iframe(page, height="content")
    else:
        components.html(page, height=700, scrolling=True)


def style_fig(fig, height=300, title=None, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=38 if title else 14, b=62 if legend else 8), title=dict(text=title, x=0.01, font=dict(size=14, color=DEEP)) if title else None,
        font=dict(family=FONT, size=12, color=INK), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"), hoverlabel=dict(bgcolor="#ffffff", font_size=12, bordercolor="#d7e2de"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor="#cfdad6", ticks="outside", tickcolor="#cfdad6", title_font=dict(size=11, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", title_font=dict(size=11, color=MUTED))
    return fig


ICONS = {
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "gauge": '<path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-6"/>',
    "stop": '<path d="M8 3h8l5 5v8l-5 5H8l-5-5V8z"/>',
    "car": '<path d="M5 16l1.5-5a2 2 0 0 1 1.9-1.4h7.2a2 2 0 0 1 1.9 1.4L19 16"/><rect x="3" y="16" width="18" height="4" rx="1.5"/><circle cx="7.5" cy="20" r="1"/><circle cx="16.5" cy="20" r="1"/>',
    "queue": '<path d="M4 7h16M4 12h12M4 17h8"/>',
    "check": '<path d="M5 12.5l4.2 4.2L19 7"/>',
}


def icon(name: str) -> str:
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{ICONS[name]}</svg>'


LOGO = ('<svg width="40" height="40" viewBox="0 0 64 64" aria-hidden="true"><rect x="19" y="3" width="26" height="58" rx="9" fill="#0f2d2a"/>'
        '<circle cx="32" cy="17" r="6.5" fill="#5b2a2a"/><circle cx="32" cy="32" r="6.5" fill="#5b4a1f"/><circle cx="32" cy="47" r="6.5" fill="#34c37f"/>'
        '<circle cx="32" cy="47" r="10" fill="#34c37f" opacity=".22"/></svg>')

HERO_ART = ('<svg width="190" height="120" viewBox="0 0 190 120" aria-hidden="true"><defs><filter id="gl" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="3.4" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
            '<rect x="6" y="46" width="178" height="30" rx="6" fill="#26343c"/><rect x="80" y="2" width="30" height="116" rx="6" fill="#26343c"/>'
            '<path d="M10 61h170M95 6v108" stroke="#f3d46b" stroke-width="1.6" stroke-dasharray="7 6" fill="none" opacity=".9"/>'
            '<rect x="80" y="46" width="30" height="30" fill="#33434d"/>'
            '<rect x="22" y="50" width="22" height="9" rx="3.5" fill="#2e9e6b"/><rect x="54" y="50" width="22" height="9" rx="3.5" fill="#f2a93b"/>'
            '<rect x="116" y="63" width="22" height="9" rx="3.5" fill="#d64545"/><rect x="146" y="63" width="22" height="9" rx="3.5" fill="#d64545"/>'
            '<rect x="85" y="14" width="9" height="20" rx="3.5" fill="#2e9e6b"/><rect x="97" y="88" width="9" height="20" rx="3.5" fill="#d64545"/>'
            '<rect x="152" y="14" width="22" height="48" rx="8" fill="#0b1f1c"/><circle cx="163" cy="26" r="5.2" fill="#6b2d2d"/><circle cx="163" cy="38" r="5.2" fill="#6b5420"/>'
            '<circle cx="163" cy="50" r="5.2" fill="#34c37f" filter="url(#gl)"/></svg>')


# --------------------------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_results():
    ctl = {"hand_tuned": FZ.HAND_TUNED.tolist()}
    for f in ("controllers.json", "controllers_robust.json", "controllers_practical.json", "controllers_practical_robust.json"):
        if (RES / f).exists():
            ctl.update(json.loads((RES / f).read_text(encoding="utf-8")))
    front = json.loads((RES / "tuning.json").read_text(encoding="utf-8"))["selected_front"] if (RES / "tuning.json").exists() else None
    ev = json.loads((RES / "evaluation.json").read_text(encoding="utf-8")) if (RES / "evaluation.json").exists() else None
    return ctl, front, ev


@st.cache_data(show_spinner=False)
def calibration():
    c = W.calibrate(0.10)
    return {k: v for k, v in c.items() if k != "mean_departures"}


CTL, FRONT, EVAL = load_results()
if "nsga2_knee" not in CTL:
    st.error("The tuned controllers are missing. Run `python experiments/11_tune.py 80 80` first, or restore the `results` folder.")
    st.stop()
CAL = calibration()

DEFS = [d for d in CONTROLLER_DEFS if d[0] in ("equal", "webster", "actuated") or d[0] in CTL]
SHORT = {d[0]: d[2] for d in DEFS} | {"pareto": "Pareto member"}
COLOR = {d[0]: d[3] for d in DEFS} | {"pareto": "#2a9d8f"}


def build_controller(key, demand, scale, pareto_genes=None):
    if key == "equal":
        return B.equal_plan()
    if key == "webster":
        plan = W.webster_plan(np.asarray(demand, float) * scale, CAL["saturation_flow_cars_per_s"], CAL["startup_lost_time_s"])
        return sim.CTRL_FIXED, plan["greens"]
    if key == "actuated":
        return B.actuated_plan()
    if key == "pareto":
        return sim.CTRL_FUZZY, np.array(pareto_genes, dtype=float)
    return sim.CTRL_FUZZY, np.array(CTL[key], dtype=float)


def traffic(preset, demand, scale, seed):
    if preset == "custom":
        return SC.custom_arrivals(np.asarray(demand, float) * scale, int(seed))
    return SC.make_arrivals(preset, int(seed), scale)


@st.cache_data(show_spinner=False, max_entries=48)
def run_one(key, preset, demand, scale, seed, rain, sigma, drop, rec_T, pareto_genes):
    ctrl, par = build_controller(key, demand, scale, pareto_genes)
    return sim.run(ctrl, par, traffic(preset, demand, scale, seed), seed=int(seed), p_slow=0.25 if rain else 0.10, sigma_q=float(sigma), dropout=float(drop), rec_T=int(rec_T))


@st.cache_data(show_spinner="Running every controller on the same traffic ...", max_entries=16)
def compare_all(keys, preset, demand, scale, seed, n_seeds, rain, sigma, drop, pareto_genes):
    rows = []
    for key in keys:
        vals = {m: [] for m in ("mean_delay", "p95_delay", "mean_stops", "throughput")}
        for s in range(int(seed), int(seed) + int(n_seeds)):
            ctrl, par = build_controller(key, demand, scale, pareto_genes)
            r = sim.run(ctrl, par, traffic(preset, demand, scale, s), seed=s, p_slow=0.25 if rain else 0.10, sigma_q=float(sigma), dropout=float(drop))
            for m in vals:
                vals[m].append(r[m])
        rows.append((key, {m: float(np.mean(v)) for m, v in vals.items()}))
    return rows


def change(value, base):
    """Percent change of value against base, or None when the reference is zero."""
    return None if base is None or abs(base) < 1e-9 else 100.0 * (value - base) / base


def chip(pct, lower_is_better=True, ref="actuated"):
    if pct is None:
        return ""
    if abs(pct) < 0.5:
        return f'<span class="chip neutral">same as {esc(ref)}</span>'
    good = (pct < 0) == lower_is_better
    arrow = "&#9660;" if pct < 0 else "&#9650;"
    return f'<span class="chip {"good" if good else "bad"}">{arrow} {abs(pct):.0f}% vs {esc(ref)}</span>'


# --------------------------------------------------------------------------------------------- styling
html_block("<style>" + (ROOT / "app" / "style.css").read_text(encoding="utf-8") + "</style>")


def new_traffic():
    st.session_state["seed"] = random.randint(1, 9999)


def use_member(genes, number):
    st.session_state["pareto_genes"] = tuple(float(x) for x in genes)
    st.session_state["pareto_no"] = int(number)


# --------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    html_block(f'<div class="brand">{LOGO}<div><b>GreenPulse</b><span>Fuzzy + NSGA-II traffic signal<br>on a car-by-car junction</span></div></div>')
    html_block('<div class="sb-h">Traffic</div>')
    preset = st.selectbox("Demand", list(SC.SCENARIOS) + ["random_mix", "custom"], index=1, format_func=lambda k: SC.LABELS.get(k, "Custom demand"))
    caption_slot = st.empty()
    if preset == "custom":
        demand = [st.slider(f"{n} (cars/h)", 0, 700, v, 10) for n, v in zip(sim.APPROACH, (300, 200, 300, 200))]
    scale = st.slider("Demand multiplier", 0.5, 1.5, 1.0, 0.05)
    seed = int(st.number_input("Random seed", 1, 9999, 1001, key="seed"))
    st.button("New random traffic", on_click=new_traffic, **STRETCH_BTN)
    if preset != "custom":
        demand = SC.mean_demand(preset, seed=seed).round(0).tolist()
        caption_slot.caption("Cars/h (N, E, S, W, time-average): " + ", ".join(f"{int(x)}" for x in demand))
    html_block('<div class="sb-h">Conditions</div>')
    with st.expander("Stress the junction"):
        rain = st.checkbox("Rain (drivers dawdle more)")
        sigma_q = st.slider("Sensor noise (cars)", 0.0, 5.0, 0.0, 0.5)
        dropout = st.slider("Detector misses (%)", 0, 40, 0, 5) / 100
    html_block('<div class="sb-foot">Soft Computing project<br>Amity School of Engineering and Technology<br>Sahil Yadav &middot; all traffic is simulated</div>')

demand_t = tuple(float(x) for x in demand)
pareto_t = st.session_state.get("pareto_genes")

CONTROLLERS = {d[1]: d[0] for d in DEFS}
if pareto_t is not None:
    CONTROLLERS[PARETO_LABEL] = "pareto"


def run_key(key, rec_T=0):
    return run_one(key, preset, demand_t, scale, seed, rain, sigma_q, dropout, rec_T, pareto_t if key == "pareto" else None)


# --------------------------------------------------------------------------------------------- header
html_block(
    '<div class="hero"><div><div class="eyebrow">Soft computing project</div><h1>GreenPulse</h1>'
    '<p>A traffic signal that decides from queues and waiting time with fuzzy rules, tuned by a genetic algorithm.</p>'
    '<span class="pill">Fuzzy logic</span><span class="pill">NSGA-II</span><span class="pill">Car-by-car simulation</span><span class="pill">Checked against Webster</span></div>'
    + HERO_ART + "</div>")
ctx = [("Demand", SC.LABELS.get(preset, "Custom demand")), ("Multiplier", f"x{scale:.2f}"), ("Seed", seed), ("Weather", "rain" if rain else "dry"),
       ("Sensors", f"noise {sigma_q:.1f}, {int(dropout * 100)}% missed" if (sigma_q or dropout) else "clean")]
html_block('<div class="ctx">' + "".join(f'<span>{esc(k)} <b>{esc(v)}</b></span>' for k, v in ctx) + "</div>")

tab_live, tab_compare, tab_inspect, tab_pareto, tab_about = st.tabs(["Live junction", "Compare controllers", "Fuzzy inspector", "Pareto lab", "About"])


# --------------------------------------------------------------------------------------------- tab 1: live junction
def kpi(ic, tone, value, unit, label, chip_html=""):
    return (f'<div class="kpi t-{tone}"><div class="ic">{icon(ic)}</div><div class="val">{esc(value)}<small>{esc(unit)}</small></div>'
            f'<div class="lab">{esc(label)}</div>{chip_html}</div>')


def green_timeline(signal):
    """Run-length encode the signal into (start, length, approach) for the Gantt-style chart."""
    runs, cur, start = [], None, 0
    for i, s in enumerate(list(signal) + [-2]):
        if s != cur:
            if cur is not None and cur >= 0:
                runs.append((start, i - start, int(cur)))
            cur, start = s, i
    return runs


with tab_live:
    c1, c2 = st.columns([3, 2])
    default_index = list(CONTROLLERS).index("Fuzzy, NSGA-II tuned") if "Fuzzy, NSGA-II tuned" in CONTROLLERS else 0
    label = c1.selectbox("Controller", list(CONTROLLERS), index=default_index)
    window = c2.slider("Animated minutes", 2, 15, 8)
    key = CONTROLLERS[label]
    out = run_key(key, window * 60)
    ref = run_key("actuated")
    ref_name = "actuated"
    is_ref = key == "actuated"

    def d(metric, lower=True):
        return "" if is_ref else chip(change(out[metric], ref[metric]), lower, ref_name)

    cards = [
        kpi("clock", "green", f"{out['mean_delay']:.1f}", " s", "mean delay per car", d("mean_delay")),
        kpi("gauge", "blue", f"{out['p95_delay']:.0f}", " s", "95th-percentile delay", d("p95_delay")),
        kpi("stop", "amber", f"{out['mean_stops']:.2f}", "", "stops per car", d("mean_stops")),
        kpi("car", "green", f"{out['throughput']:.0f}", "", "cars served per hour", d("throughput", lower=False)),
        kpi("queue", "red", f"{out['max_queue']:.0f}", "", "longest queue (cars)", d("max_queue")),
        kpi("check", "blue", f"{int(out['unserved'])}", "", "cars still waiting after 75 min", ""),
    ]
    html_block('<div class="kpis">' + "".join(cards) + "</div>")
    if is_ref:
        html_block('<div class="legend-row">This is the reference controller; the other controllers are compared against it.</div>')

    T = int(window * 60)
    occ = out["occupancy"][:T]
    digits = ["".join(chr(48 + int(v)) for v in row) for row in occ.reshape(T * 4, sim.L)]
    frames = [[digits[t * 4 + a] for a in range(4)] for t in range(T)]
    data = {"L": sim.L, "occ": frames, "sig": [int(x) for x in out["signal"][:T]], "q": [[int(v) for v in row] for row in out["queues"][:T]]}
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")        # numbers and digits only; "<" is escaped as a precaution
    page = (ROOT / "app" / "anim.html").read_text(encoding="utf-8").replace("__DATA__", payload).replace("__LAST__", str(max(T - 1, 0)))
    embed(page)

    left, right = st.columns(2)
    q = out["queues"]
    with left, card():
        fig = go.Figure()
        for a in range(4):
            col = APPROACH_COL[a]
            fig.add_trace(go.Scatter(x=lst(np.arange(len(q))), y=lst(q[:, a]), name=sim.APPROACH[a], mode="lines", line=dict(color=col, width=1.8),
                                     fill="tozeroy", fillcolor=rgba(col, 0.11)))
        fig.add_vrect(x0=0, x1=T, fillcolor="#1f7a5c", opacity=0.05, line_width=0)
        style_fig(fig, 290, "Cars waiting on each approach, whole hour")
        fig.update_xaxes(title_text="time (s)")
        fig.update_yaxes(title_text="cars")
        show(fig, key="queues")
    with right, card():
        fig = go.Figure()
        runs = green_timeline(out["signal"])
        for a in range(4):
            rs = [r for r in runs if r[2] == a]
            fig.add_trace(go.Bar(x=[r[1] for r in rs], y=[sim.APPROACH[a]] * len(rs), base=[r[0] for r in rs], orientation="h", name=sim.APPROACH[a],
                                 marker=dict(color=APPROACH_COL[a], line=dict(width=0)), hovertemplate="%{y}: green from %{base:.0f} s for %{x:.0f} s<extra></extra>"))
        style_fig(fig, 290, "Who has green, whole hour")
        fig.update_layout(barmode="overlay", bargap=0.35)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title_text="time (s)", range=[0, len(q)])
        show(fig, key="timeline")

# --------------------------------------------------------------------------------------------- tab 2: compare
with tab_compare:
    n_seeds = st.slider("Number of random traffic streams", 1, 12, 6)
    st.caption("Every controller sees exactly the same cars (common random numbers). The settings in the sidebar apply.")
    keys = tuple(CONTROLLERS.values())
    rows = compare_all(keys, preset, demand_t, scale, seed, int(n_seeds), rain, sigma_q, dropout, pareto_t)
    base = dict(rows)["actuated"]
    ranked = sorted(rows, key=lambda r: r[1]["mean_delay"])
    worst = max(r[1]["mean_delay"] for r in rows) or 1.0
    body = []
    for i, (k, v) in enumerate(ranked):
        pct = change(v["mean_delay"], base["mean_delay"])
        best = i == 0 and v["mean_delay"] < worst
        name = esc(next((lab for lab, kk in CONTROLLERS.items() if kk == k), k))
        chip_html = '<span class="chip neutral">reference</span>' if k == "actuated" else chip(pct, True, "actuated")
        badge = '<span class="best-badge">BEST</span>' if best else ""
        body.append(f'<tr class="{"best" if best else ""}"><td>{i + 1}</td><td><span class="dot" style="background:{COLOR[k]}"></span><span class="name">{name}</span>'
                    f'{badge}</td><td class="num"><b>{v["mean_delay"]:.1f}</b> s</td>'
                    f'<td>{chip_html}</td><td class="num">{v["p95_delay"]:.1f}</td><td class="num">{v["mean_stops"]:.2f}</td><td class="num">{v["throughput"]:.0f}</td></tr>')
    html_block('<table class="rank"><thead><tr><th></th><th>Controller</th><th style="text-align:right">Mean delay</th><th>vs actuated</th><th style="text-align:right">p95 (s)</th>'
               '<th style="text-align:right">Stops</th><th style="text-align:right">Cars/h</th></tr></thead><tbody>' + "".join(body) + "</tbody></table>")
    with card():
        fig = go.Figure(go.Bar(y=[next(l for l, kk in CONTROLLERS.items() if kk == k) for k, _ in ranked][::-1], x=[round(v["mean_delay"], 1) for _, v in ranked][::-1], orientation="h",
                               marker=dict(color=[COLOR[k] for k, _ in ranked][::-1], line=dict(width=0)), text=[f"{v['mean_delay']:.1f} s" for _, v in ranked][::-1], textposition="outside", cliponaxis=False))
        style_fig(fig, 60 + 38 * len(ranked), "Average delay per car", legend=False)
        fig.update_layout(margin=dict(l=8, r=40, t=38, b=8), bargap=0.3)
        fig.update_xaxes(title_text="seconds", rangemode="tozero")
        show(fig, key="compare")
    if EVAL:
        def gain(focus, ref_name_):
            hit = [r for r in EVAL["paired"] if r.get("focus", "Fuzzy + NSGA-II") == focus and r["ref"] == ref_name_ and r["scenario"] == "ALL" and r["metric"] == "mean_delay"]
            return abs(hit[0]["change_pct"]) if hit else None
        g_act, g_fix = gain("Fuzzy + NSGA-II", "Vehicle-actuated"), gain("Fuzzy + NSGA-II", "Best fixed plan")
        g_pra, g_prr = gain("Fuzzy + NSGA-II (practical)", "Vehicle-actuated"), gain("Fuzzy + NSGA-II (practical, robust)", "Vehicle-actuated")
        if g_act is not None and g_fix is not None:
            msg = (f"<b>Held-out results from the report</b> (20 seeds x 6 demand patterns): the NSGA-II controller, which may use greens as short as 4 s, cuts mean delay by "
                   f"<b>{g_act:.0f}%</b> against the vehicle-actuated controller and <b>{g_fix:.0f}%</b> against the best fixed plan.")
            if g_pra is not None and g_prr is not None:
                msg += (f" With a minimum green of 8 s, which is closer to a real signal, the gain over the actuated controller is <b>{g_pra:.0f}%</b> "
                        f"(<b>{g_prr:.0f}%</b> when the controller is also tuned on stressed traffic).")
            html_block(f'<div class="note">{msg}</div>')

# --------------------------------------------------------------------------------------------- tab 3: fuzzy inspector
INSPECT = {"hand_tuned": "Hand-tuned", "nsga2_knee": "NSGA-II tuned", "nsga2_robust_knee": "NSGA-II robust",
           "nsga2_practical_knee": "NSGA-II practical", "nsga2_practical_robust_knee": "NSGA-II practical robust"}


def rules_html(rules, top=4):
    shown = [r for r in sorted(rules, key=lambda r: -r["strength"]) if r["strength"] > 0.01][:top]
    if not shown:
        return '<div class="legend-row">No rule fires.</div>'
    out_ = []
    for r in shown:
        txt = esc(r["rule"]).replace("IF ", "<b>IF</b> ").replace(" AND ", " <b>AND</b> ").replace(" THEN ", " <b>THEN</b> ")
        out_.append(f'<div class="rule"><div class="txt">{txt}<div class="st"><i style="width:{100 * r["strength"]:.0f}%"></i></div></div><div class="pc">{r["strength"]:.2f}</div></div>')
    return "".join(out_)


def gauge_html(value, unit, lo, hi):
    pct = 100 * (value - lo) / (hi - lo)
    return f'<div class="gauge"><div class="num">{value:.1f}<small> {esc(unit)}</small></div><div class="track"><i style="width:{max(2, min(100, pct)):.0f}%"></i></div></div>'


def membership_figure(name, x, mu, terms, marker=None):
    fig = go.Figure()
    for j, term in enumerate(terms):
        col = TERM_COL[j]
        fig.add_trace(go.Scatter(x=lst(x), y=lst(mu[j]), name=term, mode="lines", line=dict(color=col, width=2.4), fill="tozeroy", fillcolor=rgba(col, 0.14)))
    if marker is not None:
        fig.add_vline(x=marker, line_dash="dash", line_color="#22303c", line_width=1.4)
    style_fig(fig, 215, name)
    fig.update_layout(legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=10)), margin=dict(l=6, r=6, t=34, b=6))
    fig.update_yaxes(range=[0, 1.08], tickvals=[0, 0.5, 1])
    return fig


with tab_inspect:
    names = [k for k in INSPECT if k in CTL]
    which = st.radio("Controller whose rules to inspect", names, format_func=lambda k: INSPECT[k], horizontal=True)
    g = np.array(CTL[which], dtype=float)
    a, b = st.columns(2)
    with a, card():
        html_block('<div class="card-h">Stage 1</div><div class="card-t">Which approach is most urgent?</div>')
        q1 = st.slider("Queue on the approach (cars)", 0, 40, 12)
        w1 = st.slider("Seconds since this approach last had green", 0, 240, 70)
        e1 = FZ.explain_urgency(float(q1), float(w1), g)
        html_block('<div class="legend-row">Urgency (0 to 100): the most urgent approach gets the next green</div>' + gauge_html(e1["urgency"], "", 0, 100)
                   + '<div class="legend-row">Rules that fired (strength)</div>' + rules_html(e1["rules"]))
    with b, card():
        html_block('<div class="card-h">Stage 2</div><div class="card-t">How long may the chosen approach keep green?</div>')
        q2 = st.slider("Queue of the chosen approach (cars)", 0, 40, 14)
        p2 = st.slider("Pressure: mean queue of the other three", 0, 40, 6)
        e2 = FZ.explain_green(float(q2), float(p2), g)
        html_block('<div class="legend-row">Green-time cap in seconds: the green also ends early when no car is near the stop line</div>' + gauge_html(e2["green_s"], "s", FZ.G_LO, FZ.G_HI)
                   + '<div class="legend-row">Rules that fired (strength)</div>' + rules_html(e2["rules"]))
        st.caption(f"Minimum green {g[18]:.0f} s. Gap-out after {g[19]:.1f} s with no car near the stop line. The cap rarely binds when the junction is not stressed.")
    curves = FZ.variable_curves(g)
    markers = {"queue (urgency FIS)": q1, "time since last green, s": w1, "queue of chosen approach": q2, "pressure from other approaches": p2}
    cols = st.columns(3)
    for i, (name, (x, mu, terms)) in enumerate(curves.items()):
        with cols[i % 3], card():
            show(membership_figure(name, x, mu, terms, markers.get(name)), key=f"mf{i}")

# --------------------------------------------------------------------------------------------- tab 4: Pareto lab
with tab_pareto:
    if FRONT is None:
        html_block('<div class="note warn">Run <b>experiments/11_tune.py</b> to create the Pareto front.</div>')
    else:
        obj = np.array(FRONT["objectives"])
        genes_all = np.array(FRONT["genes"])
        order = np.argsort(obj[:, 0])
        test = np.array(EVAL["front"]["test"]) if EVAL else None
        act_test = EVAL["front"]["actuated_test"] if EVAL else None
        html_block('<div class="note">Each dot is one complete controller. NSGA-II returns the whole trade-off between mean delay, 95th-percentile delay and stops, not one answer. '
                   'This is the front of the main tuning, where greens as short as 4 s are allowed; the fronts for a minimum green of 8 s are in the report (Figure 7.12).</div>')
        knee_pos = int(np.where(order == FRONT["knee_index"])[0][0])
        idx = st.slider("Move along the front (left = lowest mean delay)", 0, len(order) - 1, knee_pos)
        sel = int(order[idx])
        pa, pb = st.columns([3, 2])
        with pa, card():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lst(obj[:, 0]), y=lst(obj[:, 1]), mode="markers", name="front (training seeds)",
                                     marker=dict(size=10, color=lst(obj[:, 2]), colorscale=[[0, "#2a9d8f"], [0.5, "#f2a93b"], [1, "#d64545"]], line=dict(color="#fff", width=1),
                                                 colorbar=dict(title=dict(text="stops / car", side="right"), thickness=10, len=0.8)),
                                     hovertemplate="mean delay %{x:.1f} s<br>95th pct %{y:.0f} s<extra></extra>"))
            if act_test:
                fig.add_trace(go.Scatter(x=[act_test[0]], y=[act_test[1]], mode="markers", name="actuated (test)", marker=dict(size=13, symbol="square", color="#e9b44c", line=dict(color="#22303c", width=1))))
            fig.add_trace(go.Scatter(x=[float(obj[sel, 0])], y=[float(obj[sel, 1])], mode="markers", name="selected", marker=dict(size=20, symbol="star", color="#d64545", line=dict(color="#fff", width=1.5))))
            style_fig(fig, 400, "The NSGA-II front")
            fig.update_xaxes(title_text="mean delay (s)")
            fig.update_yaxes(title_text="95th-percentile delay (s)")
            show(fig, key="front")
        with pb, card():
            html_block(f'<div class="card-h">Selected controller</div><div class="card-t">Front member {idx + 1} of {len(order)}{" (knee point)" if idx == knee_pos else ""}</div>')
            html_block('<div class="kpis mini">'
                       + kpi("clock", "green", f"{obj[sel, 0]:.1f}", " s", "mean delay (training)")
                       + kpi("gauge", "blue", f"{obj[sel, 1]:.0f}", " s", "95th-percentile delay")
                       + kpi("stop", "amber", f"{obj[sel, 2]:.2f}", "", "stops per car") + "</div>")
            if test is not None:
                t_ = test[sel]
                beats = bool(act_test) and all(t_[i] < act_test[i] for i in range(3))
                html_block(f'<div class="legend-row">On held-out traffic: {t_[0]:.1f} s mean, {t_[1]:.0f} s at the 95th percentile, {t_[2]:.2f} stops.</div>'
                           + '<div style="margin-bottom:12px">' + ('<span class="chip good">beats the actuated controller on all three</span>' if beats else '<span class="chip neutral">trade-off point</span>') + "</div>")
            st.button("Use this controller in the Live junction tab", on_click=use_member, args=(genes_all[sel], idx + 1), **STRETCH_BTN)
            if st.session_state.get("pareto_no"):
                st.success(f"Saved front member {st.session_state['pareto_no']}. Pick '{PARETO_LABEL}' in the Live junction or Compare tab.")

# --------------------------------------------------------------------------------------------- tab 5: about
with tab_about:
    tiles = [
        ("What it is", "A four-way junction with one lane and one phase per approach. Cars are simulated one by one with the Nagel-Schreckenberg cellular-automaton rule (7.5 m cells, 1 s steps). The controller sees only queue lengths and waiting times."),
        ("The fuzzy controller", "FIS-1 turns (queue, time since last green) into an urgency and the most urgent approach gets green next. FIS-2 turns (queue, pressure from the other approaches) into a cap on that green. Both are Mamdani systems with three terms per input and nine rules. A green also ends early when no car is near the stop line (gap-out)."),
        ("The tuning", "NSGA-II searches 20 numbers (membership centres, minimum green, gap-out time) to minimise mean delay, 95th-percentile delay and stops per car on simulated training traffic."),
        ("Validation", f"Saturation flow {CAL['saturation_flow_cars_per_h']:.0f} cars/h; the fundamental diagram peaks near 47 cars/km; simulated delay matches Webster's formula within about 9%."),
        ("Realistic greens", "The main controllers may use greens as short as 4 s, which costs almost nothing in this model but cannot be shown by a real signal. The two practical controllers were tuned with a minimum green of at least 8 s; their gain over the actuated controller is smaller (see the note in the Compare tab) and their tail delay is longer, which is the number to quote."),
        ("Limits", "Everything is simulated: one lane per approach, no turns, no pedestrians. The numbers say how the controllers compare in this model, not how many seconds a real junction would save."),
    ]
    html_block('<div class="grid2">' + "".join(f'<div class="tile"><h4>{esc(h)}</h4><p>{esc(p)}</p></div>' for h, p in tiles) + "</div>")

html_block('<div class="foot">GreenPulse &middot; Soft Computing project &middot; Amity School of Engineering and Technology, Noida &middot; Sahil Yadav</div>')
