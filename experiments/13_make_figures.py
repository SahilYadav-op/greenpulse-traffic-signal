"""Step 13: all figures of the traffic-signal report and slides, drawn from the saved results.

Usage:  python -I experiments/13_make_figures.py
"""
import json
import site
import sys
from pathlib import Path

sys.path.append(site.getusersitepackages())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

from greenpulse import fuzzy as FZ
from greenpulse import scenarios as SC
from greenpulse import sim

OUT = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
EV = json.loads((OUT / "evaluation.json").read_text())
TU = json.loads((OUT / "tuning.json").read_text())
TR = json.loads((OUT / "tuning_robust.json").read_text())
VA = json.loads((OUT / "validation.json").read_text())
CT = json.loads((OUT / "controllers.json").read_text())
CT.update(json.loads((OUT / "controllers_robust.json").read_text()))
TRACES = np.load(OUT / "eval_traces.npz")

INK, MUTED, GRID = "#22303c", "#6b7a89", "#dfe5ea"
COL = {"Fixed equal split": "#c3ccd5", "Webster formula": "#9aa5b1", "Best fixed plan": "#6b7a89", "Vehicle-actuated": "#e9b44c",
       "Fuzzy (hand-tuned)": "#9db8d6", "Fuzzy + random search": "#c9aedb", "Fuzzy + GA": "#2a9d8f", "Fuzzy + NSGA-II": "#1f5fa5",
       "Fuzzy + NSGA-II (robust)": "#d1495b", "Fuzzy + NSGA-II (practical)": "#7b3fa0", "Fuzzy + NSGA-II (practical, robust)": "#c8702a"}
SHORT = {"Fixed equal split": "Equal split", "Webster formula": "Webster", "Best fixed plan": "Best fixed", "Vehicle-actuated": "Actuated",
         "Fuzzy (hand-tuned)": "Fuzzy hand", "Fuzzy + random search": "Fuzzy + random", "Fuzzy + GA": "Fuzzy + GA", "Fuzzy + NSGA-II": "Fuzzy + NSGA-II",
         "Fuzzy + NSGA-II (robust)": "NSGA-II robust", "Fuzzy + NSGA-II (practical)": "NSGA-II practical", "Fuzzy + NSGA-II (practical, robust)": "NSGA-II pract. robust"}
GREEN, AMBER, RED = "#2e9e6b", "#f2a93b", "#d64545"


def lab(kd):
    """Legend label that tells the clean-tuned NSGA-II controller apart from the robust and practical ones."""
    return SHORT[kd].replace("Fuzzy + NSGA-II", "NSGA-II clean")

APP_COL = ["#1f5fa5", "#e4572e", "#2a9d8f", "#e9b44c"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight", "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("saved", name)


M = EV["main"]
SC_ALL = EV["scenarios"]
PRA, PRR = "Fuzzy + NSGA-II (practical)", "Fuzzy + NSGA-II (practical, robust)"
HAVE_PRA = PRA in M and PRR in M


def mean_of(kind, scen, metric="mean_delay"):
    return M[kind][scen][metric]["mean"]


# ------------------------------------------------------------------------------------------ T1 architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)

    def box(x, y, w, h, title, sub="", fc="#eaf1f8", ec="#1f5fa5", fs=7.6, sfs=6.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), title, ha="center", va="center", fontsize=fs, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=sfs, color=MUTED, linespacing=1.2)

    def arrow(a, b, c=INK, ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=c, lw=1.2, linestyle=ls))

    ax.text(1, 51.5, "THE CONTROLLER IN OPERATION", fontsize=7.5, color=MUTED, fontweight="bold")
    y0, h0 = 33, 14
    box(1, y0, 19, h0, "Detectors", "queue q (cars)\nwait w (s since green)", "#f1f5f8", "#6b7a89")
    box(24, y0, 21, h0, "FIS-1: urgency", "q, w  ->  urgency 0-100\n9 Mamdani rules", "#eaf1f8", "#1f5fa5")
    box(49, y0, 17, h0, "Pick approach", "highest urgency;\ngap-out ends green", "#f1f5f8", "#6b7a89")
    box(70, y0, 21, h0, "FIS-2: green time", "q, pressure  ->  5-60 s\n9 Mamdani rules", "#eaf1f8", "#1f5fa5")
    box(94.5, y0, 5, h0, "", "", "#e8f5f2", GREEN)
    ax.text(97, y0 + h0 / 2, "Sig-\nnal", ha="center", va="center", fontsize=6.6, fontweight="bold")
    for x0, x1 in ((20.4, 23.4), (45.4, 48.4), (66.4, 69.4), (91.4, 94.0)):
        arrow((x0, y0 + h0 / 2), (x1, y0 + h0 / 2))
    box(24, 16, 42, 10, "Junction: car-by-car cellular automaton", "4 approaches, 40 cells each, Nagel-Schreckenberg rule, 1 s steps", "#fff7e8", AMBER, sfs=6.2)
    arrow((97, y0 - 0.5), (66.6, 22.5))
    ax.text(86, 31.2, "green for one approach", fontsize=6.6, color=MUTED, style="italic", ha="center")
    arrow((24.2, 21), (10.5, y0 - 0.5))
    ax.text(8, 27, "measure", fontsize=6.8, color=MUTED, style="italic")
    ax.text(1, 12.2, "HOW THE CONTROLLER IS TUNED", fontsize=7.5, color=MUTED, fontweight="bold")
    box(1, 1, 26, 9.5, "NSGA-II", "pop 80, 80 generations\nSBX + polynomial mutation", "#fdf0ea", "#e4572e")
    box(35, 1, 30, 9.5, "20 genes", "membership centres of both FIS,\nminimum green, gap-out time", "#fdf0ea", "#e4572e")
    box(73, 1, 26, 9.5, "3 objectives", "mean delay, 95th-percentile\ndelay, stops per car", "#fdf0ea", "#e4572e")
    arrow((27.4, 5.8), (34.6, 5.8), "#e4572e")
    arrow((72.6, 5.8), (65.4, 5.8), "#e4572e")
    arrow((50, 11), (50, 15.5), "#e4572e")
    ax.text(51.2, 12.9, "controller under test", fontsize=6.6, color="#e4572e")
    arrow((86, 11), (66.5, 17.5), "#e4572e", ls="--")
    ax.text(86.5, 13.2, "20 simulated hours", fontsize=6.6, color="#e4572e")
    save(fig, "t01_architecture")


# ------------------------------------------------------------------------------------------ T2 junction snapshot
def draw_junction(ax, occ, sig_now, queues, L_SHOW=22):
    ang = {0: 90, 1: 0, 2: 270, 3: 180}
    names = ["North", "East", "South", "West"]
    ax.add_patch(Rectangle((-2.2, -2.2), 4.4, 4.4, fc="#dfe6ec", ec=MUTED, lw=0.8))
    for a in range(4):
        phi = np.deg2rad(ang[a])
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        lane = np.array([[2.2, -1.9], [2.2 + L_SHOW, -1.9], [2.2 + L_SHOW, -0.2], [2.2, -0.2]]) @ R.T
        ax.add_patch(plt.Polygon(lane, closed=True, fc="#f4f6f8", ec="#c3ccd5", lw=0.7))
        for c in range(L_SHOW):
            v1 = occ[a, c]
            if v1 > 0:
                v = v1 - 1
                col = RED if v == 0 else (AMBER if v == 1 else GREEN)
                ctr = np.array([2.2 + c + 0.5, -1.05]) @ R.T
                w, h = 0.86, 1.0
                rect = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ R.T + ctr
                ax.add_patch(plt.Polygon(rect, closed=True, fc=col, ec="white", lw=0.4))
        stop = np.array([[2.2, -1.95], [2.2, -0.15]]) @ R.T
        ax.plot(stop[:, 0], stop[:, 1], color=INK, lw=2.2)
        sp = np.array([1.2, -1.05]) @ R.T
        ax.add_patch(Circle(sp, 0.62, fc=GREEN if sig_now == a else RED, ec="white", lw=1.0, zorder=5))
        lab = np.array([2.2 + L_SHOW * 0.55, 0.9]) @ R.T
        n = int(queues[a])
        ax.text(lab[0], lab[1], f"{names[a]}: {n} car{'s' if n != 1 else ''}", ha="center", va="center", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c3ccd5", lw=0.6))
    ax.set_aspect("equal")
    ax.set_xlim(-L_SHOW - 3.5, L_SHOW + 3.5)
    ax.set_ylim(-L_SHOW - 3.5, L_SHOW + 3.5)
    ax.axis("off")


def fig_junction():
    arr = SC.make_arrivals("balanced", 1003)
    r = sim.run(sim.CTRL_FUZZY, np.array(CT["nsga2_knee"]), arr, seed=1003, rec_T=1500)
    tot = r["queues"][:1500].sum(axis=1)
    t = int(np.argmax(np.convolve(tot, np.ones(5) / 5, mode="same")[300:1500]) + 300)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    draw_junction(ax, r["occupancy"][t], int(r["signal"][t]), r["queues"][t])
    ax.set_title(f"Junction at t = {t} s (balanced demand, tuned fuzzy controller)", fontsize=9.5)
    handles = [Rectangle((0, 0), 1, 1, fc=GREEN), Rectangle((0, 0), 1, 1, fc=AMBER), Rectangle((0, 0), 1, 1, fc=RED)]
    ax.legend(handles, ["2 cells/s", "1 cell/s", "stopped"], loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    save(fig, "t02_junction")


# ------------------------------------------------------------------------------------------ T3 validation
def fig_validation():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    dep = np.array(VA["discharge_mean_departures"])
    idx = np.arange(1, len(dep) + 1)
    axes[0].plot(idx, dep, "o", ms=3.5, color="#1f5fa5", label="simulated")
    cal = VA["calibration"]["0.1"]
    axes[0].plot(idx, cal["startup_lost_time_fit_s"] + cal["headway_s"] * idx, color="#e4572e", lw=1.5, label=f"fit: {cal['headway_s']:.2f} s per car")
    axes[0].set_xlabel("car number in the queue")
    axes[0].set_ylabel("time the car crosses the line (s)")
    axes[0].set_title(f"Discharge: {cal['saturation_flow_cars_per_h']:.0f} cars/h")
    axes[0].legend(fontsize=7.5)
    fd = VA["fundamental_diagram"]
    for p, c in (("0.0", "#9db8d6"), ("0.1", "#1f5fa5"), ("0.2", "#e4572e")):
        axes[1].plot(fd["density_veh_per_km"], fd["flow_veh_per_h"][p], marker="o", ms=2.5, color=c, label=f"dawdle p = {p}")
    axes[1].set_xlabel("density (cars/km)")
    axes[1].set_ylabel("flow (cars/h)")
    axes[1].set_title("Fundamental diagram")
    axes[1].legend(fontsize=7.5)
    wv = VA["webster_validation"]
    q = [r["flow_per_approach"] for r in wv]
    axes[2].plot(q, [r["webster_delay"] for r in wv], "s-", color="#6b7a89", label="Webster formula")
    axes[2].errorbar(q, [r["sim_delay"] for r in wv], yerr=[r["sim_delay_sd"] for r in wv], fmt="o-", color="#1f5fa5", capsize=2, label="simulation")
    axes[2].set_xlabel("flow per approach (cars/h)")
    axes[2].set_ylabel("mean delay (s)")
    axes[2].set_title(f"Delay: mean error {VA['webster_summary']['mean_abs_pct_error']:.0f}%")
    axes[2].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "t03_validation")


# ------------------------------------------------------------------------------------------ T4 / T5 membership functions
def fig_membership():
    hand, nsga = np.array(FZ.HAND_TUNED), np.array(CT["nsga2_knee"])
    spec = [("Queue (urgency FIS), cars", 0, 40, 0), ("Time since last green (s)", 3, 240, 3), ("Urgency (output)", 6, 100, 6),
            ("Queue of chosen approach", 9, 40, 9), ("Pressure from other approaches", 12, 40, 12), ("Green time (s, output)", 15, FZ.G_HI, 15)]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.2))
    for ax, (title, i0, hi, _) in zip(axes.ravel(), spec):
        lo = FZ.G_LO if "Green" in title else 0.0
        x = np.linspace(lo, hi, 400)
        for genes, c, lw, ls, lab in ((hand, "#9aa5b1", 1.5, "--", "hand-tuned"), (nsga, "#1f5fa5", 1.9, "-", "NSGA-II (clean-tuned)")):
            curves = FZ.partition_curves(x, genes[i0], genes[i0 + 1], genes[i0 + 2])
            for k in range(3):
                ax.plot(x, curves[k], color=c, lw=lw, ls=ls, label=lab if k == 0 else None, alpha=0.95)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylim(-0.03, 1.08)
    axes[0, 0].legend(fontsize=7, loc="center right")
    for ax in axes[:, 0]:
        ax.set_ylabel("membership")
    fig.tight_layout()
    save(fig, "t04_membership")


def fig_surfaces():
    hand, nsga, rob = np.array(FZ.HAND_TUNED), np.array(CT["nsga2_knee"]), np.array(CT["nsga2_robust_knee"])
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4), constrained_layout=True)
    qs, ws = np.meshgrid(np.linspace(0, 30, 61), np.linspace(0, 150, 61))
    qq, pp = np.meshgrid(np.linspace(0, 30, 61), np.linspace(0, 25, 61))
    for j, (g, nm) in enumerate(((hand, "hand-tuned"), (nsga, "NSGA-II"), (rob, "robust"))):
        u = np.vectorize(lambda a, b, g=g: FZ.urgency(float(a), float(b), g))(qs, ws)
        im = axes[0, j].contourf(qs, ws, u, levels=np.linspace(0, 100, 21), cmap="viridis")
        axes[0, j].set_title(f"urgency, {nm}", fontsize=8.5)
        axes[0, j].set_xlabel("queue (cars)")
        gt = np.vectorize(lambda a, b, g=g: FZ.green_time(float(a), float(b), g))(qq, pp)
        im2 = axes[1, j].contourf(qq, pp, gt, levels=np.linspace(FZ.G_LO, FZ.G_HI, 23), cmap="magma_r")
        axes[1, j].set_title(f"green time, {nm}", fontsize=8.5)
        axes[1, j].set_xlabel("queue of chosen approach")
    axes[0, 0].set_ylabel("seconds since last green")
    axes[1, 0].set_ylabel("pressure (mean queue elsewhere)")
    fig.colorbar(im, ax=axes[0, :], shrink=0.85, label="urgency")
    fig.colorbar(im2, ax=axes[1, :], shrink=0.85, label="green time (s)")
    save(fig, "t05_surfaces")


# ------------------------------------------------------------------------------------------ T6 GA convergence
def fig_convergence():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for r in TU["nsga2"]:
        h = r["history"]
        axes[0].plot([x["gen"] for x in h], [x["hv"] for x in h], lw=1.2, alpha=0.9)
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("hypervolume (training set)")
    axes[0].set_title("NSGA-II: 5 runs")
    for r in TU["ga"]:
        h = r["history"]
        axes[1].plot([x["evals"] for x in h], [x["best_J"] for x in h], color="#2a9d8f", lw=1.2, alpha=0.9)
    for r in TU["random"]:
        h = r["history"]
        axes[1].plot([x["evals"] for x in h], [x["best_J"] for x in h], color="#c9aedb", lw=1.2, alpha=0.9)
    axes[1].plot([], [], color="#2a9d8f", label="GA")
    axes[1].plot([], [], color="#c9aedb", label="random search")
    axes[1].axhline(1.0, color=MUTED, ls=":", lw=1)
    axes[1].set_xlabel("fitness evaluations")
    axes[1].set_ylabel("best cost J (hand-tuned = 1)")
    axes[1].set_title("GA against random search")
    axes[1].legend(fontsize=7.5)
    runs = EV["independent_runs"]
    data = [[x["mean_delay"] for x in runs["Random search"]], [x["mean_delay"] for x in runs["GA"]], [x["mean_delay"] for x in runs["NSGA-II knee"]],
            [x["mean_delay"] for x in runs["NSGA-II robust"]]]
    labs = ["random", "GA", "NSGA\nknee", "robust"]
    cols_ = ["#c9aedb", "#2a9d8f", "#1f5fa5", "#d1495b"]
    if "NSGA-II practical robust" in runs:
        data += [[x["mean_delay"] for x in runs["NSGA-II practical"]], [x["mean_delay"] for x in runs["NSGA-II practical robust"]]]
        labs += ["pract.", "pract.\nrobust"]
        cols_ += [COL[PRA], COL[PRR]]
    bp = axes[2].boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color=INK, lw=1.3))
    for patch, c in zip(bp["boxes"], cols_):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    axes[2].set_xticklabels(labs, fontsize=6.3)
    axes[2].set_ylabel("test delay (s), 4 scenarios")
    axes[2].set_title("Independent runs")
    fig.tight_layout()
    save(fig, "t06_convergence")


# ------------------------------------------------------------------------------------------ T7 Pareto front
def fig_pareto():
    F = EV["front"]
    tr, te = np.array(F["train"]), np.array(F["test"])
    act, hand = np.array(F["actuated_test"]), np.array(F["hand_test"])
    k = F["knee_index"]
    ga_pt = [np.mean([M["Fuzzy + GA"][s][m]["mean"] for s in SC.TRAIN]) for m in ("mean_delay", "p95_delay", "mean_stops")]
    rb_pt = [np.mean([M["Fuzzy + NSGA-II (robust)"][s][m]["mean"] for s in SC.TRAIN]) for m in ("mean_delay", "p95_delay", "mean_stops")]
    pr_pt = [np.mean([M[PRA][s][m]["mean"] for s in SC.TRAIN]) for m in ("mean_delay", "p95_delay", "mean_stops")] if HAVE_PRA else None
    prr_pt = [np.mean([M[PRR][s][m]["mean"] for s in SC.TRAIN]) for m in ("mean_delay", "p95_delay", "mean_stops")] if HAVE_PRA else None
    pairs = [(0, 1, "mean delay (s)", "95th-percentile delay (s)"), (0, 2, "mean delay (s)", "stops per car"), (1, 2, "95th-percentile delay (s)", "stops per car")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for ax, (i, j, xl, yl) in zip(axes, pairs):
        ax.scatter(tr[:, i], tr[:, j], s=12, color="#9db8d6", label="front, training seeds")
        ax.scatter(te[:, i], te[:, j], s=12, color="#1f5fa5", label="same members, test seeds")
        ax.scatter(te[k, i], te[k, j], s=70, marker="*", color="#e4572e", zorder=5, label="knee point")
        ax.scatter(act[i], act[j], s=45, marker="s", color="#e9b44c", zorder=4, label="actuated")
        ax.scatter(hand[i], hand[j], s=45, marker="D", color="#6b7a89", zorder=4, label="fuzzy hand-tuned")
        ax.scatter(ga_pt[i], ga_pt[j], s=45, marker="^", color="#2a9d8f", zorder=4, label="single-objective GA")
        ax.scatter(rb_pt[i], rb_pt[j], s=45, marker="P", color="#d1495b", zorder=4, label="NSGA-II robust")
        if HAVE_PRA:
            ax.scatter(pr_pt[i], pr_pt[j], s=45, marker="X", color="#7b3fa0", zorder=4, label="NSGA-II practical")
            ax.scatter(prr_pt[i], prr_pt[j], s=45, marker="v", color="#c8702a", zorder=4, label="NSGA-II practical robust")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "t07_pareto")


# ------------------------------------------------------------------------------------------ T8 main results
def fig_main_results():
    kinds = ["Best fixed plan", "Vehicle-actuated", "Fuzzy (hand-tuned)", "Fuzzy + NSGA-II", "Fuzzy + NSGA-II (robust)"] + ([PRA, PRR] if HAVE_PRA else [])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(SC_ALL))
    w = 0.78 / len(kinds)
    for i, kd in enumerate(kinds):
        vals = [mean_of(kd, s) for s in SC_ALL]
        sds = [M[kd][s]["mean_delay"]["sd"] for s in SC_ALL]
        axes[0].bar(x + (i - (len(kinds) - 1) / 2) * w, vals, w, yerr=sds, color=COL[kd], label=lab(kd), error_kw=dict(lw=0.6, capsize=1))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SC.SHORT[s] + ("*" if s in SC.UNSEEN else "") for s in SC_ALL], fontsize=7, rotation=12)
    axes[0].set_ylabel("mean delay per car (s)")
    axes[0].set_title("Mean delay, 20 held-out seeds (* unseen pattern)", fontsize=9)
    axes[0].set_ylim(0, 41)
    axes[0].legend(fontsize=6.6, ncol=4, loc="upper left")
    allv = {kd: np.concatenate([M[kd][s]["mean_delay"]["values"] for s in SC_ALL]) for kd in kinds}
    bp = axes[1].boxplot([allv[k] for k in kinds], patch_artist=True, widths=0.6, showfliers=False, medianprops=dict(color=INK, lw=1.2))
    for patch, kd in zip(bp["boxes"], kinds):
        patch.set_facecolor(COL[kd])
    axes[1].set_xticklabels(["Best\nfixed", "Actu-\nated", "Fuzzy\nhand", "NSGA\nclean", "NSGA\nrobust", "NSGA\npract.", "NSGA\np. rob."][:len(kinds)], fontsize=5.8)
    axes[1].set_title("All 120 runs")
    axes[1].set_ylabel("mean delay (s)")
    fig.tight_layout()
    save(fig, "t08_main_results")


def fig_improvement_heatmap():
    refs = ["Best fixed plan", "Vehicle-actuated", "Fuzzy (hand-tuned)"]
    metrics = [("mean_delay", "Mean delay"), ("p95_delay", "95th-percentile delay"), ("mean_stops", "Stops per car")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), constrained_layout=True)
    im = None
    for ax, (metric, title) in zip(axes, metrics):
        mat = np.zeros((len(refs), len(SC_ALL)))
        for i, ref in enumerate(refs):
            for j, s in enumerate(SC_ALL):
                a, b = mean_of("Fuzzy + NSGA-II", s, metric), mean_of(ref, s, metric)
                mat[i, j] = 100 * (a - b) / b
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=-60, vmax=60, aspect="auto")
        ax.set_xticks(range(len(SC_ALL)))
        ax.set_xticklabels([SC.SHORT[s] for s in SC_ALL], rotation=35, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(refs)))
        ax.set_yticklabels(["vs best fixed", "vs actuated", "vs fuzzy hand"] if ax is axes[0] else [], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        for i in range(len(refs)):
            for j in range(len(SC_ALL)):
                ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center", fontsize=7.5, color="black")
    fig.colorbar(im, ax=axes, shrink=0.8, label="change (negative = better)")
    save(fig, "t09_improvement")


# ------------------------------------------------------------------------------------------ T9 robustness
def fig_robustness():
    rob = EV["robustness"]
    conds = list(rob)
    kinds = ["Best fixed plan", "Vehicle-actuated", "Fuzzy (hand-tuned)", "Fuzzy + NSGA-II", "Fuzzy + NSGA-II (robust)"] + ([PRA, PRR] if HAVE_PRA else [])
    fig, ax = plt.subplots(figsize=(7.3, 3.1))
    x = np.arange(len(conds))
    w = 0.8 / len(kinds)
    for i, kd in enumerate(kinds):
        vals = [rob[c][kd]["mean_delay"] for c in conds]
        ax.bar(x + (i - (len(kinds) - 1) / 2) * w, vals, w, color=COL[kd], label=lab(kd))
        for xx, v in zip(x + (i - (len(kinds) - 1) / 2) * w, vals):
            if v > 60:
                ax.text(xx, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "Noisy sensors", "Rain", "Demand +20%", "All three together"][:len(conds)], fontsize=8)
    ax.set_ylabel("mean delay, 4 scenarios (s)")
    ax.set_title("Same controllers under stress")
    ax.legend(fontsize=6.8, ncol=4, loc="upper left")
    ax.set_ylim(0, 125)
    save(fig, "t10_robustness")


# ------------------------------------------------------------------------------------------ T10 behaviour
def fig_behaviour():
    keys = {"Best fixed plan": "Best_fixed_plan", "Vehicle-actuated": "Vehicle-actuated", "Fuzzy + NSGA-II": "Fuzzy_+_NSGA-II"}
    fig = plt.figure(figsize=(7.4, 4.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.5, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    for kd, k in keys.items():
        q = TRACES[f"surge__{k}"].sum(axis=1)
        sm = np.convolve(q, np.ones(30) / 30, mode="same")
        ax.plot(np.arange(len(sm)), sm, color=COL[kd], lw=1.5, label=SHORT[kd])
    for t_ in (1200, 2400):
        ax.axvline(t_, color=MUTED, ls=":", lw=1)
    ax.text(600, ax.get_ylim()[1] * 0.92, "220 each", ha="center", fontsize=7.5, color=MUTED)
    ax.text(1800, ax.get_ylim()[1] * 0.92, "N-S rush", ha="center", fontsize=7.5, color=MUTED)
    ax.text(3000, ax.get_ylim()[1] * 0.92, "250 each", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 3700)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cars waiting at the junction")
    ax.set_title("Surge scenario: total queue (30 s average)")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    cmap = {0: APP_COL[0], 1: APP_COL[1], 2: APP_COL[2], 3: APP_COL[3], -1: "#f2f4f6"}
    for col_i, (kd, k) in enumerate([("Vehicle-actuated", "Vehicle-actuated"), ("Fuzzy + NSGA-II", "Fuzzy_+_NSGA-II")]):
        a2 = fig.add_subplot(gs[1, col_i])
        sig = TRACES[f"sig__arterial__{k}"][:420]
        for t_, s_ in enumerate(sig):
            a2.add_patch(Rectangle((t_, 0), 1, 1, fc=cmap[int(s_)], ec="none"))
        a2.set_xlim(0, 420)
        a2.set_ylim(0, 1)
        a2.set_yticks([])
        a2.set_xlabel("time (s)")
        a2.set_title(f"{SHORT[kd]}: who has green (arterial)", fontsize=8.5)
        a2.grid(False)
        a2.spines["left"].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, fc=APP_COL[i]) for i in range(4)] + [Rectangle((0, 0), 1, 1, fc="#f2f4f6", ec=MUTED)]
    fig.legend(handles, ["North", "East", "South", "West", "all red"], loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
    save(fig, "t11_behaviour")


def fig_green_share():
    sc = "arterial"
    dem = SC.mean_demand(sc)
    dem_share = dem / dem.sum()
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(4)
    w = 0.2
    ax.bar(x - 1.5 * w, dem_share, w, color="#c3ccd5", label="demand share")
    for i, (kd, c) in enumerate((("Best fixed plan", COL["Best fixed plan"]), ("Vehicle-actuated", COL["Vehicle-actuated"]), ("Fuzzy + NSGA-II", COL["Fuzzy + NSGA-II"]))):
        b = EV["behaviour"][sc][kd]["green_share"]
        b = np.array(b) / max(sum(b), 1e-9)
        ax.bar(x + (i - 0.5) * w, b, w, color=c, label=SHORT[kd])
    ax.set_xticks(x)
    ax.set_xticklabels(["North", "East", "South", "West"])
    ax.set_ylabel("share of all green time")
    ax.set_title("Arterial scenario: green follows demand")
    ax.legend(fontsize=7.5, ncol=2)
    save(fig, "t12_green_share")


# ------------------------------------------------------------------------------------------ T11 ablation + green ends
def fig_ablation():
    ab = EV["ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=False)
    variants = ["Full controller", "Wait-only urgency", "Fixed 30 s green cap"]
    cols = ["#1f5fa5", "#e4572e", "#e9b44c"]
    for ax, cond in zip(axes, ("Clean", "All three together")):
        x = np.arange(2)
        w = 0.25
        for i, v in enumerate(variants):
            vals = [ab["NSGA-II (clean-tuned)"][cond][v]["mean_delay"], ab["NSGA-II (robust-tuned)"][cond][v]["mean_delay"]]
            ax.bar(x + (i - 1) * w, vals, w, color=cols[i], label=v)
            for xx, val in zip(x + (i - 1) * w, vals):
                ax.text(xx, val * 1.01, f"{val:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["clean-tuned\n(green cap switched off)", "robust-tuned"], fontsize=8)
        ax.set_title("Clean conditions" if cond == "Clean" else "All three stresses together")
        ax.set_ylabel("mean delay (s)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "t13_ablation")


def fig_green_ends():
    rob = EV["robustness"]
    conds = list(rob)
    kinds = ["Vehicle-actuated", "Fuzzy (hand-tuned)", "Fuzzy + NSGA-II", "Fuzzy + NSGA-II (robust)"] + ([PRA, PRR] if HAVE_PRA else [])
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(len(conds))
    w = 0.8 / len(kinds)
    for i, kd in enumerate(kinds):
        vals = [100 * rob[c][kd]["cap_share"] for c in conds]
        ax.bar(x + (i - (len(kinds) - 1) / 2) * w, vals, w, color=COL[kd], label=lab(kd))
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "Noisy sensors", "Rain", "Demand +20%", "All three"], fontsize=8)
    ax.set_ylabel("greens ended by the time cap (%)")
    ax.set_title("The green-time cap only bites under stress")
    ax.legend(fontsize=6.8, ncol=3, loc="upper left")
    save(fig, "t14_green_ends")


# ------------------------------------------------------------------------------------------ T15 realistic-green fronts
def _practical_data():
    pf = json.loads((OUT / "practical_front.json").read_text())
    act, hand = np.array(pf["actuated_test"]), np.array(pf["hand_test"])
    fronts = [("NSGA-II front, any green ≥ 4 s", np.array(EV["front"]["test"]), COL["Fuzzy + NSGA-II"], EV["front"]["knee_index"]),
              ("practical front (green ≥ 8 s)", np.array(pf["practical"]["test"]), COL[PRA], pf["practical"]["knee_index"]),
              ("practical robust front (green ≥ 8 s)", np.array(pf["practical_robust"]["test"]), COL[PRR], pf["practical_robust"]["knee_index"])]
    return act, hand, fronts


def _draw_practical(ax, i, j, xl, yl, act, hand, fronts, ms=13, star=75):
    allF = np.vstack([f[1] for f in fronts])
    lo_x, hi_x = min(allF[:, i].min(), act[i]) * 0.97, max(allF[:, i].max(), act[i]) * 1.02
    lo_y, hi_y = min(allF[:, j].min(), act[j]) * 0.97, max(allF[:, j].max(), act[j]) * 1.02
    ax.add_patch(Rectangle((lo_x, lo_y), act[i] - lo_x, act[j] - lo_y, fc="#e3f2e8", ec="none", zorder=0))
    for name, F, c, k in fronts:
        ax.scatter(F[:, i], F[:, j], s=ms, color=c, alpha=0.75, label=name, zorder=2)
        ax.scatter(F[k, i], F[k, j], s=star, marker="*", color=c, edgecolor=INK, linewidth=0.6, zorder=5)
    ax.scatter(act[i], act[j], s=star - 20, marker="s", color=COL["Vehicle-actuated"], edgecolor=INK, linewidth=0.6, zorder=6, label="actuated")
    ax.scatter(hand[i], hand[j], s=star - 27, marker="D", color=COL["Fuzzy (hand-tuned)"], edgecolor=INK, linewidth=0.6, zorder=6, label="fuzzy hand-tuned")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)


def fig_practical_front():
    act, hand, fronts = _practical_data()
    pairs = [(0, 1, "mean delay (s)", "95th-percentile delay (s)"), (0, 2, "mean delay (s)", "stops per car")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, (i, j, xl, yl) in zip(axes, pairs):
        _draw_practical(ax, i, j, xl, yl, act, hand, fronts)
    axes[0].text(0.57, 0.05, "better than actuated\non both measures", transform=axes[0].transAxes, fontsize=7, color="#2e7d4f", va="bottom", ha="center")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    save(fig, "t15_practical_front")


def fig_practical_front_slide():
    """One large panel for the presentation: mean delay against 95th-percentile delay."""
    act, hand, fronts = _practical_data()
    with plt.rc_context({"font.size": 10.5, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(6.4, 3.5))
        _draw_practical(ax, 0, 1, "mean delay (s)", "95th-percentile delay (s)", act, hand, fronts, ms=18, star=110)
        ax.text(0.57, 0.05, "better than actuated\non both measures", transform=ax.transAxes, fontsize=9, color="#2e7d4f", va="bottom", ha="center")
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.2))
        fig.tight_layout()
        save(fig, "t15b_practical_front_slide")


def main():
    for fn in (fig_architecture, fig_junction, fig_validation, fig_membership, fig_surfaces, fig_convergence, fig_pareto, fig_main_results,
               fig_improvement_heatmap, fig_robustness, fig_behaviour, fig_green_share, fig_ablation, fig_green_ends, fig_practical_front, fig_practical_front_slide):
        fn()


if __name__ == "__main__":
    main()
