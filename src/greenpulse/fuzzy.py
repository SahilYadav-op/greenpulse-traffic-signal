"""The two Mamdani fuzzy systems that drive the signal, plus the gene layout used by the GA.

FIS-1  "urgency"     inputs: queue length q, time since last green w      -> urgency 0..100
FIS-2  "green time"  inputs: queue of the chosen approach q, pressure p   -> green seconds G_LO..G_HI
                     (pressure = mean queue of the three other approaches)

Each input has three linguistic terms (Low / Medium / High) forming a strong triangular partition
defined by three centres c1 <= c2 <= c3; the output has three terms with centres o1 <= o2 <= o3.
The rule tables are fixed (they encode common sense); the GA tunes the centres and two timing
parameters. Evaluation is min for AND, max for aggregation, centroid for defuzzification.
"""
from __future__ import annotations

import os

import numpy as np
from numba import njit

NG = 101                      # points used for the centroid
G_LO, G_HI = 5.0, 60.0        # universe of the green-time output (seconds)
U_LO, U_HI = 0.0, 100.0       # universe of the urgency output

# rows: first input Low/Medium/High, columns: second input Low/Medium/High; entries: output term 0/1/2
URGENCY_RULES = np.array([[0, 0, 1],      # few vehicles : recent->Low,    moderate->Low,    long wait->Medium
                          [0, 1, 2],      # some vehicles: recent->Low,    moderate->Medium, long wait->High
                          [1, 2, 2]],     # many vehicles: recent->Medium, moderate->High,   long wait->High
                         dtype=np.int64)
GREEN_RULES = np.array([[0, 0, 0],        # few queued : Short whatever the pressure
                        [1, 1, 0],        # some queued: Medium unless pressure is high -> Short
                        [2, 2, 1]],       # many queued: Long unless pressure is high -> Medium
                       dtype=np.int64)

URG_TERMS = ("Low", "Medium", "High")
GREEN_TERMS = ("Short", "Medium", "Long")

# ---------------------------------------------------------------------------------------------
# gene layout (20 genes)
# ---------------------------------------------------------------------------------------------
GENE_NAMES = [
    "q_c1", "q_c2", "q_c3", "w_c1", "w_c2", "w_c3", "urg_o1", "urg_o2", "urg_o3",
    "qsel_c1", "qsel_c2", "qsel_c3", "p_c1", "p_c2", "p_c3", "grn_o1", "grn_o2", "grn_o3",
    "min_green", "gap_out",
]
GENE_LO = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, G_LO, G_LO, G_LO, 4.0, 1.0])
GENE_HI = np.array([40, 40, 40, 240, 240, 240, 100, 100, 100, 40, 40, 40, 40, 40, 40, G_HI, G_HI, G_HI, 14.0, 6.0])
# A real controller cannot flash greens of 4 s: the lower bound of the minimum-green gene can be raised from outside
# (environment variable, so that the worker processes of the GA see it too).
GENE_LO[18] = float(os.environ.get("GREENPULSE_MIN_GREEN_LO", GENE_LO[18]))
SORTED_TRIPLES = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18)]     # genes that must stay ordered
N_GENES = len(GENE_NAMES)

# expert-designed controller: chosen by reasoning, not by running the simulator
HAND_TUNED = np.array([
    0, 8, 20,        # queue: Few / Some / Many centred at 0, 8, 20 vehicles
    0, 50, 120,      # waiting: Recent / Moderate / Long at 0, 50, 120 s
    10, 50, 90,      # urgency output centres
    0, 8, 20,        # chosen queue
    0, 6, 15,        # pressure: Low / Medium / High
    10, 25, 45,      # green output centres (s)
    6.0, 3.0,        # minimum green, gap-out time
], dtype=np.float64)


def repair(genes: np.ndarray) -> np.ndarray:
    """Clip to bounds and keep every (c1, c2, c3) triple sorted with a small gap."""
    g = np.clip(np.asarray(genes, dtype=np.float64).copy(), GENE_LO, GENE_HI)
    for a, b in SORTED_TRIPLES:
        g[a:b] = np.sort(g[a:b])
        for i in range(a + 1, b):                      # forward pass: at least 0.5 between neighbours
            g[i] = max(g[i], g[i - 1] + 0.5)
        if g[b - 1] > GENE_HI[b - 1]:                  # pushed past the upper bound: pull the triple back down
            g[b - 1] = GENE_HI[b - 1]
            for i in range(b - 2, a - 1, -1):
                g[i] = min(g[i], g[i + 1] - 0.5)
    return g


# ---------------------------------------------------------------------------------------------
# numba kernels (used inside the simulator) -----------------------------------------------------
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def strong3(x, c1, c2, c3):
    """Memberships (low, medium, high) of x in a strong triangular partition with centres c1<=c2<=c3."""
    if x <= c1:
        return 1.0, 0.0, 0.0
    if x >= c3:
        return 0.0, 0.0, 1.0
    if x <= c2:
        d = c2 - c1
        t = (x - c1) / d if d > 1e-9 else 1.0
        return 1.0 - t, t, 0.0
    d = c3 - c2
    t = (x - c2) / d if d > 1e-9 else 1.0
    return 0.0, 1.0 - t, t


@njit(cache=True)
def mamdani3x3(x, y, g, ia, ib, io, lo, hi, table):
    """Evaluate one 3x3 Mamdani system. g is the gene vector; ia, ib, io are the start indices of the
    centres of the first input, second input and output."""
    a0, a1, a2 = strong3(x, g[ia], g[ia + 1], g[ia + 2])
    b0, b1, b2 = strong3(y, g[ib], g[ib + 1], g[ib + 2])
    ma = (a0, a1, a2)
    mb = (b0, b1, b2)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for i in range(3):
        for j in range(3):
            w = min(ma[i], mb[j])
            k = table[i, j]
            if k == 0:
                s0 = max(s0, w)
            elif k == 1:
                s1 = max(s1, w)
            else:
                s2 = max(s2, w)
    num = 0.0
    den = 0.0
    for q in range(NG):
        xg = lo + (hi - lo) * q / (NG - 1)
        m0, m1, m2 = strong3(xg, g[io], g[io + 1], g[io + 2])
        agg = max(min(s0, m0), min(s1, m1), min(s2, m2))
        num += agg * xg
        den += agg
    if den < 1e-12:
        return 0.5 * (lo + hi)
    return num / den


@njit(cache=True)
def urgency(q, w, g):
    return mamdani3x3(q, w, g, 0, 3, 6, U_LO, U_HI, URGENCY_RULES)


@njit(cache=True)
def green_time(q_sel, pressure, g):
    return mamdani3x3(q_sel, pressure, g, 9, 12, 15, G_LO, G_HI, GREEN_RULES)


# ---------------------------------------------------------------------------------------------
# python helpers for plots and the inspector ----------------------------------------------------
# ---------------------------------------------------------------------------------------------
def partition_curves(x: np.ndarray, c1: float, c2: float, c3: float) -> np.ndarray:
    """(3, len(x)) membership curves of a strong partition (vectorised twin of strong3)."""
    x = np.asarray(x, dtype=float)
    low = np.where(x <= c1, 1.0, np.where(x >= c2, 0.0, (c2 - x) / max(c2 - c1, 1e-9)))
    high = np.where(x >= c3, 1.0, np.where(x <= c2, 0.0, (x - c2) / max(c3 - c2, 1e-9)))
    mid = np.clip(1.0 - low - high, 0.0, 1.0)
    return np.vstack([low, mid, high])


def variable_curves(g: np.ndarray) -> dict:
    """All membership curves of a controller, keyed by variable, ready to plot."""
    spec = {
        "queue (urgency FIS)": (0, 40, 0, ("Few", "Some", "Many")),
        "time since last green, s": (3, 240, 3, ("Recent", "Moderate", "Long")),
        "urgency (output)": (6, 100, 6, URG_TERMS),
        "queue of chosen approach": (9, 40, 9, ("Few", "Some", "Many")),
        "pressure from other approaches": (12, 40, 12, ("Low", "Medium", "High")),
        "green time, s (output)": (15, G_HI, 15, GREEN_TERMS),
    }
    out = {}
    for name, (i0, hi, _, terms) in spec.items():
        lo = G_LO if "green time" in name else 0.0
        x = np.linspace(lo, hi, 400)
        out[name] = (x, partition_curves(x, g[i0], g[i0 + 1], g[i0 + 2]), terms)
    return out


def explain_urgency(q: float, w: float, g: np.ndarray) -> dict:
    """Memberships, rule strengths and the crisp urgency for one approach (for the app)."""
    mq = strong3(q, g[0], g[1], g[2])
    mw = strong3(w, g[3], g[4], g[5])
    rules = []
    for i, tq in enumerate(("Few", "Some", "Many")):
        for j, tw in enumerate(("Recent", "Moderate", "Long")):
            s = min(mq[i], mw[j])
            rules.append({"rule": f"IF queue is {tq} AND wait is {tw} THEN urgency is {URG_TERMS[URGENCY_RULES[i, j]]}",
                          "strength": float(s)})
    return {"queue_memberships": [float(v) for v in mq], "wait_memberships": [float(v) for v in mw],
            "rules": rules, "urgency": float(urgency(float(q), float(w), g))}


def explain_green(q: float, p: float, g: np.ndarray) -> dict:
    mq = strong3(q, g[9], g[10], g[11])
    mp = strong3(p, g[12], g[13], g[14])
    rules = []
    for i, tq in enumerate(("Few", "Some", "Many")):
        for j, tp in enumerate(("Low", "Medium", "High")):
            s = min(mq[i], mp[j])
            rules.append({"rule": f"IF queue is {tq} AND pressure is {tp} THEN green is {GREEN_TERMS[GREEN_RULES[i, j]]}",
                          "strength": float(s)})
    return {"queue_memberships": [float(v) for v in mq], "pressure_memberships": [float(v) for v in mp],
            "rules": rules, "green_s": float(green_time(float(q), float(p), g))}
