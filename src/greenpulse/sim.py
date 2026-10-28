"""Car-by-car model of an isolated four-way junction (one lane per approach, one phase per approach).

Traffic model: Nagel-Schreckenberg cellular automaton (Nagel & Schreckenberg, 1992).
  * a lane is a row of L = 40 cells, 7.5 m each (300 m); a car occupies one cell
  * one time step = 1 s; speeds are whole numbers of cells per step, vmax = 2 (15 m/s = 54 km/h)
  * every step each car: accelerates by 1 (up to vmax) -> brakes to the free gap in front ->
    dawdles with probability p (speed - 1) -> moves
  * the signal is a wall at the stop line while it is red; on green the first car simply leaves
Cars that cannot enter because the first cell is occupied wait in a virtual queue just outside the
lane, so no demand is ever lost and that waiting time counts as delay.

Controllers (code 0, 1, 2):
  FIXED     cyclic order N, E, S, W with fixed green times
  ACTUATED  vehicle-actuated: skip approaches with no vehicle waiting, gap-out / max-green
  FUZZY     fuzzy urgency picks the next approach, fuzzy green-time sets how long it keeps green
Between two greens there is always an all-red interval of INTERGREEN seconds.

Everything is compiled with numba; one simulated hour takes a few milliseconds.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from . import fuzzy as FZ

NA = 4                  # approaches: 0=N, 1=E, 2=S, 3=W
L = 40                  # cells per approach
VMAX = 2                # cells / second
CELL_M = 7.5            # metres per cell
INTERGREEN = 3          # all-red seconds between phases
DZ = 8                  # detector zone: cells 0..DZ-1 in front of the stop line (60 m)
MAXB = 4000             # virtual-queue ring-buffer size
MAXD = 9000             # most vehicles per run
T_FREE = L // VMAX      # free-flow travel time through the lane (s)
T_ARR = 3600            # seconds with arrivals
T_DRAIN = 900           # extra seconds to let queues clear
CTRL_FIXED, CTRL_ACTUATED, CTRL_FUZZY = 0, 1, 2
APPROACH = ("North", "East", "South", "West")

METRIC_NAMES = [
    "mean_delay", "p95_delay", "mean_stops", "throughput", "max_queue", "mean_queue", "served", "unserved",
    "arrived", "delay_N", "delay_E", "delay_S", "delay_W", "switches", "mean_stopped_time", "max_delay", "t_end",
    "cap_ends", "gap_ends",
]
IDX = {k: i for i, k in enumerate(METRIC_NAMES)}


@njit(cache=True)
def lane_new_speeds(pos, vel, n, first_gap, p_slow, newv):
    """Nagel-Schreckenberg speed update for the n cars of one lane (index 0 is the car nearest the stop line)."""
    for k in range(n):
        if k == 0:
            gap = first_gap
        else:
            gap = pos[k] - pos[k - 1] - 1
        v = vel[k] + 1
        if v > VMAX:
            v = VMAX
        if v > gap:
            v = gap
        if v > 0 and np.random.random() < p_slow:
            v -= 1
        newv[k] = v


@njit(cache=True)
def simulate(ctrl, params, arrivals, T_total, seed, p_slow, sigma_q, dropout, rec_T):
    """Run one junction simulation.

    arrivals : uint8 array (NA, T_arr), 1 when a car arrives in that second (common random numbers)
    params   : controller parameters (FIXED: 4 green times; ACTUATED: min, max, gap; FUZZY: 20 genes)
    sigma_q  : standard deviation (cars) of noise added to every queue measurement
    dropout  : probability that the stop-line detector misses a car in a given second
    rec_T    : number of leading seconds whose cell occupancy is recorded for animation
    """
    np.random.seed(seed)
    T_arr = arrivals.shape[1]
    pos = np.zeros((NA, L), np.int64)
    vel = np.zeros((NA, L), np.int64)
    arr_t = np.zeros((NA, L), np.int64)
    stp = np.zeros((NA, L), np.int64)
    stpd = np.zeros((NA, L), np.int64)
    n = np.zeros(NA, np.int64)
    newv = np.zeros(L, np.int64)
    buf = np.zeros((NA, MAXB), np.int64)
    bh = np.zeros(NA, np.int64)
    bc = np.zeros(NA, np.int64)
    d_delay = np.zeros(MAXD)
    d_app = np.zeros(MAXD, np.int64)
    d_stops = np.zeros(MAXD)
    d_stopped = np.zeros(MAXD)
    d_served = np.zeros(MAXD, np.int64)
    m = 0
    sig_log = np.full(T_total, -1, np.int8)
    q_log = np.zeros((T_total, NA), np.int16)
    if rec_T > 0:
        rec = np.zeros((rec_T, NA, L), np.int8)
    else:
        rec = np.zeros((1, NA, L), np.int8)

    last_end = np.full(NA, -1, np.int64)
    cur = -1
    ig_left = 0
    t_in = 0
    prev = NA - 1
    g_target = 0.0
    empty_run = 0
    switches = 0
    cap_ends = 0
    gap_ends = 0
    qm = np.zeros(NA)
    present = np.zeros(NA, np.bool_)
    sumq = 0.0
    maxq = 0.0
    arrived = 0
    t_end = T_total

    for t in range(T_total):
        # ---- arrivals join the virtual queue, then enter the lane if the entry cell is free
        if t < T_arr:
            for a in range(NA):
                if arrivals[a, t] > 0:
                    buf[a, (bh[a] + bc[a]) % MAXB] = t
                    bc[a] += 1
                    arrived += 1
        for a in range(NA):
            if bc[a] > 0 and n[a] < L:
                k = n[a]
                if k == 0 or pos[a, k - 1] < L - 1:
                    pos[a, k] = L - 1
                    if k == 0:
                        vel[a, k] = VMAX
                    else:
                        g0 = pos[a, k] - pos[a, k - 1] - 1
                        vel[a, k] = min(VMAX, g0)
                    arr_t[a, k] = buf[a, bh[a]]
                    bh[a] = (bh[a] + 1) % MAXB
                    bc[a] -= 1
                    stp[a, k] = 0
                    stpd[a, k] = 0
                    n[a] += 1

        # ---- sensors
        for a in range(NA):
            q = float(n[a] + bc[a])
            if sigma_q > 0.0:
                q += sigma_q * np.random.randn()
                if q < 0.0:
                    q = 0.0
            qm[a] = q
            seen = False
            for k in range(n[a]):
                if pos[a, k] < DZ:
                    seen = True
                    break
            if seen and dropout > 0.0 and np.random.random() < dropout:
                seen = False
            present[a] = seen

        # ---- signal controller
        if cur == -1:
            if ig_left > 0:
                ig_left -= 1
            else:
                nxt = (prev + 1) % NA
                if ctrl == CTRL_FIXED:
                    g_target = params[nxt]
                elif ctrl == CTRL_ACTUATED:
                    for s in range(1, NA + 1):
                        cand = (prev + s) % NA
                        if qm[cand] > 0.5:
                            nxt = cand
                            break
                else:
                    best = -1
                    bu = -1.0
                    anyq = False
                    for a in range(NA):
                        w = float(t - last_end[a]) if last_end[a] >= 0 else float(t + 1)
                        u = FZ.urgency(qm[a], w, params)
                        if qm[a] > 0.5:
                            anyq = True
                        if u > bu:
                            bu = u
                            best = a
                    if anyq:
                        nxt = best
                    p_mean = 0.0
                    for a in range(NA):
                        if a != nxt:
                            p_mean += qm[a]
                    p_mean /= 3.0
                    g_target = FZ.green_time(qm[nxt], p_mean, params)
                cur = nxt
                t_in = 0
                empty_run = 0
        if cur != -1:
            if present[cur]:
                empty_run = 0
            else:
                empty_run += 1
            end = False
            by_gap = False
            if ctrl == CTRL_FIXED:
                end = t_in >= g_target
            elif ctrl == CTRL_ACTUATED:
                by_gap = t_in >= params[0] and empty_run >= params[2]
                end = by_gap or t_in >= params[1]
            else:
                min_g = params[18]
                by_gap = t_in >= min_g and empty_run >= params[19]
                end = by_gap or t_in >= max(g_target, min_g)
            if end:
                if by_gap:
                    gap_ends += 1
                else:
                    cap_ends += 1
                last_end[cur] = t
                prev = cur
                cur = -1
                ig_left = INTERGREEN - 1
                switches += 1
        sig_log[t] = cur

        # ---- cars move
        for a in range(NA):
            cnt = n[a]
            if cnt == 0:
                continue
            first_gap = 1000 if cur == a else pos[a, 0]
            lane_new_speeds(pos[a], vel[a], cnt, first_gap, p_slow, newv)
            for k in range(cnt):
                if vel[a, k] > 0 and newv[k] == 0:
                    stp[a, k] += 1
                if newv[k] == 0:
                    stpd[a, k] += 1
                pos[a, k] -= newv[k]
                vel[a, k] = newv[k]
            ex = 0
            while ex < cnt and pos[a, ex] < 0:
                ex += 1
            if ex > 0:
                for e in range(ex):
                    dly = float((t + 1 - arr_t[a, e]) - T_FREE)
                    if dly < 0.0:
                        dly = 0.0
                    if m < MAXD:
                        d_delay[m] = dly
                        d_app[m] = a
                        d_stops[m] = stp[a, e]
                        d_stopped[m] = stpd[a, e]
                        d_served[m] = 1
                        m += 1
                for k in range(ex, cnt):
                    pos[a, k - ex] = pos[a, k]
                    vel[a, k - ex] = vel[a, k]
                    arr_t[a, k - ex] = arr_t[a, k]
                    stp[a, k - ex] = stp[a, k]
                    stpd[a, k - ex] = stpd[a, k]
                n[a] = cnt - ex

        if cur != -1:
            t_in += 1

        # ---- bookkeeping
        tot = 0.0
        for a in range(NA):
            qa = n[a] + bc[a]
            q_log[t, a] = qa
            tot += qa
            if qa > maxq:
                maxq = float(qa)
        if t < T_arr:
            sumq += tot
        if t < rec_T:
            for a in range(NA):
                for k in range(n[a]):
                    rec[t, a, pos[a, k]] = vel[a, k] + 1
        if t >= T_arr and tot == 0.0:
            t_end = t + 1
            break

    # ---- vehicles still in the system are counted with the delay they have already suffered
    served = m
    for a in range(NA):
        for k in range(n[a]):
            dly = float((t_end - arr_t[a, k]) - T_FREE)
            if m < MAXD:
                d_delay[m] = max(dly, 0.0)
                d_app[m] = a
                d_stops[m] = stp[a, k]
                d_stopped[m] = stpd[a, k]
                d_served[m] = 0
                m += 1
        for i in range(bc[a]):
            at = buf[a, (bh[a] + i) % MAXB]
            dly = float((t_end - at) - T_FREE)
            if m < MAXD:
                d_delay[m] = max(dly, 0.0)
                d_app[m] = a
                d_stops[m] = 0
                d_stopped[m] = 0
                d_served[m] = 0
                m += 1

    met = np.zeros(19)
    if m > 0:
        ds = np.sort(d_delay[:m])
        met[0] = d_delay[:m].mean()
        met[1] = ds[int(0.95 * (m - 1))]
        met[2] = d_stops[:m].mean()
        met[14] = d_stopped[:m].mean()
        met[15] = ds[m - 1]
        for a in range(NA):
            s = 0.0
            c = 0
            for i in range(m):
                if d_app[i] == a:
                    s += d_delay[i]
                    c += 1
            met[9 + a] = s / c if c > 0 else 0.0
    met[3] = served / (T_arr / 3600.0)
    met[4] = maxq
    met[5] = sumq / T_arr
    met[6] = served
    met[7] = m - served
    met[8] = arrived
    met[13] = switches
    met[16] = t_end
    met[17] = cap_ends
    met[18] = gap_ends
    return met, d_delay[:m].copy(), d_app[:m].copy(), d_served[:m].copy(), sig_log, q_log, rec


@njit(cache=True)
def discharge_test(p_slow, seed, queue_len, T):
    """Standing queue of `queue_len` cars released on green at t = 0: returns the departure time of each car.
    Uses exactly the same speed rule as the junction model."""
    np.random.seed(seed)
    pos = np.zeros(L, np.int64)
    vel = np.zeros(L, np.int64)
    newv = np.zeros(L, np.int64)
    n = queue_len
    for k in range(n):
        pos[k] = k
    dep = np.full(queue_len, -1, np.int64)
    out = 0
    for t in range(T):
        if n == 0:
            break
        lane_new_speeds(pos, vel, n, 1000, p_slow, newv)
        for k in range(n):
            pos[k] -= newv[k]
            vel[k] = newv[k]
        ex = 0
        while ex < n and pos[ex] < 0:
            ex += 1
        for e in range(ex):
            dep[out] = t + 1
            out += 1
        for k in range(ex, n):
            pos[k - ex] = pos[k]
            vel[k - ex] = vel[k]
        n -= ex
    return dep


@njit(cache=True)
def ring_flow(density, p_slow, seed, ring_len, steps):
    """Fundamental diagram check: the same speed rule on a closed ring road; returns mean flow (cars / cell / step)."""
    np.random.seed(seed)
    n = int(round(density * ring_len))
    pos = np.zeros(n, np.int64)
    vel = np.zeros(n, np.int64)
    newv = np.zeros(n, np.int64)
    step = ring_len // n
    for k in range(n):
        pos[k] = k * step
    # car 0 is the one with the smallest position; its leader is the last car (wrap-around)
    total = 0.0
    for t in range(steps):
        first_gap = pos[0] + ring_len - pos[n - 1] - 1
        lane_new_speeds(pos, vel, n, first_gap, p_slow, newv)
        moved = 0
        for k in range(n):
            vel[k] = newv[k]
            pos[k] -= newv[k]
            moved += newv[k]
        if pos[0] < 0:                      # car 0 wrapped past the origin -> becomes the rear-most car
            p0 = pos[0] + ring_len
            v0 = vel[0]
            for k in range(n - 1):
                pos[k] = pos[k + 1]
                vel[k] = vel[k + 1]
            pos[n - 1] = p0
            vel[n - 1] = v0
        if t >= steps // 2:
            total += moved
    return total / (steps - steps // 2) / ring_len


# ---------------------------------------------------------------------------------------------
# python-level helpers
# ---------------------------------------------------------------------------------------------
def make_params(ctrl: int, **kw) -> np.ndarray:
    if ctrl == CTRL_FIXED:
        g = kw["greens"]
        return np.array(g, dtype=np.float64)
    if ctrl == CTRL_ACTUATED:
        return np.array([kw.get("min_green", 7.0), kw.get("max_green", 45.0), kw.get("gap", 3.0)], dtype=np.float64)
    return np.asarray(kw["genes"], dtype=np.float64)


def run(ctrl: int, params: np.ndarray, arrivals: np.ndarray, seed: int = 0, p_slow: float = 0.10,
        sigma_q: float = 0.0, dropout: float = 0.0, rec_T: int = 0, drain: int = T_DRAIN):
    """Convenience wrapper returning a dict."""
    T_total = arrivals.shape[1] + drain
    met, delays, apps, served, sig, q, rec = simulate(ctrl, np.asarray(params, np.float64), arrivals.astype(np.uint8),
                                                      T_total, seed, p_slow, sigma_q, dropout, rec_T)
    out = {k: float(met[i]) for k, i in IDX.items()}
    out.update({"delays": delays, "approach": apps, "served_flag": served, "signal": sig[: int(met[16])],
                "queues": q[: int(met[16])], "occupancy": rec})
    return out
