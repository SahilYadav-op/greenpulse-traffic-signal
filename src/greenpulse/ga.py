"""Genetic algorithms for the fuzzy signal controller: a plain real-coded GA, NSGA-II and random search.

All three use the same chromosome (20 real genes, see fuzzy.py), the same simulated training scenarios and
the same number of fitness evaluations, so differences are due to the search method alone.

Operators
  selection   binary tournament (NSGA-II: rank first, crowding distance second) / size-3 tournament (GA)
  crossover   simulated binary crossover, SBX (Deb & Agrawal, 1995), eta = 15, probability 0.9
  mutation    polynomial mutation (Deb & Goyal, 1996), eta = 20, probability 1/20 per gene
  repair      clip to bounds and keep every (c1, c2, c3) triple ordered
NSGA-II: fast non-dominated sorting and crowding distance (Deb et al., 2002).
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import fuzzy as FZ
from . import scenarios as SC
from . import sim

# --------------------------------------------------------------------------------------------------
# fitness evaluation (parallel)
# --------------------------------------------------------------------------------------------------
_TRAIN: list = []
_CFG: dict = {}


def _init_worker(train_specs, cfg):
    """Each spec is (scenario, seed, demand scale) or (scenario, seed, scale, p_slow, sigma_q, dropout).
    With cfg['relative'] every objective is divided by what the vehicle-actuated controller scores on the same run,
    so stressful conditions (where all delays are large) do not drown out the easy ones."""
    global _TRAIN, _CFG
    from . import baselines as B

    _CFG = cfg
    _TRAIN = []
    for spec in train_specs:
        name, seed, scale = spec[:3]
        p_slow = spec[3] if len(spec) > 3 else cfg.get("p_slow", 0.10)
        sigma = spec[4] if len(spec) > 4 else cfg.get("sigma_q", 0.0)
        drop = spec[5] if len(spec) > 5 else cfg.get("dropout", 0.0)
        arr = SC.make_arrivals(name, seed, scale)
        ref = None
        if cfg.get("relative", False):
            m = sim.simulate(sim.CTRL_ACTUATED, np.array(B.ACTUATED), arr, arr.shape[1] + sim.T_DRAIN, seed, p_slow, sigma, drop, 0)[0]
            ref = np.array([m[0], m[1], m[2]])
        _TRAIN.append((arr, seed, p_slow, sigma, drop, ref))


def objectives(genes: np.ndarray, train=None, cfg=None) -> np.ndarray:
    """Mean delay (s), 95th-percentile delay (s), stops per car, averaged over the training runs
    (or the same three numbers relative to the actuated controller when cfg['relative'])."""
    train = _TRAIN if train is None else train
    g = FZ.repair(genes)
    tot = np.zeros(3)
    for arr, seed, p_slow, sigma, drop, ref in train:
        met = sim.simulate(sim.CTRL_FUZZY, g, arr, arr.shape[1] + sim.T_DRAIN, seed, p_slow, sigma, drop, 0)[0]
        v = np.array([met[0], met[1], met[2]])
        tot += v / ref if ref is not None else v
    return tot / len(train)


def _eval_one(genes):
    return objectives(genes)


class Evaluator:
    """Evaluates batches of chromosomes in a process pool (falls back to serial for n_workers <= 1)."""

    def __init__(self, train_specs, n_workers: int = 10, cfg: dict | None = None):
        self.train_specs = list(train_specs)
        self.cfg = cfg or {}
        self.n_workers = n_workers
        self.count = 0
        if n_workers > 1:
            self.pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"),
                                            initializer=_init_worker, initargs=(self.train_specs, self.cfg))
        else:
            self.pool = None
            _init_worker(self.train_specs, self.cfg)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        self.count += len(batch)
        if self.pool is None:
            return np.array([_eval_one(g) for g in batch])
        return np.array(list(self.pool.map(_eval_one, list(batch), chunksize=max(1, len(batch) // (4 * self.n_workers)))))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


# --------------------------------------------------------------------------------------------------
# operators
# --------------------------------------------------------------------------------------------------
def sbx(p1, p2, rng, eta=15.0, pc=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < pc:
        u = rng.random(len(p1))
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
        swap = rng.random(len(p1)) < 0.5
        a = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        b = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        c1[swap], c2[swap] = a[swap], b[swap]
    return c1, c2


def poly_mutation(x, rng, eta=20.0, pm=None):
    pm = 1.0 / len(x) if pm is None else pm
    y = x.copy()
    for i in range(len(x)):
        if rng.random() < pm:
            u = rng.random()
            delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            y[i] += delta * (FZ.GENE_HI[i] - FZ.GENE_LO[i])
    return y


def random_population(n, rng):
    return np.array([FZ.repair(FZ.GENE_LO + rng.random(FZ.N_GENES) * (FZ.GENE_HI - FZ.GENE_LO)) for _ in range(n)])


# --------------------------------------------------------------------------------------------------
# multi-objective machinery
# --------------------------------------------------------------------------------------------------
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F):
    n = len(F)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(F, front):
    d = np.zeros(len(front))
    if len(front) <= 2:
        d[:] = np.inf
        return d
    Ff = F[front]
    for m in range(F.shape[1]):
        order = np.argsort(Ff[:, m])
        d[order[0]] = d[order[-1]] = np.inf
        span = Ff[order[-1], m] - Ff[order[0], m]
        if span <= 0:
            continue
        for k in range(1, len(front) - 1):
            d[order[k]] += (Ff[order[k + 1], m] - Ff[order[k - 1], m]) / span
    return d


def rank_and_crowd(F):
    n = len(F)
    rank = np.zeros(n, int)
    crowd = np.zeros(n)
    for r, fr in enumerate(fast_non_dominated_sort(F)):
        rank[fr] = r
        crowd[fr] = crowding_distance(F, fr)
    return rank, crowd


def hv2d(P, ref):
    P = P[(P[:, 0] < ref[0]) & (P[:, 1] < ref[1])]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    # sweep along x: between consecutive x values the dominated height is ref_y minus the lowest y seen so far
    area, best_y = 0.0, ref[1]
    xs = list(P[:, 0]) + [ref[0]]
    ys = list(P[:, 1])
    for i in range(len(P)):
        best_y = min(best_y, ys[i])
        area += (xs[i + 1] - xs[i]) * (ref[1] - best_y)
    return float(area)


def hypervolume3d(F, ref):
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 2])]
    vol = 0.0
    for i in range(len(F)):
        z0 = F[i, 2]
        z1 = F[i + 1, 2] if i + 1 < len(F) else ref[2]
        vol += hv2d(F[: i + 1, :2], ref[:2]) * (z1 - z0)
    return float(vol)


def knee_point(F):
    """Member of a non-dominated set closest (after min-max scaling) to the ideal point."""
    lo, hi = F.min(axis=0), F.max(axis=0)
    Z = (F - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    return int(np.argmin(np.sqrt((Z ** 2).sum(axis=1))))


# --------------------------------------------------------------------------------------------------
# algorithms
# --------------------------------------------------------------------------------------------------
def nsga2(evaluate, pop_size, n_gen, rng, ref_obj, hv_ref=(1.6, 1.6, 1.6), callback=None):
    pop = random_population(pop_size, rng)
    F = evaluate(pop)
    rank, crowd = rank_and_crowd(F)
    hist = []

    def record(g):
        front = F[rank == 0]
        hist.append({"gen": g, "evals": pop_size * (g + 1), "hv": hypervolume3d(front / ref_obj, np.array(hv_ref)),
                     "best": F.min(axis=0).tolist(), "front_size": int(len(front)),
                     "front": front.tolist()})
        if callback:
            callback(hist[-1])

    record(0)
    for g in range(1, n_gen + 1):
        kids = []
        while len(kids) < pop_size:
            idx = []
            for _ in range(2):
                a, b = rng.integers(0, pop_size, 2)
                idx.append(a if (rank[a], -crowd[a]) < (rank[b], -crowd[b]) else b)
            c1, c2 = sbx(pop[idx[0]], pop[idx[1]], rng)
            kids += [FZ.repair(poly_mutation(c1, rng)), FZ.repair(poly_mutation(c2, rng))]
        kids = np.array(kids[:pop_size])
        Fk = evaluate(kids)
        allp, allF = np.vstack([pop, kids]), np.vstack([F, Fk])
        fronts = fast_non_dominated_sort(allF)
        chosen = []
        for fr in fronts:
            if len(chosen) + len(fr) <= pop_size:
                chosen += fr
            else:
                cd = crowding_distance(allF, fr)
                order = np.argsort(-cd)
                chosen += [fr[i] for i in order[: pop_size - len(chosen)]]
                break
        pop, F = allp[chosen], allF[chosen]
        rank, crowd = rank_and_crowd(F)
        record(g)
    front_idx = np.where(rank == 0)[0]
    return {"pop": pop, "F": F, "front_idx": front_idx, "history": hist}


def scalar_cost(F, ref_obj, weights=(1.0, 1.0, 1.0)):
    return (np.atleast_2d(F) / ref_obj * np.asarray(weights)).sum(axis=1) / sum(weights)


def simple_ga(evaluate, pop_size, n_gen, rng, ref_obj, elite=2, tour=3, callback=None):
    pop = random_population(pop_size, rng)
    F = evaluate(pop)
    J = scalar_cost(F, ref_obj)
    hist = []

    def record(g):
        i = int(np.argmin(J))
        hist.append({"gen": g, "evals": pop_size * (g + 1), "best_J": float(J[i]), "mean_J": float(J.mean()),
                     "best_obj": F[i].tolist()})
        if callback:
            callback(hist[-1])

    record(0)
    for g in range(1, n_gen + 1):
        order = np.argsort(J)
        nxt = [pop[i].copy() for i in order[:elite]]
        while len(nxt) < pop_size:
            parents = []
            for _ in range(2):
                cand = rng.integers(0, pop_size, tour)
                parents.append(pop[cand[np.argmin(J[cand])]])
            c1, c2 = sbx(parents[0], parents[1], rng)
            nxt += [FZ.repair(poly_mutation(c1, rng)), FZ.repair(poly_mutation(c2, rng))]
        pop = np.array(nxt[:pop_size])
        F = evaluate(pop)
        J = scalar_cost(F, ref_obj)
        record(g)
    i = int(np.argmin(J))
    return {"pop": pop, "F": F, "best": pop[i], "best_F": F[i], "best_J": float(J[i]), "history": hist}


def random_search(evaluate, n_eval, rng, ref_obj, batch=200):
    best_J, best_g, best_F, hist = np.inf, None, None, []
    done = 0
    while done < n_eval:
        n = min(batch, n_eval - done)
        P = random_population(n, rng)
        F = evaluate(P)
        J = scalar_cost(F, ref_obj)
        i = int(np.argmin(J))
        if J[i] < best_J:
            best_J, best_g, best_F = float(J[i]), P[i], F[i]
        done += n
        hist.append({"evals": done, "best_J": best_J})
    return {"best": best_g, "best_F": best_F, "best_J": best_J, "history": hist}
