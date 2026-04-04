"""
ampi/tuner.py — Bayesian-optimised build-parameter tuning for AMPIAffineFanIndex.

Tunes alpha (nlist = alpha·√n) and cone_top_k (K) via 1-D Gaussian Process BO
on a small data sample.  After convergence it builds the full index and sweeps
query parameters (cp, fp, w) to produce a table of suggested configs per
target recall level.

Usage
-----
    from ampi.tuner import AFanTuner

    tuner  = AFanTuner(data, queries, gt)
    result = tuner.tune()

    idx  = result['index']           # ready-to-use AMPIAffineFanIndex
    sugg = result['suggestions']     # list of (target_recall, cp, fp, w, cands, recall)
"""

import math
import time
import numpy as np
from .affine_fan import AMPIAffineFanIndex


# ── pure-numpy 1-D GP ─────────────────────────────────────────────────────

def _norm_cdf(x):
    f = np.frompyfunc(lambda v: 0.5 * (1.0 + math.erf(v / math.sqrt(2))), 1, 1)
    return f(np.asarray(x, dtype=float)).astype(float)


def _norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)


class _GP1D:
    """
    Minimal 1-D Gaussian Process with squared-exponential kernel.
    Lengthscale selected by maximising log marginal likelihood on a fixed grid.
    No external dependencies.
    """
    _LS_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    _NOISE   = 1e-3

    def __init__(self):
        self._x = self._y = self._L = self._a = self._ls = None

    def _K(self, x1, x2, ls):
        d = (np.asarray(x1)[:, None] - np.asarray(x2)[None, :]) / ls
        return np.exp(-0.5 * d ** 2)

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(x)
        best_lml, best_ls = -np.inf, 1.0
        for ls in self._LS_GRID:
            K = self._K(x, x, ls) + (self._NOISE + 1e-6) * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            a   = np.linalg.solve(L.T, np.linalg.solve(L, y))
            lml = -0.5 * float(y @ a) - np.log(np.diag(L)).sum()
            if lml > best_lml:
                best_lml, best_ls = lml, ls
        self._ls = best_ls
        K = self._K(x, x, best_ls) + (self._NOISE + 1e-6) * np.eye(n)
        self._L  = np.linalg.cholesky(K)
        self._a  = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        self._x, self._y = x, y

    def predict(self, x_new):
        x_new = np.asarray(x_new, float)
        Ks  = self._K(self._x, x_new, self._ls)
        Kss = self._K(x_new,   x_new, self._ls)
        mu  = Ks.T @ self._a
        v   = np.linalg.solve(self._L, Ks)
        var = np.diag(Kss) - (v ** 2).sum(0)
        return mu, np.maximum(var, 0.0)

    def EI(self, x_cand, xi=0.01):
        mu, var = self.predict(x_cand)
        sigma   = np.sqrt(var + 1e-12)
        f_best  = self._y.max()
        z       = (mu - f_best - xi) / sigma
        return np.maximum((mu - f_best - xi) * _norm_cdf(z) + sigma * _norm_pdf(z), 0.0)


# ── brute-force kNN (no external deps) ─────────────────────────────────────

def _brute_knn(data, queries, k):
    """Exact kNN via BLAS gemm.  No faiss required.

    ‖x−q‖² = ‖x‖² + ‖q‖² − 2x·q.  The dot-product term is a single gemm.
    Uses argpartition (O(n) per query) rather than a full sort.

    Parameters
    ----------
    data    : (n, d) float32
    queries : (q, d) float32
    k       : int

    Returns
    -------
    indices : (q, k) int32  — not sorted within each row
    """
    data    = np.asarray(data,    dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    n, d    = data.shape
    q       = queries.shape[0]
    k       = min(k, n)

    data_sq = np.sum(data    ** 2, axis=1)          # (n,)
    q_sq    = np.sum(queries ** 2, axis=1)           # (q,)

    out  = np.empty((q, k), dtype=np.int32)
    step = 256                                        # chunk queries to cap RAM
    for i in range(0, q, step):
        qc   = queries[i:i + step]                   # (step, d)
        dots = qc @ data.T                           # (step, n)
        d2   = data_sq[None, :] + q_sq[i:i+step, None] - 2.0 * dots
        out[i:i + step] = np.argpartition(
            d2, k - 1, axis=1
        )[:, :k].astype(np.int32)
    return out


# ── utilities ──────────────────────────────────────────────────────────────

def _scale_params(n, d):
    L_raw  = (d / 8) * max(1.0, math.log2(n / 5_000))
    L1     = max(16, min(64,  8 * round(L_raw / 8)))
    L2     = min(128, L1 * 2)
    w_base = max(15, int(15 * math.sqrt(n / 10_000)))
    return dict(L1=L1, L2=L2, w_base=w_base)


def _recall(gt, approx, k):
    return sum(
        len(set(g.tolist()) & set(a[:k].tolist()))
        for g, a in zip(gt, approx)
    ) / (len(gt) * k)


def _pareto_knee(pareto):
    """
    Return the knee of the (cands, recall) Pareto frontier — the point
    furthest from the line connecting the first and last Pareto points.

    This identifies the 'sweet spot': where marginal recall gain per
    additional candidate drops sharply.

    pareto : list of (cp, fp, w, recall, cands) sorted by ascending cands.
    Returns one element from pareto.
    """
    if len(pareto) <= 2:
        return pareto[-1]

    cands = np.array([r[4] for r in pareto], dtype=float)
    recs  = np.array([r[3] for r in pareto], dtype=float)

    # Normalise both axes to [0, 1] so neither dominates the distance metric
    c_range = cands[-1] - cands[0]
    r_range = recs[-1]  - recs[0]
    if c_range < 1e-10 or r_range < 1e-10:
        return pareto[-1]
    cn = (cands - cands[0]) / c_range
    rn = (recs  - recs[0])  / r_range

    # Unit vector from first to last point
    v  = np.array([cn[-1] - cn[0], rn[-1] - rn[0]])
    v /= np.linalg.norm(v)

    # Perpendicular (signed) distance of each point from that line
    dists = np.abs((cn - cn[0]) * v[1] - (rn - rn[0]) * v[0])

    return pareto[int(np.argmax(dists))]


# ── tuner ──────────────────────────────────────────────────────────────────

class AFanTuner:
    """
    Bayesian-optimised tuner for AMPIAffineFanIndex.

    Build parameters tuned
    ----------------------
    alpha     : nlist = alpha · √n  (continuous, BO over [0.25, 3.0])
    K         : cone_top_k          (discrete {1, 2, 3}, enumerated)
    F         : num_fans            (set analytically after alpha is chosen)

    Parameters
    ----------
    data, queries, gt
        Full dataset, held-out queries, ground-truth kNN indices (shape (q, k)).
    n_sample
        Data points used for BO evaluations (default: min(50k, 10 % of n)).
    n_bo_iter
        BO evaluations per K value.  Total builds = n_bo_iter × |K_candidates|.
    k
        Neighbourhood size for recall@k.
    seed
        RNG seed.
    """

    K_CANDIDATES  = [1, 2, 3]
    ALPHA_BOUNDS  = (0.25, 3.0)
    MIN_CONE_PTS  = 5
    N_BO_QUERIES  = 10    # queries used inside each BO objective call (fast)
    N_SUGG_QUERIES = 50   # queries used for the final suggestion sweep (accurate)

    def __init__(self, data, queries, gt,
                 n_sample=None, n_bo_iter=10, k=10, seed=0):
        self.data    = np.ascontiguousarray(data,    dtype=np.float32)
        self.queries = np.ascontiguousarray(queries, dtype=np.float32)
        self.gt      = np.asarray(gt)
        self.n, self.d = self.data.shape
        self.k, self.seed = k, seed
        self.n_bo_iter = n_bo_iter

        n_sample = min(n_sample or 50_000, max(5_000, int(0.1 * self.n)), self.n)
        self.n_sample = n_sample

        pr = _scale_params(self.n, self.d)
        self.L1, self.L2, self.w_base = pr['L1'], pr['L2'], pr['w_base']

        # Sample data + sample-level ground truth for BO objective
        rng              = np.random.default_rng(seed)
        sidx             = rng.choice(self.n, n_sample, replace=False)
        self.data_sample = self.data[sidx]
        bo_qs            = self.queries[:self.N_BO_QUERIES]
        gt_s             = _brute_knn(self.data_sample, bo_qs, k)
        self.bo_qs       = bo_qs
        self.gt_sample   = gt_s.astype(np.int32)

    # ── internal helpers ───────────────────────────────────────────────────

    def _viable_F(self, nlist, n=None):
        """Largest F such that expected points/cone >= MIN_CONE_PTS."""
        n = n or self.n
        cands  = sorted(set([16, 32, 64, self.L2]))
        viable = [F for F in cands if n // (nlist * F) >= self.MIN_CONE_PTS]
        return max(viable) if viable else 16

    def _build_sample(self, alpha, K):
        """Build a small index on the sample for BO objective evaluation."""
        nlist = max(16, int(alpha * math.sqrt(self.n_sample)))
        # Use F=16 for all sample builds: large F is meaningless on 50k points
        # (< 1 pt/cone), so only alpha is being selected here.
        return AMPIAffineFanIndex(
            self.data_sample, nlist=nlist, num_fans=16,
            seed=self.seed, cone_top_k=int(round(K)),
        )

    def _score(self, idx):
        """
        BO objective: average recall across three operating points covering
        low / medium / high candidate budgets.  Uses the sample-level GT.
        """
        F  = idx.F
        wb = max(5, int(self.w_base * math.sqrt(self.n_sample / self.n)))
        operating_pts = [
            (3,  max(1, F // 4), max(5, wb // 2)),
            (5,  max(1, F // 2), wb),
            (10, F,              wb),
        ]
        recalls = []
        for cp, fp, w in operating_pts:
            res    = [idx.query(q, k=self.k, window_size=w, probes=cp, fan_probes=fp)
                      for q in self.bo_qs]
            idxs   = [r[2] if isinstance(r, tuple) else r for r in res]
            padded = [np.pad(ix[:self.k], (0, max(0, self.k - len(ix))),
                             constant_values=-1) for ix in idxs]
            recalls.append(_recall(self.gt_sample, padded, self.k))
        return float(np.mean(recalls))

    # ── BO loop ────────────────────────────────────────────────────────────

    def _bo_alpha(self, K, n_iter, verbose):
        """1-D GP-BO over alpha for fixed K. Returns (best_alpha, best_score)."""
        lo, hi = self.ALPHA_BOUNDS
        x_cand = np.linspace(lo, hi, 300)
        rng    = np.random.default_rng(self.seed + K * 997)

        # Seed with 3 evaluations spread across the range
        x_obs = [lo, (lo + hi) / 2, hi]
        y_obs = [self._score(self._build_sample(a, K)) for a in x_obs]
        if verbose:
            for a, s in zip(x_obs, y_obs):
                print(f"      alpha={a:.3f}  score={s:.4f}  [seed]")

        gp = _GP1D()
        for _ in range(n_iter - 3):
            gp.fit(x_obs, y_obs)
            ei     = gp.EI(x_cand)
            next_a = float(x_cand[np.argmax(ei)])
            # Jitter if too close to an already-evaluated point
            if any(abs(next_a - a) < (hi - lo) / 40 for a in x_obs):
                next_a = float(rng.uniform(lo, hi))
            s = self._score(self._build_sample(next_a, K))
            x_obs.append(next_a)
            y_obs.append(s)
            if verbose:
                print(f"      alpha={next_a:.3f}  score={s:.4f}")

        best_i = int(np.argmax(y_obs))
        return x_obs[best_i], y_obs[best_i]

    # ── public API ─────────────────────────────────────────────────────────

    def tune(self, verbose=True):
        """
        Run BO, build the full index with best params, sweep query parameters,
        and return a result dict:

            {
              'index':       AMPIAffineFanIndex  (full dataset, ready to query),
              'nlist':       int,
              'alpha':       float,
              'K':           int,
              'F':           int,
              'suggestions': [(target_recall, cp, fp, w, cands, recall), ...],
            }
        """
        if verbose:
            print(f"[AFanTuner] BO on {self.n_sample:,}-pt sample  "
                  f"(n={self.n:,}  d={self.d}  k={self.k})")

        n_iter = max(4, self.n_bo_iter)
        best_K, best_alpha, best_score = 1, 1.0, -1.0

        for K in self.K_CANDIDATES:
            if verbose:
                print(f"  K={K}:")
            a, s = self._bo_alpha(K, n_iter, verbose)
            if verbose:
                print(f"    → best alpha={a:.3f}  score={s:.4f}")
            if s > best_score:
                best_K, best_alpha, best_score = K, a, s

        best_nlist = max(16, int(best_alpha * math.sqrt(self.n)))
        best_F     = self._viable_F(best_nlist)

        if verbose:
            print(f"\n[AFanTuner] Winner:  alpha={best_alpha:.3f} → nlist={best_nlist}"
                  f"  K={best_K}  F={best_F}")
            print("[AFanTuner] Building full index … ", end="", flush=True)

        t0  = time.perf_counter()
        idx = AMPIAffineFanIndex(
            self.data, nlist=best_nlist, num_fans=best_F,
            seed=self.seed, cone_top_k=best_K,
        )
        if verbose:
            print(f"{time.perf_counter() - t0:.1f}s")

        suggestions = self._suggest(idx, best_F, verbose)
        return dict(index=idx, nlist=best_nlist, alpha=best_alpha,
                    K=best_K, F=best_F, suggestions=suggestions)

    def _suggest(self, idx, F, verbose):
        """
        Evaluate a (cp, fp, w) grid on the full query set, build the Pareto
        frontier, and return the cheapest config for each target recall level.
        """
        wb  = self.w_base
        qs  = self.queries[:self.N_SUGG_QUERIES]
        gt  = self.gt[:self.N_SUGG_QUERIES]

        evals = []
        for cp in [5, 10, 20]:
            for fp in sorted(set([2, 4, 8, F // 4, F // 2, F])):
                for w_mult in [0.25, 0.5, 1.0, 1.5, 2.0]:
                    w      = max(5, int(wb * w_mult))
                    res    = [idx.query(q, k=self.k, window_size=w,
                                        probes=cp, fan_probes=fp) for q in qs]
                    idxs   = [r[2] if isinstance(r, tuple) else r for r in res]
                    padded = [np.pad(ix[:self.k], (0, max(0, self.k - len(ix))),
                                     constant_values=-1) for ix in idxs]
                    rec    = _recall(gt, padded, self.k)
                    cands  = int(len(idx.query_candidates(
                        qs[0], window_size=w, probes=cp, fan_probes=fp)))
                    evals.append((cp, fp, w, rec, cands))

        # Build Pareto frontier: min candidates, max recall
        evals.sort(key=lambda r: (r[4], -r[3]))
        pareto, best_rec = [], -1.0
        for row in evals:
            if row[3] > best_rec:
                best_rec = row[3]
                pareto.append(row)

        knee = _pareto_knee(pareto)

        # Cheapest Pareto config meeting each target recall
        targets     = [0.80, 0.90, 0.95, 0.99, 1.00]
        suggestions = []
        for target in targets:
            viable = [(cp, fp, w, rec, cands) for cp, fp, w, rec, cands in pareto
                      if rec >= target]
            row = min(viable, key=lambda r: r[4]) if viable else max(pareto, key=lambda r: r[3])
            suggestions.append((target, *row))

        if verbose:
            k = self.k
            print(f"\n[AFanTuner] Query-parameter suggestions"
                  f"  (estimated on {min(self.N_SUGG_QUERIES, len(qs))} queries):")
            hdr = (f"  {'Target R@'+str(k):<16}  {'cp':>4}  {'fp':>5}  {'w':>5}"
                   f"  {'cands':>9}  {'est R@'+str(k):>10}")
            print(hdr)
            print("  " + "─" * (len(hdr) - 2))
            knee_cands = knee[4]
            for target, cp, fp, w, rec, cands in suggestions:
                note = ""
                if rec < target:
                    note = "  (best available)"
                elif cands == knee_cands:
                    note = "  ← sweet spot"
                print(f"  {target:<16.2f}  {cp:>4}  {fp:>5}  {w:>5}"
                      f"  {cands:>9,}  {rec:>10.3f}{note}")
            # Always print knee if it doesn't coincide with any target row
            if knee_cands not in {cands for _, _, _, _, _, cands in suggestions}:
                cp, fp, w, rec, cands = knee
                print(f"  {'(sweet spot)':<16}  {cp:>4}  {fp:>5}  {w:>5}"
                      f"  {cands:>9,}  {rec:>10.3f}  ← knee")

        return suggestions
