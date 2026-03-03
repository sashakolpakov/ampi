"""
AMPI Hashing: p-stable LSH for L2 (Datar et al., 2004) with k-AND amplification.

Hash function
─────────────
    h_{a,b,w}(x) = floor( (a·x + b) / w )

where a ~ N(0, Iᵈ) normalised to unit length, b ~ Uniform[0, w).

Collision probability (unit-normalised projection)
───────────────────────────────────────────────────
For ‖x − q‖₂ = r:

    p(r, w) = 2·Φ(w√d / r) − 1  −  (2r / (w·√(2πd))) · (1 − exp(−w²d/(2r²)))

Strictly decreasing in r: near points always collide with higher probability.

k-AND amplification
────────────────────
M tables, each with k independently drawn hash functions concatenated.
A point is a candidate from table m only if it matches the query on ALL k hashes.

    p₁_AND = p(r_nn, w)^k        ← per-table capture rate for NN
    p₂_AND = p(r_far, w)^k       ← per-table false-positive rate

Union over M tables (OR of ANDs):
    P_captured  = 1 − (1 − p₁_AND)^M
    P_false_pos ≈ M · p₂_AND

The formal ρ exponent (Datar 2004) is unchanged:
    ρ = log(1/p₁_AND)/log(1/p₂_AND) = log(1/p₁)/log(1/p₂) < 1

But p₂_AND = p₂^k collapses exponentially with k, enabling a smaller bucket_size
(fewer points per compound bucket) while maintaining recall by increasing M.

Parameter guide
───────────────
    bucket_size ≈ σ_proj = r_nn / √d   (NN projection std-dev)
    hash_width  k = 2–4                (2 is a good default)
    num_projections (M) ≈ 4k           (more tables compensate for tighter AND)

Multi-probe with k-AND
──────────────────────
With probes=p, each table checks (2p−1)^k compound keys (ℓ∞ ball of radius p−1).
For k ≥ 3 use probes=1 (exact match); the tight AND buckets already filter well.

References
──────────
Datar, Immorlica, Indyk, Mirrokni (2004). SCG '04.
Lv, Josephson, Wang, Charikar, Li (2007). VLDB '07.
"""

import itertools
import numpy as np
from ._kernels import project_data, l2_distances
from .tomography import estimate_nn_distance


class AMPIHashIndex:
    """Approximate nearest-neighbour index via p-stable LSH.

    Supports both the original OR scheme (hash_width=1) and k-AND
    amplification (hash_width=k ≥ 2).

    Parameters
    ----------
    data            : (n, d) array_like, float32
    num_projections : M — number of hash tables.
    hash_width      : k — hash functions AND-concatenated per table (default 2).
                      Higher k → exponentially fewer false positives per table,
                      enabling a smaller bucket_size and sparser candidate sets.
                      Use k=1 only if you need the degenerate single-hash case.
    bucket_size     : w — bucket width in projection space.
                      Rule of thumb:
                        hash_width=1 → w ≈ r_nn / √d  (= σ_proj)
                        hash_width=k → w ≈ σ_proj / √k
    seed            : RNG seed
    """

    def __init__(self, data, num_projections=16, hash_width=2,
                 bucket_size="auto", seed=0, directions=None):
        self.data        = np.ascontiguousarray(data, dtype=np.float32)
        self.n, self.d   = self.data.shape
        self.M           = num_projections   # number of tables
        self.k           = hash_width        # hashes AND'd per table
        # Auto bucket_size: r_nn / sqrt(d) — adapts to dataset scale/dimensionality.
        # A fixed value (e.g. 1.0) fails for MNIST (r_nn ~2000, d=784 → need ~71)
        # and is marginal for Gaussian (r_nn ~16, d=128 → ~1.4).
        if bucket_size == "auto":
            r_nn = estimate_nn_distance(self.data, seed=seed)
            bucket_size = r_nn / np.sqrt(self.d)
        self.bucket_size = float(bucket_size)
        total = self.M * self.k

        rng = np.random.RandomState(seed)
        if directions is not None:
            dirs = np.ascontiguousarray(directions, dtype=np.float32)[:total]
            dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        else:
            dirs = rng.randn(total, self.d).astype(np.float32)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        self.proj_dirs = dirs                          # (M*k, d)
        self.offsets   = (rng.rand(total).astype(np.float32)
                          * bucket_size)               # (M*k,)

        # ── Project all data ──────────────────────────────────────────────────
        projs = project_data(self.data, self.proj_dirs)  # (M*k, n)
        # Integer bucket IDs for every (projection, point)
        all_bids = np.floor(
            (projs + self.offsets[:, None]) / bucket_size
        ).astype(np.int32)                             # (M*k, n)

        # ── Build tables ──────────────────────────────────────────────────────
        # For k=1: table[bucket_id] -> int32 array of indices
        # For k>1: table[(b1, b2, ..., bk)] -> int32 array of indices
        self.tables = []
        for m in range(self.M):
            bids_m = all_bids[m * self.k : (m + 1) * self.k, :]  # (k, n)
            table  = {}
            if self.k == 1:
                # 1-D keys: use plain integers (faster dict lookup)
                for pt in range(self.n):
                    key = int(bids_m[0, pt])
                    if key not in table:
                        table[key] = []
                    table[key].append(pt)
            else:
                # k-D compound keys: use tuples
                keys_T = bids_m.T  # (n, k)
                for pt in range(self.n):
                    key = tuple(keys_T[pt].tolist())
                    if key not in table:
                        table[key] = []
                    table[key].append(pt)
            self.tables.append(
                {key: np.array(v, dtype=np.int32) for key, v in table.items()}
            )

    # ── public API ────────────────────────────────────────────────────────────

    def query_candidates(self, q, probes=2):
        """Return the candidate pool before exact distance re-ranking.

        For hash_width=1 (OR scheme), probes adjacent buckets are checked per
        table — identical to the original multi-probe LSH behaviour.

        For hash_width=k (AND scheme), the ℓ∞ neighbourhood of radius
        (probes−1) around the query's compound key is checked per table:
        (2p−1)^k compound keys.  With k ≥ 3 keep probes=1 (exact match).

        Parameters
        ----------
        q      : (d,) float32
        probes : ℓ∞ probe radius (1 = exact match, 2 = ±1 per dimension, …)

        Returns
        -------
        candidates : (m,) int32
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        q_projs = q @ self.proj_dirs.T                          # (M*k,)
        q_bids  = np.floor(
            (q_projs + self.offsets) / self.bucket_size
        ).astype(np.int32)                                      # (M*k,)

        # All ℓ∞-offsets in k dimensions with radius (probes-1)
        offsets = list(itertools.product(
            range(-(probes - 1), probes), repeat=self.k
        ))

        parts = []
        for m in range(self.M):
            base = q_bids[m * self.k : (m + 1) * self.k]       # (k,)
            t    = self.tables[m]
            if self.k == 1:
                b = int(base[0])
                for (delta,) in offsets:
                    key = b + delta
                    if key in t:
                        parts.append(t[key])
            else:
                base_list = base.tolist()
                for delta in offsets:
                    key = tuple(base_list[j] + delta[j] for j in range(self.k))
                    if key in t:
                        parts.append(t[key])

        if not parts:
            return np.array([], dtype=np.int32)
        return np.unique(np.concatenate(parts)).astype(np.int32)

    def query(self, q, k=10, probes=2):
        """Return the k approximate nearest neighbours of q.

        Parameters
        ----------
        q      : (d,) float32
        k      : number of neighbours to return
        probes : probe radius (see query_candidates)

        Returns
        -------
        points  : (k, d) float32
        dists   : (k,)   float32  — squared L2 distances
        indices : (k,)   int32
        """
        q     = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates(q, probes)
        if len(cands) == 0:
            cands = np.random.choice(self.n, k, replace=False).astype(np.int32)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]
