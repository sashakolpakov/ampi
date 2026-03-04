"""
AMPI Affine Fan Index: FAISS k-means partition + affine cone refinement
+ sorted-projection voting.

Architecture (3 levels):
  1. Coarse partition: FAISS k-means (optimised C++ with BLAS).
  2. Within each cluster: affine fan cones around the cluster centroid.
     Global geometry-guided directions (computed once) → cone assignment on
     centred data → sorted projections per cone.
  3. Within each cone: sorted-projection vote/union query.

Query:
  1. Find P nearest centroids by L2 distance.
  2. For each probed cluster, find the best F cones via |aₖ · (q−μ_c)_hat|.
  3. Within each cone, sorted-projection vote/union → candidates.
  4. L2 rerank survivors.

The innovation is not the partition (k-means is standard) but:
  - Affine fan cones within clusters partition the cluster members by
    direction from the centroid, so within-cluster search is sublinear.
  - Sorted-projection voting within each cone further prunes candidates:
    true NNs score high votes across all projection axes, impostors don't.
  - This replaces IVF's exhaustive within-cluster scan, reducing candidate
    count while maintaining recall.
"""

import numpy as np
from ._kernels import jit_union_query, jit_vote_query, l2_distances


def _bootstrap_nn_pairs(data, S, rng):
    """Approximate NN pairs via a single random projection → (S, 2) int32."""
    n, d = data.shape
    a = rng.randn(d).astype(np.float32)
    a /= np.linalg.norm(a)
    order = np.argsort(data @ a)
    adj   = np.stack([order[:-1], order[1:]], axis=1)
    sel   = rng.choice(len(adj), min(S, len(adj)), replace=False)
    return adj[sel]


def _score_directions(data, candidates, pairs, rng):
    """score(a) = mean|a·(anchor−rand)| / mean|a·(anchor−nn)| → (C,) float32."""
    n        = data.shape[0]
    S        = len(pairs)
    anchors  = data[pairs[:, 0]]
    nns      = data[pairs[:, 1]]
    randoms  = data[rng.choice(n, S, replace=True)]
    nn_proj   = np.abs((anchors - nns)     @ candidates.T)   # (S, C)
    rand_proj = np.abs((anchors - randoms) @ candidates.T)   # (S, C)
    return (rand_proj.mean(0) / (nn_proj.mean(0) + 1e-8)).astype(np.float32)


def _global_power_iter(data, directions, steps):
    """a ← X^T(Xa)/‖·‖ over full dataset — vectorised over all L at once."""
    A = directions.T.astype(np.float64)          # (d, L)
    for _ in range(steps):
        XA = data @ A                            # (n, L)
        A  = data.T @ XA                         # (d, L)
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        A /= np.where(norms < 1e-10, 1.0, norms)
    return A.T.astype(np.float32)                # (L, d)


def _local_power_iter(data, directions, pairs, steps):
    """a ← D^T(Da)/‖·‖ where D = NN-pair difference vectors."""
    anchors = data[pairs[:, 0]].astype(np.float64)
    nns     = data[pairs[:, 1]].astype(np.float64)
    D       = anchors - nns                      # (S, d)
    A = directions.T.astype(np.float64)          # (d, L)
    for _ in range(steps):
        DA = D @ A                               # (S, L)
        A  = D.T @ DA                            # (d, L)
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        A /= np.where(norms < 1e-10, 1.0, norms)
    return A.T.astype(np.float32)                # (L, d)


def geometry_guided_directions(data, L, C_factor=5, S=500,
                                power_iter=1, local=True, seed=0):
    """Select L geometry-aware unit vectors for projecting ``data``."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    rng  = np.random.RandomState(seed)
    n, d = data.shape
    C    = C_factor * L

    pairs = _bootstrap_nn_pairs(data, S, rng)

    ii = rng.choice(n, C, replace=True)
    jj = rng.choice(n, C, replace=True)
    jj[ii == jj] = (jj[ii == jj] + 1) % n
    diffs = (data[ii] - data[jj]).astype(np.float64)
    norms = np.linalg.norm(diffs, axis=1)
    valid = norms > 1e-8
    diffs = diffs[valid] / norms[valid, None]
    shortfall = C - len(diffs)
    if shortfall > 0:
        extra = rng.randn(shortfall, d)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        diffs = np.vstack([diffs, extra])

    candidates = diffs[:C].astype(np.float32)
    scores     = _score_directions(data, candidates, pairs, rng)
    top_idx    = np.argsort(scores)[::-1][:L]
    selected   = candidates[top_idx].copy()

    if power_iter > 0:
        if local:
            selected = _local_power_iter(data, selected, pairs, steps=power_iter)
        else:
            selected = _global_power_iter(data, selected, steps=power_iter)

    return selected


def _blas_assign(data, centroids, data_sq=None):
    """Assign each row of data to nearest centroid using BLAS gemm.

    ‖x−c‖² = ‖x‖² + ‖c‖² − 2x·c.  The x·c term is a single gemm.

    Parameters
    ----------
    data      : (n, d) float32
    centroids : (k, d) float32
    data_sq   : (n,) float32 — precomputed ‖x‖², or None

    Returns
    -------
    assignments : (n,) int32
    """
    if data_sq is None:
        data_sq = np.sum(data ** 2, axis=1)
    cent_sq = np.sum(centroids ** 2, axis=1)                # (k,)
    n = data.shape[0]
    chunk = 16384  # limit peak memory: chunk × k float32
    assignments = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dots = data[start:end] @ centroids.T                # (chunk, k)
        d2 = data_sq[start:end, None] + cent_sq[None, :] - 2 * dots
        assignments[start:end] = np.argmin(d2, axis=1).astype(np.int32)
    return assignments


def _mini_batch_kmeans(data, k, seed=0):
    """Mini-batch k-means with BLAS-accelerated assignment.

    Strategy:
      1. Subsample s = min(50k, n) points for the working set.
      2. Random init: k distinct points from the subsample.
      3. Full Lloyd's iterations on the subsample in low-overhead form
         (BLAS gemm for assignment, bincount for centroid update).
      4. Final full-data assignment via BLAS.

    Lloyd's on the subsample costs O(T × s × k × d).  For s=50K, k=1000,
    d=128, T=15: ~96 billion flops, ~10s.  But the gemm is well-optimised.
    Final assignment is O(n × k × d): for n=1M, ~128B flops.

    Parameters
    ----------
    data : (n, d) float32
    k    : clusters
    seed : RNG seed

    Returns
    -------
    centroids   : (k, d) float32
    assignments : (n,) int32
    """
    rng = np.random.RandomState(seed)
    n, d = data.shape

    # Subsample
    s = min(50_000, n)
    sample_idx = rng.choice(n, s, replace=False)
    sample = data[sample_idx].copy()
    sample_sq = np.sum(sample ** 2, axis=1)                # (s,)

    # Random init
    centroids = sample[rng.choice(s, k, replace=False)].copy()

    # Lloyd's on subsample
    for it in range(20):
        # BLAS assignment: ‖x-c‖² = ‖x‖² + ‖c‖² - 2x·c
        cent_sq = np.sum(centroids ** 2, axis=1)           # (k,)
        dots = sample @ centroids.T                        # (s, k)
        d2 = sample_sq[:, None] + cent_sq[None, :] - 2 * dots
        assign = np.argmin(d2, axis=1).astype(np.int32)

        # Centroid update via bincount
        counts = np.bincount(assign, minlength=k)
        new_centroids = np.zeros((k, d), dtype=np.float64)
        for j in range(d):
            new_centroids[:, j] = np.bincount(
                assign, weights=sample[:, j].astype(np.float64), minlength=k
            )
        alive = counts > 0
        new_centroids[alive] /= counts[alive, None]
        # Respawn dead clusters at random sample points
        dead = ~alive
        if dead.any():
            n_dead = dead.sum()
            new_centroids[dead] = sample[rng.choice(s, n_dead, replace=False)].astype(np.float64)
        new_c = new_centroids.astype(np.float32)
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c

    # Final full-data assignment
    assignments = _blas_assign(data, centroids)
    return centroids, assignments


class AMPIAffineFanIndex:
    """K-means partition + affine fan cones + sorted-projection search.

    Parameters
    ----------
    data         : (n, d) float32
    nlist        : number of coarse clusters (default: sqrt(n))
    num_fans     : F — fan cones per cluster (also = sort directions)
    C_factor, S, power_iter, seed
                 : passed to geometry_guided_directions (called once globally)
    """

    def __init__(self, data, nlist=None, num_fans=16,
                 C_factor=5, S=500, power_iter=1, seed=0,
                 cone_top_k=1):

        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.n, self.d = self.data.shape
        self.F = num_fans
        self.L = num_fans
        self.cone_top_k = max(1, int(cone_top_k))

        if nlist is None:
            nlist = max(16, int(np.sqrt(self.n)))
        self.nlist = nlist

        # ── Global fan directions (computed once) ────────────────────────
        self.axes = geometry_guided_directions(
            self.data, self.F,
            C_factor=C_factor, S=S, power_iter=power_iter, seed=seed,
        )  # (F, d)

        # ── K-means partition ────────────────────────────────────────────
        self.centroids, assignments = _mini_batch_kmeans(
            self.data, nlist, seed=seed,
        )

        # ── Per-cluster: affine fan cones using global axes ──────────────
        self.cluster_global = []
        self.cluster_cones  = []

        for c in range(nlist):
            c_idx = np.where(assignments == c)[0].astype(np.int32)
            self.cluster_global.append(c_idx)

            if len(c_idx) < 2:
                self.cluster_cones.append(None)
                continue

            c_data = self.data[c_idx]
            centroid = self.centroids[c]
            centered = c_data - centroid

            # Project centred data onto global axes
            all_projs = (centered @ self.axes.T).astype(np.float32)

            # Affine cone assignment with top-K multi-assignment.
            # Each point is assigned to its top cone_top_k cones by |normalised
            # projection|.  cone_top_k=1 recovers the original hard argmax.
            # K=2,3 doubles/triples build memory but gives bounded overhead,
            # unlike a score-threshold approach which explodes in high dimensions.
            norms = np.linalg.norm(centered, axis=1, keepdims=True).astype(np.float32)
            norms = np.where(norms < 1e-10, 1.0, norms)
            normed   = np.abs(all_projs / norms)                        # (n_c, F)
            K        = min(self.cone_top_k, self.F)
            top_cones = np.argpartition(-normed, K - 1, axis=1)[:, :K]  # (n_c, K)
            soft_mask = np.zeros((normed.shape[0], self.F), dtype=bool)
            soft_mask[np.arange(normed.shape[0])[:, None], top_cones] = True

            cones = []
            for f in range(self.F):
                f_local_idx = np.where(soft_mask[:, f])[0]
                n_f = len(f_local_idx)
                if n_f == 0:
                    cones.append(None)
                    continue

                f_global = c_idx[f_local_idx]
                f_projs = all_projs[f_local_idx]

                s_idxs  = np.empty((self.F, n_f), dtype=np.int32)
                s_projs = np.empty((self.F, n_f), dtype=np.float32)
                for l in range(self.F):
                    o = np.argsort(f_projs[:, l])
                    s_idxs[l]  = o.astype(np.int32)
                    s_projs[l] = f_projs[o, l]

                cones.append(dict(
                    global_idx=f_global,
                    sorted_idxs=s_idxs,
                    sorted_projs=s_projs,
                    n=n_f,
                ))

            self.cluster_cones.append(cones)

    # ── internal ──────────────────────────────────────────────────────────

    def _best_clusters(self, q, probes):
        d2 = np.sum((self.centroids - q) ** 2, axis=1)
        return np.argsort(d2)[:probes]

    def _best_fan_cones(self, q_centered, fan_probes):
        q_norm = float(np.linalg.norm(q_centered))
        if q_norm < 1e-10:
            return np.arange(min(fan_probes, self.F), dtype=np.int32)
        proj = q_centered @ self.axes.T / q_norm
        return np.argsort(-np.abs(proj))[:fan_probes]

    # ── public API ────────────────────────────────────────────────────────

    def query_candidates(self, q, window_size=50, probes=10, fan_probes=2):
        q = np.ascontiguousarray(q, dtype=np.float32)
        clusters = self._best_clusters(q, probes)

        parts = []
        for c in clusters:
            c = int(c)
            cones = self.cluster_cones[c]
            if cones is None:
                gi = self.cluster_global[c]
                if len(gi) > 0:
                    parts.append(gi)
                continue

            centroid = self.centroids[c]
            q_centered = q - centroid
            q_proj = np.ascontiguousarray((q_centered @ self.axes.T).astype(np.float32))

            best_cones = self._best_fan_cones(q_centered, fan_probes)
            for f in best_cones:
                f = int(f)
                if f >= len(cones) or cones[f] is None:
                    continue
                cone = cones[f]
                local = jit_union_query(
                    cone['sorted_idxs'],
                    cone['sorted_projs'],
                    q_proj, window_size,
                )
                parts.append(cone['global_idx'][local])

        if not parts:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(parts))

    def query_candidates_voting(self, q, window_size=50, probes=10,
                                fan_probes=2, min_votes=None):
        q = np.ascontiguousarray(q, dtype=np.float32)
        if min_votes is None:
            min_votes = max(1, self.L // 4)
        clusters = self._best_clusters(q, probes)

        parts = []
        for c in clusters:
            c = int(c)
            cones = self.cluster_cones[c]
            if cones is None:
                gi = self.cluster_global[c]
                if len(gi) > 0:
                    parts.append(gi)
                continue

            centroid = self.centroids[c]
            q_centered = q - centroid
            q_proj = np.ascontiguousarray((q_centered @ self.axes.T).astype(np.float32))

            best_cones = self._best_fan_cones(q_centered, fan_probes)
            for f in best_cones:
                f = int(f)
                if f >= len(cones) or cones[f] is None:
                    continue
                cone = cones[f]
                local = jit_vote_query(
                    cone['sorted_idxs'],
                    cone['sorted_projs'],
                    q_proj, window_size, min_votes,
                )
                if len(local) == 0:
                    local = jit_union_query(
                        cone['sorted_idxs'],
                        cone['sorted_projs'],
                        q_proj, window_size,
                    )
                parts.append(cone['global_idx'][local])

        if not parts:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(parts))

    def query(self, q, k=10, window_size=50, probes=10, fan_probes=2):
        q     = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates(q, window_size, probes, fan_probes)
        if len(cands) < k:
            cands = np.arange(min(k, self.n), dtype=np.int32)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]

    def query_voting(self, q, k=10, window_size=50, probes=10,
                     fan_probes=2, min_votes=None):
        q     = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates_voting(q, window_size, probes,
                                             fan_probes, min_votes)
        if len(cands) < k:
            cands = np.arange(min(k, self.n), dtype=np.int32)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]
