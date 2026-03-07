"""
AMPI Affine Fan Index: k-means partition + random affine fan cones
+ sorted-projection search.

Architecture:
  1. Coarse partition: k-means (mini-batch, BLAS-accelerated).
  2. Within each cluster: F random unit-vector fan axes → cone assignment
     on centred data → sorted projections per cone.
  3. Query: find nearest cp clusters, best fp cones per cluster via
     |aₖ · (q−μ_c)_hat|, collect candidates via sorted-projection window,
     L2 rerank.

Random axes work as well as geometry-guided ones at F=128 in d=128
(near-full coverage of the space regardless), with zero build overhead.
"""

import numpy as np
from ._kernels import jit_union_query, l2_distances


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

    def __init__(self, data, nlist=None, num_fans=16, seed=0, cone_top_k=1):

        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.n, self.d = self.data.shape
        self.F = num_fans
        self.cone_top_k = max(1, int(cone_top_k))

        if nlist is None:
            nlist = max(16, int(np.sqrt(self.n)))
        self.nlist = nlist

        # ── Global fan directions (computed once) ────────────────────────
        rng  = np.random.RandomState(seed)
        axes = rng.randn(self.F, self.d).astype(np.float32)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        self.axes = axes  # (F, d)

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
            gi = self.cluster_global[c]
            if cones is None or len(gi) == 0:
                if len(gi) > 0:
                    parts.append(gi)
                continue

            # Fast path: fp >= F means every cone is visited and each cone's
            # window (w >> n_f at typical scales) returns all its points anyway.
            # Skip unnecessary JIT calls and just return the whole cluster.
            if fan_probes >= self.F:
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
                if cone['n'] <= 2 * window_size:
                    parts.append(cone['global_idx'])
                else:
                    local = jit_union_query(
                        cone['sorted_idxs'],
                        cone['sorted_projs'],
                        q_proj, window_size,
                    )
                    parts.append(cone['global_idx'][local])

        if not parts:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(parts))

    def query(self, q, k=10, window_size=200, probes=10, fan_probes=2):
        """Adaptive sorted-projection search.

        Starts with a small window and expands only until the projection lower
        bound guarantees no unvisited point can improve the current top-k.
        window_size is a ceiling; most queries stop well before reaching it.

        Stopping condition per cone: there exists an axis l whose window
        boundary projection gap >= sqrt(kth_sq). Since L2(x,q) >= |proj_l(x-q)|
        for any unit vector l, those unvisited points cannot enter the top-k.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)

        # Collect cone contexts once so cluster/cone selection is not repeated
        # on every window expansion.
        cone_ctxs = []   # list of (cone_dict, q_proj_float32)
        fallback_parts = []

        for c in map(int, self._best_clusters(q, probes)):
            cones = self.cluster_cones[c]
            gi    = self.cluster_global[c]
            if not len(gi):
                continue
            if cones is None or fan_probes >= self.F:
                fallback_parts.append(gi)
                continue
            centroid   = self.centroids[c]
            q_centered = q - centroid
            q_proj     = np.ascontiguousarray(
                (q_centered @ self.axes.T).astype(np.float32)
            )
            for f in map(int, self._best_fan_cones(q_centered, fan_probes)):
                if f < len(cones) and cones[f] is not None:
                    cone_ctxs.append((cones[f], q_proj))

        w = max(k, 8)
        cands = np.zeros(0, dtype=np.int32)

        while True:
            parts = list(fallback_parts)
            for cone, q_proj in cone_ctxs:
                if cone['n'] <= 2 * w:
                    parts.append(cone['global_idx'])
                else:
                    local = jit_union_query(
                        cone['sorted_idxs'], cone['sorted_projs'], q_proj, w,
                    )
                    parts.append(cone['global_idx'][local])

            cands = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int32)

            if w >= window_size or len(cands) < k:
                break

            dists    = l2_distances(self.data, q, cands)
            kth_sq   = float(np.partition(dists, k - 1)[k - 1])
            kth_proj = np.sqrt(max(0.0, kth_sq))

            # A cone is covered when at least one axis l has both sides of its
            # window boundary further than kth_proj in projection space.
            # Any unvisited point on that axis has L2 >= its projection gap,
            # so it cannot improve the top-k.
            all_covered = True
            for cone, q_proj in cone_ctxs:
                if cone['n'] <= 2 * w:
                    continue  # already exhausted
                n_f     = cone['n']
                covered = False
                for l in range(self.F):
                    sp  = cone['sorted_projs'][l]
                    pos = int(np.searchsorted(sp, q_proj[l]))
                    lo  = max(0, pos - w)
                    hi  = min(n_f, pos + w)
                    gap_right = float(sp[hi] - q_proj[l]) if hi < n_f else np.inf
                    gap_left  = float(q_proj[l] - sp[lo - 1]) if lo > 0 else np.inf
                    if min(gap_right, gap_left) >= kth_proj:
                        covered = True
                        break
                if not covered:
                    all_covered = False
                    break

            if all_covered:
                break

            w = min(w * 2, window_size)

        if len(cands) < k:
            cands = np.arange(min(k, self.n), dtype=np.int32)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]

