"""
AMPI Principal Fan Index: affine cone partition around the data centroid.

Build
  1. Compute the data centroid μ.
  2. Compute K geometry-guided unit directions {a₀, …, a_{K-1}} on centred
     data (x − μ) — the fan axes radiate from where the data actually lives.
  3. Assign each point to its dominant axis from the centroid:
       cone(x) = argmax_k |aₖ · (x̂ − μ̂)|
     where (x̂ − μ̂) is the unit-norm centred displacement.  The 2K half-spaces
     are the Voronoi cells of {±aₖ} around the centroid on the unit sphere of
     displacements.  This is an *affine* fan: the partition adapts to the data's
     location, not just its spread through the origin.
  4. Within each cone, sort by centred projections aₖ · (x − μ) on all K axes.

Query
  1. Centre q: δ = q − μ;  δ̂ = δ/‖δ‖.
  2. Rank cones by |aₖ · δ̂|; probe the top-P cones.
  3. Within each probed cone, run sorted-projection search (union or voting)
     on centred projections.
  4. Re-rank survivors by exact L2 against original (uncentred) data.

Why affine beats origin-centred cones
  Real datasets (SIFT, GloVe, MNIST) cluster far from the origin.  Origin-
  centred cones concentrate most data into a few cones near the centroid
  direction, leaving others empty.  Affine fans partition the data by direction
  *from the centroid*, which tracks actual cluster structure.
"""

import numpy as np
from ._kernels import jit_union_query, jit_vote_query, l2_distances


class AMPIPrincipalFanIndex:
    """Approximate nearest-neighbour index via affine cone partition.

    Parameters
    ----------
    data            : (n, d) float32
    num_fans        : K — number of cones; larger K → smaller cones → lower
                      per-cone candidate count but more boundary sensitivity
    C_factor, S, power_iter, seed
                    : passed to geometry_guided_directions
    """

    def __init__(self, data, num_fans=32,
                 C_factor=5, S=500, power_iter=1, seed=0):
        from .tomography import geometry_guided_directions

        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.n, self.d = self.data.shape
        self.K = min(num_fans, self.n)

        # Centroid — the origin of the affine fan
        self.centroid = self.data.mean(axis=0).astype(np.float32)
        centered = self.data - self.centroid  # (n, d)

        # K geometry-guided directions on centred data
        self.axes = geometry_guided_directions(
            centered, self.K,
            C_factor=C_factor, S=S, power_iter=power_iter, seed=seed,
        )  # (K, d)

        # Centred projections — used for both assignment and within-cone search
        all_projs = (centered @ self.axes.T).astype(np.float32)  # (n, K)

        # Affine cone assignment: dominant axis on centred unit sphere
        norms = np.linalg.norm(centered, axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normed_projs = all_projs / norms
        assignment = np.argmax(np.abs(normed_projs), axis=1)  # (n,)

        self.cone_global       = []   # K × (n_k,) int32  global index map
        self.cone_sorted_idxs  = []   # K × (K, n_k) int32  LOCAL indices
        self.cone_sorted_projs = []   # K × (K, n_k) float32

        for k in range(self.K):
            idx = np.where(assignment == k)[0].astype(np.int32)
            if len(idx) == 0:
                self.cone_global.append(np.zeros(0, dtype=np.int32))
                self.cone_sorted_idxs.append(None)
                self.cone_sorted_projs.append(None)
                continue

            self.cone_global.append(idx)

            sub_p = all_projs[idx]                   # (n_k, K)
            n_k   = len(idx)
            s_idxs  = np.empty((self.K, n_k), dtype=np.int32)
            s_projs = np.empty((self.K, n_k), dtype=np.float32)
            for l in range(self.K):
                o = np.argsort(sub_p[:, l])
                s_idxs[l]  = o.astype(np.int32)
                s_projs[l] = sub_p[o, l]

            self.cone_sorted_idxs.append(s_idxs)
            self.cone_sorted_projs.append(s_projs)

    # ── internal ──────────────────────────────────────────────────────────────

    def _best_cones(self, q_centered, probes):
        """Indices of the probes cones with largest |aₖ · δ̂| (centred unit sphere)."""
        q_norm = float(np.linalg.norm(q_centered))
        if q_norm < 1e-10:
            return np.arange(probes, dtype=np.int32)
        q_hat_proj = q_centered @ self.axes.T / q_norm
        return np.argsort(-np.abs(q_hat_proj))[:probes]

    # ── public API ────────────────────────────────────────────────────────────

    def query_candidates(self, q, window_size=50, probes=2):
        q         = np.ascontiguousarray(q, dtype=np.float32)
        q_centered = q - self.centroid
        q_proj    = np.ascontiguousarray(q_centered @ self.axes.T, dtype=np.float32)
        cones     = self._best_cones(q_centered, probes)

        parts = []
        for c in cones:
            if self.cone_sorted_idxs[c] is None:
                continue
            n_k = self.cone_sorted_idxs[c].shape[1]
            w = max(1, int(window_size * n_k / max(1, self.n // self.K)))
            local = jit_union_query(
                self.cone_sorted_idxs[c],
                self.cone_sorted_projs[c],
                q_proj, w,
            )
            parts.append(self.cone_global[c][local])
        if not parts:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(parts))

    def query_candidates_voting(self, q, window_size=50, probes=2, min_votes=None):
        """Like query_candidates but uses vote-thresholding within each cone."""
        q         = np.ascontiguousarray(q, dtype=np.float32)
        q_centered = q - self.centroid
        q_proj    = np.ascontiguousarray(q_centered @ self.axes.T, dtype=np.float32)
        cones     = self._best_cones(q_centered, probes)
        if min_votes is None:
            min_votes = max(1, self.K // 8)

        parts = []
        for c in cones:
            if self.cone_sorted_idxs[c] is None:
                continue
            n_k = self.cone_sorted_idxs[c].shape[1]
            w   = max(1, int(window_size * n_k / max(1, self.n // self.K)))
            local = jit_vote_query(
                self.cone_sorted_idxs[c],
                self.cone_sorted_projs[c],
                q_proj, w, min_votes,
            )
            if len(local) == 0:
                local = jit_union_query(
                    self.cone_sorted_idxs[c],
                    self.cone_sorted_projs[c],
                    q_proj, w,
                )
            parts.append(self.cone_global[c][local])
        if not parts:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(parts))

    def query_voting(self, q, k=10, window_size=50, probes=2, min_votes=None):
        """Voting-mode query: sharply fewer candidates, same or better recall."""
        q     = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates_voting(q, window_size, probes, min_votes)
        if len(cands) < k:
            cands = np.arange(min(k, self.n), dtype=np.int32)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]

    def query(self, q, k=10, window_size=50, probes=2):
        q     = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates(q, window_size, probes)
        if len(cands) < k:
            cands = np.arange(min(k, self.n), dtype=np.int32)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]
