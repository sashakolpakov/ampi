"""
AMPI Principal Fan Index: principal-direction cone partition on the unit sphere.

Build
  1. Compute K geometry-guided unit directions {a₀, …, a_{K-1}} — the fan axes.
  2. L2-normalise every data point: x̂ = x/‖x‖.
  3. Assign each point to its dominant axis on the unit sphere:
       cone(x) = argmax_k |aₖ · x̂|
     The 2K half-spaces are the Voronoi cells of {±aₖ} on the unit sphere.
     Normalising first ensures the partition is purely directional: two points
     at the same angular position land in the same cone regardless of norm,
     so nearby points on real-world datasets (which concentrate on a low-
     dimensional manifold) share a cone with high probability.
  4. Within each cone, sort by raw projections on all K axes (no extra passes).

Query
  1. Normalise q: q̂ = q/‖q‖.
  2. Rank cones by |aₖ · q̂|; probe the top-P cones.
  3. Union-query within each probed cone using jit_union_query on raw projections.
  4. Re-rank survivors by exact L2.

Note on isotropic Gaussian data
  For x ~ N(0, I_d), direction x̂ and magnitude ‖x‖ are independent, so
  unit-sphere normalisation gives the same cone assignment as the raw argmax.
  Fan's principal-cone partition is designed for structured data (real images,
  speech, text) where the manifold geometry aligns with a directional partition.
  On pure isotropic noise, no angular partition helps — and the nearest-neighbour
  problem itself is degenerate (all pairwise distances concentrate around the same
  value, so R@k measures arbitrary index assignments rather than geometry).
"""

import numpy as np
from ._kernels import jit_union_query, l2_distances


class AMPIPrincipalFanIndex:
    """Approximate nearest-neighbour index via principal-direction cone partition.

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

        # K geometry-guided directions — fan axes AND sort directions
        self.axes = geometry_guided_directions(
            self.data, self.K,
            C_factor=C_factor, S=S, power_iter=power_iter, seed=seed,
        )  # (K, d)

        # Project all data onto every axis — reused for sorting within cones
        all_projs = (self.data @ self.axes.T).astype(np.float32)  # (n, K)

        # Cone assignment: dominant axis on the unit sphere.
        # Normalise first so the partition is purely directional; two points at
        # the same angular position land in the same cone regardless of norm.
        norms = np.linalg.norm(self.data, axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normed_projs = all_projs / norms           # (n, K) — unit-sphere projections
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
                s_idxs[l]  = o.astype(np.int32)   # LOCAL indices (0..n_k-1)
                s_projs[l] = sub_p[o, l]

            self.cone_sorted_idxs.append(s_idxs)
            self.cone_sorted_projs.append(s_projs)

    # ── internal ──────────────────────────────────────────────────────────────

    def _best_cones(self, q, probes):
        """Indices of the probes cones with largest |aₖ · q̂| (unit-sphere)."""
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-10:
            return np.arange(probes, dtype=np.int32)
        q_hat_proj = q @ self.axes.T / q_norm
        return np.argsort(-np.abs(q_hat_proj))[:probes]

    # ── public API ────────────────────────────────────────────────────────────

    def query_candidates(self, q, window_size=50, probes=2):
        q      = np.ascontiguousarray(q, dtype=np.float32)
        q_proj = np.ascontiguousarray(q @ self.axes.T, dtype=np.float32)
        cones  = self._best_cones(q, probes)

        parts = []
        for c in cones:
            if self.cone_sorted_idxs[c] is None:
                continue
            local = jit_union_query(
                self.cone_sorted_idxs[c],
                self.cone_sorted_projs[c],
                q_proj, window_size,
            )
            parts.append(self.cone_global[c][local])   # map local → global
        if not parts:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(parts))

    def query(self, q, k=10, window_size=50, probes=2):
        q     = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates(q, window_size, probes)
        if len(cands) < k:
            cands = np.arange(min(k, self.n), dtype=np.int32)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]
