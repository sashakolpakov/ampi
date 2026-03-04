"""
AMPI Binary: sorted projections + binary search + adaptive sliding window.

Algorithm
─────────
Build
  1. Draw L random unit vectors a₁, …, aL ∈ ℝᵈ.
  2. For each direction i, project all n data points: vᵢ(x) = aᵢ · x.
  3. Sort the n values per direction; store sorted values and the
     original indices that produced each sorted value.

Query (for query q, window_size w, k neighbours)
  1. Project q onto each direction: qᵢ = aᵢ · q  — O(Ld).
  2. Binary-search for qᵢ in sorted projection array i.
  3. Take the w points on each side of the insertion point.
     Candidate set = union across all L directions — at most 2wL points.
  4. Re-rank candidates by exact L2 distance; return top k.

Key property: adaptive density
  Unlike hash-based methods, the window always contains exactly 2w candidates
  per projection regardless of local data density.  In dense regions this
  gives high precision; in sparse regions it reaches further to find neighbours.
"""

import numpy as np
from ._kernels import project_data, l2_distances, jit_union_query


class AMPIBinaryIndex:
    """Approximate nearest-neighbour index via sorted 1-D projections.

    Parameters
    ----------
    data            : (n, d) array_like, float32
    num_projections : L — number of random directions (more → higher recall,
                      larger candidate set, slower query)
    seed            : RNG seed for the projection directions
    """

    def __init__(self, data, num_projections=16, seed=0):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.n, self.d = self.data.shape
        self.L = num_projections

        rng = np.random.RandomState(seed)
        dirs = rng.randn(num_projections, self.d).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        self.proj_dirs = dirs

        projs = project_data(self.data, self.proj_dirs)  # (L, n)

        self.sorted_idxs  = np.empty((self.L, self.n), dtype=np.int32)
        self.sorted_projs = np.empty((self.L, self.n), dtype=np.float32)
        for i in range(self.L):
            order = np.argsort(projs[i])
            self.sorted_idxs[i]  = order
            self.sorted_projs[i] = projs[i, order]

    # ── public API ────────────────────────────────────────────────────────────

    def query_candidates(self, q, window_size=100):
        """Return the union of the 2·window_size nearest neighbours in each
        projection direction, before exact distance re-ranking.

        Parameters
        ----------
        q           : (d,) float32
        window_size : w — half-window size per projection

        Returns
        -------
        candidates : (m,) int32  — unique data indices, m ≤ 2·w·L
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        q_projs = np.ascontiguousarray(q @ self.proj_dirs.T, dtype=np.float32)
        return jit_union_query(self.sorted_idxs, self.sorted_projs, q_projs, window_size)

    def query(self, q, k=10, window_size=100):
        """Return the k approximate nearest neighbours of q.

        Parameters
        ----------
        q           : (d,) float32
        k           : number of neighbours to return
        window_size : half-window size per projection (larger → higher recall)

        Returns
        -------
        points  : (k, d) float32
        dists   : (k,)   float32  — squared L2 distances
        indices : (k,)   int32    — indices into the original data array
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates(q, window_size)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]
