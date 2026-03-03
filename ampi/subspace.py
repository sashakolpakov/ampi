"""
AMPI Subspace: d_sub-dimensional subspace hashing with data-adaptive
equal-frequency grid quantization.

Motivation
──────────
A 1-D random projection collapses all d dimensions onto a line, losing
structure perpendicular to that line.  Projecting onto a d_sub-dimensional
random subspace (d_sub = 2, 3, 4, …) retains more geometry per "slot",
giving a better precision/recall tradeoff at the same candidate budget.

Algorithm
─────────
Build
  1. Draw L random orthonormal d_sub-frames {u₁, …, u_{d_sub}}ᵢ ⊆ ℝᵈ
     (each frame is a d × d_sub matrix with orthonormal columns, i = 1…L).
  2. For each frame i and each axis ax, project all n data points:
         vᵢ_ax(x) = uᵢ_ax · x      (one call to the shared project_data kernel)
  3. For each axis, divide the n projected values into B equal-frequency bins
     using B−1 percentile thresholds (data-adaptive: bin widths automatically
     widen in sparse regions and narrow in dense regions).
  4. Assign each point to a d_sub-dimensional grid cell
         (bᵢ₁, bᵢ₂, …, bᵢ_{d_sub}) ∈ {0,…,B−1}^{d_sub}
     encoded as a single integer key = b₁·B^{d_sub−1} + b₂·B^{d_sub−2} + … + b_{d_sub}.
  5. Store a dict per frame: key → array of data indices.

Query (query q, probes p, k neighbours)
  1. Project q onto each frame — O(L · d_sub · d).
  2. For each frame i, find q's grid cell, then probe all
     (2p−1)^{d_sub} cells in the ℓ∞ neighbourhood of radius p−1.
  3. Collect union across all L frames; re-rank by exact L2.

Why d_sub > 1 improves precision
─────────────────────────────────
False-positive rate (random far point in same cell):
    p₂ = 1 / B^{d_sub}        decreases exponentially with d_sub

Capture probability for NN at distance r:
    P_single ≈ ∏_{ax} P(|δ_ax| < Δ_ax/2)
    where  δ_ax ~ N(0, r²/d),  Δ_ax = bin width at the query location.

The ρ exponent (Datar 2004) satisfies
    ρ_{d_sub} = log(1/p₁) / log(1/p₂)
with p₁ = capture probability and p₂ = false-positive rate.
Increasing d_sub lowers p₁ and p₂ proportionally, keeping ρ roughly
constant asymptotically — but in practice the equal-frequency binning
adapts to the data distribution, giving noticeably better bucket balance
and higher precision than a raw 1-D hash with the same total bucket count B^{d_sub}.

Probe cost scales as (2p−1)^{d_sub} per frame, so keep p small (1–3)
for d_sub ≥ 3.

Parameters
──────────
num_projections : L   — number of independent random frames
bins_per_axis   : B   — equal-frequency bins per axis; B^{d_sub} total buckets
subspace_dim    : d_sub — dimension of each random subspace (2, 3, 4, …)
seed            : RNG seed
"""

import itertools
import numpy as np
from ._kernels import project_data, l2_distances


class AMPISubspaceIndex:
    """Approximate nearest-neighbour index via d_sub-dimensional subspace hashing.

    Parameters
    ----------
    data            : (n, d) array_like, float32
    num_projections : L — number of random subspace frames
    bins_per_axis   : B — equal-frequency bins per axis (B^{d_sub} total buckets)
    subspace_dim    : d_sub — subspace dimension (default 2; try 3 or 4 for
                      higher-dimensional data with large n)
    seed            : RNG seed
    """

    def __init__(self, data, num_projections=16, bins_per_axis=16,
                 subspace_dim=2, seed=0, directions=None):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.n, self.d = self.data.shape
        self.L    = num_projections
        self.B    = bins_per_axis
        self.dsub = subspace_dim

        rng = np.random.RandomState(seed)

        # ── Build orthonormal frames ───────────────────────────────────────────
        # subspaces[i] is a (d, d_sub) matrix with orthonormal columns.
        # If `directions` (L*d_sub, d) is provided, orthogonalise each block
        # of d_sub rows into a frame; otherwise draw random frames.
        self.subspaces = np.zeros((num_projections, self.d, subspace_dim),
                                  dtype=np.float32)
        if directions is not None:
            dirs = np.ascontiguousarray(directions, dtype=np.float32)
            for i in range(num_projections):
                block = dirs[i * subspace_dim:(i + 1) * subspace_dim].T  # (d, d_sub)
                Q, _ = np.linalg.qr(block)
                self.subspaces[i] = Q[:, :subspace_dim]
        else:
            for i in range(num_projections):
                Q, _ = np.linalg.qr(rng.randn(self.d, subspace_dim).astype(np.float32))
                self.subspaces[i] = Q  # (d, d_sub)

        # ── Project data onto each axis of each frame ─────────────────────────
        # proj_by_axis[ax] : (L, n) float32
        proj_by_axis = [
            project_data(self.data, self.subspaces[:, :, ax])
            for ax in range(subspace_dim)
        ]

        # ── Compute equal-frequency thresholds per frame and axis ─────────────
        # thresholds[i][ax] : (B-1,) float32
        q_pcts = np.linspace(0, 100, self.B + 1)[1:-1]  # B-1 percentile values

        self.thresholds = []
        for i in range(num_projections):
            frame_thresh = []
            for ax in range(subspace_dim):
                vals = proj_by_axis[ax][i]          # (n,)
                frame_thresh.append(
                    np.percentile(vals, q_pcts).astype(np.float32)
                )
            self.thresholds.append(frame_thresh)    # list[list[array(B-1)]]

        # ── Build hash tables ─────────────────────────────────────────────────
        self.tables = []
        for i in range(num_projections):
            # bin index per axis: (d_sub, n) int32, values in [0, B-1]
            bin_ids = np.empty((subspace_dim, self.n), dtype=np.int32)
            for ax in range(subspace_dim):
                raw = np.digitize(proj_by_axis[ax][i],
                                  self.thresholds[i][ax])  # [0, B]
                bin_ids[ax] = np.clip(raw, 0, self.B - 1)

            # Encode to single int: key = b₁·B^{d_sub−1} + … + b_{d_sub}
            keys = np.zeros(self.n, dtype=np.int32)
            for ax in range(subspace_dim):
                keys = keys * self.B + bin_ids[ax]

            table = {}
            for key in np.unique(keys):
                table[int(key)] = np.where(keys == key)[0].astype(np.int32)
            self.tables.append(table)

    # ── Public API ────────────────────────────────────────────────────────────

    def query_candidates(self, q, probes=2):
        """Collect candidates from the (2·probes−1)^{d_sub} neighbouring cells
        per frame, before exact distance re-ranking.

        Parameters
        ----------
        q      : (d,) float32
        probes : p — ℓ∞ probe radius in grid space;
                 checks (2p−1)^{d_sub} cells per frame.
                 Keep ≤ 3 for d_sub ≥ 3.

        Returns
        -------
        candidates : (m,) int32
        """
        q = np.ascontiguousarray(q, dtype=np.float32)

        # Project q onto each axis of each frame: q_projs[ax] shape (L,)
        q_projs = [
            q @ self.subspaces[:, :, ax].T
            for ax in range(self.dsub)
        ]

        parts = []
        offsets = list(itertools.product(range(-(probes - 1), probes),
                                         repeat=self.dsub))

        for i in range(self.L):
            # Find query's bin on each axis
            q_bins = np.empty(self.dsub, dtype=np.int32)
            for ax in range(self.dsub):
                raw = int(np.digitize(q_projs[ax][i], self.thresholds[i][ax]))
                q_bins[ax] = np.clip(raw, 0, self.B - 1)

            table = self.tables[i]
            for off in offsets:
                key = 0
                for ax in range(self.dsub):
                    b = int(np.clip(q_bins[ax] + off[ax], 0, self.B - 1))
                    key = key * self.B + b
                if key in table:
                    parts.append(table[key])

        if not parts:
            return np.array([], dtype=np.int32)
        return np.unique(np.concatenate(parts)).astype(np.int32)

    def query(self, q, k=10, probes=2):
        """Return the k approximate nearest neighbours of q.

        Parameters
        ----------
        q      : (d,) float32
        k      : number of neighbours to return
        probes : probe radius (larger → higher recall, more candidates)

        Returns
        -------
        points  : (k, d) float32
        dists   : (k,)   float32  — squared L2 distances
        indices : (k,)   int32
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates(q, probes)
        if len(cands) == 0:
            cands = np.random.choice(self.n, k, replace=False).astype(np.int32)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]
