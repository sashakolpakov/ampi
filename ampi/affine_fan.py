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

Cone representation
-------------------
When the compiled C++ extension is available, each cone is a SortedCone
object (mutable, supports streaming insert/delete).  Otherwise cones fall
back to a _DictCone wrapper around numpy sorted arrays.  Both expose the
same interface: .size(), .all_ids(), .query(), .is_covered().

Streaming mutations
-------------------
add(x)         — insert one vector, returns its global_id.
delete(id)     — logical tombstone; compacts automatically when the
                 per-cluster tombstone fraction exceeds _TOMBSTONE_THRESHOLD.
update(id, x)  — delete + insert.

Cluster assignment uses nearest-centroid (top-K by L2).

Drift detection: per cluster a d×d covariance EMA tracks the displacement
(x − y) where y is the approximate NN of x found from the just-inserted
cones.  When the leading eigenvector has rotated more than _DRIFT_THETA
degrees from all fan axes, _local_refresh(c) rebuilds the cluster's cones.
"""

import numpy as np
from ._kernels import jit_union_query, l2_distances

try:
    from ampi._ampi_ext import (
        SortedCone             as _SortedCone,
        best_clusters          as _cpp_best_clusters,
        best_fan_cones         as _cpp_best_fan_cones,
        update_drift_and_check as _cpp_update_drift_and_check,
    )
    _HAS_SORTED_CONE = True
    _HAS_EXT = True
except ImportError:
    _HAS_SORTED_CONE = False
    _HAS_EXT = False

# ── Phase-1 constants (from DATABASE_PLAN.md §Key Constants) ─────────────────

_DRIFT_BETA           = 0.01   # EMA decay for per-cluster drift covariance
_DRIFT_THETA          = 15.0   # degrees — fan-axis refresh trigger
_TOMBSTONE_THRESHOLD  = 0.10   # compact when tombstones > 10 % of cluster size
_NN_PROBE_W           = 8      # half-window for approx-NN lookup in drift EMA


# ── fallback cone (numba / no C++ ext) ───────────────────────────────────────

class _DictCone:
    """Numpy-backed cone mimicking the SortedCone interface.

    Used as a fallback when the C++ extension is not available.
    Does not support streaming mutations (no insert/remove).
    """
    __slots__ = ('_n', '_global_idx', '_sorted_idxs', '_sorted_projs')

    def __init__(self, global_idx, sorted_idxs, sorted_projs):
        self._n            = len(global_idx)
        self._global_idx   = global_idx
        self._sorted_idxs  = sorted_idxs
        self._sorted_projs = sorted_projs

    def size(self):
        return self._n

    def all_ids(self):
        return self._global_idx

    def query(self, q_proj, window_size):
        local = jit_union_query(
            self._sorted_idxs, self._sorted_projs, q_proj, window_size)
        return self._global_idx[local]

    def is_covered(self, q_proj, w, kth_proj):
        F = self._sorted_projs.shape[0]
        for l in range(F):
            sp  = self._sorted_projs[l]
            pos = int(np.searchsorted(sp, q_proj[l]))
            lo  = max(0, pos - w)
            hi  = min(self._n, pos + w)
            gap_right = float(sp[hi]     - q_proj[l]) if hi < self._n else np.inf
            gap_left  = float(q_proj[l]  - sp[lo - 1]) if lo > 0     else np.inf
            if min(gap_right, gap_left) >= kth_proj:
                return True
        return False


def _make_cone(global_idx, sorted_idxs, sorted_projs):
    """Construct a cone object (SortedCone if available, _DictCone otherwise)."""
    if _HAS_SORTED_CONE:
        return _SortedCone.from_arrays(sorted_projs, sorted_idxs, global_idx)
    return _DictCone(global_idx, sorted_idxs, sorted_projs)


# ── k-means internals ─────────────────────────────────────────────────────────

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


def _mini_batch_kmeans(data, k, seed=0, spherical=False):
    """Mini-batch k-means with BLAS-accelerated assignment.

    Strategy:
      1. Subsample s = min(50k, n) points for the working set.
      2. Random init: k distinct points from the subsample.
      3. Full Lloyd's iterations on the subsample in low-overhead form
         (BLAS gemm for assignment, bincount for centroid update).
      4. Final full-data assignment via BLAS.

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

    s = min(50_000, n)
    sample_idx = rng.choice(n, s, replace=False)
    sample = data[sample_idx].copy()
    sample_sq = np.sum(sample ** 2, axis=1)

    centroids = sample[rng.choice(s, k, replace=False)].copy()

    for it in range(20):
        cent_sq = np.sum(centroids ** 2, axis=1)
        dots = sample @ centroids.T
        d2 = sample_sq[:, None] + cent_sq[None, :] - 2 * dots
        assign = np.argmin(d2, axis=1).astype(np.int32)

        counts = np.bincount(assign, minlength=k)
        new_centroids = np.zeros((k, d), dtype=np.float64)
        for j in range(d):
            new_centroids[:, j] = np.bincount(
                assign, weights=sample[:, j].astype(np.float64), minlength=k
            )
        alive = counts > 0
        new_centroids[alive] /= counts[alive, None]
        dead = ~alive
        if dead.any():
            n_dead = dead.sum()
            new_centroids[dead] = sample[rng.choice(s, n_dead, replace=False)].astype(np.float64)
        new_c = new_centroids.astype(np.float32)
        if spherical:
            c_norms = np.linalg.norm(new_c, axis=1, keepdims=True)
            new_c /= np.where(c_norms < 1e-10, 1.0, c_norms)
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c

    assignments = _blas_assign(data, centroids)
    return centroids, assignments


def _build_cones_for_cluster(c_idx, c_data, centroid, axes, F, cone_top_k):
    """Build sorted cone arrays for one cluster.

    Returns
    -------
    cones       : list of F cone objects (or None for empty cones)
    point_cones : dict  global_id -> [f, ...]
                  caller sets the real cluster index and promotes to (c, f)
    """
    centered  = c_data - centroid
    all_projs = (centered @ axes.T).astype(np.float32)

    norms     = np.linalg.norm(centered, axis=1, keepdims=True).astype(np.float32)
    norms     = np.where(norms < 1e-10, 1.0, norms)
    normed    = np.abs(all_projs / norms)
    K         = min(cone_top_k, F)
    top_cones = np.argpartition(-normed, K - 1, axis=1)[:, :K]
    soft_mask = np.zeros((len(c_idx), F), dtype=bool)
    soft_mask[np.arange(len(c_idx))[:, None], top_cones] = True

    cones       = []
    point_cones = {}   # local index -> [f, ...]
    for f in range(F):
        f_local_idx = np.where(soft_mask[:, f])[0]
        n_f = len(f_local_idx)
        if n_f == 0:
            cones.append(None)
            continue

        f_global = c_idx[f_local_idx]
        f_projs  = all_projs[f_local_idx]

        s_idxs  = np.empty((F, n_f), dtype=np.int32)
        s_projs = np.empty((F, n_f), dtype=np.float32)
        for l in range(F):
            o = np.argsort(f_projs[:, l])
            s_idxs[l]  = o.astype(np.int32)
            s_projs[l] = f_projs[o, l]

        cones.append(_make_cone(f_global, s_idxs, s_projs))
        for gid in f_global:
            point_cones.setdefault(int(gid), []).append(f)

    return cones, point_cones


class AMPIAffineFanIndex:
    """K-means partition + affine fan cones + sorted-projection search.

    Parameters
    ----------
    data         : (n, d) float32
    nlist        : number of coarse clusters (default: sqrt(n))
    num_fans     : F — fan cones per cluster (also = sort directions)
    seed         : RNG seed
    cone_top_k   : K — soft assignment multiplicity (1 = hard argmax)
    metric       : 'l2' or 'cosine'
    drift_theta  : angle threshold in degrees at which a cluster's cones are
                   rebuilt (default: 15.0).  Lower values = more frequent
                   rebuilds; higher values = more drift tolerance.
    """

    def __init__(self, data, nlist=None, num_fans=16, seed=0, cone_top_k=1,
                 metric='l2', drift_theta=_DRIFT_THETA):
        if metric not in ('l2', 'cosine'):
            raise ValueError(f"metric must be 'l2' or 'cosine', got {metric!r}")
        self.metric = metric
        self.drift_theta = float(drift_theta)

        self.data = np.ascontiguousarray(data, dtype=np.float32)
        if metric == 'cosine':
            norms = np.linalg.norm(self.data, axis=1, keepdims=True)
            self.data = (self.data / np.where(norms < 1e-10, 1.0, norms)).astype(np.float32)
        self.n, self.d = self.data.shape
        self.F = num_fans
        self.cone_top_k = max(1, int(cone_top_k))

        if nlist is None:
            nlist = max(16, int(np.sqrt(self.n)))
        self.nlist = nlist

        # ── Global fan directions ─────────────────────────────────────────
        rng  = np.random.RandomState(seed)
        axes = rng.randn(self.F, self.d).astype(np.float32)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        self.axes = axes  # (F, d)

        # ── K-means partition ─────────────────────────────────────────────
        self.centroids, assignments = _mini_batch_kmeans(
            self.data, nlist, seed=seed,
            spherical=(metric == 'cosine'),
        )

        # ── Per-cluster: affine fan cones ─────────────────────────────────
        self.cluster_global = []
        self.cluster_cones  = []

        # _point_cones: global_id -> [(cluster_c, cone_f), ...]
        # Built here so delete() works on initial points without a scan.
        self._point_cones = {}

        for c in range(nlist):
            c_idx = np.where(assignments == c)[0].astype(np.int32)
            self.cluster_global.append(c_idx)

            if len(c_idx) < 2:
                self.cluster_cones.append(None)
                continue

            cones, pc = _build_cones_for_cluster(
                c_idx, self.data[c_idx], self.centroids[c],
                self.axes, self.F, self.cone_top_k,
            )
            self.cluster_cones.append(cones)
            for global_id, fs in pc.items():
                for f in fs:
                    self._point_cones.setdefault(global_id, []).append((c, f))

        # ── Phase-1 streaming state ───────────────────────────────────────
        # Pre-allocated capacity buffers for data and deleted-mask.
        # Extra headroom avoids a reallocation on the first few adds;
        # buffer doubles when full (amortised O(1) per insert).
        _HEADROOM = 1024
        _cap = self.n + _HEADROOM
        self._data_buf      = np.empty((_cap, self.d), dtype=np.float32)
        self._data_buf[:self.n] = self.data
        self.data           = self._data_buf[:self.n]       # view
        self._del_mask_buf  = np.zeros(_cap, dtype=bool)
        self._deleted_mask  = self._del_mask_buf[:self.n]   # view
        self._data_capacity = _cap
        self._n_deleted          = 0

        # Per-cluster live-point counts and tombstone counts.
        self._cluster_counts     = np.array([len(g) for g in self.cluster_global],
                                            dtype=np.int64)
        self._cluster_tombstones = np.zeros(self.nlist, dtype=np.int64)

        # Per-cluster drift covariance EMA — flat (nlist, d*d) float64.
        # Each row is a row-major flattened d×d covariance matrix.
        self._sigma_drift = np.zeros((self.nlist, self.d * self.d), dtype=np.float64)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _prepare_query(self, q):
        q = np.ascontiguousarray(q, dtype=np.float32)
        if self.metric == 'cosine':
            qnorm = float(np.linalg.norm(q))
            if qnorm > 1e-10:
                q = q / qnorm
        return q

    def _best_clusters(self, q, probes):
        if _HAS_EXT:
            return _cpp_best_clusters(self.centroids, q, probes)
        d2 = np.sum((self.centroids - q) ** 2, axis=1)
        return np.argsort(d2)[:probes]

    def _best_fan_cones(self, q_centered, fan_probes):
        if _HAS_EXT:
            return _cpp_best_fan_cones(self.axes, q_centered, fan_probes)
        q_norm = float(np.linalg.norm(q_centered))
        if q_norm < 1e-10:
            return np.arange(min(fan_probes, self.F), dtype=np.int32)
        proj = q_centered @ self.axes.T / q_norm
        return np.argsort(-np.abs(proj))[:fan_probes]

    def _check_drift(self, c):
        """Power iteration on Σ_drift[c]; trigger _local_refresh if the
        leading eigenvector is > _DRIFT_THETA degrees from all fan axes.
        Python fallback — only called when C++ ext is unavailable."""
        sig = self._sigma_drift[c].reshape(self.d, self.d)
        # Warm-start with the first axis (likely already well-aligned).
        v = sig @ self.axes[0].astype(np.float64)
        for _ in range(5):
            v = sig @ v
            norm = float(np.linalg.norm(v))
            if norm < 1e-12:
                return
            v /= norm
        # Maximum cosine similarity between v and any current fan axis.
        cos_max = float(np.max(np.abs(self.axes.astype(np.float64) @ v)))
        if cos_max < np.cos(np.radians(self.drift_theta)):
            self._local_refresh(c)

    def _local_refresh(self, c):
        """Rebuild all cones for cluster c in-place.

        Evicts tombstoned entries, rebalances cone assignments using the
        current (global) fan axes, and resets the drift covariance.
        O(N_c · F · log N_c).
        """
        c_global = self.cluster_global[c]
        if len(c_global) == 0:
            return

        # Live points only.
        if self._n_deleted:
            c_idx = c_global[~self._deleted_mask[c_global]].astype(np.int32)
        else:
            c_idx = c_global.astype(np.int32)

        if len(c_idx) < 2:
            self.cluster_cones[c] = None
            self.cluster_global[c] = c_idx
            self._cluster_tombstones[c] = 0
            self._sigma_drift[c][:] = 0
            return

        # Remove all (c, *) entries from _point_cones before rebuilding.
        for gid in c_idx:
            gid_i = int(gid)
            if gid_i in self._point_cones:
                self._point_cones[gid_i] = [
                    (cc, ff) for cc, ff in self._point_cones[gid_i] if cc != c
                ]

        cones, pc = _build_cones_for_cluster(
            c_idx, self.data[c_idx], self.centroids[c],
            self.axes, self.F, self.cone_top_k,
        )
        self.cluster_cones[c]    = cones
        self.cluster_global[c]   = c_idx
        self._cluster_counts[c]  = len(c_idx)   # keep denominator in sync
        self._cluster_tombstones[c] = 0
        self._sigma_drift[c][:] = 0

        for global_id, fs in pc.items():
            for f in fs:
                self._point_cones.setdefault(global_id, []).append((c, f))

    # ── streaming insert / delete / update ────────────────────────────────────

    def add(self, x):
        """Insert one vector into the index.

        Parameters
        ----------
        x : (d,) array_like, float32

        Returns
        -------
        global_id : int — index into self.data for the inserted point.
        """
        x = np.ascontiguousarray(x, dtype=np.float32).ravel()
        if x.shape[0] != self.d:
            raise ValueError(f"expected d={self.d}, got {x.shape[0]}")
        if self.metric == 'cosine':
            xnorm = float(np.linalg.norm(x))
            if xnorm > 1e-10:
                x = x / xnorm

        global_id = self.n

        # Grow buffers if at capacity (amortised O(1) via doubling).
        if self.n >= self._data_capacity:
            new_cap = self._data_capacity * 2
            new_data_buf = np.empty((new_cap, self.d), dtype=np.float32)
            new_data_buf[:self.n] = self._data_buf[:self.n]
            self._data_buf = new_data_buf
            new_mask_buf = np.zeros(new_cap, dtype=bool)
            new_mask_buf[:self.n] = self._del_mask_buf[:self.n]
            self._del_mask_buf = new_mask_buf
            self._data_capacity = new_cap

        self._data_buf[self.n]     = x
        self._del_mask_buf[self.n] = False
        self.n            += 1
        self.data          = self._data_buf[:self.n]
        self._deleted_mask = self._del_mask_buf[:self.n]

        # ── Nearest-centroid cluster assignment (top-K by L2) ─────────────
        d2           = np.sum((self.centroids - x) ** 2, axis=1)
        top_clusters = np.argsort(d2)[:self.cone_top_k].tolist()

        self._point_cones[global_id] = []

        for c in top_clusters:
            centroid = self.centroids[c]
            centered = x - centroid
            proj     = (centered @ self.axes.T).astype(np.float32)   # (F,)

            # Top cone_top_k cones by |normalised projection|.
            cn = float(np.linalg.norm(centered))
            normed_proj = np.abs(proj) / cn if cn > 1e-10 else np.abs(proj)
            K = min(self.cone_top_k, self.F)
            top_f = np.argpartition(-normed_proj, K - 1)[:K].tolist()

            cones = self.cluster_cones[c]
            if cones is not None:
                for f in top_f:
                    cone = cones[f]
                    if cone is not None:
                        cone.insert(proj, global_id)
                        self._point_cones[global_id].append((c, f))

            # Append to cluster_global.
            self.cluster_global[c] = np.append(
                self.cluster_global[c], np.int32(global_id))

            # §1.2  Centroid EMA.
            N = self._cluster_counts[c]
            self.centroids[c] = ((N * self.centroids[c] + x) / (N + 1)).astype(np.float32)
            self._cluster_counts[c] = N + 1

            # §1.3  Drift covariance EMA: Σ ← (1-β)Σ + β·(x-y)(x-y)ᵀ
            # y = approx NN of x within this cluster, found from the cones
            # we just inserted into.  Falls back to centroid when the cone
            # is too small to provide a meaningful neighbour.
            if cn > 1e-10:
                approx_nn = None
                if cones is not None:
                    nn_parts = []
                    for f in top_f:
                        cone = cones[f]
                        if cone is not None and cone.size() > 1:
                            nn_parts.append(cone.query(proj, _NN_PROBE_W))
                    if nn_parts:
                        nn_cands = np.unique(np.concatenate(nn_parts))
                        nn_cands = nn_cands[nn_cands != global_id]
                        if len(nn_cands):
                            nn_dists = l2_distances(self.data, x, nn_cands)
                            approx_nn = self.data[nn_cands[int(np.argmin(nn_dists))]]

                displacement = (x - approx_nn) if approx_nn is not None else centered
                v = displacement.astype(np.float64)
                if _HAS_EXT:
                    if _cpp_update_drift_and_check(
                            self._sigma_drift[c], self.axes, v,
                            _DRIFT_BETA, self.drift_theta):
                        self._local_refresh(c)
                else:
                    self._sigma_drift[c] = (
                        (1.0 - _DRIFT_BETA) * self._sigma_drift[c]
                        + _DRIFT_BETA * np.outer(v, v).ravel()
                    )
                    self._check_drift(c)

        return global_id

    def delete(self, global_id):
        """Logical delete of point global_id.

        Tombstones it in every cone it belongs to.  Triggers a full
        _local_refresh for any cluster whose tombstone fraction exceeds
        _TOMBSTONE_THRESHOLD.

        Parameters
        ----------
        global_id : int
        """
        global_id = int(global_id)
        if global_id < 0 or global_id >= self.n:
            raise IndexError(f"global_id {global_id} out of range [0, {self.n})")
        if self._deleted_mask[global_id]:
            return   # already deleted

        self._deleted_mask[global_id] = True
        self._n_deleted += 1

        # Tombstone in every cone and update per-cluster tombstone counters.
        seen_clusters = set()
        for c, f in self._point_cones.get(global_id, []):
            cones = self.cluster_cones[c]
            if cones is not None and cones[f] is not None:
                cones[f].remove(global_id)
            if c not in seen_clusters:
                seen_clusters.add(c)
                self._cluster_tombstones[c] += 1

        # §1.4  Physical compaction when tombstone fraction exceeds threshold.
        for c in seen_clusters:
            if self._cluster_counts[c] > 0:
                frac = self._cluster_tombstones[c] / self._cluster_counts[c]
                if frac >= _TOMBSTONE_THRESHOLD:
                    self._local_refresh(c)

    def update(self, global_id, x):
        """Replace point global_id with a new vector x.

        Parameters
        ----------
        global_id : int — id of the point to replace
        x         : (d,) array_like, float32

        Returns
        -------
        new_global_id : int — id assigned to the replacement point.
        """
        self.delete(global_id)
        return self.add(x)

    # ── public query API ──────────────────────────────────────────────────────

    def query_candidates(self, q, window_size=50, probes=10, fan_probes=2):
        """Return the union of cone candidates before exact-distance re-ranking.

        Parameters
        ----------
        q           : (d,) array_like, float32
        window_size : half-window of sorted-projection entries per cone per axis
        probes      : number of nearest clusters to probe
        fan_probes  : number of best-aligned cones to probe per cluster

        Returns
        -------
        candidates : (m,) int32 — unique live data indices
        """
        q = self._prepare_query(q)
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

            # Fast path: probe all cones → return whole cluster.
            if fan_probes >= self.F:
                parts.append(gi)
                continue

            centroid   = self.centroids[c]
            q_centered = q - centroid
            q_proj     = np.ascontiguousarray(
                (q_centered @ self.axes.T).astype(np.float32))

            best_cones = self._best_fan_cones(q_centered, fan_probes)
            for f in best_cones:
                f = int(f)
                if f >= len(cones) or cones[f] is None:
                    continue
                cone = cones[f]
                if cone.size() <= 2 * window_size:
                    parts.append(cone.all_ids())
                else:
                    parts.append(cone.query(q_proj, window_size))

        if not parts:
            return np.zeros(0, dtype=np.int32)
        cands = np.unique(np.concatenate(parts))
        if self._n_deleted:
            cands = cands[~self._deleted_mask[cands]]
        return cands

    def query(self, q, k=10, window_size=200, probes=10, fan_probes=2):
        """Adaptive sorted-projection search.

        Starts with a small window and expands only until the projection lower
        bound guarantees no unvisited point can improve the current top-k.
        window_size is a ceiling; most queries stop well before reaching it.

        Stopping condition per cone: there exists an axis l whose window
        boundary projection gap >= sqrt(kth_sq). Since L2(x,q) >= |proj_l(x-q)|
        for any unit vector l, those unvisited points cannot enter the top-k.
        """
        q = self._prepare_query(q)

        cone_ctxs      = []
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

        w     = max(k, 8)
        cands = np.zeros(0, dtype=np.int32)

        while True:
            parts = list(fallback_parts)
            for cone, q_proj in cone_ctxs:
                if cone.size() <= 2 * w:
                    parts.append(cone.all_ids())
                else:
                    parts.append(cone.query(q_proj, w))

            cands = (np.unique(np.concatenate(parts))
                     if parts else np.zeros(0, dtype=np.int32))

            # Filter tombstoned entries that leaked through the fast path.
            if self._n_deleted and len(cands):
                cands = cands[~self._deleted_mask[cands]]

            if w >= window_size or len(cands) < k:
                break

            dists    = l2_distances(self.data, q, cands)
            kth_sq   = float(np.partition(dists, k - 1)[k - 1])
            kth_proj = np.sqrt(max(0.0, kth_sq))

            all_covered = True
            for cone, q_proj in cone_ctxs:
                if cone.size() <= 2 * w:
                    continue
                if not cone.is_covered(q_proj, w, kth_proj):
                    all_covered = False
                    break

            if all_covered:
                break

            w = min(w * 2, window_size)

        if len(cands) < k:
            cands = np.arange(min(k, self.n), dtype=np.int32)
            if self._n_deleted:
                cands = cands[~self._deleted_mask[cands]]
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]
