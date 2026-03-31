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

Drift detection: per cluster an Oja subspace sketch U_drift tracks the
displacement (x − y) where y is the approximate NN of x found from the
just-inserted cones.  When the leading eigenvector estimate has rotated
more than _DRIFT_THETA degrees from all fan axes, _local_refresh(c)
rebuilds the cluster's cones.
"""

import numpy as np
from ._kernels import jit_union_query, l2_distances

try:
    from ampi._ampi_ext import (
        SortedCone             as _SortedCone,
        best_clusters          as _cpp_best_clusters,
        best_fan_cones         as _cpp_best_fan_cones,
        update_drift_and_check as _cpp_update_drift_and_check,
        AMPIIndex              as _AMPIIndex,
    )
    _HAS_SORTED_CONE = True
    _HAS_EXT = True
except ImportError:
    _HAS_SORTED_CONE = False
    _HAS_EXT = False

# ── Metric selector ───────────────────────────────────────────────────────────

_METRIC_MAP: dict = {
    'l2':          'l2',
    'L2':          'l2',
    'euclidean':   'l2',
    'sqeuclidean': 'sqeuclidean',
    'cosine':      'cosine',
}

def _normalize_metric(metric: str) -> str:
    """Map a metric alias to its canonical name.

    Accepted aliases
    ----------------
    l2 / L2 / euclidean  →  'l2'          (Euclidean; query returns sqrt distances)
    sqeuclidean          →  'sqeuclidean'  (squared L2; faster, not a true metric)
    cosine               →  'cosine'       (index normalises vectors internally; returns 1 − cos_sim)
    """
    canon = _METRIC_MAP.get(metric)
    if canon is None:
        valid = sorted(set(_METRIC_MAP.keys()))
        raise ValueError(
            f"unknown metric {metric!r}. Valid aliases: {valid}"
        )
    return canon


# ── Phase-1 constants (from DATABASE_PLAN.md §Key Constants) ─────────────────

_DRIFT_BETA           = 0.01   # EMA decay for per-cluster drift covariance
_DRIFT_THETA          = 15.0   # degrees — fan-axis refresh trigger
_TOMBSTONE_THRESHOLD  = 0.10   # compact when tombstones > 10 % of cluster size
_NN_PROBE_W           = 8      # half-window for approx-NN lookup in drift EMA
_U_REORTH_INTERVAL = 50   # QR re-orthonormalise every N inserts per cluster

# ── Periodic cluster-merge constants ─────────────────────────────────────────
_MERGE_INTERVAL       = 0      # 0 = disabled; set > 0 to enable periodic merge
_EPS_MERGE            = 1.0    # centroid L2 distance threshold for merge check

_MERGE_QE_RATIO       = 0.5    # merge if δ_qe ≤ ratio × (mQE_i + mQE_j)


# ── C++ cone proxy ────────────────────────────────────────────────────────────
#
# When _HAS_EXT is True the cones live inside AMPIIndex (C++).
# These two thin wrappers expose them through the same list-of-lists interface
# that the Python path uses, so _py_query and all Python code keep working
# without any None-stub hacks.

class _CppClusterCones:
    """List-like view of the F SortedCone objects for one cluster."""
    __slots__ = ('_cpp', '_c')

    def __init__(self, cpp, c):
        self._cpp = cpp
        self._c   = c

    def __len__(self):
        return self._cpp.F

    def __getitem__(self, f):
        return self._cpp.get_cone(self._c, f)

    def __iter__(self):
        return (self._cpp.get_cone(self._c, f) for f in range(self._cpp.F))


class _CppConesProxy:
    """List-like view of per-cluster cones backed by AMPIIndex.get_cone().

    cluster_cones[c] returns None when the cluster has no cones yet
    (fewer than 2 points), otherwise a _CppClusterCones accessor.
    """
    __slots__ = ('_cpp',)

    def __init__(self, cpp):
        self._cpp = cpp

    def __len__(self):
        return self._cpp.nlist

    def __getitem__(self, c):
        if not self._cpp.has_cones(c):
            return None
        return _CppClusterCones(self._cpp, c)

    def __setitem__(self, c, value):
        # Only called by the Python-path _local_refresh; ignored when C++
        # owns the cones (C++ refresh updates cones in-place via local_refresh).
        pass


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
    metric       : distance metric — 'l2'/'L2'/'euclidean', 'sqeuclidean', or 'cosine'
                   (cosine normalises all vectors internally)
    drift_theta  : angle threshold in degrees at which a cluster's cones are
                   rebuilt (default: 15.0).  Lower values = more frequent
                   rebuilds; higher values = more drift tolerance.
    """

    def __init__(self, data, nlist=None, num_fans=16, seed=0, cone_top_k=1,
                 metric='l2', drift_theta=_DRIFT_THETA,
                 merge_interval=_MERGE_INTERVAL, eps_merge=_EPS_MERGE,
                 merge_qe_ratio=_MERGE_QE_RATIO, data_path=None):
        self.metric           = _normalize_metric(metric)
        self.drift_theta      = float(drift_theta)
        self.merge_interval   = int(merge_interval)
        self.eps_merge        = float(eps_merge)
        self.merge_qe_ratio   = float(merge_qe_ratio)
        self._data_path       = data_path

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
        self._point_cones   = {}

        if _HAS_EXT:
            # C++ builds cones internally in from_build().
            # Only populate cluster_global here; cluster_cones will be set to
            # a _CppConesProxy after the C++ index is constructed below.
            for c in range(nlist):
                c_idx = np.where(assignments == c)[0].astype(np.int32)
                self.cluster_global.append(c_idx)
        else:
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
        #
        # When data_path is provided the data buffer is backed by a memory-
        # mapped file (Phase-2 prerequisite).  The OS pages in only the
        # clusters being accessed, so effective RSS stays proportional to the
        # working set rather than the full dataset size.
        _HEADROOM = 1024
        _cap = self.n + _HEADROOM
        # Python-path (no C++ ext): use memmap so the OS can page out idle clusters.
        # When the C++ ext is present it owns its own mmap (passed via data_path to
        # from_build below), so we just use np.empty here as a temporary staging buffer.
        if data_path is not None and not _HAS_EXT:
            import os
            self._data_buf_file = os.path.join(data_path, '_data_buf.dat')
            self._data_buf = np.memmap(self._data_buf_file, mode='w+',
                                       dtype='float32', shape=(_cap, self.d))
        else:
            self._data_buf_file = None
            self._data_buf = np.empty((_cap, self.d), dtype=np.float32)
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

        # Per-cluster Oja subspace sketch: top-F eigenvectors of the displacement
        # covariance EMA.  Shape (nlist, d, F) float32 — ~60× smaller than the
        # full d×d covariance at d=960 (120 MB vs 7.4 GB at nlist=1000).
        self._U_drift = np.zeros((self.nlist, self.d, self.F), dtype=np.float32)
        # Per-cluster insert counter for periodic QR re-orthonormalisation.
        self._U_reorth_count = np.zeros(self.nlist, dtype=np.int32)

        # Per-cluster fan axes — list of nlist (F, d) float32 arrays, or None
        # meaning "use global self.axes".  Populated by _local_refresh via
        # _compute_cluster_axes (from U_drift Oja sketch columns).
        self.cluster_axes = [None] * self.nlist

        # Insert counter for periodic merge; only used by the Python path.
        # C++ tracks its own counter internally.
        self._insert_count = 0

        # ── C++ index (Phase 3+) ──────────────────────────────────────────────
        if _HAS_EXT:
            self._cpp = _AMPIIndex.from_build(
                self.d, self.F, self.nlist, self.cone_top_k,
                self.drift_theta, self.metric == 'cosine',
                self.axes, self.centroids,
                self._cluster_counts, self._U_drift,
                self._data_buf, self._del_mask_buf.astype(np.uint8),
                self.n,
                self.cluster_global,
                data_path=self._data_path or "",
            )
            self._cpp.set_merge_params(self.merge_interval, self.eps_merge,
                                       self.merge_qe_ratio)
            self._refresh_views()
            self.cluster_cones = _CppConesProxy(self._cpp)
        else:
            self._cpp = None

    # ── streaming factory ─────────────────────────────────────────────────────

    @classmethod
    def from_stream(cls, *, n, d, F, nlist, cone_top_k, metric, drift_theta,
                    merge_interval, eps_merge, merge_qe_ratio,
                    axes, centroids, cluster_global, cluster_counts, cones,
                    data_path):
        """Assemble an index from pre-built streaming components.

        Called by streaming.streaming_build(); do not call directly.
        Requires the C++ extension and a valid data_path/_cpp_data_buf.dat.
        Bypasses k-means and _build_cones — no random mmap access.
        """
        if not _HAS_EXT:
            raise RuntimeError(
                "from_stream requires the compiled C++ extension. "
                "Run `pip install -e .` to build it."
            )

        self = object.__new__(cls)
        self.metric          = metric
        self.drift_theta     = float(drift_theta)
        self.merge_interval  = int(merge_interval)
        self.eps_merge       = float(eps_merge)
        self.merge_qe_ratio  = float(merge_qe_ratio)
        self._data_path      = data_path
        self.n               = n
        self.d               = d
        self.F               = F
        self.cone_top_k      = cone_top_k
        self.nlist           = nlist
        self.axes            = np.ascontiguousarray(axes,      dtype=np.float32)
        self.centroids       = np.ascontiguousarray(centroids, dtype=np.float32)
        self.cluster_global  = cluster_global
        self._point_cones    = {}

        # Zero-sized staging buffers — C++ mmap owns the data after from_stream.
        self._data_buf_file  = None
        self._data_buf       = np.empty((0, d), dtype=np.float32)
        self._del_mask_buf   = np.empty(0, dtype=bool)
        self._deleted_mask   = np.empty(0, dtype=bool)
        self._data_capacity  = 0
        self._n_deleted      = 0
        self._cluster_counts     = cluster_counts.copy()
        self._cluster_tombstones = np.zeros(nlist, dtype=np.int64)
        self._U_drift        = np.zeros((nlist, d, F), dtype=np.float32)
        self._U_reorth_count = np.zeros(nlist, dtype=np.int32)
        self.cluster_axes    = [None] * nlist
        self._insert_count   = 0

        self._cpp = _AMPIIndex.from_stream(
            d, F, nlist, cone_top_k,
            drift_theta, metric == 'cosine',
            self.axes, self.centroids,
            cluster_counts, n,
            data_path or "",
            cluster_global, cones,
        )
        self._cpp.set_merge_params(merge_interval, eps_merge, merge_qe_ratio)
        self._refresh_views()
        self.cluster_cones = _CppConesProxy(self._cpp)

        return self

    # ── internal helpers ──────────────────────────────────────────────────────

    def _refresh_views(self):
        """Refresh Python references to C++ memory after any mutation."""
        self.data          = self._cpp.get_data_view()
        self._deleted_mask = self._cpp.get_deleted_mask().astype(bool)
        self.n             = self._cpp.n
        self._n_deleted    = self._cpp.n_deleted
        self.centroids     = self._cpp.get_centroids()

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

    def _compute_cluster_axes(self, c):
        """Return F fan axes for cluster c from the Oja sketch U_drift[c].

        Columns of U_drift[c] are the accumulated eigenvector estimates.
        Falls back to global axes when the sketch has insufficient signal.
        """
        U = self._U_drift[c]          # (d, F)
        norms = np.linalg.norm(U, axis=0)   # (F,)
        if norms[0] < 1e-6:
            return self.axes          # no signal yet — use global axes
        U_norm = U / np.maximum(norms, 1e-12)
        return U_norm.T.astype(np.float32)  # (F, d)

    def _check_drift(self, c):
        """Check if U_drift[:,0] has rotated > drift_theta from all fan axes.
        Python fallback — only called when C++ ext is unavailable."""
        u0 = self._U_drift[c, :, 0]
        norm = float(np.linalg.norm(u0))
        if norm < 1e-6:
            return          # no signal accumulated yet
        u0 = u0 / norm
        axes_c = self.cluster_axes[c] if self.cluster_axes[c] is not None else self.axes
        cos_max = float(np.max(np.abs(axes_c.astype(np.float64) @ u0)))
        if cos_max < np.cos(np.radians(self.drift_theta)):
            self._local_refresh(c)

    def _local_refresh(self, c):
        """Rebuild all cones for cluster c in-place.

        Evicts tombstoned entries, rebalances cone assignments using
        per-cluster axes derived from U_drift (or global axes as fallback),
        and resets the Oja sketch.  O(N_c · F · log N_c).
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
            self._U_drift[c][:] = 0
            self._U_reorth_count[c] = 0
            self.cluster_axes[c] = None
            return

        # Remove all (c, *) entries from _point_cones before rebuilding.
        for gid in c_idx:
            gid_i = int(gid)
            if gid_i in self._point_cones:
                self._point_cones[gid_i] = [
                    (cc, ff) for cc, ff in self._point_cones[gid_i] if cc != c
                ]

        # Derive per-cluster axes from accumulated U_drift before resetting it.
        axes_c = self._compute_cluster_axes(c)
        self.cluster_axes[c] = axes_c

        cones, pc = _build_cones_for_cluster(
            c_idx, self.data[c_idx], self.centroids[c],
            axes_c, self.F, self.cone_top_k,
        )
        self.cluster_cones[c]       = cones
        self.cluster_global[c]      = c_idx
        self._cluster_counts[c]     = len(c_idx)   # keep denominator in sync
        self._cluster_tombstones[c] = 0
        self._U_drift[c][:]         = 0            # reset AFTER computing axes
        self._U_reorth_count[c]     = 0

        for global_id, fs in pc.items():
            for f in fs:
                self._point_cones.setdefault(global_id, []).append((c, f))

    # ── cluster merge ─────────────────────────────────────────────────────────

    def _py_merge_clusters(self, keep, fold):
        """Merge cluster `fold` into cluster `keep` (Python-path).

        Updates centroids, cluster_global, _point_cones, and rebuilds
        cones for the merged cluster via _local_refresh.
        """
        N_k = int(self._cluster_counts[keep])
        N_f = int(self._cluster_counts[fold])
        if N_f == 0:
            return

        # Merged centroid (weighted average).
        N_total  = N_k + N_f
        mu_k     = self.centroids[keep]
        mu_f     = self.centroids[fold]
        mu_merged = ((N_k * mu_k + N_f * mu_f) / N_total).astype(np.float32)

        # Live points from fold.
        cg_fold = self.cluster_global[fold]
        if self._n_deleted and len(cg_fold):
            fold_live = cg_fold[~self._deleted_mask[cg_fold]].astype(np.int32)
        else:
            fold_live = cg_fold.astype(np.int32)

        # Remove all (fold, *) entries from _point_cones for fold's live points.
        for gid in fold_live:
            gid_i = int(gid)
            if gid_i in self._point_cones:
                self._point_cones[gid_i] = [
                    (cc, ff) for cc, ff in self._point_cones[gid_i] if cc != fold
                ]

        # Append fold's live points into keep's global list.
        self.cluster_global[keep] = np.concatenate(
            [self.cluster_global[keep], fold_live]).astype(np.int32)

        # Update centroids: keep = merged; fold = merged (redirect new inserts).
        self.centroids[keep] = mu_merged
        self.centroids[fold] = mu_merged

        # Update counts / tombstones.
        self._cluster_counts[keep]     = N_total
        self._cluster_counts[fold]     = 0
        self._cluster_tombstones[fold] = 0

        # Clear fold cluster.
        self.cluster_global[fold]  = np.array([], dtype=np.int32)
        self.cluster_cones[fold]   = None
        self.cluster_axes[fold]    = None
        self._U_drift[fold][:] = 0
        self._U_reorth_count[fold] = 0

        # Rebuild cones for the merged cluster (also resets tombstone count,
        # U_drift, and computes per-cluster axes).
        self._local_refresh(keep)

    def _py_periodic_merge(self):
        """Scan all cluster pairs; merge close pairs that reduce mean QE."""
        eps2   = self.eps_merge ** 2
        merged = [False] * self.nlist

        for i in range(self.nlist):
            if merged[i] or self._cluster_counts[i] == 0:
                continue

            best_j  = -1
            best_d2 = eps2   # only keep pairs strictly below eps2

            for j in range(i + 1, self.nlist):
                if merged[j] or self._cluster_counts[j] == 0:
                    continue
                d2 = float(np.sum((self.centroids[i] - self.centroids[j]) ** 2))
                if d2 < best_d2:
                    best_d2 = d2
                    best_j  = j

            if best_j < 0:
                continue
            j = best_j

            N_i     = int(self._cluster_counts[i])
            N_j     = int(self._cluster_counts[j])
            N_total = N_i + N_j

            # Mean quantisation error for each cluster.
            cg_i = self.cluster_global[i]
            cg_j = self.cluster_global[j]
            if self._n_deleted and len(cg_i):
                live_i = cg_i[~self._deleted_mask[cg_i]]
            else:
                live_i = cg_i
            if self._n_deleted and len(cg_j):
                live_j = cg_j[~self._deleted_mask[cg_j]]
            else:
                live_j = cg_j

            mQE_i = (float(np.mean(
                np.sum((self.data[live_i] - self.centroids[i]) ** 2, axis=1)))
                if len(live_i) else 0.0)
            mQE_j = (float(np.mean(
                np.sum((self.data[live_j] - self.centroids[j]) ** 2, axis=1)))
                if len(live_j) else 0.0)

            # The merge increases mean QE by exactly:
            #   δ = N_i*N_j / N_total² * ||μ_i - μ_j||²
            # Merge if that increase is small relative to cluster spread.
            weight   = (N_i * N_j) / (N_total ** 2)
            delta_qe = weight * best_d2
            if delta_qe <= (mQE_i + mQE_j) * 0.5:
                self._py_merge_clusters(i, j)
                merged[j] = True

    def periodic_merge(self, eps_merge=None):
        """Trigger a cluster merge pass immediately.

        Checks all centroid pairs within eps_merge; merges those that
        reduce mean quantisation error (no full model comparison).

        Parameters
        ----------
        eps_merge : float or None — centroid L2 distance threshold.
                    Defaults to self.eps_merge.
        """
        if eps_merge is None:
            eps_merge = self.eps_merge
        if self._cpp is not None:
            self._cpp.periodic_merge(float(eps_merge))
            self._refresh_views()
            return
        old_eps        = self.eps_merge
        self.eps_merge = float(eps_merge)
        self._py_periodic_merge()
        self.eps_merge = old_eps

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
        if self._cpp is not None:
            gid = self._cpp.add(x)
            self._refresh_views()
            return gid
        return self._py_add(x)

    def _py_add(self, x):
        """Python-path add (used when C++ ext unavailable)."""
        if self.metric == 'cosine':
            xnorm = float(np.linalg.norm(x))
            if xnorm > 1e-10:
                x = x / xnorm

        global_id = self.n

        # Grow buffers if at capacity (amortised O(1) via doubling).
        if self.n >= self._data_capacity:
            new_cap = self._data_capacity * 2
            if self._data_buf_file is not None:
                # memmap: copy existing data before recreating the file at
                # the larger size (mode='w+' truncates, so copy first).
                tmp = self._data_buf[:self.n].copy()
                new_data_buf = np.memmap(self._data_buf_file, mode='w+',
                                         dtype='float32', shape=(new_cap, self.d))
                new_data_buf[:self.n] = tmp
            else:
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
            # Use per-cluster fan axes if they have been computed, else global.
            axes_c   = self.cluster_axes[c] if self.cluster_axes[c] is not None else self.axes
            proj     = (centered @ axes_c.T).astype(np.float32)   # (F,)

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

            # §1.3  Oja subspace sketch: U ← (1-β)U + β·v·(vᵀU)ᵀ, then normalise columns
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
                v = displacement.astype(np.float32)
                # Oja's subspace rule: update U_drift then check angle.
                U = self._U_drift[c]                 # (d, F) view
                proj = v @ U                         # (F,)
                U *= (1.0 - _DRIFT_BETA)
                U += _DRIFT_BETA * np.outer(v, proj)
                # Column normalisation (cheap; full QR every _U_REORTH_INTERVAL steps).
                self._U_reorth_count[c] += 1
                if self._U_reorth_count[c] >= _U_REORTH_INTERVAL:
                    norms = np.linalg.norm(U, axis=0)
                    mask = norms > 1e-12
                    if mask.any():
                        U[:, mask] /= norms[mask]
                    self._U_reorth_count[c] = 0
                else:
                    norms = np.linalg.norm(U, axis=0)
                    mask = norms > 1e-12
                    if mask.any():
                        U[:, mask] /= norms[mask]
                self._check_drift(c)

        # §Merge: after every merge_interval inserts, scan cluster pairs.
        if self.merge_interval > 0:
            self._insert_count += 1
            if self._insert_count % self.merge_interval == 0:
                self._py_periodic_merge()

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
        if self._cpp is not None:
            self._cpp.remove(global_id)
            self._refresh_views()
            return
        self._py_delete(global_id)

    def batch_add(self, data):
        """Insert multiple vectors at once (single exclusive lock).

        Parameters
        ----------
        data : (m, d) array_like, float32

        Returns
        -------
        global_ids : (m,) int32 array
        """
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.d:
            raise ValueError(f"expected (m, {self.d}), got {data.shape}")
        if self._cpp is not None:
            ids = self._cpp.batch_add(data)
            self._refresh_views()
            return ids
        # Python fallback: loop
        return np.array([self._py_add(data[i]) for i in range(len(data))], dtype=np.int32)

    def batch_delete(self, global_ids):
        """Tombstone multiple points at once (single exclusive lock).

        Parameters
        ----------
        global_ids : (m,) int array
        """
        global_ids = np.asarray(global_ids, dtype=np.int32).ravel()
        if self._cpp is not None:
            self._cpp.batch_delete(global_ids)
            self._refresh_views()
            return
        for gid in global_ids.tolist():
            self._py_delete(int(gid))

    def _py_delete(self, global_id):
        """Python-path delete (used when C++ ext unavailable)."""
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
        if self._cpp is not None:
            return self._cpp.query_candidates(q, window_size, probes, fan_probes)
        return self._py_query_candidates(q, window_size, probes, fan_probes)

    def _py_query_candidates(self, q, window_size=50, probes=10, fan_probes=2):
        """Pure Python query_candidates (fallback when C++ ext unavailable)."""
        clusters = self._best_clusters(q, probes)

        parts = []
        for c in clusters:
            c = int(c)
            gi        = self.cluster_global[c]
            has_cones = self.cluster_cones[c] is not None
            if not has_cones or len(gi) == 0:
                if len(gi) > 0:
                    parts.append(gi)
                continue

            if fan_probes >= self.F:
                parts.append(gi)
                continue

            centroid   = self.centroids[c]
            q_centered = q - centroid
            axes_c     = self.cluster_axes[c] if self.cluster_axes[c] is not None else self.axes
            q_proj     = np.ascontiguousarray(
                (q_centered @ axes_c.T).astype(np.float32))

            q_norm = float(np.linalg.norm(q_centered))
            if q_norm < 1e-10:
                best_cones = np.arange(min(fan_probes, self.F), dtype=np.int32)
            else:
                best_cones = np.argsort(-np.abs(q_centered @ axes_c.T) / q_norm)[:fan_probes]

            for f in best_cones:
                f = int(f)
                if f >= len(self.cluster_cones[c]) or self.cluster_cones[c][f] is None:
                    continue
                cone = self.cluster_cones[c][f]
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
        if self._cpp is not None:
            sq_dists, ids = self._cpp.query(q, k, window_size, probes, fan_probes)
            if self.metric == 'l2':
                dists = np.sqrt(np.maximum(0.0, sq_dists))
            elif self.metric == 'sqeuclidean':
                dists = sq_dists
            else:  # cosine: vectors are already normalised; sq_l2 = 2*(1 - cos_sim)
                dists = sq_dists * 0.5
            # Use a fresh data view so results are correct even if add() was
            # called after the last _refresh_views() (e.g. via batch_add).
            data_view = self._cpp.get_data_view()
            return data_view[ids], dists, ids
        return self._py_query(q, k, window_size, probes, fan_probes)

    def _py_query(self, q, k=10, window_size=200, probes=10, fan_probes=2):
        """Pure Python query (fallback when C++ ext unavailable)."""
        cone_ctxs      = []
        fallback_parts = []

        for c in map(int, self._best_clusters(q, probes)):
            gi        = self.cluster_global[c]
            has_cones = self.cluster_cones[c] is not None
            if not len(gi):
                continue
            if not has_cones or fan_probes >= self.F:
                fallback_parts.append(gi)
                continue
            centroid   = self.centroids[c]
            q_centered = q - centroid
            axes_c     = self.cluster_axes[c] if self.cluster_axes[c] is not None else self.axes
            q_proj     = np.ascontiguousarray(
                (q_centered @ axes_c.T).astype(np.float32)
            )
            q_norm = float(np.linalg.norm(q_centered))
            if q_norm < 1e-10:
                top_fs = range(min(fan_probes, self.F))
            else:
                top_fs = np.argsort(-np.abs(q_centered @ axes_c.T) / q_norm)[:fan_probes]
            for f in map(int, top_fs):
                if f < len(self.cluster_cones[c]) and self.cluster_cones[c][f] is not None:
                    cone_ctxs.append((self.cluster_cones[c][f], q_proj))

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
        sq_dists = l2_distances(self.data, q, cands)
        top      = np.argsort(sq_dists)[:k]
        if self.metric == 'l2':
            out_dists = np.sqrt(np.maximum(0.0, sq_dists[top]))
        elif self.metric == 'sqeuclidean':
            out_dists = sq_dists[top]
        else:  # cosine: vectors are already normalised; sq_l2 = 2*(1 - cos_sim)
            out_dists = sq_dists[top] * 0.5
        return self.data[cands[top]], out_dists, cands[top]
