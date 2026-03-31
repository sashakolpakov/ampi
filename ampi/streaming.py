"""streaming.py — StreamingBuildDispatcher and streaming_build().

Builds AMPIAffineFanIndex from large datasets without loading all data into RAM.
Sequential access throughout — mmap pages are evicted by the OS as the streaming
window advances.

Memory budget (GIST 1M, d=960, F=16, K=1):
  _all_projs  : 1 000 000 × 16 × 4 B =  64 MB
  _assignments: 1 000 000 × 4 B      =   4 MB
  _all_norms  : 1 000 000 × 4 B      =   4 MB
  centroids   : 1 000 × 960 × 4 B    =   3.7 MB
  axes        : 16 × 960 × 4 B       =  61 kB
  chunk buffer: 4 096 × 960 × 4 B    =  15.7 MB
  data mmap   : 1 000 000 × 960 × 4 B = 3.58 GB  (virtual; RSS ≈ active chunk)
  ─────────────────────────────────────────────────────────────────────────────
  Peak RSS (excluding mmap backing): ≈ 90 MB.
"""

import numpy as np


class _StreamingDispatcher:
    """One-pass sequential accumulator: assign → project → store in pre-allocated arrays.

    No random mmap access during ingest: all reads are contiguous chunk slices.
    The only random access is in build_cones(), which operates within the small
    in-RAM projection buffers (≤ 72 MB for GIST).
    """

    __slots__ = ('d', 'F', 'nlist', 'centroids', 'axes', 'cone_top_k',
                 'cent_sq', '_all_projs', '_assignments', '_all_norms')

    def __init__(self, n, d, F, nlist, centroids, axes, cone_top_k):
        self.d          = d
        self.F          = F
        self.nlist      = nlist
        self.centroids  = np.ascontiguousarray(centroids, dtype=np.float32)
        self.axes       = np.ascontiguousarray(axes,       dtype=np.float32)
        self.cone_top_k = cone_top_k
        self.cent_sq    = np.einsum('ij,ij->i', centroids, centroids)
        self._all_projs   = np.empty((n, F), dtype=np.float32)
        self._assignments = np.empty(n, dtype=np.int32)
        self._all_norms   = np.empty(n, dtype=np.float32)

    def ingest(self, chunk, start_gid):
        """Ingest one contiguous chunk starting at global ID start_gid.

        chunk : (B, d) float32 — already normalised if cosine metric
        """
        B   = chunk.shape[0]
        end = start_gid + B

        chunk_sq = np.einsum('ij,ij->i', chunk, chunk)                      # (B,)
        dots     = chunk @ self.centroids.T                                  # (B, nlist)
        d2       = chunk_sq[:, None] + self.cent_sq[None, :] - 2 * dots    # (B, nlist)
        assign   = np.argmin(d2, axis=1).astype(np.int32)                   # (B,)

        assigned_mu = self.centroids[assign]                                 # (B, d)
        centered    = chunk - assigned_mu                                    # (B, d)
        projs       = (centered @ self.axes.T).astype(np.float32)           # (B, F)
        norms       = np.linalg.norm(centered, axis=1).astype(np.float32)  # (B,)

        self._assignments[start_gid:end] = assign
        self._all_projs[start_gid:end]   = projs
        self._all_norms[start_gid:end]   = norms

    def build_cones(self):
        """Build SortedCone objects and cluster membership from accumulated buffers.

        Returns
        -------
        cones          : list[list[SortedCone]]  — shape (nlist, F)
        cluster_global : list[np.ndarray int32]  — shape (nlist,)
        """
        from ampi._ampi_ext import SortedCone

        K_f = min(self.cone_top_k, self.F)

        # Sort all point indices by cluster — O(n log n), stays in RAM.
        sort_idx      = np.argsort(self._assignments, kind='stable')
        sorted_assign = self._assignments[sort_idx]
        splits        = np.searchsorted(sorted_assign, np.arange(self.nlist + 1))

        cones          = []
        cluster_global = []

        for c in range(self.nlist):
            lo, hi  = int(splits[c]), int(splits[c + 1])
            c_local = sort_idx[lo:hi]                    # row indices in _all_projs
            c_gids  = c_local.astype(np.int32)           # global IDs = row indices
            n_c     = len(c_gids)
            cluster_global.append(c_gids)

            if n_c < 2:
                cones.append([SortedCone(self.F) for _ in range(self.F)])
                continue

            c_projs = self._all_projs[c_local]           # (n_c, F)
            c_norms = self._all_norms[c_local]           # (n_c,)

            safe_norms = np.where(c_norms > 1e-10, c_norms, 1.0)
            normed     = np.abs(c_projs) / safe_norms[:, None]  # (n_c, F)

            if K_f < self.F:
                top_f = np.argpartition(-normed, K_f - 1, axis=1)[:, :K_f]
            else:
                top_f = np.tile(np.arange(self.F, dtype=np.int32), (n_c, 1))

            soft_mask = np.zeros((n_c, self.F), dtype=bool)
            soft_mask[np.arange(n_c)[:, None], top_f] = True

            c_cones = []
            for f in range(self.F):
                f_local = np.where(soft_mask[:, f])[0]
                n_f     = len(f_local)
                if n_f == 0:
                    c_cones.append(SortedCone(self.F))
                    continue

                f_gids  = c_gids[f_local]    # (n_f,) global IDs
                f_projs = c_projs[f_local]   # (n_f, F)

                sorted_projs = np.empty((self.F, n_f), dtype=np.float32)
                sorted_idxs  = np.empty((self.F, n_f), dtype=np.int32)
                for l in range(self.F):
                    o = np.argsort(f_projs[:, l])
                    sorted_projs[l] = f_projs[o, l]
                    sorted_idxs[l]  = o.astype(np.int32)

                c_cones.append(SortedCone.from_arrays(sorted_projs, sorted_idxs, f_gids))

            cones.append(c_cones)

        return cones, cluster_global


def streaming_build(
    data_source,
    n, d,
    nlist,
    num_fans=16,
    cone_top_k=1,
    seed=0,
    metric='l2',
    drift_theta=15.0,
    merge_interval=0,
    eps_merge=1.0,
    merge_qe_ratio=0.5,
    data_path=None,
    chunk_size=4096,
):
    """Build AMPIAffineFanIndex via one sequential streaming pass — O(1) peak RSS.

    Parameters
    ----------
    data_source : callable(start, end) -> (end-start, d) float32 ndarray
        Called in order: start = 0, chunk_size, 2*chunk_size, …
        Typically ``lambda s, e: data[s:e]`` for a memmap or numpy array.
    n, d        : total vectors and dimensionality
    nlist       : coarse k-means clusters
    num_fans    : F — fan directions per cluster
    cone_top_k  : soft-assignment multiplicity
    seed        : RNG seed for axes and k-means init
    metric      : 'l2' or 'cosine'
    data_path   : directory for the mmap data buffer (strongly recommended).
                  Created if absent.  Peak RSS stays ≈ 90 MB for GIST 1M.
    chunk_size  : rows processed per iteration (default 4096)

    Returns
    -------
    AMPIAffineFanIndex (C++ backend) ready for queries and streaming inserts.
    """
    import os
    from .affine_fan import AMPIAffineFanIndex, _normalize_metric, _mini_batch_kmeans

    try:
        from ampi._ampi_ext import AMPIIndex as _AMPIIndex  # noqa: F401 — presence check
    except ImportError:
        raise RuntimeError(
            "streaming_build requires the compiled C++ extension (_ampi_ext). "
            "Run `pip install -e .` to build it."
        )

    metric    = _normalize_metric(metric)
    spherical = (metric == 'cosine')
    F         = num_fans
    rng       = np.random.RandomState(seed)

    # ── Step 1: sequential sample for k-means ────────────────────────────────
    s      = min(50_000, n)
    sample = np.ascontiguousarray(data_source(0, s), dtype=np.float32)
    if spherical:
        nrm    = np.linalg.norm(sample, axis=1, keepdims=True)
        sample = (sample / np.where(nrm < 1e-10, 1.0, nrm)).astype(np.float32)

    centroids, _ = _mini_batch_kmeans(sample, nlist, seed=seed, spherical=spherical)
    del sample

    # ── Step 2: global fan axes ───────────────────────────────────────────────
    axes = rng.randn(F, d).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    # ── Step 3: open mmap data file for writing ───────────────────────────────
    data_mmap = None
    if data_path is not None:
        os.makedirs(data_path, exist_ok=True)
        mmap_path = os.path.join(data_path, '_cpp_data_buf.dat')
        data_mmap = np.memmap(mmap_path, mode='w+', dtype='float32', shape=(n, d))

    # ── Step 4: one streaming pass — assign + project + write mmap ───────────
    dispatcher = _StreamingDispatcher(n, d, F, nlist, centroids, axes, cone_top_k)
    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        chunk = np.ascontiguousarray(data_source(start, end), dtype=np.float32)
        if spherical:
            nrm   = np.linalg.norm(chunk, axis=1, keepdims=True)
            chunk = (chunk / np.where(nrm < 1e-10, 1.0, nrm)).astype(np.float32)
        if data_mmap is not None:
            data_mmap[start:end] = chunk
        dispatcher.ingest(chunk, start)

    if data_mmap is not None:
        data_mmap.flush()
        del data_mmap

    # ── Step 5: build SortedCone objects from projection buffers ─────────────
    cones, cluster_global = dispatcher.build_cones()

    # ── Step 6: assemble via C++ from_stream (no _build_cones random access) ──
    cluster_counts = np.array([len(g) for g in cluster_global], dtype=np.int64)

    return AMPIAffineFanIndex.from_stream(
        n=n, d=d, F=F, nlist=nlist, cone_top_k=cone_top_k,
        metric=metric, drift_theta=drift_theta,
        merge_interval=merge_interval, eps_merge=eps_merge,
        merge_qe_ratio=merge_qe_ratio,
        axes=axes, centroids=centroids,
        cluster_global=cluster_global, cluster_counts=cluster_counts,
        cones=cones, data_path=data_path,
    )
