"""Checkpoint serializer / deserializer for AMPIAffineFanIndex.

Requires the compiled C++ extension (``ampi._ampi_ext``).

Checkpoint file layout
----------------------
Header (66 bytes):
  magic            8 B    b"AMPI_CKP"
  version          2 B    uint16 = 1
  n                8 B    uint64
  d                4 B    uint32
  nlist            4 B    uint32
  F                4 B    uint32
  cone_top_k       4 B    uint32
  metric_id        1 B    uint8  (0=l2  1=sqeuclidean  2=cosine)
  _pad             3 B    zeros
  drift_theta      4 B    float32
  merge_interval   4 B    int32   (stored as int32; cast from Python int)
  eps_merge        4 B    float32
  merge_qe_ratio   4 B    float32  (note: NOT int32 despite `fifi` mnemonic)
  timestamp_ns     8 B    uint64
  header_crc       4 B    uint32 (CRC32 of preceding 62 bytes)

Fixed-size arrays:
  centroids        nlist × d × 4 B    float32[nlist, d]
  axes             F × d × 4 B        float32[F, d]
  cluster_counts   nlist × 8 B        int64[nlist]
  U_drift          nlist × d × F × 4 B  float32[nlist, d, F]
  (saved for completeness; not restored on load — reinitialised to zero)

Per-cluster variable-length data (repeated nlist times):
  n_c              4 B    uint32  — number of global IDs in this cluster
  global_ids       n_c × 4 B      int32[n_c]
  Per-cone (F cones per cluster):
    n_f            4 B    uint32  — pairs per cone  (0 → empty cone)
    Per-axis l = 0..F-1 (only when n_f > 0):
      projs        n_f × 4 B      float32[n_f]  — projection values, sorted
      ids          n_f × 4 B      uint32[n_f]   — global IDs, sorted by projs

Usage
-----
    from ampi.checkpoint import save_checkpoint, load_checkpoint
    from ampi.wal        import replay_wal, truncate_wal

    # --- on shutdown / periodic checkpoint ---
    ts = save_checkpoint(idx, "index.ckpt")

    # --- on startup ---
    idx = load_checkpoint("index.ckpt", data_path="/path/to/mmap/dir")
    replay_wal(idx, "mutations.wal", after_timestamp_ns=ts)
    truncate_wal("mutations.wal", idx.d)
"""
import struct
import time
import zlib

import numpy as np

_MAGIC   = b"AMPI_CKP"
_VERSION = 1

_METRIC_TO_ID = {"l2": 0, "sqeuclidean": 1, "cosine": 2}
_ID_TO_METRIC = {v: k for k, v in _METRIC_TO_ID.items()}

# magic(8) version(2) n(8) d(4) nlist(4) F(4) cone_top_k(4)
# metric_id(1) _pad(3) drift_theta(4f) merge_interval(4i)
# eps_merge(4f) merge_qe_ratio(4f) timestamp_ns(8)
# → 62 bytes body; CRC appended separately → 66 bytes total
_HDR_BODY_FMT  = "<8sHQIIIIBxxxfiffQ"
_HDR_BODY_SIZE = struct.calcsize(_HDR_BODY_FMT)  # 62


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# ── save ──────────────────────────────────────────────────────────────────────

def save_checkpoint(idx, path: str) -> int:
    """Serialize *idx* to a binary checkpoint file at *path*.

    Parameters
    ----------
    idx  : ``AMPIAffineFanIndex`` with the compiled C++ extension.
    path : destination file path (overwritten if it exists).

    Returns
    -------
    int : ``timestamp_ns`` embedded in the checkpoint header (nanoseconds
          since epoch).  Pass this to :func:`replay_wal` as
          ``after_timestamp_ns`` to replay only post-checkpoint mutations.

    Raises
    ------
    RuntimeError : if the C++ extension is not available.
    """
    cpp = getattr(idx, "_cpp", None)
    if cpp is None:
        raise RuntimeError(
            "save_checkpoint requires the compiled C++ extension. "
            "Run `pip install -e .` to build it."
        )

    metric_id = _METRIC_TO_ID.get(idx.metric, 0)
    ts        = time.time_ns()

    hdr_body = struct.pack(
        _HDR_BODY_FMT,
        _MAGIC, _VERSION,
        idx.n, idx.d, idx.nlist, idx.F, idx.cone_top_k,
        metric_id,
        float(idx.drift_theta), int(idx.merge_interval),
        float(idx.eps_merge), float(idx.merge_qe_ratio),
        ts,
    )
    hdr_crc = struct.pack("<I", _crc32(hdr_body))

    with open(path, "wb") as fh:
        fh.write(hdr_body + hdr_crc)

        # ── fixed-size arrays ─────────────────────────────────────────────
        fh.write(cpp.get_centroids().astype(np.float32).tobytes())  # [nlist, d]
        fh.write(cpp.get_axes().astype(np.float32).tobytes())       # [F, d]
        fh.write(cpp.get_cluster_counts().astype(np.int64).tobytes())  # [nlist]

        # U_drift: save for completeness; not restored on load (reinitialised
        # to zero by from_stream; drift detection resumes from scratch).
        U_all = np.empty((idx.nlist, idx.d, idx.F), dtype=np.float32)
        for c in range(idx.nlist):
            U_all[c] = cpp.get_U_drift(c)
        fh.write(U_all.tobytes())

        # ── per-cluster variable-length data ──────────────────────────────
        for c in range(idx.nlist):
            cg  = cpp.get_cluster_global(c)               # int32[n_c]
            n_c = len(cg)
            fh.write(struct.pack("<I", n_c))
            fh.write(np.asarray(cg, dtype=np.int32).tobytes())

            for f in range(idx.F):
                if not cpp.has_cones(c):
                    fh.write(struct.pack("<I", 0))        # empty cone
                    continue
                cone = cpp.get_cone(c, f)
                if cone is None or cone.size() == 0:
                    fh.write(struct.pack("<I", 0))
                    continue
                projs0, _ = cone.get_axis_pairs(0)
                n_f = len(projs0)
                fh.write(struct.pack("<I", n_f))
                for ax in range(idx.F):
                    projs_l, ids_l = cone.get_axis_pairs(ax)
                    fh.write(projs_l.astype(np.float32).tobytes())
                    fh.write(ids_l.astype(np.uint32).tobytes())

    return ts


# ── load ──────────────────────────────────────────────────────────────────────

def load_checkpoint(path: str, data_path: str = None):
    """Deserialize a checkpoint and return a ready-to-query ``AMPIAffineFanIndex``.

    Parameters
    ----------
    path       : path to the checkpoint file.
    data_path  : directory containing the mmap raw-vector file
                 (``_cpp_data_buf.dat``).  This file is **not** stored inside
                 the checkpoint; it must have been preserved from the original
                 build.  Required — ``from_stream`` will raise if absent.

    Returns
    -------
    AMPIAffineFanIndex

    Raises
    ------
    RuntimeError : if the C++ extension is not available.
    ValueError   : if the checkpoint header is corrupt.
    """
    try:
        from ampi._ampi_ext import SortedCone as _SortedCone
        from ampi.affine_fan import AMPIAffineFanIndex
    except ImportError:
        raise RuntimeError(
            "load_checkpoint requires the compiled C++ extension. "
            "Run `pip install -e .` to build it."
        )

    if data_path is None:
        raise ValueError(
            "data_path is required: load_checkpoint needs the directory that "
            "contains _cpp_data_buf.dat (the mmap raw-vector file from the "
            "original build).  The raw vectors are not stored in the checkpoint."
        )

    with open(path, "rb") as fh:
        # ── header ───────────────────────────────────────────────────────────
        hdr_body = fh.read(_HDR_BODY_SIZE)
        if len(hdr_body) < _HDR_BODY_SIZE:
            raise ValueError(f"checkpoint too short: {path}")
        (crc_on_disk,) = struct.unpack("<I", fh.read(4))
        if _crc32(hdr_body) != crc_on_disk:
            raise ValueError(f"checkpoint header CRC mismatch: {path}")

        (magic, version, n, d, nlist, F, cone_top_k,
         metric_id, drift_theta, merge_interval, eps_merge, merge_qe_ratio,
         ts_ns) = struct.unpack(_HDR_BODY_FMT, hdr_body)

        if magic != _MAGIC:
            raise ValueError(f"not an AMPI checkpoint: {path}")
        if version != _VERSION:
            raise ValueError(
                f"checkpoint version {version} != expected {_VERSION}: {path}")

        metric = _ID_TO_METRIC.get(metric_id, "l2")

        # ── fixed-size arrays ─────────────────────────────────────────────
        centroids = np.frombuffer(
            fh.read(int(nlist) * int(d) * 4),
            dtype=np.float32).reshape(nlist, d).copy()
        axes = np.frombuffer(
            fh.read(int(F) * int(d) * 4),
            dtype=np.float32).reshape(F, d).copy()
        cluster_counts = np.frombuffer(
            fh.read(int(nlist) * 8), dtype=np.int64).copy()
        fh.read(int(nlist) * int(d) * int(F) * 4)  # U_drift — skip (reinit to 0)

        # ── per-cluster variable-length data ──────────────────────────────
        cluster_global = []
        all_cones      = []   # list[list[SortedCone]]

        for _c in range(int(nlist)):
            (n_c,) = struct.unpack("<I", fh.read(4))
            cg = np.frombuffer(
                fh.read(int(n_c) * 4), dtype=np.int32).copy()
            cluster_global.append(cg)

            cones_c = []
            for _f in range(int(F)):
                (n_f,) = struct.unpack("<I", fh.read(4))
                if n_f == 0:
                    cones_c.append(_SortedCone(int(F)))   # empty cone
                    continue

                all_projs = []
                all_ids   = []
                for _l in range(int(F)):
                    projs_l = np.frombuffer(
                        fh.read(int(n_f) * 4), dtype=np.float32).copy()
                    ids_l   = np.frombuffer(
                        fh.read(int(n_f) * 4), dtype=np.uint32).copy()
                    all_projs.append(projs_l)
                    all_ids.append(ids_l)

                cone = _reconstruct_cone(all_projs, all_ids, int(F), int(n_f))
                cones_c.append(cone)

            all_cones.append(cones_c)

    idx = AMPIAffineFanIndex.from_stream(
        n=int(n), d=int(d), F=int(F), nlist=int(nlist),
        cone_top_k=int(cone_top_k),
        metric=metric,
        drift_theta=float(drift_theta),
        merge_interval=int(merge_interval),
        eps_merge=float(eps_merge),
        merge_qe_ratio=float(merge_qe_ratio),
        axes=axes,
        centroids=centroids,
        cluster_global=cluster_global,
        cluster_counts=cluster_counts,
        cones=all_cones,
        data_path=data_path,
    )
    return idx


# ── internal helpers ──────────────────────────────────────────────────────────

def _reconstruct_cone(all_projs, all_ids, F, n_f):
    """Rebuild a SortedCone from serialized per-axis (projs, ids) arrays.

    Parameters
    ----------
    all_projs : list of F float32[n_f] — projection values sorted per axis
    all_ids   : list of F uint32[n_f]  — global IDs sorted per axis
    F         : number of fan axes
    n_f       : number of points in the cone

    Returns
    -------
    SortedCone
    """
    from ampi._ampi_ext import SortedCone as _SortedCone

    # global_idx: use axis 0's sorted order as the canonical local ordering.
    global_idx = all_ids[0].astype(np.int32)      # (n_f,)

    # Build a lookup: global_id → local index in global_idx.
    # Using argsort + searchsorted avoids a Python dict for large cones.
    argsort_g0 = np.argsort(global_idx)           # global_idx[argsort_g0] sorted
    sorted_g0  = global_idx[argsort_g0]           # sorted copy for searchsorted

    sorted_projs = np.stack(all_projs)            # (F, n_f) float32
    sorted_idxs  = np.empty((F, n_f), dtype=np.int32)

    for ax in range(F):
        # pos[j]: position of all_ids[ax][j] in sorted_g0
        pos              = np.searchsorted(sorted_g0, all_ids[ax].astype(np.int32))
        sorted_idxs[ax]  = argsort_g0[pos]

    return _SortedCone.from_arrays(
        sorted_projs.astype(np.float32),
        sorted_idxs,
        global_idx,
    )
