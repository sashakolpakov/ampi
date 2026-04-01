"""Write-Ahead Log for AMPI index mutations.

Append-only binary log; one record per INSERT or DELETE mutation.
Records are replayed on restart from the last checkpoint to recover
in-memory index state without a full rebuild.

Wire format
-----------
File header (18 bytes):
  magic          : 8 bytes  (b"AMPIWAL_")
  version        : uint16
  d              : uint32
  header_crc     : uint32  (CRC32 of the preceding 14 bytes)

Per record:
  op             : uint8   (OP_INSERT=1, OP_DELETE=2)
  global_id      : uint64
  has_vector     : uint8   (1 for INSERT, 0 for DELETE)
  vector         : float32[d]  (only when has_vector=1)
  timestamp_ns   : uint64  (time.time_ns())
  record_crc     : uint32  (CRC32 of all preceding bytes in this record)
"""
import os
import struct
import time
import threading
import warnings
import zlib

import numpy as np

_MAGIC   = b"AMPIWAL_"
_VERSION = 1

OP_INSERT = 1
OP_DELETE = 2

_FILE_HDR_FMT  = "<8sHI"   # magic(8), version(2), d(4) → 14 bytes
_REC_PREFIX_FMT = "<BQB"   # op(1), global_id(8), has_vector(1) → 10 bytes


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


class WALWriter:
    """Append-only WAL writer.

    Thread-safe.  Records are flushed to the OS every ``batch_size`` writes
    (default 1 = flush on every record for durability).

    Parameters
    ----------
    path       : WAL file path (created if absent; appended if it already exists
                 with a matching header).
    d          : vector dimension — must match any existing file.
    batch_size : flush to OS every N records.  1 = flush after every write.
    """

    def __init__(self, path: str, d: int, batch_size: int = 1) -> None:
        self._d          = int(d)
        self._batch_size = max(1, int(batch_size))
        self._lock       = threading.Lock()
        self._pending    = 0

        hdr_size = struct.calcsize(_FILE_HDR_FMT) + 4  # body + CRC

        if os.path.exists(path) and os.path.getsize(path) >= hdr_size:
            with open(path, "rb") as fh:
                raw_body = fh.read(struct.calcsize(_FILE_HDR_FMT))
                (crc_on_disk,) = struct.unpack("<I", fh.read(4))
            magic, ver, d_file = struct.unpack(_FILE_HDR_FMT, raw_body)
            if magic != _MAGIC:
                raise ValueError(f"not an AMPI WAL file: {path}")
            if ver != _VERSION:
                raise ValueError(
                    f"WAL version {ver} != expected {_VERSION}: {path}")
            if d_file != d:
                raise ValueError(
                    f"WAL dimension {d_file} != {d}: {path}")
            if _crc32(raw_body) != crc_on_disk:
                raise ValueError(f"WAL header CRC mismatch: {path}")
            self._fh = open(path, "ab", buffering=0)
        else:
            self._fh = open(path, "wb", buffering=0)
            raw_body = struct.pack(_FILE_HDR_FMT, _MAGIC, _VERSION, d)
            self._fh.write(raw_body + struct.pack("<I", _crc32(raw_body)))

    # ── write helpers ─────────────────────────────────────────────────────────

    def _write_record(self, payload: bytes) -> None:
        with self._lock:
            self._fh.write(payload + struct.pack("<I", _crc32(payload)))
            self._pending += 1
            if self._pending >= self._batch_size:
                self._fh.flush()
                self._pending = 0

    def log_insert(self, global_id: int, vector: "np.ndarray") -> None:
        """Record an INSERT mutation."""
        vec    = np.ascontiguousarray(vector, dtype=np.float32).tobytes()
        prefix = struct.pack(_REC_PREFIX_FMT, OP_INSERT, int(global_id), 1)
        ts_b   = struct.pack("<Q", time.time_ns())
        self._write_record(prefix + vec + ts_b)

    def log_delete(self, global_id: int) -> None:
        """Record a DELETE mutation."""
        prefix = struct.pack(_REC_PREFIX_FMT, OP_DELETE, int(global_id), 0)
        ts_b   = struct.pack("<Q", time.time_ns())
        self._write_record(prefix + ts_b)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Force all buffered records to the OS."""
        with self._lock:
            self._fh.flush()
            self._pending = 0

    def close(self) -> None:
        """Flush and close the WAL file."""
        with self._lock:
            self._fh.flush()
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── reading & replay ──────────────────────────────────────────────────────────

def _iter_records(path: str, d: int):
    """Yield ``(op, global_id, vector_or_None, timestamp_ns)`` from a WAL file.

    Stops at the first bad CRC with a ``warnings.warn`` (handles partial
    trailing writes that result from a crash mid-record).
    """
    hdr_size    = struct.calcsize(_FILE_HDR_FMT)
    pre_size    = struct.calcsize(_REC_PREFIX_FMT)

    with open(path, "rb") as fh:
        raw_body = fh.read(hdr_size)
        if len(raw_body) < hdr_size:
            return
        (crc_on_disk,) = struct.unpack("<I", fh.read(4))
        magic, ver, d_file = struct.unpack(_FILE_HDR_FMT, raw_body)
        if magic != _MAGIC or ver != _VERSION or d_file != d:
            raise ValueError(f"WAL header invalid or mismatched: {path}")
        if _crc32(raw_body) != crc_on_disk:
            raise ValueError(f"WAL header CRC mismatch: {path}")

        while True:
            pre = fh.read(pre_size)
            if not pre:
                break
            if len(pre) < pre_size:
                warnings.warn(f"WAL truncated in record prefix: {path}",
                              stacklevel=2)
                break

            op, gid, has_vec = struct.unpack(_REC_PREFIX_FMT, pre)

            vec_raw = b""
            if has_vec:
                vec_raw = fh.read(d * 4)
                if len(vec_raw) < d * 4:
                    warnings.warn(f"WAL truncated in vector: {path}",
                                  stacklevel=2)
                    break

            ts_raw = fh.read(8)
            if len(ts_raw) < 8:
                warnings.warn(f"WAL truncated in timestamp: {path}",
                              stacklevel=2)
                break

            crc_raw = fh.read(4)
            if len(crc_raw) < 4:
                warnings.warn(f"WAL truncated in CRC: {path}", stacklevel=2)
                break
            (crc_on_disk,) = struct.unpack("<I", crc_raw)

            payload = pre + vec_raw + ts_raw
            if _crc32(payload) != crc_on_disk:
                warnings.warn(
                    f"WAL record CRC mismatch; stopping replay: {path}",
                    stacklevel=2)
                break

            (ts,) = struct.unpack("<Q", ts_raw)
            vec   = (np.frombuffer(vec_raw, dtype=np.float32).copy()
                     if has_vec else None)
            yield int(op), int(gid), vec, int(ts)


def replay_wal(idx, path: str, after_timestamp_ns: int = 0) -> int:
    """Apply WAL records with ``timestamp_ns > after_timestamp_ns`` to *idx*.

    Parameters
    ----------
    idx                : ``AMPIAffineFanIndex`` to mutate.
    path               : path to the WAL file.
    after_timestamp_ns : only replay records newer than this timestamp
                         (nanoseconds since epoch).  Pass the checkpoint's
                         ``timestamp_ns`` to replay only post-checkpoint ops.

    Returns
    -------
    int : number of records applied.
    """
    count = 0
    for op, gid, vec, ts in _iter_records(path, idx.d):
        if ts <= after_timestamp_ns:
            continue
        if op == OP_INSERT:
            idx.add(vec)
        elif op == OP_DELETE:
            idx.delete(gid)
        count += 1
    return count


def truncate_wal(path: str, d: int) -> None:
    """Rewrite the WAL file retaining only the file header (no records).

    Called after a successful checkpoint to reclaim disk space.
    """
    raw_body = struct.pack(_FILE_HDR_FMT, _MAGIC, _VERSION, d)
    with open(path, "wb", buffering=0) as fh:
        fh.write(raw_body + struct.pack("<I", _crc32(raw_body)))
