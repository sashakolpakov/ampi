"""test_persistence.py — WAL and checkpoint round-trip tests.

Runs without large datasets; uses a small synthetic index (n=2000, d=32).
"""
import os

import numpy as np
import pytest

from ampi import AMPIAffineFanIndex
from ampi.wal import (
    WALWriter, OP_INSERT, OP_DELETE,
    _iter_records, replay_wal, truncate_wal,
)
from ampi.checkpoint import save_checkpoint, load_checkpoint

try:
    import ampi._ampi_ext  # noqa: F401
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False


RNG = np.random.default_rng(0)
N, D = 2_000, 32
DATA = RNG.standard_normal((N, D)).astype(np.float32)
QS   = RNG.standard_normal((20, D)).astype(np.float32)


def _build_idx(tmp_dir, wal_path=None, wal_batch_size=1):
    return AMPIAffineFanIndex(
        DATA, nlist=32, num_fans=8, seed=0, cone_top_k=1,
        wal_path=wal_path, wal_batch_size=wal_batch_size,
    )


def _recall10(idx, queries=QS):
    from ampi.tuner import _brute_knn
    gt = _brute_knn(DATA, queries, 10)
    hits = 0
    for i, q in enumerate(queries):
        _, _, ids = idx.query(q, k=10, window_size=200, probes=8, fan_probes=8)
        hits += len(set(gt[i].tolist()) & set(ids[:10].tolist()))
    return hits / (len(queries) * 10)


# ── WAL tests ─────────────────────────────────────────────────────────────────

class TestWAL:
    def test_write_read_roundtrip(self, tmp_path):
        wal = str(tmp_path / "test.wal")
        vec = np.ones(D, dtype=np.float32)
        with WALWriter(wal, D) as w:
            w.log_insert(0, vec)
            w.log_delete(0)

        records = list(_iter_records(wal, D))
        assert len(records) == 2

        op0, gid0, v0, ts0 = records[0]
        assert op0 == OP_INSERT
        assert gid0 == 0
        assert v0 is not None
        np.testing.assert_array_equal(v0, vec)

        op1, gid1, v1, ts1 = records[1]
        assert op1 == OP_DELETE
        assert gid1 == 0
        assert v1 is None
        assert ts1 >= ts0

    def test_append_to_existing(self, tmp_path):
        wal = str(tmp_path / "append.wal")
        vec = RNG.standard_normal(D).astype(np.float32)
        with WALWriter(wal, D) as w:
            w.log_insert(0, vec)
        with WALWriter(wal, D) as w:
            w.log_insert(1, vec)
        records = list(_iter_records(wal, D))
        assert len(records) == 2

    def test_dimension_mismatch(self, tmp_path):
        wal = str(tmp_path / "bad.wal")
        with WALWriter(wal, D):
            pass
        with pytest.raises(ValueError, match="dimension"):
            WALWriter(wal, D + 1)

    def test_bad_magic(self, tmp_path):
        wal = str(tmp_path / "bad_magic.wal")
        with open(wal, "wb") as fh:
            fh.write(b"NOTAWAL_" + b"\x00" * 20)
        with pytest.raises(ValueError):
            list(_iter_records(wal, D))

    def test_truncate(self, tmp_path):
        wal = str(tmp_path / "trunc.wal")
        vec = np.ones(D, dtype=np.float32)
        with WALWriter(wal, D) as w:
            w.log_insert(0, vec)
        assert len(list(_iter_records(wal, D))) == 1
        truncate_wal(wal, D)
        assert list(_iter_records(wal, D)) == []

    def test_batch_flush(self, tmp_path):
        wal = str(tmp_path / "batch.wal")
        vec = np.ones(D, dtype=np.float32)
        with WALWriter(wal, D, batch_size=64) as w:
            for i in range(200):
                w.log_insert(i, vec)
        records = list(_iter_records(wal, D))
        assert len(records) == 200


class TestWALIntegration:
    def test_wal_logged_on_add_delete(self, tmp_path):
        if not _HAS_EXT:
            pytest.skip("C++ ext required")
        wal = str(tmp_path / "mutations.wal")
        idx = _build_idx(tmp_path, wal_path=wal)
        new_vec = RNG.standard_normal(D).astype(np.float32)
        gid = idx.add(new_vec)
        idx.delete(gid)

        records = list(_iter_records(wal, D))
        assert len(records) == 2
        assert records[0][0] == OP_INSERT
        assert records[1][0] == OP_DELETE

    def test_replay_restores_state(self, tmp_path):
        if not _HAS_EXT:
            pytest.skip("C++ ext required")
        wal = str(tmp_path / "replay.wal")
        idx = _build_idx(tmp_path, wal_path=wal)
        n_before = idx.n
        vecs = RNG.standard_normal((10, D)).astype(np.float32)
        for v in vecs:
            idx.add(v)
        assert idx.n == n_before + 10

        # Build a fresh index and replay
        idx2 = _build_idx(tmp_path)
        replayed = replay_wal(idx2, wal)
        assert replayed == 10
        assert idx2.n == n_before + 10


# ── checkpoint tests ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_EXT, reason="C++ ext required")
class TestCheckpoint:
    def test_round_trip_small(self, tmp_path):
        ckpt = str(tmp_path / "idx.ckpt")
        mmap_dir = str(tmp_path / "mmap")
        os.makedirs(mmap_dir)

        idx = AMPIAffineFanIndex(
            DATA, nlist=32, num_fans=8, seed=0, cone_top_k=1,
            data_path=mmap_dir,
        )
        rec_before = _recall10(idx)

        save_checkpoint(idx, ckpt)
        assert os.path.getsize(ckpt) > 0

        idx2 = load_checkpoint(ckpt, data_path=mmap_dir)
        assert idx2.n == idx.n
        assert idx2.d == idx.d
        assert idx2.nlist == idx.nlist
        assert idx2.F == idx.F
        assert idx2.cone_top_k == idx.cone_top_k
        np.testing.assert_allclose(idx2.centroids, idx.centroids, atol=1e-5)

        rec_after = _recall10(idx2)
        assert abs(rec_after - rec_before) < 0.05, (
            f"recall dropped after checkpoint round-trip: "
            f"{rec_before:.3f} → {rec_after:.3f}"
        )

    def test_checkpoint_returns_timestamp(self, tmp_path):
        ckpt = str(tmp_path / "ts.ckpt")
        mmap_dir = str(tmp_path / "mmap")
        os.makedirs(mmap_dir)
        idx = AMPIAffineFanIndex(
            DATA, nlist=16, num_fans=8, seed=0, data_path=mmap_dir,
        )
        import time
        t0 = time.time_ns()
        ts = save_checkpoint(idx, ckpt)
        t1 = time.time_ns()
        assert t0 <= ts <= t1

    def test_checkpoint_header_crc(self, tmp_path):
        ckpt = str(tmp_path / "crc.ckpt")
        mmap_dir = str(tmp_path / "mmap")
        os.makedirs(mmap_dir)
        idx = AMPIAffineFanIndex(
            DATA, nlist=16, num_fans=8, seed=0, data_path=mmap_dir,
        )
        save_checkpoint(idx, ckpt)
        # Corrupt one byte in the header body
        with open(ckpt, "r+b") as fh:
            fh.seek(10)
            fh.write(b"\xff")
        with pytest.raises(ValueError, match="CRC"):
            load_checkpoint(ckpt, data_path=mmap_dir)

    def test_checkpoint_then_wal_replay(self, tmp_path):
        ckpt    = str(tmp_path / "idx.ckpt")
        wal     = str(tmp_path / "mutations.wal")
        mmap_dir = str(tmp_path / "mmap")
        os.makedirs(mmap_dir)

        idx = AMPIAffineFanIndex(
            DATA, nlist=32, num_fans=8, seed=0, cone_top_k=1,
            data_path=mmap_dir, wal_path=wal,
        )
        ts = save_checkpoint(idx, ckpt)

        # Post-checkpoint mutations
        new_vecs = RNG.standard_normal((5, D)).astype(np.float32)
        new_gids = [idx.add(v) for v in new_vecs]
        idx.delete(new_gids[0])
        n_final = idx.n

        # Restore from checkpoint + WAL replay
        idx2 = load_checkpoint(ckpt, data_path=mmap_dir)
        replayed = replay_wal(idx2, wal, after_timestamp_ns=ts)
        assert replayed == 6   # 5 inserts + 1 delete
        assert idx2.n == n_final

        truncate_wal(wal, D)
        assert list(_iter_records(wal, D)) == []

    def test_load_without_data_path_raises(self, tmp_path):
        ckpt = str(tmp_path / "idx.ckpt")
        mmap_dir = str(tmp_path / "mmap")
        os.makedirs(mmap_dir)
        idx = AMPIAffineFanIndex(
            DATA, nlist=16, num_fans=8, seed=0, data_path=mmap_dir,
        )
        save_checkpoint(idx, ckpt)
        with pytest.raises(ValueError, match="data_path"):
            load_checkpoint(ckpt)

    def test_no_cpp_raises(self, tmp_path, monkeypatch):
        ckpt = str(tmp_path / "idx.ckpt")
        mmap_dir = str(tmp_path / "mmap")
        os.makedirs(mmap_dir)
        idx = AMPIAffineFanIndex(
            DATA, nlist=16, num_fans=8, seed=0, data_path=mmap_dir,
        )
        # Simulate no C++ ext by removing _cpp
        monkeypatch.setattr(idx, "_cpp", None)
        with pytest.raises(RuntimeError, match="C\\+\\+ extension"):
            save_checkpoint(idx, ckpt)
