"""
Phase 6 microbenchmark: batch_add(m) must not be slower than m individual add() calls.

All correctness checks from Phase 6 have been promoted to smoke_test.py and
stress_test.py.  Only the timing assertion stays here because throughput ratios
are environment-dependent and unsuitable for the deterministic CI suite.
"""
import time
import numpy as np
import pytest

try:
    from ampi._ampi_ext import AMPIIndex as _AMPIIndex
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

from ampi import AMPIAffineFanIndex

RNG  = np.random.default_rng(42)
N, D = 5_000, 64
DATA = RNG.standard_normal((N, D)).astype(np.float32)


def _build():
    idx = AMPIAffineFanIndex(DATA, nlist=10, num_fans=8, seed=0)
    assert _HAS_CPP and idx._cpp is not None, "C++ ext not available"
    return idx


@pytest.mark.skipif(not _HAS_CPP, reason="C++ ext not built")
def test_batch_add_not_slower_than_loop():
    """batch_add(1000) should not be slower than 1000 individual add() calls."""
    pts = RNG.standard_normal((1000, D)).astype(np.float32)

    REPS = 3
    t0 = time.perf_counter()
    for _ in range(REPS):
        _build()._cpp.batch_add(pts)
    t_batch = (time.perf_counter() - t0) / REPS

    t0 = time.perf_counter()
    for _ in range(REPS):
        idx = _build()
        for row in pts:
            idx._cpp.add(row)
    t_loop = (time.perf_counter() - t0) / REPS

    print(f"\nbatch_add: {t_batch*1e3:.1f} ms, loop add: {t_loop*1e3:.1f} ms, "
          f"ratio: {t_loop/t_batch:.2f}×")
    assert t_batch <= t_loop * 1.2, \
        f"batch_add ({t_batch*1e3:.1f} ms) much slower than loop ({t_loop*1e3:.1f} ms)"
