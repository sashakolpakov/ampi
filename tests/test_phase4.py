"""
Phase 4 microbenchmark: C++ query() must be >= 1.5× faster than Python-path query().

All correctness checks from Phase 4 have been promoted to smoke_test.py and
stress_test.py.  Only the timing assertion stays here because speedup ratios
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
QS   = RNG.standard_normal((50, D)).astype(np.float32)
K    = 10


@pytest.mark.skipif(not _HAS_CPP, reason="C++ ext not built")
def test_cpp_query_faster_than_python():
    """C++ query should be at least 1.5× faster than Python-path query."""
    idx = AMPIAffineFanIndex(DATA, nlist=20, num_fans=16, seed=0)
    assert idx._cpp is not None

    # Warm up
    for q in QS[:5]:
        idx.query(q, k=K, window_size=200, probes=10, fan_probes=2)
        idx._py_query(q, k=K, window_size=200, probes=10, fan_probes=2)

    REPS = 5
    t0 = time.perf_counter()
    for _ in range(REPS):
        for q in QS:
            idx.query(q, k=K, window_size=200, probes=10, fan_probes=2)
    t_cpp = (time.perf_counter() - t0) / (REPS * len(QS))

    t0 = time.perf_counter()
    for _ in range(REPS):
        for q in QS:
            idx._py_query(q, k=K, window_size=200, probes=10, fan_probes=2)
    t_py = (time.perf_counter() - t0) / (REPS * len(QS))

    speedup = t_py / t_cpp
    print(f"\nQuery latency — C++: {t_cpp*1e3:.3f} ms, Python: {t_py*1e3:.3f} ms, "
          f"speedup: {speedup:.2f}×")
    assert speedup >= 1.5, \
        f"C++ query speedup too low: {speedup:.2f}× (expected >= 1.5×)"
