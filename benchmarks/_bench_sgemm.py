"""
Microbenchmark: old C++ scalar loop (replicated in numba) vs new Accelerate SGEMM.
"""
import numpy as np, time
import ampi._ampi_ext as ext
from numba import jit

@jit(nopython=True, parallel=True)
def project_data_old(data, proj_dirs):
    """Exact replica of the old triple loop that lived in _ext.cpp."""
    L = proj_dirs.shape[0]
    n = data.shape[0]
    d = data.shape[1]
    out = np.empty((L, n), dtype=np.float32)
    for i in range(L):
        for k in range(n):
            dot = np.float32(0.0)
            for j in range(d):
                dot += proj_dirs[i, j] * data[k, j]
            out[i, k] = dot
    return out

rng = np.random.default_rng(0)

# warm up numba JIT
_d = rng.standard_normal((100, 64)).astype(np.float32)
_p = rng.standard_normal((4, 64)).astype(np.float32)
project_data_old(_d, _p)

print("%-30s  %10s  %10s  %8s" % ("shape (n x d, L)", "old loop", "SGEMM", "speedup"))
print("-" * 65)

configs = [
    (10_000,   64, 16),
    (10_000,  128, 16),
    (100_000, 128, 16),
    (500_000, 128, 16),
    (10_000,  512, 32),
    (100_000, 512, 32),
]

for n, d, L in configs:
    data = rng.standard_normal((n, d)).astype(np.float32)
    dirs = rng.standard_normal((L, d)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    for _ in range(2):
        project_data_old(data, dirs)
        ext.project_data(data, dirs)

    reps = max(3, int(2e9 / (n * d * L)))

    t0 = time.perf_counter()
    for _ in range(reps):
        project_data_old(data, dirs)
    old_ms = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        ext.project_data(data, dirs)
    new_ms = (time.perf_counter() - t0) / reps * 1000

    label = "n=%7d  d=%4d  L=%2d" % (n, d, L)
    print("%-30s  %10.2f  %10.2f  %7.1fx" % (label, old_ms, new_ms, old_ms / new_ms))
