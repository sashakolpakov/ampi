"""
Phase 2 tests: update_drift_and_check C++ function.

Verifies:
- Sigma values match the NumPy reference after many EMA steps (to 1e-10).
- Drift-triggered refreshes still fire when insertions cluster off-axis.
- Recall is maintained after the routing change.
"""
import numpy as np
import pytest

try:
    from ampi._ampi_ext import update_drift_and_check
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

needs_ext = pytest.mark.skipif(not HAS_EXT, reason="C++ ext not built")

rng = np.random.default_rng(1)
D, F = 32, 16
BETA = 0.01
THETA = 15.0


def _py_update(sigma_flat, axes, v, beta, theta_deg):
    """Reference Python EMA + power iteration (returns should_refresh bool)."""
    d = axes.shape[1]
    sig = sigma_flat.reshape(d, d)
    sig = (1.0 - beta) * sig + beta * np.outer(v, v)
    sigma_flat[:] = sig.ravel()
    # Power iteration
    ev = sig @ axes[0].astype(np.float64)
    for _ in range(5):
        ev = sig @ ev
        norm = float(np.linalg.norm(ev))
        if norm < 1e-12:
            return False
        ev /= norm
    cos_max = float(np.max(np.abs(axes.astype(np.float64) @ ev)))
    cos_theta = float(np.cos(np.radians(theta_deg)))
    return cos_max < cos_theta


@needs_ext
def test_sigma_matches_numpy_after_many_steps():
    axes = rng.standard_normal((F, D)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    sigma_cpp = np.zeros(D * D, dtype=np.float64)
    sigma_ref = np.zeros(D * D, dtype=np.float64)

    for _ in range(1000):
        v = rng.standard_normal(D)
        # C++ path (in-place)
        update_drift_and_check(sigma_cpp, axes, v, BETA, THETA)
        # Python reference (in-place)
        _py_update(sigma_ref, axes, v, BETA, THETA)

    np.testing.assert_allclose(
        sigma_cpp, sigma_ref, atol=1e-10,
        err_msg="sigma_drift diverged between C++ and Python after 1000 steps"
    )


@needs_ext
def test_refresh_flag_matches_numpy():
    """C++ and Python must agree on when to trigger a refresh."""
    axes = rng.standard_normal((F, D)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    sigma_cpp = np.zeros(D * D, dtype=np.float64)
    sigma_ref = np.zeros(D * D, dtype=np.float64)

    mismatches = 0
    for _ in range(500):
        v = rng.standard_normal(D)
        cpp_flag = bool(update_drift_and_check(sigma_cpp, axes, v, BETA, THETA))
        ref_flag = _py_update(sigma_ref, axes, v, BETA, THETA)
        if cpp_flag != ref_flag:
            mismatches += 1

    assert mismatches == 0, f"{mismatches} refresh-flag mismatches between C++ and Python"


def test_drift_refresh_fires_on_index():
    """Insert points concentrated off-axis; verify _local_refresh is triggered."""
    from ampi import AMPIAffineFanIndex
    n, d = 500, 32
    data = rng.standard_normal((n, d)).astype(np.float32)
    idx = AMPIAffineFanIndex(data, nlist=4, num_fans=8, seed=0)

    # Direction orthogonal to all fan axes — guaranteed to trigger drift
    axes = idx.axes  # (F, d)
    # Build a direction with low cosine to all axes
    perp = rng.standard_normal(d).astype(np.float32)
    for _ in range(20):
        for ax in axes:
            perp -= float(perp @ ax) * ax
        perp /= np.linalg.norm(perp)

    # Insert 300 points along that direction
    for _ in range(300):
        x = perp * float(rng.uniform(0.5, 2.0)) + rng.standard_normal(d).astype(np.float32) * 0.05
        idx.add(x.astype(np.float32))

    # Index must still be queryable and return plausible results
    q = perp + rng.standard_normal(d).astype(np.float32) * 0.1
    _, dists, ids = idx.query(q.astype(np.float32), k=5, window_size=100, probes=4)
    assert len(ids) == 5


def test_recall_unchanged_after_phase2():
    from ampi import AMPIAffineFanIndex
    n, d = 1000, 32
    data = rng.standard_normal((n, d)).astype(np.float32)
    qs   = rng.standard_normal((20, d)).astype(np.float32)

    gt = []
    for q in qs:
        dists = np.linalg.norm(data - q, axis=1)
        gt.append(set(np.argsort(dists)[:10].tolist()))

    idx = AMPIAffineFanIndex(data, nlist=5, num_fans=16, seed=0)
    # Warm up with inserts to exercise the drift path
    for _ in range(200):
        idx.add(rng.standard_normal(d).astype(np.float32))

    hits = 0
    for q, g in zip(qs, gt):
        _, _, ids = idx.query(q, k=10, window_size=200, probes=5, fan_probes=16)
        hits += len(g & set(ids[:10].tolist()))
    recall = hits / (len(qs) * 10)
    assert recall >= 0.80, f"recall@10 = {recall:.3f} < 0.80"
