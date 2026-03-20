"""
Phase 1 tests: best_clusters and best_fan_cones C++ functions.

Compares C++ output against the NumPy reference implementation and checks
that AMPIAffineFanIndex recall is unaffected after the routing change.
"""
import numpy as np
import pytest

try:
    from ampi._ampi_ext import best_clusters, best_fan_cones
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

needs_ext = pytest.mark.skipif(not HAS_EXT, reason="C++ ext not built")

rng = np.random.default_rng(0)
NLIST, F, D = 200, 64, 32


def _np_best_clusters(centroids, q, probes):
    d2 = np.sum((centroids - q) ** 2, axis=1)
    return np.argsort(d2)[:probes]


def _np_best_fan_cones(axes, q_centered, fan_probes):
    q_norm = float(np.linalg.norm(q_centered))
    if q_norm < 1e-10:
        return np.arange(min(fan_probes, len(axes)), dtype=np.int32)
    proj = q_centered @ axes.T / q_norm
    return np.argsort(-np.abs(proj))[:fan_probes]


@needs_ext
def test_best_clusters_matches_numpy():
    centroids = rng.standard_normal((NLIST, D)).astype(np.float32)
    queries   = rng.standard_normal((100, D)).astype(np.float32)
    probes = 5
    for q in queries:
        cpp = best_clusters(centroids, q, probes)
        ref = _np_best_clusters(centroids, q, probes)
        assert list(cpp) == list(ref), f"mismatch: cpp={list(cpp)} ref={list(ref)}"


@needs_ext
def test_best_clusters_probes_clamp():
    centroids = rng.standard_normal((10, D)).astype(np.float32)
    q = rng.standard_normal(D).astype(np.float32)
    out = best_clusters(centroids, q, probes=999)
    assert len(out) == 10


@needs_ext
def test_best_fan_cones_matches_numpy():
    axes = rng.standard_normal((F, D)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    queries = rng.standard_normal((100, D)).astype(np.float32)
    fan_probes = 4
    for q in queries:
        cpp = best_fan_cones(axes, q, fan_probes)
        ref = _np_best_fan_cones(axes, q, fan_probes)
        assert list(cpp) == list(ref), f"mismatch: cpp={list(cpp)} ref={list(ref)}"


@needs_ext
def test_best_fan_cones_zero_query():
    axes = rng.standard_normal((F, D)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    q_zero = np.zeros(D, dtype=np.float32)
    fan_probes = 4
    out = best_fan_cones(axes, q_zero, fan_probes)
    assert len(out) == fan_probes
    assert out.dtype == np.int32


def test_affinefan_recall_unchanged():
    """End-to-end: recall@10 must stay >= 0.80 after Phase 1 routing change."""
    from ampi import AMPIAffineFanIndex
    n, d = 1000, 32
    data = rng.standard_normal((n, d)).astype(np.float32)
    qs   = rng.standard_normal((20, d)).astype(np.float32)

    # Brute-force ground truth (no FAISS needed)
    gt = []
    for q in qs:
        dists = np.linalg.norm(data - q, axis=1)
        gt.append(set(np.argsort(dists)[:10].tolist()))

    idx = AMPIAffineFanIndex(data, nlist=5, num_fans=16, seed=0)
    hits = 0
    for q, g in zip(qs, gt):
        _, _, ids = idx.query(q, k=10, window_size=200, probes=5, fan_probes=16)
        hits += len(g & set(ids[:10].tolist()))
    recall = hits / (len(qs) * 10)
    assert recall >= 0.80, f"recall@10 = {recall:.3f} < 0.80"
