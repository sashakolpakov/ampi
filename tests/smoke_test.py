"""
Smoke test: build small indexes, verify exact NN is found at high recall.
Runs in ~10s on a laptop, no datasets required.
"""
import numpy as np
import faiss

from ampi import AMPIBinaryIndex, AMPIAffineFanIndex

rng  = np.random.default_rng(42)
n, d = 5_000, 64
data = rng.standard_normal((n, d)).astype("float32")
qs   = rng.standard_normal((50, d)).astype("float32")

# Ground truth
flat = faiss.IndexFlatL2(d)
flat.add(data)
_, gt = flat.search(qs, 10)


def recall10(gt, found):
    hits = sum(len(set(g.tolist()) & set(f[:10].tolist())) for g, f in zip(gt, found))
    return hits / (len(gt) * 10)


# ── Binary ────────────────────────────────────────────────────────────────────
idx_b = AMPIBinaryIndex(data, num_projections=64, seed=0)
results = [idx_b.query(q, k=10, window_size=100) for q in qs]
found_b = [r[2] for r in results]
rec_b   = recall10(gt, found_b)
print(f"Binary   recall@10 = {rec_b:.3f}")
assert rec_b >= 0.80, f"Binary recall too low: {rec_b:.3f}"

# ── AffineFan ─────────────────────────────────────────────────────────────────
idx_a = AMPIAffineFanIndex(data, nlist=5, num_fans=16, seed=0)
results = [idx_a.query(q, k=10, window_size=200, probes=5, fan_probes=16) for q in qs]
found_a = [r[2] for r in results]
rec_a   = recall10(gt, found_a)
print(f"AffineFan recall@10 = {rec_a:.3f}")
assert rec_a >= 0.80, f"AffineFan recall too low: {rec_a:.3f}"

print("OK")

# ── AMPIIndex C++ accessor compatibility ──────────────────────────────────────
# Verifies that every public C++ field and method of AMPIIndex is reachable
# from Python.  This test must be updated whenever new members are added to
# the C++ class to ensure the pybind11 binding stays complete.

try:
    from ampi._ampi_ext import AMPIIndex as _AMPIIndex
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

if _HAS_CPP:
    cpp = idx_a._cpp
    assert cpp is not None, "AMPIIndex not constructed"

    # ── scalar properties ────────────────────────────────────────────────────
    assert isinstance(cpp.n,             int), "n not int"
    assert isinstance(cpp.n_deleted,     int), "n_deleted not int"
    assert isinstance(cpp.capacity,      int), "capacity not int"
    assert isinstance(cpp.nlist,         int), "nlist not int"
    assert isinstance(cpp.F,             int), "F not int"
    assert isinstance(cpp.d,             int), "d not int"
    assert isinstance(cpp.cone_top_k,    int), "cone_top_k not int"
    assert isinstance(cpp.cosine_metric, bool), "cosine_metric not bool"
    assert isinstance(cpp.drift_theta,   float), "drift_theta not float"

    # drift_theta must be writable
    old_theta = cpp.drift_theta
    cpp.drift_theta = 20.0
    assert cpp.drift_theta == 20.0, "drift_theta not writable"
    cpp.drift_theta = old_theta

    # ── array views ──────────────────────────────────────────────────────────
    import numpy as np
    dv = cpp.get_data_view();        assert dv.shape == (cpp.n, cpp.d),     "data_view shape"
    dm = cpp.get_deleted_mask();     assert dm.shape == (cpp.n,),           "del_mask shape"
    cv = cpp.get_centroids();        assert cv.shape == (cpp.nlist, cpp.d), "centroids shape"
    ax = cpp.get_axes();             assert ax.shape == (cpp.F, cpp.d),     "axes shape"
    cc = cpp.get_cluster_counts();   assert len(cc) == cpp.nlist,           "cluster_counts len"
    ct = cpp.get_cluster_tombstones(); assert len(ct) == cpp.nlist,         "cluster_tombstones len"

    # ── per-cluster accessors ────────────────────────────────────────────────
    for c in range(cpp.nlist):
        gi = cpp.get_cluster_global(c)
        assert isinstance(gi, np.ndarray), f"cluster_global[{c}] not ndarray"
        hc = cpp.has_cones(c)
        assert isinstance(hc, bool), f"has_cones({c}) not bool"
        for f in range(cpp.F):
            cone = cpp.get_cone(c, f)
            # SortedCone methods accessible
            _ = cone.size()
            _ = cone.all_ids()

    # ── per-point inverse map ─────────────────────────────────────────────────
    pc = cpp.get_point_cones(0)
    assert isinstance(pc, list), "get_point_cones not list"

    # ── mutating methods ──────────────────────────────────────────────────────
    x_test = np.zeros(cpp.d, dtype=np.float32)
    gid = cpp.add(x_test)
    assert isinstance(gid, int), "add() did not return int"
    cpp.remove(gid)

    print("AMPIIndex accessor compatibility: OK")
else:
    print("AMPIIndex accessor compatibility: SKIPPED (C++ ext not built)")


# ── Metric alias testing ───────────────────────────────────────────────────────
# Verify all documented aliases construct an index and return correctly-shaped
# distances, and that an unknown alias raises ValueError.

from ampi.affine_fan import _normalize_metric

# canonical round-trip
assert _normalize_metric('l2')          == 'l2'
assert _normalize_metric('L2')          == 'l2'
assert _normalize_metric('euclidean')   == 'l2'
assert _normalize_metric('sqeuclidean') == 'sqeuclidean'
assert _normalize_metric('cosine')      == 'cosine'

# unknown alias raises ValueError
try:
    _normalize_metric('manhattan')
    raise AssertionError("expected ValueError for unknown metric")
except ValueError:
    pass

q = qs[0]

# l2 / euclidean aliases give identical results and non-negative distances
idx_l2  = AMPIAffineFanIndex(data, nlist=5, num_fans=16, seed=0, metric='l2')
idx_eu  = AMPIAffineFanIndex(data, nlist=5, num_fans=16, seed=0, metric='euclidean')
idx_L2  = AMPIAffineFanIndex(data, nlist=5, num_fans=16, seed=0, metric='L2')
_, d_l2, ids_l2 = idx_l2.query(q, k=10, window_size=200, probes=5, fan_probes=16)
_, d_eu, ids_eu = idx_eu.query(q, k=10, window_size=200, probes=5, fan_probes=16)
_, d_L2, ids_L2 = idx_L2.query(q, k=10, window_size=200, probes=5, fan_probes=16)
assert (d_l2 >= 0).all(),                 "l2 distances not non-negative"
np.testing.assert_array_equal(ids_l2, ids_eu, err_msg="l2 vs euclidean ids differ")
np.testing.assert_array_equal(ids_l2, ids_L2, err_msg="l2 vs L2 ids differ")
np.testing.assert_allclose(d_l2, d_eu, rtol=1e-5, err_msg="l2 vs euclidean dists differ")

# sqeuclidean distances are squares of l2 distances
idx_sq = AMPIAffineFanIndex(data, nlist=5, num_fans=16, seed=0, metric='sqeuclidean')
_, d_sq, ids_sq = idx_sq.query(q, k=10, window_size=200, probes=5, fan_probes=16)
assert (d_sq >= 0).all(), "sqeuclidean distances not non-negative"
np.testing.assert_array_equal(ids_l2, ids_sq, err_msg="l2 vs sqeuclidean ids differ")
np.testing.assert_allclose(d_l2 ** 2, d_sq, rtol=1e-4, err_msg="sqeuclidean != l2^2")

# cosine: index normalises internally — pass raw vectors, distances in [0, 1]
idx_cos   = AMPIAffineFanIndex(data, nlist=5, num_fans=16, seed=0, metric='cosine')
_, d_cos, _ = idx_cos.query(q, k=10, window_size=200, probes=5, fan_probes=16)
assert (d_cos >= -1e-5).all() and (d_cos <= 1 + 1e-5).all(), \
    f"cosine distances out of [0,1]: min={d_cos.min():.4f} max={d_cos.max():.4f}"

print("Metric alias testing: OK")
