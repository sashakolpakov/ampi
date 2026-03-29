"""
Smoke test: build small indexes, verify exact NN is found at high recall.
Runs in ~10s on a laptop, no datasets required.
"""
import threading
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

    # ── cone population after from_build ─────────────────────────────────────
    total_live = sum(
        cpp.get_cone(c, f).size()
        for c in range(cpp.nlist) if cpp.has_cones(c)
        for f in range(cpp.F)
    )
    n_live = cpp.n - cpp.n_deleted
    assert total_live >= n_live, \
        f"too few cone entries after from_build: {total_live} < {n_live}"

    # ── batch_add / batch_delete ──────────────────────────────────────────────
    _batch = np.random.default_rng(6).standard_normal((20, cpp.d)).astype(np.float32)
    _bids  = idx_a.batch_add(_batch)
    assert _bids.shape == (20,) and len(set(_bids.tolist())) == 20, \
        "batch_add did not return 20 unique IDs"
    idx_a.batch_delete(_bids)
    _, _, _found = idx_a.query(
        _batch[0], k=5, window_size=200, probes=idx_a.nlist, fan_probes=idx_a.F)
    assert not (set(_bids.tolist()) & set(_found.tolist())), \
        "batch-deleted points appeared in query"

    print("AMPIIndex accessor compatibility: OK")
else:
    print("AMPIIndex accessor compatibility: SKIPPED (C++ ext not built)")


# ── C++ query correctness ─────────────────────────────────────────────────────
# 1. C++ and Python-path must agree when both use full-cluster scan (fan_probes=F).
# 2. query_candidates() must be a superset of query() top-k IDs.

if _HAS_CPP:
    _fp = idx_a.F  # fan_probes == F triggers cluster-level fallback in both paths
    _cpp_ids, _py_ids = [], []
    for _q in qs:
        _, _, _ic = idx_a.query(_q, k=10, window_size=200, probes=5, fan_probes=_fp)
        _, _, _ip = idx_a._py_query(_q, k=10, window_size=200, probes=5, fan_probes=_fp)
        _cpp_ids.append(set(_ic.tolist()))
        _py_ids.append(set(_ip.tolist()))

    _agree = sum(len(a & b) for a, b in zip(_cpp_ids, _py_ids)) / (len(qs) * 10)
    assert _agree >= 0.99, f"C++ vs Python recall agreement too low: {_agree:.3f}"

    for _q in qs[:20]:
        _, _, _ids = idx_a.query(_q, k=10, window_size=200, probes=5, fan_probes=_fp)
        _cands = idx_a.query_candidates(_q, window_size=200, probes=5, fan_probes=_fp)
        _cand_set = set(_cands.tolist())
        for _gid in _ids.tolist():
            assert _gid in _cand_set, \
                f"top-k id {_gid} missing from query_candidates output"

    print("C++ query correctness: OK")
else:
    print("C++ query correctness: SKIPPED (C++ ext not built)")


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


# ── C++ routing: best_clusters / best_fan_cones ───────────────────────────────
if _HAS_CPP:
    from ampi._ampi_ext import best_clusters, best_fan_cones

    _rng1   = np.random.default_rng(0)
    _NLIST1 = 200
    _F1, _D1 = 64, 32

    def _np_best_clusters(centroids, q, probes):
        d2 = np.sum((centroids - q) ** 2, axis=1)
        return np.argsort(d2)[:probes]

    def _np_best_fan_cones(axes, q_c, fan_probes):
        qn = float(np.linalg.norm(q_c))
        if qn < 1e-10:
            return np.arange(min(fan_probes, len(axes)), dtype=np.int32)
        return np.argsort(-np.abs(q_c @ axes.T / qn))[:fan_probes]

    _cent1 = _rng1.standard_normal((_NLIST1, _D1)).astype(np.float32)
    _qs1   = _rng1.standard_normal((100, _D1)).astype(np.float32)
    for _q1 in _qs1:
        assert list(best_clusters(_cent1, _q1, 5)) == list(_np_best_clusters(_cent1, _q1, 5)), \
            "best_clusters mismatch"

    _cent_small = _rng1.standard_normal((10, _D1)).astype(np.float32)
    assert len(best_clusters(_cent_small, _rng1.standard_normal(_D1).astype(np.float32), 999)) == 10, \
        "best_clusters probes not clamped"

    _axes1 = _rng1.standard_normal((_F1, _D1)).astype(np.float32)
    _axes1 /= np.linalg.norm(_axes1, axis=1, keepdims=True)
    for _q1 in _rng1.standard_normal((100, _D1)).astype(np.float32):
        assert list(best_fan_cones(_axes1, _q1, 4)) == list(_np_best_fan_cones(_axes1, _q1, 4)), \
            "best_fan_cones mismatch"

    _out_zero = best_fan_cones(_axes1, np.zeros(_D1, dtype=np.float32), 4)
    assert len(_out_zero) == 4 and _out_zero.dtype == np.int32, \
        "best_fan_cones zero-vector: wrong output"

    print("C++ routing (best_clusters / best_fan_cones): OK")
else:
    print("C++ routing: SKIPPED (C++ ext not built)")


# ── C++ drift EMA: update_drift_and_check ─────────────────────────────────────
if _HAS_CPP:
    from ampi._ampi_ext import update_drift_and_check

    _rng2        = np.random.default_rng(1)
    _D2, _F2     = 32, 16
    _BETA2, _TH2 = 0.01, 15.0

    def _py_drift(sigma_flat, axes, v, beta, theta_deg):
        sig = sigma_flat.reshape(axes.shape[1], axes.shape[1])
        sig = (1.0 - beta) * sig + beta * np.outer(v, v)
        sigma_flat[:] = sig.ravel()
        ev = sig @ axes[0].astype(np.float64)
        for _ in range(5):
            ev = sig @ ev
            norm = float(np.linalg.norm(ev))
            if norm < 1e-12:
                return False
            ev /= norm
        cos_max = float(np.max(np.abs(axes.astype(np.float64) @ ev)))
        return cos_max < float(np.cos(np.radians(theta_deg)))

    _axes2  = _rng2.standard_normal((_F2, _D2)).astype(np.float32)
    _axes2 /= np.linalg.norm(_axes2, axis=1, keepdims=True)

    _sc, _sr = np.zeros(_D2 * _D2, dtype=np.float64), np.zeros(_D2 * _D2, dtype=np.float64)
    for _ in range(1000):
        _v2 = _rng2.standard_normal(_D2)
        update_drift_and_check(_sc, _axes2, _v2, _BETA2, _TH2)
        _py_drift(_sr, _axes2, _v2, _BETA2, _TH2)
    np.testing.assert_allclose(_sc, _sr, atol=1e-10,
        err_msg="sigma_drift C++ vs Python diverged after 1000 steps")

    _sc2, _sr2, _mm = np.zeros(_D2 * _D2, dtype=np.float64), np.zeros(_D2 * _D2, dtype=np.float64), 0
    for _ in range(500):
        _v2 = _rng2.standard_normal(_D2)
        if bool(update_drift_and_check(_sc2, _axes2, _v2, _BETA2, _TH2)) != \
           _py_drift(_sr2, _axes2, _v2, _BETA2, _TH2):
            _mm += 1
    assert _mm == 0, f"{_mm} refresh-flag mismatches C++ vs Python"

    print("C++ drift EMA (update_drift_and_check): OK")
else:
    print("C++ drift EMA: SKIPPED (C++ ext not built)")


# ── local_refresh GIL safety ──────────────────────────────────────────────────
# local_refresh() releases the GIL; it must not deadlock when a Python thread
# calls it while the main thread holds the GIL.

if _HAS_CPP:
    _cpp_r = idx_a._cpp
    _c_r = next(c for c in range(_cpp_r.nlist) if _cpp_r.has_cones(c))
    _errors_r = []

    def _refresh_worker():
        try:
            _cpp_r.local_refresh(_c_r)
        except Exception as e:
            _errors_r.append(e)

    _t = threading.Thread(target=_refresh_worker)
    _t.start()
    _t.join(timeout=5.0)
    assert not _t.is_alive(), "local_refresh timed out — possible GIL deadlock"
    assert not _errors_r, f"local_refresh raised: {_errors_r}"
    print("local_refresh GIL safety: OK")
else:
    print("local_refresh GIL safety: SKIPPED (C++ ext not built)")


# ── add / delete / update API ──────────────────────────────────────────────────
_rng3       = np.random.default_rng(2)
_N3, _D3    = 1000, 32
_data3      = _rng3.standard_normal((_N3, _D3)).astype(np.float32)

# sequential IDs
_idx3 = AMPIAffineFanIndex(_data3, nlist=5, num_fans=8, seed=0)
_n0   = _idx3.n
for _i in range(20):
    _gid = _idx3.add(_rng3.standard_normal(_D3).astype(np.float32))
    assert _gid == _n0 + _i, f"add(): expected id {_n0+_i}, got {_gid}"

# data view reflects inserts
_idx3b = AMPIAffineFanIndex(_data3, nlist=5, num_fans=8, seed=0)
_x_ins = _rng3.standard_normal(_D3).astype(np.float32)
_gid_ins = _idx3b.add(_x_ins)
np.testing.assert_allclose(_idx3b.data[_gid_ins], _x_ins, atol=1e-6,
    err_msg="data view does not reflect insert")

# delete hides point from query
_q3   = _rng3.standard_normal(_D3).astype(np.float32)
_q3  /= np.linalg.norm(_q3)
_gid_del = _idx3b.add((_q3 * 0.01).astype(np.float32))
_, _, _ids_pre = _idx3b.query(_q3, k=10, window_size=200, probes=_idx3b.nlist, fan_probes=_idx3b.F)
assert _gid_del in _ids_pre.tolist(), "inserted point not found before delete"
_idx3b.delete(_gid_del)
_, _, _ids_post = _idx3b.query(_q3, k=10, window_size=200, probes=_idx3b.nlist, fan_probes=_idx3b.F)
assert _gid_del not in _ids_post.tolist(), "deleted point leaked into results"

# delete marks mask
_idx3c = AMPIAffineFanIndex(_data3, nlist=5, num_fans=8, seed=0)
_idx3c.delete(42)
assert _idx3c._deleted_mask[42], "deleted_mask not set after delete"

# update replaces point
_idx3d  = AMPIAffineFanIndex(_data3, nlist=5, num_fans=8, seed=0)
_q_upd  = _rng3.standard_normal(_D3).astype(np.float32)
_q_upd /= np.linalg.norm(_q_upd)
_gid_old = _idx3d.add((_q_upd * 0.01).astype(np.float32))
_idx3d.update(_gid_old, (-_q_upd * 100.0).astype(np.float32))
_, _, _ids_upd = _idx3d.query(_q_upd, k=10, window_size=200,
                               probes=_idx3d.nlist, fan_probes=_idx3d.F)
assert _gid_old not in _ids_upd.tolist(), "old point still in results after update"

print("add / delete / update API: OK")
