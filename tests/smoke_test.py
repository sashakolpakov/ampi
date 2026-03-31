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
        # get_U_drift: (d, F) float32 Oja sketch for each cluster
        _U = cpp.get_U_drift(c)
        assert _U.shape == (cpp.d, cpp.F), \
            f"get_U_drift({c}) wrong shape: {_U.shape}"
        assert _U.dtype == np.float32, \
            f"get_U_drift({c}) wrong dtype: {_U.dtype}"
        for f in range(cpp.F):
            cone = cpp.get_cone(c, f)
            # SortedCone methods accessible
            _ = cone.size()
            _ = cone.all_ids()
            # get_axis_pairs: (projs: float32[n_f], ids: uint32[n_f])
            _projs, _ids = cone.get_axis_pairs(0)
            assert _projs.dtype == np.float32, \
                f"get_axis_pairs projs wrong dtype: {_projs.dtype}"
            assert _ids.dtype == np.uint32, \
                f"get_axis_pairs ids wrong dtype: {_ids.dtype}"
            assert _projs.shape == _ids.shape, \
                "get_axis_pairs projs/ids shape mismatch"

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

    # ── merge / per-cluster-axes fields (gap 4) ───────────────────────────────
    assert isinstance(cpp.merge_qe_ratio,  float), "merge_qe_ratio not float"

    for _c in range(cpp.nlist):
        _ax_c = cpp.get_cluster_axes(_c)
        assert _ax_c.shape == (cpp.F, cpp.d), \
            f"get_cluster_axes({_c}) wrong shape: {_ax_c.shape}"
        assert _ax_c.dtype == np.float32, \
            f"get_cluster_axes({_c}) wrong dtype: {_ax_c.dtype}"

    # set_merge_params must update all three fields atomically
    cpp.set_merge_params(42, 3.14, 0.25)
    assert cpp.merge_interval   == 42,                    "merge_interval not set"
    assert abs(cpp.eps_merge    - 3.14) < 1e-9,          "eps_merge not set"
    assert abs(cpp.merge_qe_ratio - 0.25) < 1e-9,        "merge_qe_ratio not set"
    cpp.set_merge_params(0, 1.0, 0.5)   # restore defaults

    # ── cosine_metric flag is True for cosine indexes (gap 7) ─────────────────
    _idx_cos_compat = AMPIAffineFanIndex(
        data, nlist=5, num_fans=16, seed=0, metric='cosine')
    assert _idx_cos_compat._cpp is not None, "cosine index has no C++ backend"
    assert _idx_cos_compat._cpp.cosine_metric is True, \
        "cosine_metric not True for metric='cosine'"

    print("AMPIIndex accessor compatibility: OK")
else:
    print("AMPIIndex accessor compatibility: SKIPPED (C++ ext not built)")


# ── Binary query_candidates (gap 6) ──────────────────────────────────────────
_q0 = qs[0]
_cands_b = idx_b.query_candidates(_q0, window_size=100)
assert _cands_b.ndim == 1,              "binary query_candidates not 1-D"
assert _cands_b.dtype == np.int32,      "binary query_candidates wrong dtype"
_, _, _ids_b = idx_b.query(_q0, k=10, window_size=100)
assert set(_ids_b.tolist()).issubset(set(_cands_b.tolist())), \
    "binary query_candidates not a superset of query top-k"
print("Binary query_candidates: OK")


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


# ── C++ Oja sketch: update_drift_and_check ────────────────────────────────────
if _HAS_CPP:
    from ampi._ampi_ext import update_drift_and_check

    _rng2        = np.random.default_rng(1)
    _D2, _F2     = 32, 16
    _BETA2, _TH2 = 0.01, 15.0

    def _py_oja(U, axes, v, beta, theta_deg):
        """Python reference Oja update; mirrors C++ update_drift_and_check."""
        v = v.astype(np.float32)
        proj = v @ U                         # (F,)
        U *= (1.0 - beta)
        U += beta * np.outer(v, proj)
        norms = np.linalg.norm(U, axis=0)
        mask = norms > 1e-12
        if mask.any():
            U[:, mask] /= norms[mask]
        u0 = U[:, 0].copy()
        norm0 = float(np.linalg.norm(u0))
        if norm0 < 1e-6:
            return False
        u0 /= norm0
        cos_max = float(np.max(np.abs(axes.astype(np.float64) @ u0)))
        return cos_max < float(np.cos(np.radians(theta_deg)))

    _axes2  = _rng2.standard_normal((_F2, _D2)).astype(np.float32)
    _axes2 /= np.linalg.norm(_axes2, axis=1, keepdims=True)

    # Verify U_drift state tracks between C++ and Python after 1000 steps.
    _Uc = np.zeros((_D2, _F2), dtype=np.float32)
    _Ur = np.zeros((_D2, _F2), dtype=np.float32)
    for _ in range(1000):
        _v2 = _rng2.standard_normal(_D2).astype(np.float32)
        update_drift_and_check(_Uc, _axes2, _v2, _BETA2, _TH2)
        _py_oja(_Ur, _axes2, _v2.copy(), _BETA2, _TH2)
    np.testing.assert_allclose(_Uc, _Ur, atol=1e-5,
        err_msg="U_drift C++ vs Python diverged after 1000 steps")

    # Verify refresh flags match between C++ and Python.
    _Uc2 = np.zeros((_D2, _F2), dtype=np.float32)
    _Ur2 = np.zeros((_D2, _F2), dtype=np.float32)
    _mm = 0
    for _ in range(500):
        _v2 = _rng2.standard_normal(_D2).astype(np.float32)
        cpp_flag = bool(update_drift_and_check(_Uc2, _axes2, _v2, _BETA2, _TH2))
        py_flag  = _py_oja(_Ur2, _axes2, _v2.copy(), _BETA2, _TH2)
        if cpp_flag != py_flag:
            _mm += 1
    assert _mm == 0, f"{_mm} refresh-flag mismatches C++ vs Python"

    print("C++ Oja sketch (update_drift_and_check): OK")
else:
    print("C++ Oja sketch: SKIPPED (C++ ext not built)")


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


# ── AFanTuner smoke (gap 5) ───────────────────────────────────────────────────
# Verifies that AFanTuner constructs without error, returns all expected keys,
# and that the tuned index achieves reasonable recall on tiny data.

from ampi import AFanTuner

_rng_t   = np.random.default_rng(7)
_n_t, _d_t = 2_000, 16
_data_t  = _rng_t.standard_normal((_n_t, _d_t)).astype("float32")
_qs_t    = _rng_t.standard_normal((20, _d_t)).astype("float32")

_flat_t  = faiss.IndexFlatL2(_d_t)
_flat_t.add(_data_t)
_, _gt_t = _flat_t.search(_qs_t, 10)

_tuner  = AFanTuner(_data_t, _qs_t, _gt_t, n_bo_iter=4)
_result = _tuner.tune(verbose=False)

assert set(_result.keys()) >= {"index", "nlist", "alpha", "K", "F", "suggestions"}, \
    f"tune() result missing keys: {_result.keys()}"
assert isinstance(_result["index"], AMPIAffineFanIndex), \
    "tune() 'index' not AMPIAffineFanIndex"
assert isinstance(_result["suggestions"], list), \
    "tune() 'suggestions' not a list"
assert _result["nlist"] >= 1,   "nlist < 1"
assert _result["F"]     >= 1,   "F < 1"
assert _result["K"]     in (1, 2, 3), f"K={_result['K']} not in {{1,2,3}}"

# Tuned index must achieve at least 0.5 recall on the same queries
_tuned_idx = _result["index"]
_found_t   = [_tuned_idx.query(_q, k=10, window_size=50,
                                probes=min(5, _tuned_idx.nlist),
                                fan_probes=min(8, _tuned_idx.F))[2] for _q in _qs_t]
_rec_t = sum(
    len(set(_gt_t[i].tolist()) & set(_found_t[i].tolist()))
    for i in range(len(_qs_t))
) / (len(_qs_t) * 10)
assert _rec_t >= 0.50, f"AFanTuner index recall too low: {_rec_t:.3f}"

print("AFanTuner smoke: OK")


# ── mmap-backed data buffer ───────────────────────────────────────────────────
# Verifies both the C++ mmap path (data_path= kwarg, _HAS_EXT=True) and the
# Python fallback memmap path (_HAS_EXT=False is hard to force, so we verify
# the file and dtype directly).

import os, tempfile

with tempfile.TemporaryDirectory() as _mmap_dir:
    _mmap_rng  = np.random.default_rng(99)
    _mmap_data = _mmap_rng.standard_normal((800, 32)).astype(np.float32)
    _mmap_idx  = AMPIAffineFanIndex(_mmap_data, nlist=8, num_fans=4,
                                     seed=0, data_path=_mmap_dir)

    if _HAS_CPP:
        # C++ extension: mmap file is owned by C++; Python _data_buf is np.empty.
        _cpp_file = os.path.join(_mmap_dir, "_cpp_data_buf.dat")
        assert os.path.exists(_cpp_file), "C++ mmap file not created"
        _expected_bytes = _mmap_idx._cpp.capacity * 32 * 4   # capacity * d * sizeof(float)
        _actual_bytes   = os.path.getsize(_cpp_file)
        assert _actual_bytes >= _expected_bytes, \
            f"C++ mmap file too small: {_actual_bytes} < {_expected_bytes}"
        assert not isinstance(_mmap_idx._data_buf, np.memmap), \
            "Python _data_buf should be np.empty when C++ ext is present"
    else:
        # Python fallback: _data_buf should be a memmap.
        _py_file = os.path.join(_mmap_dir, "_data_buf.dat")
        assert os.path.exists(_py_file), "Python mmap file not created"
        assert isinstance(_mmap_idx._data_buf, np.memmap), \
            "Python _data_buf should be np.memmap when data_path is set"

    # Queries must still return correct results regardless of mmap mode.
    _mmap_q   = _mmap_rng.standard_normal(32).astype(np.float32)
    _, _, _mmap_ids = _mmap_idx.query(_mmap_q, k=5, window_size=200,
                                       probes=_mmap_idx.nlist,
                                       fan_probes=_mmap_idx.F)
    assert len(_mmap_ids) == 5, f"mmap query returned wrong count: {len(_mmap_ids)}"

    # Streaming adds must work (may trigger C++ _grow_buffers → mmap remap).
    for _ in range(20):
        _mmap_idx.add(_mmap_rng.standard_normal(32).astype(np.float32))
    assert _mmap_idx.n == len(_mmap_data) + 20, \
        f"n wrong after mmap adds: {_mmap_idx.n}"

print("mmap-backed data buffer: OK")


# ── streaming_build correctness ───────────────────────────────────────────────
# Verifies that streaming_build produces a fully functional index:
#   - correct metadata (n, d, F, nlist)
#   - mmap file written at the expected path and size
#   - spike added post-build is the exact NN
#   - deleting the spike hides it from subsequent queries
#   - recall@10 is reasonable vs brute-force GT

import os, tempfile
from ampi.streaming import streaming_build

if _HAS_CPP:
    with tempfile.TemporaryDirectory() as _sb_dir:
        _sb_rng  = np.random.default_rng(55)
        _sb_n, _sb_d = 2000, 32
        _sb_data = _sb_rng.standard_normal((_sb_n, _sb_d)).astype(np.float32)
        _sb_path = os.path.join(_sb_dir, 'stream')

        _sb_idx = streaming_build(
            lambda s, e: _sb_data[s:e],
            n=_sb_n, d=_sb_d, nlist=16, num_fans=8,
            cone_top_k=1, seed=0, metric='l2',
            data_path=_sb_path,
        )

        # Structural checks
        assert _sb_idx.n     == _sb_n,  f"streaming n: {_sb_idx.n} != {_sb_n}"
        assert _sb_idx.nlist == 16,     f"streaming nlist: {_sb_idx.nlist}"
        assert _sb_idx.F     == 8,      f"streaming F: {_sb_idx.F}"
        assert _sb_idx.d     == _sb_d,  f"streaming d: {_sb_idx.d}"
        assert _sb_idx.metric == 'l2',  f"streaming metric: {_sb_idx.metric}"
        assert _sb_idx._cpp  is not None, "streaming_build: no C++ index"

        # mmap file must exist and be exactly n * d * 4 bytes
        _cpp_f = os.path.join(_sb_path, '_cpp_data_buf.dat')
        assert os.path.exists(_cpp_f), "streaming: _cpp_data_buf.dat missing"
        assert os.path.getsize(_cpp_f) == _sb_n * _sb_d * 4, \
            f"streaming: mmap size {os.path.getsize(_cpp_f)} != {_sb_n * _sb_d * 4}"

        # Exact NN: spike added after build is the trivially unique NN
        _sb_spike = np.zeros(_sb_d, dtype=np.float32)
        _sb_spike[0] = 1e4
        _sb_gid = _sb_idx.add(_sb_spike)
        _, _, _sb_ids = _sb_idx.query(_sb_spike, k=1, window_size=200,
                                       probes=_sb_idx.nlist, fan_probes=_sb_idx.F)
        assert _sb_gid in _sb_ids.tolist(), "streaming: spike not found as NN"

        # Delete hides the spike
        _sb_idx.delete(_sb_gid)
        _, _, _sb_after = _sb_idx.query(_sb_spike, k=5, window_size=200,
                                         probes=_sb_idx.nlist, fan_probes=_sb_idx.F)
        assert _sb_gid not in _sb_after.tolist(), "streaming: deleted spike leaked"

        # Recall@10 vs FAISS brute-force
        _sb_qs  = _sb_rng.standard_normal((30, _sb_d)).astype(np.float32)
        _sb_fl  = faiss.IndexFlatL2(_sb_d)
        _sb_fl.add(_sb_data)
        _, _sb_gt = _sb_fl.search(_sb_qs, 10)
        _sb_found = [_sb_idx.query(q, k=10, window_size=200,
                                    probes=_sb_idx.nlist, fan_probes=_sb_idx.F)[2]
                     for q in _sb_qs]
        _sb_rec = recall10(_sb_gt, _sb_found)
        assert _sb_rec >= 0.70, f"streaming recall@10 = {_sb_rec:.3f} < 0.70"

    print("streaming_build correctness: OK")
else:
    print("streaming_build correctness: SKIPPED (C++ ext not built)")
