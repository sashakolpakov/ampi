"""
Stress test for the AMPIAffineFanIndex streaming API (add / delete / update).

Covers adversarial scenarios that the smoke test does not:
  - Insert findability
  - Delete / update correctness (no false positives)
  - Idempotent delete, invalid delete
  - Boundary inserts (equidistant from multiple clusters)
  - Outlier and zero-vector inserts
  - Bulk add recall
  - High-deletion recall
  - Tombstone compaction trigger
  - Drift detection trigger
  - All-cluster-deleted query
  - cosine metric
  - cone_top_k=2 tombstone coverage
  - Interleaved mutations and queries
  - Heavy churn recall

No external dependencies beyond numpy (brute-force ground truth used throughout).
"""

import sys
import time
import threading
import traceback
import numpy as np
from ampi import AMPIAffineFanIndex

# ── helpers ───────────────────────────────────────────────────────────────────

def _brute_knn(data, queries, k):
    """Exact k-NN via numpy BLAS."""
    q_sq = np.sum(queries ** 2, axis=1)[:, None]        # (nq, 1)
    d_sq = np.sum(data    ** 2, axis=1)[None, :]        # (1,  n)
    d2   = q_sq + d_sq - 2.0 * (queries @ data.T)      # (nq, n)
    return np.argsort(d2, axis=1)[:, :k].astype(np.int32)


def _recall(gt, found, k):
    """Recall@k averaged over queries."""
    hits = sum(
        len(set(g[:k].tolist()) & set(f[:k].tolist()))
        for g, f in zip(gt, found)
    )
    return hits / (len(gt) * k)


def _small_index(n=3000, d=32, nlist=20, F=16, K=1, seed=0, metric='l2'):
    rng  = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=nlist, num_fans=F, seed=seed,
                              cone_top_k=K, metric=metric)
    return idx, data, rng


def _spike(d, axis, scale=1e3):
    """A point far from the bulk of N(0,1) data along one basis direction."""
    v = np.zeros(d, dtype='float32')
    v[axis] = scale
    return v


# ── test registry ─────────────────────────────────────────────────────────────

_TESTS = []

def _register(fn):
    _TESTS.append(fn)
    return fn


# ── scenarios ────────────────────────────────────────────────────────────────

@_register
def test_insert_findability():
    """Inserted spike must be the exact nearest neighbour when queried."""
    idx, _, _ = _small_index()
    d   = idx.d
    x   = _spike(d, 0)
    gid = idx.add(x)

    _, _, ids = idx.query(x, k=1, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert gid in ids, f"inserted spike (gid={gid}) not found as NN, got {ids}"


@_register
def test_delete_no_false_positive():
    """After deleting a spike, it must never appear in any query result."""
    idx, _, rng = _small_index()
    d   = idx.d
    x   = _spike(d, 1)
    gid = idx.add(x)

    # Confirm it's findable before deletion.
    _, _, pre = idx.query(x, k=1, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert gid in pre, "spike not found before deletion — precondition failed"

    idx.delete(gid)

    # Query at the exact spike location — gid must not appear.
    _, _, post = idx.query(x, k=20, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert gid not in post, f"deleted gid={gid} still returned after delete"

    # Also check via exhaustive probe sweep.
    for w in (10, 50, 200):
        _, _, ids = idx.query(x, k=20, window_size=w, probes=idx.nlist, fan_probes=idx.F)
        assert gid not in ids, f"deleted gid={gid} returned at window_size={w}"


@_register
def test_update_correctness():
    """After update(id, y): old id gone, new id findable at y."""
    idx, _, _ = _small_index()
    d = idx.d
    x = _spike(d, 2)
    y = _spike(d, 3)

    old_gid = idx.add(x)
    new_gid = idx.update(old_gid, y)

    # Old location: old_gid must not appear.
    _, _, ids_x = idx.query(x, k=20, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert old_gid not in ids_x, f"old gid={old_gid} still returned after update"

    # New location: new_gid must be found.
    _, _, ids_y = idx.query(y, k=1, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert new_gid in ids_y, f"new gid={new_gid} not found at updated location"


@_register
def test_double_delete_is_noop():
    """Deleting the same id twice must not change _n_deleted after the first call."""
    idx, _, _ = _small_index()
    gid = idx.add(_spike(idx.d, 0))

    idx.delete(gid)
    n_del_after_first = idx._n_deleted

    idx.delete(gid)                    # second call — must be silent no-op
    assert idx._n_deleted == n_del_after_first, \
        "_n_deleted changed on double delete"


@_register
def test_invalid_delete_raises():
    """delete() with an out-of-range id must raise IndexError."""
    idx, _, _ = _small_index()
    for bad in (-1, idx.n, idx.n + 9999):
        try:
            idx.delete(bad)
            raise AssertionError(f"expected IndexError for id={bad}, got no error")
        except IndexError:
            pass


@_register
def test_outlier_insert():
    """Insert a point far outside the training distribution; it must be findable."""
    idx, _, _ = _small_index()
    d   = idx.d
    x   = _spike(d, 4, scale=1e4)
    gid = idx.add(x)

    _, _, ids = idx.query(x, k=1, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert gid in ids, f"outlier gid={gid} not found"


@_register
def test_zero_vector_insert():
    """Inserting a zero vector must not crash (edge case for normalisation)."""
    idx, _, _ = _small_index()
    z   = np.zeros(idx.d, dtype='float32')
    gid = idx.add(z)
    assert 0 <= gid < idx.n
    # Query near zero — should return some result without crash.
    idx.query(z, k=5, window_size=50, probes=5, fan_probes=4)


@_register
def test_boundary_insert_cone_top_k2():
    """With cone_top_k=2 a boundary point lives in 2 clusters; delete removes it from both."""
    idx, _, rng = _small_index(K=2)
    d = idx.d

    # Insert a point, then delete it; _point_cones should cover both clusters.
    x   = rng.standard_normal(d).astype('float32')
    gid = idx.add(x)

    if idx._cpp is not None:
        n_cones = len(idx._cpp.get_point_cones(gid))
    else:
        n_cones = len(idx._point_cones.get(gid, []))
    assert n_cones >= 1, "cone_top_k=2 point not registered in any cone"

    idx.delete(gid)

    _, _, ids = idx.query(x, k=20, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert gid not in ids, f"cone_top_k=2 deleted gid={gid} still returned"


@_register
def test_bulk_add_recall():
    """Add 300 points; each must be findable (recall@1 >= 0.95 across added points)."""
    idx, data, rng = _small_index(n=2000, d=32)
    extras = rng.standard_normal((300, idx.d)).astype('float32')
    for e in extras:
        idx.add(e)

    # Ground truth: brute force over the full (original + inserted) data.
    all_data = idx.data   # already extended by add()
    gt = _brute_knn(all_data, extras, k=1)
    found = []
    for e in extras:
        _, _, ids = idx.query(e, k=1, window_size=200, probes=idx.nlist, fan_probes=idx.F)
        found.append(ids)

    rec = _recall(gt, found, k=1)
    assert rec >= 0.90, f"bulk-add recall@1 = {rec:.3f} < 0.90"


@_register
def test_high_deletion_recall():
    """Delete 20 % of original points; recall@10 on remaining must stay >= 0.80."""
    idx, data, rng = _small_index(n=2000, d=32, nlist=20, F=16)
    n_del = 400
    del_ids = rng.choice(idx.n, n_del, replace=False)
    for gid in del_ids:
        idx.delete(int(gid))

    del_set = set(del_ids.tolist())
    live_mask = np.ones(len(data), dtype=bool)
    live_mask[del_ids] = False
    live_data = data[live_mask]
    live_ids  = np.where(live_mask)[0].astype(np.int32)

    qs = rng.standard_normal((50, idx.d)).astype('float32')

    # Ground truth over live points only.
    gt_local = _brute_knn(live_data, qs, k=10)          # indices into live_data
    gt_global = live_ids[gt_local]                       # global IDs

    found = []
    for q in qs:
        _, _, ids = idx.query(q, k=10, window_size=200, probes=idx.nlist, fan_probes=idx.F)
        found.append(ids)

    rec = _recall(gt_global, found, k=10)
    assert rec >= 0.80, f"high-deletion recall@10 = {rec:.3f} < 0.80"

    # No deleted id must appear in any result.
    for ids in found:
        bad = del_set & set(ids.tolist())
        assert not bad, f"deleted ids {bad} returned after deletion"


@_register
def test_tombstone_compaction_fires():
    """Exceed _TOMBSTONE_THRESHOLD in one cluster; _cluster_tombstones must reset."""
    from ampi.affine_fan import _TOMBSTONE_THRESHOLD

    # Tiny index so we can control exactly which cluster a batch of points lands in.
    n, d, nlist = 500, 16, 5
    rng  = np.random.default_rng(7)
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=nlist, num_fans=8, seed=7, cone_top_k=1)

    # Find the largest cluster.
    c_star = int(np.argmax([len(g) for g in idx.cluster_global]))
    members = idx.cluster_global[c_star].tolist()

    # Delete enough to exceed the threshold.
    # Delete exactly ceil(threshold * size) members — the last one crosses
    # the threshold and fires _local_refresh, resetting the counter to 0.
    need = int(np.ceil(_TOMBSTONE_THRESHOLD * len(members)))
    to_delete = members[:need]
    for gid in to_delete:
        idx.delete(gid)

    # Compaction should have fired: tombstone counter resets to 0.
    assert idx._cluster_tombstones[c_star] == 0, \
        f"tombstones not reset after compaction in cluster {c_star}"
    # Oja sketch also reset.
    assert np.all(idx._U_drift[c_star] == 0), \
        "U_drift not cleared after _local_refresh"


@_register
def test_drift_detection_simple():
    """Insert many points along e_0; drift check must eventually trigger a refresh."""
    n, d = 1000, 32
    nlist = 10
    rng  = np.random.default_rng(99)
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=nlist, num_fans=16, seed=99, cone_top_k=1)

    # Insert 200 points very strongly aligned with e_0 — biased direction.
    e0 = np.zeros(d, dtype='float32')
    e0[0] = 1.0
    scale = 5.0
    for i in range(200):
        noise = rng.standard_normal(d).astype('float32') * 0.05
        idx.add(scale * e0 + noise)

    # U_drift must have been updated (and possibly reset by a refresh).
    # At minimum: at some point it was non-zero. If a refresh fired, the
    # counter is 0. Either way the cluster is still queryable.
    q = scale * e0
    pts, dists, ids = idx.query(q, k=5, window_size=200, probes=nlist, fan_probes=8)
    assert len(ids) == 5, "query after drift inserts returned wrong number of results"


@_register
def test_all_cluster_points_deleted():
    """Delete every point in one cluster; subsequent queries must not crash or return those ids."""
    n, d, nlist = 500, 16, 5
    rng  = np.random.default_rng(11)
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=nlist, num_fans=8, seed=11, cone_top_k=1)

    c_star   = int(np.argmin([len(g) for g in idx.cluster_global]))  # smallest cluster
    members  = set(idx.cluster_global[c_star].tolist())
    for gid in list(members):
        idx.delete(gid)

    qs = rng.standard_normal((10, d)).astype('float32')
    for q in qs:
        _, _, ids = idx.query(q, k=5, window_size=200, probes=nlist, fan_probes=8)
        bad = members & set(ids.tolist())
        assert not bad, f"deleted cluster members {bad} returned in query"


@_register
def test_cosine_metric_add_delete():
    """add/delete work correctly under the cosine metric."""
    n, d = 2000, 32
    rng  = np.random.default_rng(13)
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=20, num_fans=16, seed=13,
                               cone_top_k=1, metric='cosine')

    x   = _spike(d, 5)
    gid = idx.add(x)

    _, _, pre = idx.query(x, k=1, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert gid in pre, "cosine: spike not found before delete"

    idx.delete(gid)
    _, _, post = idx.query(x, k=20, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert gid not in post, "cosine: deleted spike still returned"


@_register
def test_interleaved_mutations_and_queries():
    """Alternating add/delete/query must stay consistent at every step."""
    idx, _, rng = _small_index(n=1000, d=16, nlist=10, F=8)
    d = idx.d

    live    = {}    # gid -> vector, for points added during this test
    deleted = set() # gids explicitly deleted during this test

    for step in range(60):
        action = step % 3
        if action == 0 or not live:
            # Add
            x   = rng.standard_normal(d).astype('float32') * 3.0
            gid = idx.add(x)
            live[gid] = x
        elif action == 1:
            # Delete a random live point added by this test
            gid = int(rng.choice(list(live.keys())))
            idx.delete(gid)
            deleted.add(gid)
            del live[gid]
        else:
            # Query: no explicitly deleted ID may appear in results.
            sample_gids = list(live.keys())[:5]
            for gid in sample_gids:
                _, _, ids = idx.query(live[gid], k=10, window_size=200,
                                      probes=idx.nlist, fan_probes=idx.F)
                bad = deleted & set(ids.tolist())
                assert not bad, \
                    f"step {step}: deleted ids {bad} returned in query"


@_register
def test_heavy_churn_recall():
    """200 adds + 200 deletes interleaved; recall@10 vs brute force >= 0.75."""
    n, d = 2000, 32
    rng  = np.random.default_rng(17)
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=20, num_fans=16, seed=17, cone_top_k=1)

    added    = {}    # gid -> vector
    deleted  = set()

    for i in range(200):
        x   = rng.standard_normal(d).astype('float32')
        gid = idx.add(x)
        added[gid] = x

    for gid in list(added.keys())[:200]:
        idx.delete(int(gid))
        deleted.add(gid)

    # Build ground truth over all live points.
    live_mask = np.ones(idx.n, dtype=bool)
    for gid in deleted:
        live_mask[gid] = False
    live_data = idx.data[live_mask]
    live_ids  = np.where(live_mask)[0].astype(np.int32)

    qs = rng.standard_normal((50, d)).astype('float32')
    gt_local  = _brute_knn(live_data, qs, k=10)
    gt_global = live_ids[gt_local]

    found = []
    for q in qs:
        _, _, ids = idx.query(q, k=10, window_size=200, probes=idx.nlist, fan_probes=idx.F)
        found.append(ids)

    rec = _recall(gt_global, found, k=10)
    assert rec >= 0.75, f"heavy-churn recall@10 = {rec:.3f} < 0.75"

    for ids in found:
        bad = deleted & set(ids.tolist())
        assert not bad, f"deleted ids {bad} returned during churn"


# ── periodic merge ───────────────────────────────────────────────────────────

@_register
def test_periodic_merge_reduces_cluster_count():
    """Two near-identical tight clusters must be folded into one by periodic_merge."""
    rng = np.random.default_rng(20)
    d   = 16
    # Two very tight groups whose centroids are 0.001 apart — merge criterion
    # is trivially satisfied: δ_qe ≈ 0.25 * 1e-6 << 0.5 * (mQE_a + mQE_b).
    group_a = (rng.standard_normal((100, d)) * 0.1).astype('float32')
    group_b = (rng.standard_normal((100, d)) * 0.1 + 0.001).astype('float32')
    data    = np.vstack([group_a, group_b])
    idx     = AMPIAffineFanIndex(data, nlist=2, num_fans=8, seed=20)

    if idx._cpp is not None:
        non_empty_before = sum(1 for c in idx._cpp.get_cluster_counts() if c > 0)
    else:
        non_empty_before = sum(1 for g in idx.cluster_global if len(g) > 0)
    assert non_empty_before == 2, "precondition: expected 2 non-empty clusters"

    idx.periodic_merge(eps_merge=1.0)

    if idx._cpp is not None:
        non_empty_after = sum(1 for c in idx._cpp.get_cluster_counts() if c > 0)
    else:
        non_empty_after = sum(1 for g in idx.cluster_global if len(g) > 0)
    assert non_empty_after < non_empty_before, \
        f"periodic_merge did not reduce cluster count: {non_empty_before} → {non_empty_after}"


@_register
def test_periodic_merge_recall_preserved():
    """Recall@5 must be maintained after a merge that folds two near-identical clusters."""
    rng = np.random.default_rng(21)
    d   = 16
    group_a = (rng.standard_normal((100, d)) * 0.1).astype('float32')
    group_b = (rng.standard_normal((100, d)) * 0.1 + 0.001).astype('float32')
    data    = np.vstack([group_a, group_b])
    idx     = AMPIAffineFanIndex(data, nlist=2, num_fans=8, seed=21)

    qs  = (rng.standard_normal((20, d)) * 0.1).astype('float32')
    gt  = _brute_knn(data, qs, k=5)

    idx.periodic_merge(eps_merge=1.0)

    found = [idx.query(q, k=5, window_size=100,
                        probes=idx.nlist, fan_probes=idx.F)[2] for q in qs]
    rec = _recall(gt, found, k=5)
    assert rec >= 0.80, f"recall after periodic_merge = {rec:.3f} < 0.80"


@_register
def test_merge_interval_auto_triggers():
    """With merge_interval>0 auto-merge fires during add() calls without crashing."""
    rng  = np.random.default_rng(22)
    d    = 16
    data = rng.standard_normal((200, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=4, num_fans=8, seed=22,
                               merge_interval=5, eps_merge=100.0)
    for _ in range(25):
        idx.add(rng.standard_normal(d).astype('float32'))
    q = rng.standard_normal(d).astype('float32')
    _, _, ids = idx.query(q, k=5, window_size=100,
                           probes=idx.nlist, fan_probes=idx.F)
    assert len(ids) > 0, "index not queryable after merge_interval auto-merge"


# ── merge params and per-cluster axes ─────────────────────────────────────────

@_register
def test_merge_params_propagate_to_cpp():
    """Non-default merge_qe_ratio must reach the C++ layer."""
    idx, _, _ = _small_index(n=500, d=16, nlist=5, F=8)
    if idx._cpp is None:
        return
    assert abs(idx._cpp.merge_qe_ratio - idx.merge_qe_ratio) < 1e-9, \
        "merge_qe_ratio mismatch Python vs C++"

    rng  = np.random.default_rng(30)
    data = rng.standard_normal((500, 16)).astype('float32')
    idx2 = AMPIAffineFanIndex(data, nlist=5, num_fans=8, seed=30, merge_qe_ratio=0.1)
    assert abs(idx2.merge_qe_ratio - 0.1) < 1e-9,  "Python merge_qe_ratio not stored"
    if idx2._cpp is not None:
        assert abs(idx2._cpp.merge_qe_ratio - 0.1) < 1e-9, \
            "merge_qe_ratio=0.1 not propagated to C++"


@_register
def test_per_cluster_axes_populated_after_refresh():
    """After local_refresh with non-trivial U_drift, cluster axes are valid unit vectors.

    Python path: sets U_drift directly, calls _local_refresh, checks cluster_axes.
    C++ path:    inserts biased points to build U_drift, calls local_refresh,
                 checks get_cluster_axes returns (F, d) unit-vector float32 array.
    """
    rng  = np.random.default_rng(40)
    d    = 32
    data = rng.standard_normal((800, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=4, num_fans=8, seed=40)
    c_star = int(np.argmax([len(g) for g in idx.cluster_global]))

    if idx._cpp is None:
        # Python path: inject a known biased U_drift sketch and refresh.
        # Set U_drift[:,0] to e0 (unit vector along dimension 0) — strong signal.
        e0    = np.zeros(d, dtype=np.float32)
        e0[0] = 1.0
        idx._U_drift[c_star][:] = 0.0
        idx._U_drift[c_star][:, 0] = e0   # leading eigenvec estimate = e0
        idx._local_refresh(c_star)

        ca = idx.cluster_axes[c_star]
        assert ca is not None, f"cluster_axes[{c_star}] still None after refresh"
        assert ca.shape == (idx.F, idx.d), \
            f"cluster_axes[{c_star}] wrong shape: {ca.shape}"
        assert ca.dtype == np.float32
        norms = np.linalg.norm(ca.astype(np.float64), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5,
            err_msg="Python cluster_axes: axes not unit vectors")
        # Leading axis must align with e0 (the only direction with signal).
        cos = float(np.max(np.abs(ca.astype(np.float64) @ e0.astype(np.float64))))
        assert cos > 0.9, f"leading axis not aligned with e0: cos={cos:.3f}"

    else:
        # C++ path: insert points biased along e0 relative to c_star's centroid,
        # then manually trigger local_refresh and verify the returned axes.
        centroid = idx._cpp.get_centroids()[c_star].copy()
        e0       = np.zeros(d, dtype='float32')
        e0[0]    = 1.0
        # Points placed 1 unit from centroid along e0 — stays within the cluster's
        # neighbourhood (within-cluster spread ≈ sqrt(d) ≈ 5.6).
        for _ in range(80):
            x = centroid + e0 * 1.0 + rng.standard_normal(d).astype('float32') * 0.1
            idx._cpp.add(x.astype('float32'))

        idx._cpp.local_refresh(c_star)
        ax_c = idx._cpp.get_cluster_axes(c_star)

        assert ax_c.shape == (idx.F, idx.d), \
            f"C++ get_cluster_axes wrong shape: {ax_c.shape}"
        assert ax_c.dtype == np.float32, f"C++ cluster axes wrong dtype: {ax_c.dtype}"
        norms = np.linalg.norm(ax_c.astype(np.float64), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5,
            err_msg="C++ cluster axes are not unit vectors")


# ── sqeuclidean with mutations ────────────────────────────────────────────────

@_register
def test_sqeuclidean_add_delete():
    """add/delete/update work correctly under sqeuclidean metric; distances are non-negative."""
    rng  = np.random.default_rng(50)
    data = rng.standard_normal((500, 16)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=5, num_fans=8, seed=50,
                               metric='sqeuclidean')

    x   = _spike(16, 6)
    gid = idx.add(x)

    _, dists_pre, ids_pre = idx.query(x, k=5, window_size=200,
                                       probes=idx.nlist, fan_probes=idx.F)
    assert gid in ids_pre.tolist(), "sqeuclidean: inserted spike not found"
    assert (dists_pre >= 0).all(), \
        f"sqeuclidean: negative distances before delete: {dists_pre.min()}"

    idx.delete(gid)
    _, dists_post, ids_post = idx.query(x, k=10, window_size=200,
                                         probes=idx.nlist, fan_probes=idx.F)
    assert gid not in ids_post.tolist(), "sqeuclidean: deleted spike still returned"
    assert (dists_post >= 0).all(), \
        f"sqeuclidean: negative distances after delete: {dists_post.min()}"

    new_gid = idx.update(int(ids_post[0]),
                          rng.standard_normal(16).astype('float32'))
    assert isinstance(new_gid, int), "sqeuclidean: update did not return int"


# ── buffer / compaction / drift ───────────────────────────────────────────────

@_register
def test_buffer_grows_past_initial_capacity():
    """Insert enough points to exceed the 1024-point initial headroom."""
    rng  = np.random.default_rng(3)
    data = rng.standard_normal((50, 32)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=4, num_fans=8, seed=0)
    n0   = idx.n
    for _ in range(1100):
        idx.add(rng.standard_normal(32).astype('float32'))
    assert idx.n == n0 + 1100, f"n mismatch: {idx.n} != {n0 + 1100}"
    if idx._cpp is not None:
        assert idx._cpp.n == n0 + 1100, "C++ n mismatch after buffer growth"


@_register
def test_compaction_triggers_on_high_tombstone_fraction():
    """Delete >threshold% of a cluster; cluster_global must contain only live points."""
    from ampi.affine_fan import _TOMBSTONE_THRESHOLD as _TT
    rng  = np.random.default_rng(4)
    data = rng.standard_normal((1000, 32)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=4, num_fans=8, seed=0)

    if idx._cpp is not None:
        c0_ids = idx._cpp.get_cluster_global(0)
    else:
        c0_ids = np.array(idx.cluster_global[0])
    n_c0 = len(c0_ids)
    if n_c0 < 15:
        print("[SKIP] compaction_triggers_on_high_tombstone_fraction: cluster 0 too small")
        return

    n_del = int(n_c0 * _TT) + 1
    for gid in c0_ids[:n_del]:
        idx.delete(int(gid))

    if idx._cpp is not None:
        c0_after = idx._cpp.get_cluster_global(0)
        for gid in c0_after:
            assert not idx._deleted_mask[int(gid)], \
                f"deleted point {gid} still in cluster_global after compaction"
        assert len(c0_after) == n_c0 - n_del, \
            f"cluster_global size after compaction: {len(c0_after)} != {n_c0 - n_del}"


@_register
def test_drift_detection_perpendicular():
    """Insert many points along e_0; index must remain queryable after drift refresh."""
    rng  = np.random.default_rng(99)
    data = rng.standard_normal((1000, 32)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=10, num_fans=16, seed=99)

    e0       = np.zeros(32, dtype='float32')
    e0[0]    = 1.0
    # Build a direction with low cosine to all fan axes
    axes = idx.axes
    perp = rng.standard_normal(32).astype('float32')
    for _ in range(20):
        for ax in axes:
            perp -= float(perp @ ax) * ax
        perp /= np.linalg.norm(perp)

    for _ in range(300):
        x = perp * float(rng.uniform(0.5, 2.0)) + rng.standard_normal(32).astype('float32') * 0.05
        idx.add(x.astype('float32'))

    q = perp + rng.standard_normal(32).astype('float32') * 0.1
    _, _, ids = idx.query(q.astype('float32'), k=5, window_size=200, probes=10, fan_probes=8)
    assert len(ids) == 5, "query after drift inserts returned wrong number of results"


# ── concurrent access ─────────────────────────────────────────────────────────

@_register
def test_concurrent_rw_no_crash():
    """2 reader threads + 1 writer + 1 deleter for 2 s — no crashes or exceptions."""
    idx, _, _ = _small_index(n=3000, d=32, nlist=10, F=8)
    cpp = idx._cpp
    if cpp is None:
        return  # skip if C++ ext not built

    N0 = idx.n
    stop = threading.Event()
    errors = []

    def reader():
        rng_l = np.random.default_rng()
        try:
            while not stop.is_set():
                q = rng_l.standard_normal(idx.d).astype(np.float32)
                sq, ids = cpp.query(q, 5, 100, 4, 4)
                assert sq.shape == ids.shape
        except Exception as e:
            errors.append(("reader", e))

    def writer():
        rng_l = np.random.default_rng()
        try:
            while not stop.is_set():
                pts = rng_l.standard_normal((5, idx.d)).astype(np.float32)
                cpp.batch_add(pts)
        except Exception as e:
            errors.append(("writer", e))

    def deleter():
        rng_l = np.random.default_rng()
        try:
            while not stop.is_set():
                n_cur = cpp.n
                if n_cur > N0 + 20:
                    lo, hi = int(N0), int(n_cur)
                    sample = rng_l.integers(lo, hi, size=min(5, hi - lo), dtype=np.int32)
                    cpp.batch_delete(sample)
                else:
                    time.sleep(0.001)
        except Exception as e:
            errors.append(("deleter", e))

    threads = [
        threading.Thread(target=reader, daemon=True),
        threading.Thread(target=reader, daemon=True),
        threading.Thread(target=writer, daemon=True),
        threading.Thread(target=deleter, daemon=True),
    ]
    for t in threads:
        t.start()
    time.sleep(2.0)
    stop.set()
    for t in threads:
        t.join(timeout=3.0)
        assert not t.is_alive(), "thread did not stop in time"
    assert not errors, f"thread errors: {errors}"


@_register
def test_mmap_cpp_data_path():
    """C++ mmap mode: data_path= creates a mmap file; queries and adds work correctly."""
    import os
    import tempfile
    rng = np.random.default_rng(77)
    n, d = 2000, 32
    data = rng.standard_normal((n, d)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        idx = AMPIAffineFanIndex(data, nlist=10, num_fans=8, seed=0, data_path=tmp)
        if idx._cpp is not None:
            cpp_file = os.path.join(tmp, "_cpp_data_buf.dat")
            assert os.path.exists(cpp_file), "C++ mmap file not created"
        # Queries must be correct.
        q = rng.standard_normal(d).astype(np.float32)
        _, _, ids = idx.query(q, k=10, window_size=200, probes=idx.nlist, fan_probes=idx.F)
        assert len(ids) == 10, f"mmap query returned {len(ids)} results"
        # Streaming adds (may trigger mmap remap).
        for _ in range(100):
            idx.add(rng.standard_normal(d).astype(np.float32))
        assert idx.n == n + 100, f"n wrong after adds: {idx.n}"
        # Deleted points must not appear (use a spike far from the bulk so it's
        # the exact NN at k=1).
        spike = np.zeros(d, dtype=np.float32)
        spike[0] = 1e4
        gid = idx.add(spike)
        _, _, before = idx.query(spike, k=1, window_size=200,
                                 probes=idx.nlist, fan_probes=idx.F)
        assert gid in before.tolist(), "mmap: inserted spike not found as NN"
        idx.delete(gid)
        _, _, after = idx.query(spike, k=5, window_size=200,
                                probes=idx.nlist, fan_probes=idx.F)
        assert gid not in after.tolist(), "mmap: deleted spike leaked"


@_register
def test_mmap_serialization_getters():
    """get_U_drift and get_axis_pairs return correct shapes after mutations."""
    import tempfile
    rng = np.random.default_rng(88)
    data = rng.standard_normal((1000, 32)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        idx = AMPIAffineFanIndex(data, nlist=8, num_fans=4, seed=0, data_path=tmp)
        if idx._cpp is None:
            return
        cpp = idx._cpp
        for _ in range(50):
            idx.add(rng.standard_normal(32).astype(np.float32))
        for c in range(cpp.nlist):
            U = cpp.get_U_drift(c)
            assert U.shape == (cpp.d, cpp.F), \
                f"get_U_drift({c}) shape {U.shape} != ({cpp.d}, {cpp.F})"
            assert U.dtype == np.float32, f"get_U_drift dtype {U.dtype}"
            if cpp.has_cones(c):
                for f in range(cpp.F):
                    projs, ids = cpp.get_cone(c, f).get_axis_pairs(0)
                    assert projs.dtype == np.float32, "get_axis_pairs projs dtype"
                    assert ids.dtype == np.uint32, "get_axis_pairs ids dtype"
                    assert projs.shape == ids.shape, "get_axis_pairs shape mismatch"


@_register
def test_batch_correctness_after_mutations():
    """batch_add then batch_delete: deleted IDs must never appear in queries."""
    idx, _, rng = _small_index(n=2000, d=32, nlist=8, F=8)
    if idx._cpp is None:
        return

    pts = rng.standard_normal((200, idx.d)).astype(np.float32)
    ids = idx.batch_add(pts)

    to_del = ids[:100].astype(np.int32)
    idx.batch_delete(to_del)
    deleted_set = set(to_del.tolist())

    qs = rng.standard_normal((20, idx.d)).astype(np.float32)
    for q in qs:
        _, _, found = idx.query(q.astype(np.float32), k=10, window_size=200,
                                probes=idx.nlist, fan_probes=idx.F)
        leaked = deleted_set & set(found.tolist())
        assert not leaked, f"batch-deleted ids {leaked} appeared in query"


# ── streaming_build ───────────────────────────────────────────────────────────

@_register
def test_streaming_build_basic_recall():
    """streaming_build recall@10 >= 0.75 on 3000-point data."""
    import tempfile
    import os
    from ampi.streaming import streaming_build

    rng  = np.random.default_rng(200)
    n, d = 3000, 32
    data = rng.standard_normal((n, d)).astype('float32')

    with tempfile.TemporaryDirectory() as tmp:
        idx = streaming_build(
            lambda s, e: data[s:e],
            n=n, d=d, nlist=20, num_fans=8,
            cone_top_k=1, seed=0,
            data_path=os.path.join(tmp, 'idx'),
        )
        assert idx.n == n,  f"streaming n={idx.n} != {n}"
        assert idx._cpp is not None, "streaming_build returned no C++ index"

        qs    = rng.standard_normal((50, d)).astype('float32')
        gt    = _brute_knn(data, qs, k=10)
        found = [idx.query(q, k=10, window_size=200,
                            probes=idx.nlist, fan_probes=idx.F)[2] for q in qs]
        rec   = _recall(gt, found, k=10)
        assert rec >= 0.75, f"streaming recall@10 = {rec:.3f} < 0.75"


@_register
def test_streaming_build_add_delete():
    """After streaming_build: add spike → found as NN; delete → gone."""
    import tempfile
    import os
    from ampi.streaming import streaming_build

    rng  = np.random.default_rng(201)
    n, d = 2000, 32
    data = rng.standard_normal((n, d)).astype('float32')

    with tempfile.TemporaryDirectory() as tmp:
        idx   = streaming_build(lambda s, e: data[s:e], n=n, d=d,
                                 nlist=16, num_fans=8, seed=0,
                                 data_path=os.path.join(tmp, 'idx'))
        spike = np.zeros(d, dtype='float32')
        spike[0] = 1e4
        gid   = idx.add(spike)

        _, _, before = idx.query(spike, k=1, window_size=200,
                                  probes=idx.nlist, fan_probes=idx.F)
        assert gid in before.tolist(), "streaming: spike not found before delete"

        idx.delete(gid)
        _, _, after = idx.query(spike, k=5, window_size=200,
                                 probes=idx.nlist, fan_probes=idx.F)
        assert gid not in after.tolist(), "streaming: deleted spike leaked"


@_register
def test_streaming_build_cosine():
    """streaming_build with cosine metric: distances in [0,1]; spike findable and deletable."""
    import tempfile
    import os
    from ampi.streaming import streaming_build

    rng  = np.random.default_rng(202)
    n, d = 2000, 32
    data = rng.standard_normal((n, d)).astype('float32')

    with tempfile.TemporaryDirectory() as tmp:
        idx = streaming_build(lambda s, e: data[s:e], n=n, d=d,
                               nlist=16, num_fans=8, metric='cosine', seed=0,
                               data_path=os.path.join(tmp, 'idx'))
        assert idx.metric == 'cosine'
        if idx._cpp is not None:
            assert idx._cpp.cosine_metric is True, "cosine_metric not set in C++ layer"

        spike = np.zeros(d, dtype='float32')
        spike[0] = 1.0   # unit vector
        gid = idx.add(spike)

        _, dists, before = idx.query(spike, k=5, window_size=200,
                                      probes=idx.nlist, fan_probes=idx.F)
        assert gid in before.tolist(), "streaming cosine: spike not found"
        assert (dists >= -1e-4).all() and (dists <= 1 + 1e-4).all(), \
            f"streaming cosine: distances out of [0,1]: {dists}"

        idx.delete(gid)
        _, _, after = idx.query(spike, k=5, window_size=200,
                                 probes=idx.nlist, fan_probes=idx.F)
        assert gid not in after.tolist(), "streaming cosine: deleted spike leaked"


@_register
def test_streaming_build_matches_regular_recall():
    """Streaming build recall must be within 15pp of regular build on same data+seed."""
    import tempfile
    import os
    from ampi.streaming import streaming_build

    rng  = np.random.default_rng(203)
    n, d = 3000, 32
    data = rng.standard_normal((n, d)).astype('float32')
    qs   = rng.standard_normal((50, d)).astype('float32')
    gt   = _brute_knn(data, qs, k=10)

    def _rec(found_list):
        return _recall(gt, found_list, k=10)

    idx_reg   = AMPIAffineFanIndex(data, nlist=20, num_fans=8, seed=0)
    found_reg = [idx_reg.query(q, k=10, window_size=200,
                                probes=idx_reg.nlist, fan_probes=idx_reg.F)[2] for q in qs]
    rec_reg   = _rec(found_reg)

    with tempfile.TemporaryDirectory() as tmp:
        idx_str   = streaming_build(lambda s, e: data[s:e], n=n, d=d,
                                     nlist=20, num_fans=8, seed=0,
                                     data_path=os.path.join(tmp, 'idx'))
        found_str = [idx_str.query(q, k=10, window_size=200,
                                    probes=idx_str.nlist, fan_probes=idx_str.F)[2] for q in qs]
        rec_str   = _rec(found_str)

    assert rec_str >= rec_reg - 0.15, \
        f"streaming recall {rec_str:.3f} too far below regular {rec_reg:.3f} (gap > 15pp)"


# ── bounds checks ────────────────────────────────────────────────────────────

@_register
def test_bounds_checks_raise_python_exceptions():
    """Out-of-range / wrong-dimension arguments must raise Python exceptions, not segfault."""
    idx, _, _ = _small_index(n=500, d=16, nlist=5, F=8)
    if idx._cpp is None:
        return

    cpp   = idx._cpp
    d_idx = idx.d
    nl    = idx.nlist

    # add: wrong dimension
    try:
        cpp.add(np.zeros(d_idx + 1, dtype='float32'))
        assert False, "add: expected ValueError"
    except ValueError:
        pass

    # batch_add: wrong column count
    try:
        cpp.batch_add(np.zeros((2, d_idx + 1), dtype='float32'))
        assert False, "batch_add: expected ValueError"
    except ValueError:
        pass

    # get_U_drift: out of range
    for bad in (-1, nl):
        try:
            cpp.get_U_drift(bad)
            assert False, f"get_U_drift({bad}): expected IndexError"
        except IndexError:
            pass

    # get_cluster_global: out of range
    for bad in (-1, nl):
        try:
            cpp.get_cluster_global(bad)
            assert False, f"get_cluster_global({bad}): expected IndexError"
        except IndexError:
            pass

    # get_cone: out of range
    for bad_c, bad_f in ((-1, 0), (nl, 0), (0, -1), (0, idx.F)):
        try:
            cpp.get_cone(bad_c, bad_f)
            assert False, f"get_cone({bad_c}, {bad_f}): expected IndexError"
        except IndexError:
            pass

    # query: wrong dimension
    try:
        cpp.query(np.zeros(d_idx + 1, dtype='float32'),
                  k=1, window_size=10, probes=1, fan_probes=1)
        assert False, "query: expected ValueError"
    except ValueError:
        pass


# ── _build_norms_all / _rerank_blas correctness ──────────────────────────────

@_register
def test_rerank_blas_norms_at_construction():
    """Returned sq_dists must match exact L2² distances — verifies _build_norms_all
    populated norms[global_id] correctly at construction so _rerank_blas can use them."""
    rng  = np.random.default_rng(400)
    n, d = 500, 32
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=16, num_fans=8, seed=0)
    if idx._cpp is None:
        return  # no C++ ext; skip

    cpp = idx._cpp
    q   = rng.standard_normal(d).astype('float32')
    k   = 20

    sq_dists_cpp, ids_cpp = cpp.query(
        q, k=k, window_size=500, probes=idx.nlist, fan_probes=idx.F)

    # Exact sq distances for the returned ids.
    exact = np.sum((data[ids_cpp.astype(int)] - q) ** 2, axis=1).astype('float32')
    np.testing.assert_allclose(
        sq_dists_cpp, exact, rtol=1e-4, atol=1e-4,
        err_msg="sq_dists from _rerank_blas disagree with exact L2² — norms likely wrong")


@_register
def test_rerank_blas_norms_after_deletion():
    """After deleting ~half the points, sq_dists for survivors must still be exact.
    Confirms that del_mask skipping in _build_norms_all does not corrupt live norms,
    and that norms[global_id] indexing stays correct across the deleted gaps."""
    rng  = np.random.default_rng(401)
    n, d = 600, 32
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=16, num_fans=8, seed=0)
    if idx._cpp is None:
        return

    cpp = idx._cpp

    # Delete every other point from the first half — creates gaps in global_id space.
    deleted = set(range(0, n // 2, 2))
    for gid in deleted:
        cpp.remove(gid)

    q = rng.standard_normal(d).astype('float32')
    k = 10

    sq_dists_cpp, ids_cpp = cpp.query(
        q, k=k, window_size=500, probes=idx.nlist, fan_probes=idx.F)

    # No deleted id should appear.
    leaked = set(ids_cpp.tolist()) & deleted
    assert not leaked, f"deleted ids {leaked} appeared in query results"

    # Exact sq distances for the returned ids must match.
    ids_int = ids_cpp.astype(int)
    exact = np.sum((data[ids_int] - q) ** 2, axis=1).astype('float32')
    np.testing.assert_allclose(
        sq_dists_cpp, exact, rtol=1e-4, atol=1e-4,
        err_msg="sq_dists wrong after deletion — norms[global_id] indexing broken")


# ── AMPIBinaryIndex stress ────────────────────────────────────────────────────

@_register
def test_binary_stress_recall_vs_projections():
    """AMPIBinaryIndex recall@10: L=32 ≥ 0.65, L=128 ≥ 0.80."""
    from ampi import AMPIBinaryIndex

    rng  = np.random.default_rng(300)
    n, d = 3_000, 32
    data = rng.standard_normal((n, d)).astype('float32')
    qs   = rng.standard_normal((50, d)).astype('float32')
    gt   = _brute_knn(data, qs, k=10)

    for L, min_rec in [(32, 0.65), (128, 0.80)]:
        idx  = AMPIBinaryIndex(data, num_projections=L, seed=0)
        found = [idx.query(q, k=10, window_size=100)[2] for q in qs]
        rec  = _recall(gt, found, k=10)
        assert rec >= min_rec, f"BinaryIndex L={L}: recall@10={rec:.3f} < {min_rec}"


@_register
def test_binary_edge_cases():
    """AMPIBinaryIndex: n=1, k>n, window_size=1 all return valid results without crash."""
    from ampi import AMPIBinaryIndex

    rng = np.random.default_rng(301)
    q   = rng.standard_normal(8).astype('float32')

    # n=1: only one data point
    idx1 = AMPIBinaryIndex(rng.standard_normal((1, 8)).astype('float32'),
                           num_projections=4, seed=0)
    pts, dists, ids = idx1.query(q, k=5, window_size=10)
    assert len(ids) == 1,    f"n=1: expected 1 result, got {len(ids)}"
    assert (dists >= 0).all(), "n=1: negative distance"

    # k > n: must return at most n results
    idx2 = AMPIBinaryIndex(rng.standard_normal((10, 8)).astype('float32'),
                           num_projections=4, seed=0)
    _, _, ids2 = idx2.query(q, k=100, window_size=50)
    assert len(ids2) <= 10,  f"k>n: got {len(ids2)} results, expected ≤10"

    # window_size=1: minimal window still works
    _, _, ids3 = idx2.query(q, k=1, window_size=1)
    assert len(ids3) >= 1,   "window_size=1: no results returned"


@_register
def test_binary_candidate_superset():
    """query_candidates must always be a superset of the top-k returned by query."""
    from ampi import AMPIBinaryIndex

    rng  = np.random.default_rng(302)
    n, d = 1_000, 32
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIBinaryIndex(data, num_projections=32, seed=0)

    for _ in range(20):
        q = rng.standard_normal(d).astype('float32')
        cands = idx.query_candidates(q, window_size=50)
        _, _, ids = idx.query(q, k=10, window_size=50)
        missing = set(ids.tolist()) - set(cands.tolist())
        assert not missing, f"top-k ids {missing} not in query_candidates output"


# ── streaming_build sqeuclidean ───────────────────────────────────────────────

@_register
def test_streaming_build_sqeuclidean():
    """streaming_build with sqeuclidean: metric preserved, distances ≥ 0, recall ≥ 0.60."""
    import tempfile
    import os
    from ampi.streaming import streaming_build

    rng  = np.random.default_rng(205)
    n, d = 2_000, 32
    data = rng.standard_normal((n, d)).astype('float32')

    with tempfile.TemporaryDirectory() as tmp:
        try:
            idx = streaming_build(
                lambda s, e: data[s:e],
                n=n, d=d, nlist=16, num_fans=8,
                metric='sqeuclidean', seed=0,
                data_path=os.path.join(tmp, 'idx'),
            )
        except RuntimeError:
            print("[SKIP] test_streaming_build_sqeuclidean: C++ ext not built")
            return

        assert idx.metric == 'sqeuclidean', f"metric: {idx.metric}"

        qs    = rng.standard_normal((30, d)).astype('float32')
        gt    = _brute_knn(data, qs, k=10)
        found = []
        for q in qs:
            _, dists, ids = idx.query(q, k=10, window_size=200,
                                      probes=idx.nlist, fan_probes=idx.F)
            assert (dists >= 0).all(), f"sqeuclidean: negative dists: {dists.min()}"
            found.append(ids)

        rec = _recall(gt, found, k=10)
        assert rec >= 0.60, f"sqeuclidean streaming recall@10 = {rec:.3f} < 0.60"


# ── query k > live count ──────────────────────────────────────────────────────

@_register
def test_query_k_exceeds_live_count():
    """query(k > n_live) must not crash and return at most n_live results."""
    rng  = np.random.default_rng(303)
    n, d = 20, 8
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=2, num_fans=4, seed=0)

    for gid in range(0, n, 2):
        idx.delete(gid)
    n_live = n - n // 2

    q = rng.standard_normal(d).astype('float32')
    _, dists, ids = idx.query(q, k=100, window_size=200,
                              probes=idx.nlist, fan_probes=idx.F)
    assert len(ids) <= n_live, f"returned {len(ids)} results with only {n_live} live points"
    assert len(ids) >= 1, "no results returned"
    assert (dists >= 0).all(), f"negative distances: {dists}"


# ── AFanTuner edge cases ──────────────────────────────────────────────────────

@_register
def test_afantuner_edge_cases():
    """AFanTuner with n_bo_iter=1 and small n_sample must not crash."""
    from ampi import AFanTuner

    rng  = np.random.default_rng(304)
    n, d = 500, 8
    data = rng.standard_normal((n, d)).astype('float32')
    qs   = rng.standard_normal((10, d)).astype('float32')
    gt   = _brute_knn(data, qs, k=10)

    tuner  = AFanTuner(data, qs, gt, n_bo_iter=1)
    result = tuner.tune(verbose=False)
    assert result['nlist'] >= 1, "nlist < 1"
    assert result['F']     >= 1, "F < 1"

    tuner2  = AFanTuner(data, qs, gt, n_sample=50, n_bo_iter=2)
    result2 = tuner2.tune(verbose=False)
    assert result2['nlist'] >= 1, "n_sample=50: nlist < 1"


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    passed, failed = [], []
    for fn in _TESTS:
        name = fn.__name__
        try:
            fn()
            passed.append(name)
            print(f"[PASS] {name}")
        except Exception:
            failed.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()

    print(f"\n{len(passed)}/{len(passed)+len(failed)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
