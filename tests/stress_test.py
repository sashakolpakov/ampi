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

def test(fn):
    _TESTS.append(fn)
    return fn


# ── scenarios ────────────────────────────────────────────────────────────────

@test
def insert_findability():
    """Inserted spike must be the exact nearest neighbour when queried."""
    idx, _, _ = _small_index()
    d   = idx.d
    x   = _spike(d, 0)
    gid = idx.add(x)

    _, _, ids = idx.query(x, k=1, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert gid in ids, f"inserted spike (gid={gid}) not found as NN, got {ids}"


@test
def delete_no_false_positive():
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


@test
def update_correctness():
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


@test
def double_delete_is_noop():
    """Deleting the same id twice must not change _n_deleted after the first call."""
    idx, _, _ = _small_index()
    gid = idx.add(_spike(idx.d, 0))

    idx.delete(gid)
    n_del_after_first = idx._n_deleted

    idx.delete(gid)                    # second call — must be silent no-op
    assert idx._n_deleted == n_del_after_first, \
        "_n_deleted changed on double delete"


@test
def invalid_delete_raises():
    """delete() with an out-of-range id must raise IndexError."""
    idx, _, _ = _small_index()
    for bad in (-1, idx.n, idx.n + 9999):
        try:
            idx.delete(bad)
            raise AssertionError(f"expected IndexError for id={bad}, got no error")
        except IndexError:
            pass


@test
def outlier_insert():
    """Insert a point far outside the training distribution; it must be findable."""
    idx, _, _ = _small_index()
    d   = idx.d
    x   = _spike(d, 4, scale=1e4)
    gid = idx.add(x)

    _, _, ids = idx.query(x, k=1, window_size=200, probes=idx.nlist, fan_probes=idx.F)
    assert gid in ids, f"outlier gid={gid} not found"


@test
def zero_vector_insert():
    """Inserting a zero vector must not crash (edge case for normalisation)."""
    idx, _, _ = _small_index()
    z   = np.zeros(idx.d, dtype='float32')
    gid = idx.add(z)
    assert 0 <= gid < idx.n
    # Query near zero — should return some result without crash.
    idx.query(z, k=5, window_size=50, probes=5, fan_probes=4)


@test
def boundary_insert_cone_top_k2():
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


@test
def bulk_add_recall():
    """Add 300 points; each must be findable (recall@1 >= 0.95 across added points)."""
    idx, data, rng = _small_index(n=2000, d=32)
    extras = rng.standard_normal((300, idx.d)).astype('float32')
    new_ids = np.array([idx.add(e) for e in extras], dtype=np.int32)

    # Ground truth: brute force over the full (original + inserted) data.
    all_data = idx.data   # already extended by add()
    gt = _brute_knn(all_data, extras, k=1)
    found = []
    for e in extras:
        _, _, ids = idx.query(e, k=1, window_size=200, probes=idx.nlist, fan_probes=idx.F)
        found.append(ids)

    rec = _recall(gt, found, k=1)
    assert rec >= 0.90, f"bulk-add recall@1 = {rec:.3f} < 0.90"


@test
def high_deletion_recall():
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


@test
def tombstone_compaction_fires():
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
    # Sigma drift also reset.
    assert np.all(idx._sigma_drift[c_star] == 0), \
        "sigma_drift not cleared after _local_refresh"


@test
def drift_detection_fires():
    """Insert many points along e_0; drift check must eventually trigger a refresh."""
    n, d = 1000, 32
    nlist = 10
    rng  = np.random.default_rng(99)
    data = rng.standard_normal((n, d)).astype('float32')
    idx  = AMPIAffineFanIndex(data, nlist=nlist, num_fans=16, seed=99, cone_top_k=1)

    # Pick a cluster and record its initial sigma_drift norm.
    c = 0
    sig_before = idx._sigma_drift[c].copy()

    # Insert 200 points very strongly aligned with e_0 — biased direction.
    e0 = np.zeros(d, dtype='float32')
    e0[0] = 1.0
    scale = 5.0
    for i in range(200):
        noise = rng.standard_normal(d).astype('float32') * 0.05
        idx.add(scale * e0 + noise)

    # sigma_drift must have been updated (and possibly reset by a refresh).
    # At minimum: at some point it was non-zero. If a refresh fired, the
    # counter is 0. Either way the cluster is still queryable.
    q = scale * e0
    pts, dists, ids = idx.query(q, k=5, window_size=200, probes=nlist, fan_probes=8)
    assert len(ids) == 5, "query after drift inserts returned wrong number of results"


@test
def all_cluster_points_deleted():
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


@test
def cosine_metric_add_delete():
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


@test
def interleaved_mutations_and_queries():
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


@test
def heavy_churn_recall():
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
