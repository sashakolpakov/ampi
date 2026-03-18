# AMPI — Architecture & Roadmap

## 1. Streaming Insertion (Phase 1 — complete)

### Cluster Assignment — Nearest-Centroid (top-K)
- [x] Assign new point to top-K clusters by L2 distance to centroids.
- [x] Within each cluster, assign to top-K cones by |normalised projection|.
- [x] Update cluster centroid via EMA: μ_c ← (N_c·μ_c + x) / (N_c + 1).
- [x] Pre-allocated capacity buffer for `self.data` and `self._deleted_mask`:
      doubling buffer in `add()`, amortised O(1) per insert.
- [ ] Periodic cluster merge: after every `merge_interval` inserts, check pairs of
      clusters whose centroids are within ε_merge; merge if it reduces mean
      quantisation error (no full DP model comparison needed).

### Direction Drift Detection
- [x] Per cluster, maintain EMA of outer products: Σ_drift ← (1−β)·Σ_drift + β·v·vᵀ
      where v = x − y and y is the approximate NN of x found from the just-inserted
      cones (window=8); falls back to v = x − μ_c when the cone has only one point.
- [x] Power iteration (5 steps) to find leading eigenvector of Σ_drift.
- [x] If leading eigenvector is > θ_drift degrees from all fan axes, trigger _local_refresh.
- [x] _local_refresh: rebuild all cones for cluster c from current live points + current
      global axes, evict tombstones, reset Σ_drift and tombstone counter.
- [ ] Per-cluster fan axes (instead of global): recompute F axes from Σ_drift eigenvectors
      on refresh. Currently kept global (random ≈ geometry-guided at F=128).

### Sorted-Array Insert
- [x] SortedCone C++ class: F sorted `std::vector<std::pair<float,uint32_t>>` per cone.
- [x] `insert(proj_values, global_id)` — O(F·(log n + n)) via `std::lower_bound` + vector shift.
- [x] `remove(global_id)` — O(1) tombstone in `std::unordered_set`.
- [x] `compact()` — O(F·n) physical eviction of tombstoned entries.
- [x] `query(q_projs, window_size)` — union-window, returns sorted global IDs.
- [x] `is_covered(q_projs, w, kth_proj)` — early-stopping coverage check.
- [ ] Replace `std::vector` with B-tree or skip-list per cone when n_cone > 10k to get
      O(log n) insert instead of O(n) shift.

### Delete / Update
- [x] `delete(global_id)` — tombstones in all cones via `_point_cones` inverse index;
      fires `_local_refresh` when cluster tombstone fraction ≥ 10 %.
- [x] `update(global_id, x)` — delete + insert.
- [x] `_deleted_mask` boolean array filters tombstoned entries from query fast-path
      (when fan_probes ≥ F and cluster_global is returned directly).

See **DATABASE_PLAN.md** for the concrete phased implementation plan.

---

## 2. Persistence Layer (Phase 2 — not started)

### Write-Ahead Log
- [ ] Append-only binary WAL: one record per mutation (INSERT|DELETE, global_id,
      vector, timestamp, checksum).
- [ ] Flush on every insert (or micro-batches of 64 for throughput).
- [ ] Replay WAL on startup to rebuild in-memory state from last checkpoint.

### Checkpointing
- [ ] Checkpoint serializer: header + centroids + axes + per-cluster cone pairs.
- [ ] mmap-friendly layout for read-only serving while new checkpoint is being written.
- [ ] Truncate WAL after successful checkpoint.

---

## 3. Distributed Architecture (Phase 3 — not started)

### Sharding
- [ ] One cluster per shard; shard owns raw vectors, cone arrays, fan axes, centroid, WAL.
- [ ] Coordinator holds centroid table + fan-axis table + shard map (tiny footprint).
- [ ] K=2 soft assignment means boundary points live on 2 shards — cross-shard recall
      without a separate replication layer.

### Coordinator & Query Fan-Out
- [ ] Coordinator routes inserts: find top-K_route nearest centroids, forward to shards.
- [ ] Query fan-out: coordinator fans to cp shards in parallel; shards return local
      top-k candidates; coordinator k-way heap merge with early termination.

### Cluster Splits & Rebalancing
- [ ] Cluster split when N_c > N_max: shard runs mini k-means (k=2) locally, ships
      new centroid pair to coordinator for atomic epoch bump.
- [ ] Rebalancing: background task migrates whole clusters (not vectors) between shards.

See **DATABASE_PLAN.md** for the phased build sequence and default constants.

---

## 4. C++ Port

### Done
- [x] Hot-path kernels: `project_data`, `l2_distances`, `union_query`
      (`ampi/_ext.cpp`, compiled to `_ampi_ext.so`).
- [x] `_kernels.py` transparently falls back to numba JIT if the extension is absent.
- [x] `SortedCone` class: mutable sorted cone with `insert`, `remove`, `compact`,
      `query`, `is_covered`, `all_ids`, `from_arrays` — replaces immutable NumPy dicts.

### Remaining
- [x] BLAS-accelerated `project_data`: `ampi/_gemm.hpp` dispatches to Accelerate /
      OpenBLAS / MKL at compile time; AVX2 / NEON tiled micro-kernel fallback.
      Measured 20–112× speedup over the old scalar loop (`benchmarks/_bench_sgemm.py`).
- [ ] Move centroid EMA, drift-covariance update, and power iteration into C++ for the
      insert hot-path (currently Python + numpy, adequate for Phase 1 throughput).
- [ ] Replace `std::vector` cone with B-tree / skip-list when n_cone > 10k.

### Notes
- K=2 QPS collapse on SIFT (0.5–6 vs K=1's 3–19) is a Python/NumPy cache artifact:
  K=2 doubles memory footprint (2×nlist×F sorted arrays), thrashing L3 at 1M×128-D.
  The algorithmic recall improvement is real (K=2: 0.977 vs K=1: 0.973 at equal
  candidates). The mutable C++ cone layout will reduce this gap.
- Python API surface stays identical; tuner.py unchanged.
- `benchmark.py` moved to `benchmarks/`.

---

## 5. Benchmarking TODOs

- [x] Fashion-MNIST (60k, d=784)
- [x] SIFT-128 full 1M
- [x] Recall@1 / Recall@100 curves (benchmarks/benchmark.py reports all three)
- [ ] GIST (1M, d=960) — high-d stress test
- [ ] Profile per-cluster fan-axis variance to validate drift-detection threshold θ_drift

---

## 6. Packaging & Testing

- [x] CI: smoke test on every push/PR (`.github/workflows/ci.yml`)
- [x] CI: stress test covering adversarial add/delete/update scenarios (`tests/stress_test.py`)
- [x] Pin numba/numpy versions in `pyproject.toml`
- [ ] Publish to PyPI once recall is competitive and streaming insert is stable
