# AMPI — Architecture & Roadmap

## 1. Streaming Insertion (Phase 1 — complete)

### Cluster Assignment — Nearest-Centroid (top-K)
- [x] Assign new point to top-K clusters by L2 distance to centroids.
- [x] Within each cluster, assign to top-K cones by |normalised projection|.
- [x] Update cluster centroid via EMA: μ_c ← (N_c·μ_c + x) / (N_c + 1).
- [x] Pre-allocated capacity buffer for `self.data` and `self._deleted_mask`:
      doubling buffer in `add()`, amortised O(1) per insert.
- [x] Periodic cluster merge: after every `merge_interval` inserts, check pairs of
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
- [x] Per-cluster fan axes (instead of global): recompute F axes from Σ_drift eigenvectors
      on refresh via deflated power iteration (10 steps per axis, F axes total).
- [x] Replace full-rank Σ_drift (d×d float64, O(nlist·d²)) with rank-F Oja sketch
      (d×F float32) — done. At d=960/nlist=1000: 7.4 GB → 120 MB. See DATABASE_PLAN.md §Memory.

### Sorted-Array Insert
- [x] SortedCone C++ class: F sorted `std::vector<std::pair<float,uint32_t>>` per cone.
- [x] `insert(proj_values, global_id)` — O(F·(log n + n)) via `std::lower_bound` + vector shift.
- [x] `remove(global_id)` — O(1) tombstone in `std::unordered_set`.
- [x] `compact()` — O(F·n) physical eviction of tombstoned entries.
- [x] `query(q_projs, window_size)` — union-window, returns sorted global IDs.
- [x] `is_covered(q_projs, w, kth_proj)` — early-stopping coverage check.
- [x] Replace `std::vector` with B-tree or skip-list per cone when n_cone > 10k to get
      O(log n) insert instead of O(n) shift.

### Delete / Update
- [x] `delete(global_id)` — tombstones in all cones via `_point_cones` inverse index;
      fires `_local_refresh` when cluster tombstone fraction ≥ 10 %.
- [x] `update(global_id, x)` — delete + insert.
- [x] `_deleted_mask` boolean array filters tombstoned entries from query fast-path
      (when fan_probes ≥ F and cluster_global is returned directly).

See **DATABASE_PLAN.md** for the concrete phased implementation plan.

---

## 2. Persistence Layer (Phase 2 — complete)

### Phase-2 Prerequisites (branch `phase-2-prereqs`)
- [x] `AMPIIndex::get_U_drift(c)` — returns `(d, F) float32` copy of the Oja sketch;
      needed by the checkpoint serializer to snapshot per-cluster drift state.
- [x] `SortedCone::get_axis_pairs(l)` — returns `(projs, ids)` for axis l; needed by
      the checkpoint serializer to read cone pairs back out of C++.
- [x] mmap-backed `_data_buf` (`data_path=` kwarg on `AMPIAffineFanIndex`): replaces
      `np.empty` with `np.memmap`; OS pages in only touched clusters — prerequisite
      for single-node GIST 1M on a 16 GB machine.
- [x] `AMPIIndex::from_stream` C++ factory: assembles index from pre-built `SortedCone`
      objects + existing mmap data file; skips `_build_cones` random mmap access.
- [x] `StreamingBuildDispatcher` + `streaming_build()` (`ampi/streaming.py`): single
      sequential pass over data — 50k-row sample for k-means, one streaming pass
      accumulates per-(cluster,cone) projections in ≈72 MB RAM (n-independent), builds
      `SortedCone` objects. Peak RSS ≈90 MB for GIST 1M d=960. Benchmark auto-selects
      streaming for n > 200k when data_path is set.

### Write-Ahead Log
- [x] Append-only binary WAL (`ampi/wal.py`): one record per mutation (INSERT|DELETE,
      global_id, vector, timestamp_ns, CRC32).
- [x] Flush on every insert (configurable `wal_batch_size` for throughput).
- [x] `replay_wal(idx, path, after_timestamp_ns)` — replays post-checkpoint mutations.
- [x] `truncate_wal(path, d)` — resets WAL to header-only after checkpoint.

### Checkpointing
- [x] Checkpoint serializer (`ampi/checkpoint.py`): 66-byte header (CRC32) + centroids
      + axes + cluster_counts + U_drift + per-cluster (global_ids, cone pairs).
- [x] `save_checkpoint(idx, path) → timestamp_ns`; `load_checkpoint(path, data_path)`.
- [x] Calls `AMPIAffineFanIndex.from_stream` on load — mmap raw-vector file preserved.
- [x] WAL `wal_path=` / `wal_batch_size=` kwargs on `AMPIAffineFanIndex.__init__` and
      `from_stream`; `add()` / `delete()` log mutations automatically.

### Query Performance (done 2026-03-31)
- [x] `l2_distances`: gather candidates into contiguous buffer → sequential GEMV
      (eliminates random mmap access interleaved with compute; compiler auto-vectorises).
- [x] `union_query`: replaced two O(n) mask scans with sort+unique on the hit list
      (eliminates 2× O(n=1M) passes → ~2.7 ms on GIST-scale n).
- [x] `AMPIIndex::query` / `query_candidates`: GIL released for full computation section
      (input copied to C++ buffer before release; output arrays allocated after reacquire)
      → Python threads can now call `query` in parallel without serialisation.

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

## 4. C++ Port — complete

All phases merged (branch `cpp-pipeline`).

- [x] Hot-path kernels: `project_data`, `l2_distances`, `union_query`.
- [x] `_kernels.py` falls back to numba JIT when the extension is absent.
- [x] `SortedCone`: mutable sorted cone with `insert`, `remove`, `compact`, `query`, `is_covered`.
- [x] BLAS-accelerated `project_data` via `ampi/_gemm.hpp` (Accelerate / OpenBLAS / MKL /
      AVX2 / NEON); 20–112× over scalar loop.
- [x] `best_clusters`, `best_fan_cones`: C++ nth_element replaces NumPy argsort per query.
- [x] `update_drift_and_check`: Oja rank-F subspace sketch + leading-eigenvec angle check in C++.
- [x] `AMPIIndex` C++ class: owns all mutable state; `add`, `remove`, `local_refresh`,
      `build_cones`, full adaptive query loop, `batch_add`, `batch_delete`.
- [x] `std::shared_mutex` — concurrent reads, serialised writes; GIL released on hot paths.
- [x] `_CppConesProxy` / `_CppClusterCones` — Python `cluster_cones[c][f]` proxies to live
      C++ `SortedCone` objects (no `None` stubs).

### Open
- [x] Replace `std::vector` per-cone sorted array with B-tree / skip-list when
      n_cone > 10k (reduces insert from O(n_cone) shift to O(log n_cone)).

### Notes
- K=2 QPS collapse on SIFT (0.5–6 vs K=1's 3–19) is a memory cache artifact:
  K=2 doubles footprint (2×nlist×F arrays), thrashing L3 at 1M×128-D.
  Recall improvement is real (K=2: 0.977 vs K=1: 0.973). Now mitigated by the C++ query loop.
- Python API surface unchanged; tuner.py unchanged.

---

## 5. Benchmarking TODOs

- [x] Fashion-MNIST (60k, d=784)
- [x] SIFT-128 full 1M
- [x] Recall@1 / Recall@100 curves (benchmarks/benchmark.py reports all three)
- [x] GIST (1M, d=960) — 200k cap (historical); full 1M now unblocked via streaming build.
- [ ] GIST full 1M benchmark — unblocked by `streaming_build`; run to validate recall/QPS.
- [x] Profile per-cluster fan-axis variance to validate drift-detection threshold θ_drift
- [ ] Vote-distribution analysis: for a sample of queries, record how many projection
      windows each candidate appears in (vote count).  Applies to both Binary (L random
      projections) and AffineFan (F cone axes × probed clusters).  Plot vote histogram
      for true NNs vs false positives to find a natural threshold.  Only then consider
      adding a `min_votes` parameter to `union_query` — must verify the true-NN vote
      distribution stays well above any threshold before cutting candidates.

---

## 6. Packaging & Testing

- [x] CI: smoke test on every push/PR (`.github/workflows/ci.yml`)
- [x] CI: stress test covering adversarial add/delete/update scenarios (`tests/stress_test.py`)
- [x] CI: cppcheck + two-pass LLM C++ review on every push (`.github/workflows/cpp-verify.yml`)
- [x] Pin numba/numpy versions in `pyproject.toml`
- [ ] Publish to PyPI once recall is competitive and streaming insert is stable
