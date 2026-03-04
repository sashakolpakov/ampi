# AMPI — Architecture & Roadmap

## 1. Bayesian Streaming Insertion

### Motivation
Batch rebuild of the full index (k-means + fan axes) is O(n·F) and impractical for live systems. We want O(F·log n_cone) amortised insertion with correctness guarantees.

### Cluster Assignment — Dirichlet-Process Mixture
- Maintain a soft Dirichlet-process prior over cluster memberships.
- For a new point x, compute responsibilities r_c ∝ N_c · p(x | μ_c, Σ_c).
- Assign to top-K clusters (consistent with cone_top_k=K for the query path).
- Periodically merge clusters whose centroids drift within ε of each other (Bayesian model comparison on marginal likelihoods).

### Direction Drift Detection
- Per cluster, maintain an exponential moving average of outer products of nearest-neighbour pairs: Σ_drift ← (1-β)·Σ_drift + β·(x-y)(x-y)^T.
- When the leading eigenvector of Σ_drift rotates more than θ_drift degrees from the current fan axis, flag the cluster for fan-axis refresh.
- Fan-axis refresh is local (one cluster, one set of F axes) — O(n_cluster · F) not O(n · F).

### Sorted-Array Insert
- Each cone stores a sorted array of (projection_value, global_id) pairs.
- Insert is `bisect_left` + `list.insert` → O(log n_cone + n_cone) amortised, or O(log n_cone) with a skip-list / B-tree per cone.
- Total per-point cost: K cones × O(F · log n_cone) = O(K·F·log n_cone).

### Delete / Update
- Mark-and-sweep: logical delete with tombstone bit; physical removal during next cluster refresh.
- Update = delete + insert.

---

## 2. Distributed Vector Database

### Sharding Strategy
- Each k-means cluster maps to exactly one shard.
- Shard owns: raw vectors, cone sorted arrays, fan axes, centroid.
- Number of shards = number of clusters (or multiple clusters per shard for small datasets).

### Coordinator
- Holds replicated centroid table (one centroid per cluster, tiny footprint).
- Holds replicated fan axes per cluster (F · d floats per cluster).
- Routes queries: find top-K_route nearest centroids, fan-out to those shards.
- K=2 soft assignment at insert time means boundary points live on 2 shards → recall is preserved across shard boundaries without a separate replication layer.

### Query Fan-In
- Coordinator sends (query, fp, w) to each selected shard.
- Shards return their local candidate lists (sorted by projection distance).
- Coordinator merges with a k-way heap, applies global reranking if needed.
- Early termination: once merged candidate pool has stable top-K for two rounds, stop fan-in.

### Writes
- WAL per shard for durability.
- Insertion hits coordinator → coordinate cluster assignment → forwarded to 1 or 2 shards.
- Coordinator applies drift-detection heuristic; schedules local fan-axis refresh on the owning shard when triggered.

### Cluster Splits
- When a shard's cluster grows beyond N_max, split locally (mini k-means on the shard).
- Coordinator updates centroid table and fan-axis table atomically (epoch bump).
- No global IVF rebuild required; other shards are unaffected.

### Rebalancing
- Periodic background task compares shard sizes; migrates whole clusters (not individual vectors) between shard hosts.
- Cluster migration = ship sorted arrays + fan axes + WAL tail, then atomic centroid-table update.

---

## 3. C++ Port (next major milestone)

- Core index (affine_fan.py) → C++ with pybind11 wrapper.
- Sorted arrays → `std::vector<std::pair<float,uint32_t>>` with `std::lower_bound`.
- SIMD projection: AVX2 dot products for all-F-axes at once.
- Expected 20-50× QPS improvement over current Python path.
- K=2 QPS collapse on SIFT (0.5–6 vs K=1's 3–19) is a Python/NumPy cache artifact:
  K=2 doubles memory footprint (2×nlist×F sorted arrays), thrashing L3 at 1M×128-D.
  The algorithmic recall improvement is real (K=2 tops 0.977 vs K=1's 0.973 at equal
  candidates). A C++ implementation with tighter memory layout will close this gap.
- Python API surface stays identical; tuner.py and benchmark.py unchanged.

---

## 4. Benchmarking TODOs

- [x] Fashion-MNIST (60k, d=784)
- [x] SIFT-128 full 1M
- [ ] GIST (1M, d=960) — high-d stress test
- [ ] Add recall@1 / recall@100 curves in addition to recall@10
- [ ] Profile per-cluster fan-axis variance to validate drift-detection threshold θ_drift
- [ ] ann-benchmarks wrapper: `module.py` with `fit` / `set_query_arguments` / `query`, `Dockerfile`, `config.yml`
- [ ] Target: competitive on MNIST and SIFT before opening ann-benchmarks PR

---

## 5. Packaging

- [ ] Publish to PyPI once recall is competitive
- [ ] Add CI: import + smoke tests on push
- [ ] Pin numba/numpy versions in pyproject.toml
