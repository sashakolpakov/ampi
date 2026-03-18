# AMPI Vector Database — Construction Plan

This document translates the architecture notes in `TODO.md` into a concrete
build sequence, component by component, with the current Python/C++ codebase
as the foundation.

---

## Phase 0 — Foundations (pre-DB, current state)

**What exists:**
- `AMPIAffineFanIndex` — in-memory, single-process, build-time-only.
- `_ampi_ext.so` — C++ kernels (`project_data`, `l2_distances`, `union_query`).
  `project_data` uses `ampi/_gemm.hpp` (Accelerate / OpenBLAS / MKL / AVX2+NEON
  fallback); 20–112× faster than the prior scalar loop.
- `AFanTuner` — GP-BO over alpha/K, Pareto suggestion sweep.

**Prerequisite before Phase 1:**
- Complete the C++ port of the full index hot path (TODO §3) so that the
  insertion primitives we add in Phase 1 are fast from day one.
- Specifically: move the sorted-array cone structures into C++ objects exposed
  via pybind11. `std::vector<std::pair<float,uint32_t>>` per cone, with
  `std::lower_bound` insert — this is the in-place mutable structure that
  streaming insertion in Phase 1 needs.

---

## Phase 1 — Mutable Single-Node Index

Goal: `add(x)` and `delete(id)` without full rebuild.

### 1.1  Sorted-Array Insert

**Data structure change (C++ side):**

```
Per cone:
  std::vector<std::pair<float, uint32_t>>  sorted by projection value
```

Replace the current immutable `(sorted_idxs, sorted_projs)` NumPy arrays with
this resizable structure. The pybind11 wrapper exposes:

```python
cone.insert(proj_value: float, global_id: int)   # O(log n + n) amortised
cone.remove(global_id: int)                       # tombstone, O(1)
cone.compact()                                    # remove tombstones, O(n)
```

For the O(log n) `insert` case, replace `std::vector` with a B-tree or
`std::deque`-backed skip-list if n_cone > 10k. At smaller sizes the vector
cache locality wins.

### 1.2  Dirichlet-Process Cluster Assignment

When inserting point `x`:

1. Compute soft responsibilities:
   ```
   r_c  ∝  N_c · N(x | μ_c, σ²_c·I)       for each cluster c
   ```
   Only evaluate the top-`cone_top_k` clusters by L2 distance to centroid
   (prune the rest — they'll have negligible probability).

2. Assign `x` to the `cone_top_k` highest-responsibility clusters.

3. Within each assigned cluster `c`:
   - Project `x - μ_c` onto all F axes.
   - Find the top-`cone_top_k` cones by `|projection| / ‖x - μ_c‖`.
   - Call `cone.insert(proj_value, global_id)` for each selected cone.

4. Update cluster centroid via exponential moving average:
   ```
   μ_c ← (N_c · μ_c + x) / (N_c + 1),   N_c ← N_c + 1
   ```

**Periodic cluster merge:** after every `merge_interval` inserts, for each
pair of clusters whose centroids satisfy `‖μ_i - μ_j‖ < ε_merge`, compute
Bayesian model comparison (BIC approximation: compare two Gaussians vs one).
Merge the pair if the single-Gaussian model wins. Cost: O(nlist²) centroid
comparisons — negligible for nlist ≤ 2000.

### 1.3  Direction Drift Detection

Per cluster, maintain an EMA of the outer product of nearest-neighbour pairs:

```
Σ_drift_c ← (1 − β) · Σ_drift_c  +  β · (x − y)(x − y)^T
```

where `y` = the approximate nearest neighbour of `x` within cluster `c`
(cheaply found from the cone candidate set before L2 rerank).

After each insert, check whether the leading eigenvector of `Σ_drift_c`
(power iteration, 3–5 steps) has rotated more than `θ_drift` degrees from the
current fan axes. If so, schedule a **local fan-axis refresh** for cluster `c`:

1. Recompute F axes from `Σ_drift_c`'s top-F eigenvectors (or keep random —
   empirically random ≈ geometry-guided at F=128).
2. Re-project all `N_c` points in cluster `c` onto the new axes.
3. Rebuild cone sorted arrays for cluster `c` — O(N_c · F · log N_c).
4. Swap atomically (write-lock cluster `c` for the swap, then release).

### 1.4  Delete / Update

- **Logical delete:** set tombstone bit in each cone that contains `global_id`.
  Query paths skip tombstoned entries already (they just inflate the candidate
  pool slightly).
- **Physical compaction:** during fan-axis refresh (§1.3) or when tombstone
  fraction exceeds `tombstone_threshold` (default 10%), call `cone.compact()`
  on all cones in the cluster.
- **Update = delete + insert.** The caller provides the new vector; the index
  removes the old entry and inserts the new one.

---

## Phase 2 — Persistence Layer

Single-node durability before distributing.

### 2.1  Write-Ahead Log (WAL)

Append-only binary log, one record per mutation:

```
record := { op: INSERT|DELETE, global_id: u64, vector: float32[d],
            timestamp: u64, checksum: u32 }
```

- Flush to disk on every insert (or in micro-batches of 64 for throughput).
- On startup, replay WAL to rebuild in-memory state from the last checkpoint.

### 2.2  Checkpointing

Periodically serialize the full index to disk:

```
checkpoint := {
  header:     { version, n, d, nlist, F, K, seed, timestamp },
  centroids:  float32[nlist, d],
  axes:       float32[F, d],
  per_cluster: [
    { centroid: float32[d], n_c: u32,
      sigma_drift: float32[d, d],
      cones: [ { n_f: u32, pairs: (float32, u32)[n_f] }, ... F cones ] }
  ],
  global_ids: u64[n],   # maps internal slot → user-provided ID
}
```

Use `mmap`-friendly layout: fixed-size header, then variable-length cone
blocks. This allows memory-mapped read-only serving of a checkpoint while a
new one is being written.

Checkpoint cadence: every `checkpoint_interval` WAL records, or triggered
manually. Truncate WAL after successful checkpoint.

---

## Phase 3 — Distributed Architecture

### 3.1  Component Map

```
┌──────────────────────────────────────────────┐
│                  Coordinator                 │
│  - centroid table  (nlist × d float32)       │
│  - fan-axis table  (nlist × F × d float32)   │
│  - shard map       (cluster_id → shard_addr) │
│  - Raft log for the above (tiny)             │
└────────────┬─────────────────────────────────┘
             │ gRPC / ZeroMQ
    ┌────────┴─────────────────────────────────┐
    │  Shard 0   │  Shard 1   │  ...  │ Shard S│
    │  clusters  │  clusters  │       │clusters│
    │  0..k0     │  k0..k1    │       │ks..K-1 │
    │  WAL       │  WAL       │       │ WAL    │
    │  checkpoint│  checkpoint│       │checkpt │
    └────────────┴────────────┴───────┴────────┘
```

### 3.2  Sharding Strategy

- **One cluster per shard** (or multiple small clusters per shard for datasets
  with nlist > desired shard count).
- Shard owns: raw vectors, cone sorted arrays, fan axes, centroid, WAL,
  checkpoint.
- Coordinator owns only: centroid table + fan-axis table + shard map.
  This is tiny — nlist=1000, d=128 → 512 KB. Fully replicable in memory on
  every coordinator replica.

### 3.3  Insert Flow

```
client  →  coordinator:  insert(vector, user_id)
coordinator:
  1. compute top-K_route nearest centroids  (BLAS on centroid table)
  2. for each of the top-K_route clusters:
       forward (vector, user_id, cluster_id) → owning shard
shard:
  3. acquire cluster write-lock
  4. WAL append
  5. Phase-1 insertion (cone insert, centroid EMA, drift check)
  6. release lock
  7. ack → coordinator
coordinator:
  8. ack → client after all K_route shards ack
```

K_route = cone_top_k (typically 1 or 2). Boundary points live on 2 shards,
giving cross-shard recall without a separate replication layer.

### 3.4  Query Flow

```
client  →  coordinator:  query(vector, k, probes=cp, fan_probes=fp, window=w)
coordinator:
  1. find top-cp nearest centroids  (BLAS on centroid table)
  2. fan-out: send (vector, fp, w) to each of the cp owning shards in parallel
shard i:
  3. local query_candidates(vector, w, probes=1, fan_probes=fp)
     (shard already owns the relevant cluster; probes=1)
  4. L2 rerank → return top-k local candidates with distances
coordinator:
  5. k-way heap merge of all shard results
  6. return global top-k → client
```

**Early termination:** once the merged heap's k-th distance is smaller than
the (k+1)-th distance from any pending shard, cancel remaining RPCs.

### 3.5  Cluster Splits

When shard `s`'s cluster `c` grows beyond `N_max`:

1. Shard runs mini k-means (k=2) on its local data → two new centroids.
2. Shard partitions cones into two groups and rebuilds sorted arrays locally.
3. Shard sends new centroid pair + new fan axes to coordinator.
4. Coordinator atomically:
   - Removes cluster `c` entry.
   - Adds clusters `c'`, `c''` with new centroids + fan axes.
   - Bumps epoch counter.
5. In-flight queries that arrived before the epoch bump are served stale —
   acceptable (they'll miss at most one cluster worth of points, bounded miss).

No global IVF rebuild. Other shards are unaffected.

### 3.6  Rebalancing

Background task on coordinator:

1. Periodically polls all shard sizes (vector counts).
2. Identifies overloaded shards (size > 1.5× mean) and underloaded shards.
3. Selects a cluster to migrate: smallest cluster on the most overloaded shard.
4. Migration protocol:
   a. Source shard: snapshot checkpoint for cluster `c`.
   b. Source ships checkpoint + WAL tail to destination shard.
   c. Destination replays and acks.
   d. Coordinator atomic update: cluster `c` → new shard address.
   e. Source shard physically deletes cluster `c` data.

Cluster migration granularity (not individual vectors) keeps the protocol
simple and the coordinator state small.

---

## Phase 4 — Implementation Sequence

| Step | What | Builds on |
|------|------|-----------|
| 4.1 | C++ mutable cone (§1.1) | `_ext.cpp` |
| 4.2 | Python `add(x)` / `delete(id)` wrappers (§1.2–1.4) | 4.1 |
| 4.3 | WAL + checkpoint serializer (§2.1–2.2) | 4.2 |
| 4.4 | Single-node server (Python asyncio / gRPC) | 4.3 |
| 4.5 | Coordinator (centroid table, routing) | 4.4 |
| 4.6 | Multi-shard query fan-out | 4.5 |
| 4.7 | Cluster split protocol | 4.6 |
| 4.8 | Rebalancing + background drift-check | 4.7 |

Each step is independently testable: 4.1–4.2 via unit tests on small datasets,
4.3 via round-trip checkpoint tests, 4.4–4.8 via integration tests on a local
multi-process cluster.

---

## Key Constants (starting defaults, tune per dataset)

| Parameter | Default | Note |
|-----------|---------|------|
| `cone_top_k` K | 1 or 2 | 2 for better cross-boundary recall |
| `merge_interval` | 10 000 inserts | cluster merge check cadence |
| `ε_merge` | 0.05 × median inter-centroid dist | merge threshold |
| `β` (drift EMA) | 0.01 | slow decay for stability |
| `θ_drift` | 15° | fan-axis refresh trigger |
| `tombstone_threshold` | 0.10 | fraction before compaction |
| `checkpoint_interval` | 100 000 WAL records | or ~1 min, whichever first |
| `N_max` (split trigger) | 3 × (n / nlist) | 3× expected cluster size |
| `K_route` | = cone_top_k | shards to fan out to on insert |
