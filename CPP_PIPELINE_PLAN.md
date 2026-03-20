# C++ Pipeline Plan: Full Query / Insertion / Deletion in C++

Branch: `cpp-pipeline`
Created: 2026-03-19

---

## Current State (as of 2026-03-19)

### In C++ (`ampi/_ext.cpp`) — completed

| Symbol | What it does | Phase |
|--------|-------------|-------|
| `project_data(data, proj_dirs)` | BLAS-dispatched SGEMM: (n,d)×(F,d)ᵀ → (F,n) | pre-1 |
| `l2_distances(data, query, cands)` | Squared L2 from query to m candidates | pre-1 |
| `union_query(sorted_idxs, sorted_projs, q_projs, w)` | Binary-search + window union on static sorted arrays | pre-1 |
| `SortedCone` class | Mutable per-cone sorted structure: `insert`, `remove`, `compact`, `size`, `all_ids`, `query`, `is_covered` | pre-1 |
| `best_clusters(centroids, q, probes)` | Nearest-centroid top-K via nth_element | 1 ✅ |
| `best_fan_cones(axes, q_centered, fan_probes)` | Top-K cones by \|norm proj\| via nth_element | 1 ✅ |
| `update_drift_and_check(sigma, axes, v, beta, theta)` | Fused EMA + 5-step power iteration; returns refresh flag | 2 ✅ |
| `AMPIIndex` class | Owns all mutable state; `add()`, `remove()`, `_local_refresh()`, `_build_cones()` fully in C++ | 3 ✅ |

### Still in Python (`ampi/affine_fan.py`)

| Python code | Hot? | Target phase |
|-------------|------|-------------|
| `query()` — adaptive window expansion loop + coverage check | Every query | 4 |
| `query_candidates()` — candidate pool before rerank | Every query | 4 |
| `_mini_batch_kmeans` + `_build_cones_for_cluster` | Build-time only | 5 (partial) |

---

## Architecture of the New C++ Layer

We introduce a new C++ class `AMPIIndex` that owns the full mutable index state.
Python becomes a thin pybind11 wrapper — no hot-path logic in Python.

```
┌─────────────────────────────────────────────────────┐
│  Python AMPIAffineFanIndex  (thin wrapper)          │
│  __init__: k-means + cone build stays Python        │
│  add / delete / update / query → delegate to C++    │
└────────────────────┬────────────────────────────────┘
                     │ pybind11
┌────────────────────▼────────────────────────────────┐
│  C++ AMPIIndex                                      │
│  ┌─────────────────┐  ┌────────────────────────┐   │
│  │  data store     │  │  cluster metadata       │   │
│  │  float* buf     │  │  centroids (nlist×d)    │   │
│  │  bool* del_mask │  │  cluster_global lists   │   │
│  │  atomic n       │  │  cluster_cones (F each) │   │
│  └─────────────────┘  │  sigma_drift (nlist,d,d)│   │
│                       │  cluster_counts         │   │
│                       └────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐  │
│  │  SortedCone (already in C++)                 │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

State moved from Python dicts/lists to C++ structs:
- `cluster_global[c]` → `std::vector<uint32_t>` per cluster
- `cluster_cones[c]` → `std::vector<SortedCone>` per cluster
- `_point_cones[gid]` → `std::vector<std::pair<uint16_t,uint16_t>>` per global_id (cluster, cone)
- `_sigma_drift[c]` → flat `double[d*d]` per cluster
- `centroids` → `float[nlist*d]`

---

## Phased Plan

---

### Phase 1 — Move `_best_clusters` and `_best_fan_cones` to C++ ✅ DONE

**Goal:** Eliminate the two NumPy calls on every query.
**Scope:** Add two standalone C++ functions; no class yet.

#### New C++ functions (add to `_ext.cpp`)

```cpp
// Returns indices of the `probes` nearest centroids to query q.
// centroids: (nlist, d) float32;  q: (d,) float32
py::array_t<int32_t> best_clusters(
    py::array_t<float> centroids, py::array_t<float> q, int probes);

// Returns indices of the `fan_probes` cones with highest |norm proj|.
// axes: (F, d) float32;  q_centered: (d,) float32
py::array_t<int32_t> best_fan_cones(
    py::array_t<float> axes, py::array_t<float> q_centered, int fan_probes);
```

Both functions use partial-sort (`std::partial_sort` or `nth_element`) to avoid
a full sort when probes ≪ nlist.

#### Changes to `affine_fan.py`

Replace `_best_clusters` and `_best_fan_cones` method bodies with calls to the
new C++ functions when `_HAS_EXT` is True, keep NumPy fallback otherwise.

#### Checkpoint / Test

- `tests/smoke_test.py` must still pass (recall@10 ≥ 0.80).
- Add `tests/test_phase1.py`: compare C++ and NumPy outputs on random inputs
  for 100 random queries, assert identical results.

---

### Phase 2 — Move drift covariance EMA and power iteration to C++ ✅ DONE

**Goal:** Eliminate the d×d NumPy outer product + 5-step matrix-vector product
on every `add()`.  At d=128 this is 128²=16k multiplications per insert, all in
Python overhead.

#### New C++ function

```cpp
// Performs one in-place EMA step on sigma_drift and runs 5-step power iteration.
// Returns true if the leading eigenvector is > theta_deg from all axes
// (i.e. _local_refresh should be called).
//
// sigma: (d*d,) float64, modified in-place
// axes:  (F, d) float32
// displacement: (d,) float64
// beta: EMA decay (0.01)
// theta_deg: drift angle threshold (15.0)
bool update_drift_and_check(
    py::array_t<double> sigma,   // (d*d,) flat, modified in-place
    py::array_t<float>  axes,    // (F, d)
    py::array_t<double> displacement,  // (d,)
    double beta, double theta_deg);
```

The function combines `_check_drift` + the EMA update in `add()`:
```
Σ ← (1-β)Σ + β·v·vᵀ
then 5×: v ← Σ·v / ‖Σ·v‖
return cos_max < cos(θ)
```

#### Data structure change

`self._sigma_drift` changes from a Python list of (d,d) arrays to a single
`(nlist, d*d)` float64 contiguous array so the C++ function can operate in-place
without copy.

#### Changes to `affine_fan.py`

- Replace the EMA + `_check_drift` call block in `add()` with
  `update_drift_and_check(self._sigma_drift[c], self.axes, v, ...)`.
- Keep `_sigma_drift` as a flat `(nlist, d*d)` array; reshape to `(nlist, d, d)`
  only if needed externally.

#### Checkpoint / Test

- `tests/smoke_test.py` still passes.
- `tests/test_phase2.py`: run 1000 inserts on a 5k-point index, compare
  `_sigma_drift` values against a reference Python implementation to within 1e-10.
- Verify that drift-triggered refreshes still fire correctly (insert points
  clustered in a region not covered by any axis).

---

### Phase 3 — C++ `AMPIIndex` class: `add` and `delete` ✅ DONE

**Goal:** Move the full `add()` and `delete()` orchestration into a C++ class,
eliminating all per-insert Python overhead (GIL, dict lookups, np.append, etc.).

This is the largest phase. It introduces a new C++ class `AMPIIndex` that owns
all mutable state except the initial data buffer (which is still shared as a
NumPy array).

#### New C++ class skeleton

```cpp
class AMPIIndex {
public:
    // Immutable geometry (set at construction, never changed)
    int n_init, d, F, nlist, cone_top_k;
    std::vector<float> axes;       // (F*d) row-major

    // Mutable data store (shared with Python as a numpy buffer)
    float*  data_buf;              // pointer into numpy array
    bool*   del_mask;
    std::atomic<uint32_t> n;
    uint32_t capacity;

    // Per-cluster state
    std::vector<float>    centroids;           // (nlist*d)
    std::vector<int64_t>  cluster_counts;      // (nlist,)
    std::vector<int64_t>  cluster_tombstones;  // (nlist,)
    std::vector<std::vector<uint32_t>>    cluster_global;  // per-cluster global IDs
    std::vector<std::vector<SortedCone>>  cluster_cones;   // per-cluster F cones
    std::vector<std::vector<double>>      sigma_drift;     // per-cluster d*d

    // Inverse map: global_id → list of (cluster, cone) pairs
    std::vector<std::vector<std::pair<uint16_t,uint16_t>>> point_cones;

    uint32_t add(const float* x);
    void     remove(uint32_t global_id);
    void     _local_refresh(int c);
    void     _grow_buffers();
};
```

#### Construction protocol

The Python `__init__` still runs k-means and `_build_cones_for_cluster` (build
time, not hot path). It then calls `AMPIIndex.from_build(...)` passing the
resulting NumPy arrays. The C++ class absorbs the cone objects and cluster lists.

#### `add(x)` in C++

Maps directly from the current Python `add()`:
1. Normalize (cosine) if needed
2. `global_id = n.fetch_add(1)` (atomic; thread-safe for future concurrency)
3. Grow buffers if needed (double capacity)
4. Write `data_buf[global_id]` and `del_mask[global_id]`
5. Top-K centroid search (reuse Phase 1 `best_clusters` logic inlined)
6. For each top cluster:
   - Center, project onto axes (dot product loop, d ≤ 512 so no BLAS needed)
   - Top-K cones by |norm proj| (partial sort)
   - `cone.insert(proj, global_id)` for each selected cone
   - Append to `cluster_global[c]`, update `point_cones[global_id]`
   - Centroid EMA (simple fused multiply-add)
   - Drift EMA + power iteration (Phase 2 logic inlined)
   - If drift triggered: `_local_refresh(c)`

#### `remove(global_id)` in C++

Maps directly from `delete()`:
1. Bounds check + double-delete guard via `del_mask`
2. `del_mask[global_id] = true`, `n_deleted++`
3. For each `(c, f)` in `point_cones[global_id]`:
   - `cluster_cones[c][f].remove(global_id)` (tombstone)
   - Increment `cluster_tombstones[c]`
4. For each affected cluster: if fraction ≥ threshold → `_local_refresh(c)`

#### `_local_refresh(c)` in C++

Same logic as Python: extract live IDs, call `_build_cones_for_cluster` (C++ version of the Python helper), rebuild cone objects, reset drift + tombstone counters.

#### New pybind11 bindings

```python
ext.AMPIIndex(d, F, nlist, cone_top_k, axes_np)     # constructor
idx.add(x_np)           # → int global_id
idx.remove(global_id)   # → None
idx.get_data_view()     # → (n, d) float32 numpy view (no copy)
idx.get_deleted_mask()  # → (n,) bool numpy view
idx.get_centroids()     # → (nlist, d) float32 numpy view
```

#### Changes to `affine_fan.py`

`AMPIAffineFanIndex` becomes:
```python
class AMPIAffineFanIndex:
    def __init__(self, data, ...):
        # k-means + cone build (Python, build-time)
        ...
        if _HAS_EXT:
            self._cpp = _ampi_ext.AMPIIndex(...)
            # transfer cone objects to C++
        else:
            # existing Python path

    def add(self, x):
        return self._cpp.add(x) if self._cpp else self._py_add(x)

    def delete(self, global_id):
        if self._cpp: self._cpp.remove(global_id)
        else: self._py_delete(global_id)
```

#### Checkpoint / Test

- `tests/smoke_test.py` still passes.
- `tests/test_phase3.py`:
  - Build index on n=5k, d=64.
  - Insert 500 points one by one via `add()`.
  - Delete 100 random points.
  - Run `query()` on 50 queries; compare results against Python-path index
    built on same data. Assert recall@10 difference < 2pp.
  - Verify `get_data_view()` returns a valid zero-copy numpy array.
- Microbenchmark: measure `add()` throughput in Python-path vs C++-path
  (target: ≥ 3× speedup at d=128).

---

### Phase 4 — C++ query loop

**Goal:** Move `query()` and `query_candidates()` into C++ `AMPIIndex`.

#### New C++ methods

```cpp
// Returns (distances, global_ids) for the top-k nearest neighbours.
std::pair<py::array_t<float>, py::array_t<int32_t>>
AMPIIndex::query(const float* q, int k, int window_size, int probes, int fan_probes);

// Returns candidate pool (no rerank).
py::array_t<int32_t>
AMPIIndex::query_candidates(const float* q, int window_size, int probes, int fan_probes);
```

The adaptive window expansion loop (`while True`) and coverage check become a
simple C++ while-loop with no Python callbacks:

```cpp
int w = std::max(k, 8);
while (true) {
    // collect candidates from cone_ctxs + fallback_parts
    // filter tombstones
    if (w >= window_size || n_cands < k) break;
    // compute l2 distances, find kth_sq
    // check coverage via cone.is_covered(...)
    if (all_covered) break;
    w = std::min(w * 2, window_size);
}
// l2_distances + argpartition → top-k
```

Key implementation detail: the candidate union uses a flat `uint8_t mask[n]`
(already present in `union_query`) rather than a hash set — avoids allocation
on every loop iteration.

#### Changes to `affine_fan.py`

```python
def query(self, q, k=10, window_size=200, probes=10, fan_probes=2):
    q = self._prepare_query(q)
    if self._cpp:
        dists, ids = self._cpp.query(q, k, window_size, probes, fan_probes)
        return self.data[ids], dists, ids
    return self._py_query(q, k, window_size, probes, fan_probes)
```

#### Checkpoint / Test

- `tests/smoke_test.py` still passes.
- `tests/test_phase4.py`:
  - 50 queries on a freshly built 5k index.
  - Assert results identical to Python-path query (same candidate sets at
    same window_size, or recall difference < 1pp due to float ordering).
  - Microbenchmark: QPS improvement (target: ≥ 2× vs Phase 3 baseline at
    d=128, probes=10, fan_probes=2, window=200).

---

### Phase 5 — `_build_cones_for_cluster` and `_local_refresh` fully in C++

**Goal:** Move the last Python-only code path (`_build_cones_for_cluster`)
into C++ so that `_local_refresh` is a pure C++ operation with no GIL
re-acquisition.

#### New C++ helper

```cpp
// Rebuilds all F cones for cluster c from scratch using live_ids.
// Called from _local_refresh and from AMPIIndex::from_build (construction).
void AMPIIndex::_build_cones_cpp(
    int c,
    const uint32_t* live_ids, int n_live,
    const float* centroid);
```

Logic:
1. Gather centered vectors: `x_c[i] = data[live_ids[i]] - centroid`
2. Project all: `projs (F×n_live) = axes @ x_c^T` (reuse `project_data` SGEMM)
3. For each point i, find top-K cones by |normed proj|
4. For each cone f, collect its assigned points → sort by projection → `SortedCone::from_arrays`
5. Write into `cluster_cones[c]`, rebuild `point_cones` entries for live_ids

#### Checkpoint / Test

- `tests/smoke_test.py` still passes.
- `tests/test_phase5.py`: trigger a drift refresh by inserting 2000 points
  all in one direction; verify cones are rebuilt and recall is maintained.
- Check that `_local_refresh` no longer acquires the GIL (use `py::gil_scoped_release`
  on the C++ side and confirm no Python exceptions escape).

---

### Phase 6 — Thread-safety and batch API

**Goal:** Enable concurrent reads and serialised writes; add batch insert.

#### Changes

1. Add `std::shared_mutex index_mutex` to `AMPIIndex`.
   - `query()` acquires shared lock (`std::shared_lock`).
   - `add()` / `remove()` acquire exclusive lock (`std::unique_lock`).
2. Add `batch_add(data_2d_np) → int32_t[]` that takes `(m, d)` float32 and
   inserts m points, returning their global IDs.
3. Add `batch_delete(ids_np)` for bulk tombstoning before a single `_local_refresh`.

#### Checkpoint / Test

- `tests/test_phase6.py`: spawn 4 threads — 2 querying, 1 inserting, 1 deleting.
  Run for 2 seconds. Assert no crashes, no data races (run with ThreadSanitizer).
- Benchmark `batch_add` vs looped `add` at m=1000 (target: ≤ 10% overhead per
  point vs single `add`).

---

## Testing Strategy (all phases)

Every phase leaves the smoke test passing at recall@10 ≥ 0.80.  The per-phase
tests focus on:

1. **Correctness against Python baseline** — build the same index twice (Python
   path and C++ path) on the same seed, assert query results agree.
2. **Mutation consistency** — interleave add/delete/query; verify no phantom
   tombstoned IDs leak into results.
3. **Refresh correctness** — force `_local_refresh` by exceeding the tombstone
   threshold; verify cone rebuild produces valid sorted arrays.
4. **Performance regression guard** — track QPS and insert/s with a lightweight
   timing fixture; fail if C++ path is slower than Python path.

Run the full suite with:
```
python -m pytest tests/ -v
```

---

## File Change Summary

| File | Change |
|------|--------|
| `ampi/_ext.cpp` | Phases 1–6: add `best_clusters`, `best_fan_cones`, `update_drift_and_check`, `AMPIIndex` class |
| `ampi/affine_fan.py` | Phases 3–5: delegate hot paths to C++; keep Python fallback |
| `tests/test_phase1.py` | New |
| `tests/test_phase2.py` | New |
| `tests/test_phase3.py` | New |
| `tests/test_phase4.py` | New |
| `tests/test_phase5.py` | New |
| `tests/test_phase6.py` | New |
| `setup.py` | No change needed (already compiles `_ext.cpp`) |

---

## Non-Goals (deferred to later roadmap)

- WAL + checkpointing (DATABASE_PLAN.md Phase 2)
- gRPC server (DATABASE_PLAN.md Phase 3)
- B-tree / skip-list for cones > 10k points (TODO.md)
- k-means build path moved to C++ (build-time cost, not hot path)
