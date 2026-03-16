# AMPI — Affine Multi-Projection Index

Approximate nearest-neighbour search via k-means partition + random affine fan cones
+ sorted-projection candidate selection.  Supports streaming `add` / `delete` /
`update` without a full rebuild.

```bash
pip install -e .
```

For a full mathematical treatment of the algorithm see **[ALGORITHM.md](ALGORITHM.md)**.

---

## Algorithm overview

### AMPIAffineFanIndex

The main index. Three-level structure:

1. **k-means partition** — mini-batch Lloyd's (BLAS-accelerated) partitions data into
   `nlist` clusters, each with centroid μ_c.
2. **Affine fan cones** — within each cluster, `F` random unit vectors define projection
   axes. Each data point is assigned to its top-K most-aligned cones (by
   `|〈x−μ_c, aₗ〉| / ‖x−μ_c‖`). Cone membership is stored as F sorted arrays of
   (projection value, global id) pairs.
3. **Sorted-projection query** — project the query, probe `cp` nearest clusters,
   select `fp` best-aligned cones per cluster, collect candidates from a window of
   `w` entries around the query's projection rank on each axis, L2-rerank the union.
   A Cauchy-Schwarz coverage certificate (`‖x−q‖ ≥ |proj_a(x) − proj_a(q)|`) allows
   early stopping: when the window boundary gap on any axis exceeds the current k-th
   distance, no unvisited point can improve the result.

**Streaming mutations** are fully supported without rebuilding the global index:

- `add(x)` — insert one vector; returns its `global_id`.
- `delete(global_id)` — logical tombstone; auto-compacts when the cluster's tombstone
  fraction exceeds 10%.
- `update(global_id, x)` — delete + insert.

Per-cluster drift detection (covariance EMA + power iteration) triggers a local cone
refresh when the dominant displacement direction rotates > 15° from all fan axes.

### AMPIBinaryIndex

Simpler baseline: `L` random projections over the full dataset, sorted. Query collects
a window of `w` points around the query's rank on each projection, returns the union.
No partition; density-adaptive (window always covers exactly `2w` points per projection
regardless of local density). Equivalent to AffineFan with a single cluster.

### Backend

Hot-path kernels (`project_data`, `l2_distances`, `union_query`) and the mutable
`SortedCone` data structure are implemented in C++ via pybind11 (`ampi/_ext.cpp`).
A numba JIT fallback is used automatically when the compiled extension is absent.

---

## Quick Start

```python
import numpy as np
from ampi import AMPIAffineFanIndex, AMPIBinaryIndex

rng  = np.random.default_rng(0)
data = rng.standard_normal((100_000, 128)).astype("float32")
q    = rng.standard_normal(128).astype("float32")

# AffineFan — main index
idx = AMPIAffineFanIndex(data, nlist=100, num_fans=128, seed=0)
pts, dists, ids = idx.query(q, k=10, window_size=150, probes=10, fan_probes=32)

# Binary — simpler baseline
idx = AMPIBinaryIndex(data, num_projections=128, seed=0)
pts, dists, ids = idx.query(q, k=10, window_size=200)

# Streaming mutations
new_id = idx.add(rng.standard_normal(128).astype("float32"))
idx.delete(new_id)
idx.update(42, rng.standard_normal(128).astype("float32"))

# Inspect candidate pool before reranking
cands = idx.query_candidates(q, window_size=200)   # (m,) int32 indices
```

### Parameter guide

| Parameter | Affects | Rule of thumb |
|---|---|---|
| `nlist` | cluster count | `alpha × sqrt(n)`, tune alpha ∈ [0.25, 3.0] |
| `num_fans` F | cones per cluster | largest F s.t. `n/(nlist×F) ≥ w_base` |
| `cone_top_k` K | soft assignment | K=1 fast; K=2 better recall at cluster boundaries |
| `probes` cp | clusters probed per query | 5–20 |
| `fan_probes` fp | cones probed per cluster | F/4 … F |
| `window_size` w | candidates per cone per axis | scales with `sqrt(n)` |

See [ALGORITHM.md §9](ALGORITHM.md) for the derivation of the scaling rules.

### Auto-tuning

```python
from ampi import AFanTuner

tuner  = AFanTuner(data, queries, gt)   # gt: (n_queries, k) true NN indices
result = tuner.tune()

idx  = result["index"]        # ready-to-use AMPIAffineFanIndex
sugg = result["suggestions"]  # [(target_recall, cp, fp, w, cands, recall), ...]
```

`AFanTuner` runs Gaussian-Process Bayesian optimisation over `(alpha, cone_top_k)`,
maximising recall@10 subject to a QPS target, then returns the full Pareto frontier.

---

## Benchmark

SIFT-1M (n=1M, d=128) and MNIST (n=60k, d=784), 200 held-out queries, Recall@10.

| Method | Recall@10 | Recall@100 | Candidates |
|---|---|---|---|
| IVF nprobe=50 (SIFT) | 0.994 | 0.986 | 50,000 |
| AFan K=1 cp=10 fp=128 (SIFT) | 0.992 | 0.984 | 109,468 |
| AFan K=1 cp=20 fp=128 (SIFT) | 0.999 | 0.999 | 209,880 |
| IVF nprobe=50 (MNIST) | 0.998 | 0.999 | 12,250 |
| AFan K=1 cp=20 fp=64 (MNIST) | 0.994 | 0.991 | 10,139 |

AFan matches IVF recall on both datasets. On MNIST it does so with *fewer* candidates
than IVF. On SIFT it requires ~2× more candidates.

Both support cheap inserts without a global rebuild. As the data distribution shifts,
IVF eventually needs a full global retrain (O(T·n·M·d)), while AFan triggers only
per-cluster local refreshes (O(N_c·F·log N_c)) — see [ALGORITHM.md §10](ALGORITHM.md)
for the formal comparison.

```bash
python benchmark.py sift
python benchmark.py mnist
python benchmark.py all
```

Benchmark output includes Recall@1, Recall@10, Recall@100, QPS, and distance ratio for
each method, with Pareto frontier plots saved to `figures/`.

---

## Installation

```bash
git clone <repo>
cd ampi
pip install -e .
```

The C++ extension is built automatically by setuptools (requires a C++17 compiler and
`pybind11`). If the build fails, the package still works via the numba fallback.

**Dependencies**: `numpy`, `numba`. Benchmarks additionally need `faiss-cpu`, `h5py`, `matplotlib`.

---

## Repository Layout

```
ampi/
├── ampi/
│   ├── __init__.py
│   ├── _kernels.py       # C++ ext wrapper + numba fallback
│   ├── _ext.cpp          # pybind11 C++ kernels + SortedCone class
│   ├── affine_fan.py     # AMPIAffineFanIndex (streaming insert/delete/update)
│   ├── binary.py         # AMPIBinaryIndex
│   └── tuner.py          # AFanTuner (GP-BO over alpha, Pareto knee detection)
├── tests/
│   ├── smoke_test.py     # fast unit test, no datasets needed
│   └── stress_test.py    # adversarial add/delete/update/churn scenarios
├── benchmark.py          # recall@1/10/100 vs FAISS IVF
├── ALGORITHM.md          # full mathematical algorithm description
├── DATABASE_PLAN.md      # phased implementation plan (persistence + distributed DB)
├── TODO.md               # task tracking
└── pyproject.toml
```

## Roadmap

Phase 1 (streaming insert/delete/update) is complete. The next milestones are:

- **Phase 2** — persistence: WAL + checkpoint serializer for single-node durability.
- **Phase 3** — distributed: coordinator + multi-shard query fan-out, cluster splits,
  rebalancing.

See `DATABASE_PLAN.md` for the phased implementation plan and `TODO.md` for current
task status.
