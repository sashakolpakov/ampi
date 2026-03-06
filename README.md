# AMPI — Affine Multi-Projection Index

Approximate nearest-neighbour search via k-means partition + random affine fan cones
+ sorted-projection candidate selection.

```bash
pip install -e .
```

---

## Algorithms

### AMPIAffineFanIndex

The main index. Three-level structure:

1. **k-means partition** — coarse assignment of data points to `nlist` clusters.
2. **Affine fan cones** — within each cluster, `F` random unit vectors define cones
   around the centroid. Each data point is assigned to its top-K most-aligned cones
   (`cone_top_k`). Cone membership is recorded as sorted projection arrays.
3. **Sorted-projection query** — project the query, probe `cp` nearest clusters,
   select the `fp` best-aligned cones per cluster, collect candidates from a window
   of size `w` around the query's projection rank, L2-rerank.

Key insight: with `nlist` small enough that each cluster has O(n/nlist) points and
`F` cones per cluster, the effective per-query scan is `cp × fp × (n / (nlist × F))`
— sublinear in n while maintaining high recall.

### AMPIBinaryIndex

Simpler baseline: `L` random projections over the full dataset, sorted.
Query collects a window of size `w` around the query rank on each projection, returns
the union. No partition, density-adaptive (window always covers exactly `2w` points per
projection regardless of local density).

### Backend

Hot-path kernels (`project_data`, `l2_distances`, `union_query`) are implemented in
C++ via pybind11 (`ampi/_ext.cpp`). A numba JIT fallback is used automatically if the
compiled extension is not present.

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

# Inspect candidate pool before reranking
cands = idx.query_candidates(q, window_size=200)   # (m,) int32 indices
```

### Parameter guide

| Parameter | Affects | Rule of thumb |
|---|---|---|
| `nlist` | cluster count | `alpha × sqrt(n)`, tune alpha ∈ [0.1, 2.0] |
| `num_fans` F | cones per cluster | largest F s.t. `n/(nlist×F) ≥ w` |
| `cone_top_k` K | soft assignment | K=1 fast, K=2 better recall at boundaries |
| `probes` cp | clusters probed per query | 5–20 |
| `fan_probes` fp | cones probed per cluster | F/4 … F |
| `window_size` w | candidates per cone | scales with `sqrt(n)` |

### Auto-tuning

```python
from ampi import AFanTuner

tuner  = AFanTuner(data, queries, gt)   # gt: (n_queries, k) true NN indices
result = tuner.tune()

idx  = result["index"]        # ready-to-use AMPIAffineFanIndex
sugg = result["suggestions"]  # [(target_recall, cp, fp, w, cands, recall), ...]
```

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

AFan matches IVF recall quality on both datasets. On MNIST it does so with *fewer*
candidates than IVF. On SIFT it requires more candidates (~2×) but provides a
flexible, dynamically-updatable structure (no global Voronoi rebuild for inserts).

Run benchmarks:

```bash
python benchmark.py sift
python benchmark.py mnist
python benchmark.py all
```

Benchmark output includes Recall@1, Recall@10, Recall@100, QPS, and distance ratio
for each method, with per-family Pareto frontier plots saved to `figures/`.

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
│   ├── _ext.cpp          # pybind11 C++ kernels
│   ├── affine_fan.py     # AMPIAffineFanIndex
│   ├── binary.py         # AMPIBinaryIndex
│   └── tuner.py          # AFanTuner (GP-BO over alpha, Pareto knee detection)
├── tests/
│   └── smoke_test.py
├── benchmark.py          # recall@1/10/100 curves vs FAISS IVF
├── DATABASE_PLAN.md      # phased implementation plan for streaming insert + distributed DB
├── TODO.md               # architecture notes and task tracking
└── pyproject.toml
```

## Roadmap

The next major milestone is streaming insertion (no full rebuild on `add`/`delete`)
and a distributed multi-shard architecture. See `DATABASE_PLAN.md` for the phased
implementation plan, and `TODO.md` for current task status.
