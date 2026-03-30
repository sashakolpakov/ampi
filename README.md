# AMPI — Adaptive Multi-Projection Index

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sashakolpakov/ampi/blob/main/demo.ipynb)

ANN index for **live, mutable vector collections**.  Insert, delete, and update
individual vectors at any time — no rebuild required.

Most ANN indexes assume the dataset is static or nearly so.  HNSW requires a full
rebuild (minutes to hours) to recover quality after significant churn, and has no
true delete — only a `mark_deleted` that leaves ghost nodes polluting every search
path.  FAISS IVF drifts silently as data shifts and eventually needs a full global
retrain.  AMPI is designed around the assumption that your data changes continuously.

**How AMPI stays fresh without rebuilding:**
- Inserts and deletes are O(1) amortised — no sorted arrays are rebuilt globally.
- Per-cluster covariance drift detection (EMA + power iteration) triggers a
  *local* cone refresh only for the affected cluster — O(N_c · F · d) vs
  O(T · n · M · d) for a full IVF retrain.
- Tombstone compaction is automatic when a cluster's delete fraction exceeds 10%.

For a full mathematical treatment see **[ALGORITHM.md](ALGORITHM.md)**.
For benchmark results see **[BENCHMARKS.md](BENCHMARKS.md)**.

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
   early stopping when the window boundary gap exceeds the current k-th distance.

**Streaming mutations** (no global rebuild at any step):

| Operation | Cost | Notes |
|---|---|---|
| `add(x)` | O(M·d + K·F·n_f) | inserts into sorted arrays in affected cluster |
| `delete(global_id)` | O(K·F) | tombstone; auto-compacts at 10% threshold |
| `update(global_id, x)` | delete + add | returns new global_id |
| local refresh | O(N_c·F·d) | triggered by drift or compaction; recomputes per-cluster axes |
| `periodic_merge(eps)` | O(nlist²) | merges centroid pairs within eps; runs automatically every `merge_interval` inserts |

### AMPIBinaryIndex

Simpler baseline: `L` random projections over the full dataset, sorted. Query collects
a window of `w` points around the query's rank on each projection, returns the union.
No partition; density-adaptive. Equivalent to AffineFan with a single cluster.

### Backend

The entire hot path is implemented in C++ via pybind11 (`ampi/_ext.cpp`):
`SortedCone`, `AMPIIndex` (owns all mutable state — data buffer, cones, drift
covariance), `add`, `remove`, `batch_add`, `batch_delete`, the full adaptive
query loop, cone build, and local refresh.  A `std::shared_mutex` allows
concurrent reads with serialised writes.  `project_data` dispatches to
`cblas_sgemm` via `ampi/_gemm.hpp` (Accelerate on macOS, OpenBLAS / MKL on
Linux/Windows, or a tiled AVX2/NEON micro-kernel fallback) — 20–112× faster
than a scalar loop.  A numba JIT fallback is used automatically when the
compiled extension is absent.

---

## Quick Start

```python
import numpy as np
from ampi import AMPIAffineFanIndex, AMPIBinaryIndex

rng  = np.random.default_rng(0)
data = rng.standard_normal((100_000, 128)).astype("float32")
q    = rng.standard_normal(128).astype("float32")

# AffineFan — main index
afan = AMPIAffineFanIndex(data, nlist=100, num_fans=128, seed=0)
pts, dists, ids = afan.query(q, k=10, window_size=150, probes=10, fan_probes=32)

# Cosine similarity (normalises data internally)
afan_cos = AMPIAffineFanIndex(data, nlist=100, num_fans=128, seed=0, metric='cosine')

# Binary — simpler baseline (no partition, no streaming mutations)
binary = AMPIBinaryIndex(data, num_projections=128, seed=0)
pts, dists, ids = binary.query(q, k=10, window_size=200)

# Streaming mutations (AffineFan only)
new_id = afan.add(rng.standard_normal(128).astype("float32"))
afan.delete(new_id)
afan.update(42, rng.standard_normal(128).astype("float32"))

# Inspect candidate pool before reranking
afan_cands   = afan.query_candidates(q, window_size=150, probes=10, fan_probes=32)
binary_cands = binary.query_candidates(q, window_size=200)   # (m,) int32 indices
```

### Parameter guide

| Parameter | Affects | Rule of thumb |
|---|---|---|
| `nlist` | cluster count | `alpha × sqrt(n)`, tune alpha ∈ [0.25, 3.0] |
| `num_fans` F | cones per cluster | largest F s.t. `n/(nlist×F) ≥ w_base` |
| `cone_top_k` K | soft assignment | K=1 fast; K=2 better recall at cluster boundaries |
| `metric` | distance function | `'l2'`/`'L2'`/`'euclidean'` (Euclidean), `'sqeuclidean'` (squared L2), `'cosine'` (normalises internally) |
| `merge_interval` | periodic merge cadence | 0 = disabled; set e.g. 1000 to merge every 1000 inserts |
| `eps_merge` | merge threshold (L2) | centroid distance below which a pair is a merge candidate |
| `probes` cp | clusters probed per query | 5–20 |
| `fan_probes` fp | cones probed per cluster | F/4 … F |
| `window_size` w | candidates per cone per axis | scales with `sqrt(n)` |

See [ALGORITHM.md §9](ALGORITHM.md) for the derivation of the scaling rules.

### Auto-tuning

```python
from ampi import AFanTuner

tuner  = AFanTuner(data, queries, gt)   # gt: (n_queries, k) true NN indices
result = tuner.tune()

idx   = result["index"]        # ready-to-use AMPIAffineFanIndex
sugg  = result["suggestions"]  # [(target_recall, cp, fp, w, cands, recall), ...]
# also available: result["nlist"], result["alpha"], result["K"], result["F"]
```

`AFanTuner` runs Gaussian-Process Bayesian optimisation over `alpha` (where
`nlist = alpha × sqrt(n)`) and `cone_top_k` ∈ {1, 2, 3}, then builds the full
index with the best params and sweeps query parameters to return a Pareto frontier
of `(probes, fan_probes, window_size)` configs — one per target recall level.

---

## Benchmark

Full results and analysis: **[BENCHMARKS.md](BENCHMARKS.md)**.

### Static recall (single build, no mutations)

AMPI is competitive with FAISS IVF on candidate efficiency and beats it on GloVe.
HNSW is faster in raw QPS — but only on a freshly built, static index.

| Dataset | IVF R@10 | IVF cands | AFan R@10 | AFan cands |
|---|---:|---:|---:|---:|
| MNIST 60k | 0.996 | 6,125 | 0.981 | **2,120** (3× fewer) |
| Fashion-MNIST 60k | 1.000 | 6,125 | 1.000 | **3,055** (2× fewer) |
| SIFT 1M | 0.992 | 50,000 | 0.990 | **39,703** (20% fewer) |
| GloVe 1.18M | 0.879 | 54,400 | **0.897** | 59,929 (wins on recall) |
| GIST 200k d=960 | 0.982 | 22,350 | 0.964 | **16,760** (25% fewer) |

### The rebuild problem

HNSW's headline QPS numbers assume a freshly-built, never-modified index.
In a live system, this assumption breaks quickly:

| Operation | HNSW | FAISS IVF | AMPI |
|---|---|---|---|
| Insert | graph degrades silently | O(1) append, no quality loss | O(1) amortised, guaranteed membership |
| Delete | `mark_deleted` only — ghost nodes pollute graph | unsupported or full scan | O(1) tombstone, auto-compacted |
| Update | mark + insert, compounding both problems | unsupported | delete + insert |
| Recovery from drift | **full rebuild** (4–6 min at 1M vectors) | **full global retrain** | per-cluster local refresh only |

HNSW's speed is real — if you build once and never touch it again.  If your dataset
has any churn, you are either accepting silent quality degradation or paying full
rebuild costs on a schedule.

### Running the benchmarks

Dataset files are downloaded automatically on first run.

```bash
python benchmarks/benchmark_vs_faiss.py all   # vs FAISS Flat L2 + IVF
python benchmarks/benchmark_vs_hnsw.py all    # vs hnswlib HNSW M=16
python benchmarks/benchmark_vs_faiss.py all --force  # re-download data
```

**Datasets** (downloaded automatically from [ann-benchmarks.com](http://ann-benchmarks.com)):

| Key | Description | Size |
|---|---|---|
| `gauss` | Synthetic N(0,1), n=10k, d=128 | — (generated) |
| `mnist` | MNIST digits, n=60k, d=784, Euclidean | ~55 MB |
| `fashion` | Fashion-MNIST, n=60k, d=784, Euclidean | ~55 MB |
| `sift` | SIFT descriptors, n=1M, d=128, Euclidean | ~350 MB |
| `glove` | GloVe Twitter, n=1.18M, d=100, cosine | ~500 MB |
| `gist` | GIST, n=1M, d=960, Euclidean (high-d stress test; benchmarked at 200k) | ~3.6 GB |

To download manually or inspect which files are present:

```bash
python benchmarks/download_data.py           # download all (~1 GB)
python benchmarks/download_data.py mnist     # specific dataset
python benchmarks/download_data.py --list    # check status without downloading
```

---

## Installation

**Requirements**: Python ≥ 3.8, a C++17 compiler (gcc ≥ 9 / clang ≥ 10 / MSVC 2019),
and `pybind11` (installed automatically by the build step below).

### From source

```bash
git clone https://github.com/sashakolpakov/ampi.git
cd ampi
pip install -e .
```

The C++ extension (`ampi/_ext.cpp`) is compiled automatically by setuptools during
install. If the build fails, the package falls back to a numba JIT implementation
transparently — no extra action needed.

### Dependencies

| Package | Purpose | Installed automatically |
|---|---|---|
| `numpy ≥ 1.24` | array operations | yes |
| `numba ≥ 0.57` | JIT fallback kernels | yes |
| `pybind11 ≥ 2.11` | C++ extension build | yes (build-only) |
| `faiss-cpu` | benchmark baseline | no |
| `h5py` | loading `.hdf5` datasets | no |
| `matplotlib` | Pareto frontier plots | no |

To install benchmark extras:

```bash
pip install faiss-cpu h5py matplotlib
```

---

## Repository Layout

```
ampi/
├── ampi/
│   ├── __init__.py
│   ├── _kernels.py       # C++ ext wrapper + numba fallback
│   ├── _ext.cpp          # pybind11 C++ kernels + SortedCone class
│   ├── _gemm.hpp         # portable SGEMM dispatcher (Accelerate/OpenBLAS/MKL/AVX2/NEON)
│   ├── affine_fan.py     # AMPIAffineFanIndex (streaming insert/delete/update)
│   ├── binary.py         # AMPIBinaryIndex
│   ├── tuner.py          # AFanTuner (GP-BO over alpha, Pareto knee detection)
│   └── README.md         # package-level pointer to this document
├── benchmarks/
│   ├── benchmark_vs_faiss.py  # recall@1/10/100 vs FAISS (Flat L2 + IVF)
│   ├── benchmark_vs_hnsw.py   # recall@1/10/100 vs hnswlib
│   ├── _bench_common.py       # shared dataset loaders, evaluation, AMPI builder
│   ├── download_data.py            # dataset downloader (auto-called by benchmarks)
│   ├── profile_drift_threshold.py  # per-cluster fan-axis variance profiler
│   └── _bench_sgemm.py             # project_data microbenchmark (scalar loop vs SGEMM)
├── tests/
│   ├── smoke_test.py     # fast unit test, no datasets needed
│   └── stress_test.py    # adversarial add/delete/update/churn scenarios
├── figures/              # Pareto frontier plots saved by benchmark scripts (gitignored)
├── .github/workflows/
│   ├── ci.yml            # CI: lint + smoke test on push
│   └── cpp-verify.yml    # CI: cppcheck + two-pass LLM review on every push
├── demo.ipynb            # interactive walkthrough
├── setup.py              # C++ extension build + BLAS detection
├── pyproject.toml        # project metadata and dependencies
├── ALGORITHM.md          # full mathematical algorithm description
├── BENCHMARKS.md         # full benchmark results (FAISS + hnswlib)
├── DATABASE_PLAN.md      # phased implementation plan (persistence + distributed DB)
├── TODO.md               # task tracking
└── LICENSE               # MIT
```

## Roadmap

The full hot path is in C++: routing, drift EMA, mutable index state, query loop,
`batch_add`/`batch_delete`, and concurrent reader/writer locking (`std::shared_mutex`).

Near-term milestones:

- **Persistence** — WAL + checkpoint serializer for single-node durability.
- **Distributed** — coordinator + multi-shard query fan-out, cluster splits, rebalancing.

See `DATABASE_PLAN.md` for the persistence/distributed plan and `TODO.md` for current
task status.
