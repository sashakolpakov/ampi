# AMPI — Adaptive Multi-Projection Index

A lightweight, pure-Python/Numba library for approximate nearest-neighbour (ANN) search,
implementing three complementary projection-based algorithms with a unified API.

```bash
pip install -e .          # from repo root
```

```python
from ampi import AMPIBinaryIndex, AMPIHashIndex, AMPISubspaceIndex

idx = AMPIBinaryIndex(data, num_projections=16)
idx = AMPIHashIndex(data, num_projections=16, bucket_size=1.0)
idx = AMPISubspaceIndex(data, num_projections=16, bins_per_axis=16, subspace_dim=2)

points, dists, indices = idx.query(q, k=10)
candidates              = idx.query_candidates(q)   # before re-ranking
```

---

## 1. Introduction

Exact nearest-neighbour search costs O(nd) per query, prohibitive for large n or high d.
Projection-based ANN methods reduce this by hashing or sorting data onto low-dimensional
projections, then re-ranking a small candidate pool by exact L2 distance.

AMPI provides three such algorithms under one API, with no external C dependencies:
only NumPy and Numba (JIT-compiled inner loops). The three backends offer different
precision–recall–latency tradeoffs and suit different deployment contexts.

---

## 2. Algorithms

### 2.1 AMPI Binary — Sorted Projections

**Build.**
Draw L random unit vectors a₁, …, aL ∈ ℝᵈ. For each direction i, project all n data
points and sort:

    vᵢ(x) = aᵢ · x
    sorted_projs[i] = argsort(vᵢ(x₁), …, vᵢ(xₙ))

Memory: O(Ln).

**Query.**
Project q onto each direction, binary-search for its rank, take the w nearest points on
each side:

    Cᵢ = { x_(r−w), …, x_(r+w) },   r = searchsorted(vᵢ(q))

Return the union C = ⋃ᵢ Cᵢ (at most 2wL candidates) and re-rank by exact ℓ².

**Key property — density adaptivity.**
The window always contains exactly 2w points per projection regardless of local data
density. In dense regions the window is geometrically narrow (high precision); in sparse
regions it automatically extends further (high recall). No threshold to calibrate.

**Theoretical guarantee.**
Let r = ‖q − x*‖₂. The projection difference δ = aᵢ · (x* − q) ~ N(0, r²/d).
Let Δ_w be the projection-space half-width of a w-point window at q's rank position.
Capture probability for the nearest neighbour:

    P(capture | projection i)  =  2Φ(Δ_w √d / r) − 1

    P(capture | L projections) =  1 − (1 − P_single)^L

Δ_w grows linearly with w up to the local point spacing, so recall increases
monotonically in both w and L. Density adaptivity implies Δ_w is automatically large in
sparse regions and small in dense regions — no parameter calibration needed.

---

### 2.2 AMPI Hashing — p-stable LSH

Implements the p-stable hash of Datar et al. (2004) with multi-probe extension (Lv et al., 2007).

**Hash function.**

    h_{a,b,w}(x) = floor( (a · x + b) / w )

where a ~ N(0, Iᵈ), unit-normalised, and b ~ Uniform[0, w).

**Collision probability** (Datar et al., 2004):

    p(r, w) = 2Φ(w√d / r) − 1  −  (2r / (w√(2πd))) · (1 − exp(−w²d / (2r²)))

where Φ is the standard normal CDF. This is strictly decreasing in r: near points always
collide with higher probability than far points.

**Multi-probe.**
With probe radius p, check the (2p−1) adjacent buckets [h(q)−(p−1), …, h(q)+(p−1)]
per projection. Effective window half-width in projection space: (p−1)·w.

Approximate capture probability for the nearest neighbour at distance r:

    P_single ≈ 2Φ((2p−1)·w·√d / (2r)) − 1

Amplification over L independent hash functions:

    P_captured = 1 − (1 − P_single)^L

**Formal (c, r)-ANN guarantee** (Datar et al., 2004).
For any c > 1, there exist parameters (w, L) such that the data structure answers
(c, r)-ANN queries in time O(nᵖ · d) with constant probability, where

    ρ = log(1/p₁) / log(1/p₂) < 1,    p₁ = p(r, w),  p₂ = p(cr, w)

For L2 in ℝᵈ, ρ ≤ 1/c² asymptotically (Andoni & Indyk, 2006). The space requirement
is O(n^(1+ρ)).

**Calibration rule.**
Set `bucket_size` w ≈ r_NN / √d (= σ_proj, the projection std of the nearest-neighbour
difference). For N(0,1) data in d = 128 with n = 10⁵: σ_proj ≈ 1.12.
For MNIST (d = 784): σ_proj ≈ 0.18.
Below ~0.1·σ_proj recall collapses; above σ_proj it saturates — see Figure 3.

---

### 2.3 AMPI Subspace — Multi-dimensional Subspace Hashing

A 1-D projection collapses all d dimensions onto a line, discarding all structure
perpendicular to it. Projecting onto a d_sub-dimensional random subspace (d_sub = 2, 3, 4)
retains more geometry per hash slot.

**Build.**
Draw L orthonormal frames Uᵢ ∈ ℝ^(d × d_sub) (QR decomposition of Gaussian matrices).
For each frame i and each axis ax, project all n points:

    v_{i,ax}(x) = U_{i,·,ax} · x

Quantise each axis independently into B equal-frequency bins (data-adaptive: bin widths
narrow in dense regions, widen in sparse ones). Encode the d_sub bin indices as one key:

    key = b₁·B^(d_sub−1) + b₂·B^(d_sub−2) + … + b_{d_sub}

One hash table per frame: key → array of data indices.

**Query.**
Project q onto each frame's d_sub axes, find q's grid cell, probe all (2p−1)^{d_sub}
cells in the ℓ∞ ball of radius p−1. Collect the union across L frames; re-rank by ℓ².

**False-positive rate.**
A random far point lands in the same d_sub-dimensional cell with probability

    p₂ = 1 / B^{d_sub}

This decreases exponentially with d_sub. For B = 16, d_sub = 2: p₂ = 1/256;
for d_sub = 3: p₂ = 1/4096.

**Capture probability.**
Each axis ax contributes projection difference δ_ax ~ N(0, r²/d). With bin width Δ_ax:

    P_frame ≈ ∏_{ax} P(|δ_ax| < Δ_ax/2)

    P_captured = 1 − (1 − P_frame)^L

The equal-frequency quantisation ensures balanced bins regardless of the data distribution,
giving well-calibrated precision and avoiding the empty-bucket pathology of fixed-width hashing.

**ρ exponent.**
The LSH ρ exponent ρ = log(1/p₁)/log(1/p₂) with p₁ = P_frame, p₂ = 1/B^{d_sub}.
Both p₁ and p₂ decrease as d_sub increases, keeping ρ roughly constant asymptotically.
In practice the adaptivity of equal-frequency bins gives noticeably better bucket balance
than fixed-width hashing at the same total bucket count B^{d_sub}.

**Probe cost** scales as (2p−1)^{d_sub} per frame; keep p ≤ 3 for d_sub ≥ 3.

---

## 3. Empirical Results

Benchmarks on two datasets, 300 held-out query points, Recall@10 reported.
All AMPI indices use L = 16 projections. Hashing uses bucket_size = σ_proj (empirically
measured). Subspace uses B = 16 bins/axis (d_sub = 2) or B = 8 (d_sub = 3).
FAISS IVF (nlist = 1 % of n) is the competitive baseline.

| Dataset | n | d | σ_proj |
|---|---|---|---|
| Synthetic Gaussian | 100 000 | 128 | 1.12 |
| MNIST (pixels/255) | 70 000  | 784 | 0.18 |

### Figure 1 — Recall@10 vs Candidates Examined

![Recall vs Candidates](figures/fig1_recall_vs_cands.png)

On MNIST (right), Subspace (d=2, d=3) reaches Recall@10 > 0.99 examining 28–52 k
candidates — well ahead of Binary at the same budget, and with a clear trade-off curve.
On synthetic Gaussian data (left) — the hardest possible benchmark (no cluster structure,
nearly uniform distance distribution) — IVF dominates the low-candidate regime. All
projection methods must examine a substantial fraction of the corpus to achieve high
recall, which is expected: at d = 128 with iid Gaussian data the gap between nearest and
median neighbour distances is narrow and hard to exploit by projection alone.

### Figure 2 — Recall@10 vs Query Latency

![Recall vs Time](figures/fig2_recall_vs_time.png)

On MNIST, Subspace(d=2,3) delivers > 0.99 recall in ~30 ms/query (single-threaded
Python/Numba). FAISS IVF reaches 0.97 recall in < 1 ms — the cost of C optimisation and
inverted-index structure. The latency gap is expected; AMPI targets transparency and
a self-contained implementation over raw throughput.

### Figure 3 — Bucket-size Calibration (Hashing)

![Calibration](figures/fig3_calibration.png)

Recall transitions sharply near bucket_size ≈ σ_proj (dotted vertical line) on both
datasets. Below 0.1·σ_proj the bucket is too narrow and many probes fail to capture the
nearest neighbour. Above σ_proj recall saturates quickly with probes = 3, at the cost of
a larger candidate set. The rule `bucket_size ≈ r_NN / √d` is empirically confirmed.

### Figure 4 — Subspace Dimension

![Subspace Dimension](figures/fig4_subspace_dim.png)

On MNIST (right), d_sub = 3 reaches Recall@10 ≈ 0.48 with only ~2 300 candidates
(p = 1), versus 0.11 for d_sub = 1 at a similar budget. The d_sub-dimensional projection
retains more geometry per hash slot, reducing false positives (1/B^{d_sub} vs 1/B).
On the harder Gaussian benchmark (left) the advantage is present but more modest.

---

## 4. Method Comparison

| Property | Binary | Hashing | Subspace |
|---|---|---|---|
| Projection dim | 1 | 1 | d_sub (2–4) |
| Bucket type | sorted-order window | fixed-width range | d_sub-dim grid |
| Density adaptation | yes — window = fixed count | no — window = fixed range | partial — equal-freq bins |
| Formal (c,r)-ANN guarantee | — | yes (Datar 2004) | analogous ρ bound |
| False-positive rate | ∝ local density | 1/B | 1/B^{d_sub} |
| Probe cost | O(wL) | O((2p−1)L) | O((2p−1)^{d_sub} L) |
| Best suited for | heterogeneous data | theoretical guarantees | high-d structured data |
| Key calibration param | window_size w | bucket_size (set ≈ σ_proj) | bins_per_axis B, subspace_dim |

---

## 5. Installation

```bash
git clone <repo>
cd ampi
pip install -e .
```

**Dependencies**: `numpy >= 1.20`, `numba >= 0.55`. No C/C++ compilation.

**Optional** (benchmarks and demo only): `faiss-cpu`, `matplotlib`.

```bash
python bench_figures.py   # reproduces all figures above
python benchmark.py       # detailed degradation curves
python compare.py         # cross-method comparison tables
```

---

## 6. Quick Start

```python
import numpy as np
from ampi import AMPIBinaryIndex, AMPIHashIndex, AMPISubspaceIndex

rng  = np.random.default_rng(0)
data = rng.standard_normal((50_000, 128)).astype("float32")
q    = rng.standard_normal(128).astype("float32")

# Binary: adaptive, no calibration needed
idx = AMPIBinaryIndex(data, num_projections=16)
pts, dists, ids = idx.query(q, k=10, window_size=100)

# Hashing: calibrate bucket_size ≈ r_NN/sqrt(d)
idx = AMPIHashIndex(data, num_projections=16, bucket_size=0.9)
pts, dists, ids = idx.query(q, k=10, probes=2)

# Subspace: higher precision via 2D grid
idx = AMPISubspaceIndex(data, num_projections=16, bins_per_axis=16, subspace_dim=2)
pts, dists, ids = idx.query(q, k=10, probes=2)

# Inspect the candidate pool before re-ranking
cands = idx.query_candidates(q, probes=2)   # (m,) int32 indices
print(f"{len(cands)} candidates → top-10 by exact L2")
```

---

## 7. Repository Layout

```
ampi/
├── pyproject.toml          # pip install -e .
├── README.md
├── bench_figures.py        # reproduces all four figures
├── benchmark.py            # degradation curves, all methods
├── compare.py              # cross-method tables
├── demo.ipynb              # interactive notebook (MNIST + optional PyKeOps)
├── data/MNIST/raw/         # MNIST IDX files
├── figures/                # output of bench_figures.py
└── ampi/
    ├── __init__.py
    ├── _kernels.py         # shared Numba JIT: project_data, l2_distances
    ├── binary.py           # AMPIBinaryIndex
    ├── hashing.py          # AMPIHashIndex
    └── subspace.py         # AMPISubspaceIndex
```

---

## 8. References

- Datar, Immorlica, Indyk, Mirrokni (2004). *Locality-sensitive hashing scheme based on
  p-stable distributions.* SCG '04.
- Lv, Josephson, Wang, Charikar, Li (2007). *Multi-probe LSH: efficient indexing for
  high-dimensional similarity search.* VLDB '07.
- Andoni, Indyk (2006). *Near-optimal hashing algorithms for approximate nearest neighbor
  in high dimensions.* FOCS '06.
- Johnson, Lindenstrauss (1984). *Extensions of Lipschitz maps into a Hilbert space.*
  Contemporary Mathematics.
