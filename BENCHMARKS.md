# AMPI Benchmark Results

Recall@1 / Recall@10 / Recall@100, QPS, and candidate counts across all datasets.
Ground truth computed via exact brute-force (BLAS gemm, no FAISS).
All runs: 200 held-out queries, single-threaded query path, macOS, Apple M-series.

## What these benchmarks measure — and what they don't

The numbers below measure **static recall**: one build, no mutations, queries on the
original corpus.  This is the fairest comparison for algorithmic efficiency, but it
favours HNSW and IVF whose designs assume the dataset never changes.

**HNSW caveats (not reflected in the numbers below):**
- Insertions after the initial build degrade graph quality — newly added nodes receive
  fewer connections than nodes built into the index, and the graph is never rebalanced.
- Deletions are not supported. `mark_deleted()` leaves ghost nodes in the graph that
  participate in every traversal and are filtered only at result-return time.
- Recovery from significant churn requires a **full rebuild**: 34 s (MNIST 60k),
  60 s (Fashion-MNIST 60k), 249 s (SIFT 1M), 351 s (GloVe 1.18M) — all
  single-index, 8 CPU cores, Apple M-series.
- There is no partial repair mechanism.

**FAISS IVF caveats:**
- Insertions are O(1) but the Voronoi boundaries computed at build time become
  inaccurate as data drifts, silently degrading recall.
- Deletes are not supported natively.
- Recovery requires a full global retrain: O(T · n · M · d).

**AMPI in a live system:**
- Inserts and deletes are O(1) amortised with guaranteed correctness.
- Drift triggers a per-cluster local refresh — only the affected cluster is rebuilt,
  not the full index.
- There is no scheduled rebuild; the index self-maintains.

The QPS gap between AMPI and HNSW reflects AMPI's current Python query path — the
inner kernels are in C++, but the query orchestration is not yet.  Candidate count
is the implementation-neutral efficiency metric; see the "Notes" section.

---

## vs FAISS

Baselines: **Flat L2** (exact brute-force) and **IVF** (FAISS IndexIVFFlat,
`nlist = sqrt(n)`).  AMPI variants: **Binary** (sorted-projection baseline) and
**AFan** (AffineFan index, K=1 hard assignment / K=2 soft assignment).

### Gaussian 10k  ·  d=128  ·  Euclidean

> Structureless dataset — all points are roughly equidistant from every query.
> Numbers are not indicative of real-world performance; included only as a
> sanity check (Flat L2 should score 1.0).

| Method | R@1 | R@10 | R@100 | QPS | Cands |
|---|---:|---:|---:|---:|---:|
| Flat L2 | 1.000 | 1.000 | 1.000 | 4,104 | n |
| IVF nprobe=25 | 0.570 | 0.595 | 0.548 | 10,174 | 2,500 |
| AFan K=1 cp=20 fp=16 w=5 | 0.450 | 0.442 | 0.374 | 1,993 | 1,643 |
| AFan K=2 cp=20 fp=16 w=5 | 0.450 | 0.442 | 0.374 | 1,995 | 1,643 |

---

### MNIST 60k  ·  d=784  ·  Euclidean

| Method | R@1 | R@10 | R@100 | QPS | Cands |
|---|---:|---:|---:|---:|---:|
| Flat L2 | 1.000 | 1.000 | 1.000 | 136 | n |
| IVF nprobe=25 | 1.000 | **0.996** | 0.994 | **1,310** | 6,125 |
| AFan K=1 cp=10 fp=32 w=9 | 0.980 | **0.981** | 0.941 | 329 | **2,120** |
| AFan K=1 cp=5 fp=32 w=9 *(best QPS @ R@10 ≥ 0.90)* | 0.950 | 0.928 | — | **558** | 1,012 |
| AFan K=2 cp=10 fp=32 w=9 | 0.980 | 0.981 | 0.941 | 323 | 2,120 |

AFan K=1 reaches R@10=0.981 with **2,120 candidates** vs IVF's 0.996 at 6,125 —
**~3× fewer candidates** for near-identical recall.  IVF is faster per query due
to FAISS's highly optimised C++ inner loop.

---

### Fashion-MNIST 60k  ·  d=784  ·  Euclidean

| Method | R@1 | R@10 | R@100 | QPS | Cands |
|---|---:|---:|---:|---:|---:|
| Flat L2 | 1.000 | 1.000 | 1.000 | 151 | n |
| IVF nprobe=25 | 1.000 | **1.000** | 0.999 | **1,270** | 6,125 |
| AFan K=1 cp=20 fp=16 w=9 | 1.000 | **1.000** | 0.994 | 221 | **3,055** |
| AFan K=1 cp=10 fp=16 w=18 *(best QPS @ R@10 ≥ 0.90)* | 1.000 | 0.992 | — | **411** | 1,540 |
| AFan K=2 cp=20 fp=16 w=9 | 1.000 | 1.000 | 0.994 | 224 | 3,055 |

Perfect recall on both IVF and AFan.  AFan achieves this with **~2× fewer candidates**.

---

### SIFT 1M  ·  d=128  ·  Euclidean

| Method | R@1 | R@10 | R@100 | QPS | Cands |
|---|---:|---:|---:|---:|---:|
| Flat L2 | 1.000 | 0.999 | 1.000 | 40 | n |
| IVF nprobe=50 | 0.995 | **0.992** | 0.986 | **638** | 50,000 |
| AFan K=1 cp=50 fp=64 w=37 | 0.990 | **0.990** | 0.973 | 60 | **39,703** |
| AFan K=1 cp=20 fp=64 w=37 *(best QPS @ R@10 ≥ 0.90)* | 0.960 | 0.931 | — | **130** | 17,113 |
| AFan K=2 cp=50 fp=64 w=37 | 0.990 | 0.990 | 0.973 | 59 | 39,703 |

AFan matches IVF recall (0.990 vs 0.992) with **~20% fewer candidates**.
IVF has a large QPS advantage (638 vs 60) due to FAISS's native BLAS path —
this gap will narrow as AMPI's C++ query path matures.

---

### GloVe 1.18M  ·  d=100  ·  Cosine

| Method | R@1 | R@10 | R@100 | QPS | Cands |
|---|---:|---:|---:|---:|---:|
| Flat L2 | 1.000 | 1.000 | 1.000 | 39 | n |
| IVF nprobe=50 | 0.905 | 0.879 | 0.820 | **470** | 54,400 |
| AFan K=1 cp=50 fp=64 w=40 | **0.925** | **0.897** | **0.850** | 42 | 59,929 |
| AFan K=2 cp=50 fp=64 w=40 | 0.925 | 0.897 | 0.850 | 38 | 59,929 |

**AFan outperforms IVF on GloVe** across all recall metrics (R@1: +2%, R@10: +1.8%,
R@100: +3%) while using a similar candidate budget.  Cosine similarity with
non-uniform data distribution appears to favour the affine fan geometry over
IVF's Voronoi partition.

---

## vs hnswlib

HNSW parameters: M=16, ef\_construction=200, ef swept over [10, 20, 50, 100, 200, 400, 800].
Build uses all available CPU cores; query is single-threaded.

> **Note on x-axis:** HNSW's `ef` is a *control parameter* (dynamic candidate list size
> during graph traversal), not a direct count of distance computations.  It is not
> comparable to AMPI's candidate counts.  **QPS vs Recall is the meaningful comparison.**

### Gaussian 10k  ·  d=128

Not meaningful (structureless dataset). HNSW build: 1.9 s.

---

### MNIST 60k  ·  d=784  ·  HNSW build: 34 s

| Method | R@1 | R@10 | R@100 | QPS |
|---|---:|---:|---:|---:|
| HNSW ef=10 | 1.000 | **0.999** | 0.990 | **662** |
| HNSW ef=200 | 1.000 | 1.000 | 0.998 | 487 |
| HNSW ef=400 | 1.000 | 1.000 | 1.000 | 295 |
| AFan K=1 cp=50 fp=16 w=9 | 1.000 | **1.000** | — | 192 |
| AFan K=1 cp=5 fp=16 w=9 *(best QPS @ R@10 ≥ 0.90)* | 1.000 | 0.900 | — | **410** |

HNSW ef=10 achieves near-perfect recall at 662 QPS — faster than AFan's best QPS
config at equivalent recall.  AFan matches perfect recall but at ~3× lower throughput
in the current Python query path.

---

### Fashion-MNIST 60k  ·  d=784  ·  HNSW build: 60 s

| Method | R@1 | R@10 | R@100 | QPS |
|---|---:|---:|---:|---:|
| HNSW ef=10 | 1.000 | **0.999** | 0.994 | **906** |
| HNSW ef=400 | 1.000 | 1.000 | 1.000 | 368 |
| AFan K=1 cp=20 fp=32 w=9 | 1.000 | **1.000** | — | 192 |
| AFan K=1 cp=5 fp=32 w=9 *(best QPS @ R@10 ≥ 0.90)* | 1.000 | 0.969 | — | **495** |

HNSW ef=10 hits R@10=0.999 at 906 QPS.  AFan reaches perfect recall at 192 QPS.

---

### SIFT 1M  ·  d=128  ·  HNSW build: 249 s (4.2 min)

| Method | R@1 | R@10 | R@100 | QPS |
|---|---:|---:|---:|---:|
| HNSW ef=10 | 0.990 | **0.983** | 0.928 | **529** |
| HNSW ef=200 | 0.995 | 0.995 | 0.979 | **962** |
| HNSW ef=400 | 1.000 | 0.998 | 0.996 | 573 |
| AFan K=1 cp=50 fp=64 w=37 | 0.990 | **0.986** | — | 60 |

HNSW dominates SIFT: ef=200 gives R@10=0.995 at 962 QPS vs AFan's 0.986 at 60 QPS.
This is the dataset where HNSW's graph structure is most efficient.

---

### GloVe 1.18M  ·  d=100  ·  Cosine  ·  HNSW build: 351 s (5.9 min)

| Method | R@1 | R@10 | R@100 | QPS |
|---|---:|---:|---:|---:|
| HNSW ef=400 | 0.920 | **0.905** | 0.839 | **487** |
| HNSW ef=800 | 0.960 | 0.943 | 0.896 | 268 |
| AFan K=2 cp=50 fp=32 w=40 | — | **0.826** | — | 38 |

HNSW outperforms AFan on GloVe: ef=400 reaches R@10=0.905 at 487 QPS, while AFan K=2
reaches only 0.826.  (Note: AFan K=1 reached 0.897 vs FAISS IVF — the HNSW and FAISS
benchmark runs use independently re-tuned AMPI parameters, which can vary slightly due
to the random subsample used during GP-BO tuning.)

---

## Summary

**vs FAISS IVF (static):** AMPI uses 2–3× fewer candidates on MNIST/Fashion for
equivalent recall, and outperforms IVF on GloVe.  IVF is faster in raw QPS due to
FAISS's mature C++ implementation.

**vs HNSW (static):** HNSW wins on QPS across all datasets.  ef=10 on SIFT gives
R@10=0.983 at 529 QPS; AMPI's best is R@10=0.986 at 60 QPS.  HNSW's graph structure
is highly efficient for static corpora.

**The dynamic picture:** HNSW's QPS advantage is only valid on a freshly built,
static index.  Every insertion degrades the graph; every delete is a ghost node;
recovery requires a full rebuild.  At 1M+ vectors that rebuild costs minutes even
with all cores.  AMPI's local-refresh architecture keeps quality without any
scheduled downtime — that is the design goal, and the static QPS gap reflects
current implementation maturity, not algorithmic limits.

---

## Notes

- **Candidate count** is the primary efficiency metric: it measures how many
  vectors are distance-computed before reranking, independent of implementation
  speed.  It is a fair cross-implementation comparison; QPS is not (FAISS has a
  far more mature query-path implementation than AMPI's current Python layer).
- **Gaussian** results are not meaningful for ANN evaluation — see note in that
  section.
- All AMPI parameters (nlist, F) are chosen automatically via GP-BO on a data
  subsample.  Query parameters (cp, fp, w) are swept exhaustively; the table
  shows Pareto-optimal configs.
- Figures (recall vs candidates, QPS vs candidates, dist-ratio) are saved to
  `figures/` by the benchmark scripts.
