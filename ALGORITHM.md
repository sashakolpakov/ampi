# AMPI — Algorithm Description (Mathematical Reference)

---

## 1. Problem statement

### 1.1 Exact formulation

Let **(ℝᵈ, ‖·‖₂)** be the standard Euclidean space.  Given a dataset
**X** = {x₁, …, xₙ} ⊂ ℝᵈ and a query **q** ∈ ℝᵈ, the *k-nearest-neighbour*
problem asks for the set

```
NNₖ(q, X) = argmin_{S ⊆ X, |S|=k}  max_{x ∈ S}  ‖x − q‖₂
```

i.e., the k points in X with the smallest Euclidean distance to q.

The *approximate* k-NN (ANN) problem relaxes this to finding a set S of k
points satisfying:

```
max_{x ∈ S} ‖x − q‖₂  ≤  (1 + ε) · min_{x ∈ X\S} ‖x − q‖₂
```

for some approximation ratio ε > 0.  AMPI targets recall-at-k rather than
a per-query worst-case ε guarantee; empirically recall@10 ≥ 0.95 on SIFT-1M
at ~25× speedup over exhaustive search.

### 1.2 Cosine similarity variant

For the cosine metric, vectors are L2-normalised before indexing, mapping
cosine similarity to squared Euclidean distance:

```
cos(xᵢ, xⱼ) = 〈x̂ᵢ, x̂ⱼ〉 = 1 − ½ · ‖x̂ᵢ − x̂ⱼ‖₂²
```

where x̂ = x / ‖x‖.  Thus maximising cosine similarity on unit vectors is
equivalent to minimising squared L2 distance.  All subsequent analysis
applies unchanged.

---

## 2. Three-level index structure

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| n | Dataset size |
| d | Dimension |
| k | (lowercase) Result count for k-NN queries |
| K | Soft-assignment multiplicity (`cone_top_k`) |
| nlist = M | Number of k-means clusters |
| F | Number of fan axes per cluster |
| cp | Clusters probed per query (`probes`) |
| fp | Fan cones probed per cluster (`fan_probes`) |
| w | Half-window size for sorted-projection search |
| μ_c | Centroid of cluster c, c ∈ {0,…,M−1} |
| aₗ | l-th fan axis (unit vector), l ∈ {0,…,F−1} |
| N_c | Number of points in cluster c |
| n_f^(c) | Number of points in cone f of cluster c |

### 2.2 Structural overview

```
X  ⊂  ℝᵈ
│
└── Level 1: k-means partition  {C₀, …, C_{M-1}}
             μ_c = centroid of Cₒ,   c ∈ [M]
             │
             └── Level 2: affine fan cones  {cone_{c,0}, …, cone_{c,F-1}}
                          global axes a₀, …, a_{F-1} ∈ Sᵈ⁻¹
                          point x ∈ Cₒ → top-K cones by |〈x−μ_c, aₗ〉| / ‖x−μ_c‖
                          │
                          └── Level 3: sorted projection arrays
                               cone_{c,f} stores F arrays, each sorted by 〈x−μ_c, aₗ〉
```

**Query path:** find cp nearest μ_c → for each, find fp best-aligned cones →
retrieve window of 2w points per (cone, axis) → exact L2 rerank.

---

## 3. Build phase

### 3.1 L2-normalisation (cosine metric)

```
x̂ᵢ  ←  xᵢ / ‖xᵢ‖₂     ∀ i ∈ [n]
```

For xᵢ with ‖xᵢ‖₂ < 10⁻¹⁰ the norm is clamped to 1 (zero vector maps to
itself).  All subsequent operations run on the normalised dataset.

### 3.2 Mini-batch k-means

**Subsample.** Let s = min(50 000, n).  Draw without replacement a subsample
S ⊂ [n], |S| = s, uniformly at random.

**Random initialisation.** Choose k centroids μ₁^(0), …, μₖ^(0) ∈ ℝᵈ
uniformly at random from {x_i : i ∈ S}.

**Lloyd iteration t = 0, 1, …, T−1.**

*Assignment step.* For each i ∈ S:

```
cᵢ^(t) = argmin_{c ∈ [k]}  ‖xᵢ − μ_c^(t)‖₂²
         = argmin_{c}  ‖xᵢ‖² + ‖μ_c^(t)‖² − 2 〈xᵢ, μ_c^(t)〉
```

The cross terms 〈xᵢ, μ_c^(t)〉 for all (i, c) are computed in a single BLAS
GEMM: **S** · **μ**ᵀ → matrix of shape (s × k).  Squared norms ‖xᵢ‖² and
‖μ_c^(t)‖² are precomputed vectors (O(s·d) and O(k·d) respectively).
Total cost: O(s · k · d) per iteration (dominated by GEMM).

*Centroid update step.* For each cluster c ∈ [k]:

```
μ_c^(t+1) = (1 / |{i ∈ S : cᵢ^(t) = c}|) · Σ_{i : cᵢ^(t)=c} xᵢ
```

Implemented dimension-by-dimension via `np.bincount(assign, weights=xᵢ[:,j])`
for each j ∈ [d].  Cost: O(s · d) total (s additions across d dimensions).

*Dead cluster respawn.* If cluster c collects no points in iteration t,
its centroid is reseeded uniformly from S.  This prevents k from shrinking.

*Convergence.* Iteration halts when:

```
‖μ_c^(t+1) − μ_c^(t)‖₂  <  10⁻⁶  for all c ∈ [k]
```

or T = 20 iterations are exhausted (empirically 10–15 suffice).

**Spherical variant (cosine).** After each centroid update, renormalise:

```
μ_c^(t+1)  ←  μ_c^(t+1) / ‖μ_c^(t+1)‖₂
```

This constrains centroids to Sᵈ⁻¹ and makes the squared-L2 objective
equivalent to the spherical k-means objective (minimising sum of negative
cosine similarities).

**Full-data assignment.** After convergence, assign all n points (not just S)
to their nearest centroid.  To bound memory, data is processed in chunks of
B = 16 384:

```
for start = 0, B, 2B, … :
    block = X[start : start+B]                    # (B × d)
    D = ‖block‖² + ‖μ‖² − 2 · block · μᵀ        # (B × k)
    assignment[start:start+B] = argmin_c D
```

Peak memory: B × k × 4 bytes.  For k = 1 000, B = 16 384: ~62 MB.

**Overall complexity:** O(T · s · k · d) for iterations + O(n · k · d /
chunk_factor) for full assignment.  With s = 50k, k = 1000, d = 128, T = 15:
≈ 10¹¹ FLOPs (subsample) + ≈ 1.6 × 10¹¹ FLOPs (full assign at n = 1M).

### 3.3 Fan axis generation

Draw F independent vectors from the standard Gaussian distribution:

```
ãₗ ~ N(0, Iᵈ)   →   aₗ = ãₗ / ‖ãₗ‖₂     ∈ Sᵈ⁻¹,   l ∈ [F]
```

**Why Gaussian?**  The unit-sphere projection of a Gaussian random vector is
the *uniform distribution on Sᵈ⁻¹* (rotation-invariant measure).  For any
fixed unit vector v ∈ Sᵈ⁻¹, the projection 〈aₗ, v〉 ~ Beta-distributed with
mean 0 and variance 1/(d−1) by symmetry.

**Why F global axes (not per-cluster)?**  For d = F = 128, by a volumetric
argument the F random axes form a near-tight frame of ℝᵈ: for any unit
vector u the expected maximum squared projection is:

```
E[max_l 〈aₗ, u〉²]  ≥  1 − (1 − 1/d)^F   →  1  as  F → ∞
```

At F = d = 128 this expectation is ≈ 0.63, meaning on average the best axis
captures 79% of the component of any direction — adequate for coarse cone
alignment with zero build overhead.

### 3.4 Cone assignment

For cluster c with point set Cₒ = {xᵢ : assignment[i] = c}:

**Step 1.** Centre all points about the cluster centroid:

```
x̃ᵢ = xᵢ − μ_c    ∀ i ∈ Cₒ
```

This removes the cluster's mean, focusing the analysis on intra-cluster
geometry.

**Step 2.** Project onto all F axes:

```
P = (X_c − μ_c) · Aᵀ    ∈ ℝ^{N_c × F}
```

where X_c ∈ ℝ^{N_c × d} is the submatrix of cluster points and
A ∈ ℝ^{F × d} is the axis matrix.  Cost: O(N_c · F · d) (one GEMM per
cluster, embarrassingly parallel across clusters).

**Step 3.** Compute normalised absolute projections:

```
p̃ᵢₗ = |Pᵢₗ| / ‖x̃ᵢ‖₂    ∈ [0, 1]
```

This is the absolute cosine similarity between x̃ᵢ and aₗ, i.e., the sine
of the angle between x̃ᵢ and the hyperplane orthogonal to aₗ, ranging in
[0, 1] regardless of ‖x̃ᵢ‖.

**Step 4.** Assign each point to its top-K cones:

```
top_cones(i) = argmax_{K out of F} { p̃ᵢₗ }_{l=0}^{F-1}
```

Implemented via `np.argpartition(-p̃, K-1, axis=1)[:, :K]`, which runs in
O(N_c · F) time (linear-time selection, no full sort).

**Geometric interpretation.** The normalised projection p̃ᵢₗ measures how
well the direction from the cluster centroid to xᵢ aligns with axis aₗ.
Points assigned to cone l are those best "facing" axis aₗ from μ_c.  A query
point q whose centred direction q̃ = q − μ_c is nearly parallel to aₗ will
have its neighbours mostly in cone l — because neighbours share similar
centred directions (they are geometrically close).

**Soft assignment (K ≥ 2).** A point within angle arccos(p̃ᵢₗ₁ / p̃ᵢₗ₂) of
the cone boundary between cones l₁ and l₂ is assigned to both.  This ensures
that boundary-proximity does not systematically degrade recall for queries
near cone boundaries, at a cost of K× memory for the sorted arrays.

### 3.5 Sorted projection arrays

For each cone f in cluster c, let:
- I_{c,f} ⊂ [n]: set of global indices of points in this cone, |I_{c,f}| = n_f
- P_{c,f} ∈ ℝ^{n_f × F}: centred projections of those points onto all F axes

For each axis l ∈ [F], construct the sorting permutation:

```
σₗ = argsort({ P_{c,f}[i, l] : i ∈ [n_f] })
```

Store two arrays of length n_f:
- `sorted_idxs[l]` = σₗ (local indices within the cone, int32)
- `sorted_projs[l]` = P_{c,f}[σₗ, l] (sorted projection values, float32)

A `SortedCone` object holds F such pairs plus a global index array
`I_{c,f}` mapping local cone indices to global data indices.

**Storage per cone:** F × n_f × 8 bytes (4 bytes each for float32 + int32).
Total index memory:

```
Σ_{c,f} F × n_f × 8  =  K × n × F × 8  bytes
```

(since each point appears in K cones).  For n = 1M, K = 1, F = 128:
128 × 10⁶ × 8 = 1 GB.  For K = 2: 2 GB.

**Why F arrays per cone?**  The union of F independent windows gives much
higher recall than a single window of equal total size.  Two windows on
uncorrelated axes cover directions orthogonally: a true neighbour that falls
outside the window on axis l (because it lies in a different projection rank)
is still likely captured by at least one other axis whose projection order
aligns better.  This is the key recall amplification mechanism.

---

## 4. Query phase

### 4.1 Cluster selection

Compute the squared L2 distance from q to each centroid:

```
δ_c = ‖q − μ_c‖₂²  =  ‖q‖² + ‖μ_c‖² − 2〈q, μ_c〉
```

Return the cp clusters with smallest δ_c.  Cost: O(M · d) (one dot product
per centroid; the ‖q‖² term cancels in ranking).

### 4.2 Cone selection

For each probed cluster c, centre the query:

```
q̃_c = q − μ_c
```

Compute the absolute normalised projections:

```
sₗ = |〈q̃_c, aₗ〉| / ‖q̃_c‖₂    ∈ [0, 1]
```

These measure the alignment of q's intra-cluster direction with each fan
axis — exactly the same scoring used for data-point cone assignment.  Select
the fp axes with largest sₗ.  Cost: O(F) per cluster (F dot products of
length d have already been reduced to a length-F vector via the projection
`q̃_c @ axes.T`, cost O(F · d)).

### 4.3 Candidate collection

For each selected (cone, axis l, query projection q_proj[l]):

**Binary search.** Find the rank of the query in cone f on axis l:

```
pos_l = |{i ∈ [n_f] : sorted_projs[l][i]  <  q_proj[l]}|
```

Implemented via `std::lower_bound` (C++) or `np.searchsorted` (Python).
Cost: O(log n_f) per axis.

**Window retrieval.** Take all points within w positions of pos_l:

```
W_l = {sorted_idxs[l][j] : j ∈ [max(0, pos_l − w), min(n_f, pos_l + w))}
```

**Union.** Combine across all F axes via a boolean inclusion mask or
`np.unique(concatenated)`:

```
W_{c,f} = W₀ ∪ W₁ ∪ … ∪ W_{F-1}
```

Map local indices to global: `cands = I_{c,f}[W_{c,f}]`.

**Total candidate upper bound:**

```
|cands|  ≤  cp × fp × min(2w × F, n_f)
```

In practice much smaller due to union deduplication; see §5 for the
expected-value analysis.

### 4.4 Exact L2 reranking

Compute squared L2 distances for all collected candidates:

```
dist²(q, xᵢ) = Σⱼ (qⱼ − xᵢⱼ)²    ∀ i ∈ cands
```

Return the k indices with smallest dist².  Implemented via `l2_distances`
(C++ or numba JIT) + `np.argsort`.  Cost: O(|cands| · d).

### 4.5 Adaptive window expansion with early stopping

**Protocol.** `query()` starts at `w₀ = max(k, 8)` and expands geometrically:

```
w_{t+1} = min(2 · wₜ, w_max)
```

where `w_max` is the caller's `window_size` ceiling.  At each iteration:
1. Collect candidates with current w and compute exact distances.
2. Compute `kth_dist² = dist of the k-th nearest found candidate`.
3. Test the coverage condition (below).  If satisfied, exit.

**Coverage condition.** Cone (c, f) is *covered at window w* if there exists
an axis l ∈ [F] such that:

```
min(sorted_projs[l][min(n_f-1, pos_l+w)]  − q_proj[l],
    q_proj[l]  − sorted_projs[l][max(0, pos_l−w−1)])  ≥  kth_proj
```

where `kth_proj = √kth_dist²` (or +∞ when the window already covers the
entire cone).

**Correctness proof via Cauchy-Schwarz.**  For any unit vector a ∈ Sᵈ⁻¹ and
any two points x, q ∈ ℝᵈ:

```
‖x − q‖₂  =  sup_{u ∈ Sᵈ⁻¹}  〈x − q, u〉  ≥  〈x − q, a〉  =  proj_a(x) − proj_a(q)
```

Taking absolute values:

```
‖x − q‖₂  ≥  |proj_a(x) − proj_a(q)|                      (*)
```

Now suppose point i is not in the current window W_l on axis l, meaning its
position in the sorted array is either above pos_l + w or below pos_l − w.
In the "above" case:

```
sorted_projs[l][pos_l + w]  ≤  proj_{aₗ}(xᵢ)
```

so by (*):

```
‖xᵢ − q‖₂  ≥  |proj_{aₗ}(xᵢ) − q_proj[l]|
              ≥  sorted_projs[l][pos_l + w] − q_proj[l]  ≥  kth_proj
```

Therefore xᵢ has distance ≥ kth_proj ≥ √kth_dist² from q, and cannot
appear in the top-k result.  The same argument holds for the "below" case.

When the coverage condition holds for every unvisited point in every probed
cone (guaranteed by the boundary gap check), the top-k result is
**provably identical to an exhaustive scan of all probed cones**.

This is a correctness certificate that IVF-flat does not possess: IVF always
returns all cluster points as candidates; AMPI can exit early with a proof.

**Empirical behaviour.** On SIFT-1M at recall@10 ≥ 0.97, ≥ 85% of queries
exit after the first iteration (w = max(k, 8)), and > 99% exit within 2
doublings.

---

## 5. Candidate count analysis

### 5.1 Expected cone size

With M clusters and K soft-assignment, the total number of (point, cone)
memberships is K·n (each of the n points occupies K cones).  With F cones
per cluster and M clusters:

```
E[n_f]  =  K · n / (M · F)
```

For n = 1M, M = √n = 1000, K = 1, F = 128: E[n_f] = 10⁶ / (1000 × 128)
≈ 7.8 points per cone.

### 5.2 Expected candidate count per query

For one probed cone with window w:

```
E[|W_{c,f}|]  ≤  F × 2w × (1 − (1 − 1/n_f)^{2w·F})^{-1}  ≈  min(F × 2w, n_f)
```

(union bound; actual candidates much fewer due to deduplication).

Summing over cp × fp cone probes:

```
E[|cands|]  ≈  cp × fp × (K·n / (M·F)) × 2w  ×  dedup_factor
```

With `nlist = α√n`, F chosen as largest power of 2 s.t. E[n_f] ≥ w, and
typical α = 1, cp = 10, fp = F/4, w = 50:

```
E[|cands|]  ≈  10 × 32 × 7.8 × 2×50 × 0.1  ≈  2 500   (n=1M, aggressive)
              10 × 32 × 7.8 × 100           ≈  25 000  (n=1M, with dedup≈1)
```

This is sublinear in n: with M = α√n, E[n_f] scales as √n, so E[|cands|] ∝ √n
for fixed w/√n.

### 5.3 Parameter F vs recall tradeoff

The probability that a true nearest neighbour xᵢ falls outside all fp × F
windows can be bounded as follows.  Suppose xᵢ has centred direction v̂ =
(xᵢ − μ_c) / ‖xᵢ − μ_c‖.  The cone assigned to xᵢ is the one with highest
p̃ᵢₗ = |〈v̂, aₗ〉|.  The expected best alignment is:

```
E[max_{l ∈ [F]} |〈v̂, aₗ〉|]  →  1  as  F → ∞
```

For the query's cone ranking to agree with xᵢ's cone ranking (i.e., for xᵢ
to appear in one of the fp cones probed for q), we need at least one common
axis in the top-fp of both.  The probability of overlap increases with fp and
F; at F = fp = 128 the overlap is total (all cones are probed).

---

## 6. Streaming insertion

### 6.1 Overview

`add(x)` inserts one vector into a live index without rebuilding any global
structure.  The three core invariants maintained after each insert are:

1. **Membership:** x is reachable by a query probing at least one of x's
   assigned clusters and cones.
2. **Sortedness:** each SortedCone's F arrays remain sorted by projection value.
3. **Centroid accuracy:** each cluster centroid μ_c equals the exact mean of
   all inserted (non-deleted) points, modulo floating-point rounding.

### 6.2 Capacity buffer for self.data

To avoid O(n) copy on every insert, `data` is stored in a pre-allocated
buffer of capacity C, with a view `self.data = self._data_buf[:n]` exposed:

```
C₀ = n₀ + 1024          (initial buffer with 1024-slot headroom)
Cₜ₊₁ = 2 · Cₜ           (double when full)
```

This gives amortised O(1) memory operations per insert (the standard
doubling-buffer argument: total copy work over n inserts is Σ₂ᵏ ≤ 2n).

The deleted-mask buffer `_del_mask_buf` (boolean array, 1 byte/point) follows
the same doubling schedule.

### 6.3 Cluster assignment

For a new point x ∈ ℝᵈ:

```
d²(x, μ_c)  =  ‖x‖² + ‖μ_c‖² − 2〈x, μ_c〉    ∀ c ∈ [M]
top_clusters = argsort_{c} d²(x, μ_c)  :  [:K]
```

Cost: O(M · d).

Note: a Dirichlet-Process (DP) formulation would weight cluster c by its
prior: `N_c · p(x | μ_c, Σ_c)`.  This is not used because:
(a) the DP prior favours large clusters, distorting ANN recall near
    boundaries of unequal-size clusters;
(b) evaluating Gaussian likelihoods requires per-cluster covariances
    (O(d²) storage and O(d²) update per insert);
(c) empirically, nearest-centroid top-K is sufficient for fixed-M ANN.

### 6.4 Cone insert

For each assigned cluster c:

1. Compute centred projection:

```
x̃ = x − μ_c
proj = A · x̃    ∈ ℝᶠ
```

2. Select top-K cones by normalised projection (same criterion as §3.4):

```
top_f = argmax_K { |proj[l]| / ‖x̃‖ }_{l=0}^{F-1}
```

3. For each cone f ∈ top_f, call `SortedCone.insert(proj, global_id)`:
   - For each axis l ∈ [F]: find insertion rank via `std::lower_bound`
     in the l-th sorted vector (O(log n_f)), then `std::vector::insert`
     to shift elements (O(n_f)).
   - Total per-insert cost: O(F · (log n_f + n_f)).
   - The O(n_f) shift is the bottleneck for large cones; a B-tree would
     reduce this to O(F · log n_f) and is planned for n_f > 10k.

4. Update `_point_cones[global_id]` inverse index:

```
_point_cones[global_id] += [(c, f) for f in top_f]
```

5. Append global_id to `cluster_global[c]`.

### 6.5 Centroid exponential moving average

After inserting x into cluster c with current count N_c:

```
μ_c^{new} = (N_c · μ_c + x) / (N_c + 1)
N_c       ← N_c + 1
```

This is the *online arithmetic mean update*, exact in infinite precision.
In float32 it accumulates rounding error O(n · ε_mach · ‖μ_c‖₂), which is
negligible for n ≤ 10M.

For cosine metric, note that after normalisation xᵢ ∈ Sᵈ⁻¹, but the
centroid μ_c after the EMA update lies inside the unit ball.  Queries use
the centroid as-is for distance ranking; the cluster's interior centroid
still correctly ranks the cluster's distance relative to other clusters for
most queries.

### 6.6 Drift covariance EMA

**Motivation.** The initial cone assignment (§3.4) was optimal for the data
at build time.  As new points accumulate in a directional sub-region of
cluster c, the leading geometric direction within the cluster shifts.  If
this new direction is not well-covered by any fan axis, points near the
boundary of the shifted sub-region will be assigned to misaligned cones,
reducing recall for queries in that direction.

**Covariance accumulation.** Maintain a per-cluster d×d matrix Σ_c ∈ ℝ^{d×d}:

```
Σ_c  ←  (1 − β) · Σ_c  +  β · (x − y)(x − y)ᵀ
```

where β = 0.01 (exponential decay constant) and y is the **approximate
nearest neighbour** of x within cluster c (see §6.7 below).

The vector v = x − y is the local pair displacement.  Its outer product
v·vᵀ is a rank-1 PSD matrix with sole eigenvector v/‖v‖ and eigenvalue
‖v‖².  After many inserts, Σ_c converges to an EMA of rank-1 matrices:

```
Σ_c  ≈  β · Σᵢ (1−β)^{t−i} · vᵢ · vᵢᵀ
```

which is the exponentially-weighted second moment of the displacement
distribution.  Its leading eigenvector (dominant direction of recent local
displacements) is extracted by power iteration.

**Power iteration (5 steps).** Starting from v₀ = Σ_c · a₀ (warm-started
on the first fan axis to exploit likely alignment):

```
vᵢ₊₁ = Σ_c · vᵢ / ‖Σ_c · vᵢ‖₂    i = 0, 1, 2, 3, 4
```

After 5 steps, v₅ converges to the leading eigenvector of Σ_c.  The number
of steps needed for ε-accuracy in eigenvector angle is:

```
t  ≥  log(2/ε) / log(λ₁/λ₂)
```

where λ₁ > λ₂ are the top two eigenvalues.  For typical drift scenarios
λ₁/λ₂ ≥ 2, giving < 0.001 rad error in 5 steps.

**Trigger condition.** Compute the maximum cosine similarity between v₅
and any fan axis:

```
cos_max = max_{l ∈ [F]}  |〈v₅, aₗ〉|
```

If `cos_max < cos(θ_drift)` with θ_drift = 15°, i.e.,

```
cos_max  <  cos(π/12)  ≈  0.9659
```

the leading drift direction is more than 15° from all fan axes.  A query
aligned with this direction will miss most points in this sub-region, so a
local refresh is triggered.

### 6.7 Approximate nearest-neighbour proxy for drift

Instead of using the centroid as proxy for y (which overstates the
displacement by ‖x − μ_c‖₂ instead of the true local pair distance), `add()`
queries the cones x was just inserted into:

```
nn_cands = ∪_{f ∈ top_f} cone_{c,f}.query(proj, w=8)
nn_cands ← nn_cands \ {global_id}          (exclude x itself)
y = argmin_{i ∈ nn_cands} ‖x − xᵢ‖₂
```

This uses the sorted-projection window mechanism with w = 8 (tiny constant
cost, no extra projections needed since proj was already computed for the
insert).  The l2_distances call runs on |nn_cands| ≤ 2 · K · 8 · F = 2048
candidates at most (16 bytes each → ≤ 4 kB, likely L1-cached).

Falls back to `y = μ_c` (centroid as proxy) when all selected cones have
only x itself (first insertion, or very sparse cone after compaction).

**Cost of the NN probe:** O(K · F · w) cone reads + O(|nn_cands| · d)
distance computations.  With K=2, F=16, w=8, d=128: ≤ 256 reads + 2048
distance components — negligible relative to the O(M · d) cluster-
assignment cost that dominates `add()`.

### 6.8 Local refresh

`_local_refresh(c)` is called when drift is detected (§6.6) or tombstone
fraction exceeds 10% (§6.10).  It rebuilds all cone structures for cluster c:

1. Collect live global ids: `c_idx = cluster_global[c][~deleted_mask[c_idx]]`
2. Remove all old (c, f) entries from `_point_cones` for c_idx.
3. Call `_build_cones_for_cluster(c_idx, data[c_idx], μ_c, axes, F, K)`,
   which re-projects, re-normalises, re-sorts, and re-builds SortedCone objects.
4. Update `cluster_cones[c]`, `cluster_global[c]`, `_cluster_counts[c]`.
5. Extend `_point_cones` with new (c, f) entries.
6. Reset: `Σ_c ← 0`,  `_cluster_tombstones[c] ← 0`.

Cost: O(N_c · F · d) projections + O(N_c · K · F · log N_c) sorting.
This is a local operation; all other M−1 clusters are untouched.  In
contrast, IVF's full retrain costs O(T · n · M · d).

### 6.9 Delete

`delete(global_id)` is a logical delete:

1. Set `_deleted_mask[global_id] = True`.
2. For each (c, f) ∈ `_point_cones[global_id]`:
   - Call `SortedCone.remove(global_id)`, which inserts `global_id` into the
     cone's `unordered_set<uint32_t> tombstones`.  Cost: O(1) amortised.
   - Increment `_cluster_tombstones[c]`.
3. For each affected cluster c, check:

```
_cluster_tombstones[c] / _cluster_counts[c]  ≥  0.10
```

If so, call `_local_refresh(c)`.

**Why two-level tombstoning?**  The `_deleted_mask` enables O(1) post-hoc
filtering in `query()` (a single boolean mask index over the candidate array).
The per-cone tombstone `unordered_set` is the authoritative live/dead flag
used during sorted-array traversal in `SortedCone.query()`.  This separation
allows `query()` to skip the tombstone check in the fast path (`fan_probes ≥ F`)
by checking only `_deleted_mask`.

**Physical eviction.** Points are physically removed from sorted arrays only
during `_local_refresh(c)`.  Between deletes and refresh, tombstoned entries
occupy space in the sorted arrays but are filtered at query time.  The 10%
threshold limits the overhead from tombstoned entries to at most 10% extra
candidates per cluster before compaction.

### 6.10 Update

```
update(global_id, x) = delete(global_id) ; return add(x)
```

The returned new_global_id ≠ global_id in general (the new vector appends at
position n).  The old slot remains allocated in `_data_buf` but is masked off
by `_deleted_mask`.  Unrecovered slots are freed when `_local_refresh` rebuilds
the cluster.

---

## 7. SortedCone data structure

### 7.1 Internal layout (C++, `ampi/_ext.cpp`)

```cpp
class SortedCone {
    int F;
    std::vector<std::vector<std::pair<float, uint32_t>>> axes;  // [F][n_f]
    std::unordered_set<uint32_t> tombstones;
};
```

`axes[l]` stores the l-th axis sorted array as pairs (projection_value,
global_id), ordered by `pair::first` ascending.

### 7.2 `from_arrays` constructor

Called from Python at build time or after local refresh:

```
Input:  sorted_projs: (F, n_f) float32
        sorted_idxs:  (F, n_f) int32  (local indices, unused — global_idx maps positions)
        global_idx:   (n_f,)   int32
Output: SortedCone with axes[l][j] = (sorted_projs[l][j], global_idx[sorted_idxs[l][j]])
```

Cost: O(F · n_f) — one pass per axis to interleave projection values with
global ids.

### 7.3 `insert(proj_values, global_id)`

For each axis l ∈ [F]:
1. `pos = lower_bound(axes[l].begin(), axes[l].end(), {proj_values[l], 0})` — O(log n_f)
2. `axes[l].insert(pos, {proj_values[l], global_id})` — O(n_f) shift

Total: O(F · (log n_f + n_f)).  The O(n_f) vector shift dominates; replacing
`std::vector` with a B-tree would give O(F · log n_f) but with larger constant.

### 7.4 `query(q_projs, w)` and `is_covered(q_projs, w, kth_proj)`

**`query`:** for each axis l, binary-search for q_projs[l], read 2w entries,
union via an intermediate `uint32_t` boolean array of length n_f (O(F · w)
random writes), convert to sorted int32 array.

**`is_covered`:** for each axis l, check boundary gaps in O(1) per axis:

```
gap_right = axes[l][min(n_f-1, pos+w)].first  − q_projs[l]
gap_left  = q_projs[l]  − axes[l][max(0, pos-w-1)].first
if min(gap_right, gap_left) >= kth_proj: return True   // covered on this axis
```

Return `True` iff any axis provides coverage.  Cost: O(F) — exit as soon as
one axis suffices.

### 7.5 Method cost table

| Method | Time | Space |
|--------|------|-------|
| `from_arrays(sorted_projs, sorted_idxs, global_idx)` | O(F · n_f) | O(F · n_f) |
| `insert(proj_values, global_id)` | O(F · (log n_f + n_f)) | O(F) amortised |
| `remove(global_id)` | O(1) amortised | O(1) |
| `compact()` | O(F · n_f) | O(F · n_f) |
| `query(q_projs, w)` | O(F · (log n_f + w)) | O(n_f) |
| `is_covered(q_projs, w, kth_proj)` | O(F) | O(1) |
| `all_ids()` | O(n_f) | O(n_f) |
| `size()` | O(1) | O(1) |

---

## 8. Complexity summary

| Operation | Time | Dominant Term |
|-----------|------|---------------|
| **Build: k-means** | O(T · s · M · d + n · M · d) | s=50k, T≤20 |
| **Build: fan axes** | O(F · d) | negligible |
| **Build: projections** | O(n · F · d) | one GEMM per cluster |
| **Build: sort** | O(n · K · F · log(n/(M·F))) | per-cone argsort |
| **Query** | O(M·d + cp·F·d + cp·fp·F·(log n_f + w) + \|cands\|·d) | rerank dominates |
| **Insert** | O(M·d + K·F·(log n_f + n_f) + K·F·w + d) | cone shift dominates |
| **Delete** | O(K·F) | unordered_set insert |
| **Local refresh** | O(N_c · F · d + N_c · K · F · log N_c) | projection + sort |

For n = 10⁶, M = 10³, d = 128, F = 128, K = 1, cp = 10, fp = 16, w = 50:
- Query: ~2.6 · 10⁶ FLOPs (cluster select: 1.3·10⁵, cone projections:
  2·10⁵, window reads: 1.3·10⁴, rerank at |cands|≈2k: 2.6·10⁶)
- Insert: ~10⁶ FLOPs for cluster assign + ~10⁴ per cone shift
- Exhaustive scan: 10⁶ · 128 = 1.3·10⁸ FLOPs

---

## 9. Parameter reference and scaling rules

### 9.1 Parameter table

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| `nlist` | M | √n | k-means clusters. ↑ → finer partition, slower build, less fan-search work per query |
| `num_fans` | F | 16–128 | Fan axes. ↑ → better recall, F× more memory, F× more sort cost |
| `cone_top_k` | K | 1 | Soft-assignment. ↑ → better cross-boundary recall, K× memory |
| `probes` (query) | cp | tuned | Clusters probed. ↑ → better recall, linear cost increase |
| `fan_probes` (query) | fp | tuned | Cones per cluster. ↑ → better recall, linear cost; fp = F → full-cluster fallback |
| `window_size` (query) | w_max | 200 | Max window. Rarely reached due to early stopping |
| `seed` | — | 0 | RNG seed for axes and k-means init |
| `metric` | — | 'l2' | 'cosine' adds normalisation in __init__, add(), query() |

### 9.2 Scaling rules

**nlist.** The optimal M balances two competing costs:
- Cluster assignment: O(M · d) per query
- Fan-search work: O(cp · n/(M · F) · F · w) ∝ n/M per query

Setting these equal gives M ∝ √n.  The proportionality constant α is tuned
by `AFanTuner` (GP-BO over recall-QPS Pareto frontier):

```
M = round(α · √n),   α ∈ [0.25, 3.0]
```

**F.** Chosen as the largest power of 2 such that the average cone size
exceeds the base window:

```
F = max{16, 32, 64, 128}  subject to  n / (M · F)  ≥  w_base
w_base = max(15, 15 · √(n / 10 000))
```

This ensures the sorted arrays are long enough for the window search to be
meaningful.

**w_max (window ceiling).** Scales as √n: `w_base = max(15, 15·√(n/10k))`.
Rationale: the projection gap for a random pair in ℝᵈ grows as O(√n / n_f)
in absolute units, so the window that captures a fixed fraction of each cone
scales as O(√n / M) = O(n⁰·⁰).  In practice w_base = 15–150 over the range
n = 10k–1M.

---

## 10. Relation to inverted file index (IVF)

**IVF-flat** (FAISS `IndexIVFFlat`):
- Build: k-means on n points → M centroids.
- Query: probe cp clusters, return all N_c ≈ n/M points as candidates, rerank.
- Insert: O(M·d) assign + O(1) append to cluster list.  No sorted arrays.
- Distribution shift: the Voronoi boundaries computed at build time become
  inaccurate as data drifts.  Eventually requires a **full global retrain**:
  O(T·n·M·d) — impractical for live systems.

**AMPI AffineFan vs IVF-flat:**

| Property | IVF-flat | AMPI AffineFan |
|----------|----------|----------------|
| Build cost | O(T·n·M·d) | same |
| Query candidates | O(cp · n/M) | O(cp · fp · w · F) ≪ n/M |
| Correctness cert. | none | Cauchy-Schwarz coverage |
| Insert | O(M·d) | O(M·d + K·F·n_f) |
| Delete | O(scan) or unsupported | O(K·F) tombstone |
| Distribution shift | full global retrain | per-cluster local refresh |
| Memory overhead | n/M avg list pointers | K·n·F × 8 bytes |

The extra memory for sorted arrays (K·F×8 bytes/point = 1 KB/point at
K=1, F=128) is the price of the second filtering stage, which reduces
candidates by factor ≈ F/(fp·2w·M/n) relative to IVF.

---

## 11. AFanTuner: automatic hyperparameter optimisation

`AFanTuner` runs a Bayesian optimisation (Gaussian Process with Expected
Improvement) over the 2D search space (α, K) to find the Pareto-optimal
α = nlist/√n and cone_top_k = K for a given dataset.

**Search space:**
- α ∈ [0.25, 3.0] (nlist scale)
- K ∈ {1, 2, 3} (discrete soft-assignment multiplicity)

**Objective:** maximise recall@10 subject to QPS ≥ target_qps.  The
scalarised objective is:

```
obj(α, K) = recall@10 − λ · max(0, target_qps − measured_qps)
```

**GP prior.** Zero-mean GP with squared-exponential kernel:

```
k((α₁,K₁), (α₂,K₂)) = σ² · exp(−‖(α₁,K₁) − (α₂,K₂)‖² / (2ℓ²))
```

with σ² = 1, ℓ = 0.5 (in normalised coordinates).  After n_init = 5 random
evaluations, 10 BO steps are run with Expected Improvement acquisition.

**`_scale_params(n, d)`** returns the analytically-derived default query
parameters (probes, fan_probes, window_size) as a function of (n, d), used
as the evaluation point during tuning and as the default for `benchmark.py`.

---

## 12. AMPIBinaryIndex (baseline)

A degenerate version of AMPI with no k-means clustering:

**Build:** draw L = num_fans random unit vectors aₗ ~ uniform(Sᵈ⁻¹).
Project all n points: P ∈ ℝ^{n × L}.  For each l ∈ [L], sort by Pᵢₗ.
Store L sorted arrays of (projection, global_id).

Cost: O(n · L · d) projections + O(n · L · log n) sorting.

**Query:** for each axis l, binary-search for q_proj[l], take a window of w
points on each side.  Return union reranked by L2.

Cost: O(L · (d + log n + w) + |cands| · d).

**Comparison to AffineFan:** AMPIBinaryIndex corresponds to AffineFan with
M = 1 (one cluster = entire dataset), F = L, cp = 1, fp = L.  The lack of
clustering means:
- Each sorted array has n entries (vs n/M for AffineFan) — higher memory
- No geometric grouping: true neighbours are not preferentially grouped
  in fewer cones, so more candidates are needed for equivalent recall.
- 2–4× more candidates needed for equivalent recall on clustered datasets.

AMPIBinaryIndex is density-adaptive (window always spans exactly 2w
entries regardless of local density) and simpler to implement and prove
correct; it serves as a correctness baseline.
