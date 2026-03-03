"""
AMPI Tomographic: geometry-guided projections for sorted-projection ANN.

Build
  1. Bootstrap S approximate (anchor, NN) pairs via one random projection.
  2. Generate C = C_factor·L candidate directions from data difference vectors.
  3. Score each candidate by separability: score(a) = mean|a·(anchor−rand)| / mean|a·(anchor−nn)|
  4. Keep the L highest-scoring directions.
  5. Refine with power iteration (global or local):
       global: a ← X^T(Xa)  — biases toward global PCs (good for SIFT, bad for MNIST)
       local:  a ← D^T(Da)  where D = difference vectors of NN pairs
               biases toward directions that separate local neighbours, not global clusters
  6. Sort all n projections per direction.

Query modes
  union  : take all points in ±window per projection, union → re-rank (classic)
  voting : count votes per point across projections, keep votes≥min_votes → re-rank
           True NNs appear in ~all L windows; random false positives in ~0.
           Shrinks candidate pool by 10-100x at negligible recall cost.
"""

import numpy as np
from ._kernels import project_data, l2_distances, jit_union_query, jit_vote_query


# ── internal helpers ──────────────────────────────────────────────────────────

def _bootstrap_nn_pairs(data, S, rng):
    """Approximate NN pairs via a single random projection → (S, 2) int32."""
    n, d = data.shape
    a = rng.randn(d).astype(np.float32)
    a /= np.linalg.norm(a)
    order = np.argsort(data @ a)
    adj   = np.stack([order[:-1], order[1:]], axis=1)
    sel   = rng.choice(len(adj), min(S, len(adj)), replace=False)
    return adj[sel]


def _score_directions(data, candidates, pairs, rng):
    """score(a) = mean|a·(anchor−rand)| / mean|a·(anchor−nn)| → (C,) float32."""
    n        = data.shape[0]
    S        = len(pairs)
    anchors  = data[pairs[:, 0]]
    nns      = data[pairs[:, 1]]
    randoms  = data[rng.choice(n, S, replace=True)]
    nn_proj   = np.abs((anchors - nns)     @ candidates.T)   # (S, C)
    rand_proj = np.abs((anchors - randoms) @ candidates.T)   # (S, C)
    return (rand_proj.mean(0) / (nn_proj.mean(0) + 1e-8)).astype(np.float32)


def _global_power_iter(data, directions, steps):
    """a ← X^T(Xa)/‖·‖ over full dataset — vectorised over all L at once."""
    A = directions.T.astype(np.float64)          # (d, L)
    for _ in range(steps):
        XA = data @ A                            # (n, L)
        A  = data.T @ XA                         # (d, L)
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        A /= np.where(norms < 1e-10, 1.0, norms)
    return A.T.astype(np.float32)                # (L, d)


def _local_power_iter(data, directions, pairs, steps):
    """a ← D^T(Da)/‖·‖ where D = NN-pair difference vectors.

    Biases directions toward explaining *local* neighbour separations rather
    than global variance.  Critical for datasets where global PCs do not align
    with local NN geometry (e.g. MNIST: global PCs separate digit classes, but
    NN queries are within-class).
    """
    anchors = data[pairs[:, 0]].astype(np.float64)
    nns     = data[pairs[:, 1]].astype(np.float64)
    D       = anchors - nns                      # (S, d) — local displacements
    A = directions.T.astype(np.float64)          # (d, L)
    for _ in range(steps):
        DA = D @ A                               # (S, L)
        A  = D.T @ DA                            # (d, L)
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        A /= np.where(norms < 1e-10, 1.0, norms)
    return A.T.astype(np.float32)                # (L, d)


# ── public utilities ──────────────────────────────────────────────────────────

def estimate_nn_distance(data, S=500, seed=0):
    """Median approximate NN distance estimated from bootstrap pairs.

    Useful for auto-tuning bucket_size in AMPIHashIndex:
        bucket_size = estimate_nn_distance(data) / sqrt(d)
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    rng  = np.random.RandomState(seed)
    pairs = _bootstrap_nn_pairs(data, S, rng)
    diffs = data[pairs[:, 0]] - data[pairs[:, 1]]
    return float(np.median(np.sqrt(np.sum(diffs ** 2, axis=1))))


def geometry_guided_directions(data, L, C_factor=5, S=500,
                                power_iter=1, local=True, seed=0):
    """Select L geometry-aware unit vectors for projecting ``data``.

    Parameters
    ----------
    data       : (n, d) float32
    L          : number of directions to return
    C_factor   : C = C_factor * L candidates screened
    S          : bootstrap NN-pairs for scoring and local power iter
    power_iter : refinement steps (0 = off)
    local      : if True use local power iteration (NN-pair differences);
                 if False use global (full-data covariance).
                 Local is better for structured data; global for SIFT-like data.
    seed       : RNG seed

    Returns
    -------
    directions : (L, d) float32
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    rng  = np.random.RandomState(seed)
    n, d = data.shape
    C    = C_factor * L

    pairs = _bootstrap_nn_pairs(data, S, rng)

    ii = rng.choice(n, C, replace=True)
    jj = rng.choice(n, C, replace=True)
    jj[ii == jj] = (jj[ii == jj] + 1) % n
    diffs = (data[ii] - data[jj]).astype(np.float64)
    norms = np.linalg.norm(diffs, axis=1)
    valid = norms > 1e-8
    diffs = diffs[valid] / norms[valid, None]
    shortfall = C - len(diffs)
    if shortfall > 0:
        extra = rng.randn(shortfall, d)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        diffs = np.vstack([diffs, extra])

    candidates = diffs[:C].astype(np.float32)
    scores     = _score_directions(data, candidates, pairs, rng)
    top_idx    = np.argsort(scores)[::-1][:L]
    selected   = candidates[top_idx].copy()

    if power_iter > 0:
        if local:
            selected = _local_power_iter(data, selected, pairs, steps=power_iter)
        else:
            selected = _global_power_iter(data, selected, steps=power_iter)

    return selected


# ── index ─────────────────────────────────────────────────────────────────────

class AMPITomographicIndex:
    """Approximate nearest-neighbour index with geometry-guided projections.

    Parameters
    ----------
    data            : (n, d) array_like, float32
    num_projections : L
    C_factor        : candidate multiplier for direction screening
    S               : bootstrap NN-pairs
    power_iter      : refinement steps (0 = off)
    local           : use local power iteration (default True; set False for
                      datasets where global PCA aligns with local structure)
    seed            : RNG seed
    """

    def __init__(self, data, num_projections=16, C_factor=5, S=500,
                 power_iter=1, local=True, seed=0):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.n, self.d = self.data.shape
        self.L = num_projections

        self.proj_dirs = geometry_guided_directions(
            self.data, self.L,
            C_factor=C_factor, S=S, power_iter=power_iter, local=local, seed=seed,
        )

        projs = project_data(self.data, self.proj_dirs)
        self.sorted_idxs  = np.empty((self.L, self.n), dtype=np.int32)
        self.sorted_projs = np.empty((self.L, self.n), dtype=np.float32)
        for i in range(self.L):
            order = np.argsort(projs[i])
            self.sorted_idxs[i]  = order
            self.sorted_projs[i] = projs[i, order]

    # ── public API ────────────────────────────────────────────────────────────

    def query_candidates(self, q, window_size=100):
        """Union of ±window_size neighbours across all projections."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        q_projs = np.ascontiguousarray(q @ self.proj_dirs.T, dtype=np.float32)
        return jit_union_query(self.sorted_idxs, self.sorted_projs, q_projs, window_size)

    def query_candidates_voting(self, q, window_size=100, min_votes=None, min_return=0):
        """Voting-based candidates: keep points appearing in ≥min_votes projections.

        If the filtered set has fewer than min_return points, falls back to the
        full union so the caller always gets a usable candidate set.  Pass
        min_return=k to match the behaviour of query_voting.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        if min_votes is None:
            min_votes = max(1, self.L // 4)
        q_projs = np.ascontiguousarray(q @ self.proj_dirs.T, dtype=np.float32)
        cands   = jit_vote_query(
            self.sorted_idxs, self.sorted_projs, q_projs, window_size, min_votes
        )
        if len(cands) < min_return:
            cands = jit_union_query(
                self.sorted_idxs, self.sorted_projs, q_projs, window_size
            )
        return cands

    def query(self, q, k=10, window_size=100):
        """Union-mode query (baseline)."""
        q     = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates(q, window_size)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]

    def query_voting(self, q, k=10, window_size=100, min_votes=None):
        """Voting-mode query: far fewer candidates, same or better recall."""
        q     = np.ascontiguousarray(q, dtype=np.float32)
        cands = self.query_candidates_voting(q, window_size, min_votes, min_return=k)
        dists = l2_distances(self.data, q, cands)
        top   = np.argsort(dists)[:k]
        return self.data[cands[top]], dists[top], cands[top]


# ── two-stage pipeline ────────────────────────────────────────────────────────

class AMPITwoStageIndex:
    """Two-stage pipeline: Sub-tomo coarse retrieval → tomo voting precision pass.

    Stage 1  AMPISubspaceIndex built with geometry-guided directions → broad
             candidate set (fast, good recall).
    Stage 2  Project stage-1 candidates onto a separate set of tomo directions;
             vote: give each candidate one vote per projection window it wins;
             keep only candidates with ≥ min_votes votes (high precision).
    Stage 3  L2 rerank the survivors → top k.

    Parameters
    ----------
    data            : (n, d) array_like, float32
    num_projections : L — projections used in both stages
    bins_per_axis   : B — equal-frequency bins for the subspace stage
    subspace_dim    : d_sub — subspace dimension for the coarse stage
    C_factor, S, power_iter, local, seed
                    : geometry_guided_directions kwargs (applied to both stages)
    """

    def __init__(self, data, num_projections=16, bins_per_axis=8, subspace_dim=2,
                 C_factor=5, S=500, power_iter=1, local=True, seed=0):
        from .subspace import AMPISubspaceIndex

        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.n, self.d = self.data.shape
        self.L = num_projections

        # Directions for coarse subspace stage (L * d_sub directions, grouped into L frames)
        dirs_coarse = geometry_guided_directions(
            self.data, num_projections * subspace_dim,
            C_factor=C_factor, S=S, power_iter=power_iter, local=local, seed=seed,
        )
        # Directions for fine voting stage (L independent directions, different seed)
        dirs_fine = geometry_guided_directions(
            self.data, num_projections,
            C_factor=C_factor, S=S, power_iter=power_iter, local=local, seed=seed + 1,
        )

        self.sub_index = AMPISubspaceIndex(
            data, num_projections=num_projections, bins_per_axis=bins_per_axis,
            subspace_dim=subspace_dim, seed=seed, directions=dirs_coarse,
        )
        self.proj_dirs = dirs_fine  # (L, d) for voting pass

    # ── internal ──────────────────────────────────────────────────────────────

    def _vote_filter(self, q_projs, coarse, fine_window, min_votes, min_return):
        """Project coarse candidates onto fine directions and vote-filter.

        fine_window : int   → absolute count (top-w per projection direction)
                      float → fraction of coarse set, e.g. 0.10 = top-10%
        """
        m = len(coarse)
        if m == 0:
            return coarse
        # resolve fractional fine_window relative to coarse set size
        if isinstance(fine_window, float) and fine_window < 1.0:
            w = max(1, int(fine_window * m))
        else:
            w = max(1, min(int(fine_window), m))
        cand_projs = self.data[coarse] @ self.proj_dirs.T  # (m, L)
        votes = np.zeros(m, dtype=np.int32)
        for i in range(self.L):
            diffs = np.abs(cand_projs[:, i] - q_projs[i])
            top_w = np.argpartition(diffs, w - 1)[:w]
            votes[top_w] += 1
        fine = coarse[votes >= min_votes]
        if len(fine) < min_return:
            fine = coarse
        return fine

    # ── public API ────────────────────────────────────────────────────────────

    def query_candidates(self, q, sub_probes=2, fine_window=0.10, min_votes=None, min_return=0):
        """Return fine candidates after coarse retrieval + voting filter.

        fine_window : fraction of coarse set (float < 1) or absolute count (int).
                      Default 0.10 means top-10% of coarse per projection direction.
        min_return  : if fine set < min_return, fall back to full coarse set.
                      Pass min_return=k to match the behaviour of query().
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        if min_votes is None:
            min_votes = max(1, self.L // 4)
        coarse  = self.sub_index.query_candidates(q, probes=sub_probes)
        q_projs = q @ self.proj_dirs.T
        return self._vote_filter(q_projs, coarse, fine_window, min_votes, min_return=min_return)

    def query(self, q, k=10, sub_probes=2, fine_window=0.10, min_votes=None):
        """Two-stage ANN query.

        Parameters
        ----------
        q           : (d,) float32
        k           : neighbours to return
        sub_probes  : coarse probe radius in the subspace stage
        fine_window : candidates per projection direction in the voting pass
        min_votes   : minimum votes to survive the precision filter
                      (default L//4)

        Returns
        -------
        points  : (k, d) float32
        dists   : (k,)   float32 — squared L2 distances
        indices : (k,)   int32
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        if min_votes is None:
            min_votes = max(1, self.L // 4)
        coarse  = self.sub_index.query_candidates(q, probes=sub_probes)
        if len(coarse) == 0:
            coarse = np.arange(min(k, self.n), dtype=np.int32)
        q_projs = q @ self.proj_dirs.T
        fine    = self._vote_filter(q_projs, coarse, fine_window, min_votes, min_return=k)
        dists   = l2_distances(self.data, q, fine)
        top     = np.argsort(dists)[:k]
        return self.data[fine[top]], dists[top], fine[top]
