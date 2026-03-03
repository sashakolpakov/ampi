"""Shared Numba JIT kernels used by all AMPI backends."""

import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True, parallel=True)
def project_data(data, proj_dirs):
    """Project n data points onto L directions.

    Parameters
    ----------
    data      : (n, d) float32
    proj_dirs : (L, d) float32  — unit vectors

    Returns
    -------
    out : (L, n) float32  — proj_dirs[i] · data[k]  for each i, k
    """
    n, d = data.shape
    L = proj_dirs.shape[0]
    out = np.zeros((L, n), dtype=np.float32)
    for i in prange(L):
        for k in range(n):
            dot = 0.0
            for j in range(d):
                dot += proj_dirs[i, j] * data[k, j]
            out[i, k] = dot
    return out


@jit(nopython=True, cache=True, parallel=True)
def l2_distances(data, query, candidate_indices):
    """Squared L2 distances from query to a subset of data rows.

    Parameters
    ----------
    data              : (n, d) float32
    query             : (d,)   float32
    candidate_indices : (m,)   int32

    Returns
    -------
    dists : (m,) float32  — ‖data[candidate_indices[i]] − query‖²
    """
    m = len(candidate_indices)
    d = data.shape[1]
    dists = np.zeros(m, dtype=np.float32)

    q_norm2 = 0.0
    for j in range(d):
        q_norm2 += query[j] * query[j]

    for i in prange(m):
        idx = candidate_indices[i]
        dot = 0.0
        x_norm2 = 0.0
        for j in range(d):
            x = data[idx, j]
            x_norm2 += x * x
            dot += x * query[j]
        dists[i] = x_norm2 + q_norm2 - 2.0 * dot

    return dists


@jit(nopython=True, cache=True)
def jit_union_query(sorted_idxs, sorted_projs, q_projs, window_size):
    """Union-mode candidate selection — fully JIT compiled.

    For each projection i, every point within ±window_size of the query's
    position gets included.  Returns the union as a sorted int32 array.

    Uses a dense boolean mask (O(n) memory, O(L·w + n) work) — fast and
    cache-friendly; no Python overhead at all.
    """
    L, n = sorted_idxs.shape
    in_window = np.zeros(n, dtype=np.uint8)
    for i in range(L):
        pos = np.searchsorted(sorted_projs[i], q_projs[i])
        lo  = max(np.int64(0), pos - window_size)
        hi  = min(np.int64(n), pos + window_size)
        for j in range(lo, hi):
            in_window[sorted_idxs[i, j]] = 1
    count = np.int64(0)
    for k in range(n):
        if in_window[k]:
            count += 1
    result = np.empty(count, dtype=np.int32)
    c = np.int64(0)
    for k in range(n):
        if in_window[k]:
            result[c] = k
            c += 1
    return result


@jit(nopython=True, cache=True)
def jit_vote_query(sorted_idxs, sorted_projs, q_projs, window_size, min_votes):
    """Voting-mode candidate selection — fully JIT compiled.

    Each point accumulates one vote per projection window it appears in.
    Only points with vote_count >= min_votes are returned.

    True NNs appear in ~all L windows; random false positives in ~L·2w/n ≈ 0.
    At min_votes=2 and typical parameters this shrinks the candidate pool
    10–100× vs union while preserving recall for true nearest neighbours.
    """
    L, n = sorted_idxs.shape
    votes = np.zeros(n, dtype=np.int32)
    for i in range(L):
        pos = np.searchsorted(sorted_projs[i], q_projs[i])
        lo  = max(np.int64(0), pos - window_size)
        hi  = min(np.int64(n), pos + window_size)
        for j in range(lo, hi):
            votes[sorted_idxs[i, j]] += 1
    count = np.int64(0)
    for k in range(n):
        if votes[k] >= min_votes:
            count += 1
    result = np.empty(count, dtype=np.int32)
    c = np.int64(0)
    for k in range(n):
        if votes[k] >= min_votes:
            result[c] = k
            c += 1
    return result
