"""
AMPI: Fast version with numba - dict-based hash tables.
"""

import numpy as np
from numba import jit, prange
from collections import defaultdict


@jit(nopython=True, cache=True, parallel=True)
def _compute_projections(data, proj_dirs):
    n, d = data.shape
    L = proj_dirs.shape[0]
    projections = np.zeros((L, n), dtype=np.float32)
    
    for i in prange(L):
        for k in range(n):
            dot = 0.0
            for j in range(d):
                dot += data[k, j] * proj_dirs[i, j]
            projections[i, k] = dot
    
    return projections


@jit(nopython=True, cache=True, parallel=True)
def _compute_distances(data, query, candidate_indices):
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


class AMPIIndex:
    def __init__(self, data, num_projections=10, bucket_size=1.0):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.n, self.d = self.data.shape
        self.num_projections = num_projections
        self.bucket_size = bucket_size
        
        rng = np.random.RandomState(0)
        self.proj_dirs = rng.randn(num_projections, self.d).astype(np.float32)
        norms = np.linalg.norm(self.proj_dirs, axis=1, keepdims=True)
        self.proj_dirs /= norms
        
        projections = _compute_projections(self.data, self.proj_dirs)
        
        self.tables = []
        for i in range(num_projections):
            buckets = defaultdict(list)
            for idx, p in enumerate(projections[i]):
                bucket_id = int(p / bucket_size)
                buckets[bucket_id].append(idx)
            # Convert to dict with numpy arrays
            table = {}
            for k, v in buckets.items():
                table[k] = np.array(v, dtype=np.int32)
            self.tables.append(table)
        
        self.data_norms = np.sum(self.data ** 2, axis=1)
    
    def query(self, q, k=5, probes=3):
        q = np.ascontiguousarray(q, dtype=np.float32)
        
        candidates = set()
        q_projs = q @ self.proj_dirs.T
        
        for i in range(self.num_projections):
            proj = q_projs[i]
            bucket_id = int(proj / self.bucket_size)
            
            for b in range(bucket_id - probes + 1, bucket_id + probes):
                if b in self.tables[i]:
                    candidates.update(self.tables[i][b])
        
        if len(candidates) == 0:
            indices = np.random.choice(self.n, k, replace=False)
            points = self.data[indices]
            dists = np.sum((points - q) ** 2, axis=1)
            return points, dists, indices
        
        candidates = np.array(list(candidates))
        
        dists = _compute_distances(self.data, q, candidates)
        order = np.argsort(dists)[:k]
        
        return self.data[candidates[order]], dists[order], candidates[order]


__version__ = "0.1.0"
__all__ = ["AMPIIndex"]
