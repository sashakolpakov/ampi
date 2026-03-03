# AMPI: Adaptive Multi-Projection Index

See the [top-level README](../README.md) for full documentation including
mathematical guarantees, parameter guide, and worked examples.

## Usage

```python
from ampi_hashing import AMPIIndex   # or ampi_binary, ampi_subspace

index = AMPIIndex(data, num_projections=16, bucket_size=1.0, seed=0)
points, dists, indices = index.query(query, k=10, probes=2)
candidates = index.query_candidates(query, probes=2)  # pre-rerank pool
```

## Choosing `bucket_size`

Set `bucket_size ≈ r_nn / √d` where `r_nn` is the typical nearest-neighbor
distance in your dataset.  For `N(0,1)` data in `d=128` with `n=100k`,
this is `≈ 0.88`.  See the top-level README for the full derivation.
