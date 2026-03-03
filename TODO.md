# AMPI TODO

## 1. Algorithm & index

- [ ] **C++ core** — the QPS gap vs FAISS IVF is purely implementation; recall/candidates
      curve is competitive on MNIST; a C++ extension with Python bindings (pybind11)
      would close the gap. Priority: `jit_union_query`, `l2_distances`, `_vote_filter`.
- [ ] **TwoStage parameter auto-tuning** — `fine_window` (fraction) and `min_votes`
      currently need manual sweep; could be auto-set from a small calibration sample.
- [ ] **Incremental insert** — add new vectors without full rebuild:
      - TomographicIndex: insert into sorted projection arrays (O(log n) per direction)
      - SubspaceIndex / TwoStageIndex: hash new point, append to existing tables
- [ ] **Persistence** — `save(path)` / `load(path)` via `np.savez` or HDF5
- [ ] **Filtered search** — `query(q, k, filter=fn)` on metadata; major differentiator
      vs FAISS (which has no native metadata filtering)

---

## 2. Benchmarking

### Datasets
- [x] Fashion-MNIST (60k, d=784)
- [x] SIFT-128 full 1M
- [x] GloVe 1.2M d=100 (L2-normalised)
- [ ] GIST (1M, d=960) — high-d stress test

### Before ann-benchmarks submission
- [ ] Run on full SIFT-1M — structured data should favour AMPI over iid Gaussian
- [ ] Tune `bins_per_axis`, `subspace_dim`, `fine_window` per dataset
- [ ] Write ann-benchmarks wrapper:
      `module.py` with `fit` / `set_query_arguments` / `query`, `Dockerfile`, `config.yml`
- [ ] Target: competitive on MNIST and SIFT before opening PR

---

## 3. Packaging

- [ ] Publish to PyPI once recall is competitive
- [ ] Add CI: import + smoke tests on push
- [ ] Pin numba/numpy versions in pyproject.toml
