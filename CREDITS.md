# Credits

## Axiom — High-Performance C++ Tensor Library

**Noah Kay, 2025** — https://github.com/frikallo/axiom

`ampi/_gemm.hpp` was designed by studying Axiom's BLAS dispatch layer.
Specifically, the following ideas came from Axiom:

- **Compile-time 3-tier dispatch**: the pattern of selecting Accelerate →
  OpenBLAS → MKL → native fallback via preprocessor defines set by the build
  system, rather than runtime linking.
- **Tile sizes for the native micro-kernel**: Axiom's `NativeBlasBackend`
  uses 64×64×256 blocking (A-strip + B-strip + C-tile fit in L2 cache), which
  we adopted directly.
- **AVX2 horizontal-sum pattern**: the `_mm256_extractf128_ps` + double
  `_mm_hadd_ps` reduction used in `ampi_dot` mirrors Axiom's implementation.

What we did **not** take from Axiom: lazy computation graphs, the `Tensor`
class (shape/strides/dtype metadata), Metal/GPU dispatch, batch matmul,
type promotion, or any of the higher-level operator machinery. We extracted
only the narrow SGEMM path needed for `project_data`.

The underlying BLAS calls (`cblas_sgemm`) are Apple's Accelerate framework
or OpenBLAS — not Axiom's code. Axiom showed us how to structure the dispatch
cleanly in a self-contained header.

```bibtex
@misc{axiom2025,
  title  = {Axiom: High-Performance Tensor Library for C++},
  author = {Noah Kay},
  year   = {2025},
  url    = {https://github.com/frikallo/axiom}
}
```
