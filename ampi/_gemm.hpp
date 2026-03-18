#pragma once
/*
 * ampi/_gemm.hpp  —  portable single-header SGEMM dispatcher
 *
 * Implements:  C = op(A) * op(B),  row-major float32,  alpha=1, beta=0.
 *
 * Architecture credit: the compile-time 3-tier BLAS dispatch structure
 * (Accelerate → OpenBLAS → MKL → native fallback), the tile sizes for the
 * native micro-kernel, and the AVX2 horizontal-sum pattern were informed by
 * Axiom (Noah Kay, 2025); https://github.com/frikallo/axiom
 *
 * Compile-time priority (set by setup.py via -D flags):
 *   AMPI_USE_ACCELERATE  →  Apple Accelerate framework  (cblas_sgemm)
 *   AMPI_USE_OPENBLAS    →  OpenBLAS                    (cblas_sgemm)
 *   AMPI_USE_MKL         →  Intel MKL                   (cblas_sgemm)
 *   (none)               →  built-in tiled micro-kernel (see below)
 *
 * SIMD micro-kernel (native path only, auto-detected from compiler flags):
 *   __AVX2__             →  8-wide FMA   (_mm256_fmadd_ps)  x86/x86_64
 *   __ARM_NEON           →  4-wide FMA   (vfmaq_f32)        ARM / Apple Silicon
 *   (neither)            →  scalar loop  (compiler auto-vectorises with
 *                            -O3 -march=native / -mcpu=native)
 *
 * "Assembly kernels" on CPU:
 *   SIMD intrinsics are the C++ equivalent of GPU kernels — code that
 *   operates on N values in parallel using vector registers.
 *   AVX2 = 256-bit register → 8 float32s per FMA instruction.
 *   ARM NEON = 128-bit → 4 float32s per FMA instruction.
 *   Production BLAS (OpenBLAS, BLIS) write the innermost 8×4 register
 *   micro-kernel in hand-tuned .s assembly, one file per CPU
 *   microarchitecture.  Intrinsics compile to the same instructions with
 *   much less complexity and are sufficient for our hot path.
 *
 * Public API (namespace ampi):
 *   ampi::sgemm(M, N, K, A, lda, B, ldb, C, ldc,
 *               transA=false, transB=false)
 */

#include <algorithm>
#include <cstring>

// ── BLAS backend selection ────────────────────────────────────────────────────

#if defined(AMPI_USE_ACCELERATE)
#  include <Accelerate/Accelerate.h>
#  define AMPI_HAVE_CBLAS 1

#elif defined(AMPI_USE_OPENBLAS)
#  include <cblas.h>
#  define AMPI_HAVE_CBLAS 1

#elif defined(AMPI_USE_MKL)
#  include <mkl_cblas.h>
#  define AMPI_HAVE_CBLAS 1
#endif

// ── SIMD includes (native fallback only) ─────────────────────────────────────

#if !defined(AMPI_HAVE_CBLAS)
#  if defined(__AVX2__)
#    include <immintrin.h>
#  elif defined(__ARM_NEON)
#    include <arm_neon.h>
#  endif
#endif

// ═════════════════════════════════════════════════════════════════════════════
// Internal implementation details — all in an anonymous namespace so that
// multiple translation units can include this header without ODR violations.
// ═════════════════════════════════════════════════════════════════════════════

namespace {

// ── Tile sizes ────────────────────────────────────────────────────────────────
// C-tile (MC×NC) + A-strip (MC×KC) + B-strip (NC×KC) ≈ 147 KB → fits in L2.
static constexpr int AMPI_MC = 64;
static constexpr int AMPI_NC = 64;
static constexpr int AMPI_KC = 256;

// ── SIMD dot-product micro-kernel ─────────────────────────────────────────────
// dot(a, b, len): contiguous float32 dot product.
// This is the "register kernel" — processes W floats per cycle using vector FMA.

#if defined(AMPI_HAVE_CBLAS)
// Not used when BLAS is available; keep compiler happy with a no-op stub.
static inline float ampi_dot(const float*, const float*, int) noexcept { return 0.f; }

#elif defined(__AVX2__)
// x86 / x86_64: 8-wide FMA.  Horizontal reduce via hadd + extract.
static inline float ampi_dot(const float* __restrict__ a,
                              const float* __restrict__ b, int len) noexcept
{
    __m256 acc = _mm256_setzero_ps();
    int j = 0;
    for (; j + 8 <= len; j += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + j),
                              _mm256_loadu_ps(b + j), acc);
    // Horizontal sum of 8 lanes
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float s = _mm_cvtss_f32(sum);
    for (; j < len; ++j) s += a[j] * b[j];   // scalar tail
    return s;
}

#elif defined(__ARM_NEON)
// ARM / Apple Silicon: 4-wide FMA.  Reduce with vaddvq_f32.
static inline float ampi_dot(const float* __restrict__ a,
                              const float* __restrict__ b, int len) noexcept
{
    float32x4_t acc = vdupq_n_f32(0.f);
    int j = 0;
    for (; j + 4 <= len; j += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + j), vld1q_f32(b + j));
    float s = vaddvq_f32(acc);
    for (; j < len; ++j) s += a[j] * b[j];   // scalar tail
    return s;
}

#else
// Scalar fallback — compiler will auto-vectorise with -O3 -march=native.
static inline float ampi_dot(const float* __restrict__ a,
                              const float* __restrict__ b, int len) noexcept
{
    float s = 0.f;
    for (int j = 0; j < len; ++j) s += a[j] * b[j];
    return s;
}
#endif  // SIMD selection

// ── Tiled native SGEMM ────────────────────────────────────────────────────────
//
// Hot path: transA=false, transB=true  (project_data: P @ D^T).
// Both A[i, j0:jhi] and B[k, j0:jhi] are contiguous row slices →
// ampi_dot() vectorises fully with no gather overhead.
//
// Other transpose combinations use a scalar inner loop — they don't occur in
// the current AMPI kernels and can be optimised later if needed.

#if !defined(AMPI_HAVE_CBLAS)
static void ampi_sgemm_native(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float*       C, int ldc,
    bool transA, bool transB) noexcept
{
    for (int i = 0; i < M; ++i)
        std::memset(C + (size_t)i * ldc, 0, N * sizeof(float));

    for (int i0 = 0; i0 < M; i0 += AMPI_MC) {
        const int ihi = std::min(i0 + AMPI_MC, M);
        for (int k0 = 0; k0 < N; k0 += AMPI_NC) {
            const int khi = std::min(k0 + AMPI_NC, N);
            for (int j0 = 0; j0 < K; j0 += AMPI_KC) {
                const int jhi  = std::min(j0 + AMPI_KC, K);
                const int jlen = jhi - j0;

                if (!transA && transB) {
                    // Both rows are contiguous — full SIMD throughput
                    for (int i = i0; i < ihi; ++i) {
                        const float* Arow = A + (size_t)i * lda + j0;
                        for (int k = k0; k < khi; ++k) {
                            C[(size_t)i * ldc + k] +=
                                ampi_dot(Arow, B + (size_t)k * ldb + j0, jlen);
                        }
                    }
                } else if (!transA && !transB) {
                    // A row contiguous, B column-strided — scalar path
                    for (int i = i0; i < ihi; ++i) {
                        const float* Arow = A + (size_t)i * lda + j0;
                        for (int k = k0; k < khi; ++k) {
                            float acc = 0.f;
                            for (int j = 0; j < jlen; ++j)
                                acc += Arow[j] * B[(size_t)(j0 + j) * ldb + k];
                            C[(size_t)i * ldc + k] += acc;
                        }
                    }
                } else if (transA && !transB) {
                    for (int i = i0; i < ihi; ++i) {
                        for (int k = k0; k < khi; ++k) {
                            float acc = 0.f;
                            for (int j = 0; j < jlen; ++j)
                                acc += A[(size_t)(j0 + j) * lda + i]
                                     * B[(size_t)(j0 + j) * ldb + k];
                            C[(size_t)i * ldc + k] += acc;
                        }
                    }
                } else {
                    // transA && transB
                    for (int i = i0; i < ihi; ++i) {
                        for (int k = k0; k < khi; ++k) {
                            float acc = 0.f;
                            for (int j = 0; j < jlen; ++j)
                                acc += A[(size_t)(j0 + j) * lda + i]
                                     * B[(size_t)k * ldb + (j0 + j)];
                            C[(size_t)i * ldc + k] += acc;
                        }
                    }
                }
            }
        }
    }
}
#endif  // !AMPI_HAVE_CBLAS

}  // anonymous namespace

// ═════════════════════════════════════════════════════════════════════════════
// Public API
// ═════════════════════════════════════════════════════════════════════════════

namespace ampi {

// C (M×N) = op(A (M×K)) @ op(B (K×N)),  row-major float32, alpha=1, beta=0.
// lda/ldb/ldc are the column counts of the *stored* (untransposed) matrix.
inline void sgemm(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float*       C, int ldc,
    bool transA = false, bool transB = false)
{
#if defined(AMPI_HAVE_CBLAS)
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K,
                1.0f, A, lda, B, ldb,
                0.0f, C, ldc);
#else
    ampi_sgemm_native(M, N, K, A, lda, B, ldb, C, ldc, transA, transB);
#endif
}

}  // namespace ampi
