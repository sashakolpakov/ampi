/*
 * AMPI C++ kernel extension — pybind11
 *
 * Implements the four hot-path functions from _kernels.py:
 *   project_data    (n,d)×(L,d)^T → (L,n)   build-time projection
 *   l2_distances    squared L2 to a candidate subset
 *   union_query     binary search + boolean mask union
 *   vote_query      binary search + per-point vote counting
 *
 * Build with:
 *   pip install pybind11
 *   pip install -e .        (runs setup.py which compiles this file)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>   // std::max, std::min, std::lower_bound
#include <cstdint>
#include <vector>

namespace py = pybind11;

// ── helpers ──────────────────────────────────────────────────────────────────

// Branchless lower-bound on a contiguous float array.
static inline int64_t lb_float(const float* arr, int64_t n, float val) {
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (arr[mid] < val) lo = mid + 1;
        else                hi = mid;
    }
    return lo;
}

// ── project_data ─────────────────────────────────────────────────────────────

py::array_t<float> project_data(
    py::array_t<float, py::array::c_style | py::array::forcecast> data,
    py::array_t<float, py::array::c_style | py::array::forcecast> proj_dirs)
{
    auto D  = data.unchecked<2>();
    auto P  = proj_dirs.unchecked<2>();
    const int64_t n = D.shape(0), d = D.shape(1), L = P.shape(0);

    auto out     = py::array_t<float>({L, n});
    auto out_buf = out.mutable_unchecked<2>();

    for (int64_t i = 0; i < L; ++i) {
        const float* dir = &P(i, 0);
        for (int64_t k = 0; k < n; ++k) {
            const float* row = &D(k, 0);
            float dot = 0.f;
            for (int64_t j = 0; j < d; ++j)
                dot += dir[j] * row[j];
            out_buf(i, k) = dot;
        }
    }
    return out;
}

// ── l2_distances ─────────────────────────────────────────────────────────────

py::array_t<float> l2_distances(
    py::array_t<float,   py::array::c_style | py::array::forcecast> data,
    py::array_t<float,   py::array::c_style | py::array::forcecast> query,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> candidate_indices)
{
    auto D    = data.unchecked<2>();
    auto q    = query.unchecked<1>();
    auto cand = candidate_indices.unchecked<1>();
    const int64_t m = cand.shape(0);
    const int64_t d = q.shape(0);

    // Precompute ‖q‖²
    float q_norm2 = 0.f;
    for (int64_t j = 0; j < d; ++j)
        q_norm2 += q(j) * q(j);

    auto out     = py::array_t<float>(m);
    auto out_buf = out.mutable_unchecked<1>();

    for (int64_t i = 0; i < m; ++i) {
        const float* row = &D(cand(i), 0);
        float dot = 0.f, x_norm2 = 0.f;
        for (int64_t j = 0; j < d; ++j) {
            float x = row[j];
            x_norm2 += x * x;
            dot     += x * q(j);
        }
        out_buf(i) = x_norm2 + q_norm2 - 2.f * dot;
    }
    return out;
}

// ── union_query ───────────────────────────────────────────────────────────────

py::array_t<int32_t> union_query(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> sorted_idxs,
    py::array_t<float,   py::array::c_style | py::array::forcecast> sorted_projs,
    py::array_t<float,   py::array::c_style | py::array::forcecast> q_projs,
    int64_t window_size)
{
    auto SI = sorted_idxs.unchecked<2>();
    auto SP = sorted_projs.unchecked<2>();
    auto QP = q_projs.unchecked<1>();
    const int64_t L = SI.shape(0), n = SI.shape(1);

    std::vector<uint8_t> mask(n, 0);

    for (int64_t i = 0; i < L; ++i) {
        const float*   sp = &SP(i, 0);
        const int32_t* si = &SI(i, 0);
        int64_t pos = lb_float(sp, n, QP(i));
        int64_t lo  = std::max(int64_t(0), pos - window_size);
        int64_t hi  = std::min(n,           pos + window_size);
        for (int64_t j = lo; j < hi; ++j)
            mask[si[j]] = 1;
    }

    int64_t count = 0;
    for (int64_t k = 0; k < n; ++k)
        if (mask[k]) ++count;

    auto out     = py::array_t<int32_t>(count);
    auto out_buf = out.mutable_unchecked<1>();
    int64_t c = 0;
    for (int64_t k = 0; k < n; ++k)
        if (mask[k]) out_buf(c++) = static_cast<int32_t>(k);

    return out;
}

// ── module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_ampi_ext, m) {
    m.doc() = "AMPI C++ kernel extension (pybind11)";
    m.def("project_data", &project_data,
          "Project (n,d) data onto L unit directions → (L,n) float32",
          py::arg("data"), py::arg("proj_dirs"));
    m.def("l2_distances", &l2_distances,
          "Squared L2 distances from query to candidate subset",
          py::arg("data"), py::arg("query"), py::arg("candidate_indices"));
    m.def("union_query",  &union_query,
          "Union-mode candidate selection via sorted projections",
          py::arg("sorted_idxs"), py::arg("sorted_projs"),
          py::arg("q_projs"), py::arg("window_size"));
}
