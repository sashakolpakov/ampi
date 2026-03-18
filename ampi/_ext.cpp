/*
 * AMPI C++ kernel extension — pybind11
 *
 * Implements the four hot-path functions from _kernels.py:
 *   project_data    (n,d)×(L,d)^T → (L,n)   build-time projection
 *   l2_distances    squared L2 to a candidate subset
 *   union_query     binary search + boolean mask union
 *   vote_query      binary search + per-point vote counting
 *
 * Also exposes:
 *   SortedCone      mutable per-cone sorted structure for streaming insertion
 *
 * Build with:
 *   pip install pybind11
 *   pip install -e .        (runs setup.py which compiles this file)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "_gemm.hpp"   // portable SGEMM dispatcher (Accelerate / OpenBLAS / MKL / native SIMD)

#include <algorithm>   // std::max, std::min, std::lower_bound, std::sort, std::remove_if
#include <cstdint>
#include <limits>
#include <unordered_set>
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
//
// Computes out (L×n) = proj_dirs (L×d) @ data (n×d)^T.
// Delegated to ampi::sgemm which dispatches to the best available backend:
//   Accelerate / OpenBLAS / MKL → cblas_sgemm
//   fallback                    → tiled AVX2 / NEON / scalar micro-kernel

py::array_t<float> project_data(
    py::array_t<float, py::array::c_style | py::array::forcecast> data,
    py::array_t<float, py::array::c_style | py::array::forcecast> proj_dirs)
{
    auto D = data.unchecked<2>();
    auto P = proj_dirs.unchecked<2>();
    const int n = static_cast<int>(D.shape(0));
    const int d = static_cast<int>(D.shape(1));
    const int L = static_cast<int>(P.shape(0));

    auto out = py::array_t<float>({(py::ssize_t)L, (py::ssize_t)n});

    // C (L×n) = P (L×d) @ D^T (d×n)   →  transA=false, transB=true
    ampi::sgemm(L, n, d,
                &P(0, 0), d,    // A = proj_dirs, lda = d
                &D(0, 0), d,    // B = data,      ldb = d  (transposed)
                out.mutable_data(), n,
                /*transA=*/false, /*transB=*/true);

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

// ── SortedCone ────────────────────────────────────────────────────────────────
//
// Mutable per-cone data structure for streaming insertion (Phase 1).
//
// Each cone maintains F sorted arrays, one per fan axis.  Each entry is a
// (projection_value, global_id) pair.  Logical deletes use a tombstone set;
// physical removal is deferred to compact().

class SortedCone {
public:
    int F;
    // axes[l] = pairs sorted by .first (projection value on fan axis l)
    std::vector<std::vector<std::pair<float, uint32_t>>> axes;
    std::unordered_set<uint32_t> tombstones;

    explicit SortedCone(int F_) : F(F_), axes(F_) {}

    // Build from the numpy sorted arrays produced by the Python build path.
    //   sorted_projs : (F, n_f) float32  — proj value at sorted position [l][i]
    //   sorted_idxs  : (F, n_f) int32    — local index at sorted position [l][i]
    //   global_idx   : (n_f,)   int32    — local index → global id
    static SortedCone from_arrays(
        py::array_t<float,   py::array::c_style | py::array::forcecast> sorted_projs,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> sorted_idxs,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> global_idx)
    {
        auto SP = sorted_projs.unchecked<2>();
        auto SI = sorted_idxs.unchecked<2>();
        auto GI = global_idx.unchecked<1>();
        int F_  = (int)SP.shape(0);
        int n_f = (int)SP.shape(1);

        SortedCone cone(F_);
        for (int l = 0; l < F_; ++l) {
            cone.axes[l].reserve(n_f);
            for (int i = 0; i < n_f; ++i)
                cone.axes[l].push_back({SP(l, i), (uint32_t)GI(SI(l, i))});
        }
        return cone;
    }

    // Insert a new point into all F sorted arrays.  O(F * (log n + n)) due to
    // vector shift; acceptable for n_cone ≤ ~10k (cache-friendly).
    // proj_values : (F,) float32
    void insert(
        py::array_t<float, py::array::c_style | py::array::forcecast> proj_values,
        uint32_t global_id)
    {
        auto PV = proj_values.unchecked<1>();
        tombstones.erase(global_id);   // re-insert clears any prior tombstone
        for (int l = 0; l < F; ++l) {
            auto& ax  = axes[l];
            float  pv = PV(l);
            auto   it = std::lower_bound(ax.begin(), ax.end(),
                            std::make_pair(pv, uint32_t(0)),
                            [](const std::pair<float,uint32_t>& a,
                               const std::pair<float,uint32_t>& b) {
                                return a.first < b.first;
                            });
            ax.insert(it, {pv, global_id});
        }
    }

    // Logical delete: O(1).
    void remove(uint32_t global_id) {
        tombstones.insert(global_id);
    }

    // Physical removal of all tombstoned entries: O(F * n).
    void compact() {
        if (tombstones.empty()) return;
        for (int l = 0; l < F; ++l) {
            auto& ax = axes[l];
            ax.erase(
                std::remove_if(ax.begin(), ax.end(),
                    [&](const std::pair<float,uint32_t>& p) {
                        return tombstones.count(p.second) > 0;
                    }),
                ax.end());
        }
        tombstones.clear();
    }

    // Number of live (non-tombstoned) entries.
    int size() const {
        int total = axes.empty() ? 0 : (int)axes[0].size();
        return total - (int)tombstones.size();
    }

    // All live global IDs (sorted ascending).  Used when the whole cone is
    // returned without window filtering.
    py::array_t<int32_t> all_ids() const {
        std::vector<int32_t> ids;
        if (!axes.empty()) {
            ids.reserve(axes[0].size());
            for (auto& p : axes[0]) {
                if (!tombstones.count(p.second))
                    ids.push_back((int32_t)p.second);
            }
        }
        std::sort(ids.begin(), ids.end());
        auto out = py::array_t<int32_t>((py::ssize_t)ids.size());
        auto buf = out.mutable_unchecked<1>();
        for (size_t i = 0; i < ids.size(); ++i) buf((py::ssize_t)i) = ids[i];
        return out;
    }

    // Union query: binary search + window on each of F axes, return live
    // global IDs (sorted ascending).
    // q_projs : (F,) float32,  window_size : half-window per axis
    py::array_t<int32_t> query(
        py::array_t<float, py::array::c_style | py::array::forcecast> q_projs,
        int64_t window_size) const
    {
        auto QP = q_projs.unchecked<1>();
        std::unordered_set<uint32_t> seen;

        for (int l = 0; l < F; ++l) {
            const auto& ax = axes[l];
            int64_t n = (int64_t)ax.size();
            if (n == 0) continue;
            float qp = QP(l);
            // Binary search for lower bound of qp
            int64_t lo = 0, hi = n;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (ax[mid].first < qp) lo = mid + 1;
                else hi = mid;
            }
            int64_t start = std::max(int64_t(0), lo - window_size);
            int64_t end   = std::min(n, lo + window_size);
            for (int64_t i = start; i < end; ++i) {
                uint32_t gid = ax[i].second;
                if (!tombstones.count(gid))
                    seen.insert(gid);
            }
        }

        std::vector<int32_t> result(seen.begin(), seen.end());
        std::sort(result.begin(), result.end());
        auto out = py::array_t<int32_t>((py::ssize_t)result.size());
        auto buf = out.mutable_unchecked<1>();
        for (size_t i = 0; i < result.size(); ++i)
            buf((py::ssize_t)i) = result[i];
        return out;
    }

    // Early-stopping coverage check (used by the adaptive query loop).
    // Returns true if any axis l has both window boundaries >= kth_proj away
    // from the query projection, guaranteeing no unvisited point can improve
    // the current top-k (by the Cauchy-Schwarz lower bound L2 >= |proj_l gap|).
    bool is_covered(
        py::array_t<float, py::array::c_style | py::array::forcecast> q_projs,
        int64_t w, float kth_proj) const
    {
        auto QP = q_projs.unchecked<1>();
        const float inf = std::numeric_limits<float>::infinity();
        for (int l = 0; l < F; ++l) {
            const auto& ax = axes[l];
            int64_t n = (int64_t)ax.size();
            if (n == 0) continue;
            float qp = QP(l);
            // Binary search
            int64_t lo = 0, hi = n;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (ax[mid].first < qp) lo = mid + 1;
                else hi = mid;
            }
            int64_t win_lo = std::max(int64_t(0), lo - w);
            int64_t win_hi = std::min(n, lo + w);
            float gap_right = (win_hi < n)  ? (ax[win_hi].first - qp)       : inf;
            float gap_left  = (win_lo > 0)  ? (qp - ax[win_lo - 1].first)   : inf;
            if (std::min(gap_right, gap_left) >= kth_proj)
                return true;
        }
        return false;
    }
};

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

    py::class_<SortedCone>(m, "SortedCone",
        "Mutable sorted-projection cone supporting streaming insert/delete.\n\n"
        "Each cone holds F sorted arrays of (proj_value, global_id) pairs.\n"
        "Logical deletes use tombstones; compact() does physical removal.")
        .def(py::init<int>(), py::arg("F"),
             "Create an empty cone with F fan axes.")
        .def_static("from_arrays", &SortedCone::from_arrays,
             py::arg("sorted_projs"), py::arg("sorted_idxs"), py::arg("global_idx"),
             "Build a SortedCone from the numpy arrays produced by the Python build path.\n\n"
             "  sorted_projs : (F, n_f) float32\n"
             "  sorted_idxs  : (F, n_f) int32   (local indices into global_idx)\n"
             "  global_idx   : (n_f,)   int32")
        .def("insert",  &SortedCone::insert,
             py::arg("proj_values"), py::arg("global_id"),
             "Insert a point.  proj_values must be (F,) float32.")
        .def("remove",  &SortedCone::remove,
             py::arg("global_id"),
             "Tombstone a point (logical delete, O(1)).")
        .def("compact", &SortedCone::compact,
             "Physically remove all tombstoned entries from the sorted arrays.")
        .def("size",    &SortedCone::size,
             "Number of live (non-tombstoned) entries.")
        .def("all_ids", &SortedCone::all_ids,
             "All live global IDs as a sorted int32 array.")
        .def("query",   &SortedCone::query,
             py::arg("q_projs"), py::arg("window_size"),
             "Union-window query.  Returns sorted int32 global IDs.")
        .def("is_covered", &SortedCone::is_covered,
             py::arg("q_projs"), py::arg("w"), py::arg("kth_proj"),
             "True if any axis l guarantees no unvisited point can enter the top-k.");
}
