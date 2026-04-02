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
#include <mutex>
#include <numeric>     // std::iota
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>    // open, O_CREAT, O_RDWR
#include <fstream>    // std::ifstream (sketch load)
#include <sys/mman.h> // mmap, munmap, MAP_SHARED, PROT_READ, PROT_WRITE
#include <unistd.h>   // ftruncate, close

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
    const float* qptr = q.data(0);
    float q_norm2 = 0.f;
    for (int64_t j = 0; j < d; ++j)
        q_norm2 += qptr[j] * qptr[j];

    auto out     = py::array_t<float>(m);
    auto out_buf = out.mutable_unchecked<1>();

    if (m == 0) return out;

    // Gather candidates into a contiguous buffer then use BLAS for the dot
    // products.  Accessing `data` via a gather of row pointers causes random
    // mmap page accesses; contiguous writes here are not free, but the BLAS
    // M×d @ d×1 is then a single vectorised GEMV call instead of m scalar
    // inner loops, giving 4–10× speedup on warm-cache data.
    std::vector<float> gathered((size_t)m * d);
    std::vector<float> row_norms2(m, 0.f);
    for (int64_t i = 0; i < m; ++i) {
        const float* row = &D(cand(i), 0);
        float* dst = gathered.data() + i * d;
        float  ns  = 0.f;
        for (int64_t j = 0; j < d; ++j) { dst[j] = row[j]; ns += row[j] * row[j]; }
        row_norms2[i] = ns;
    }

    // dots[i] = gathered[i,:] · q  — sequential access, auto-vectorised by compiler
    std::vector<float> dots(m, 0.f);
    for (int64_t i = 0; i < m; ++i) {
        const float* row = gathered.data() + i * d;
        float dot = 0.f;
        for (int64_t j = 0; j < d; ++j) dot += row[j] * qptr[j];
        dots[i] = dot;
    }

    for (int64_t i = 0; i < m; ++i)
        out_buf(i) = row_norms2[i] + q_norm2 - 2.f * dots[i];
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

    // Collect window-hit IDs into a local vector, then sort+unique.
    // Avoids two O(n) scans over the mask array (n can be 1 M for GIST-1M).
    // sort+unique gives the same sorted-ID output as the mask approach.
    std::vector<int32_t> hits;
    hits.reserve(static_cast<size_t>(L) * static_cast<size_t>(window_size) * 2);

    for (int64_t i = 0; i < L; ++i) {
        const float*   sp = &SP(i, 0);
        const int32_t* si = &SI(i, 0);
        int64_t pos = lb_float(sp, n, QP(i));
        int64_t lo  = std::max(int64_t(0), pos - window_size);
        int64_t hi  = std::min(n,           pos + window_size);
        for (int64_t j = lo; j < hi; ++j)
            hits.push_back(si[j]);
    }

    std::sort(hits.begin(), hits.end());
    hits.erase(std::unique(hits.begin(), hits.end()), hits.end());

    auto out     = py::array_t<int32_t>(static_cast<py::ssize_t>(hits.size()));
    auto out_buf = out.mutable_unchecked<1>();
    for (int64_t k = 0; k < static_cast<int64_t>(hits.size()); ++k)
        out_buf(k) = hits[k];

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

    // ── internal raw methods (no numpy overhead, called from AMPIIndex) ──────

    void insert_raw(const float* proj_values, uint32_t global_id) {
        tombstones.erase(global_id);
        for (int l = 0; l < F; ++l) {
            auto& ax = axes[l];
            float pv = proj_values[l];
            auto it = std::lower_bound(ax.begin(), ax.end(),
                          std::make_pair(pv, uint32_t(0)),
                          [](const std::pair<float,uint32_t>& a,
                             const std::pair<float,uint32_t>& b) {
                              return a.first < b.first;
                          });
            ax.insert(it, {pv, global_id});
        }
    }

    std::vector<uint32_t> query_raw(const float* q_projs, int64_t window_size) const {
        std::unordered_set<uint32_t> seen;
        for (int l = 0; l < F; ++l) {
            const auto& ax = axes[l];
            int64_t n_ax = (int64_t)ax.size();
            if (n_ax == 0) continue;
            float qp = q_projs[l];
            int64_t lo = 0, hi = n_ax;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (ax[mid].first < qp) lo = mid + 1; else hi = mid;
            }
            int64_t start = std::max(int64_t(0), lo - window_size);
            int64_t end   = std::min(n_ax, lo + window_size);
            for (int64_t i = start; i < end; ++i) {
                uint32_t gid = ax[i].second;
                if (!tombstones.count(gid)) seen.insert(gid);
            }
        }
        std::vector<uint32_t> result(seen.begin(), seen.end());
        std::sort(result.begin(), result.end());
        return result;
    }

    bool is_covered_raw(const float* q_projs, int64_t w, float kth_proj) const {
        const float inf = std::numeric_limits<float>::infinity();
        for (int l = 0; l < F; ++l) {
            const auto& ax = axes[l];
            int64_t n_ax = (int64_t)ax.size();
            if (n_ax == 0) continue;
            float qp = q_projs[l];
            int64_t lo = 0, hi = n_ax;
            while (lo < hi) {
                int64_t mid = (lo + hi) >> 1;
                if (ax[mid].first < qp) lo = mid + 1; else hi = mid;
            }
            int64_t win_lo = std::max(int64_t(0), lo - w);
            int64_t win_hi = std::min(n_ax, lo + w);
            float gap_right = (win_hi < n_ax) ? (ax[win_hi].first - qp)     : inf;
            float gap_left  = (win_lo > 0)    ? (qp - ax[win_lo-1].first)   : inf;
            if (std::min(gap_right, gap_left) >= kth_proj) return true;
        }
        return false;
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

    // Return all (proj_value, global_id) pairs for axis l as two parallel arrays.
    // Includes tombstoned entries — call compact() first if a clean snapshot is needed.
    // Used by the Phase-2 checkpoint serializer.
    py::tuple get_axis_pairs(int l) const {
        if (l < 0 || l >= (int)axes.size())
            throw py::index_error("axis index " + std::to_string(l) +
                                  " out of range [0, " + std::to_string(axes.size()) + ")");
        const auto& ax = axes[l];
        py::ssize_t n_f = (py::ssize_t)ax.size();
        auto projs = py::array_t<float>(n_f);
        auto ids   = py::array_t<uint32_t>(n_f);
        auto bp = projs.mutable_unchecked<1>();
        auto bi = ids.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n_f; ++i) {
            bp(i) = ax[i].first;
            bi(i) = (uint32_t)ax[i].second;
        }
        return py::make_tuple(projs, ids);
    }
};

// ── AMPIIndex ─────────────────────────────────────────────────────────────────
//
// Owns all mutable index state: data buffer, centroid table, per-cluster
// SortedCones, drift covariance, and the inverse (global_id → cone) map.
//
// Phase 3: add() and remove() are pure C++.
// Phase 4 will add query() here.

class AMPIIndex {
public:
    int d, F, nlist, cone_top_k;
    double drift_theta;
    bool cosine_metric;

    // Heap-mode data buffer.  Stored behind shared_ptr so that numpy arrays
    // returned by get_data_view() keep the allocation alive after a grow
    // creates a new one.  Null in mmap mode.
    std::shared_ptr<std::vector<float>> _data_buf_sp;
    float*               _data_ptr = nullptr; // always valid: _data_buf_sp->data() or mmap addr
    // ── mmap backing store (set by from_build when data_path is provided) ────
    std::string          _mmap_path;
    int                  _mmap_fd   = -1;
    void*                _mmap_addr = MAP_FAILED;
    size_t               _mmap_size = 0;
    std::vector<uint8_t> del_mask;   // capacity (0 = live, 1 = deleted)
    uint32_t n, capacity, n_deleted;

    // Phase 6: reader-writer lock.  query/query_candidates hold shared lock;
    // add/remove/batch_add/batch_delete/local_refresh hold exclusive lock.
    // Stored behind unique_ptr so AMPIIndex remains movable (shared_mutex is not).
    std::unique_ptr<std::shared_mutex> p_mutex = std::make_unique<std::shared_mutex>();

    std::vector<float> axes;       // F * d (immutable after build)
    std::vector<float> centroids;  // nlist * d (EMA-updated)

    std::vector<int64_t>  cluster_counts;
    std::vector<int64_t>  cluster_tombstones;
    std::vector<bool>     cluster_has_cones;
    std::vector<std::vector<uint32_t>>   cluster_global;
    std::vector<std::vector<SortedCone>> cluster_cones;   // [nlist][F]
    std::vector<std::vector<float>>      U_drift;         // [nlist][d*F], row-major (d rows, F cols)
    // Per-cluster fan axes (F*d floats each); empty vector means use global axes.
    // Populated by _compute_cluster_axes() during _local_refresh.
    std::vector<std::vector<float>>      cluster_axes;    // [nlist]

    // Sketch table: sketch[gid*F + f] = dot(x_gid, global_axis_f).
    // Bessel: ||sketch(q) - sketch(x)||² ≤ ||q - x||² for any orthonormal A.
    // Used as a cheap in-RAM lower bound to prune candidates before mmap access.
    std::vector<float> sketch;   // capacity × F, row-major

    // point_cones[gid] = {(cluster, fan), ...}
    std::vector<std::vector<std::pair<uint16_t,uint16_t>>> point_cones;

    // Periodic cluster merge parameters.
    int      merge_interval   = 0;    // 0 = disabled
    double   eps_merge        = 1.0;  // centroid L2 distance threshold
    double   merge_qe_ratio   = 0.5;  // merge if δ_qe ≤ ratio*(mQE_i+mQE_j)
    uint64_t insert_count     = 0;    // total inserts processed

    // cppcheck-suppress uninitMemberVar
    AMPIIndex() = default;

    // Move constructor: transfer mmap ownership so the destructor only runs
    // on the live object.  Needed because ~AMPIIndex() suppresses the implicit
    // move constructor (C++11 rule of five).
    AMPIIndex(AMPIIndex&& o) noexcept
        : _data_buf_sp(std::move(o._data_buf_sp)),
          _data_ptr(o._data_ptr),
          _mmap_path(std::move(o._mmap_path)),
          _mmap_fd(o._mmap_fd),
          _mmap_addr(o._mmap_addr),
          _mmap_size(o._mmap_size),
          del_mask(std::move(o.del_mask)),
          n(o.n), capacity(o.capacity), n_deleted(o.n_deleted),
          p_mutex(std::move(o.p_mutex)),
          d(o.d), F(o.F), nlist(o.nlist), cone_top_k(o.cone_top_k),
          drift_theta(o.drift_theta), cosine_metric(o.cosine_metric),
          axes(std::move(o.axes)),
          centroids(std::move(o.centroids)),
          cluster_counts(std::move(o.cluster_counts)),
          cluster_tombstones(std::move(o.cluster_tombstones)),
          U_drift(std::move(o.U_drift)),
          cluster_global(std::move(o.cluster_global)),
          cluster_cones(std::move(o.cluster_cones)),
          cluster_has_cones(std::move(o.cluster_has_cones)),
          cluster_axes(std::move(o.cluster_axes)),
          sketch(std::move(o.sketch)),
          point_cones(std::move(o.point_cones)),
          merge_interval(o.merge_interval),
          eps_merge(o.eps_merge),
          merge_qe_ratio(o.merge_qe_ratio),
          insert_count(o.insert_count)
    {
        // Prevent double-close/unmap in the moved-from destructor.
        o._data_ptr  = nullptr;
        o._mmap_fd   = -1;
        o._mmap_addr = MAP_FAILED;
        o._mmap_size = 0;
    }

    // ── construction ─────────────────────────────────────────────────────────

    static AMPIIndex from_build(
        int d, int F, int nlist, int cone_top_k,
        double drift_theta, bool cosine,
        py::array_t<float,   py::array::c_style | py::array::forcecast> axes_np,
        py::array_t<float,   py::array::c_style | py::array::forcecast> centroids_np,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> cluster_counts_np,
        py::array_t<float,   py::array::c_style | py::array::forcecast> U_drift_np,
        py::array_t<float,   py::array::c_style | py::array::forcecast> data_np,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> del_mask_np,
        int n_init,
        py::list  cluster_global_list,  // [nlist] of int32 numpy arrays
        const std::string& data_path = "")     // when non-empty: mmap data to {data_path}/_cpp_data_buf.dat
    {
        AMPIIndex idx;
        idx.d             = d;
        idx.F             = F;
        idx.nlist         = nlist;
        idx.cone_top_k    = cone_top_k;
        idx.drift_theta   = drift_theta;
        idx.cosine_metric = cosine;
        idx.n             = (uint32_t)n_init;
        idx.n_deleted     = 0;

        const int64_t cap = (int64_t)data_np.shape(0);
        idx.capacity = (uint32_t)cap;

        // axes
        auto AX = axes_np.unchecked<2>();
        idx.axes.resize((size_t)F * d);
        for (int l = 0; l < F; ++l)
            for (int j = 0; j < d; ++j)
                idx.axes[l * d + j] = AX(l, j);

        // centroids
        auto CN = centroids_np.unchecked<2>();
        idx.centroids.resize((size_t)nlist * d);
        for (int c = 0; c < nlist; ++c)
            for (int j = 0; j < d; ++j)
                idx.centroids[c * d + j] = CN(c, j);

        // data buffer (full pre-allocated capacity)
        // mmap mode: write to a memory-mapped file so the OS can page out idle
        // clusters, keeping RSS proportional to the working set rather than n.
        size_t data_sz = (size_t)cap * d * sizeof(float);
        auto DN = data_np.unchecked<2>();
        if (!data_path.empty()) {
            std::string fpath = data_path + "/_cpp_data_buf.dat";
            idx._mmap_open(fpath, data_sz);
            float* mp = idx._data_ptr;
            for (int64_t i = 0; i < (int64_t)n_init; ++i)
                for (int j = 0; j < d; ++j)
                    mp[i * d + j] = DN(i, j);
            // headroom slots are already zeroed by the OS (mmap of a new file)
        } else {
            idx._data_buf_sp = std::make_shared<std::vector<float>>((size_t)cap * d);
            for (int64_t i = 0; i < cap; ++i)
                for (int j = 0; j < d; ++j)
                    (*idx._data_buf_sp)[i * d + j] = DN(i, j);
            idx._data_ptr = idx._data_buf_sp->data();
        }

        // deleted mask
        idx.del_mask.resize((size_t)cap, 0);
        auto DM = del_mask_np.unchecked<1>();
        int64_t dm_len = del_mask_np.shape(0);
        for (int64_t i = 0; i < dm_len; ++i)
            idx.del_mask[i] = DM(i);

        // per-cluster counts
        auto CC = cluster_counts_np.unchecked<1>();
        idx.cluster_counts.resize(nlist);
        for (int c = 0; c < nlist; ++c)
            idx.cluster_counts[c] = CC(c);
        idx.cluster_tombstones.assign(nlist, 0);

        // Oja sketch: U_drift[c] is (d*F) float32, row-major (d rows, F cols)
        idx.U_drift.resize(nlist);
        auto UD = U_drift_np.unchecked<3>();
        for (int c = 0; c < nlist; ++c) {
            idx.U_drift[c].resize((size_t)d * F, 0.f);
            for (int j = 0; j < d; ++j)
                for (int l = 0; l < F; ++l)
                    idx.U_drift[c][j * F + l] = UD(c, j, l);
        }

        // cluster_global
        idx.cluster_global.resize(nlist);
        for (int c = 0; c < nlist; ++c) {
            py::object obj = cluster_global_list[c];
            if (obj.is_none()) continue;
            auto arr = obj.cast<py::array_t<int32_t>>();
            auto buf = arr.unchecked<1>();
            idx.cluster_global[c].reserve((size_t)buf.shape(0));
            for (int64_t i = 0; i < buf.shape(0); ++i)
                idx.cluster_global[c].push_back((uint32_t)buf(i));
        }

        // Build cones entirely in C++ (Phase 5: no Python cone objects passed in).
        idx.cluster_cones.resize(nlist);
        idx.cluster_has_cones.assign(nlist, false);
        idx.cluster_axes.resize(nlist);   // all empty = use global axes
        idx.point_cones.resize((size_t)cap);
        for (int c = 0; c < nlist; ++c) {
            idx.cluster_cones[c].assign(F, SortedCone(F));
            if (!idx.cluster_global[c].empty())
                idx._build_cones(c, idx.cluster_global[c]);
        }

        idx._build_sketch_all();
        return idx;
    }

    // ── from_stream ───────────────────────────────────────────────────────────
    //
    // Assemble an AMPIIndex from pre-built streaming components, bypassing
    // _build_cones (which would cause random access across the full mmap).
    //
    // The mmap data file at data_path/_cpp_data_buf.dat must already be written
    // (by the Python streaming pass) before this factory is called.
    //
    // cones_list : [nlist] of list-of-F SortedCone objects (built by dispatcher)

    static AMPIIndex from_stream(
        int d, int F, int nlist, int cone_top_k,
        double drift_theta, bool cosine,
        py::array_t<float,   py::array::c_style | py::array::forcecast> axes_np,
        py::array_t<float,   py::array::c_style | py::array::forcecast> centroids_np,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> cluster_counts_np,
        uint32_t n_total,
        const std::string& data_path,
        py::list  cluster_global_list,
        py::list  cones_list
    )
    {
        if (data_path.empty())
            throw std::invalid_argument("from_stream: data_path must not be empty");

        AMPIIndex idx;
        idx.d             = d;
        idx.F             = F;
        idx.nlist         = nlist;
        idx.cone_top_k    = cone_top_k;
        idx.drift_theta   = drift_theta;
        idx.cosine_metric = cosine;
        idx.n             = n_total;
        idx.capacity      = n_total;
        idx.n_deleted     = 0;

        // axes
        auto AX = axes_np.unchecked<2>();
        idx.axes.resize((size_t)F * d);
        for (int l = 0; l < F; ++l)
            for (int j = 0; j < d; ++j)
                idx.axes[l * d + j] = AX(l, j);

        // centroids
        auto CN = centroids_np.unchecked<2>();
        idx.centroids.resize((size_t)nlist * d);
        for (int c = 0; c < nlist; ++c)
            for (int j = 0; j < d; ++j)
                idx.centroids[c * d + j] = CN(c, j);

        // data buffer: open existing mmap file written by Python streaming pass.
        // _mmap_open uses ftruncate which is a no-op when size matches.
        std::string fpath = data_path + "/_cpp_data_buf.dat";
        size_t data_sz = (size_t)n_total * d * sizeof(float);
        idx._mmap_open(fpath, data_sz);

        // deleted mask: all zeros (fresh build, no deletions)
        idx.del_mask.resize((size_t)n_total, 0);

        // per-cluster counts
        auto CC = cluster_counts_np.unchecked<1>();
        idx.cluster_counts.resize(nlist);
        for (int c = 0; c < nlist; ++c)
            idx.cluster_counts[c] = CC(c);
        idx.cluster_tombstones.assign(nlist, 0);

        // Oja drift sketch: zero-initialised
        idx.U_drift.resize(nlist);
        for (int c = 0; c < nlist; ++c)
            idx.U_drift[c].assign((size_t)d * F, 0.f);

        // cluster_global
        idx.cluster_global.resize(nlist);
        for (int c = 0; c < nlist; ++c) {
            py::object obj = cluster_global_list[c];
            if (obj.is_none()) continue;
            auto arr = obj.cast<py::array_t<int32_t>>();
            auto buf = arr.unchecked<1>();
            idx.cluster_global[c].reserve((size_t)buf.shape(0));
            for (int64_t i = 0; i < buf.shape(0); ++i)
                idx.cluster_global[c].push_back((uint32_t)buf(i));
        }

        // Pre-built cones: copy from Python SortedCone objects.
        // Rebuild point_cones from axes[0] (same gid set as any other axis).
        idx.cluster_cones.resize(nlist);
        idx.cluster_has_cones.assign(nlist, false);
        idx.cluster_axes.resize(nlist);   // all empty = use global axes
        idx.point_cones.resize((size_t)n_total);

        for (int c = 0; c < nlist; ++c) {
            py::list cone_c = cones_list[c].cast<py::list>();
            idx.cluster_cones[c].resize(F, SortedCone(F));
            for (int f = 0; f < F; ++f) {
                const SortedCone& sc = cone_c[f].cast<const SortedCone&>();
                idx.cluster_cones[c][f] = sc;   // copy
                const auto& ax0 = idx.cluster_cones[c][f].axes;
                if (!ax0.empty() && !ax0[0].empty()) {
                    idx.cluster_has_cones[c] = true;
                    for (auto& p : ax0[0]) {
                        uint32_t gid = (uint32_t)p.second;
                        if (gid < n_total)
                            idx.point_cones[gid].push_back({(uint16_t)c, (uint16_t)f});
                    }
                }
            }
        }

        // Load pre-computed sketch from data_path/_sketch.dat if present;
        // otherwise compute from mmap via BLAS (sequential scan, OS-prefetchable).
        idx.sketch.resize((size_t)n_total * F, 0.f);
        {
            std::string sketch_path = data_path + "/_sketch.dat";
            std::ifstream sf(sketch_path, std::ios::binary);
            size_t expected = (size_t)n_total * F * sizeof(float);
            if (sf.good()) {
                sf.read(reinterpret_cast<char*>(idx.sketch.data()), (std::streamsize)expected);
                if (!sf || (size_t)sf.gcount() != expected)
                    idx._build_sketch_all();   // partial read → recompute
            } else {
                idx._build_sketch_all();
            }
        }

        return idx;
    }

    void set_merge_params(int interval, double eps, double qe_ratio) {
        if (interval < 0)
            throw std::invalid_argument("merge_interval must be >= 0");
        if (eps <= 0.0)
            throw std::invalid_argument("eps must be > 0");
        if (qe_ratio < 0.0 || qe_ratio > 1.0)
            throw std::invalid_argument("qe_ratio must be in [0, 1]");
        merge_interval = interval;
        eps_merge      = eps;
        merge_qe_ratio = qe_ratio;
    }

    // ── add ──────────────────────────────────────────────────────────────────

    uint32_t add(py::array_t<float, py::array::c_style | py::array::forcecast> x_np) {
        if (x_np.shape(0) != d)
            throw py::value_error("add: expected vector of length " + std::to_string(d) +
                                  ", got " + std::to_string(x_np.shape(0)));
        auto X = x_np.unchecked<1>();
        std::vector<float> x(d);
        if (cosine_metric) {
            float norm2 = 0.f;
            for (int j = 0; j < d; ++j) norm2 += X(j) * X(j);
            float inv = (norm2 > 1e-20f) ? 1.f / std::sqrt(norm2) : 1.f;
            for (int j = 0; j < d; ++j) x[j] = X(j) * inv;
        } else {
            for (int j = 0; j < d; ++j) x[j] = X(j);
        }
        std::unique_lock<std::shared_mutex> lk(*p_mutex);
        return _add_raw(x.data());
    }

    // ── batch_add ─────────────────────────────────────────────────────────────
    //
    // Insert m points at once under a single exclusive lock.
    // Returns (m,) int32 array of assigned global IDs.
    // Allocates numpy output and normalises rows while holding the GIL, then
    // releases the GIL for the locked insertion loop.

    py::array_t<int32_t> batch_add(
        py::array_t<float, py::array::c_style | py::array::forcecast> data_np)
    {
        auto D2 = data_np.unchecked<2>();  // GIL held: safe pointer extraction
        if (D2.shape(1) != d)
            throw py::value_error("batch_add: expected vectors of length " + std::to_string(d) +
                                  ", got " + std::to_string(D2.shape(1)));
        int m = (int)D2.shape(0);

        // Normalise all rows while GIL is held (pure arithmetic, no lock needed)
        std::vector<float> xs((size_t)m * d);
        for (int i = 0; i < m; ++i) {
            float* xi = &xs[i * d];
            if (cosine_metric) {
                float norm2 = 0.f;
                for (int j = 0; j < d; ++j) norm2 += D2(i, j) * D2(i, j);
                float inv = (norm2 > 1e-20f) ? 1.f / std::sqrt(norm2) : 1.f;
                for (int j = 0; j < d; ++j) xi[j] = D2(i, j) * inv;
            } else {
                for (int j = 0; j < d; ++j) xi[j] = D2(i, j);
            }
        }

        // Allocate output while GIL is held (numpy requires the GIL)
        auto out    = py::array_t<int32_t>(m);
        int32_t* op = out.mutable_data();

        {
            py::gil_scoped_release release;  // release GIL for the insertion loop
            std::unique_lock<std::shared_mutex> lk(*p_mutex);
            for (int i = 0; i < m; ++i)
                op[i] = (int32_t)_add_raw(&xs[i * d]);
        }
        return out;
    }

    // ── _add_raw (private, called under unique_lock) ──────────────────────────

    uint32_t _add_raw(const float* x) {
        uint32_t global_id = n;
        if (n >= capacity) _grow_buffers();

        std::copy(x, x + d, _data_ptr + (size_t)n * d);
        del_mask[n] = 0;
        _update_sketch_point(global_id, x);
        ++n;

        if (point_cones.size() <= global_id)
            point_cones.resize(global_id + 1);
        point_cones[global_id].clear();

        // Top-K cluster assignment
        int K_c = std::min(cone_top_k, nlist);
        std::vector<std::pair<float,int>> cdists(nlist);
        for (int c = 0; c < nlist; ++c) {
            float acc = 0.f;
            const float* cent = &centroids[c * d];
            for (int j = 0; j < d; ++j) {
                float diff = cent[j] - x[j]; acc += diff * diff;
            }
            cdists[c] = {acc, c};
        }
        std::nth_element(cdists.begin(), cdists.begin() + K_c, cdists.end());
        std::sort(cdists.begin(), cdists.begin() + K_c);

        for (int ki = 0; ki < K_c; ++ki) {
            int c = cdists[ki].second;
            const float* cent = &centroids[c * d];

            std::vector<float> centered(d);
            float cn2 = 0.f;
            for (int j = 0; j < d; ++j) {
                centered[j] = x[j] - cent[j];
                cn2 += centered[j] * centered[j];
            }
            float cn = (cn2 > 1e-20f) ? std::sqrt(cn2) : 0.f;

            // Project onto per-cluster axes (or global if not yet computed).
            const float* local_axes_c = cluster_axes[c].empty()
                                        ? axes.data() : cluster_axes[c].data();
            std::vector<float> proj(F, 0.f);
            for (int l = 0; l < F; ++l)
                for (int j = 0; j < d; ++j)
                    proj[l] += centered[j] * local_axes_c[l * d + j];

            // Top-K fan cones by |normed proj|
            int K_f = std::min(cone_top_k, F);
            std::vector<std::pair<float,int>> fsc(F);
            float inv_cn = (cn > 1e-10f) ? 1.f / cn : 1.f;
            for (int l = 0; l < F; ++l)
                fsc[l] = {-std::abs(proj[l]) * inv_cn, l};
            std::nth_element(fsc.begin(), fsc.begin() + K_f, fsc.end());

            std::vector<int> top_f;
            top_f.reserve(K_f);
            for (int k = 0; k < K_f; ++k) top_f.push_back(fsc[k].second);

            // Insert into selected cones
            if (cluster_has_cones[c]) {
                for (int f : top_f) {
                    cluster_cones[c][f].insert_raw(proj.data(), global_id);
                    point_cones[global_id].push_back({(uint16_t)c, (uint16_t)f});
                }
            } else {
                // Cluster had no cones; cones are already empty-initialised
                // in from_build — just insert and mark active once >= 2 points.
                for (int f : top_f) {
                    cluster_cones[c][f].insert_raw(proj.data(), global_id);
                    point_cones[global_id].push_back({(uint16_t)c, (uint16_t)f});
                }
                // After this push_back, cluster_global[c] will have size+1.
                if (cluster_global[c].size() >= 1)
                    cluster_has_cones[c] = true;
            }

            cluster_global[c].push_back(global_id);

            // Centroid EMA: μ ← (N·μ + x) / (N+1)
            int64_t N = cluster_counts[c];
            float inv_Np1 = 1.f / float(N + 1);
            float* cp = &centroids[c * d];
            for (int j = 0; j < d; ++j)
                cp[j] = (float(N) * cp[j] + x[j]) * inv_Np1;
            cluster_counts[c] = N + 1;

            // Drift EMA + power iteration
            if (cn > 1e-10f) {
                // Approx NN from cones
                std::unordered_set<uint32_t> nn_set;
                for (int f : top_f) {
                    auto ids = cluster_cones[c][f].query_raw(proj.data(), 8);
                    for (uint32_t gid2 : ids)
                        if (gid2 != global_id && !del_mask[gid2])
                            nn_set.insert(gid2);
                }
                std::vector<float> v(d);
                if (!nn_set.empty()) {
                    float best_d2 = std::numeric_limits<float>::max();
                    const float* best_ptr = nullptr;
                    for (uint32_t gid2 : nn_set) {
                        const float* p2 = _data_ptr + (size_t)gid2 * d;
                        float d2 = 0.f;
                        for (int j = 0; j < d; ++j) { float dj = x[j]-p2[j]; d2 += dj*dj; }
                        if (d2 < best_d2) { best_d2 = d2; best_ptr = p2; }
                    }
                    for (int j = 0; j < d; ++j) v[j] = x[j] - best_ptr[j];
                } else {
                    for (int j = 0; j < d; ++j) v[j] = centered[j];
                }
                if (_update_drift_and_check(c, v.data()))
                    _local_refresh(c);
            }
        }

        // Periodic cluster merge.
        if (merge_interval > 0) {
            ++insert_count;
            if (insert_count % (uint64_t)merge_interval == 0)
                _periodic_merge();
        }

        return global_id;
    }

    // ── remove ───────────────────────────────────────────────────────────────

    void remove(uint32_t global_id) {
        std::unique_lock<std::shared_mutex> lk(*p_mutex);
        _remove_raw(global_id);
    }

    // ── batch_delete ──────────────────────────────────────────────────────────
    //
    // Tombstone m points under a single exclusive lock.
    // Copies IDs into a plain vector while holding the GIL, then releases it.

    void batch_delete(
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> ids_np)
    {
        auto IDS = ids_np.unchecked<1>();  // GIL held: safe
        int m = (int)IDS.shape(0);
        std::vector<uint32_t> ids_vec(m);
        for (int i = 0; i < m; ++i) ids_vec[i] = (uint32_t)IDS(i);

        py::gil_scoped_release release;
        std::unique_lock<std::shared_mutex> lk(*p_mutex);
        for (int i = 0; i < m; ++i)
            _remove_raw(ids_vec[i]);
    }

    // ── _remove_raw (private, called under unique_lock) ───────────────────────

    void _remove_raw(uint32_t global_id) {
        if (global_id >= n || del_mask[global_id]) return;

        del_mask[global_id] = 1;
        ++n_deleted;

        std::unordered_set<int> seen;
        for (auto& cf : point_cones[global_id]) {
            int c = cf.first, f = cf.second;
            cluster_cones[c][f].remove(global_id);
            seen.insert(c);
        }
        for (int c : seen) {
            ++cluster_tombstones[c];
            if (cluster_counts[c] > 0) {
                double frac = double(cluster_tombstones[c]) / double(cluster_counts[c]);
                if (frac >= 0.10) _local_refresh(c);
            }
        }
    }

    // ── numpy views (call _refresh_views() in Python after any add/remove) ──

    py::array_t<float> get_data_view() {
        py::capsule capsule;
        if (_data_buf_sp) {
            // Heap mode: capsule holds a copy of the shared_ptr, keeping the
            // current allocation alive even after a grow replaces _data_buf_sp.
            auto* owner = new std::shared_ptr<std::vector<float>>(_data_buf_sp);
            capsule = py::capsule(owner, [](void* p) {
                delete static_cast<std::shared_ptr<std::vector<float>>*>(p);
            });
        } else {
            // mmap mode: region stays mapped for the lifetime of the index.
            capsule = py::capsule(_data_ptr, [](void*){});
        }
        return py::array_t<float>(
            {(py::ssize_t)n, (py::ssize_t)d},
            {(py::ssize_t)(d * sizeof(float)), (py::ssize_t)sizeof(float)},
            _data_ptr, capsule);
    }

    py::array_t<uint8_t> get_deleted_mask() {
        py::capsule dummy(del_mask.data(), [](void*){});
        return py::array_t<uint8_t>(
            {(py::ssize_t)n},
            {(py::ssize_t)sizeof(uint8_t)},
            del_mask.data(), dummy);
    }

    py::array_t<float> get_centroids() {
        py::capsule dummy(centroids.data(), [](void*){});
        return py::array_t<float>(
            {(py::ssize_t)nlist, (py::ssize_t)d},
            {(py::ssize_t)(d * sizeof(float)), (py::ssize_t)sizeof(float)},
            centroids.data(), dummy);
    }

    py::array_t<float> get_sketch() {
        py::capsule dummy(sketch.data(), [](void*){});
        return py::array_t<float>(
            {(py::ssize_t)n, (py::ssize_t)F},
            {(py::ssize_t)(F * sizeof(float)), (py::ssize_t)sizeof(float)},
            sketch.data(), dummy);
    }

    bool has_cones(int c) const {
        return c >= 0 && c < nlist && cluster_has_cones[c];
    }

    py::array_t<int32_t> get_cluster_global(int c) {
        if (c < 0 || c >= nlist)
            throw py::index_error("cluster index " + std::to_string(c) +
                                  " out of range [0, " + std::to_string(nlist) + ")");
        auto& cg = cluster_global[c];
        auto out = py::array_t<int32_t>((py::ssize_t)cg.size());
        auto buf = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < (py::ssize_t)cg.size(); ++i)
            buf(i) = (int32_t)cg[i];
        return out;
    }

    SortedCone& get_cone(int c, int f) {
        if (c < 0 || c >= nlist || f < 0 || f >= F)
            throw py::index_error("get_cone: (c=" + std::to_string(c) +
                                  ", f=" + std::to_string(f) + ") out of range");
        return cluster_cones[c][f];
    }

    py::array_t<float> get_axes() const {
        py::capsule dummy(const_cast<float*>(axes.data()), [](void*){});
        return py::array_t<float>(
            {(py::ssize_t)F, (py::ssize_t)d},
            {(py::ssize_t)(d * sizeof(float)), (py::ssize_t)sizeof(float)},
            axes.data(), dummy);
    }

    py::array_t<int64_t> get_cluster_counts() const {
        auto out = py::array_t<int64_t>((py::ssize_t)nlist);
        auto buf = out.mutable_unchecked<1>();
        for (int c = 0; c < nlist; ++c) buf(c) = cluster_counts[c];
        return out;
    }

    py::array_t<int64_t> get_cluster_tombstones() const {
        auto out = py::array_t<int64_t>((py::ssize_t)nlist);
        auto buf = out.mutable_unchecked<1>();
        for (int c = 0; c < nlist; ++c) buf(c) = cluster_tombstones[c];
        return out;
    }

    // Returns the (cluster, fan) pairs that contain global_id.
    py::list get_point_cones(uint32_t gid) const {
        py::list result;
        if (gid < (uint32_t)point_cones.size()) {
            for (auto& cf : point_cones[gid])
                result.append(py::make_tuple((int)cf.first, (int)cf.second));
        }
        return result;
    }

    // ── query ─────────────────────────────────────────────────────────────────
    //
    // Full adaptive window-expansion query entirely in C++.
    // Returns py::tuple(sq_dists float32[k], ids int32[k]) sorted nearest-first.
    // Python wrapper applies metric conversion (sqrt / *0.5 / identity).

    py::tuple query(
        py::array_t<float, py::array::c_style | py::array::forcecast> q_np,
        int k, int window_size, int probes, int fan_probes)
    {
        if (q_np.shape(0) != d)
            throw py::value_error("query: expected vector of length " + std::to_string(d) +
                                  ", got " + std::to_string(q_np.shape(0)));

        // Copy query to a local C++ buffer so the GIL can be released for
        // the entire computation section below.
        std::vector<float> q_buf(d);
        std::vector<float> q_sketch(F);
        {
            auto Q = q_np.unchecked<1>();
            for (int j = 0; j < d; ++j) q_buf[j] = Q(j);
            // Compute global sketch projection while GIL is still held.
            const float* ax = axes.data();
            for (int f = 0; f < F; ++f) {
                float acc = 0.f;
                const float* row = ax + (size_t)f * d;
                for (int j = 0; j < d; ++j) acc += q_buf[j] * row[j];
                q_sketch[f] = acc;
            }
        }
        const float* qptr = q_buf.data();

        // Results filled by the GIL-free section.
        std::vector<float>    result_dists;
        std::vector<int32_t>  result_ids;

        {
            py::gil_scoped_release rel;
            std::shared_lock<std::shared_mutex> lk(*p_mutex);

        // ── Step 1: top probes clusters ───────────────────────────────────────
        int K_c = std::min(probes, nlist);
        std::vector<std::pair<float,int>> cdists(nlist);
        for (int c = 0; c < nlist; ++c) {
            float acc = 0.f;
            const float* cent = &centroids[c * d];
            for (int j = 0; j < d; ++j) { float diff = cent[j]-qptr[j]; acc += diff*diff; }
            cdists[c] = {acc, c};
        }
        std::nth_element(cdists.begin(), cdists.begin() + K_c, cdists.end());
        std::sort(cdists.begin(), cdists.begin() + K_c);

        // ── Step 2: build cone contexts and fallback parts ────────────────────
        struct ConeCtx {
            const SortedCone* cone;
            std::vector<float> q_proj;  // (F,) projection for this cluster
        };
        std::vector<ConeCtx> cone_ctxs;
        std::vector<const std::vector<uint32_t>*> fallback_parts;

        for (int ki = 0; ki < K_c; ++ki) {
            int c = cdists[ki].second;
            const auto& cg = cluster_global[c];
            if (cg.empty()) continue;

            if (!cluster_has_cones[c] || fan_probes >= F) {
                fallback_parts.push_back(&cg);
                continue;
            }

            const float* cent = &centroids[c * d];
            std::vector<float> q_centered(d);
            float qnorm2 = 0.f;
            for (int j = 0; j < d; ++j) {
                q_centered[j] = qptr[j] - cent[j];
                qnorm2 += q_centered[j] * q_centered[j];
            }
            float inv_qnorm = (qnorm2 > 1e-20f) ? 1.f / std::sqrt(qnorm2) : 1.f;

            // Use per-cluster axes if available, else global.
            const float* local_axes_c = cluster_axes[c].empty()
                                        ? axes.data() : cluster_axes[c].data();
            std::vector<float> q_proj(F, 0.f);
            for (int l = 0; l < F; ++l) {
                const float* al = local_axes_c + l * d;
                float dot = 0.f;
                for (int j = 0; j < d; ++j) dot += q_centered[j] * al[j];
                q_proj[l] = dot;
            }

            int K_f = std::min(fan_probes, F);
            std::vector<std::pair<float,int>> fsc(F);
            for (int l = 0; l < F; ++l)
                fsc[l] = {-std::abs(q_proj[l]) * inv_qnorm, l};
            std::nth_element(fsc.begin(), fsc.begin() + K_f, fsc.end());

            for (int ki2 = 0; ki2 < K_f; ++ki2) {
                int f = fsc[ki2].second;
                if (cluster_cones[c][f].size() > 0)
                    cone_ctxs.push_back({&cluster_cones[c][f], q_proj});
            }
        }

        // ── Step 3: adaptive window expansion ─────────────────────────────────
        int w = std::max(k, 8);
        std::vector<uint8_t> mask(n, 0);  // reused across iterations
        std::vector<uint32_t> cands;

        while (true) {
            std::fill(mask.begin(), mask.end(), 0);

            // Fallback: all live points in cluster
            for (const auto* part : fallback_parts)
                for (uint32_t gid : *part)
                    if (gid < n && !del_mask[gid]) mask[gid] = 1;

            // Cone candidates
            for (const auto& ctx : cone_ctxs) {
                if (ctx.cone->size() <= 2 * w) {
                    if (!ctx.cone->axes.empty())
                        for (const auto& p : ctx.cone->axes[0])
                            if (!ctx.cone->tombstones.count(p.second) && p.second < n)
                                mask[p.second] = 1;
                } else {
                    auto ids = ctx.cone->query_raw(ctx.q_proj.data(), w);
                    for (uint32_t gid : ids) if (gid < n) mask[gid] = 1;
                }
            }

            cands.clear();
            for (uint32_t i = 0; i < n; ++i)
                if (mask[i] && !del_mask[i]) cands.push_back(i);

            if (w >= window_size || (int)cands.size() < k) break;

            // Compute kth squared distance for coverage check
            int m = (int)cands.size();
            std::vector<float> dists_tmp(m);
            for (int i = 0; i < m; ++i) {
                const float* p = _data_ptr + (size_t)cands[i] * d;
                float acc = 0.f;
                for (int j = 0; j < d; ++j) { float diff = p[j]-qptr[j]; acc += diff*diff; }
                dists_tmp[i] = acc;
            }
            std::nth_element(dists_tmp.begin(), dists_tmp.begin() + k - 1, dists_tmp.end());
            float kth_sq   = dists_tmp[k - 1];
            float kth_proj = std::sqrt(std::max(0.f, kth_sq));

            // Coverage check
            bool all_covered = true;
            for (const auto& ctx : cone_ctxs) {
                if (ctx.cone->size() <= 2 * w) continue;
                if (!ctx.cone->is_covered_raw(ctx.q_proj.data(), w, kth_proj)) {
                    all_covered = false;
                    break;
                }
            }
            if (all_covered) break;

            w = std::min(w * 2, window_size);
        }

        // Fallback: not enough candidates — return first k live points
        if ((int)cands.size() < k) {
            cands.clear();
            for (uint32_t i = 0; i < n && (int)cands.size() < k; ++i)
                if (!del_mask[i]) cands.push_back(i);
        }

            // Final rerank — sketch-pruned two-pass.
            // Pass 1: rank all m candidates by sketch lower-bound (in RAM).
            //         Compute exact dist for top-M2 only (mmap access).
            // Pass 2: for the remaining candidates, prune by lower bound;
            //         compute exact only for survivors.
            // Correctness: sketch_dist ≤ true_dist (Bessel), so no false negatives.
            int m = (int)cands.size();
            int actual_k = std::min(k, m);
            int M2 = (m > 3 * k) ? std::min(m, std::max(3 * k, 50)) : m;

            std::vector<float> sq_dists(m, std::numeric_limits<float>::infinity());

            if (M2 >= m || sketch.empty()) {
                // Small candidate set or no sketch — skip sketch overhead, exact rerank all.
                for (int i = 0; i < m; ++i) {
                    const float* p = _data_ptr + (size_t)cands[i] * d;
                    float acc = 0.f;
                    for (int j = 0; j < d; ++j) { float diff = p[j]-qptr[j]; acc += diff*diff; }
                    sq_dists[i] = acc;
                }
            } else {
                // Compute sketch distances for all m candidates (pure RAM).
                std::vector<float> sk_dists(m);
                const float* sk_table = sketch.data();
                const float* qsk      = q_sketch.data();
                for (int i = 0; i < m; ++i) {
                    const float* sk = sk_table + (size_t)cands[i] * F;
                    float acc = 0.f;
                    for (int f = 0; f < F; ++f) { float df = qsk[f]-sk[f]; acc += df*df; }
                    sk_dists[i] = acc;
                }

                // Partial-sort: indices of top-M2 by sketch distance.
                std::vector<int> order(m);
                std::iota(order.begin(), order.end(), 0);
                std::nth_element(order.begin(), order.begin() + M2, order.end(),
                    [&](int a, int b){ return sk_dists[a] < sk_dists[b]; });

                // Pass 1: exact distances for top-M2 (mmap access).
                for (int ii = 0; ii < M2; ++ii) {
                    int i = order[ii];
                    const float* p = _data_ptr + (size_t)cands[i] * d;
                    float acc = 0.f;
                    for (int j = 0; j < d; ++j) { float diff = p[j]-qptr[j]; acc += diff*diff; }
                    sq_dists[i] = acc;
                }

                // kth exact distance among top-M2 (upper bound on true kth-neighbor dist).
                std::vector<float> top_m2(M2);
                for (int ii = 0; ii < M2; ++ii) top_m2[ii] = sq_dists[order[ii]];
                std::nth_element(top_m2.begin(), top_m2.begin() + actual_k - 1, top_m2.end());
                float kth_sq = top_m2[actual_k - 1];

                // Pass 2: remaining candidates — prune by lower bound, exact for survivors.
                for (int ii = M2; ii < m; ++ii) {
                    int i = order[ii];
                    if (sk_dists[i] > kth_sq) continue;    // safe: lower bound > threshold
                    const float* p = _data_ptr + (size_t)cands[i] * d;
                    float acc = 0.f;
                    for (int j = 0; j < d; ++j) { float diff = p[j]-qptr[j]; acc += diff*diff; }
                    sq_dists[i] = acc;
                    if (acc < kth_sq) kth_sq = acc;        // tighten threshold
                }
            }

            std::vector<int> top_idx(m);
            std::iota(top_idx.begin(), top_idx.end(), 0);
            std::nth_element(top_idx.begin(), top_idx.begin() + actual_k, top_idx.end(),
                [&](int a, int b) { return sq_dists[a] < sq_dists[b]; });
            std::sort(top_idx.begin(), top_idx.begin() + actual_k,
                [&](int a, int b) { return sq_dists[a] < sq_dists[b]; });

            result_dists.resize(actual_k);
            result_ids.resize(actual_k);
            for (int i = 0; i < actual_k; ++i) {
                result_dists[i] = sq_dists[top_idx[i]];
                result_ids[i]   = (int32_t)cands[top_idx[i]];
            }
        }  // GIL reacquired here; lk released here

        // Allocate and fill output numpy arrays (requires GIL).
        int actual_k = (int)result_ids.size();
        auto out_sq  = py::array_t<float>(actual_k);
        auto out_ids = py::array_t<int32_t>(actual_k);
        auto od = out_sq.mutable_unchecked<1>();
        auto oi = out_ids.mutable_unchecked<1>();
        for (int i = 0; i < actual_k; ++i) {
            od(i) = result_dists[i];
            oi(i) = result_ids[i];
        }
        return py::make_tuple(out_sq, out_ids);
    }

    // ── query_candidates ──────────────────────────────────────────────────────
    //
    // Single-pass candidate collection (no adaptive loop, no rerank).
    // Returns sorted int32 array of unique live global IDs.

    py::array_t<int32_t> query_candidates(
        py::array_t<float, py::array::c_style | py::array::forcecast> q_np,
        int window_size, int probes, int fan_probes)
    {
        if (q_np.shape(0) != d)
            throw py::value_error("query_candidates: expected vector of length " + std::to_string(d) +
                                  ", got " + std::to_string(q_np.shape(0)));

        std::vector<float> q_buf(d);
        {
            auto Q = q_np.unchecked<1>();
            for (int j = 0; j < d; ++j) q_buf[j] = Q(j);
        }
        const float* qptr = q_buf.data();

        std::vector<uint32_t> cands_result;

        {
            py::gil_scoped_release rel;
            std::shared_lock<std::shared_mutex> lk(*p_mutex);

        int K_c = std::min(probes, nlist);
        std::vector<std::pair<float,int>> cdists(nlist);
        for (int c = 0; c < nlist; ++c) {
            float acc = 0.f;
            const float* cent = &centroids[c * d];
            for (int j = 0; j < d; ++j) { float diff = cent[j]-qptr[j]; acc += diff*diff; }
            cdists[c] = {acc, c};
        }
        std::nth_element(cdists.begin(), cdists.begin() + K_c, cdists.end());
        std::sort(cdists.begin(), cdists.begin() + K_c);

        std::vector<uint8_t> mask(n, 0);

        for (int ki = 0; ki < K_c; ++ki) {
            int c = cdists[ki].second;
            const auto& cg = cluster_global[c];
            if (cg.empty()) continue;

            if (!cluster_has_cones[c] || fan_probes >= F) {
                for (uint32_t gid : cg)
                    if (gid < n && !del_mask[gid]) mask[gid] = 1;
                continue;
            }

            const float* cent = &centroids[c * d];
            std::vector<float> q_centered(d);
            float qnorm2 = 0.f;
            for (int j = 0; j < d; ++j) {
                q_centered[j] = qptr[j] - cent[j];
                qnorm2 += q_centered[j] * q_centered[j];
            }
            float inv_qnorm = (qnorm2 > 1e-20f) ? 1.f / std::sqrt(qnorm2) : 1.f;

            const float* local_axes_c = cluster_axes[c].empty()
                                        ? axes.data() : cluster_axes[c].data();
            std::vector<float> q_proj(F, 0.f);
            for (int l = 0; l < F; ++l) {
                const float* al = local_axes_c + l * d;
                float dot = 0.f;
                for (int j = 0; j < d; ++j) dot += q_centered[j] * al[j];
                q_proj[l] = dot;
            }

            int K_f = std::min(fan_probes, F);
            std::vector<std::pair<float,int>> fsc(F);
            for (int l = 0; l < F; ++l)
                fsc[l] = {-std::abs(q_proj[l]) * inv_qnorm, l};
            std::nth_element(fsc.begin(), fsc.begin() + K_f, fsc.end());

            for (int ki2 = 0; ki2 < K_f; ++ki2) {
                int f = fsc[ki2].second;
                const SortedCone& cone = cluster_cones[c][f];
                int cone_sz = cone.size();
                if (cone_sz == 0) continue;
                if (cone_sz <= 2 * window_size) {
                    if (!cone.axes.empty())
                        for (const auto& p : cone.axes[0])
                            if (!cone.tombstones.count(p.second) && p.second < n)
                                mask[p.second] = 1;
                } else {
                    auto ids = cone.query_raw(q_proj.data(), window_size);
                    for (uint32_t gid : ids) if (gid < n) mask[gid] = 1;
                }
            }
        }

        for (uint32_t i = 0; i < n; ++i)
            if (mask[i] && !del_mask[i]) cands_result.push_back(i);

        }  // GIL reacquired here

        auto out = py::array_t<int32_t>((py::ssize_t)cands_result.size());
        auto buf = out.mutable_unchecked<1>();
        for (size_t i = 0; i < cands_result.size(); ++i)
            buf((py::ssize_t)i) = (int32_t)cands_result[i];
        return out;
    }

    // ── local_refresh (public, GIL-free) ──────────────────────────────────────

    void local_refresh(int c) {
        std::unique_lock<std::shared_mutex> lk(*p_mutex);
        if (c >= 0 && c < nlist)
            _local_refresh(c);
    }

    // ── periodic_merge (public) ───────────────────────────────────────────────
    //
    // Manually trigger a merge pass with the given eps_merge threshold.
    // Temporarily overrides the stored eps_merge for this call.

    void periodic_merge(double eps) {
        if (eps <= 0.0)
            throw std::invalid_argument("eps must be > 0");
        std::unique_lock<std::shared_mutex> lk(*p_mutex);
        const double saved = eps_merge;
        eps_merge = eps;
        _periodic_merge();
        eps_merge = saved;
    }

    // ── get_cluster_axes ─────────────────────────────────────────────────────
    //
    // Returns (F, d) float32 array of the current fan axes for cluster c.
    // Returns the global axes if per-cluster axes have not been computed yet.

    py::array_t<float> get_cluster_axes(int c) const {
        const float* ptr = (c >= 0 && c < nlist && !cluster_axes[c].empty())
                           ? cluster_axes[c].data() : axes.data();
        py::capsule dummy(const_cast<float*>(ptr), [](void*){});
        return py::array_t<float>(
            {(py::ssize_t)F, (py::ssize_t)d},
            {(py::ssize_t)(d * sizeof(float)), (py::ssize_t)sizeof(float)},
            ptr, dummy);
    }

    // Return the (d, F) float32 Oja subspace sketch for cluster c.
    // Used by the Phase-2 checkpoint serializer.
    py::array_t<float> get_U_drift(int c) const {
        if (c < 0 || c >= nlist)
            throw py::index_error("cluster index " + std::to_string(c) +
                                  " out of range [0, " + std::to_string(nlist) + ")");
        auto out = py::array_t<float>({(py::ssize_t)d, (py::ssize_t)F});
        auto buf = out.mutable_unchecked<2>();
        const float* U = U_drift[c].data();   // row-major: U[j*F + l]
        for (int j = 0; j < d; ++j)
            for (int l = 0; l < F; ++l)
                buf(j, l) = U[j * F + l];
        return out;
    }

    ~AMPIIndex() { _mmap_close(); }

private:
    // ── mmap helpers ─────────────────────────────────────────────────────────

    // Open (or create) a file at path, extend to sz bytes, and MAP_SHARED it.
    // Sets _mmap_fd, _mmap_addr, _mmap_size, _mmap_path, _data_ptr.
    void _mmap_open(const std::string& path, size_t sz) {
        _mmap_path = path;
        _mmap_fd   = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (_mmap_fd < 0) throw std::runtime_error("mmap open failed: " + path);
        if (::ftruncate(_mmap_fd, (off_t)sz) < 0) {
            ::close(_mmap_fd); _mmap_fd = -1;
            throw std::runtime_error("ftruncate failed: " + path);
        }
        _mmap_addr = ::mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, _mmap_fd, 0);
        if (_mmap_addr == MAP_FAILED) {
            ::close(_mmap_fd); _mmap_fd = -1;
            throw std::runtime_error("mmap failed: " + path);
        }
        _mmap_size = sz;
        _data_ptr  = static_cast<float*>(_mmap_addr);
    }

    // Grow the mmap file to new_sz bytes (macOS-compatible: unmap then remap).
    void _mmap_grow(size_t new_sz) {
        ::munmap(_mmap_addr, _mmap_size);
        _mmap_addr = MAP_FAILED;  // mark invalid before any failure point
        _data_ptr  = nullptr;
        if (::ftruncate(_mmap_fd, (off_t)new_sz) < 0) {
            ::close(_mmap_fd); _mmap_fd = -1;
            throw std::runtime_error("ftruncate grow failed");
        }
        _mmap_addr = ::mmap(nullptr, new_sz, PROT_READ | PROT_WRITE, MAP_SHARED, _mmap_fd, 0);
        if (_mmap_addr == MAP_FAILED) {
            ::close(_mmap_fd); _mmap_fd = -1;
            throw std::runtime_error("mmap grow failed");
        }
        _mmap_size = new_sz;
        _data_ptr  = static_cast<float*>(_mmap_addr);
    }

    void _mmap_close() {
        if (_mmap_addr != MAP_FAILED) { ::munmap(_mmap_addr, _mmap_size); _mmap_addr = MAP_FAILED; }
        if (_mmap_fd   >= 0)          { ::close(_mmap_fd);                _mmap_fd   = -1; }
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    void _grow_buffers() {
        uint32_t new_cap = capacity * 2;
        if (_mmap_addr != MAP_FAILED) {
            // mmap mode: extend the file and remap.
            _mmap_grow((size_t)new_cap * d * sizeof(float));
        } else {
            // Allocate fresh — do not resize in place, which would invalidate
            // raw pointers held by any numpy arrays returned by get_data_view().
            auto new_buf = std::make_shared<std::vector<float>>((size_t)new_cap * d, 0.f);
            std::copy(_data_ptr, _data_ptr + (size_t)n * d, new_buf->data());
            _data_buf_sp = std::move(new_buf);
            _data_ptr = _data_buf_sp->data();
        }
        del_mask.resize(new_cap, 0);
        point_cones.resize(new_cap);
        sketch.resize((size_t)new_cap * F, 0.f);
        capacity = new_cap;
    }

    // Compute sketch[gid*F .. +F] = x @ global_axes^T  (F dot products).
    // x is the raw (un-normalised, un-centered) stored vector.
    void _update_sketch_point(uint32_t gid, const float* x) {
        float* sk = sketch.data() + (size_t)gid * F;
        const float* ax = axes.data();
        for (int f = 0; f < F; ++f) {
            float acc = 0.f;
            const float* row = ax + (size_t)f * d;
            for (int j = 0; j < d; ++j) acc += x[j] * row[j];
            sk[f] = acc;
        }
    }

    // Build sketch for all live points via BLAS: sketch (n×F) = data (n×d) @ axes^T (F×d).
    void _build_sketch_all() {
        sketch.assign((size_t)capacity * F, 0.f);
        if (n == 0) return;
        // Gather live rows into a contiguous buffer, then GEMM.
        // For mmap-backed data the sequential read is OS-prefetchable.
        ampi::sgemm((int)n, F, d,
                    _data_ptr, d,
                    axes.data(), d,
                    sketch.data(), F,
                    /*transA=*/false, /*transB=*/true);
    }

    bool _update_drift_and_check(int c, const float* v) {
        float* U = U_drift[c].data();   // (d, F) row-major: U[j*F + l]
        constexpr float beta  = 0.01f;
        constexpr float omB   = 1.0f - beta;

        // proj[l] = Σ_j v[j] * U[j*F + l]
        std::vector<float> proj(F, 0.f);
        for (int j = 0; j < d; ++j) {
            float vj = v[j];
            for (int l = 0; l < F; ++l)
                proj[l] += vj * U[j * F + l];
        }

        // U[j*F+l] = omB * U[j*F+l] + beta * v[j] * proj[l]
        for (int j = 0; j < d; ++j) {
            float vj = v[j];
            for (int l = 0; l < F; ++l)
                U[j * F + l] = omB * U[j * F + l] + beta * vj * proj[l];
        }

        // Normalise each column
        for (int l = 0; l < F; ++l) {
            float norm2 = 0.f;
            for (int j = 0; j < d; ++j) norm2 += U[j * F + l] * U[j * F + l];
            if (norm2 > 1e-24f) {
                float inv = 1.f / std::sqrt(norm2);
                for (int j = 0; j < d; ++j) U[j * F + l] *= inv;
            }
        }

        // Drift check: compare U[:, 0] (leading eigenvec estimate) against fan axes.
        float norm0 = 0.f;
        for (int j = 0; j < d; ++j) norm0 += U[j * F] * U[j * F];
        if (norm0 < 1e-12f) return false;   // no signal yet

        const float* cmp_axes = cluster_axes[c].empty()
                                ? axes.data() : cluster_axes[c].data();
        float cos_max = 0.f;
        for (int l = 0; l < F; ++l) {
            const float* al = cmp_axes + (size_t)l * d;
            float dot = 0.f;
            for (int j = 0; j < d; ++j) dot += al[j] * U[j * F];
            float ab = std::abs(dot);
            if (ab > cos_max) cos_max = ab;
        }

        const float cos_theta = std::cos((float)drift_theta * (3.14159265f / 180.f));
        return cos_max < cos_theta;
    }

    void _build_cones(int c, const std::vector<uint32_t>& live_ids) {
        int n_c = (int)live_ids.size();
        cluster_cones[c].assign(F, SortedCone(F));
        cluster_has_cones[c] = false;
        if (n_c < 2) return;

        const float* cent = &centroids[c * d];

        // Step 1: gather centered vectors x_c (n_c × d) and compute norms.
        std::vector<float> x_c((size_t)n_c * d);
        std::vector<float> norms(n_c, 0.f);
        for (int i = 0; i < n_c; ++i) {
            const float* xi = _data_ptr + (size_t)live_ids[i] * d;
            float s2 = 0.f;
            for (int j = 0; j < d; ++j) {
                float cj = xi[j] - cent[j];
                x_c[i * d + j] = cj;
                s2 += cj * cj;
            }
            norms[i] = (s2 > 1e-20f) ? std::sqrt(s2) : 1.f;
        }

        // Step 2: projs (n_c × F) = x_c (n_c × d) @ local_axes^T (F × d).
        // ampi::sgemm(M, N, K, A, lda, B, ldb, C, ldc, transA=false, transB=true)
        const float* local_axes = (cluster_axes[c].empty())
                                  ? axes.data() : cluster_axes[c].data();
        std::vector<float> projs((size_t)n_c * F, 0.f);
        ampi::sgemm(n_c, F, d,
                    x_c.data(), d,
                    local_axes, d,
                    projs.data(), F,
                    /*transA=*/false, /*transB=*/true);

        // Top-K cone assignment per point
        int K_f = std::min(cone_top_k, F);
        std::vector<std::vector<int>> cone_pts(F);
        std::vector<std::pair<float,int>> tmp(F);
        for (int i = 0; i < n_c; ++i) {
            float inv_n = 1.f / norms[i];
            for (int l = 0; l < F; ++l)
                tmp[l] = {-std::abs(projs[i * F + l]) * inv_n, l};
            std::nth_element(tmp.begin(), tmp.begin() + K_f, tmp.end());
            for (int k = 0; k < K_f; ++k) cone_pts[tmp[k].second].push_back(i);
        }

        // Clear old (c, *) entries from point_cones for these IDs
        for (uint32_t gid : live_ids) {
            auto& pc = point_cones[gid];
            pc.erase(std::remove_if(pc.begin(), pc.end(),
                [c](const std::pair<uint16_t,uint16_t>& p) {
                    return p.first == (uint16_t)c;
                }), pc.end());
        }

        // Build sorted cone arrays
        for (int f = 0; f < F; ++f) {
            auto& pts = cone_pts[f];
            if (pts.empty()) continue;
            cluster_has_cones[c] = true;
            SortedCone& cone = cluster_cones[c][f];
            for (int l = 0; l < F; ++l) {
                auto& ax_l = cone.axes[l];
                ax_l.clear();
                ax_l.reserve(pts.size());
                for (int local_i : pts)
                    ax_l.push_back({projs[local_i * F + l], live_ids[local_i]});
                std::sort(ax_l.begin(), ax_l.end());
            }
            for (int local_i : pts)
                point_cones[live_ids[local_i]].push_back({(uint16_t)c, (uint16_t)f});
        }
    }

    void _local_refresh(int c) {
        auto& cg = cluster_global[c];
        std::vector<uint32_t> live;
        live.reserve(cg.size());
        for (uint32_t gid : cg)
            if (!del_mask[gid]) live.push_back(gid);

        // Derive per-cluster axes from accumulated U_drift BEFORE resetting it.
        _compute_cluster_axes(c);

        if (live.size() < 2) {
            cluster_has_cones[c] = false;
            cluster_cones[c].assign(F, SortedCone(F));
        } else {
            _build_cones(c, live);
        }
        cluster_global[c]       = std::move(live);
        cluster_counts[c]       = (int64_t)cluster_global[c].size();
        cluster_tombstones[c]   = 0;
        std::fill(U_drift[c].begin(), U_drift[c].end(), 0.f);
    }

    // ── per-cluster axes (from Oja sketch U_drift) ───────────────────────────
    //
    // Copies and normalises the F columns of U_drift[c] into cluster_axes[c]
    // as (F*d) floats.  Falls back to clearing cluster_axes[c]
    // (= use global axes) when the leading column has insufficient signal.

    void _compute_cluster_axes(int c) {
        const float* U = U_drift[c].data();   // (d, F) row-major

        // Check if the leading column has signal.
        float norm0 = 0.f;
        for (int j = 0; j < d; ++j) norm0 += U[j * F] * U[j * F];
        if (norm0 < 1e-12f) {
            cluster_axes[c].clear();   // signal: use global axes
            return;
        }

        // Copy and normalise columns → cluster_axes[c] as (F, d) float32.
        cluster_axes[c].resize((size_t)F * d, 0.f);
        for (int l = 0; l < F; ++l) {
            float norm2 = 0.f;
            for (int j = 0; j < d; ++j) norm2 += U[j * F + l] * U[j * F + l];
            float inv = (norm2 > 1e-24f) ? (1.f / std::sqrt(norm2)) : 0.f;
            for (int j = 0; j < d; ++j)
                cluster_axes[c][l * d + j] = U[j * F + l] * inv;
        }
    }


    // ── cluster merge helpers ─────────────────────────────────────────────────

    // Merge cluster `fold` into cluster `keep`.  Called under exclusive lock.
    void _merge_clusters(int keep, int fold) {
        if (keep == fold) return;
        if (cluster_counts[fold] == 0) return;

        int64_t N_k     = cluster_counts[keep];
        int64_t N_f     = cluster_counts[fold];
        int64_t N_total = N_k + N_f;

        // Merged centroid.
        float* mu_k      = &centroids[keep * d];
        const float* mu_f = &centroids[fold * d];
        float inv_N = 1.0f / (float)N_total;
        for (int j = 0; j < d; ++j)
            mu_k[j] = ((float)N_k * mu_k[j] + (float)N_f * mu_f[j]) * inv_N;
        // Redirect fold's centroid so new inserts near that region go to keep.
        for (int j = 0; j < d; ++j)
            centroids[fold * d + j] = mu_k[j];

        // Append fold's live points to keep's cluster_global.
        auto& cg_fold = cluster_global[fold];
        auto& cg_keep = cluster_global[keep];
        for (uint32_t gid : cg_fold)
            if (!del_mask[gid]) cg_keep.push_back(gid);

        // Clear fold cluster state.
        cg_fold.clear();
        cluster_counts[fold]     = 0;
        cluster_tombstones[fold] = 0;
        cluster_has_cones[fold]  = false;
        cluster_cones[fold].assign(F, SortedCone(F));
        std::fill(U_drift[fold].begin(), U_drift[fold].end(), 0.f);
        cluster_axes[fold].clear();

        // Compact keep's cluster_global (remove deleted entries).
        std::vector<uint32_t> live;
        live.reserve(cg_keep.size());
        for (uint32_t gid : cg_keep)
            if (!del_mask[gid]) live.push_back(gid);
        cg_keep               = std::move(live);
        cluster_counts[keep]  = (int64_t)cg_keep.size();
        cluster_tombstones[keep] = 0;
        std::fill(U_drift[keep].begin(), U_drift[keep].end(), 0.f);
        cluster_axes[keep].clear();

        // Rebuild cones for the merged cluster using global axes initially;
        // per-cluster axes will be derived on the next _local_refresh.
        _build_cones(keep, cg_keep);
    }

    // Scan all cluster pairs within eps_merge; merge qualifying pairs.
    // Called under exclusive lock from _add_raw (when insert_count % merge_interval == 0)
    // or from the public periodic_merge() method.
    void _periodic_merge() {
        if (eps_merge <= 0.0) return;
        const double eps2 = eps_merge * eps_merge;
        std::vector<bool> merged(nlist, false);

        for (int i = 0; i < nlist; ++i) {
            if (merged[i] || cluster_counts[i] == 0) continue;

            double best_d2 = eps2;
            int    best_j  = -1;

            for (int j = i + 1; j < nlist; ++j) {
                if (merged[j] || cluster_counts[j] == 0) continue;
                const float* mi = &centroids[i * d];
                const float* mj = &centroids[j * d];
                double d2 = 0.0;
                for (int k = 0; k < d; ++k) { double df = mi[k]-mj[k]; d2 += df*df; }
                if (d2 < best_d2) { best_d2 = d2; best_j = j; }
            }

            if (best_j < 0) continue;
            const int j = best_j;

            // Mean QE for each cluster.
            const int64_t N_i     = cluster_counts[i];
            const int64_t N_j     = cluster_counts[j];
            const int64_t N_total = N_i + N_j;
            const float* mu_i = &centroids[i * d];
            const float* mu_j = &centroids[j * d];

            double mQE_i = 0.0;
            for (uint32_t gid : cluster_global[i]) {
                if (del_mask[gid]) continue;
                const float* xi = _data_ptr + (size_t)gid * d;
                double d2 = 0.0;
                for (int k = 0; k < d; ++k) { double df = xi[k]-mu_i[k]; d2 += df*df; }
                mQE_i += d2;
            }
            if (N_i > 0) mQE_i /= (double)N_i;

            double mQE_j = 0.0;
            for (uint32_t gid : cluster_global[j]) {
                if (del_mask[gid]) continue;
                const float* xj = _data_ptr + (size_t)gid * d;
                double d2 = 0.0;
                for (int k = 0; k < d; ++k) { double df = xj[k]-mu_j[k]; d2 += df*df; }
                mQE_j += d2;
            }
            if (N_j > 0) mQE_j /= (double)N_j;

            // Merge if the QE increase from using μ_merged (instead of separate
            // centroids) is small relative to the combined within-cluster spread:
            //   δ_qe = N_i*N_j/N_total² * ||μ_i - μ_j||²  ≤  (mQE_i + mQE_j)/2
            double weight   = (double)(N_i * N_j) / ((double)N_total * N_total);
            double delta_qe = weight * best_d2;
            if (delta_qe <= (mQE_i + mQE_j) * merge_qe_ratio) {
                _merge_clusters(i, j);
                merged[j] = true;
            }
        }
    }
};

// ── update_drift_and_check ────────────────────────────────────────────────────
//
// Oja subspace sketch update + drift check (standalone, for Python path).
//
// U_drift      : (d, F)  float32, modified in-place (Oja eigenvector sketch)
// axes         : (F, d)  float32
// displacement : (d,)    float32
// beta         : EMA decay (e.g. 0.01)
// theta_deg    : angle threshold in degrees (e.g. 15.0)
//
// Performs Oja's rule: proj = Uᵀ·v, U ← (1-β)U + β·v·projᵀ, normalise cols.
// Returns true if U[:,0] is > theta_deg from all fan axes,
// meaning _local_refresh should be triggered.

bool update_drift_and_check(
    py::array_t<float,  py::array::c_style | py::array::forcecast> U_np,
    py::array_t<float,  py::array::c_style | py::array::forcecast> axes,
    py::array_t<float,  py::array::c_style | py::array::forcecast> displacement,
    double beta, double theta_deg)
{
    auto U  = U_np.mutable_unchecked<2>();   // (d, F)
    auto A  = axes.unchecked<2>();           // (F, d)
    auto V  = displacement.unchecked<1>();   // (d,)
    const int64_t d = U.shape(0);
    const int64_t F = U.shape(1);

    const float omB  = (float)(1.0 - beta);
    const float betaf = (float)beta;

    // proj[l] = Σ_j V[j] * U[j, l]
    std::vector<float> proj(F, 0.f);
    for (int64_t j = 0; j < d; ++j) {
        float vj = V(j);
        for (int64_t l = 0; l < F; ++l)
            proj[l] += vj * U(j, l);
    }

    // U[j, l] = omB * U[j, l] + beta * V[j] * proj[l]
    for (int64_t j = 0; j < d; ++j) {
        float vj = V(j);
        for (int64_t l = 0; l < F; ++l)
            U(j, l) = omB * U(j, l) + betaf * vj * proj[l];
    }

    // Normalise each column
    for (int64_t l = 0; l < F; ++l) {
        float norm2 = 0.f;
        for (int64_t j = 0; j < d; ++j) norm2 += U(j, l) * U(j, l);
        if (norm2 > 1e-24f) {
            float inv = 1.f / std::sqrt(norm2);
            for (int64_t j = 0; j < d; ++j) U(j, l) *= inv;
        }
    }

    // Drift check: use U[:, 0] as leading eigenvec estimate
    float norm0 = 0.f;
    for (int64_t j = 0; j < d; ++j) norm0 += U(j, 0) * U(j, 0);
    if (norm0 < 1e-12f) return false;

    float cos_max = 0.f;
    for (int64_t l = 0; l < F; ++l) {
        float dot = 0.f;
        for (int64_t j = 0; j < d; ++j)
            dot += A(l, j) * U(j, 0);
        float ab = std::abs(dot);
        if (ab > cos_max) cos_max = ab;
    }

    const float cos_theta = std::cos((float)(theta_deg * (3.14159265358979323846 / 180.0)));
    return cos_max < cos_theta;
}

// ── best_clusters ─────────────────────────────────────────────────────────────
//
// Returns (probes,) int32 — indices of the `probes` nearest centroids to q,
// sorted nearest-first.  Uses nth_element (O(nlist)) + sort of the top slice.

py::array_t<int32_t> best_clusters(
    py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
    py::array_t<float, py::array::c_style | py::array::forcecast> q,
    int probes)
{
    auto C  = centroids.unchecked<2>();
    auto Q  = q.unchecked<1>();
    const int64_t nlist = C.shape(0);
    const int64_t d     = C.shape(1);

    probes = static_cast<int>(std::min((int64_t)probes, nlist));

    std::vector<std::pair<float, int32_t>> dists(nlist);
    const float* qptr = &Q(0);
    for (int64_t i = 0; i < nlist; ++i) {
        float acc = 0.f;
        const float* row = &C(i, 0);
        for (int64_t j = 0; j < d; ++j) {
            float diff = row[j] - qptr[j];
            acc += diff * diff;
        }
        dists[i] = {acc, static_cast<int32_t>(i)};
    }

    std::nth_element(dists.begin(), dists.begin() + probes, dists.end(),
                     [](const std::pair<float,int32_t>& a,
                        const std::pair<float,int32_t>& b) {
                         return a.first < b.first;
                     });
    std::sort(dists.begin(), dists.begin() + probes,
              [](const std::pair<float,int32_t>& a,
                 const std::pair<float,int32_t>& b) {
                  return a.first < b.first;
              });

    auto out = py::array_t<int32_t>(probes);
    auto buf = out.mutable_unchecked<1>();
    for (int i = 0; i < probes; ++i)
        buf(i) = dists[i].second;
    return out;
}

// ── best_fan_cones ────────────────────────────────────────────────────────────
//
// Returns (fan_probes,) int32 — indices of cones with highest
// |axis_l · q_centered| / ||q_centered||, sorted highest-first.

py::array_t<int32_t> best_fan_cones(
    py::array_t<float, py::array::c_style | py::array::forcecast> axes,
    py::array_t<float, py::array::c_style | py::array::forcecast> q_centered,
    int fan_probes)
{
    auto A  = axes.unchecked<2>();
    auto QC = q_centered.unchecked<1>();
    const int64_t F = A.shape(0);
    const int64_t d = A.shape(1);

    fan_probes = static_cast<int>(std::min((int64_t)fan_probes, F));

    float qnorm2 = 0.f;
    const float* qptr = &QC(0);
    for (int64_t j = 0; j < d; ++j) qnorm2 += qptr[j] * qptr[j];
    const float qnorm = (qnorm2 > 1e-20f) ? std::sqrt(qnorm2) : 1.f;

    // Store negated score so nth_element brings the best (largest |proj|) first.
    std::vector<std::pair<float, int32_t>> scores(F);
    for (int64_t l = 0; l < F; ++l) {
        float dot = 0.f;
        const float* row = &A(l, 0);
        for (int64_t j = 0; j < d; ++j) dot += row[j] * qptr[j];
        scores[l] = {-std::abs(dot) / qnorm, static_cast<int32_t>(l)};
    }

    std::nth_element(scores.begin(), scores.begin() + fan_probes, scores.end(),
                     [](const std::pair<float,int32_t>& a,
                        const std::pair<float,int32_t>& b) {
                         return a.first < b.first;
                     });
    std::sort(scores.begin(), scores.begin() + fan_probes,
              [](const std::pair<float,int32_t>& a,
                 const std::pair<float,int32_t>& b) {
                  return a.first < b.first;
              });

    auto out = py::array_t<int32_t>(fan_probes);
    auto buf = out.mutable_unchecked<1>();
    for (int i = 0; i < fan_probes; ++i)
        buf(i) = scores[i].second;
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
    py::class_<AMPIIndex>(m, "AMPIIndex",
        "C++ index owning all mutable state: data, cones, drift covariance.\n\n"
        "Construct via AMPIIndex.from_build(...).  Call add()/remove() for "
        "streaming mutations.  Use get_data_view() / get_deleted_mask() / "
        "get_centroids() to get zero-copy numpy views (refresh after add/remove).")
        .def_static("from_build", &AMPIIndex::from_build,
             py::arg("d"), py::arg("F"), py::arg("nlist"), py::arg("cone_top_k"),
             py::arg("drift_theta"), py::arg("cosine"),
             py::arg("axes_np"), py::arg("centroids_np"),
             py::arg("cluster_counts_np"), py::arg("U_drift_np"),
             py::arg("data_np"), py::arg("del_mask_np"), py::arg("n_init"),
             py::arg("cluster_global_list"),
             py::arg("data_path") = std::string(""))
        .def_static("from_stream", &AMPIIndex::from_stream,
             py::arg("d"), py::arg("F"), py::arg("nlist"), py::arg("cone_top_k"),
             py::arg("drift_theta"), py::arg("cosine"),
             py::arg("axes_np"), py::arg("centroids_np"),
             py::arg("cluster_counts_np"), py::arg("n_total"),
             py::arg("data_path"),
             py::arg("cluster_global_list"),
             py::arg("cones_list"),
             "Build an AMPIIndex from pre-computed streaming components.\n\n"
             "Skips _build_cones entirely — no random mmap access.\n"
             "data_path/_cpp_data_buf.dat must already be written.")
        .def("add",    &AMPIIndex::add,    py::arg("x"),
             py::call_guard<py::gil_scoped_release>(),
             "Insert one (d,) float32 vector.  Returns int global_id.")
        .def("remove", &AMPIIndex::remove, py::arg("global_id"),
             py::call_guard<py::gil_scoped_release>(),
             "Logical-delete a point.  Triggers compaction if needed.")
        .def("batch_add", &AMPIIndex::batch_add, py::arg("data"),
             "Insert (m, d) float32 array under one exclusive lock.\n\n"
             "Returns (m,) int32 array of assigned global IDs.")
        .def("batch_delete", &AMPIIndex::batch_delete, py::arg("ids"),
             "Tombstone all ids in (m,) int32 array under one exclusive lock.")
        .def("get_data_view",      &AMPIIndex::get_data_view,
             "Zero-copy (n, d) float32 numpy view into the data buffer.")
        .def("get_deleted_mask",   &AMPIIndex::get_deleted_mask,
             "Zero-copy (n,) uint8 numpy view (1 = deleted).")
        .def("get_centroids",      &AMPIIndex::get_centroids,
             "Zero-copy (nlist, d) float32 numpy view of centroids.")
        .def("get_sketch",         &AMPIIndex::get_sketch,
             "Zero-copy (n, F) float32 view of global-axis sketch table.\n\n"
             "sketch[gid, f] = dot(x_gid, global_axis_f).\n"
             "||sketch(q)-sketch(x)||² ≤ ||q-x||² (Bessel lower bound).")
        .def("has_cones",          &AMPIIndex::has_cones, py::arg("c"),
             "True if cluster c has initialised SortedCones.")
        .def("get_cluster_global", &AMPIIndex::get_cluster_global, py::arg("c"),
             "Return (m,) int32 array of global IDs belonging to cluster c.")
        .def("get_cone",           &AMPIIndex::get_cone, py::arg("c"), py::arg("f"),
             py::return_value_policy::reference_internal,
             "Reference to SortedCone for cluster c, fan axis f.")
        .def("get_point_cones",   &AMPIIndex::get_point_cones, py::arg("gid"),
             "Returns list of (cluster, fan) tuples containing global_id.")
        .def("get_axes",               &AMPIIndex::get_axes,
             "Zero-copy (F, d) float32 numpy view of the fan axes.")
        .def("get_cluster_counts",     &AMPIIndex::get_cluster_counts,
             "Returns (nlist,) int64 array of live-point counts per cluster.")
        .def("get_cluster_tombstones", &AMPIIndex::get_cluster_tombstones,
             "Returns (nlist,) int64 array of tombstone counts per cluster.")
        .def_readonly("n",             &AMPIIndex::n,
             "Current total number of inserted points (live + deleted).")
        .def_readonly("n_deleted",     &AMPIIndex::n_deleted,
             "Number of logically deleted points.")
        .def_readonly("capacity",      &AMPIIndex::capacity,
             "Allocated data-buffer capacity.")
        .def_readonly("nlist",         &AMPIIndex::nlist,
             "Number of coarse clusters.")
        .def_readonly("F",             &AMPIIndex::F,
             "Number of fan axes (sort directions) per cluster.")
        .def_readonly("d",             &AMPIIndex::d,
             "Vector dimensionality.")
        .def_readonly("cone_top_k",    &AMPIIndex::cone_top_k,
             "Soft-assignment multiplicity (number of clusters/cones per point).")
        .def_readonly("cosine_metric", &AMPIIndex::cosine_metric,
             "True if the index uses cosine distance.")
        .def_readwrite("drift_theta",  &AMPIIndex::drift_theta,
             "Drift-angle threshold in degrees.  Writable — change at any time.")
        .def_readwrite("merge_interval", &AMPIIndex::merge_interval,
             "Inserts between periodic cluster-merge checks (0 = disabled).")
        .def_readwrite("eps_merge",       &AMPIIndex::eps_merge,
             "Centroid L2 distance threshold for the merge check.")
        .def_readwrite("merge_qe_ratio",  &AMPIIndex::merge_qe_ratio,
             "Merge if δ_qe ≤ ratio*(mQE_i+mQE_j).")
        .def("set_merge_params", &AMPIIndex::set_merge_params,
             py::arg("interval"), py::arg("eps"), py::arg("qe_ratio"),
             "Set merge_interval, eps_merge, and merge_qe_ratio in one call.")
        .def("query", &AMPIIndex::query,
             py::arg("q"), py::arg("k"), py::arg("window_size"),
             py::arg("probes"), py::arg("fan_probes"),
             "Adaptive sorted-projection query entirely in C++.\n\n"
             "Returns (sq_dists float32[k], ids int32[k]) sorted nearest-first.\n"
             "Python wrapper applies metric conversion (sqrt / *0.5 / identity).")
        .def("query_candidates", &AMPIIndex::query_candidates,
             py::arg("q"), py::arg("window_size"),
             py::arg("probes"), py::arg("fan_probes"),
             "Single-pass candidate collection without reranking.\n\n"
             "Returns sorted int32 array of unique live global IDs.")
        .def("local_refresh", &AMPIIndex::local_refresh,
             py::arg("c"),
             py::call_guard<py::gil_scoped_release>(),
             "Rebuild all cones for cluster c without holding the GIL.\n\n"
             "Resets tombstone count and drift covariance for cluster c.\n"
             "No-op if c is out of range.")
        .def("periodic_merge", &AMPIIndex::periodic_merge,
             py::arg("eps"),
             py::call_guard<py::gil_scoped_release>(),
             "Trigger one cluster-merge pass with the given eps_merge threshold.\n\n"
             "Merges centroid pairs within eps whose QE increase is small.\n"
             "Calls _refresh_views() on the Python side are not needed —\n"
             "use AMPIAffineFanIndex.periodic_merge() instead.")
        .def("get_U_drift", &AMPIIndex::get_U_drift, py::arg("c"),
             "Copy of the (d, F) float32 Oja subspace sketch for cluster c.\n\n"
             "Row-major layout (d rows, F columns). Used by the Phase-2 checkpoint serializer.")
        .def("get_cluster_axes", &AMPIIndex::get_cluster_axes,
             py::arg("c"),
             "Return (F, d) float32 axes for cluster c.\n\n"
             "Returns global axes if per-cluster axes not yet computed.");

    m.def("update_drift_and_check", &update_drift_and_check,
          "Oja subspace sketch update + drift check.\n\n"
          "  U_drift      : (d, F)  float32, modified in-place\n"
          "  axes         : (F, d)  float32\n"
          "  displacement : (d,)    float32\n"
          "  beta         : float   EMA decay\n"
          "  theta_deg    : float   refresh threshold in degrees\n"
          "Returns True if _local_refresh should be triggered.",
          py::arg("U_drift"), py::arg("axes"), py::arg("displacement"),
          py::arg("beta"), py::arg("theta_deg"));
    m.def("best_clusters", &best_clusters,
          "Indices of the `probes` nearest centroids to q, sorted nearest-first.\n\n"
          "  centroids : (nlist, d) float32\n"
          "  q         : (d,)       float32\n"
          "  probes    : int\n"
          "Returns (probes,) int32.",
          py::arg("centroids"), py::arg("q"), py::arg("probes"));
    m.def("best_fan_cones", &best_fan_cones,
          "Indices of the `fan_probes` cones with highest |normed projection|.\n\n"
          "  axes       : (F, d) float32\n"
          "  q_centered : (d,)   float32\n"
          "  fan_probes : int\n"
          "Returns (fan_probes,) int32, sorted highest-first.",
          py::arg("axes"), py::arg("q_centered"), py::arg("fan_probes"));

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
             "True if any axis l guarantees no unvisited point can enter the top-k.")
        .def("get_axis_pairs", &SortedCone::get_axis_pairs, py::arg("l"),
             "Returns (projs: float32[n_f], ids: uint32[n_f]) for axis l.\n\n"
             "Includes tombstoned entries; call compact() first for a clean snapshot.\n"
             "Used by the Phase-2 checkpoint serializer.");
}
