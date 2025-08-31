
// ampi_index.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

// Include the BLAS header.
#include <cblas.h>

namespace py = pybind11;

// Helper: Convert a 2D numpy array (float32) to a std::vector<std::vector<float>>
std::vector<std::vector<float>> numpy_to_vector2d(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input should be a 2-D numpy array");
    int n = buf.shape[0];
    int d = buf.shape[1];
    float* ptr = static_cast<float*>(buf.ptr);
    std::vector<std::vector<float>> vec(n, std::vector<float>(d));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            vec[i][j] = ptr[i * d + j];
        }
    }
    return vec;
}

// Helper: Convert a 1D numpy array (float32) to std::vector<float>
std::vector<float> numpy_to_vector1d(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input should be a 1-D numpy array");
    int n = buf.shape[0];
    float* ptr = static_cast<float*>(buf.ptr);
    std::vector<float> vec(n);
    for (int i = 0; i < n; i++) {
        vec[i] = ptr[i];
    }
    return vec;
}

// Helper: Convert a std::vector<std::vector<float>> to a 2D numpy array (float32)
py::array_t<float> vector2d_to_numpy(const std::vector<std::vector<float>>& vec) {
    if (vec.empty())
        return py::array_t<float>();
    size_t k = vec.size();
    size_t d = vec[0].size();
    std::vector<ssize_t> shape = { static_cast<ssize_t>(k), static_cast<ssize_t>(d) };
    py::array_t<float> arr(shape);
    py::buffer_info buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < d; j++) {
            ptr[i * d + j] = vec[i][j];
        }
    }
    return arr;
}

// AMPIIndex: Adaptive Multi-Projection Index
class AMPIIndex {
public:
    std::vector<std::vector<float>> data;         // Data: (n x d)
    int n, d, L;
    std::vector<std::vector<float>> proj_dirs;      // Projection directions: (L x d)
    std::vector<std::vector<float>> sorted_projs;   // Sorted projection values per direction: (L x n)
    std::vector<std::vector<int>> sorted_idxs;      // Corresponding sorted indices: (L x n)
    std::vector<float> data_norms;                  // Precomputed squared L2 norms for each data point.

    // Constructor: build index from data and number of projections.
    AMPIIndex(const std::vector<std::vector<float>>& data_in, int num_projections) {
        data = data_in;
        n = data.size();
        d = (n > 0) ? data[0].size() : 0;
        L = num_projections;

        // Precompute squared norms for each data point using BLAS.
        data_norms.resize(n, 0.0f);
        for (int i = 0; i < n; i++) {
            // Compute ||data[i]||^2 = data[i] · data[i]
            data_norms[i] = cblas_sdot(d, data[i].data(), 1, data[i].data(), 1);
        }

        // Initialize random generator with seed 0.
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Generate L random projection directions and normalize each.
        proj_dirs.resize(L, std::vector<float>(d, 0.0f));
        for (int i = 0; i < L; i++) {
            float norm = 0.0f;
            for (int j = 0; j < d; j++) {
                proj_dirs[i][j] = dist(rng);
                norm += proj_dirs[i][j] * proj_dirs[i][j];
            }
            norm = std::sqrt(norm);
            for (int j = 0; j < d; j++) {
                proj_dirs[i][j] /= norm;
            }
        }

        // Precompute projections for each direction.
        sorted_projs.resize(L, std::vector<float>(n, 0.0f));
        sorted_idxs.resize(L, std::vector<int>(n, 0));
        for (int i = 0; i < L; i++) {
            std::vector<float> projs(n, 0.0f);
            std::vector<int> idxs(n);
            for (int k = 0; k < n; k++) {
                float dot = 0.0f;
                for (int j = 0; j < d; j++) {
                    dot += data[k][j] * proj_dirs[i][j];
                }
                projs[k] = dot;
                idxs[k] = k;
            }
            // Sort indices by corresponding projection values.
            std::vector<int> sorted_indices = idxs;
            std::sort(sorted_indices.begin(), sorted_indices.end(), [&projs](int a, int b) {
                return projs[a] < projs[b];
            });
            // Save sorted projections and indices.
            for (int k = 0; k < n; k++) {
                sorted_projs[i][k] = projs[sorted_indices[k]];
                sorted_idxs[i][k] = sorted_indices[k];
            }
        }
    }

    // Extract candidate indices from projections given a query vector q.
    std::vector<int> query_candidates(const std::vector<float>& q, int window_size = 10) {
        std::vector<std::vector<int>> candidate_matrix(L, std::vector<int>(2 * window_size, 0));

        // Loop over each projection direction (parallelized if OpenMP is enabled)
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int i = 0; i < L; i++) {
            float q_val = 0.0f;
            for (int j = 0; j < d; j++) {
                q_val += proj_dirs[i][j] * q[j];
            }
            // Binary search in sorted_projs[i]
            auto& proj = sorted_projs[i];
            int idx = std::lower_bound(proj.begin(), proj.end(), q_val) - proj.begin();
            int slice_size = 2 * window_size;
            int start = idx - window_size;
            if (start < 0)
                start = 0;
            if (start > n - slice_size)
                start = n - slice_size;
            for (int k = 0; k < slice_size; k++) {
                candidate_matrix[i][k] = sorted_idxs[i][start + k];
            }
        }
        std::vector<int> flat;
        flat.reserve(L * 2 * window_size);
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < 2 * window_size; j++) {
                flat.push_back(candidate_matrix[i][j]);
            }
        }
        std::sort(flat.begin(), flat.end());
        auto last = std::unique(flat.begin(), flat.end());
        flat.erase(last, flat.end());
        return flat;
    }

    // Compute squared Euclidean distances between query q and candidate points using BLAS.
    std::vector<float> compute_dists(const std::vector<float>& q, const std::vector<int>& candidate_indices) {
        std::vector<float> dists(candidate_indices.size(), 0.0f);
        // Precompute the squared norm of the query vector.
        float q_norm2 = cblas_sdot(d, q.data(), 1, q.data(), 1);
        for (size_t i = 0; i < candidate_indices.size(); i++) {
            int idx = candidate_indices[i];
            // Compute dot product between data[idx] and q using BLAS.
            float dot = cblas_sdot(d, data[idx].data(), 1, q.data(), 1);
            // Use the precomputed squared norm of data[idx] and the query.
            float dist2 = data_norms[idx] + q_norm2 - 2 * dot;
            dists[i] = dist2;
        }
        return dists;
    }

    // Query for k nearest neighbors.
    void query(const std::vector<float>& q, int window_size, int k,
               std::vector<std::vector<float>>& final_points,
               std::vector<float>& final_dists,
               std::vector<int>& final_indices) {
        std::vector<int> candidate_indices = query_candidates(q, window_size);
        std::vector<float> dists = compute_dists(q, candidate_indices);
        std::vector<int> order(candidate_indices.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&dists](int a, int b) {
            return dists[a] < dists[b];
        });
        final_points.clear();
        final_dists.clear();
        final_indices.clear();
        for (int i = 0; i < k && i < (int)order.size(); i++) {
            int idx = candidate_indices[order[i]];
            final_points.push_back(data[idx]);
            final_dists.push_back(dists[order[i]]);
            final_indices.push_back(idx);
        }
    }

    // ---- Python Wrapper Methods ----

    std::vector<int> py_query_candidates(py::array_t<float> q, int window_size = 10) {
        std::vector<float> q_vec = numpy_to_vector1d(q);
        return query_candidates(q_vec, window_size);
    }

    py::tuple py_query(py::array_t<float> q, int window_size, int k) {
        std::vector<float> q_vec = numpy_to_vector1d(q);
        std::vector<std::vector<float>> final_points;
        std::vector<float> final_dists;
        std::vector<int> final_indices;
        query(q_vec, window_size, k, final_points, final_dists, final_indices);
        py::array_t<float> np_points = vector2d_to_numpy(final_points);
        return py::make_tuple(np_points, final_dists, final_indices);
    }
};

PYBIND11_MODULE(ampi_index, m) {
    m.doc() = "AMPIIndex module compiled with C++ and pybind11 (with BLAS for vectorized operations)";
    m.def("create_ampi_index", [](py::array_t<float> data, int num_projections) {
        auto vec = numpy_to_vector2d(data);
        return new AMPIIndex(vec, num_projections);
    }, py::return_value_policy::take_ownership, py::arg("data"), py::arg("num_projections"));

    py::class_<AMPIIndex>(m, "AMPIIndex")
        .def("query_candidates", &AMPIIndex::py_query_candidates, py::arg("q"), py::arg("window_size") = 10)
        .def("query", &AMPIIndex::py_query, py::arg("q"), py::arg("window_size"), py::arg("k"));
}
