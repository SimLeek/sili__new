// Profiling driver for Phase 7.5's investigation (see
// TODO_BATCH_BLOCKING.md): where does disldo_forward's SCATTERED path
// actually spend its time -- the per-batch-sample accumulate loop
// (which a transpose fix would target), or the per-synapse CSR
// traversal/decode (which it wouldn't)? Few reps (callgrind slowdown).
#include "../sili/lib/headers/linear_disldo.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

int main() {
    const std::size_t n_in = 288, n_out = 288;
    const SIZE_TYPE batch = 1024;
    const float density = 0.1f;
    const int num_cpus = 1; // single-threaded: clean attribution, no GOMP artifact
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> idist(0.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
    std::vector<SIZE_TYPE> idx;
    std::vector<float> w, imp;
    const std::size_t k = std::max<std::size_t>(1, std::size_t(density * float(n_out)));
    std::vector<SIZE_TYPE> col_pool(n_out);
    std::iota(col_pool.begin(), col_pool.end(), SIZE_TYPE(0));
    for (std::size_t r = 0; r < n_in; ++r) {
        std::shuffle(col_pool.begin(), col_pool.end(), rng);
        std::vector<SIZE_TYPE> row_cols(col_pool.begin(), col_pool.begin() + std::ptrdiff_t(k));
        std::sort(row_cols.begin(), row_cols.end());
        for (SIZE_TYPE c : row_cols) {
            idx.push_back(c);
            w.push_back(wdist(rng));
            imp.push_back(idist(rng));
        }
        ptrs[r + 1] = ptrs[r] + SIZE_TYPE(row_cols.size());
    }

    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        ptrs, idx, w, imp, n_in, n_out, 1u << 20, 1u << 20, 0.2f);
    weights.recompute_stats();

    std::vector<float> x(std::size_t(batch) * n_in);
    for (auto& v : x)
        v = xdist(rng);
    std::vector<float> y(std::size_t(batch) * n_out);

    for (int i = 0; i < 2; ++i) {
        std::fill(y.begin(), y.end(), 0.f);
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
            x.data(), batch, SIZE_TYPE(n_in), weights, y.data(), num_cpus);
    }
    for (int i = 0; i < 5; ++i) {
        std::fill(y.begin(), y.end(), 0.f);
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
            x.data(), batch, SIZE_TYPE(n_in), weights, y.data(), num_cpus);
    }
    return 0;
}
