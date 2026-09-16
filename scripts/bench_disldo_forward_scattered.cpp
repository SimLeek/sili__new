// Real-kernel A/B harness for Phase 4 of the batch-blocking rollout (see
// TODO_BATCH_BLOCKING.md): times disldo_forward's SCATTERED (non-block4,
// dc.empty()==false) path specifically -- a random 10%-density CSR
// layer with no block4 content at all, so every call exercises only the
// `if (!dc.empty())` branch this phase touched.
#include "../sili/lib/headers/linear_disldo.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

int main(int argc, char** argv) {
    const int num_cpus = argc > 1 ? std::atoi(argv[1]) : 4;
    const std::size_t n_in = 288, n_out = 288;
    const float density = 0.1f;
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

    std::printf("# disldo_forward SCATTERED real-kernel bench: n_in=%zu n_out=%zu num_cpus=%d "
                "density=%.2f nnz=%zu block4_tiles=%zu\n",
                n_in, n_out, num_cpus, double(density), idx.size(), weights.block4.n_tiles());

    for (int batch : {1, 4, 8, 16, 32, 64, 256, 1024}) {
        std::vector<float> x(std::size_t(batch) * n_in);
        for (auto& v : x)
            v = xdist(rng);

        std::vector<float> y(std::size_t(batch) * n_out);
        const int n_warmup = 5;
        for (int i = 0; i < n_warmup; ++i) {
            std::fill(y.begin(), y.end(), 0.f);
            disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                x.data(), SIZE_TYPE(batch), SIZE_TYPE(n_in), weights, y.data(), num_cpus);
        }
        const int n_reps = std::max(10, 20000 / std::max(batch, 1));
        std::vector<double> call_ns(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            std::fill(y.begin(), y.end(), 0.f);
            auto t0 = std::chrono::steady_clock::now();
            disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                x.data(), SIZE_TYPE(batch), SIZE_TYPE(n_in), weights, y.data(), num_cpus);
            auto t1 = std::chrono::steady_clock::now();
            call_ns[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        std::sort(call_ns.begin(), call_ns.end());
        const double median_ns = call_ns[std::size_t(n_reps) / 2];
        const double mean_ns = std::accumulate(call_ns.begin(), call_ns.end(), 0.0) / n_reps;
        std::printf("batch=%5d  reps=%5d  mean=%12.1f ns/call  median=%12.1f ns/call\n", batch,
                    n_reps, mean_ns, median_ns);
    }
    return 0;
}
