// Real-kernel A/B harness for Phase 5 of the batch-blocking rollout (see
// TODO_BATCH_BLOCKING.md): times sisldo_forward's block4 branch, which
// (like disldo_forward's old block4 code) had both a fresh-per-call
// buffer allocation AND a fully serial final reduction.
#include "../sili/lib/headers/delta_csr_memory.hpp"
#include "../sili/lib/headers/sisldo_ops.hpp"
#include "../sili/lib/headers/csr.hpp"
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
    const float input_density = 0.1f;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> idist(0.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    std::vector<float> weight_values(n_in * n_out), importance_values(n_in * n_out);
    for (auto& w : weight_values)
        w = wdist(rng);
    for (auto& im : importance_values)
        im = idist(rng);

    Weights weights;
    weights.connections.layout.rows = n_in;
    weights.connections.layout.cols = n_out;
    block4_load_dense_fp32<SIZE_TYPE, COL_TYPE>(weights, weight_values.data(),
                                                importance_values.data(), n_in, n_out);

    std::printf("# sisldo_forward BLOCK4 real-kernel bench: n_in=%zu n_out=%zu num_cpus=%d "
                "input_density=%.2f tiles=%zu\n",
                n_in, n_out, num_cpus, double(input_density), weights.block4.n_tiles());

    const std::size_t k = std::max<std::size_t>(1, std::size_t(input_density * float(n_in)));
    for (int batch : {1, 4, 8, 16, 32, 64, 256, 1024}) {
        std::vector<SIZE_TYPE> ptrs(std::size_t(batch) + 1, 0);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> vals;
        std::vector<SIZE_TYPE> col_pool(n_in);
        std::iota(col_pool.begin(), col_pool.end(), SIZE_TYPE(0));
        for (int b = 0; b < batch; ++b) {
            std::shuffle(col_pool.begin(), col_pool.end(), rng);
            std::vector<SIZE_TYPE> row_cols(col_pool.begin(), col_pool.begin() + std::ptrdiff_t(k));
            std::sort(row_cols.begin(), row_cols.end());
            for (SIZE_TYPE c : row_cols) {
                idx.push_back(c);
                vals.push_back(xdist(rng));
            }
            ptrs[std::size_t(b) + 1] = ptrs[std::size_t(b)] + SIZE_TYPE(row_cols.size());
        }
        auto x_sparse =
            make_csr_input<SIZE_TYPE, float>(SIZE_TYPE(batch), SIZE_TYPE(n_in), ptrs, idx, vals);

        std::vector<float> y(std::size_t(batch) * n_out);
        const int n_warmup = 5;
        for (int i = 0; i < n_warmup; ++i) {
            std::fill(y.begin(), y.end(), 0.f);
            sisldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x_sparse, weights,
                                                                         y.data(), num_cpus);
        }
        const int n_reps = std::max(10, 20000 / std::max(batch, 1));
        std::vector<double> call_ns(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            std::fill(y.begin(), y.end(), 0.f);
            auto t0 = std::chrono::steady_clock::now();
            sisldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x_sparse, weights,
                                                                         y.data(), num_cpus);
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
