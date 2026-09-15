// Real-kernel A/B harness for Phase 1 of the batch-blocked accumulation
// rollout (TODO_BATCH_BLOCKING.md): times the ACTUAL disldo_forward wide
// (column-partitioned) path across a batch sweep, not the isolated PoC.
// Build this same file against two different checkouts of
// linear_disldo_forward.hpp/block4.hpp (pre- and post-Phase 1) to get a
// real before/after comparison -- see the shell driver that compiles it
// twice.
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
    const int num_cpus = argc > 1 ? std::atoi(argv[1]) : 8;
    const std::size_t n_in = 288, n_out = 288; // matches the PoC's dims
    std::mt19937 rng(42);
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

    std::printf("# disldo_forward real-kernel bench: n_in=%zu n_out=%zu num_cpus=%d tiles=%zu\n",
                n_in, n_out, num_cpus, weights.block4.n_tiles());

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
