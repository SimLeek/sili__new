// Minimal driver for callgrind profiling of disldo_forward's NARROW
// (tree-reduction) path specifically -- see TODO_BATCH_BLOCKING.md
// Phase 2's null-result finding. Few reps (callgrind is ~50-100x
// slowdown) at a shape/thread-count forced into the narrow dispatch
// (cols_per_thread = ceil(n_out/4)/num_cpus < 12).
#include "../sili/lib/headers/linear_disldo.hpp"
#include <random>
#include <vector>

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

int main() {
    const std::size_t n_in = 128, n_out = 128;
    const SIZE_TYPE batch = 1024;
    const int num_cpus = 4; // cols_per_thread = (128/4)/4 = 8 < 12 -- narrow path
    std::mt19937 rng(1);
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

    std::vector<float> x(std::size_t(batch) * n_in);
    for (auto& v : x)
        v = xdist(rng);
    std::vector<float> y(std::size_t(batch) * n_out);

    // Warmup (not profiled -- callgrind instrumentation starts at
    // process start, so "warmup" here just primes allocator/OS pages;
    // real isolation of the profiled region would need
    // CALLGRIND_START_INSTRUMENTATION, skipped for simplicity since we
    // want total-call attribution anyway).
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
