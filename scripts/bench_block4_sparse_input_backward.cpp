// Does delta_csr_backward_sparse_grad's block4 gradient-gather phase
// (backward_sparse's underlying op) beat disldo_backward (dense gradient,
// backward_dense) at a given density? Fresh process per density point,
// matching bench_block4_sparse_input_forward.cpp's methodology.
//
// Measured result (see TODO_DUAL_BLOCK4.md): sparse wins at every tested
// density from 90% down to 5% (1.63x-12.82x). Timed at learning_rate=0
// (read-only) specifically -- isolates the gather/decode cost from
// stochastic-quantize write-back cost, which the two implementations
// don't spend identically (see TODO_DUAL_BLOCK4.md's row-workspace
// writeup for why a real learning step isn't directly comparable this
// way between the two paths).
//
// Usage: g++ -std=c++20 -O3 -ffast-math -march=native -fopenmp -I<repo>
//   bench_block4_sparse_input_backward.cpp -o bench_b4_sparse_bwd
//   ./bench_b4_sparse_bwd <grad_density 0.0-1.0> <num_cpus>
#include "sili/lib/headers/delta_csr_memory.hpp"
#include "sili/lib/headers/delta_csr_ops.hpp"
#include "sili/lib/headers/linear_disldo.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>

using SIZE_TYPE = int;
using COL_TYPE  = uint32_t;
using Weights   = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static double best_of(int reps, const std::function<void()>& fn) {
    fn();
    double best = 1e18;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        best = std::min(best, std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
    }
    return best;
}

int main(int argc, char** argv) {
    const double density = argc > 1 ? std::atof(argv[1]) : 0.5;
    const int num_cpus = argc > 2 ? std::atoi(argv[2]) : 4;
    const int n_in = 512, n_out = 512, reps = 200;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> val_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> data_dist(-1.0f, 1.0f);
    std::bernoulli_distribution keep(density);

    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        std::vector<SIZE_TYPE>(n_in + 1, 0), {}, {}, {}, n_in, n_out, 4096, 64, 0.1f);
    weights.block4.init(n_in, n_out);
    weights.block4.switch_point = 0;
    weights.out_degree.assign(n_out, SIZE_TYPE(0));
    const int tiles_r = n_in / 4, tiles_c = n_out / 4;
    for (int br = 0; br < tiles_r; ++br)
        for (int bc = 0; bc < tiles_c; ++bc) {
            auto tile = weights.block4.get_or_create(uint32_t(br), uint32_t(bc));
            for (int li = 0; li < 4; ++li)
                for (int lj = 0; lj < 4; ++lj) {
                    tile.at(li, lj) = fp4_quantize(val_dist(rng)) | (fp4_quantize(0.5f) << 4);
                    ++weights.out_degree[bc * 4 + lj];
                }
        }
    weights.recompute_stats();

    std::vector<float> input(n_in);
    for (auto& v : input) v = data_dist(rng);

    std::vector<float> dy_dense(n_out, 0.0f);
    std::vector<SIZE_TYPE> g_idx;
    std::vector<float> g_vals;
    for (int c = 0; c < n_out; ++c) {
        if (!keep(rng)) continue;
        const float v = data_dist(rng);
        dy_dense[c] = v;
        g_idx.push_back(c);
        g_vals.push_back(v);
    }
    const double real_density = double(g_idx.size()) / n_out;

    CSRInput<SIZE_TYPE, float> dy_sparse;
    dy_sparse.rows = 1; dy_sparse.cols = n_out;
    dy_sparse.ptrs[0] = std::make_shared<std::vector<SIZE_TYPE>>(
        std::vector<SIZE_TYPE>{0, static_cast<SIZE_TYPE>(g_idx.size())});
    dy_sparse.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(g_idx);
    dy_sparse.values[0]  = std::make_shared<std::vector<float>>(g_vals);

    // lr=0: read-only comparison, isolates the gather/decode cost itself
    // from stochastic-quantize write-back cost (RNG calls aren't free and
    // differ in count/order between the two implementations -- see
    // conversation on why lr>0 correctness can't be checked bit-exact).
    std::vector<float> dx_dense(n_in), ni_a(n_in, 0.0f), ng_a(n_out, 0.0f);
    const double dense_ms = best_of(reps, [&]() {
        std::fill(dx_dense.begin(), dx_dense.end(), 0.0f);
        disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            input.data(), 1, n_in, dy_dense.data(), weights,
            dx_dense.data(), ni_a.data(), ng_a.data(), 0.0f, num_cpus, false, true);
    }) * 1e3;

    std::vector<float> dx_sparse(n_in), ni_a2(n_in, 0.0f), ng_a2(n_out, 0.0f);
    const double sparse_ms = best_of(reps, [&]() {
        std::fill(dx_sparse.begin(), dx_sparse.end(), 0.0f);
        delta_csr_backward_sparse_grad<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            input.data(), 1, weights, dy_sparse,
            dx_sparse.data(), ni_a2.data(), ng_a2.data(), 0.0f, num_cpus, false);
    }) * 1e3;

    std::printf("density=%.3f (real=%.3f) num_cpus=%d n_tiles=%zu\n",
                density, real_density, num_cpus, weights.block4.n_tiles());
    std::printf("  dense (disldo_backward):            %.4f ms\n", dense_ms);
    std::printf("  sparse (delta_csr_backward_sparse_grad b4): %.4f ms\n", sparse_ms);
    std::printf("  speedup (dense/sparse): %.2fx %s\n", dense_ms / sparse_ms,
                dense_ms / sparse_ms > 1.0 ? "(sparse wins)" : "(dense wins)");
    return 0;
}
