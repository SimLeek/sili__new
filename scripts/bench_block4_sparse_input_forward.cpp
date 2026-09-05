// Does sisldo_forward's block4 sparse-input phase (forward_sparse's
// underlying op; Design A: work_offsets pre-pass, reused from the
// scattered path) beat disldo_forward (dense input, forward_dense) at a
// given density? Fresh process per density point -- see
// bench_block4_vs_dense_fp4.cpp's own established methodology note on why
// an in-process density loop gives unreliable numbers.
//
// Measured result (see TODO_DUAL_BLOCK4.md): sparse wins at every tested
// density from 90% down to 5% (1.73x-8.92x), once a redundant-rescan bug
// (find() instead of at_index()) was fixed -- before that fix, dense won
// from 30-90% density.
//
// Usage: g++ -std=c++20 -O3 -ffast-math -march=native -fopenmp -I<repo>
//   bench_block4_sparse_input_forward.cpp -o bench_b4_sparse_fwd
//   ./bench_b4_sparse_fwd <input_density 0.0-1.0> <num_cpus>
#include "sili/lib/headers/delta_csr_memory.hpp"
#include "sili/lib/headers/sisldo_ops.hpp"
#include "sili/lib/headers/linear_disldo.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static double best_of(int reps, const std::function<void()>& fn) {
    fn();
    double best = 1e18;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        best = std::min(
            best,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
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

    // Fully block4-resident layer -- isolates block4's own sparse-input
    // path from the scattered path entirely (this investigation is
    // specifically about whether block4's NEW gather is worth its cost).
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

    std::vector<float> x_dense(n_in, 0.0f);
    std::vector<SIZE_TYPE> x_idx;
    std::vector<float> x_vals;
    for (int i = 0; i < n_in; ++i) {
        if (!keep(rng))
            continue;
        const float v = data_dist(rng);
        x_dense[i] = v;
        x_idx.push_back(i);
        x_vals.push_back(v);
    }
    const double real_density = double(x_idx.size()) / n_in;

    CSRInput<SIZE_TYPE, float> x_sparse;
    x_sparse.rows = 1;
    x_sparse.cols = n_in;
    x_sparse.ptrs[0] = std::make_shared<std::vector<SIZE_TYPE>>(
        std::vector<SIZE_TYPE>{0, static_cast<SIZE_TYPE>(x_idx.size())});
    x_sparse.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(x_idx);
    x_sparse.values[0] = std::make_shared<std::vector<float>>(x_vals);

    std::vector<float> y_dense(n_out);
    const double dense_ms =
        best_of(reps,
                [&]() {
                    std::fill(y_dense.begin(), y_dense.end(), 0.0f);
                    disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
                        x_dense.data(), 1, n_in, weights, y_dense.data(), 0.0f, num_cpus);
                }) *
        1e3;

    std::vector<float> y_sparse(n_out);
    const double sparse_ms = best_of(reps,
                                     [&]() {
                                         std::fill(y_sparse.begin(), y_sparse.end(), 0.0f);
                                         sisldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
                                             x_sparse, weights, y_sparse.data(), 0.0f, num_cpus);
                                     }) *
                             1e3;

    std::printf("density=%.3f (real=%.3f) num_cpus=%d n_tiles=%zu\n", density, real_density,
                num_cpus, weights.block4.n_tiles());
    std::printf("  dense (disldo_forward):        %.4f ms\n", dense_ms);
    std::printf("  sparse (sisldo_forward b4):  %.4f ms\n", sparse_ms);
    std::printf("  speedup (dense/sparse): %.2fx %s\n", dense_ms / sparse_ms,
                dense_ms / sparse_ms > 1.0 ? "(sparse wins)" : "(dense wins)");
    return 0;
}
