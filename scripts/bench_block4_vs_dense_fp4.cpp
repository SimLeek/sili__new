// Fair block4-vs-dense-FP4-floor benchmark, batch=1 (sili's real-time
// target -- see TODO_DUAL_BLOCK4.md's Part C). Corrects two real
// methodology bugs from earlier versions and the result once fixed:
// see docs/research/bench_block4_vs_dense_fp4.rst:
// bench_block4_vs_dense_fp4.methodology_bugs.
//
// Usage: run once per density point, in a FRESH process each time --
// see docs/research/bench_block4_vs_dense_fp4.rst:
// bench_block4_vs_dense_fp4.fresh_process_per_density for why.
//   g++ -std=c++20 -O3 -ffast-math -march=native -fopenmp \
//     -I <repo_root> scripts/bench_block4_vs_dense_fp4.cpp -o bench_b4_vs_dense
//   ./bench_b4_vs_dense 0.10   # tile-fill fraction, 0.0-1.0
//   ./bench_b4_vs_dense 1.00
//
// NOT wired into ctest -- a timing report, not a pass/fail correctness
// check; matches this repo's existing scripts/bench_block4_layer.py
// convention (benchmark, not test).
#include "../sili/lib/headers/delta_csr_memory.hpp"
#include "../sili/lib/headers/linear_disldo.hpp"
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <functional>

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
    const double tile_fraction = argc > 1 ? std::atof(argv[1]) : 1.0;
    const int n_in = 512, n_out = 512, batch = 1, reps = 200;
    const float learning_rate = 0.001f;
    const bool damp_by_importance = true;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> val_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> data_dist(-1.0f, 1.0f);
    std::bernoulli_distribution keep(tile_fraction);

    std::vector<float> x(batch * n_in), dy(batch * n_out);
    for (auto& v : x)
        v = data_dist(rng);
    for (auto& v : dy)
        v = data_dist(rng);

    // block4 side: bulk-load tile_fraction of all tile positions directly
    // (matches the planned "load network into block4, then prune/grow"
    // workflow, not growth-driven promotion).
    Weights w_b4;
    std::vector<SIZE_TYPE> empty_ptrs(n_in + 1, 0);
    w_b4.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        empty_ptrs, {}, {}, {}, n_in, n_out, 4096, 64, 0.1f);
    w_b4.block4.init(n_in, n_out);
    w_b4.block4.switch_point =
        0; // dense-loaded network: no compression, matches the planned workflow's initial state
    w_b4.out_degree.assign(n_out, SIZE_TYPE(0));
    const int tiles_r = n_in / 4, tiles_c = n_out / 4;
    int n_tiles = 0;
    for (int br = 0; br < tiles_r; ++br)
        for (int bc = 0; bc < tiles_c; ++bc) {
            if (!keep(rng))
                continue;
            ++n_tiles;
            auto tile = w_b4.block4.get_or_create(uint32_t(br), uint32_t(bc));
            for (int li = 0; li < 4; ++li)
                for (int lj = 0; lj < 4; ++lj) {
                    tile.at(li, lj) = fp4_quantize(val_dist(rng)) | (fp4_quantize(0.5f) << 4);
                    ++w_b4.out_degree[bc * 4 + lj];
                }
        }
    w_b4.recompute_stats();

    fp4_seed_stochastic_rng(0);
    std::vector<float> out(n_out);
    const double b4_fwd_ms = best_of(reps,
                                     [&]() {
                                         disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
                                             x.data(), batch, n_in, w_b4, out.data(), 0.0f, 1);
                                     }) *
                             1e3;

    std::vector<float> dx(n_in), nia(n_in), nga(n_out);
    fp4_seed_stochastic_rng(0);
    const double b4_bwd_ms = best_of(reps,
                                     [&]() {
                                         disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
                                             x.data(), batch, n_in, dy.data(), w_b4, dx.data(),
                                             nia.data(), nga.data(), learning_rate, 1,
                                             /*lr_per_row_nnz=*/false, damp_by_importance);
                                     }) *
                             1e3;

    // Fair dense-FP4 floor: same per-element math as block4's
    // disldo_forward/disldo_backward, flat n_in x n_out byte array, no
    // tile/CSR indirection. Always processes the FULL matrix (a dense
    // loop has no sparsity to exploit) -- the fixed floor block4 is
    // compared against at every density.
    std::vector<uint8_t> Wd(std::size_t(n_in) * n_out);
    for (auto& v : Wd)
        v = fp4_quantize(val_dist(rng)) | (fp4_quantize(0.5f) << 4);
    std::vector<float> value_scale(n_in, 1.0f), importance_scale(n_in, 1.0f);
    std::vector<float> output_scale(n_out, 1.0f), output_importance_scale(n_out, 1.0f);

    std::vector<float> yd(n_out);
    const double dense_fwd_ms =
        best_of(reps,
                [&]() {
                    std::fill(yd.begin(), yd.end(), 0.0f);
                    for (int row = 0; row < n_in; ++row) {
                        const float xi = x[std::size_t(row)], vs = value_scale[std::size_t(row)];
                        const uint8_t* Wrow = &Wd[std::size_t(row) * n_out];
                        for (int col = 0; col < n_out; ++col) {
                            const float w =
                                FP4_TABLE[Wrow[col] & 0xFu] * vs * output_scale[std::size_t(col)];
                            yd[std::size_t(col)] += xi * w;
                        }
                    }
                }) *
        1e3;

    fp4_seed_stochastic_rng(0);
    std::vector<float> dxd(n_in);
    const double dense_bwd_ms =
        best_of(reps,
                [&]() {
                    std::fill(dxd.begin(), dxd.end(), 0.0f);
                    for (int row = 0; row < n_in; ++row) {
                        const float val_scale_r = value_scale[std::size_t(row)];
                        const float imp_scale_r = importance_scale[std::size_t(row)];
                        const float xi = x[std::size_t(row)];
                        uint8_t* Wrow = &Wd[std::size_t(row) * n_out];
                        float dx_row = 0.0f;
                        for (int col = 0; col < n_out; ++col) {
                            uint8_t byte = Wrow[col];
                            const float out_scale = output_scale[std::size_t(col)];
                            const float out_imp_scale = output_importance_scale[std::size_t(col)];
                            const float combined_scale = val_scale_r * out_scale;
                            const float combined_imp_scale = imp_scale_r * out_imp_scale;

                            float cw = FP4_TABLE[byte & 0xFu] * combined_scale; // -> true units
                            float ci = FP4_TABLE[(byte >> 4) & 0xFu] * combined_imp_scale;

                            const float dyv = dy[std::size_t(col)];
                            const float g = dyv * xi;
                            ci -= g * learning_rate;
                            const float delta = damp_by_importance
                                                    ? (-learning_rate * g) / (1.0f + std::abs(ci))
                                                    : (-learning_rate * g);
                            cw += delta;
                            dx_row += cw * dyv;

                            const uint8_t new_w = fp4_quantize_stochastic(cw / combined_scale);
                            const uint8_t new_imp =
                                fp4_quantize_stochastic(ci / combined_imp_scale);
                            Wrow[col] = uint8_t((new_imp << 4) | new_w);
                        }
                        dxd[std::size_t(row)] = dx_row;
                    }
                }) *
        1e3;

    std::printf("tile_fraction=%.2f n_tiles=%d/%d\n", tile_fraction, n_tiles, tiles_r * tiles_c);
    std::printf("%-22s %10s %10s %8s\n", "", "dense floor", "block4", "speedup");
    std::printf("%-22s %10.4f %10.4f %7.2fx\n", "forward (ms)", dense_fwd_ms, b4_fwd_ms,
                dense_fwd_ms / b4_fwd_ms);
    std::printf("%-22s %10.4f %10.4f %7.2fx\n", "backward (ms)", dense_bwd_ms, b4_bwd_ms,
                dense_bwd_ms / b4_bwd_ms);
    std::printf("(speedup = dense/block4, >1x means block4 is faster)\n");
    return 0;
}
