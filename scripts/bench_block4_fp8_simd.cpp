// FP8 block4 backward: SIMD accumulate path vs its own scalar fallback,
// same benchmarking convention as scripts/bench_block4_vs_dense_fp4.cpp.
// Measured results + why FP8's decode/encode stays scalar while
// accumulation goes SIMD: see docs/research/bench_block4_fp8_simd.rst.
//
// Build (SIMD, default):
//   g++ -std=c++20 -O3 -ffast-math -march=native -fopenmp \
//     -I <repo_root> scripts/bench_block4_fp8_simd.cpp -o bench_fp8_simd
// Build (scalar fallback forced):
//   g++ -std=c++20 -O3 -ffast-math -march=native -fopenmp \
//     -DSILI_BLOCK4_FORCE_SCALAR_BACKWARD=1 \
//     -I <repo_root> scripts/bench_block4_fp8_simd.cpp -o bench_fp8_scalar
//   ./bench_fp8_simd 1     # batch size
//   ./bench_fp8_simd 32
//
// NOT wired into ctest -- a timing report, not a pass/fail correctness
// check; matches this repo's existing bench_block4_vs_dense_fp4.cpp
// convention.
#include "../sili/lib/headers/delta_csr_memory.hpp"
#include "../sili/lib/headers/linear_disldo.hpp"
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <algorithm>

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP8BiValues, COL_TYPE>;

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
    const int n_in = 512, n_out = 512, reps = 200;
    const int batch = argc > 1 ? std::atoi(argv[1]) : 1;
    const float learning_rate = 0.001f;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> val_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> data_dist(-1.0f, 1.0f);

    std::vector<float> x(std::size_t(batch) * n_in), dy(std::size_t(batch) * n_out);
    for (auto& v : x)
        v = data_dist(rng);
    for (auto& v : dy)
        v = data_dist(rng);

    Weights w;
    std::vector<SIZE_TYPE> empty_ptrs(n_in + 1, 0);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP8BiValues, COL_TYPE>(
        empty_ptrs, {}, {}, {}, n_in, n_out, 4096, 64, 0.1f);
    w.block4.init(n_in, n_out);
    w.block4.switch_point =
        0; // fully dense-loaded, no compression -- matches the "load dense, then prune" workflow
    w.out_degree.assign(n_out, SIZE_TYPE(0));

    const int tiles_r = n_in / 4, tiles_c = n_out / 4;
    for (int br = 0; br < tiles_r; ++br) {
        for (int bc = 0; bc < tiles_c; ++bc) {
            auto tile = w.block4.get_or_create(uint32_t(br), uint32_t(bc));
            for (uint32_t li = 0; li < 4; ++li)
                for (uint32_t lj = 0; lj < 4; ++lj) {
                    tile.at_weight(li, lj) = fp8_quantize(val_dist(rng));
                    tile.at_importance(li, lj) = fp8_quantize(std::abs(val_dist(rng)));
                }
        }
    }
    for (int c = 0; c < n_out; ++c)
        w.out_degree[c] = tiles_r * 4;

    std::vector<float> dx(std::size_t(batch) * n_in), nia(n_in, 0.f), nga(n_out, 0.f);

    const double t = best_of(reps, [&]() {
        std::fill(dx.begin(), dx.end(), 0.f);
        disldo_backward<SIZE_TYPE, FP8BiValues, COL_TYPE>(x.data(), batch, n_in, dy.data(), w,
                                                          dx.data(), nia.data(), nga.data(),
                                                          learning_rate, 1, false, true);
    });

#if SILI_BLOCK4_FORCE_SCALAR_BACKWARD
    std::printf("scalar: batch=%d  %.6f s/call (best of %d), n_tiles=%zu\n", batch, t, reps,
                w.block4.n_tiles());
#else
    std::printf("simd:   batch=%d  %.6f s/call (best of %d), n_tiles=%zu\n", batch, t, reps,
                w.block4.n_tiles());
#endif
    return 0;
}
