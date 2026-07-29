// bench_block_kernel.cpp
// ===========================================================================
// Feasibility check ONLY (per direct instruction: verify, don't assume) --
// does a fixed-size (BS x BS) DENSE block matvec kernel actually vectorize
// well, before building any block-sparse network infrastructure around it?
//
// Design being tested: partition the weight matrix into BSxBS tiles; store
// each ACTIVE tile fully dense (all BS*BS slots, most possibly zero-weight)
// so there is no within-tile indexing/gather at all -- the tile's own
// memory layout is contiguous and regular, unlike disldo's CSR-per-row scan
// or Fable's hash/permutation-addressed gather. Skip fully-empty tiles for
// the GLOBAL sparsity (this is the classic Block-Sparse-Row / block-ELLPACK
// idea). This file tests ONLY the innermost per-tile compute kernel in
// isolation, both auto-vectorized (checked via -fopt-info-vec) and via
// hand-written AVX2 intrinsics, to see whether either actually achieves
// real SIMD throughput on a single tile before scaling up to a full network.
// ===========================================================================
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <immintrin.h>

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- Auto-vectorization candidate: column-major dense BSxBS matvec ----
// y[0..BS) += sum_j W[:,j] * x[j] -- W stored column-major (BS floats per
// column, contiguous), so for FIXED j, W[:,j] is a contiguous BS-wide read
// and the update to y[0..BS) is a single vectorizable AXPY per column. No
// gather anywhere: every read/write is a fixed, compile-time-known offset
// pattern (BS known at compile time via template).
template <int BS>
void block_matvec_auto(const float* W_colmajor, const float* x, float* y) {
    for (int j = 0; j < BS; ++j) {
        const float xj = x[j];
        #pragma omp simd
        for (int i = 0; i < BS; ++i) y[i] += W_colmajor[j * BS + i] * xj;
    }
}

// ---- Hand-written AVX2 intrinsics, BS=8 (exactly one __m256 register) ----
void block_matvec_avx2_8x8(const float* W_colmajor, const float* x, float* y) {
    __m256 acc = _mm256_loadu_ps(y);
    for (int j = 0; j < 8; ++j) {
        __m256 wcol = _mm256_loadu_ps(W_colmajor + j * 8);
        __m256 xj   = _mm256_set1_ps(x[j]);
        acc = _mm256_fmadd_ps(wcol, xj, acc);
    }
    _mm256_storeu_ps(y, acc);
}

// ---- Hand-written AVX2 intrinsics, BS=16 (two __m256 registers/column) ----
void block_matvec_avx2_16x16(const float* W_colmajor, const float* x, float* y) {
    __m256 acc0 = _mm256_loadu_ps(y), acc1 = _mm256_loadu_ps(y + 8);
    for (int j = 0; j < 16; ++j) {
        __m256 w0 = _mm256_loadu_ps(W_colmajor + j * 16);
        __m256 w1 = _mm256_loadu_ps(W_colmajor + j * 16 + 8);
        __m256 xj = _mm256_set1_ps(x[j]);
        acc0 = _mm256_fmadd_ps(w0, xj, acc0);
        acc1 = _mm256_fmadd_ps(w1, xj, acc1);
    }
    _mm256_storeu_ps(y, acc0);
    _mm256_storeu_ps(y + 8, acc1);
}

template <int BS, class F>
double bench_ntiles(F&& fn, int n_tiles, int reps = 200) {
    fn();  // warm
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_s();
        fn();
        best = std::min(best, now_s() - t0);
    }
    return best / n_tiles;  // seconds per tile
}

int main() {
    const int N_TILES = 100000;  // enough tiles to make timing loop overhead negligible
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> wv(-1, 1);

    // ---- BS=8 ----
    {
        std::vector<float> W(size_t(N_TILES) * 64), x(size_t(N_TILES) * 8), y(size_t(N_TILES) * 8, 0.f);
        for (auto& v : W) v = wv(rng);
        for (auto& v : x) v = wv(rng);

        double t_auto = bench_ntiles<8>([&] {
            for (int t = 0; t < N_TILES; ++t)
                block_matvec_auto<8>(&W[size_t(t) * 64], &x[size_t(t) * 8], &y[size_t(t) * 8]);
        }, N_TILES);
        std::fill(y.begin(), y.end(), 0.f);
        double t_avx2 = bench_ntiles<8>([&] {
            for (int t = 0; t < N_TILES; ++t)
                block_matvec_avx2_8x8(&W[size_t(t) * 64], &x[size_t(t) * 8], &y[size_t(t) * 8]);
        }, N_TILES);
        // scalar reference for both speed floor and correctness
        std::fill(y.begin(), y.end(), 0.f);
        double t_scalar = bench_ntiles<8>([&] {
            for (int t = 0; t < N_TILES; ++t) {
                float* yt = &y[size_t(t) * 8];
                const float* Wt = &W[size_t(t) * 64];
                const float* xt = &x[size_t(t) * 8];
                for (int j = 0; j < 8; ++j)
                    for (int i = 0; i < 8; ++i) yt[i] += Wt[j * 8 + i] * xt[j];
            }
        }, N_TILES);

        printf("=== BS=8 (64 slots/tile) ===\n");
        printf("scalar          : %8.2f ns/tile  %8.1f Mtile/s  %8.1f Mslot/s\n",
               t_scalar * 1e9, 1.0 / t_scalar * 1e-6, 64.0 / t_scalar * 1e-6);
        printf("auto (omp simd) : %8.2f ns/tile  %8.1f Mtile/s  %8.1f Mslot/s  (%.2fx vs scalar)\n",
               t_auto * 1e9, 1.0 / t_auto * 1e-6, 64.0 / t_auto * 1e-6, t_scalar / t_auto);
        printf("hand AVX2       : %8.2f ns/tile  %8.1f Mtile/s  %8.1f Mslot/s  (%.2fx vs scalar)\n\n",
               t_avx2 * 1e9, 1.0 / t_avx2 * 1e-6, 64.0 / t_avx2 * 1e-6, t_scalar / t_avx2);
    }

    // ---- BS=16 ----
    {
        std::vector<float> W(size_t(N_TILES) * 256), x(size_t(N_TILES) * 16), y(size_t(N_TILES) * 16, 0.f);
        for (auto& v : W) v = wv(rng);
        for (auto& v : x) v = wv(rng);

        double t_auto = bench_ntiles<16>([&] {
            for (int t = 0; t < N_TILES; ++t)
                block_matvec_auto<16>(&W[size_t(t) * 256], &x[size_t(t) * 16], &y[size_t(t) * 16]);
        }, N_TILES);
        std::fill(y.begin(), y.end(), 0.f);
        double t_avx2 = bench_ntiles<16>([&] {
            for (int t = 0; t < N_TILES; ++t)
                block_matvec_avx2_16x16(&W[size_t(t) * 256], &x[size_t(t) * 16], &y[size_t(t) * 16]);
        }, N_TILES);
        std::fill(y.begin(), y.end(), 0.f);
        double t_scalar = bench_ntiles<16>([&] {
            for (int t = 0; t < N_TILES; ++t) {
                float* yt = &y[size_t(t) * 16];
                const float* Wt = &W[size_t(t) * 256];
                const float* xt = &x[size_t(t) * 16];
                for (int j = 0; j < 16; ++j)
                    for (int i = 0; i < 16; ++i) yt[i] += Wt[j * 16 + i] * xt[j];
            }
        }, N_TILES);

        printf("=== BS=16 (256 slots/tile) ===\n");
        printf("scalar          : %8.2f ns/tile  %8.1f Mtile/s  %8.1f Mslot/s\n",
               t_scalar * 1e9, 1.0 / t_scalar * 1e-6, 256.0 / t_scalar * 1e-6);
        printf("auto (omp simd) : %8.2f ns/tile  %8.1f Mtile/s  %8.1f Mslot/s  (%.2fx vs scalar)\n",
               t_auto * 1e9, 1.0 / t_auto * 1e-6, 256.0 / t_auto * 1e-6, t_scalar / t_auto);
        printf("hand AVX2       : %8.2f ns/tile  %8.1f Mtile/s  %8.1f Mslot/s  (%.2fx vs scalar)\n",
               t_avx2 * 1e9, 1.0 / t_avx2 * 1e-6, 256.0 / t_avx2 * 1e-6, t_scalar / t_avx2);
    }

    return 0;
}
