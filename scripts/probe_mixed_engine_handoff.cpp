// Validates the "quite important" secondary concern directly, with REAL
// code on both sides -- not two more synthetic proxies. Calls the actual
// sili disldo_forward kernel (Group A: sparse/block4-native, its own
// #pragma omp regions, built the same way setup.py/CMakeLists.txt build
// it: -fopenmp -> libgomp) immediately followed by a real MKL
// cblas_sgemv call (Group B: dense-BLAS, MKL_THREADING_LAYER=GNU ->
// also libgomp), to measure the REAL Group-A-to-Group-B layer-boundary
// handoff cost, rather than trusting the synthetic single-dispatch
// probe's ~1.0x number to generalize.
#include "../sili/lib/headers/linear_disldo.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

enum CBLAS_ORDER { CblasRowMajor = 101 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
extern "C" void cblas_sgemv(int, int, int, int, float, const float*, int, const float*, int, float,
                            float*, int);

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main() {
    const int n_in = 1024, n_out = 1024, num_cpus = 8;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Small-ish dense-ish block4 layer (Group A) -- just needs to
    // exercise disldo_forward's real #pragma omp regions with
    // realistic-sized work, not be the exact shape of anything else in
    // this investigation.
    std::vector<SIZE_TYPE> ptrs(std::size_t(n_in) + 1, 0);
    std::vector<SIZE_TYPE> idx;
    std::vector<float> wv, imp;
    for (int r = 0; r < n_in; ++r) {
        for (int c = 0; c < n_out; ++c) {
            idx.push_back(c);
            wv.push_back(dist(rng));
            imp.push_back(0.5f);
        }
        ptrs[std::size_t(r) + 1] = ptrs[std::size_t(r)] + n_out;
    }
    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        ptrs, idx, wv, imp, std::size_t(n_in), std::size_t(n_out), std::size_t(1) << 22,
        std::size_t(1) << 22, 0.2f);
    weights.recompute_stats();

    const int batch = 64; // above BLOCK4_BATCH_BLOCK_THRESHOLD, real omp path
    std::vector<float> x(std::size_t(batch) * n_in), y(std::size_t(batch) * n_out);
    for (auto& v : x)
        v = dist(rng);

    // MKL side (Group B), n=1024 batch=1 GEMV, same shape used throughout
    // this whole investigation.
    std::vector<float> w2(std::size_t(n_in) * n_out), x2(n_in), y2(n_out);
    for (auto& v : w2)
        v = dist(rng);
    for (auto& v : x2)
        v = dist(rng);
    auto mkl_fwd = [&]() {
        cblas_sgemv(CblasRowMajor, CblasTrans, n_in, n_out, 1.0f, w2.data(), n_out, x2.data(), 1,
                    0.0f, y2.data(), 1);
    };

    // warm-up both
    for (int i = 0; i < 5; ++i) {
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x.data(), batch, n_in, weights,
                                                                     y.data(), num_cpus);
        mkl_fwd();
    }

    std::printf("MKL sgemv right after a REAL disldo_forward call, 15 iters (ms):\n");
    for (int r = 0; r < 15; ++r) {
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x.data(), batch, n_in, weights,
                                                                     y.data(), num_cpus);
        double t0 = now_ms();
        mkl_fwd();
        double dt = now_ms() - t0;
        std::printf("%.4f ", dt);
    }
    std::printf("\n");

    std::printf("MKL sgemv isolated (no disldo_forward in between), 15 iters (ms):\n");
    for (int r = 0; r < 15; ++r) {
        double t0 = now_ms();
        mkl_fwd();
        double dt = now_ms() - t0;
        std::printf("%.4f ", dt);
    }
    std::printf("\n");
    return 0;
}
