// DIDLDO proof-of-concept bench (TODO_BATCH_BLOCKING.md, "Also queued" ->
// full writeup and measurements there, not repeated here). Plain forward/
// backward GEMM sequence via numpy's bundled OpenBLAS
// (libscipy_openblas), plus an in-place RMSprop update fused into
// backward, timed against torch_ms already recorded in the Block4 Bench
// dataset for the same shapes/batches/thread count.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <random>
#include <vector>

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };

extern "C" void scipy_cblas_sgemm(const int Order, const int TransA, const int TransB, const int M,
                                  const int N, const int K, const float alpha, const float* A,
                                  const int lda, const float* B, const int ldb, const float beta,
                                  float* C, const int ldc);
extern "C" void scipy_cblas_sgemv(const int Order, const int TransA, const int M, const int N,
                                  const float alpha, const float* A, const int lda, const float* X,
                                  const int incX, const float beta, float* Y, const int incY);
extern "C" void scipy_openblas_set_num_threads(int);

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// batch==1 dispatches to sgemv not sgemm -- see probe_didldo_sgemv.cpp
// and TODO_BATCH_BLOCKING.md for the A/B (sgemm pays fixed blocked/
// packed-GEMM setup cost an M=1 call doesn't need).
static void didldo_forward(const float* x, const float* w, float* y, int batch, int n_in,
                           int n_out) {
    if (batch == 1) {
        // y[n_out] = w[n_in,n_out]^T @ x[n_in]
        scipy_cblas_sgemv(CblasRowMajor, CblasTrans, n_in, n_out, 1.0f, w, n_out, x, 1, 0.0f, y, 1);
        return;
    }
    scipy_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, n_out, n_in, 1.0f, x, n_in,
                      w, n_out, 0.0f, y, n_out);
}

// Single backward call, disldo_backward's own shape: lr=0 is grad-only,
// lr!=0 is grad+update, no separate optimizer call.
static void didldo_backward(const float* x, const float* dy, float* w, float* square_avg, float* dx,
                            float* dw_scratch, int batch, int n_in, int n_out, float lr,
                            int num_cpus) {
    // dx: same M=1 GEMV dispatch as didldo_forward.
    if (batch == 1) {
        scipy_cblas_sgemv(CblasRowMajor, CblasNoTrans, n_in, n_out, 1.0f, w, n_out, dy, 1, 0.0f, dx,
                          1);
    } else {
        scipy_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, n_in, n_out, 1.0f, dy,
                          n_out, w, n_out, 0.0f, dx, n_in);
    }
    // dw: K=1 (sger shape) at batch=1 not special-cased -- future micro-opt.
    scipy_cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_in, n_out, batch, 1.0f, x, n_in,
                      dy, n_out, 0.0f, dw_scratch, n_out);
    if (lr == 0.0f)
        return;
    constexpr float alpha = 0.99f;
    constexpr float eps = 1e-8f;
    const std::size_t n = std::size_t(n_in) * n_out;
    // num_threads() clause, not a global omp_set_num_threads() call.
#pragma omp parallel for num_threads(num_cpus)
    for (std::size_t i = 0; i < n; ++i) {
        square_avg[i] = alpha * square_avg[i] + (1.0f - alpha) * dw_scratch[i] * dw_scratch[i];
        w[i] -= lr * dw_scratch[i] / (std::sqrt(square_avg[i]) + eps);
    }
}

int main(int argc, char** argv) {
    int n_in = argc > 1 ? std::atoi(argv[1]) : 288;
    int n_out = argc > 2 ? std::atoi(argv[2]) : 288;
    int num_cpus = argc > 3 ? std::atoi(argv[3]) : 8;
    int reps = argc > 4 ? std::atoi(argv[4]) : 30;
    scipy_openblas_set_num_threads(num_cpus);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<int> batches = {1, 32, 64, 128, 256, 512, 1024};
    std::vector<float> w(std::size_t(n_in) * n_out);
    for (auto& v : w)
        v = dist(rng);
    // Persists across warm-up and every timed rep, matching torch's RMSprop.
    std::vector<float> square_avg(w.size(), 0.0f);

    // bwd0/bwdX match bench_sili_vs_torch_matrix.py's op set (fwd+grad,
    // fwd+grad+step) for an apples-to-apples ns comparison.
    std::printf("n_in=%d n_out=%d num_cpus=%d\n", n_in, n_out, num_cpus);
    std::printf("%6s %14s %14s %14s\n", "batch", "fwd_ms", "bwd0_ms", "bwdX_ms");

    for (int batch : batches) {
        std::vector<float> x(std::size_t(batch) * n_in), y(std::size_t(batch) * n_out),
            dy(std::size_t(batch) * n_out), dx(std::size_t(batch) * n_in), dw(w.size());
        for (auto& v : x)
            v = dist(rng);
        for (auto& v : dy)
            v = dist(rng);

        // warm-up must exercise every op the timed loop calls (incl.
        // lr!=0) or the first real call pays one-time spin-up cost.
        for (int i = 0; i < 5; ++i) {
            didldo_forward(x.data(), w.data(), y.data(), batch, n_in, n_out);
            didldo_backward(x.data(), dy.data(), w.data(), square_avg.data(), dx.data(), dw.data(),
                            batch, n_in, n_out, 1e-3f, num_cpus);
        }

        double t_fwd = 0, t_bwd0 = 0, t_bwdX = 0;
        for (int r = 0; r < reps; ++r) {
            double t0 = now_ms();
            didldo_forward(x.data(), w.data(), y.data(), batch, n_in, n_out);
            double t1 = now_ms();
            t_fwd += t1 - t0;

            t0 = now_ms();
            didldo_forward(x.data(), w.data(), y.data(), batch, n_in, n_out);
            didldo_backward(x.data(), dy.data(), w.data(), square_avg.data(), dx.data(), dw.data(),
                            batch, n_in, n_out, 0.0f, num_cpus);
            t1 = now_ms();
            t_bwd0 += t1 - t0;

            t0 = now_ms();
            didldo_forward(x.data(), w.data(), y.data(), batch, n_in, n_out);
            didldo_backward(x.data(), dy.data(), w.data(), square_avg.data(), dx.data(), dw.data(),
                            batch, n_in, n_out, 1e-3f, num_cpus);
            t1 = now_ms();
            t_bwdX += t1 - t0;
        }
        std::printf("%6d %14.4f %14.4f %14.4f\n", batch, t_fwd / reps, t_bwd0 / reps,
                    t_bwdX / reps);
    }
    return 0;
}
