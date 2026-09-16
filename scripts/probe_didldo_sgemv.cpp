// Probe: is DIDLDO forward's batch=1 slowness (vs torch/numpy) because
// scripts/bench_didldo_kernel.cpp calls cblas_sgemm unconditionally, even
// for the M=1 (batch=1) case, where BLAS's blocked/packed GEMM setup has
// fixed overhead a dedicated GEMV call skips entirely? numpy's `x @ w`
// for a (1,n)x(n,n) shape almost certainly special-cases into sgemv, not
// sgemm -- this checks whether calling sgemv directly here closes the gap.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

enum CBLAS_ORDER { CblasRowMajor = 101 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };

extern "C" void scipy_cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K,
                                  float alpha, const float* A, int lda, const float* B, int ldb,
                                  float beta, float* C, int ldc);
extern "C" void scipy_cblas_sgemv(int Order, int TransA, int M, int N, float alpha, const float* A,
                                  int lda, const float* X, int incX, float beta, float* Y,
                                  int incY);
extern "C" void scipy_openblas_set_num_threads(int);

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char** argv) {
    int n_in = argc > 1 ? std::atoi(argv[1]) : 288;
    int n_out = argc > 2 ? std::atoi(argv[2]) : 288;
    int num_cpus = argc > 3 ? std::atoi(argv[3]) : 8;
    scipy_openblas_set_num_threads(num_cpus);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> w(std::size_t(n_in) * n_out), x(n_in), y(n_out);
    for (auto& v : w)
        v = dist(rng);
    for (auto& v : x)
        v = dist(rng);

    const int reps = 200;
    // y[n_out] = w^T[n_out,n_in] @ x[n_in] -- w is stored row-major
    // [n_in,n_out], so "w^T" here means CblasTrans on the row-major
    // matrix as stored (A=w, lda=n_out, M=n_in rows, N=n_out cols,
    // TransA=Trans -> computes A^T @ x = [n_out] result). This is the
    // exact y=x@w forward, just expressed as a GEMV.
    for (int i = 0; i < 20; ++i)
        scipy_cblas_sgemv(CblasRowMajor, CblasTrans, n_in, n_out, 1.0f, w.data(), n_out, x.data(),
                          1, 0.0f, y.data(), 1);
    double t_gemv = 0;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_ms();
        scipy_cblas_sgemv(CblasRowMajor, CblasTrans, n_in, n_out, 1.0f, w.data(), n_out, x.data(),
                          1, 0.0f, y.data(), 1);
        t_gemv += now_ms() - t0;
    }

    for (int i = 0; i < 20; ++i)
        scipy_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, n_out, n_in, 1.0f, x.data(),
                          n_in, w.data(), n_out, 0.0f, y.data(), n_out);
    double t_gemm = 0;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_ms();
        scipy_cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, n_out, n_in, 1.0f, x.data(),
                          n_in, w.data(), n_out, 0.0f, y.data(), n_out);
        t_gemm += now_ms() - t0;
    }

    std::printf("n_in=%d n_out=%d num_cpus=%d batch=1: sgemv=%.4fms sgemm=%.4fms\n", n_in, n_out,
                num_cpus, t_gemv / reps, t_gemm / reps);
    return 0;
}
