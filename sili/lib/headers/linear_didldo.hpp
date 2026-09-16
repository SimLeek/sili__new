#pragma once
#ifdef SILI_HAVE_MKL
#include <mkl.h>
#endif
#include <cmath>
#include <cstddef>
#include <vector>

// ── DIDLDO: Dense Input, Dense Linear, Dense Output ──────────────────────────
//
// Group B of the two-engine-group architecture: DISLDO/SISLDO (Group A)
// share sparse/block4 weight storage; DIDLDO/SIDLDO (Group B) share this
// DENSE weight storage instead. Full rationale, the sgemv-vs-sgemm
// dispatch measurements, and the MKL threading-handoff investigation all
// live in TODO_BATCH_BLOCKING.md ("Also queued -> DIDLDO/SIDLDO") -- not
// repeated here. fp32 only for now (BLAS is inherently single-precision;
// fp8/fp4 would need a dense staging buffer, separate future work). Only
// compiled in when the optional `mkl`/`mkl-include` pip extra was present
// at build time (`pip install sili[mkl]`, see setup.py's `_find_mkl()`);
// every symbol here is guarded behind `SILI_HAVE_MKL`.

#ifdef SILI_HAVE_MKL

// Persistent dense weight + RMSprop state. square_avg/dw_scratch are
// backward-only scratch, allocated once via resize(), never per-call.
struct DenseLinearWeights {
    std::vector<float> w;          // [n_in x n_out] row-major
    std::vector<float> square_avg; // [n_in x n_out] RMSprop running avg, same shape as w
    std::vector<float> dw_scratch; // [n_in x n_out] backward-only scratch
    std::size_t n_in = 0, n_out = 0;

    void resize(std::size_t rows, std::size_t cols) {
        n_in = rows;
        n_out = cols;
        w.assign(rows * cols, 0.0f);
        square_avg.assign(rows * cols, 0.0f);
        dw_scratch.resize(rows * cols);
    }
};

// y[batch,n_out] = x[batch,n_in] @ w[n_in,n_out]. batch==1 dispatches to
// sgemv, not sgemm -- see TODO_BATCH_BLOCKING.md for why (sgemm pays
// fixed blocked/packed-GEMM setup cost an M=1 call doesn't need).
inline void didldo_forward(const float* x, DenseLinearWeights& weights, float* y, int batch,
                           int num_cpus) {
    (void)num_cpus; // MKL's own thread count is fixed at link/env time, not per-call here
    const int n_in = static_cast<int>(weights.n_in);
    const int n_out = static_cast<int>(weights.n_out);
    if (batch == 1) {
        cblas_sgemv(CblasRowMajor, CblasTrans, n_in, n_out, 1.0f, weights.w.data(), n_out, x, 1,
                    0.0f, y, 1);
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, n_out, n_in, 1.0f, x, n_in,
                weights.w.data(), n_out, 0.0f, y, n_out);
}

// Always computes dx+dw; gates an in-place RMSprop update on lr != 0
// (lr=0.0 = grad-only, lr!=0 = grad+update, one function, no separate
// optimizer call -- matches disldo_backward's own shape). dx is
// [batch,n_in], caller-owned.
inline void didldo_backward(const float* x, const float* dy, DenseLinearWeights& weights, float* dx,
                            int batch, int num_cpus, float lr) {
    (void)num_cpus;
    const int n_in = static_cast<int>(weights.n_in);
    const int n_out = static_cast<int>(weights.n_out);
    float* w = weights.w.data();
    float* dw = weights.dw_scratch.data();

    // dx[batch,n_in] = dy[batch,n_out] @ w[n_in,n_out]^T -- same M=1 GEMV
    // dispatch as didldo_forward, same reason.
    if (batch == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n_in, n_out, 1.0f, w, n_out, dy, 1, 0.0f, dx, 1);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, n_in, n_out, 1.0f, dy, n_out, w,
                    n_out, 0.0f, dx, n_in);
    }
    // dw[n_in,n_out] = x[batch,n_in]^T @ dy[batch,n_out]. K=batch=1
    // (outer-product/sger shape) not special-cased -- future micro-opt.
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_in, n_out, batch, 1.0f, x, n_in, dy,
                n_out, 0.0f, dw, n_out);

    if (lr == 0.0f)
        return;

    // In-place RMSprop routed entirely through MKL (VML + BLAS-1), zero
    // hand-rolled #pragma omp -- see TODO_BATCH_BLOCKING.md for why.
    static thread_local std::vector<float> tmp;
    static thread_local std::vector<float> denom;
    const std::size_t n = weights.n_in * weights.n_out;
    tmp.resize(n);
    denom.resize(n);
    constexpr float alpha = 0.99f;
    constexpr float eps = 1e-8f;
    const int ni = static_cast<int>(n);

    vsMul(ni, dw, dw, tmp.data());                        // tmp = dw^2
    cblas_sscal(ni, alpha, weights.square_avg.data(), 1); // square_avg *= alpha
    cblas_saxpy(ni, 1.0f - alpha, tmp.data(), 1, weights.square_avg.data(), 1);
    vsSqrt(ni, weights.square_avg.data(), denom.data());
    vsLinearFrac(ni, denom.data(), denom.data(), 1.0f, eps, 0.0f, 1.0f, denom.data());
    vsDiv(ni, dw, denom.data(), tmp.data());
    cblas_saxpy(ni, -lr, tmp.data(), 1, w, 1);
}

#endif // SILI_HAVE_MKL
