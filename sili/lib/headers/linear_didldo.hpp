#pragma once
#ifdef SILI_HAVE_MKL
// NOT <mkl.h> -- its sparse-matrix-checker sub-header declares its own
// global `sparse_struct`, colliding with this codebase's own
// sparse_struct TEMPLATE (delta_csr_types.hpp). Only cblas+VML used
// here; neither narrower header pulls in the sparse-checker API.
#include <mkl_cblas.h>
#include <mkl_vml_functions.h>
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

// Persistent dense weight + optimizer state. ci/dw_scratch/contrib_scratch
// are backward-only scratch, allocated once via resize(), never per-call.
// ci is NOT vanilla RMSprop's square_avg -- see didldo_backward's own
// docstring for why (importance IS the optimizer,
// feedback_importance_is_already_the_optimizer memory).
struct DenseLinearWeights {
    std::vector<float> w;               // [n_in x n_out] row-major
    std::vector<float> ci;              // [n_in x n_out] BoundedRMSpropSynapsePolicy state
    std::vector<float> dw_scratch;      // [n_in x n_out] backward-only scratch
    std::vector<float> contrib_scratch; // [n_in x n_out] backward-only scratch
    std::size_t n_in = 0, n_out = 0;

    void resize(std::size_t rows, std::size_t cols) {
        n_in = rows;
        n_out = cols;
        w.assign(rows * cols, 0.0f);
        ci.assign(rows * cols, 0.0f);
        dw_scratch.resize(rows * cols);
        contrib_scratch.resize(rows * cols);
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

// Always computes dx+dw; gates an in-place update on lr != 0 (lr=0.0 =
// grad-only, lr!=0 = grad+update, one function, no separate optimizer
// call -- matches disldo_backward's own shape). dx is [batch,n_in],
// caller-owned. NOT vanilla RMSprop -- BoundedRMSpropSynapsePolicy
// (delta_csr_types.hpp), same optimizer disldo_backward uses -- see
// docs/research/sparse_rnn.rst:sparse_rnn.didldo_same_optimizer_as_disldo
// for the full why (importance IS the optimizer) and the float-vs-double
// accumulation caveat. scale_invariant is a no-op (S is always 1 here);
// accepted for signature parity with DISLDOLayerV::backward_dense only.
inline void didldo_backward(const float* x, const float* dy, DenseLinearWeights& weights, float* dx,
                            int batch, int num_cpus, float lr, bool lr_per_row_nnz = false,
                            bool damp_by_importance = true, float beta2 = 0.999f, float eps = 1e-8f,
                            float min_decay_frac = 0.0f, float max_abs_delta = 2.0f,
                            float max_ci = 100.0f, bool scale_invariant = false) {
    (void)num_cpus;
    (void)scale_invariant; // no S concept here -- see docstring above
    const int n_in = static_cast<int>(weights.n_in);
    const int n_out = static_cast<int>(weights.n_out);
    float* w = weights.w.data();
    float* dw = weights.dw_scratch.data(); // this call's g_agg, per weight

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

    // contrib_agg[i,j] = w[i,j] * sum_b(x[b,i]) -- see the RST anchor above.
    static thread_local std::vector<float> colsum_x;
    static thread_local std::vector<float> ones_batch;
    colsum_x.resize(std::size_t(n_in));
    ones_batch.assign(std::size_t(batch), 1.0f);
    cblas_sgemv(CblasRowMajor, CblasTrans, batch, n_in, 1.0f, x, n_in, ones_batch.data(), 1, 0.0f,
                colsum_x.data(), 1);
    float* contrib = weights.contrib_scratch.data();
    for (int i = 0; i < n_in; ++i) {
        const float s = colsum_x[std::size_t(i)];
        const float* wr = w + std::size_t(i) * std::size_t(n_out);
        float* cr = contrib + std::size_t(i) * std::size_t(n_out);
        for (int j = 0; j < n_out; ++j)
            cr[j] = wr[j] * s;
    }

    // ci = clip(beta2*ci + (1-beta2)*(g^2+contrib^2), min_decay_frac*ci, max_ci)
    // -- BoundedRMSpropSynapsePolicy::update_ci, verbatim.
    const std::size_t n = weights.n_in * weights.n_out;
    const int ni = static_cast<int>(n);
    float* ci = weights.ci.data();
    for (int i = 0; i < ni; ++i) {
        const float g = dw[std::size_t(i)];
        const float c = contrib[std::size_t(i)];
        const float ema = beta2 * ci[std::size_t(i)] + (1.0f - beta2) * (g * g + c * c);
        const float floor_v = min_decay_frac * ci[std::size_t(i)];
        ci[std::size_t(i)] = std::min(std::max(ema, floor_v), max_ci);
    }

    // effective_lr: nnz_row==n_out for every row here (fully dense), so
    // lr_per_row_nnz reduces to one constant divisor, not per-row.
    const float effective_lr = lr_per_row_nnz ? (lr / float(n_out)) : lr;

    // Clip BEFORE the lr multiply -- delta_csr_types.hpp:synapse_policy.
    // clip_order_and_lr_ceiling.
    for (int i = 0; i < ni; ++i) {
        const float g = dw[std::size_t(i)];
        float raw = damp_by_importance ? (-g) / (std::sqrt(ci[std::size_t(i)]) + eps) : (-g);
        raw = std::min(std::max(raw, -max_abs_delta), max_abs_delta);
        w[std::size_t(i)] += effective_lr * raw;
    }
}

#endif // SILI_HAVE_MKL
