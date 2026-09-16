#pragma once
#include "linear_didldo.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>

// ── SIDLDO: Sparse Input, Dense Linear, Dense Output ─────────────────────────
//
// Group B sibling of DIDLDO (see TODO_BATCH_BLOCKING.md "Also queued"),
// sharing the same DenseLinearWeights storage -- interchangeable with
// DIDLDO per-call, same as DISLDO/SISLDO share sparse/block4 storage.
// Forward gathers only the weight ROWS touched by the batch's union of
// nonzero input positions -- degrades toward DIDLDO as batch/density
// grow (the union fills in), so this targets low batch specifically; see
// the TODO for the high-batch feature-major-CSR design, not implemented
// here. fp32 only, same reason as DIDLDO.

#ifdef SILI_HAVE_MKL

// Forward-only scratch, persistent and grow-only, kept separate from
// DenseLinearWeights so a DIDLDO-only layer never allocates it.
struct SidldoForwardScratch {
    std::vector<int> active_row;
    std::vector<int> inverse_map;
    std::vector<float> w_compact;
    std::vector<float> x_compact;

    void ensure(std::size_t n_in, std::size_t n_out, std::size_t batch) {
        if (inverse_map.size() < n_in)
            inverse_map.assign(n_in, -1);
        if (active_row.capacity() < n_in)
            active_row.reserve(n_in);
        if (w_compact.size() < n_in * n_out)
            w_compact.resize(n_in * n_out);
        if (x_compact.size() < batch * n_in)
            x_compact.resize(batch * n_in);
    }
};

// x_ptrs[batch+1]/x_idx[nnz]/x_val[nnz]: per-sample CSR, row=sample,
// column=input feature index. y[batch,n_out] overwritten.
inline void sidldo_forward(const int* x_ptrs, const int* x_idx, const float* x_val, int batch,
                           DenseLinearWeights& weights, SidldoForwardScratch& scratch, float* y) {
    const int n_out = static_cast<int>(weights.n_out);
    scratch.ensure(weights.n_in, weights.n_out, std::size_t(batch));

    // Serial union + gather, deliberately not parallelized -- see file
    // header comment.
    scratch.active_row.clear();
    const int nnz = x_ptrs[batch];
    for (int i = 0; i < nnz; ++i) {
        const int c = x_idx[i];
        if (scratch.inverse_map[c] == -1) {
            scratch.inverse_map[c] = static_cast<int>(scratch.active_row.size());
            scratch.active_row.push_back(c);
        }
    }
    const int num_active = static_cast<int>(scratch.active_row.size());

    for (int i = 0; i < num_active; ++i) {
        const float* src = weights.w.data() + std::size_t(scratch.active_row[i]) * n_out;
        float* dst = scratch.w_compact.data() + std::size_t(i) * n_out;
        std::copy(src, src + n_out, dst);
    }

    std::fill(scratch.x_compact.begin(),
              scratch.x_compact.begin() + std::size_t(batch) * num_active, 0.0f);
    for (int b = 0; b < batch; ++b) {
        float* row = scratch.x_compact.data() + std::size_t(b) * num_active;
        for (int i = x_ptrs[b]; i < x_ptrs[b + 1]; ++i)
            row[scratch.inverse_map[x_idx[i]]] = x_val[i];
    }
    for (int i = 0; i < num_active; ++i)
        scratch.inverse_map[scratch.active_row[i]] = -1;

    if (num_active == 0) {
        std::fill(y, y + std::size_t(batch) * n_out, 0.0f);
        return;
    }
    if (batch == 1) {
        cblas_sgemv(CblasRowMajor, CblasTrans, num_active, n_out, 1.0f, scratch.w_compact.data(),
                    n_out, scratch.x_compact.data(), 1, 0.0f, y, 1);
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, n_out, num_active, 1.0f,
                scratch.x_compact.data(), num_active, scratch.w_compact.data(), n_out, 0.0f, y,
                n_out);
}

#endif // SILI_HAVE_MKL
