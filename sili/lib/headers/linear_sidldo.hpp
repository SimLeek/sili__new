#pragma once
#include "linear_didldo.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <vector>

// ── SIDLDO: Sparse Input, Dense Linear, Dense Output ─────────────────────────
//
// Group B sibling of DIDLDO (see TODO_BATCH_BLOCKING.md "Also queued"),
// sharing the same DenseLinearWeights storage -- interchangeable with
// DIDLDO per-call, same as DISLDO/SISLDO share sparse/block4 storage.
// Two forward paths, dispatched by batch: low-batch gathers only the
// weight ROWS touched by the union of the batch's nonzero input
// positions (one MKL sgemv/sgemm call); high-batch instead transposes
// the batch's sparsity into a feature-major CSR and does a custom,
// nnz-balanced, hand-rolled accumulate -- a direct A/B
// (probe_nested_mkl.cpp) found MKL's own sger ~3x SLOWER than a plain
// scalar loop for this per-row-broadcast shape, so this path calls MKL
// nowhere at all. fp32 only, same reason as DIDLDO.

#ifdef SILI_HAVE_MKL

// Forward-only scratch, persistent and grow-only, kept separate from
// DenseLinearWeights so a DIDLDO-only layer never allocates it. Covers
// both forward paths in one object so callers need only one scratch.
struct SidldoForwardScratch {
    // low-batch (union gather)
    std::vector<int> active_row;
    std::vector<int> inverse_map;
    std::vector<float> w_compact;
    std::vector<float> x_compact;
    // high-batch (feature-major CSR + nnz-balanced accumulate)
    std::vector<int> fm_ptrs;
    std::vector<int> fm_sample;
    std::vector<float> fm_val;
    std::vector<int> fm_cursor;

    // Per-thread compact output: touched_samples/local_out only cover
    // samples THIS thread actually wrote to (not the full batch) --
    // sample_local_map is the same union/inverse-map trick the low-batch
    // path already uses, applied to the OUTPUT side. sample_local_map is
    // grow-only and only ever reset at the TOUCHED entries (never a full
    // clear), so its steady-state per-call cost is proportional to real
    // work, not batch. See sidldo_forward_high_batch for why this
    // replaced an earlier full-[batch x n_out]-per-thread design: that
    // one's reduction cost was O(threads*batch*n_out), independent of
    // density, and measurably lost to sidldo_forward_union_gather at
    // high batch + low density as a result.
    struct ThreadLocal {
        std::vector<int> sample_local_map; // size >= batch, -1 sentinel
        std::vector<int> touched_samples;
        std::vector<float> local_out; // touched_samples.size() * n_out

        void ensure_batch(std::size_t batch) {
            if (sample_local_map.size() < batch)
                sample_local_map.resize(batch, -1);
        }
    };
    std::vector<ThreadLocal> thread_local_data;

    void ensure_union_gather(std::size_t n_in, std::size_t n_out, std::size_t batch) {
        if (inverse_map.size() < n_in)
            inverse_map.assign(n_in, -1);
        if (active_row.capacity() < n_in)
            active_row.reserve(n_in);
        if (w_compact.size() < n_in * n_out)
            w_compact.resize(n_in * n_out);
        if (x_compact.size() < batch * n_in)
            x_compact.resize(batch * n_in);
    }

    void ensure_high_batch(std::size_t n_in, std::size_t /*n_out*/, std::size_t /*batch*/,
                           std::size_t nnz, int num_threads) {
        if (fm_ptrs.size() < n_in + 1)
            fm_ptrs.resize(n_in + 1);
        if (fm_cursor.size() < n_in)
            fm_cursor.resize(n_in);
        if (fm_sample.size() < nnz)
            fm_sample.resize(nnz);
        if (fm_val.size() < nnz)
            fm_val.resize(nnz);
        if (thread_local_data.size() < std::size_t(num_threads))
            thread_local_data.resize(std::size_t(num_threads));
    }
};

// x_ptrs[batch+1]/x_idx[nnz]/x_val[nnz]: per-sample CSR, row=sample,
// column=input feature index. y[batch,n_out] overwritten.
inline void sidldo_forward_union_gather(const int* x_ptrs, const int* x_idx, const float* x_val,
                                        int batch, DenseLinearWeights& weights,
                                        SidldoForwardScratch& scratch, float* y) {
    const int n_out = static_cast<int>(weights.n_out);
    scratch.ensure_union_gather(weights.n_in, weights.n_out, std::size_t(batch));

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

// Transposes the batch's per-sample CSR into a feature-major CSR (one
// row per input feature, listing which samples touch it), partitions
// its total nnz evenly across threads, and has each thread accumulate
// into its OWN compact touched-sample buffer -- a sample's active
// features can legitimately split across two threads' nnz ranges, so
// per-thread-private buffers + a final combine avoid that write race
// (same shape disldo_forward/backward already use for their own
// parallel reductions, but sized by actual touched work here, not the
// full batch -- see the ThreadLocal comment above for why that mattered).
// No BLAS calls anywhere in this function -- see the file header comment
// for why.
inline void sidldo_forward_high_batch(const int* x_ptrs, const int* x_idx, const float* x_val,
                                      int batch, DenseLinearWeights& weights,
                                      SidldoForwardScratch& scratch, float* y, int num_cpus) {
    const int n_in = static_cast<int>(weights.n_in);
    const int n_out = static_cast<int>(weights.n_out);
    const int nnz = x_ptrs[batch];
    scratch.ensure_high_batch(std::size_t(n_in), std::size_t(n_out), std::size_t(batch),
                              std::size_t(nnz), num_cpus);

    if (nnz == 0) {
        std::fill(y, y + std::size_t(batch) * n_out, 0.0f);
        return;
    }

    // Counting-sort transpose, serial: O(nnz+n_in), small relative to
    // the accumulate step below at this engine's target (high) batch.
    std::fill(scratch.fm_ptrs.begin(), scratch.fm_ptrs.begin() + n_in + 1, 0);
    for (int i = 0; i < nnz; ++i)
        ++scratch.fm_ptrs[x_idx[i] + 1];
    for (int r = 0; r < n_in; ++r)
        scratch.fm_ptrs[r + 1] += scratch.fm_ptrs[r];
    std::copy(scratch.fm_ptrs.begin(), scratch.fm_ptrs.begin() + n_in, scratch.fm_cursor.begin());
    for (int b = 0; b < batch; ++b)
        for (int i = x_ptrs[b]; i < x_ptrs[b + 1]; ++i) {
            const int c = x_idx[i];
            const int pos = scratch.fm_cursor[c]++;
            scratch.fm_sample[std::size_t(pos)] = b;
            scratch.fm_val[std::size_t(pos)] = x_val[i];
        }

#pragma omp parallel num_threads(num_cpus)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const int lo = int((std::int64_t(nnz) * tid) / nthreads);
        const int hi = int((std::int64_t(nnz) * (tid + 1)) / nthreads);

        auto& ts = scratch.thread_local_data[std::size_t(tid)];
        ts.ensure_batch(std::size_t(batch));
        ts.touched_samples.clear();
        ts.local_out.clear();

        if (lo < hi) {
            int row = int(std::upper_bound(scratch.fm_ptrs.begin(),
                                           scratch.fm_ptrs.begin() + n_in + 1, lo) -
                          scratch.fm_ptrs.begin()) -
                      1;
            int pos = lo;
            while (pos < hi) {
                const int row_end = std::min(hi, scratch.fm_ptrs[std::size_t(row) + 1]);
                const float* w_row = weights.w.data() + std::size_t(row) * n_out;
                for (int i = pos; i < row_end; ++i) {
                    const int b = scratch.fm_sample[std::size_t(i)];
                    const float v = scratch.fm_val[std::size_t(i)];
                    int local = ts.sample_local_map[std::size_t(b)];
                    if (local == -1) {
                        local = int(ts.touched_samples.size());
                        ts.sample_local_map[std::size_t(b)] = local;
                        ts.touched_samples.push_back(b);
                        ts.local_out.resize(ts.local_out.size() + std::size_t(n_out), 0.0f);
                    }
                    float* out_row = ts.local_out.data() + std::size_t(local) * n_out;
                    for (int c = 0; c < n_out; ++c)
                        out_row[c] += v * w_row[c];
                }
                pos = row_end;
                ++row;
            }
        }
        // Targeted reset -- only the entries this thread actually
        // touched, maintaining the -1 invariant without a full clear.
        for (int b : ts.touched_samples)
            ts.sample_local_map[std::size_t(b)] = -1;
    }

    // Combine: y must end up fully dense regardless of density (one
    // unavoidable O(batch*n_out) fill), but the per-thread contributions
    // are only as many as were actually touched -- serial, since the
    // same sample can appear in more than one thread's touched list
    // (features of one sample split across threads) and a parallel
    // combine would need its own disjoint-ownership scheme for no clear
    // win at this scale (num_cpus is small, touched work is bounded by
    // nnz).
    std::fill(y, y + std::size_t(batch) * n_out, 0.0f);
    for (int t = 0; t < num_cpus; ++t) {
        auto& ts = scratch.thread_local_data[std::size_t(t)];
        for (std::size_t j = 0; j < ts.touched_samples.size(); ++j) {
            const int b = ts.touched_samples[j];
            const float* src = ts.local_out.data() + j * std::size_t(n_out);
            float* dst = y + std::size_t(b) * n_out;
            for (int c = 0; c < n_out; ++c)
                dst[c] += src[c];
        }
    }
}

// Dispatches on batch -- see TODO_BATCH_BLOCKING.md for where the
// crossover was measured.
constexpr int SIDLDO_HIGH_BATCH_THRESHOLD = 64;

inline void sidldo_forward(const int* x_ptrs, const int* x_idx, const float* x_val, int batch,
                           DenseLinearWeights& weights, SidldoForwardScratch& scratch, float* y,
                           int num_cpus) {
    if (batch >= SIDLDO_HIGH_BATCH_THRESHOLD)
        sidldo_forward_high_batch(x_ptrs, x_idx, x_val, batch, weights, scratch, y, num_cpus);
    else
        sidldo_forward_union_gather(x_ptrs, x_idx, x_val, batch, weights, scratch, y);
}

// ── backward ──────────────────────────────────────────────────────────────────
//
// Dense x, SPARSE dy (CSR) -- matches disldo_backward_sparse_grad's own
// established convention (dense input, sparse output gradient), not
// forward's sparse-input convention: dy's sparsity here comes from
// whatever downstream sparsifying op produced it, independent of
// whether THIS layer's own forward input was sparse. Restricts to the
// COLUMNS of W where dy is nonzero (the transpose of forward's row
// restriction) for both dx and the dW/RMSprop update. Union-gather only
// for now, no high-batch variant yet -- see TODO_BATCH_BLOCKING.md.

// Backward-only scratch, persistent and grow-only.
struct SidldoBackwardScratch {
    std::vector<int> active_col;
    std::vector<int> inverse_map;     // [n_out], -1 sentinel
    std::vector<float> w_col_compact; // [num_active x n_in]: row i = W[:,active_col[i]]
    std::vector<float> dy_compact;    // [batch x num_active]
    std::vector<float> dw_compact;    // [n_in x num_active]

    void ensure(std::size_t n_in, std::size_t n_out, std::size_t batch) {
        if (inverse_map.size() < n_out)
            inverse_map.assign(n_out, -1);
        if (active_col.capacity() < n_out)
            active_col.reserve(n_out);
        if (w_col_compact.size() < n_out * n_in)
            w_col_compact.resize(n_out * n_in);
        if (dy_compact.size() < batch * n_out)
            dy_compact.resize(batch * n_out);
        if (dw_compact.size() < n_in * n_out)
            dw_compact.resize(n_in * n_out);
    }
};

// x[batch,n_in] dense. dy_ptrs[batch+1]/dy_idx[nnz]/dy_val[nnz]: per-
// sample CSR, row=sample, column=output feature index. dx[batch,n_in]
// overwritten (caller-owned). Always computes dx+dw, gates the in-place
// RMSprop update on lr != 0 (lr=0.0 = grad-only, lr!=0 = grad+update),
// matching disldo_backward/didldo_backward's own shape -- no separate
// optimizer call anywhere.
inline void sidldo_backward(const float* x, const int* dy_ptrs, const int* dy_idx,
                            const float* dy_val, int batch, DenseLinearWeights& weights,
                            SidldoBackwardScratch& scratch, float* dx, float lr) {
    const int n_in = static_cast<int>(weights.n_in);
    const int n_out = static_cast<int>(weights.n_out);
    scratch.ensure(weights.n_in, weights.n_out, std::size_t(batch));

    // Serial union + gather, same reasoning as forward's union-gather:
    // targets a batch range small enough that a custom #pragma omp
    // region here would just pay handoff tax for no benefit.
    scratch.active_col.clear();
    const int nnz = dy_ptrs[batch];
    for (int i = 0; i < nnz; ++i) {
        const int c = dy_idx[i];
        if (scratch.inverse_map[c] == -1) {
            scratch.inverse_map[c] = static_cast<int>(scratch.active_col.size());
            scratch.active_col.push_back(c);
        }
    }
    const int num_active = static_cast<int>(scratch.active_col.size());

    // Gather W's active COLUMNS into a compact, contiguous [num_active x
    // n_in] buffer -- a strided read from row-major W (unavoidable, W's
    // layout is fixed by forward/DIDLDO's needs), but a contiguous write,
    // and the buffer this produces is exactly the shape both GEMMs below
    // need (no further transpose).
    for (int i = 0; i < num_active; ++i) {
        const int c = scratch.active_col[i];
        float* dst = scratch.w_col_compact.data() + std::size_t(i) * n_in;
        for (int r = 0; r < n_in; ++r)
            dst[r] = weights.w[std::size_t(r) * n_out + c];
    }

    std::fill(scratch.dy_compact.begin(),
              scratch.dy_compact.begin() + std::size_t(batch) * num_active, 0.0f);
    for (int b = 0; b < batch; ++b) {
        float* row = scratch.dy_compact.data() + std::size_t(b) * num_active;
        for (int i = dy_ptrs[b]; i < dy_ptrs[b + 1]; ++i)
            row[scratch.inverse_map[dy_idx[i]]] = dy_val[i];
    }
    for (int i = 0; i < num_active; ++i)
        scratch.inverse_map[scratch.active_col[i]] = -1;

    if (num_active == 0) {
        std::fill(dx, dx + std::size_t(batch) * n_in, 0.0f);
        return;
    }

    // dx[batch,n_in] = dy_compact[batch,num_active] @ w_col_compact[num_active,n_in]
    if (batch == 1) {
        cblas_sgemv(CblasRowMajor, CblasTrans, num_active, n_in, 1.0f, scratch.w_col_compact.data(),
                    n_in, scratch.dy_compact.data(), 1, 0.0f, dx, 1);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, n_in, num_active, 1.0f,
                    scratch.dy_compact.data(), num_active, scratch.w_col_compact.data(), n_in, 0.0f,
                    dx, n_in);
    }

    // dw_compact[n_in,num_active] = x[batch,n_in]^T @ dy_compact[batch,num_active].
    // K=batch=1 (sger shape) not special-cased, same precedent as DIDLDO.
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_in, num_active, batch, 1.0f, x, n_in,
                scratch.dy_compact.data(), num_active, 0.0f, scratch.dw_compact.data(), num_active);

    if (lr == 0.0f)
        return;

    // In-place RMSprop, applied directly at the real (strided) weight
    // positions -- hand-rolled, not VML, since VML's vector ops need
    // contiguous input and gathering/scattering a compact square_avg
    // buffer just to use VML would cost more than this plain loop saves
    // (same lesson as the high-batch forward path: a hand-rolled loop
    // often beats routing a small, memory-bound op through a library
    // built for large contiguous ones). Reuses dw_compact, already
    // contiguous from the GEMM above.
    constexpr float alpha = 0.99f;
    constexpr float eps = 1e-8f;
    for (int i = 0; i < num_active; ++i) {
        const int c = scratch.active_col[i];
        for (int r = 0; r < n_in; ++r) {
            const float g = scratch.dw_compact[std::size_t(r) * num_active + i];
            float& sq = weights.square_avg[std::size_t(r) * n_out + c];
            sq = alpha * sq + (1.0f - alpha) * g * g;
            weights.w[std::size_t(r) * n_out + c] -= lr * g / (std::sqrt(sq) + eps);
        }
    }
}

#endif // SILI_HAVE_MKL
