#pragma once
#include "csr.hpp"
#include "sparse_struct.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// ── DISLDO: Dense Input, Sparse Linear, Dense Output ─────────────────────────
//
// Generic over VALUES_TYPE via ValueAccessor -- works identically for
// FP4BiPacked (default, 4-bit) and DeltaCSRBiValues<float> (32-bit fallback),
// matching delta_csr_forward (the SISLDO/sparse-input forward equivalent
// in delta_csr_ops.hpp) and delta_csr_synap_row_step / delta_csr_build_probes,
// which already use this same pattern.
//
// Supersedes the previous float32/absolute-CSR disldo_forward/disldo_backward
// (which never used DeltaCSRLayout/FP4BiPacked at all -- see conversation).
// Dense-input walk is embarrassingly parallel by input row, unlike the
// sparse-input SISLDO path which needs a work-offset table to balance
// threads across a variable-density CSR batch.

// ── forward ───────────────────────────────────────────────────────────────────

/**
 * @brief Dense-input forward pass with inline Hebbian importance update.
 *
 * @param input          [batch x in_cols] row-major dense.
 * @param batch, in_cols Input dimensions.
 * @param weights        Layer state (importance updated in place if learning_rate != 0).
 * @param output         [batch x out_cols] accumulated into (caller zeroes first).
 * @param learning_rate  Hebbian importance update rate (0 = off).
 * @param num_cpus       OpenMP thread count.
 *
 * NOTE (test): with learning_rate=0, output must equal the dense matmul
 * input @ W_dense where W_dense[r,c] = weight of synapse (r->c). Same
 * reference check used for delta_csr_forward and for this session's
 * standalone disldo_ops.hpp (see conversation) -- both passed it.
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void disldo_forward(
    const typename ValueAccessor<VALUES_TYPE>::value_type* input,
    SIZE_TYPE    batch,
    SIZE_TYPE    in_cols,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    typename ValueAccessor<VALUES_TYPE>::value_type* output,
    typename ValueAccessor<VALUES_TYPE>::value_type  learning_rate = 0.01f,
    int          num_cpus = 4)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;
    const auto& L  = dc.layout;
    if (dc.empty()) return;

    const std::size_t n_in  = L.rows;
    const std::size_t n_out = L.cols;
    const std::size_t ost   = static_cast<std::size_t>(batch) * n_out;

    std::vector<value_type> t_out(static_cast<std::size_t>(num_cpus) * ost, value_type(0));

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid = omp_get_thread_num();
        value_type* mo = t_out.data() + static_cast<std::size_t>(tid) * ost;

        // Per-thread local accumulators -- see update_importance_stats()'s
        // THREAD SAFETY comment (delta_csr_types.hpp). Calling
        // weights.update_importance_stats() directly from inside this
        // parallel loop would race on the shared importance_l1/l2_sq/
        // max_abs fields (a real bug found and fixed -- see conversation).
        // Each thread sums locally here; one aggregate call per thread
        // (not per synapse) after the loop applies the combined total.
        double local_sum_abs_new = 0.0, local_sum_abs_old = 0.0;
        double local_sum_sq_new  = 0.0, local_sum_sq_old  = 0.0;
        value_type local_max_new = value_type(0);

        #pragma omp for schedule(static)
        for (std::size_t r = 0; r < n_in; ++r) {
            const std::size_t n_row = L.row_nnz(r);
            if (n_row == 0) continue;

            auto cursor = dc.row_cursor(r);
            const value_type imp_scale = weights.get_importance_scale(r);
            const value_type val_scale = weights.get_value_scale(r);
            for (std::size_t e = 0; e < n_row; ++e) {
                const COL_TYPE    col = cursor.advance();
                const std::size_t vb  = L.elem_start[r] + e;
                const value_type  w_stored = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                const value_type  w        = w_stored * val_scale;   // -> true units

                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                    if (iv == value_type(0)) continue;
                    const value_type contrib = w * iv;
                    mo[static_cast<std::size_t>(b) * n_out + col] += contrib;

                    if (learning_rate != value_type(0)) {
                        const value_type stored_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                        value_type imp = stored_imp * imp_scale;   // -> true units
                        imp += contrib * learning_rate / (value_type(1) + std::abs(imp));
                        ValueAccessor<VALUES_TYPE>::set(dc.values, vb, w_stored, imp / imp_scale);
                        // Read back the ACTUAL post-quantization stored value -- FP4BiPacked
                        // rounds to the nearest FP4_TABLE entry, so it can differ from what
                        // was just written. Stats must track what's really in the buffer.
                        const value_type actual_stored = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                        local_sum_abs_new += std::abs(static_cast<double>(actual_stored));
                        local_sum_abs_old += std::abs(static_cast<double>(stored_imp));
                        local_sum_sq_new  += static_cast<double>(actual_stored) * actual_stored;
                        local_sum_sq_old  += static_cast<double>(stored_imp) * stored_imp;
                        local_max_new = std::max(local_max_new, std::abs(actual_stored));
                    }
                }
            }
        }

        // One aggregate call per THREAD (not per synapse) -- critical
        // section cost is now O(num_cpus), not O(nnz).
        if (learning_rate != value_type(0)) {
            #pragma omp critical
            {
                weights.update_importance_stats_aggregate(
                    local_sum_abs_new, local_sum_abs_old,
                    local_sum_sq_new,  local_sum_sq_old,
                    local_max_new);
            }
        }
    }

    for (int t = 0; t < num_cpus; ++t) {
        const value_type* s = t_out.data() + static_cast<std::size_t>(t) * ost;
        for (std::size_t i = 0; i < ost; ++i) output[i] += s[i];
    }
}

// ── backward ─────────────────────────────────────────────────────────────────

/**
 * @brief Dense-input backward: weight + importance update, dx, accumulators.
 *
 * Weight/importance update is parallelised over ROWS (not synapses) since
 * DeltaCSRRowCursor decodes sequentially within a row -- each row is
 * independent (unique elem_start range), so no races.
 *
 * @param input             [batch x in_cols].
 * @param output_grad       [batch x out_cols].
 * @param weights           Layer state, modified in place.
 * @param input_grad        [batch x in_cols], accumulated into (caller zeroes).
 * @param neuron_input_accum [in_cols]  |input| accumulator for synaptogenesis.
 * @param neuron_grad_accum  [out_cols] |output_grad| accumulator for synaptogenesis.
 * @param learning_rate     Update step.
 * @param num_cpus          Thread count.
 *
 * NOTE (test): with learning_rate=0, input_grad must equal W_dense^T @ output_grad
 * per batch sample, weights/importance unchanged. Same reference check as
 * delta_csr_backward.
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void disldo_backward(
    const typename ValueAccessor<VALUES_TYPE>::value_type* input,
    SIZE_TYPE    batch,
    SIZE_TYPE    in_cols,
    const typename ValueAccessor<VALUES_TYPE>::value_type* output_grad,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    typename ValueAccessor<VALUES_TYPE>::value_type* input_grad,
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_input_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_grad_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type  learning_rate = 0.01f,
    int          num_cpus = 4,
    bool         lr_per_row_nnz = false)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;
    const auto& L = dc.layout;
    const std::size_t n_in  = L.rows;
    const std::size_t n_out = L.cols;

    for (SIZE_TYPE b = 0; b < batch; ++b) {
        for (std::size_t r = 0; r < n_in; ++r)
            neuron_input_accum[r] += std::abs(input[static_cast<std::size_t>(b) * in_cols + r]);
        for (std::size_t c = 0; c < n_out; ++c)
            neuron_grad_accum[c]  += std::abs(output_grad[static_cast<std::size_t>(b) * n_out + c]);
    }

    if (dc.empty()) return;

    const std::size_t dst = static_cast<std::size_t>(batch) * in_cols;
    std::vector<value_type> t_dx(static_cast<std::size_t>(num_cpus) * dst, value_type(0));

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid = omp_get_thread_num();
        value_type* mdx = t_dx.data() + static_cast<std::size_t>(tid) * dst;

        // Per-thread local accumulators -- see disldo_forward's comment
        // and update_importance_stats()'s THREAD SAFETY note.
        double local_sum_abs_new_w = 0.0, local_sum_abs_old_w = 0.0;
        double local_sum_sq_new_w  = 0.0, local_sum_sq_old_w  = 0.0;
        value_type local_max_new_w = value_type(0);
        double local_sum_abs_new_i = 0.0, local_sum_abs_old_i = 0.0;
        double local_sum_sq_new_i  = 0.0, local_sum_sq_old_i  = 0.0;
        value_type local_max_new_i = value_type(0);

        #pragma omp for schedule(static)
        for (std::size_t r = 0; r < n_in; ++r) {
            const std::size_t n_row = L.row_nnz(r);
            if (n_row == 0) continue;
            // lr_row/row_nnz (per conversation): a row with more synapses
            // gets more simultaneous per-synapse nudges each backward pass,
            // so the AGGREGATE shift in that row's behavior scales roughly
            // with row_nnz for a fixed learning_rate -- dividing by row_nnz
            // keeps the aggregate update comparable across rows regardless
            // of connection count (matters here specifically because
            // synaptogenesis makes row_nnz genuinely vary within one layer).
            // The layer-wide equivalent (lr_layer/nnz) needs no kernel
            // support at all -- a caller can just pre-divide learning_rate
            // by layer.nnz themselves, since that quantity doesn't vary
            // within a single call the way row_nnz does.
            const value_type effective_lr = lr_per_row_nnz
                ? learning_rate / static_cast<value_type>(n_row)
                : learning_rate;

            auto cursor = dc.row_cursor(r);
            const value_type imp_scale = weights.get_importance_scale(r);
            const value_type val_scale = weights.get_value_scale(r);
            for (std::size_t e = 0; e < n_row; ++e) {
                const COL_TYPE    col = cursor.advance();
                const std::size_t vb  = L.elem_start[r] + e;
                const value_type  cw_orig = ValueAccessor<VALUES_TYPE>::get_w  (dc.values, vb);
                const value_type  ci_orig = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                value_type cw  = cw_orig * val_scale;   // -> true units
                value_type ci  = ci_orig * imp_scale;   // -> true units

                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type iv  = input[static_cast<std::size_t>(b) * in_cols + r];
                    const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col];
                    const value_type g   = dyv * iv;

                    if (learning_rate != value_type(0)) {
                        ci -= g * effective_lr;
                        cw += (-effective_lr * g) / (value_type(1) + std::abs(ci));
                    }
                    mdx[static_cast<std::size_t>(b) * in_cols + r] += cw * dyv;
                }
                if (learning_rate != value_type(0)) {
                    ValueAccessor<VALUES_TYPE>::set(dc.values, vb, cw / val_scale, ci / imp_scale);
                    // Read back post-quantization actuals -- see disldo_forward's comment.
                    const value_type actual_w   = ValueAccessor<VALUES_TYPE>::get_w  (dc.values, vb);
                    const value_type actual_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                    local_sum_abs_new_w += std::abs(static_cast<double>(actual_w));
                    local_sum_abs_old_w += std::abs(static_cast<double>(cw_orig));
                    local_sum_sq_new_w  += static_cast<double>(actual_w) * actual_w;
                    local_sum_sq_old_w  += static_cast<double>(cw_orig) * cw_orig;
                    local_max_new_w = std::max(local_max_new_w, std::abs(actual_w));
                    local_sum_abs_new_i += std::abs(static_cast<double>(actual_imp));
                    local_sum_abs_old_i += std::abs(static_cast<double>(ci_orig));
                    local_sum_sq_new_i  += static_cast<double>(actual_imp) * actual_imp;
                    local_sum_sq_old_i  += static_cast<double>(ci_orig) * ci_orig;
                    local_max_new_i = std::max(local_max_new_i, std::abs(actual_imp));
                }
            }
        }

        if (learning_rate != value_type(0)) {
            #pragma omp critical
            {
                weights.update_value_stats_aggregate(
                    local_sum_abs_new_w, local_sum_abs_old_w,
                    local_sum_sq_new_w,  local_sum_sq_old_w, local_max_new_w);
                weights.update_importance_stats_aggregate(
                    local_sum_abs_new_i, local_sum_abs_old_i,
                    local_sum_sq_new_i,  local_sum_sq_old_i, local_max_new_i);
            }
        }
    }

    for (int t = 0; t < num_cpus; ++t) {
        const value_type* s = t_dx.data() + static_cast<std::size_t>(t) * dst;
        for (std::size_t i = 0; i < dst; ++i) input_grad[i] += s[i];
    }
}
