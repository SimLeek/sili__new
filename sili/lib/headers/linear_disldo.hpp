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
 * @brief Dense-input forward pass with inline importance tracking update.
 *
 * @param input          [batch x in_cols] row-major dense.
 * @param batch, in_cols Input dimensions.
 * @param weights        Layer state (importance updated in place if learning_rate != 0).
 * @param output         [batch x out_cols] accumulated into (caller zeroes first).
 * @param learning_rate  Importance update rate (0 = off). Controls activity-based
 *                       importance tracking (|x|*|h|*lr). Does NOT change weight
 *                       values -- those are updated only by backward_dense() via
 *                       the task gradient.
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

    // value_scale is gradient-trainable (disldo_backward), so -- same as
    // every per-synapse weight -- it needs a forward-side importance
    // update too (Hebbian activity correlation, not just the backward
    // gradient step). Pre-size for safe indexed writes inside the
    // per-row parallel loop below (each row is thread-exclusive, so no
    // race once sized).
    if (weights.value_scale_importance.size() < n_in)
        weights.value_scale_importance.resize(n_in, value_type(0));

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
            // value_scale's own forward importance signal: this row's
            // total contribution to the output, same activity-correlation
            // update as a per-synapse weight's, applied once per row
            // (sum first) for the same reason value_scale's backward
            // update sums first -- see disldo_backward.
            double row_contrib_sum = 0.0;
            for (std::size_t e = 0; e < n_row; ++e) {
                const COL_TYPE    col = cursor.advance();
                const std::size_t vb  = L.elem_start[r] + e;
                const value_type  w_stored = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                const value_type  out_scale = weights.get_output_scale(col);
                const value_type  w        = w_stored * val_scale * out_scale;   // -> true units

                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                    if (iv == value_type(0)) continue;
                    const value_type contrib = w * iv;
                    mo[static_cast<std::size_t>(b) * n_out + col] += contrib;
                    row_contrib_sum += static_cast<double>(contrib);

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
            if (learning_rate != value_type(0) && row_contrib_sum != 0.0) {
                // Normalize by n_row -- same reasoning as backward's
                // scale_eff_lr = lr/nnz_this_row: row_contrib_sum grows
                // with fan-out (more synapses -> larger raw sum), and
                // without this a wide row's importance would blow up
                // and freeze value_scale's own backward step immediately.
                const value_type avg_contrib = static_cast<value_type>(row_contrib_sum)
                                              / static_cast<value_type>(n_row);
                value_type vs_imp = weights.value_scale_importance[r];
                vs_imp += avg_contrib * learning_rate / (value_type(1) + std::abs(vs_imp));
                weights.value_scale_importance[r] = vs_imp;
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

    // output_scale's forward importance: its own "contrib" is the final
    // output value at that column (post fold reduction above), same
    // activity-correlation formula as everything else. Only when
    // output_scale is actually trainable (see output_scale_is_trainable);
    // each column is independent, so this parallelizes trivially.
    if (learning_rate != value_type(0) && weights.output_scale_is_trainable) {
        if (weights.output_scale_importance.size() < n_out)
            weights.output_scale_importance.resize(n_out, value_type(0));
        #pragma omp parallel for num_threads(num_cpus) schedule(static)
        for (std::size_t c = 0; c < n_out; ++c) {
            const std::size_t deg = c < weights.out_degree.size()
                ? static_cast<std::size_t>(weights.out_degree[c]) : 0;
            if (deg == 0) continue;
            double col_contrib_sum = 0.0;
            for (SIZE_TYPE b = 0; b < batch; ++b)
                col_contrib_sum += static_cast<double>(output[static_cast<std::size_t>(b) * n_out + c]);
            if (col_contrib_sum == 0.0) continue;
            // Normalize by out_degree[c] -- same reasoning as backward's
            // col_eff_lr = lr/out_degree[c]: output[.,c] grows with
            // fan-in (more rows feeding this column -> larger sum).
            const value_type avg_contrib = static_cast<value_type>(col_contrib_sum)
                                          / static_cast<value_type>(deg);
            value_type os_imp = weights.output_scale_importance[c];
            os_imp += avg_contrib * learning_rate / (value_type(1) + std::abs(os_imp));
            weights.output_scale_importance[c] = os_imp;
        }
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

    // output_scale's own gradient, symmetric to value_scale's: a column
    // can be touched by many rows, spread across threads by the outer
    // #pragma omp for (over rows) -- each thread accumulates into its own
    // [n_out]-sized slice, reduced after the parallel region (same
    // pattern as t_dx above). Only applied if output_scale_is_trainable
    // (set by set_output_scale_raw) -- not a size check, since the resize
    // below runs unconditionally for safe reads regardless of mode.
    std::vector<value_type> t_col_grad(static_cast<std::size_t>(num_cpus) * n_out, value_type(0));
    const bool output_scale_trainable = weights.output_scale_is_trainable;

    // Pre-size value_scale/output_scale so that direct indexed writes
    // from within the parallel region are safe (resize would race if
    // called per-thread).
    if (weights.value_scale.size() < n_in)
        weights.value_scale.resize(n_in, value_type(1));
    if (weights.output_scale.size() < n_out)
        weights.output_scale.resize(n_out, value_type(1));
    if (weights.value_scale_importance.size() < n_in)
        weights.value_scale_importance.resize(n_in, value_type(0));
    if (weights.output_scale_importance.size() < n_out)
        weights.output_scale_importance.resize(n_out, value_type(0));

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid = omp_get_thread_num();
        value_type* mdx  = t_dx.data() + static_cast<std::size_t>(tid) * dst;
        value_type* mcol = t_col_grad.data() + static_cast<std::size_t>(tid) * n_out;

        // Per-thread importance stats accumulators -- see disldo_forward's
        // comment and update_importance_stats()'s THREAD SAFETY note.
        // Value stats (update_value_stats_aggregate) are intentionally NOT
        // tracked here: stored weight values are only changed by backward,
        // but value_scale is learned directly via gradient descent (see the
        // scale_eff_lr update below), so there's no Hoyer-based adaptive
        // policy decision that needs live weight stats between explicit
        // recompute_stats() calls. Importance stats ARE needed in backward
        // because importance gets gradient updates here (backward) as well as
        // activity-correlation updates in forward_dense() (forward).
        double local_sum_abs_new_i = 0.0, local_sum_abs_old_i = 0.0;
        double local_sum_sq_new_i  = 0.0, local_sum_sq_old_i  = 0.0;
        value_type local_max_new_i = value_type(0);

        #pragma omp for schedule(static)
        for (std::size_t r = 0; r < n_in; ++r) {
            const std::size_t nnz_this_row = L.row_nnz(r);
            if (nnz_this_row == 0) continue;
            // lr_row/nnz_this_row (per conversation): a row with more
            // synapses gets more simultaneous per-synapse nudges each
            // backward pass, so the AGGREGATE shift in that row's behavior
            // scales roughly with nnz_this_row for a fixed learning_rate --
            // dividing by nnz_this_row keeps the aggregate update comparable
            // across rows regardless of connection count (matters here
            // specifically because synaptogenesis makes nnz_this_row
            // genuinely vary within one layer). The layer-wide equivalent
            // (lr_layer/total_nnz) needs no kernel support at all -- a
            // caller can just pre-divide learning_rate by layer.nnz
            // themselves, since that quantity doesn't vary within a call.
            const value_type effective_lr = lr_per_row_nnz
                ? learning_rate / static_cast<value_type>(nnz_this_row)
                : learning_rate;

            // value_scale gradient ALWAYS divides by nnz_this_row,
            // independent of lr_per_row_nnz. Reason: scale_grad_sum
            // accumulates nnz_this_row*batch contributions (one per
            // synapse per batch sample), so it's approximately
            // nnz_this_row * batch * average_contribution. Dividing by
            // nnz_this_row normalizes that back to the average, matching
            // the semantics of a gradient on a single scalar parameter
            // (not a vector of n weights).
            const value_type scale_eff_lr =
                learning_rate / static_cast<value_type>(nnz_this_row);

            auto cursor = dc.row_cursor(r);
            const value_type imp_scale = weights.get_importance_scale(r);
            const value_type val_scale = weights.get_value_scale(r);
            // value_scale gradient: sum first across ALL (synapse, batch)
            // pairs for this row, then apply lr ONCE. See conversation:
            // applying lr per-individual-contribution inside the innermost
            // loop risks each increment falling below ULP(value_scale) in
            // float32 and disappearing. The double accumulator + single
            // application avoids that.
            double scale_grad_sum = 0.0;
            for (std::size_t e = 0; e < nnz_this_row; ++e) {
                const COL_TYPE    col = cursor.advance();
                const std::size_t vb  = L.elem_start[r] + e;
                const value_type  cw_orig = ValueAccessor<VALUES_TYPE>::get_w  (dc.values, vb);
                const value_type  ci_orig = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                const value_type  out_scale     = weights.get_output_scale(col);
                const value_type  combined_scale = val_scale * out_scale;
                value_type cw  = cw_orig * combined_scale;   // -> true units
                value_type ci  = ci_orig * imp_scale;   // -> true units

                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type iv  = input[static_cast<std::size_t>(b) * in_cols + r];
                    const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col];
                    const value_type g   = dyv * iv;

                    if (learning_rate != value_type(0)) {
                        ci -= g * effective_lr;
                        cw += (-effective_lr * g) / (value_type(1) + std::abs(ci));
                        // dL/d(val_scale[r]) = stored_w * out_scale[col] * dy * input
                        // dL/d(out_scale[col]) = stored_w * val_scale[r] * dy * input
                        // (true_w = stored_w * val_scale * out_scale, so each
                        // scale's gradient holds the OTHER factor fixed)
                        scale_grad_sum += static_cast<double>(cw_orig) * static_cast<double>(out_scale) * g;
                        mcol[col] += cw_orig * val_scale * g;
                    }
                    mdx[static_cast<std::size_t>(b) * in_cols + r] += cw * dyv;
                }
                if (learning_rate != value_type(0)) {
                    ValueAccessor<VALUES_TYPE>::set(dc.values, vb, cw / combined_scale, ci / imp_scale);
                    const value_type actual_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                    local_sum_abs_new_i += std::abs(static_cast<double>(actual_imp));
                    local_sum_abs_old_i += std::abs(static_cast<double>(ci_orig));
                    local_sum_sq_new_i  += static_cast<double>(actual_imp) * actual_imp;
                    local_sum_sq_old_i  += static_cast<double>(ci_orig) * ci_orig;
                    local_max_new_i = std::max(local_max_new_i, std::abs(actual_imp));
                }
            }
            if (learning_rate != value_type(0)) {
                // Same damping pattern as a per-synapse weight: importance
                // updates first (undamped), then the value_scale step
                // itself is damped by the freshly-updated importance.
                const value_type raw_update = static_cast<value_type>(scale_eff_lr * scale_grad_sum);
                weights.value_scale_importance[r] -= raw_update;
                const value_type vs_imp = weights.value_scale_importance[r];
                weights.value_scale[r] -= raw_update / (value_type(1) + std::abs(vs_imp));
            }
        }

        if (learning_rate != value_type(0)) {
            #pragma omp critical
            {
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

    if (learning_rate != value_type(0) && output_scale_trainable) {
        // output_scale[c]'s gradient, reduced across threads then applied
        // once per column -- same "sum first, apply lr once" reasoning as
        // value_scale's own update. Normalizes by out_degree[c] (how many
        // rows feed this output), the column-axis equivalent of
        // nnz_this_row; a column with zero connections is skipped.
        for (std::size_t c = 0; c < n_out; ++c) {
            const std::size_t deg = c < weights.out_degree.size()
                ? static_cast<std::size_t>(weights.out_degree[c]) : 0;
            if (deg == 0) continue;
            double col_grad_sum = 0.0;
            for (int t = 0; t < num_cpus; ++t)
                col_grad_sum += t_col_grad[static_cast<std::size_t>(t) * n_out + c];
            const value_type col_eff_lr = learning_rate / static_cast<value_type>(deg);
            // Same importance-damping pattern as value_scale's own update.
            const value_type raw_update = static_cast<value_type>(col_eff_lr * col_grad_sum);
            weights.output_scale_importance[c] -= raw_update;
            const value_type os_imp = weights.output_scale_importance[c];
            weights.output_scale[c] -= raw_update / (value_type(1) + std::abs(os_imp));
        }
    }
}
