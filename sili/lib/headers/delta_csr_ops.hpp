#ifndef __DELTA_CSR_OPS_HPP_
#define __DELTA_CSR_OPS_HPP_

// Split out of sparse_struct.hpp (see conversation). Whole-structure memory
// operations (compact/expand_headroom -- opposite operations, see their own
// docstrings) and the actual forward/backward computation: delta_csr_forward
// (SISLDO -- sparse input) and delta_csr_backward_sparse_grad (dense input,
// sparse gradient -- deliberately NOT sparse input; see its own docstring
// for why sparse-input backward permanently loses the ability to correct
// "didn't fire & should have").

#include "delta_csr_types.hpp"
#include "delta_csr_memory.hpp"

// ── compact ────────────────────────────────────────────────────────────────────

/**
 * @brief Repack a DeltaCSRWeights so every row occupies exactly its active
 * bytes/elements, zero inter-row blank space -- both the index buffer
 * (byte_start/byte_end) AND the values buffer (elem_start/elem_end) are
 * separate growth-headroom axes and both get compacted here.
 *
 * delta_csr_from_absolute()'s reserved headroom (the blank_fraction fixed
 * earlier this session) is correct and necessary for a LIVE, training
 * model -- rows need O(1) append room for synaptogenesis. For a freshly
 * converted or long-since-pruned model being saved/measured for
 * deployment, that headroom is pure unused padding that nnz()/
 * total_alloc_bytes() otherwise count as consumed. Use compact() before
 * saving/measuring; call reserve_indices()/reserve_values() again after
 * loading if this model is about to resume training rather than just be
 * measured or deployed.
 *
 * Generic over VALUES_TYPE via ValueAccessor -- one implementation for both
 * FP4BiPacked and DeltaCSRBiValues<float>, matching the rest of this file's
 * pattern (delta_csr_forward/backward/build_probes/synap_row_step).
 *
 * NOTE (test): must be lossless -- decode every synapse from the input and
 * the output (column indices via row_cursor, weight/importance via
 * ValueAccessor::get_w/get_imp), compare row by row; must match exactly.
 * Also verify total_alloc_bytes()/total_alloc_elems() strictly decrease (or
 * stay equal) after compacting a delta_csr_from_absolute()-constructed
 * layer, and that a second compact() call is idempotent (sizes unchanged).
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> compact(
    const DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    const auto& L = dc.layout;

    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> out;
    out.layout.rows = L.rows;
    out.layout.cols = L.cols;
    out.layout.byte_start.resize(L.rows + 1);
    out.layout.byte_end.resize(L.rows);
    out.layout.elem_start.resize(L.rows + 1);
    out.layout.elem_end.resize(L.rows);

    std::size_t total_bytes = 0, total_elems = 0;
    for (std::size_t r = 0; r < L.rows; ++r) {
        total_bytes += L.row_byte_len(r);
        total_elems += L.row_nnz(r);
    }

    out.indices_buf.assign(total_bytes, uint8_t(0));
    ValueAccessor<VALUES_TYPE>::resize(out.values, total_elems, value_type(0));

    std::size_t bcursor = 0, ecursor = 0;
    for (std::size_t r = 0; r < L.rows; ++r) {
        const std::size_t blen = L.row_byte_len(r);
        if (blen > 0)
            std::memcpy(out.indices_buf.data() + bcursor,
                       dc.indices_buf.data() + L.byte_start[r], blen);
        out.layout.byte_start[r] = bcursor;
        out.layout.byte_end[r]   = bcursor + blen;
        bcursor += blen;

        const std::size_t n = L.row_nnz(r);
        for (std::size_t k = 0; k < n; ++k) {
            const value_type w   = ValueAccessor<VALUES_TYPE>::get_w  (dc.values, L.elem_start[r] + k);
            const value_type imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, L.elem_start[r] + k);
            ValueAccessor<VALUES_TYPE>::set(out.values, ecursor + k, w, imp);
        }
        out.layout.elem_start[r] = ecursor;
        out.layout.elem_end[r]   = ecursor + n;
        ecursor += n;
    }
    out.layout.byte_start[L.rows] = bcursor;
    out.layout.elem_start[L.rows] = ecursor;
    out.layout.total_nnz = L.total_nnz;

    out.max_indices_bytes = dc.max_indices_bytes;
    out.max_values_bytes  = dc.max_values_bytes;

    return out;
}
// ── expand ─────────────────────────────────────────────────────────────────────

/**
 * @brief Opposite of compact(): restore growth headroom to a DeltaCSRWeights
 * that has none (or not enough) -- typically because compact() removed it.
 *
 * Reuses delta_csr_from_absolute()'s already-tested headroom-reservation
 * logic (extract to absolute CSR via delta_csr_to_absolute, then rebuild)
 * rather than duplicating it. blank_fraction is the SAME parameter
 * delta_csr_from_absolute takes -- 0.2 (20%) restores the same headroom a
 * freshly-converted layer gets by default; pass a larger value before a
 * synaptogenesis-heavy phase, smaller if memory is tight and only modest
 * growth is expected.
 *
 * NOTE (test): after compact() then expand(), row_rebuild/synap_row_step
 * must succeed on rows that failed immediately post-compact (this is the
 * actual bug this function exists to let callers work around -- see
 * conversation, "silent failure is the worst case"). Also verify expand()
 * is lossless (same content as compact() already checks).
 *
 * Behavior note: expand() NORMALIZES headroom to exactly blank_fraction of
 * current content size -- it does not add blank_fraction on top of
 * whatever headroom the input already had (delta_csr_to_absolute extracts
 * only the actual synapses, not existing slack, so there's nothing to add
 * to). Calling expand() on an already-roomy layer with a smaller
 * blank_fraction than it currently has will shrink its headroom, same as
 * compact() would, just not all the way to zero. Consistent with compact()
 * normalizing to exactly 0% -- expand() normalizes to exactly
 * blank_fraction, not "at least blank_fraction."
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> expand_headroom(
    const DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc,
    float blank_fraction = 0.2f)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    std::vector<SIZE_TYPE>  ptrs, idx;
    std::vector<value_type> w, imp;
    delta_csr_to_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(dc, ptrs, idx, w, imp);

    const std::size_t n = idx.size();
    return delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
        ptrs, idx, w, imp, dc.layout.rows, dc.layout.cols,
        n * (1.0 + blank_fraction) * (uleb128_max_bytes<COL_TYPE>() + 1) + 4096,
        static_cast<std::size_t>(n * (1.0 + blank_fraction)) + 64,
        blank_fraction);
}
// ── Forward pass ─────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_forward(
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& input_tensor,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    typename ValueAccessor<VALUES_TYPE>::value_type* output,
    typename ValueAccessor<VALUES_TYPE>::value_type   learning_rate = 0.01f,
    const int    num_cpus = 4,
    typename ValueAccessor<VALUES_TYPE>::value_type* original_contributions_output = nullptr)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;
    if (dc.empty()) return;

    const auto& L           = dc.layout;
    const std::size_t out_cols    = L.cols;
    const std::size_t num_outputs = static_cast<std::size_t>(input_tensor.rows) * out_cols;
    const std::size_t num_inputs  = L.rows;

    std::vector<value_type> all_outputs(static_cast<std::size_t>(num_cpus) * num_outputs,
                                        value_type(0));
    std::vector<value_type> all_contributions(
        original_contributions_output
            ? static_cast<std::size_t>(num_cpus) * num_inputs : 0,
        value_type(0));

    std::vector<SIZE_TYPE> work_offsets;

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid      = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        value_type* thread_output = all_outputs.data() + static_cast<std::size_t>(tid) * num_outputs;
        value_type* thread_contrib = original_contributions_output
            ? all_contributions.data() + static_cast<std::size_t>(tid) * num_inputs
            : nullptr;

        // Per-thread local accumulators, persisting across all batches --
        // see disldo_forward's THREAD SAFETY comment. Aggregated once,
        // after the batch loop below, not per synapse.
        double local_sum_abs_new = 0.0, local_sum_abs_old = 0.0;
        double local_sum_sq_new  = 0.0, local_sum_sq_old  = 0.0;
        value_type local_max_new = value_type(0);

        for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
            const SIZE_TYPE batch_start  = (*input_tensor.ptrs[0])[batch];
            const SIZE_TYPE batch_end    = (*input_tensor.ptrs[0])[batch + 1];
            const SIZE_TYPE batch_nnz    = batch_end - batch_start;
            const SIZE_TYPE batch_offset = batch * static_cast<SIZE_TYPE>(out_cols);

            #pragma omp single
            {
                work_offsets.resize(batch_nnz + 1);
                work_offsets[0] = 0;
                for (SIZE_TYPE i = 0; i < batch_nnz; ++i) {
                    const SIZE_TYPE in_idx = (*input_tensor.indices[0])[batch_start + i];
                    work_offsets[i + 1] = work_offsets[i]
                        + static_cast<SIZE_TYPE>(L.row_nnz(in_idx));
                }
            }

            const SIZE_TYPE total_work = work_offsets[batch_nnz];
            const SIZE_TYPE chunk      = (total_work + nthreads - 1) / nthreads;
            const SIZE_TYPE w_start    = std::min(static_cast<SIZE_TYPE>(tid) * chunk, total_work);
            const SIZE_TYPE w_end      = std::min(w_start + chunk, total_work);

            if (w_start < w_end) {
                SIZE_TYPE ip = static_cast<SIZE_TYPE>(
                    std::upper_bound(work_offsets.begin(), work_offsets.end(), w_start)
                    - work_offsets.begin()) - 1;

                SIZE_TYPE last_ip = std::numeric_limits<SIZE_TYPE>::max();
                DeltaCSRRowCursor<COL_TYPE> cursor;

                for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                    while (ip + 1 < batch_nnz && work_offsets[ip + 1] <= w) ++ip;

                    const SIZE_TYPE  in_idx      = (*input_tensor.indices[0])[batch_start + ip];
                    const value_type in_val      = (*input_tensor.values[0]) [batch_start + ip];
                    const SIZE_TYPE  elem_offset = w - work_offsets[ip];

                    if (ip != last_ip) {
                        cursor  = DeltaCSRRowCursor<COL_TYPE>(dc.indices_buf.data(), L, in_idx);
                        cursor.advance_to(elem_offset);
                        last_ip = ip;
                    } else {
                        cursor.advance();
                    }

                    const SIZE_TYPE   out_idx = static_cast<SIZE_TYPE>(cursor.col());
                    const std::size_t wptr    = L.elem_start[in_idx] + elem_offset;
                    const value_type  wval_stored = ValueAccessor<VALUES_TYPE>::get_w(dc.values, wptr);
                    // Scale lookups per-synapse, not hoisted: in_idx (the row) varies
                    // within this loop (work-offset iteration, not a simple per-row
                    // loop) -- unlike disldo_forward/backward, can't fix it once per
                    // outer iteration.
                    const value_type  val_scale = weights.get_value_scale(in_idx);
                    const value_type  wval      = wval_stored * val_scale;   // -> true units
                    const value_type  contrib   = wval * in_val;

                    if (learning_rate != 0) {
                        const value_type imp_scale  = weights.get_importance_scale(in_idx);
                        const value_type stored_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, wptr);
                        value_type cur_imp = stored_imp * imp_scale;   // -> true units
                        cur_imp += contrib * learning_rate / (value_type(1) + std::abs(cur_imp));
                        ValueAccessor<VALUES_TYPE>::set(dc.values, wptr, wval_stored, cur_imp / imp_scale);
                        // Read back post-quantization actual -- see disldo_forward's comment.
                        const value_type actual_stored = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, wptr);
                        local_sum_abs_new += std::abs(static_cast<double>(actual_stored));
                        local_sum_abs_old += std::abs(static_cast<double>(stored_imp));
                        local_sum_sq_new  += static_cast<double>(actual_stored) * actual_stored;
                        local_sum_sq_old  += static_cast<double>(stored_imp) * stored_imp;
                        local_max_new = std::max(local_max_new, std::abs(actual_stored));
                    }

                    thread_output[batch_offset + out_idx] += contrib;
                    if (thread_contrib)
                        thread_contrib[in_idx] += in_val * wval;
                }
            }
            #pragma omp barrier
        }

        if (learning_rate != 0) {
            #pragma omp critical
            {
                weights.update_importance_stats_aggregate(
                    local_sum_abs_new, local_sum_abs_old,
                    local_sum_sq_new,  local_sum_sq_old, local_max_new);
            }
        }

        for (int stride = 1; stride < nthreads; stride <<= 1) {
            #pragma omp barrier
            const int src = tid + stride;
            if (tid % (stride << 1) == 0 && src < nthreads) {
                const value_type* src_out = all_outputs.data() +
                                            static_cast<std::size_t>(src) * num_outputs;
                for (std::size_t i = 0; i < num_outputs; ++i)
                    thread_output[i] += src_out[i];
                if (thread_contrib) {
                    const value_type* src_con =
                        all_contributions.data() + static_cast<std::size_t>(src) * num_inputs;
                    for (std::size_t i = 0; i < num_inputs; ++i)
                        thread_contrib[i] += src_con[i];
                }
            }
        }
    }

    for (std::size_t i = 0; i < num_outputs; ++i)
        output[i] += all_outputs[i];
    if (original_contributions_output)
        for (std::size_t i = 0; i < num_inputs; ++i)
            original_contributions_output[i] += all_contributions[i];
}

// delta_csr_backward (sparse input + sparse gradient) removed here -- see
// conversation. Confirmed wrong design: sparse input in backward permanently
// loses the ability to correct "didn't fire & should have" (a row not in the
// sparse input representation has no computational path to receive gradient
// at all, regardless of how strong the signal is). Only "fired & shouldn't
// have" could ever be fixed. Replaced by delta_csr_backward_sparse_grad
// below (dense input, sparse gradient) -- the only sparse-gradient backward
// variant that should exist. Confirmed zero real callers before removal
// (only this file's own definition matched a search for "delta_csr_backward(").

// ── Backward pass (dense input, sparse gradient) ────────────────────────────
//
// Per conversation: this is the ONLY sparse-gradient backward variant --
// there is deliberately no sparse-INPUT backward. Input is always dense
// here (available regardless of which forward path was used, since
// sparsification never destroys the underlying dense array). Only the
// GRADIENT toggles sparse/dense, matching the actual performance
// bottleneck (backward's cost is dominated by the gradient side, forward's
// by the activation side -- these are independent axes, not mirror images
// of each other).
//
// Why dense input specifically (not just "simpler to implement"):
// dx[r] = sum_c W[r,c]*dy[c] depends only on weights and the gradient, not
// on input[r] itself -- so a row whose OWN activation was zero/near-zero
// this pass still gets a correct dx, correctly telling whatever produced
// this input "you should have fired more here." A sparse-input design
// would skip that row entirely (it's not in the sparse representation at
// all), permanently losing the ability to correct this. The weight update
// DOES scale with input[r] (via `grad = dy_val * in_val`), so it naturally
// stays small for rows that didn't fire -- appropriately conservative,
// without needing to skip the row. Net effect: dense input covers both
// "fired & shouldn't have" (weight update, scales with the real input
// value) and "didn't fire & should have" (dx, weight-only, reaches the
// row regardless of its own value) -- sparse input would only ever cover
// the first.

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_backward_sparse_grad(
    const typename ValueAccessor<VALUES_TYPE>::value_type* input,   // dense [batch, n_inputs]
    SIZE_TYPE batch,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& out_grad_sparse,
    typename ValueAccessor<VALUES_TYPE>::value_type* input_gradients,  // dense [batch, n_inputs], accumulated
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_input_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_grad_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type   learning_rate = 0.01f,
    const int    num_cpus = 4,
    bool         lr_per_row_nnz = false)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;
    const auto& L = dc.layout;
    const std::size_t n_inputs = L.rows;

    for (SIZE_TYPE b = 0; b < batch; ++b)
        for (std::size_t r = 0; r < n_inputs; ++r)
            neuron_input_accum[r] += std::abs(input[b * n_inputs + r]);
    for (SIZE_TYPE i = 0; i < out_grad_sparse.rows; ++i)
        for (SIZE_TYPE j = (*out_grad_sparse.ptrs[0])[i]; j < (*out_grad_sparse.ptrs[0])[i+1]; ++j)
            neuron_grad_accum[(*out_grad_sparse.indices[0])[j]] +=
                std::abs((*out_grad_sparse.values[0])[j]);

    if (dc.empty()) return;

    // Across-batches accumulators -- each batch's #pragma omp parallel for
    // is a SEPARATE parallel region (re-created every batch iteration, not
    // one persistent region like disldo_forward/backward), so per-thread
    // locals can't persist across batches here the same way. reduction()
    // handles the within-one-batch thread-safety; these accumulate each
    // batch's reduced total, with ONE final aggregate call after the whole
    // loop -- see disldo_forward's THREAD SAFETY comment for the bug this
    // is all avoiding.
    double total_sum_abs_new_w = 0.0, total_sum_abs_old_w = 0.0;
    double total_sum_sq_new_w  = 0.0, total_sum_sq_old_w  = 0.0;
    value_type total_max_new_w = value_type(0);
    double total_sum_abs_new_i = 0.0, total_sum_abs_old_i = 0.0;
    double total_sum_sq_new_i  = 0.0, total_sum_sq_old_i  = 0.0;
    value_type total_max_new_i = value_type(0);

    // value_scale gradient: serial per-row vector accumulated across batches
    // (within each batch's parallel for, each r is unique per thread, so
    // += into scale_grad_sums[r] is race-free; across batch iterations the
    // outer loop is serial, so also race-free). Applied once after all
    // batches -- "sum first, then apply lr" per conversation.
    std::vector<double> scale_grad_sums(n_inputs, 0.0);
    if (weights.value_scale.size() < n_inputs)
        weights.value_scale.resize(n_inputs, value_type(1));

    for (SIZE_TYPE b = 0; b < batch; ++b) {
        const SIZE_TYPE og_start = (*out_grad_sparse.ptrs[0])[b];
        const SIZE_TYPE og_end   = (*out_grad_sparse.ptrs[0])[b + 1];

        double batch_sum_abs_new_w = 0.0, batch_sum_abs_old_w = 0.0;
        double batch_sum_sq_new_w  = 0.0, batch_sum_sq_old_w  = 0.0;
        value_type batch_max_new_w = value_type(0);
        double batch_sum_abs_new_i = 0.0, batch_sum_abs_old_i = 0.0;
        double batch_sum_sq_new_i  = 0.0, batch_sum_sq_old_i  = 0.0;
        value_type batch_max_new_i = value_type(0);

        #pragma omp parallel for num_threads(num_cpus) schedule(static) \
            reduction(+:batch_sum_abs_new_w, batch_sum_abs_old_w, batch_sum_sq_new_w, batch_sum_sq_old_w) \
            reduction(+:batch_sum_abs_new_i, batch_sum_abs_old_i, batch_sum_sq_new_i, batch_sum_sq_old_i) \
            reduction(max:batch_max_new_w, batch_max_new_i)
        for (std::size_t r = 0; r < n_inputs; ++r) {
            const std::size_t n_row = L.row_nnz(r);
            if (n_row == 0) continue;
            const value_type in_val = input[b * n_inputs + r];
            // lr_row/row_nnz -- see disldo_backward's comment for the full
            // reasoning (a row with more synapses gets more simultaneous
            // per-synapse nudges each pass, so dividing by row_nnz keeps
            // the aggregate update comparable across rows of different
            // connection counts).
            const value_type effective_lr = lr_per_row_nnz
                ? learning_rate / static_cast<value_type>(n_row)
                : learning_rate;

            auto cursor = dc.row_cursor(r);
            SIZE_TYPE  og_ptr   = og_start;   // fresh per row -- each row does its own merge
            value_type dx_accum = value_type(0);
            const value_type imp_scale = weights.get_importance_scale(r);
            const value_type val_scale = weights.get_value_scale(r);

            for (std::size_t e = 0; e < n_row; ++e) {
                const COL_TYPE    col = cursor.advance();
                const std::size_t vb  = L.elem_start[r] + e;
                const value_type  w_stored = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                const value_type  w        = w_stored * val_scale;   // -> true units

                // Merge-advance (both this row's columns and the gradient's
                // columns are sorted ascending) -- O(row_nnz + grad_nnz)
                // per row, not a search per synapse.
                while (og_ptr < og_end &&
                       (*out_grad_sparse.indices[0])[og_ptr] < static_cast<SIZE_TYPE>(col))
                    ++og_ptr;
                if (og_ptr >= og_end ||
                    (*out_grad_sparse.indices[0])[og_ptr] != static_cast<SIZE_TYPE>(col))
                    continue;   // this output has no significant gradient this pass -- skip

                const value_type dy_val = (*out_grad_sparse.values[0])[og_ptr];
                dx_accum += w * dy_val;   // weight-only -- reaches this row regardless of in_val

                if (learning_rate != value_type(0)) {
                    const value_type grad = dy_val * in_val;   // scales with true input value
                    const value_type stored_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                    value_type imp = stored_imp * imp_scale;   // -> true units
                    imp -= grad * effective_lr;
                    const value_type new_w = w + (-effective_lr * grad)
                                              / (value_type(1) + std::abs(imp));
                    ValueAccessor<VALUES_TYPE>::set(dc.values, vb, new_w / val_scale, imp / imp_scale);
                    // Read back post-quantization actuals -- see disldo_forward's comment.
                    const value_type actual_w   = ValueAccessor<VALUES_TYPE>::get_w  (dc.values, vb);
                    const value_type actual_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                    batch_sum_abs_new_w += std::abs(static_cast<double>(actual_w));
                    batch_sum_abs_old_w += std::abs(static_cast<double>(w_stored));
                    batch_sum_sq_new_w  += static_cast<double>(actual_w) * actual_w;
                    batch_sum_sq_old_w  += static_cast<double>(w) * w;
                    batch_max_new_w = std::max(batch_max_new_w, std::abs(actual_w));
                    batch_sum_abs_new_i += std::abs(static_cast<double>(actual_imp));
                    batch_sum_abs_old_i += std::abs(static_cast<double>(stored_imp));
                    batch_sum_sq_new_i  += static_cast<double>(actual_imp) * actual_imp;
                    batch_sum_sq_old_i  += static_cast<double>(stored_imp) * stored_imp;
                    batch_max_new_i = std::max(batch_max_new_i, std::abs(actual_imp));
                    // value_scale gradient: w_stored * dy_val * in_val
                    // (chain rule: output += w_stored * val_scale * in_val,
                    // so d(output)/d(val_scale) = w_stored * in_val).
                    // Accumulated in scale_grad_sums[r] (serial across
                    // batches, parallel-safe within a batch since r is
                    // unique per thread) -- applied once after all batches.
                    scale_grad_sums[r] += static_cast<double>(w_stored) * dy_val * in_val;
                }
            }
            input_gradients[b * n_inputs + r] += dx_accum;
        }

        total_sum_abs_new_w += batch_sum_abs_new_w; total_sum_abs_old_w += batch_sum_abs_old_w;
        total_sum_sq_new_w  += batch_sum_sq_new_w;  total_sum_sq_old_w  += batch_sum_sq_old_w;
        total_max_new_w = std::max(total_max_new_w, batch_max_new_w);
        total_sum_abs_new_i += batch_sum_abs_new_i; total_sum_abs_old_i += batch_sum_abs_old_i;
        total_sum_sq_new_i  += batch_sum_sq_new_i;  total_sum_sq_old_i  += batch_sum_sq_old_i;
        total_max_new_i = std::max(total_max_new_i, batch_max_new_i);
    }

    if (learning_rate != value_type(0)) {
        weights.update_value_stats_aggregate(
            total_sum_abs_new_w, total_sum_abs_old_w,
            total_sum_sq_new_w,  total_sum_sq_old_w, total_max_new_w);
        weights.update_importance_stats_aggregate(
            total_sum_abs_new_i, total_sum_abs_old_i,
            total_sum_sq_new_i,  total_sum_sq_old_i, total_max_new_i);

        // Apply value_scale gradient once per row, after ALL batches.
        for (std::size_t r = 0; r < n_inputs; ++r) {
            if (scale_grad_sums[r] == 0.0) continue;
            const std::size_t n_row = L.row_nnz(r);
            if (n_row == 0) continue;
            const value_type eff_lr = lr_per_row_nnz
                ? learning_rate / static_cast<value_type>(n_row)
                : learning_rate;
            weights.value_scale[r] -= static_cast<value_type>(eff_lr * scale_grad_sums[r]);
        }
    }
}

#endif
