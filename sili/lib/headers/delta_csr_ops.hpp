#ifndef __DELTA_CSR_OPS_HPP_
#define __DELTA_CSR_OPS_HPP_

// Split out of sparse_struct.hpp (see conversation). Whole-structure memory
// operations (compact/expand_headroom -- opposite operations, see their own
// docstrings) and the actual forward/backward computation
// (delta_csr_forward/delta_csr_backward, the SISLDO sparse-input path).

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
                    const value_type  wval    = ValueAccessor<VALUES_TYPE>::get_w(dc.values, wptr);
                    const value_type  contrib = wval * in_val;

                    if (learning_rate != 0) {
                        value_type cur_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, wptr);
                        cur_imp += contrib * learning_rate / (value_type(1) + std::abs(cur_imp));
                        ValueAccessor<VALUES_TYPE>::set(dc.values, wptr, wval, cur_imp);
                    }

                    thread_output[batch_offset + out_idx] += contrib;
                    if (thread_contrib)
                        thread_contrib[in_idx] += in_val * wval;
                }
            }
            #pragma omp barrier
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

// ── Backward pass ─────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_backward(
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& in_tensor,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& out_grad_sparse,
    typename ValueAccessor<VALUES_TYPE>::value_type* input_gradients,
    typename ValueAccessor<VALUES_TYPE>::value_type* output_gradients,
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_input_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_grad_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type   learning_rate = 0.01f,
    const int    num_cpus = 4)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    
    for (SIZE_TYPE i = 0; i < in_tensor.rows; ++i)
        for (SIZE_TYPE j = (*in_tensor.ptrs[0])[i]; j < (*in_tensor.ptrs[0])[i+1]; ++j)
            neuron_input_accum[(*in_tensor.indices[0])[j]] +=
                std::abs((*in_tensor.values[0])[j]);

    for (SIZE_TYPE i = 0; i < out_grad_sparse.rows; ++i)
        for (SIZE_TYPE j = (*out_grad_sparse.ptrs[0])[i]; j < (*out_grad_sparse.ptrs[0])[i+1]; ++j)
            neuron_grad_accum[(*out_grad_sparse.indices[0])[j]] +=
                std::abs((*out_grad_sparse.values[0])[j]);

    auto& dc = weights.connections;
    if (dc.empty()) return;

    const auto& L           = dc.layout;
    const SIZE_TYPE batch_size  = in_tensor.rows;
    const std::size_t num_inputs  = L.rows;

    std::vector<SIZE_TYPE> weight_grad_offsets;
    std::vector<SIZE_TYPE> input_grad_offsets;

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid      = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        for (SIZE_TYPE batch = 0; batch < batch_size; ++batch) {
            const SIZE_TYPE batch_start = (*in_tensor.ptrs[0])[batch];
            const SIZE_TYPE batch_end   = (*in_tensor.ptrs[0])[batch + 1];
            const SIZE_TYPE batch_nnz   = batch_end - batch_start;
            const SIZE_TYPE og_start    = (*out_grad_sparse.ptrs[0])[batch];
            const SIZE_TYPE og_end      = (*out_grad_sparse.ptrs[0])[batch + 1];

            #pragma omp single
            {
                weight_grad_offsets.resize(batch_nnz + 1);
                weight_grad_offsets[0] = 0;
                for (SIZE_TYPE i = 0; i < batch_nnz; ++i) {
                    const SIZE_TYPE in_idx = (*in_tensor.indices[0])[batch_start + i];
                    weight_grad_offsets[i + 1] = weight_grad_offsets[i]
                        + static_cast<SIZE_TYPE>(L.row_nnz(in_idx));
                }
                
                input_grad_offsets.resize(num_inputs + 1);
                input_grad_offsets[0] = 0;
                for (std::size_t i = 0; i < num_inputs; ++i)
                    input_grad_offsets[i + 1] = input_grad_offsets[i]
                        + static_cast<SIZE_TYPE>(L.row_nnz(i));
            }

            // ── weight update ─────────────────────────────────────────────────
            {
                const SIZE_TYPE total_work = weight_grad_offsets[batch_nnz];
                const SIZE_TYPE chunk      = (total_work + nthreads - 1) / nthreads;
                const SIZE_TYPE w_start    = std::min(static_cast<SIZE_TYPE>(tid) * chunk, total_work);
                const SIZE_TYPE w_end      = std::min(w_start + chunk, total_work);

                if (w_start < w_end) {
                    SIZE_TYPE ip = static_cast<SIZE_TYPE>(
                        std::upper_bound(weight_grad_offsets.begin(),
                                         weight_grad_offsets.end(), w_start)
                        - weight_grad_offsets.begin()) - 1;

                    SIZE_TYPE last_ip = std::numeric_limits<SIZE_TYPE>::max();
                    DeltaCSRRowCursor<COL_TYPE> cursor;

                    for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                        while (ip + 1 < batch_nnz && weight_grad_offsets[ip + 1] <= w) ++ip;

                        const SIZE_TYPE  in_idx      = (*in_tensor.indices[0])[batch_start + ip];
                        const value_type in_val      = (*in_tensor.values[0]) [batch_start + ip];
                        const SIZE_TYPE  elem_offset = w - weight_grad_offsets[ip];

                        if (ip != last_ip) {
                            cursor  = DeltaCSRRowCursor<COL_TYPE>(dc.indices_buf.data(), L, in_idx);
                            cursor.advance_to(elem_offset);
                            last_ip = ip;
                        } else {
                            cursor.advance();
                        }

                        const SIZE_TYPE   out_idx = static_cast<SIZE_TYPE>(cursor.col());
                        const std::size_t wptr    = L.elem_start[in_idx] + elem_offset;
                        const value_type  grad    = output_gradients[out_idx * batch_size + batch] * in_val;

                        value_type cur_w   = ValueAccessor<VALUES_TYPE>::get_w(dc.values, wptr);
                        value_type cur_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, wptr);

                        cur_imp -= grad * learning_rate;
                        cur_w   += (-learning_rate * grad) / (value_type(1) + std::abs(cur_imp));

                        ValueAccessor<VALUES_TYPE>::set(dc.values, wptr, cur_w, cur_imp);
                    }
                }
            }

            // ── input gradients ───────────────────────────────────────────────
            {
                const SIZE_TYPE total_work = input_grad_offsets[num_inputs];
                const SIZE_TYPE chunk      = (total_work + nthreads - 1) / nthreads;
                const SIZE_TYPE w_start    = std::min(static_cast<SIZE_TYPE>(tid) * chunk, total_work);
                const SIZE_TYPE w_end      = std::min(w_start + chunk, total_work);

                if (w_start < w_end) {
                    SIZE_TYPE in_idx = static_cast<SIZE_TYPE>(
                        std::upper_bound(input_grad_offsets.begin(),
                                         input_grad_offsets.end(), w_start)
                        - input_grad_offsets.begin()) - 1;

                    SIZE_TYPE last_in_idx = std::numeric_limits<SIZE_TYPE>::max();
                    DeltaCSRRowCursor<COL_TYPE> cursor;
                    SIZE_TYPE og_ptr = og_start;

                    for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                        while (in_idx + 1 < static_cast<SIZE_TYPE>(num_inputs)
                               && input_grad_offsets[in_idx + 1] <= w) {
                            ++in_idx;
                            og_ptr = og_start;
                        }

                        const SIZE_TYPE elem_offset = w - input_grad_offsets[in_idx];

                        if (in_idx != last_in_idx) {
                            cursor = DeltaCSRRowCursor<COL_TYPE>(dc.indices_buf.data(), L, in_idx);
                            cursor.advance_to(elem_offset);
                            last_in_idx = in_idx;
                        } else {
                            cursor.advance();
                        }

                        const SIZE_TYPE  out_idx  = static_cast<SIZE_TYPE>(cursor.col());
                        const std::size_t wptr    = L.elem_start[in_idx] + elem_offset;

                        while (og_ptr < og_end &&
                               (*out_grad_sparse.indices[0])[og_ptr] < out_idx)
                            ++og_ptr;
                        if (og_ptr >= og_end ||
                            (*out_grad_sparse.indices[0])[og_ptr] != out_idx)
                            continue;

                        const value_type og_val = (*out_grad_sparse.values[0])[og_ptr];
                        input_gradients[in_idx] += ValueAccessor<VALUES_TYPE>::get_w(dc.values, wptr) * og_val;
                    }
                }
            }

            #pragma omp barrier
        }
    }
}

#endif
