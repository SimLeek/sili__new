#ifndef __SISLDO_OPS_HPP_
#define __SISLDO_OPS_HPP_

// See docs/research/sisldo_ops.rst:sisldo_ops.file_split_context.

#include "delta_csr_types.hpp"
#include "delta_csr_memory.hpp"

// ── compact ────────────────────────────────────────────────────────────────────

/**
 * @brief Repack a DeltaCSRWeights so every row occupies exactly its active
 * bytes/elements, zero inter-row blank space (both index and values
 * buffers). See docs/research/sisldo_ops.rst:compact.headroom_removal.
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>
compact(const DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc) {
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
            std::memcpy(out.indices_buf.data() + bcursor, dc.indices_buf.data() + L.byte_start[r],
                        blen);
        out.layout.byte_start[r] = bcursor;
        out.layout.byte_end[r] = bcursor + blen;
        bcursor += blen;

        const std::size_t n = L.row_nnz(r);
        for (std::size_t k = 0; k < n; ++k) {
            const value_type w = ValueAccessor<VALUES_TYPE>::get_w(dc.values, L.elem_start[r] + k);
            const value_type imp =
                ValueAccessor<VALUES_TYPE>::get_imp(dc.values, L.elem_start[r] + k);
            ValueAccessor<VALUES_TYPE>::set(out.values, ecursor + k, w, imp);
        }
        out.layout.elem_start[r] = ecursor;
        out.layout.elem_end[r] = ecursor + n;
        ecursor += n;
    }
    out.layout.byte_start[L.rows] = bcursor;
    out.layout.elem_start[L.rows] = ecursor;
    out.layout.total_nnz = L.total_nnz;

    out.max_indices_bytes = dc.max_indices_bytes;
    out.max_values_bytes = dc.max_values_bytes;

    return out;
}
// ── expand ─────────────────────────────────────────────────────────────────────

/**
 * @brief Opposite of compact(): restore growth headroom to a
 * DeltaCSRWeights that has none (or not enough).
 * See docs/research/sisldo_ops.rst:expand_headroom.budget_propagation_bug.
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>
expand_headroom(const DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc,
                float blank_fraction = 0.2f) {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    std::vector<SIZE_TYPE> ptrs, idx;
    std::vector<value_type> w, imp;
    delta_csr_to_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(dc, ptrs, idx, w, imp);

    const std::size_t n = idx.size();
    // Propagate the INPUT dc's own hard limits through -- see
    // docs/research/sisldo_ops.rst:expand_headroom.budget_propagation_bug.
    return delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
        ptrs, idx, w, imp, dc.layout.rows, dc.layout.cols,
        n * (1.0 + blank_fraction) * (uleb128_max_bytes<COL_TYPE>() + 1) + 4096,
        static_cast<std::size_t>(n * (1.0 + blank_fraction)) + 64, blank_fraction,
        dc.max_indices_bytes, dc.max_values_bytes);
}

// Like expand_headroom() but sizes for a minimum per-row connection count.
// See docs/research/sisldo_ops.rst:expand_headroom_to.per_row_budget.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>
expand_headroom_to(const DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc,
                   std::size_t min_nnz_per_row, float blank_fraction = 0.2f) {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    std::vector<SIZE_TYPE> ptrs, idx;
    std::vector<value_type> w, imp;
    delta_csr_to_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(dc, ptrs, idx, w, imp);
    const std::size_t rows = dc.layout.rows;
    const std::size_t n = idx.size();
    const std::size_t budget = std::max(n, rows * min_nnz_per_row);
    // See docs/research/sisldo_ops.rst:expand_headroom_to.per_row_budget.
    return delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
        ptrs, idx, w, imp, rows, dc.layout.cols,
        budget * (1.0 + blank_fraction) * (uleb128_max_bytes<COL_TYPE>() + 1) + 4096,
        static_cast<std::size_t>(budget * (1.0 + blank_fraction)) + 64, blank_fraction,
        dc.max_indices_bytes, dc.max_values_bytes);
}
// ── Forward pass ─────────────────────────────────────────────────────────────

// No learning_rate parameter, deliberately -- see
// docs/research/sisldo_ops.rst:sisldo_forward.no_learning_rate_param.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void sisldo_forward(
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& input_tensor,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    typename ValueAccessor<VALUES_TYPE>::value_type* output, const int num_cpus = 4,
    typename ValueAccessor<VALUES_TYPE>::value_type* original_contributions_output = nullptr) {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;

    // dc.empty() does NOT mean "nothing to do" -- see
    // docs/research/sisldo_ops.rst:sisldo_forward.dc_empty_block4_bug.
    const auto& L = dc.layout;
    const std::size_t out_cols = L.cols;
    const std::size_t num_outputs = static_cast<std::size_t>(input_tensor.rows) * out_cols;
    const std::size_t num_inputs = L.rows;

    std::vector<value_type> all_outputs(static_cast<std::size_t>(num_cpus) * num_outputs,
                                        value_type(0));
    std::vector<value_type> all_contributions(
        original_contributions_output ? static_cast<std::size_t>(num_cpus) * num_inputs : 0,
        value_type(0));

    std::vector<SIZE_TYPE> work_offsets;

    if (!dc.empty()) {
#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();

            value_type* thread_output =
                all_outputs.data() + static_cast<std::size_t>(tid) * num_outputs;
            value_type* thread_contrib =
                original_contributions_output
                    ? all_contributions.data() + static_cast<std::size_t>(tid) * num_inputs
                    : nullptr;

            for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
                const SIZE_TYPE batch_start = (*input_tensor.ptrs[0])[batch];
                const SIZE_TYPE batch_end = (*input_tensor.ptrs[0])[batch + 1];
                const SIZE_TYPE batch_nnz = batch_end - batch_start;
                const SIZE_TYPE batch_offset = batch * static_cast<SIZE_TYPE>(out_cols);

#pragma omp single
                {
                    work_offsets.resize(batch_nnz + 1);
                    work_offsets[0] = 0;
                    for (SIZE_TYPE i = 0; i < batch_nnz; ++i) {
                        const SIZE_TYPE in_idx = (*input_tensor.indices[0])[batch_start + i];
                        work_offsets[i + 1] =
                            work_offsets[i] + static_cast<SIZE_TYPE>(L.row_nnz(in_idx));
                    }
                }

                const SIZE_TYPE total_work = work_offsets[batch_nnz];
                const SIZE_TYPE chunk = (total_work + nthreads - 1) / nthreads;
                const SIZE_TYPE w_start = std::min(static_cast<SIZE_TYPE>(tid) * chunk, total_work);
                const SIZE_TYPE w_end = std::min(w_start + chunk, total_work);

                if (w_start < w_end) {
                    SIZE_TYPE ip =
                        static_cast<SIZE_TYPE>(
                            std::upper_bound(work_offsets.begin(), work_offsets.end(), w_start) -
                            work_offsets.begin()) -
                        1;

                    SIZE_TYPE last_ip = std::numeric_limits<SIZE_TYPE>::max();
                    DeltaCSRRowCursor<COL_TYPE> cursor;

                    for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                        while (ip + 1 < batch_nnz && work_offsets[ip + 1] <= w)
                            ++ip;

                        const SIZE_TYPE in_idx = (*input_tensor.indices[0])[batch_start + ip];
                        const value_type in_val = (*input_tensor.values[0])[batch_start + ip];
                        const SIZE_TYPE elem_offset = w - work_offsets[ip];

                        if (ip != last_ip) {
                            cursor = DeltaCSRRowCursor<COL_TYPE>(dc.indices_buf.data(), L, in_idx);
                            cursor.advance_to(elem_offset);
                            last_ip = ip;
                        } else {
                            cursor.advance();
                        }

                        const SIZE_TYPE out_idx = static_cast<SIZE_TYPE>(cursor.col());
                        const std::size_t wptr = L.elem_start[in_idx] + elem_offset;
                        const value_type wval_stored =
                            ValueAccessor<VALUES_TYPE>::get_w(dc.values, wptr);
                        // Per-synapse scale lookup (not hoisted -- in_idx varies within
                        // this loop). Rank-N scale; fixes a real bug where output_scale
                        // was silently dropped. See
                        // docs/research/sisldo_ops.rst:sisldo_forward.output_scale_read_bug.
                        const value_type wval =
                            wval_stored * weights.get_scale(in_idx, out_idx); // -> true units
                        const value_type contrib = wval * in_val;

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
                    const value_type* src_out =
                        all_outputs.data() + static_cast<std::size_t>(src) * num_outputs;
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
    }

    for (std::size_t i = 0; i < num_outputs; ++i)
        output[i] += all_outputs[i];
    if (original_contributions_output)
        for (std::size_t i = 0; i < num_inputs; ++i)
            original_contributions_output[i] += all_contributions[i];

    // ── block4 contribution ─────────────────────────────────────────────────
    // Real bug fixed: block4-resident synapses were never read here. See
    // docs/research/sisldo_ops.rst:sisldo_forward.block4_gather_design.
    if constexpr (std::is_same_v<VALUES_TYPE, FP4BiPacked>) {
        if (weights.block4.n_tiles() > 0) {
            const auto& BL4 = weights.block4.block_layout;

            std::vector<value_type> all_b4_outputs(static_cast<std::size_t>(num_cpus) * num_outputs,
                                                   value_type(0));

            // Per-batch scratch, reused across batches (not reallocated per batch).
            std::vector<SIZE_TYPE> win_br;           // active window's block-row index
            std::vector<value_type> win_vals;        // flat, 4 per window: win_vals[4*w + li]
            std::vector<SIZE_TYPE> win_work_offsets; // cumulative block4 tile count per window

#pragma omp parallel num_threads(num_cpus)
            {
                const int tid = omp_get_thread_num();
                const int nthreads = omp_get_num_threads();
                value_type* thread_output =
                    all_b4_outputs.data() + static_cast<std::size_t>(tid) * num_outputs;

                for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
                    const SIZE_TYPE batch_start = (*input_tensor.ptrs[0])[batch];
                    const SIZE_TYPE batch_end = (*input_tensor.ptrs[0])[batch + 1];
                    const SIZE_TYPE batch_nnz = batch_end - batch_start;
                    const SIZE_TYPE batch_offset = batch * static_cast<SIZE_TYPE>(out_cols);

#pragma omp single
                    {
                        win_br.clear();
                        win_vals.clear();
                        win_work_offsets.clear();
                        win_work_offsets.push_back(0);

                        SIZE_TYPE i = 0;
                        while (i < batch_nnz) {
                            const SIZE_TYPE idx0 = (*input_tensor.indices[0])[batch_start + i];
                            const SIZE_TYPE br = idx0 / static_cast<SIZE_TYPE>(BLOCK4_TILE);
                            const SIZE_TYPE window_lo = br * static_cast<SIZE_TYPE>(BLOCK4_TILE);
                            const SIZE_TYPE window_hi =
                                window_lo + static_cast<SIZE_TYPE>(BLOCK4_TILE);

                            // Gather this window's entries -- sorted input means a
                            // single forward scan, not a search.
                            value_type local[4] = {value_type(0), value_type(0), value_type(0),
                                                   value_type(0)};
                            SIZE_TYPE j = i;
                            while (j < batch_nnz) {
                                const SIZE_TYPE idxj = (*input_tensor.indices[0])[batch_start + j];
                                if (idxj >= window_hi)
                                    break;
                                local[idxj - window_lo] =
                                    (*input_tensor.values[0])[batch_start + j];
                                ++j;
                            }

                            const std::size_t row_nnz_b4 =
                                static_cast<std::size_t>(br) < BL4.rows ? BL4.row_nnz(br) : 0;
                            if (row_nnz_b4 > 0) {
                                win_br.push_back(br);
                                win_vals.push_back(local[0]);
                                win_vals.push_back(local[1]);
                                win_vals.push_back(local[2]);
                                win_vals.push_back(local[3]);
                                win_work_offsets.push_back(win_work_offsets.back() +
                                                           static_cast<SIZE_TYPE>(row_nnz_b4));
                            }
                            i = j;
                        }
                    }

                    const SIZE_TYPE n_windows = static_cast<SIZE_TYPE>(win_br.size());
                    const SIZE_TYPE total_work = win_work_offsets.back();

                    if (total_work > 0) {
                        const SIZE_TYPE chunk = (total_work + nthreads - 1) / nthreads;
                        const SIZE_TYPE w_start =
                            std::min(static_cast<SIZE_TYPE>(tid) * chunk, total_work);
                        const SIZE_TYPE w_end = std::min(w_start + chunk, total_work);

                        if (w_start < w_end) {
                            SIZE_TYPE wi = static_cast<SIZE_TYPE>(
                                               std::upper_bound(win_work_offsets.begin(),
                                                                win_work_offsets.end(), w_start) -
                                               win_work_offsets.begin()) -
                                           1;

                            SIZE_TYPE last_wi = std::numeric_limits<SIZE_TYPE>::max();
                            DeltaCSRRowCursor<uint32_t> bc_cursor;
                            std::size_t elem_pos = 0, byte_pos = 0;

                            for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                                while (wi + 1 < n_windows && win_work_offsets[wi + 1] <= w)
                                    ++wi;

                                const SIZE_TYPE br = win_br[wi];
                                const SIZE_TYPE tile_offset = w - win_work_offsets[wi];
                                const value_type* local =
                                    &win_vals[static_cast<std::size_t>(wi) * 4];

                                if (wi != last_wi) {
                                    // Incremental walk avoids find()'s redundant rescan. See
                                    // docs/research/sisldo_ops.rst:sisldo_forward.block4_incremental_walk_perf.
                                    bc_cursor =
                                        weights.block4.row_cursor(static_cast<std::size_t>(br));
                                    elem_pos = BL4.elem_start[br];
                                    byte_pos = weights.block4.tile_byte_start[br];
                                    for (SIZE_TYPE s = 0; s < tile_offset; ++s) {
                                        bc_cursor.advance();
                                        byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
                                        ++elem_pos;
                                    }
                                    last_wi = wi;
                                } else {
                                    byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
                                    ++elem_pos;
                                }
                                const uint32_t bc = bc_cursor.advance();

                                // Read-only lookup, does not mark the handle dirty.
                                const auto tile = weights.block4.at_index(static_cast<uint32_t>(br),
                                                                          bc, elem_pos, byte_pos);
                                const uint8_t* tdata = tile.raw_data();

                                // Window-level zero-skip -- see
                                // docs/research/sisldo_ops.rst:sisldo_forward.block4_zero_skip.
                                if (local[0] == value_type(0) && local[1] == value_type(0) &&
                                    local[2] == value_type(0) && local[3] == value_type(0)) {
                                    continue;
                                }

                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    const std::size_t col =
                                        static_cast<std::size_t>(bc) * BLOCK4_TILE + lj;
                                    if (col >= out_cols)
                                        continue;

                                    value_type acc = value_type(0);
                                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                                        const std::size_t row =
                                            static_cast<std::size_t>(br) * BLOCK4_TILE + li;
                                        if (row >= num_inputs)
                                            continue;
                                        // Per-li skip: zeroed input needs no decode.
                                        if (local[li] == value_type(0))
                                            continue;
                                        const uint8_t byte = tdata[Block4Tile::slot_index(li, lj)];
                                        if (byte == 0)
                                            continue;
                                        const value_type w_decoded = FP4_TABLE[byte & 0xFu];
                                        // Rank-N scale.
                                        const value_type w_true =
                                            w_decoded * weights.get_scale(row, col);
                                        acc += w_true * local[li];
                                    }
                                    thread_output[batch_offset + col] += acc;
                                }
                            }
                        }
                    }
#pragma omp barrier
                }
            }

            // Sum EVERY thread's private slice, not just thread 0's.
            for (int t = 0; t < num_cpus; ++t) {
                const value_type* s =
                    all_b4_outputs.data() + static_cast<std::size_t>(t) * num_outputs;
                for (std::size_t i = 0; i < num_outputs; ++i)
                    output[i] += s[i];
            }
        }
    }

    // ── AQRS additive branch ─────────────────────────────────────────────────
    // See docs/research/sisldo_ops.rst:sisldo_forward.additive_branch_port.
    if (weights.additive_rank > 0) {
        std::vector<value_type> proj(
            static_cast<std::size_t>(input_tensor.rows) * weights.additive_rank, value_type(0));
        for (SIZE_TYPE b = 0; b < input_tensor.rows; ++b) {
            const SIZE_TYPE batch_start = (*input_tensor.ptrs[0])[b];
            const SIZE_TYPE batch_end = (*input_tensor.ptrs[0])[b + 1];
            value_type* p_row = proj.data() + static_cast<std::size_t>(b) * weights.additive_rank;
            for (SIZE_TYPE i = batch_start; i < batch_end; ++i) {
                const SIZE_TYPE r = (*input_tensor.indices[0])[i];
                const value_type iv = (*input_tensor.values[0])[i];
                if (iv == value_type(0))
                    continue;
                for (std::size_t k = 0; k < weights.additive_rank; ++k)
                    p_row[k] += weights.get_additive_u_k(static_cast<std::size_t>(r), k) * iv;
            }
        }
        for (SIZE_TYPE b = 0; b < input_tensor.rows; ++b) {
            const value_type* p_row =
                proj.data() + static_cast<std::size_t>(b) * weights.additive_rank;
            value_type* out_row = output + static_cast<std::size_t>(b) * out_cols;
            for (std::size_t c = 0; c < out_cols; ++c) {
                value_type acc = value_type(0);
                for (std::size_t k = 0; k < weights.additive_rank; ++k)
                    acc +=
                        weights.get_additive_gamma_k(k) * weights.get_additive_v_k(c, k) * p_row[k];
                out_row[c] += acc;
            }
        }
    }
}

// delta_csr_backward (sparse input + sparse gradient) removed here -- see
// docs/research/sisldo_ops.rst:sisldo_ops.delta_csr_backward_removed.

// ── Backward pass (dense input, sparse gradient) ─────────────────────────────
// See docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.dense_input_rationale.

// Template-parameter parity with disldo_backward (task #100); also fixes a
// true-weight-space vs. code-space scale bug. See
// docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.template_parity_and_scale_space_bug.
template <
    typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t,
    typename ScalePolicy = RMSpropScalePolicy<typename ValueAccessor<VALUES_TYPE>::value_type>,
    bool StochasticRounding = true,
    template <typename> class SynapsePolicyT = BoundedRMSpropSynapsePolicy>
void disldo_backward_sparse_grad(
    const typename ValueAccessor<VALUES_TYPE>::value_type* input, // dense [batch, n_inputs]
    SIZE_TYPE batch, SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& out_grad_sparse,
    typename ValueAccessor<VALUES_TYPE>::value_type*
        input_gradients, // dense [batch, n_inputs], accumulated
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_input_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_grad_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type learning_rate = 0.01f, const int num_cpus = 4,
    bool lr_per_row_nnz = false, bool damp_by_importance = true,
    typename ValueAccessor<VALUES_TYPE>::value_type beta2 = 0.999f,
    typename ValueAccessor<VALUES_TYPE>::value_type eps = 1e-8f,
    typename ValueAccessor<VALUES_TYPE>::value_type min_decay_frac = 0.0f,
    typename ValueAccessor<VALUES_TYPE>::value_type max_abs_delta = 1e30f,
    typename ValueAccessor<VALUES_TYPE>::value_type max_ci = 1e30f, bool scale_invariant = false) {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    using SynapsePolicy = SynapsePolicyT<value_type>;
    auto& dc = weights.connections;
    const auto& L = dc.layout;
    const std::size_t n_inputs = L.rows;
    const std::size_t out_cols = L.cols;

    for (SIZE_TYPE b = 0; b < batch; ++b)
        for (std::size_t r = 0; r < n_inputs; ++r)
            neuron_input_accum[r] += std::abs(input[b * n_inputs + r]);
    for (SIZE_TYPE i = 0; i < out_grad_sparse.rows; ++i)
        for (SIZE_TYPE j = (*out_grad_sparse.ptrs[0])[i]; j < (*out_grad_sparse.ptrs[0])[i + 1];
             ++j)
            neuron_grad_accum[(*out_grad_sparse.indices[0])[j]] +=
                std::abs((*out_grad_sparse.values[0])[j]);

    // AQRS neurogenesis-trigger normalization fix -- see
    // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.dy_density_normalization_fix.
    const SIZE_TYPE dy_total_nnz =
        batch > 0 ? (*out_grad_sparse.ptrs[0])[batch] - (*out_grad_sparse.ptrs[0])[0]
                  : SIZE_TYPE(0);
    const value_type dy_density =
        (batch > 0 && out_cols > 0)
            ? std::max(static_cast<value_type>(dy_total_nnz) /
                           (static_cast<value_type>(batch) * static_cast<value_type>(out_cols)),
                       value_type(1e-6))
            : value_type(1);

    // ── AQRS rank-N scale scaffolding ─────────────────────────────────────────
    // See docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.rank_backfill_pattern.
    const std::size_t rank = weights.scale_rank;
    std::vector<value_type> t_col_grad(static_cast<std::size_t>(num_cpus) * out_cols * rank,
                                       value_type(0));
    std::vector<value_type> t_col_grad_contrib(static_cast<std::size_t>(num_cpus) * out_cols * rank,
                                               value_type(0));
    const bool output_scale_trainable = weights.output_scale_is_trainable;
    // AQRS gamma's own gradient (task #273/#283 parity) -- layer-wide, sized num_cpus*rank.
    std::vector<value_type> t_gamma_grad(static_cast<std::size_t>(num_cpus) * rank, value_type(0));
    std::vector<value_type> t_gamma_grad_contrib(static_cast<std::size_t>(num_cpus) * rank,
                                                 value_type(0));

    // Pre-size value_scale/output_scale -- backfill pattern, see
    // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.rank_backfill_pattern.
    if (weights.value_scale.size() < n_inputs * rank) {
        const std::size_t old_size = weights.value_scale.size();
        weights.value_scale.resize(n_inputs * rank, value_type(0));
        for (std::size_t idx = old_size; idx < weights.value_scale.size(); ++idx)
            if (idx % rank == 0)
                weights.value_scale[idx] = value_type(1);
    }
    if (weights.output_scale.size() < out_cols * rank) {
        const std::size_t old_size = weights.output_scale.size();
        weights.output_scale.resize(out_cols * rank, value_type(0));
        for (std::size_t idx = old_size; idx < weights.output_scale.size(); ++idx)
            if (idx % rank == 0)
                weights.output_scale[idx] = value_type(1);
    }
    if (weights.value_scale_importance.size() < n_inputs * rank)
        weights.value_scale_importance.resize(n_inputs * rank, value_type(0));
    if (weights.output_scale_importance.size() < out_cols * rank)
        weights.output_scale_importance.resize(out_cols * rank, value_type(0));
    if (weights.value_scale_step.size() < n_inputs * rank)
        weights.value_scale_step.resize(n_inputs * rank, 0);

    // dc.empty() does NOT mean "nothing to do" -- see sisldo_forward.dc_empty_block4_bug above.
    if (!dc.empty()) {
        // Importance stats accumulators across batches -- see
        // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.batch_outer_row_inner_layout.
        double total_sum_abs_new_i = 0.0, total_sum_abs_old_i = 0.0;
        double total_sum_sq_new_i = 0.0, total_sum_sq_old_i = 0.0;
        value_type total_max_new_i = value_type(0);

        // value_scale gradient accumulator layout differs from disldo_backward's
        // (batch-outer/row-inner here) -- see
        // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.batch_outer_row_inner_layout.
        std::vector<double> scale_grad_sums_rank(n_inputs * rank, 0.0);
        // Parallel forward-contribution accumulator, mirroring disldo_backward's own.
        std::vector<double> scale_grad_sums_rank_contrib(n_inputs * rank, 0.0);

        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const SIZE_TYPE og_start = (*out_grad_sparse.ptrs[0])[b];
            const SIZE_TYPE og_end = (*out_grad_sparse.ptrs[0])[b + 1];

            double batch_sum_abs_new_i = 0.0, batch_sum_abs_old_i = 0.0;
            double batch_sum_sq_new_i = 0.0, batch_sum_sq_old_i = 0.0;
            value_type batch_max_new_i = value_type(0);

#pragma omp parallel for num_threads(num_cpus) schedule(static)                                    \
    reduction(+ : batch_sum_abs_new_i, batch_sum_abs_old_i, batch_sum_sq_new_i,                    \
                  batch_sum_sq_old_i) reduction(max : batch_max_new_i)
            for (std::size_t r = 0; r < n_inputs; ++r) {
                const std::size_t nnz_this_row = L.row_nnz(r);
                if (nnz_this_row == 0)
                    continue;
                const value_type in_val = input[b * n_inputs + r];
                // lr/nnz_this_row keeps updates comparable across rows of different
                // connection counts.
                const value_type effective_lr =
                    lr_per_row_nnz ? learning_rate / static_cast<value_type>(nnz_this_row)
                                   : learning_rate;
                // value_scale's own scale_eff_lr is applied once after all batches (see below).

                auto cursor = dc.row_cursor(r);
                SIZE_TYPE og_ptr = og_start; // fresh per row -- each row does its own merge
                value_type dx_accum = value_type(0);
                const value_type imp_scale = weights.get_importance_scale(r);

                // Per-thread rank-N accumulator lambdas.
                const int tid = omp_get_thread_num();
                // cppcheck-suppress constVariablePointer -- false positive, mutated indirectly via
                // the m*_at lambdas below (see linear_disldo.hpp's mcol_base for the full
                // rationale)
                value_type* mcol_base =
                    t_col_grad.data() + static_cast<std::size_t>(tid) * out_cols * rank;
                auto mcol_at = [&](std::size_t col_, std::size_t k) -> value_type& {
                    return mcol_base[col_ * rank + k];
                };
                // cppcheck-suppress constVariablePointer
                value_type* mcol_contrib_base =
                    t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * out_cols * rank;
                auto mcol_at_contrib = [&](std::size_t col_, std::size_t k) -> value_type& {
                    return mcol_contrib_base[col_ * rank + k];
                };
                // cppcheck-suppress constVariablePointer
                value_type* mgamma_base =
                    t_gamma_grad.data() + static_cast<std::size_t>(tid) * rank;
                auto mgamma_at = [&](std::size_t k) -> value_type& { return mgamma_base[k]; };
                // cppcheck-suppress constVariablePointer
                value_type* mgamma_contrib_base =
                    t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * rank;
                auto mgamma_at_contrib = [&](std::size_t k) -> value_type& {
                    return mgamma_contrib_base[k];
                };

                for (std::size_t e = 0; e < nnz_this_row; ++e) {
                    const COL_TYPE col = cursor.advance();
                    const std::size_t vb = L.elem_start[r] + e;
                    const value_type cw_orig = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                    // S: real per-synapse scale, rank-N. See
                    // template_parity_and_scale_space_bug above.
                    const value_type S = weights.get_scale(r, col);
                    const value_type w = cw_orig * S; // -> true units, for dx only

                    // Merge-advance, O(nnz_this_row + grad_nnz) per row.
                    while (og_ptr < og_end &&
                           (*out_grad_sparse.indices[0])[og_ptr] < static_cast<SIZE_TYPE>(col))
                        ++og_ptr;
                    if (og_ptr >= og_end ||
                        (*out_grad_sparse.indices[0])[og_ptr] != static_cast<SIZE_TYPE>(col))
                        continue; // this output has no significant gradient this pass -- skip

                    const value_type dy_val = (*out_grad_sparse.values[0])[og_ptr];
                    dx_accum += w * dy_val; // weight-only -- reaches this row regardless of in_val

                    if (learning_rate != value_type(0)) {
                        const value_type out_scale = weights.get_output_scale(col);
                        const value_type out_imp_scale = weights.get_output_importance_scale(col);
                        const value_type combined_imp_scale = imp_scale * out_imp_scale;
                        const value_type grad = dy_val * in_val; // scales with true input value
                        const value_type ci_orig =
                            ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                        value_type ci = ci_orig * combined_imp_scale; // -> true units
                        // Additive contrib combination -- see
                        // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.merge_scan_design.
                        const value_type contrib = in_val * w;
                        ci = SynapsePolicy::update_ci(ci, grad, contrib, beta2, min_decay_frac,
                                                      max_ci);
                        value_type quant =
                            cw_orig; // code-space accumulator, matches disldo_backward
                        quant += SynapsePolicy::update_cw(grad, ci, S, effective_lr, eps,
                                                          damp_by_importance, max_abs_delta,
                                                          scale_invariant);
                        if constexpr (StochasticRounding) {
                            ValueAccessor<VALUES_TYPE>::set_stochastic_live(
                                dc.values, vb, quant, ci / combined_imp_scale);
                        } else {
                            ValueAccessor<VALUES_TYPE>::set_live(dc.values, vb, quant,
                                                                 ci / combined_imp_scale);
                        }
                        const value_type actual_imp =
                            ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                        batch_sum_abs_new_i += std::abs(static_cast<double>(actual_imp));
                        batch_sum_abs_old_i += std::abs(static_cast<double>(ci_orig));
                        batch_sum_sq_new_i += static_cast<double>(actual_imp) * actual_imp;
                        batch_sum_sq_old_i += static_cast<double>(ci_orig) * ci_orig;
                        batch_max_new_i = std::max(batch_max_new_i, std::abs(actual_imp));

                        // dL/d(value_scale_k) -- see merge_scan_design in
                        // docs/research/sisldo_ops.rst.
                        const value_type quant_floor =
                            (cw_orig == value_type(0)) ? value_type(0.1f) : cw_orig;
                        for (std::size_t k = 0; k < rank; ++k) {
                            const value_type out_scale_k = weights.get_output_scale_k(col, k);
                            const value_type val_scale_k = weights.get_value_scale_k(r, k);
                            const value_type gamma_k = weights.get_scale_gamma_k(k);
                            scale_grad_sums_rank[r * rank + k] += static_cast<double>(quant_floor) *
                                                                  static_cast<double>(out_scale_k) *
                                                                  static_cast<double>(gamma_k) *
                                                                  grad;
                            mcol_at(col, k) += quant_floor * val_scale_k * gamma_k * grad;
                            mgamma_at(k) += quant_floor * val_scale_k * out_scale_k * grad;
                            scale_grad_sums_rank_contrib[r * rank + k] +=
                                static_cast<double>(quant_floor) *
                                static_cast<double>(out_scale_k) * static_cast<double>(gamma_k) *
                                static_cast<double>(contrib);
                            mcol_at_contrib(col, k) +=
                                quant_floor * val_scale_k * gamma_k * contrib;
                            mgamma_at_contrib(k) +=
                                quant_floor * val_scale_k * out_scale_k * contrib;
                        }
                    }
                }
                input_gradients[b * n_inputs + r] += dx_accum;
            }

            total_sum_abs_new_i += batch_sum_abs_new_i;
            total_sum_abs_old_i += batch_sum_abs_old_i;
            total_sum_sq_new_i += batch_sum_sq_new_i;
            total_sum_sq_old_i += batch_sum_sq_old_i;
            total_max_new_i = std::max(total_max_new_i, batch_max_new_i);
        }

        if (learning_rate != value_type(0)) {
            weights.update_importance_stats_aggregate(total_sum_abs_new_i, total_sum_abs_old_i,
                                                      total_sum_sq_new_i, total_sum_sq_old_i,
                                                      total_max_new_i);

            // Apply value_scale gradient once per (row,k), after ALL batches.
            for (std::size_t r = 0; r < n_inputs; ++r) {
                const std::size_t nnz_this_row = L.row_nnz(r);
                if (nnz_this_row == 0)
                    continue;
                const value_type scale_eff_lr =
                    learning_rate / static_cast<value_type>(nnz_this_row);
                for (std::size_t k = 0; k < rank; ++k) {
                    if (scale_grad_sums_rank[r * rank + k] == 0.0 &&
                        scale_grad_sums_rank_contrib[r * rank + k] == 0.0)
                        continue;
                    const value_type g_agg =
                        static_cast<value_type>(scale_grad_sums_rank[r * rank + k]);
                    const value_type contrib_agg =
                        static_cast<value_type>(scale_grad_sums_rank_contrib[r * rank + k]);
                    ScalePolicy::update(weights.value_scale[r * rank + k],
                                        weights.value_scale_importance[r * rank + k], g_agg,
                                        scale_eff_lr, beta2, eps, contrib_agg,
                                        &weights.get_value_scale_step_k(r, k), scale_invariant);
                }
            }
        }
    } // closes if (!dc.empty())

    // ── block4 contribution ─────────────────────────────────────────────────
    // See docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.block4_backward_design.
    if constexpr (std::is_same_v<VALUES_TYPE, FP4BiPacked>) {
        if (weights.block4.n_tiles() > 0) {
            const auto& BL4 = weights.block4.block_layout;
            const std::size_t tiles_r = BL4.rows;

            // Safe to write directly (row-exclusive br ownership, see
            // block4_backward_design above). Rank-N (task #331), sized
            // n_inputs*rank, indexed [row*rank+k].
            std::vector<double> row_scale_grad_sums_rank(n_inputs * rank, 0.0);
            // Parallel forward-contribution accumulator (see scattered path above).
            std::vector<double> row_scale_grad_sums_rank_contrib(n_inputs * rank, 0.0);

            double b4_total_sum_abs_new = 0.0, b4_total_sum_abs_old = 0.0;
            double b4_total_sum_sq_new = 0.0, b4_total_sum_sq_old = 0.0;
            value_type b4_total_max_new = value_type(0);

            for (SIZE_TYPE b = 0; b < batch; ++b) {
                const SIZE_TYPE og_start = (*out_grad_sparse.ptrs[0])[b];
                const SIZE_TYPE og_end = (*out_grad_sparse.ptrs[0])[b + 1];

                double batch_sum_abs_new = 0.0, batch_sum_abs_old = 0.0;
                double batch_sum_sq_new = 0.0, batch_sum_sq_old = 0.0;
                value_type batch_max_new = value_type(0);

#pragma omp parallel for num_threads(num_cpus) schedule(static)                                    \
    reduction(+ : batch_sum_abs_new, batch_sum_abs_old, batch_sum_sq_new, batch_sum_sq_old)        \
    reduction(max : batch_max_new)
                for (std::size_t br = 0; br < tiles_r; ++br) {
                    const std::size_t row_nnz_b4 = BL4.row_nnz(br);
                    if (row_nnz_b4 == 0)
                        continue;
                    // Total live slots across ALL of this row's tiles this call.
                    const std::size_t nnz_row = row_nnz_b4 * BLOCK4_TILE;

                    value_type dx_accum[BLOCK4_TILE] = {0, 0, 0, 0};

                    // Per-thread rank-N accumulator lambdas (tid stable across this br's work).
                    const int tid = omp_get_thread_num();
                    // cppcheck-suppress constVariablePointer
                    value_type* mcol_base =
                        t_col_grad.data() + static_cast<std::size_t>(tid) * out_cols * rank;
                    auto mcol_at = [&](std::size_t col_, std::size_t k) -> value_type& {
                        return mcol_base[col_ * rank + k];
                    };
                    // cppcheck-suppress constVariablePointer
                    value_type* mcol_contrib_base =
                        t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * out_cols * rank;
                    auto mcol_at_contrib = [&](std::size_t col_, std::size_t k) -> value_type& {
                        return mcol_contrib_base[col_ * rank + k];
                    };
                    // cppcheck-suppress constVariablePointer
                    value_type* mgamma_base =
                        t_gamma_grad.data() + static_cast<std::size_t>(tid) * rank;
                    auto mgamma_at = [&](std::size_t k) -> value_type& { return mgamma_base[k]; };
                    // cppcheck-suppress constVariablePointer
                    value_type* mgamma_contrib_base =
                        t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * rank;
                    auto mgamma_at_contrib = [&](std::size_t k) -> value_type& {
                        return mgamma_contrib_base[k];
                    };

                    if (learning_rate == value_type(0)) {
                        // Read-only path, no concurrency hazard. See
                        // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.block4_workspace_concurrency.
                        auto bc_cursor = weights.block4.row_cursor(br);
                        std::size_t elem_pos = BL4.elem_start[br];
                        std::size_t byte_pos = weights.block4.tile_byte_start[br];
                        SIZE_TYPE og_ptr = og_start;

                        for (std::size_t e = 0; e < row_nnz_b4; ++e) {
                            const uint32_t bc = bc_cursor.advance();
                            const std::size_t window_lo =
                                static_cast<std::size_t>(bc) * BLOCK4_TILE;
                            const std::size_t window_hi = window_lo + BLOCK4_TILE;

                            while (og_ptr < og_end &&
                                   static_cast<std::size_t>((*out_grad_sparse.indices[0])[og_ptr]) <
                                       window_lo)
                                ++og_ptr;

                            value_type dy_local[BLOCK4_TILE] = {0, 0, 0, 0};
                            bool any = false;
                            for (SIZE_TYPE p = og_ptr;
                                 p < og_end && static_cast<std::size_t>(
                                                   (*out_grad_sparse.indices[0])[p]) < window_hi;
                                 ++p) {
                                const std::size_t col =
                                    static_cast<std::size_t>((*out_grad_sparse.indices[0])[p]);
                                dy_local[col - window_lo] = (*out_grad_sparse.values[0])[p];
                                any = true;
                            }

                            if (any) {
                                const auto tile = weights.block4.at_index(static_cast<uint32_t>(br),
                                                                          bc, elem_pos, byte_pos);
                                const uint8_t* tdata = tile.raw_data();
                                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                                    const std::size_t row = br * BLOCK4_TILE + li;
                                    if (row >= n_inputs)
                                        continue;
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (dy_local[lj] == value_type(0))
                                            continue;
                                        const std::size_t col = window_lo + lj;
                                        if (col >= out_cols)
                                            continue;
                                        const uint8_t byte = tdata[Block4Tile::slot_index(li, lj)];
                                        const value_type w_decoded = FP4_TABLE[byte & 0xFu];
                                        // Rank-N, matches the scattered read-only path.
                                        const value_type S = weights.get_scale(row, col);
                                        const value_type w = w_decoded * S;
                                        dx_accum[li] += w * dy_local[lj];
                                    }
                                }
                            }
                            byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
                            ++elem_pos;
                        }
                    } else {
                        // Writing path -- row-local workspace, ASan-confirmed hazard. See
                        // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.block4_workspace_concurrency.
                        auto ws = weights.block4.snapshot_row(br);
                        SIZE_TYPE og_ptr = og_start;
                        std::size_t local_pos = 0;

                        for (std::size_t e = 0; e < row_nnz_b4; ++e) {
                            const uint32_t bc = ws.bc[e];
                            const std::size_t window_lo =
                                static_cast<std::size_t>(bc) * BLOCK4_TILE;
                            const std::size_t window_hi = window_lo + BLOCK4_TILE;

                            while (og_ptr < og_end &&
                                   static_cast<std::size_t>((*out_grad_sparse.indices[0])[og_ptr]) <
                                       window_lo)
                                ++og_ptr;

                            value_type dy_local[BLOCK4_TILE] = {0, 0, 0, 0};
                            bool any = false;
                            for (SIZE_TYPE p = og_ptr;
                                 p < og_end && static_cast<std::size_t>(
                                                   (*out_grad_sparse.indices[0])[p]) < window_hi;
                                 ++p) {
                                const std::size_t col =
                                    static_cast<std::size_t>((*out_grad_sparse.indices[0])[p]);
                                dy_local[col - window_lo] = (*out_grad_sparse.values[0])[p];
                                any = true;
                            }

                            const std::size_t this_local_pos = local_pos;
                            if (any) {
                                uint8_t scratch[BLOCK4_TILE_SLOTS];
                                weights.block4.unpack_workspace_tile(ws, e, this_local_pos,
                                                                     scratch);
                                bool dirty = false;

                                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                                    const std::size_t row = br * BLOCK4_TILE + li;
                                    if (row >= n_inputs)
                                        continue;
                                    const value_type in_val =
                                        input[static_cast<std::size_t>(b) * n_inputs + row];
                                    const value_type imp_scale = weights.get_importance_scale(row);
                                    const value_type effective_lr =
                                        lr_per_row_nnz
                                            ? learning_rate / static_cast<value_type>(nnz_row)
                                            : learning_rate;
                                    // value_scale's own scale_eff_lr is applied
                                    // once after all batches now, not folded in
                                    // per-synapse -- see the final application
                                    // loop below (recomputed there from nnz_row).

                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (dy_local[lj] == value_type(0))
                                            continue;
                                        const std::size_t col = window_lo + lj;
                                        if (col >= out_cols)
                                            continue;

                                        const uint8_t byte =
                                            scratch[Block4Tile::slot_index(li, lj)];
                                        const value_type w_decoded = FP4_TABLE[byte & 0xFu];
                                        // Rank-N (task #331): S = weights.get_scale(row, col),
                                        // matching the scattered write path's identical swap --
                                        // threading S through SynapsePolicy::update_cw (code-space)
                                        // instead of the old true-weight-space `new_w /
                                        // combined_scale` store fixes the ~1/S^2 bug here too.
                                        const value_type S = weights.get_scale(row, col);
                                        const value_type w = w_decoded * S;
                                        const value_type dy_val = dy_local[lj];
                                        dx_accum[li] += w * dy_val;

                                        const value_type out_imp_scale =
                                            weights.get_output_importance_scale(col);
                                        const value_type combined_imp_scale =
                                            imp_scale * out_imp_scale;
                                        const value_type imp_decoded =
                                            FP4_TABLE[(byte >> 4) & 0xFu];
                                        const value_type grad = dy_val * in_val;
                                        value_type ci = imp_decoded * combined_imp_scale;
                                        // Additive contrib combination, matching the scattered path
                                        // above.
                                        const value_type contrib = in_val * w;
                                        ci = SynapsePolicy::update_ci(ci, grad, contrib, beta2,
                                                                      min_decay_frac, max_ci);
                                        value_type quant = w_decoded; // code-space accumulator,
                                                                      // matches disldo_backward
                                        quant += SynapsePolicy::update_cw(
                                            grad, ci, S, effective_lr, eps, damp_by_importance,
                                            max_abs_delta, scale_invariant);
                                        // was_live gate -- see
                                        // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.block4_workspace_concurrency.
                                        const bool was_live = (byte != 0);
                                        const value_type imp_ratio = ci / combined_imp_scale;
                                        uint8_t new_w_code, new_imp_code;
                                        if constexpr (StochasticRounding) {
                                            new_w_code = was_live
                                                             ? fp4_quantize_stochastic_live(quant)
                                                             : fp4_quantize_stochastic(quant);
                                            new_imp_code =
                                                was_live
                                                    ? fp4_quantize_stochastic_live_nonneg(imp_ratio)
                                                    : fp4_quantize_stochastic(imp_ratio);
                                        } else {
                                            new_w_code = was_live ? fp4_quantize_live(quant)
                                                                  : fp4_quantize(quant);
                                            new_imp_code = was_live ? fp4_quantize_live(imp_ratio)
                                                                    : fp4_quantize(imp_ratio);
                                        }
                                        scratch[Block4Tile::slot_index(li, lj)] =
                                            uint8_t((new_imp_code << 4) | new_w_code);
                                        dirty = true;

                                        const value_type actual_imp =
                                            FP4_TABLE[new_imp_code] * combined_imp_scale;
                                        const value_type stored_imp =
                                            imp_decoded * combined_imp_scale;
                                        batch_sum_abs_new +=
                                            std::abs(static_cast<double>(actual_imp));
                                        batch_sum_abs_old +=
                                            std::abs(static_cast<double>(stored_imp));
                                        batch_sum_sq_new +=
                                            static_cast<double>(actual_imp) * actual_imp;
                                        batch_sum_sq_old +=
                                            static_cast<double>(stored_imp) * stored_imp;
                                        batch_max_new =
                                            std::max(batch_max_new, std::abs(actual_imp));

                                        // Rank-N (task #331): per-k gradient accumulation, direct
                                        // port of the scattered write path above. Race-free: this
                                        // br exclusively owns `row` for the whole function.
                                        const value_type quant_floor = (w_decoded == value_type(0))
                                                                           ? value_type(0.1f)
                                                                           : w_decoded;
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            const value_type out_scale_k =
                                                weights.get_output_scale_k(col, k);
                                            const value_type val_scale_k =
                                                weights.get_value_scale_k(row, k);
                                            const value_type gamma_k = weights.get_scale_gamma_k(k);
                                            row_scale_grad_sums_rank[row * rank + k] +=
                                                static_cast<double>(quant_floor) *
                                                static_cast<double>(out_scale_k) *
                                                static_cast<double>(gamma_k) * grad;
                                            mcol_at(col, k) +=
                                                quant_floor * val_scale_k * gamma_k * grad;
                                            mgamma_at(k) +=
                                                quant_floor * val_scale_k * out_scale_k * grad;
                                            row_scale_grad_sums_rank_contrib[row * rank + k] +=
                                                static_cast<double>(quant_floor) *
                                                static_cast<double>(out_scale_k) *
                                                static_cast<double>(gamma_k) *
                                                static_cast<double>(contrib);
                                            mcol_at_contrib(col, k) +=
                                                quant_floor * val_scale_k * gamma_k * contrib;
                                            mgamma_at_contrib(k) +=
                                                quant_floor * val_scale_k * out_scale_k * contrib;
                                        }
                                    }
                                }
                                if (dirty)
                                    weights.block4.commit_dirty_tile_in_workspace(
                                        ws, e, this_local_pos, scratch);
                            }
                            local_pos +=
                                block4_stored_tile_len(ws.is_sparse[e], &ws.bytes[this_local_pos]);
                        } // tiles in this row

                        // Merge back -- see
                        // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.block4_workspace_concurrency.
                        weights.block4.merge_row_workspace(
                            br, ws,
                            [&](std::size_t ev_row, std::size_t ev_col,
                                uint8_t ev_imp_code) -> double {
                                const value_type imp_scale = weights.get_importance_scale(ev_row);
                                const value_type out_imp_scale =
                                    weights.get_output_importance_scale(ev_col);
                                return static_cast<double>(FP4_TABLE[ev_imp_code & 0xFu]) *
                                       static_cast<double>(imp_scale) *
                                       static_cast<double>(out_imp_scale);
                            });
                    }

                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = br * BLOCK4_TILE + li;
                        if (row >= n_inputs)
                            continue;
                        input_gradients[static_cast<std::size_t>(b) * n_inputs + row] +=
                            dx_accum[li];
                    }
                } // br

                b4_total_sum_abs_new += batch_sum_abs_new;
                b4_total_sum_abs_old += batch_sum_abs_old;
                b4_total_sum_sq_new += batch_sum_sq_new;
                b4_total_sum_sq_old += batch_sum_sq_old;
                b4_total_max_new = std::max(b4_total_max_new, batch_max_new);
            } // batch

            if (learning_rate != value_type(0)) {
                weights.update_importance_stats_aggregate(b4_total_sum_abs_new,
                                                          b4_total_sum_abs_old, b4_total_sum_sq_new,
                                                          b4_total_sum_sq_old, b4_total_max_new);

                for (std::size_t row = 0; row < n_inputs; ++row) {
                    // nnz_row for this row's block-row (same derivation as above).
                    const std::size_t br = row / BLOCK4_TILE;
                    const std::size_t nnz_row = (br < BL4.rows ? BL4.row_nnz(br) : 0) * BLOCK4_TILE;
                    if (nnz_row == 0)
                        continue;
                    const value_type scale_eff_lr =
                        learning_rate / static_cast<value_type>(nnz_row);
                    // Rank-N (task #331): per-(row,k) apply, direct port of the scattered path
                    // above.
                    for (std::size_t k = 0; k < rank; ++k) {
                        if (row_scale_grad_sums_rank[row * rank + k] == 0.0 &&
                            row_scale_grad_sums_rank_contrib[row * rank + k] == 0.0)
                            continue;
                        const value_type g_agg =
                            static_cast<value_type>(row_scale_grad_sums_rank[row * rank + k]);
                        const value_type contrib_agg = static_cast<value_type>(
                            row_scale_grad_sums_rank_contrib[row * rank + k]);
                        ScalePolicy::update(weights.value_scale[row * rank + k],
                                            weights.value_scale_importance[row * rank + k], g_agg,
                                            scale_eff_lr, beta2, eps, contrib_agg,
                                            &weights.get_value_scale_step_k(row, k),
                                            scale_invariant);
                    }
                }
            }
        }
    }

    // ── output_scale's own gradient reduction ─────────────────────────────────
    // See
    // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.output_scale_and_gamma_reduction.
    if (learning_rate != value_type(0) && output_scale_trainable) {
        for (std::size_t c = 0; c < out_cols; ++c) {
            const std::size_t deg =
                c < weights.out_degree.size() ? static_cast<std::size_t>(weights.out_degree[c]) : 0;
            if (deg == 0)
                continue;
            const value_type col_eff_lr = learning_rate / static_cast<value_type>(deg);
            for (std::size_t k = 0; k < rank; ++k) {
                double col_grad_sum = 0.0, col_grad_sum_contrib = 0.0;
                for (int t = 0; t < num_cpus; ++t) {
                    col_grad_sum +=
                        t_col_grad[static_cast<std::size_t>(t) * out_cols * rank + c * rank + k];
                    col_grad_sum_contrib +=
                        t_col_grad_contrib[static_cast<std::size_t>(t) * out_cols * rank +
                                           c * rank + k];
                }
                const value_type g_agg = static_cast<value_type>(col_grad_sum);
                const value_type contrib_agg = static_cast<value_type>(col_grad_sum_contrib);
                ScalePolicy::update(weights.output_scale[c * rank + k],
                                    weights.output_scale_importance[c * rank + k], g_agg,
                                    col_eff_lr, beta2, eps, contrib_agg,
                                    &weights.get_output_scale_step_k(c, k), scale_invariant);
            }
        }
    }

    // ── AQRS scale_gamma's own update ──────────────────────────────────────────
    // See
    // docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.output_scale_and_gamma_reduction.
    if (learning_rate != value_type(0) && weights.scale_gamma_is_trainable) {
        std::vector<value_type> g_agg_by_k(rank);
        for (std::size_t k = 0; k < rank; ++k) {
            double gamma_grad_sum = 0.0, gamma_grad_sum_contrib = 0.0;
            for (int t = 0; t < num_cpus; ++t) {
                gamma_grad_sum += t_gamma_grad[static_cast<std::size_t>(t) * rank + k];
                gamma_grad_sum_contrib +=
                    t_gamma_grad_contrib[static_cast<std::size_t>(t) * rank + k];
            }
            const value_type g_agg = static_cast<value_type>(gamma_grad_sum);
            const value_type contrib_agg = static_cast<value_type>(gamma_grad_sum_contrib);
            g_agg_by_k[k] = g_agg;
            weights.set_scale_gamma_raw_k(k, weights.get_scale_gamma_k(k));
            ScalePolicy::update(weights.scale_gamma[k], weights.get_scale_gamma_state_k(k), g_agg,
                                learning_rate, beta2, eps, contrib_agg,
                                &weights.get_scale_gamma_step_k(k), false);
        }
        value_type gamma_l1_sum = value_type(0);
        for (std::size_t k = 0; k < rank; ++k)
            gamma_l1_sum += std::fabs(weights.scale_gamma[k]);
        const value_type grad_norm_divisor =
            static_cast<value_type>(n_inputs) * static_cast<value_type>(out_cols) * dy_density;
        for (std::size_t k = 0; k < rank; ++k) {
            const value_type abs_gamma_k = std::fabs(weights.scale_gamma[k]);
            const value_type share_k =
                gamma_l1_sum > value_type(0) ? abs_gamma_k / gamma_l1_sum : value_type(0);
            weights.update_scale_gamma_ema_k(k, abs_gamma_k, share_k,
                                             std::fabs(g_agg_by_k[k]) / grad_norm_divisor);
        }
    }

    // ── AQRS additive branch backward ──────────────────────────────────────────
    // See docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.additive_branch_backward.
    if (weights.additive_rank > 0) {
        const std::size_t r_o = weights.additive_rank;
        std::vector<value_type> P(static_cast<std::size_t>(batch) * r_o, value_type(0));
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* in_row = input + static_cast<std::size_t>(b) * n_inputs;
            value_type* p_row = P.data() + static_cast<std::size_t>(b) * r_o;
            for (std::size_t r = 0; r < n_inputs; ++r) {
                const value_type iv = in_row[r];
                if (iv == value_type(0))
                    continue;
                for (std::size_t k = 0; k < r_o; ++k)
                    p_row[k] += weights.get_additive_u_k(r, k) * iv;
            }
        }
        // dP[b,k] = sum_c V(c,k)*dy[b,c] -- CSR walk over out_grad_sparse.
        std::vector<value_type> dP(static_cast<std::size_t>(batch) * r_o, value_type(0));
        // dV_accum[c,k] = sum_b dy[b,c]*P[b,k] -- built in the SAME CSR walk as dP.
        std::vector<value_type> dV_accum(out_cols * r_o, value_type(0));
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const SIZE_TYPE og_start = (*out_grad_sparse.ptrs[0])[b];
            const SIZE_TYPE og_end = (*out_grad_sparse.ptrs[0])[b + 1];
            value_type* dp_row = dP.data() + static_cast<std::size_t>(b) * r_o;
            const value_type* p_row = P.data() + static_cast<std::size_t>(b) * r_o;
            for (SIZE_TYPE i = og_start; i < og_end; ++i) {
                const std::size_t c = static_cast<std::size_t>((*out_grad_sparse.indices[0])[i]);
                const value_type dy = (*out_grad_sparse.values[0])[i];
                for (std::size_t k = 0; k < r_o; ++k) {
                    dp_row[k] += weights.get_additive_v_k(c, k) * dy;
                    dV_accum[c * r_o + k] += dy * p_row[k];
                }
            }
        }
        // dX = sum_k U(r,k) * gamma_k * dP_raw[b,k] -- direct port.
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* dp_row = dP.data() + static_cast<std::size_t>(b) * r_o;
            value_type* dx_row = input_gradients + static_cast<std::size_t>(b) * n_inputs;
            for (std::size_t r = 0; r < n_inputs; ++r) {
                value_type acc = value_type(0);
                for (std::size_t k = 0; k < r_o; ++k)
                    acc += weights.get_additive_gamma_k(k) * weights.get_additive_u_k(r, k) *
                           dp_row[k];
                dx_row[r] += acc;
            }
        }
        if (learning_rate != value_type(0)) {
            // dU[r,k] = gamma_k * sum_b dP_raw[b,k]*X[b,r] -- direct port.
            for (std::size_t r = 0; r < n_inputs; ++r) {
                for (std::size_t k = 0; k < r_o; ++k) {
                    value_type dU_rk = value_type(0);
                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        const value_type iv = input[static_cast<std::size_t>(b) * n_inputs + r];
                        if (iv == value_type(0))
                            continue;
                        dU_rk += dP[static_cast<std::size_t>(b) * r_o + k] * iv;
                    }
                    dU_rk *= weights.get_additive_gamma_k(k);
                    if (dU_rk == value_type(0))
                        continue;
                    value_type u_val = weights.get_additive_u_k(r, k);
                    AdamScalePolicy<value_type>::update(
                        u_val, weights.get_additive_u_state_k(r, k),
                        weights.get_additive_u_momentum_k(r, k), dU_rk, learning_rate,
                        value_type(0.9f), beta2, eps, &weights.get_additive_u_step_k(r, k));
                    weights.set_additive_u_raw_k(r, k, u_val);
                }
            }
            // dV[c,k] = gamma_k * dV_accum[c,k] -- already reduced above via the CSR walk.
            for (std::size_t c = 0; c < out_cols; ++c) {
                for (std::size_t k = 0; k < r_o; ++k) {
                    value_type dV_ck = dV_accum[c * r_o + k] * weights.get_additive_gamma_k(k);
                    if (dV_ck == value_type(0))
                        continue;
                    value_type v_val = weights.get_additive_v_k(c, k);
                    AdamScalePolicy<value_type>::update(
                        v_val, weights.get_additive_v_state_k(c, k),
                        weights.get_additive_v_momentum_k(c, k), dV_ck, learning_rate,
                        value_type(0.9f), beta2, eps, &weights.get_additive_v_step_k(c, k));
                    weights.set_additive_v_raw_k(c, k, v_val);
                }
            }

            // additive_gamma's own update -- direct port, see additive_branch_backward above.
            if (weights.additive_gamma_is_trainable) {
                std::vector<value_type> dgamma_by_k(r_o);
                for (std::size_t k = 0; k < r_o; ++k) {
                    double dgamma_sum = 0.0;
                    for (SIZE_TYPE b = 0; b < batch; ++b)
                        dgamma_sum +=
                            static_cast<double>(P[static_cast<std::size_t>(b) * r_o + k]) *
                            static_cast<double>(dP[static_cast<std::size_t>(b) * r_o + k]);
                    const value_type dgamma_k = static_cast<value_type>(dgamma_sum);
                    dgamma_by_k[k] = dgamma_k;
                    weights.set_additive_gamma_raw_k(k, weights.get_additive_gamma_k(k));
                    ScalePolicy::update(weights.additive_gamma[k],
                                        weights.get_additive_gamma_state_k(k), dgamma_k,
                                        learning_rate, beta2, eps, value_type(0),
                                        &weights.get_additive_gamma_step_k(k), false);
                }
                value_type gamma_l1_sum = value_type(0);
                for (std::size_t k = 0; k < r_o; ++k)
                    gamma_l1_sum += std::fabs(weights.additive_gamma[k]);
                const value_type grad_norm_divisor = static_cast<value_type>(n_inputs) *
                                                     static_cast<value_type>(out_cols) * dy_density;
                for (std::size_t k = 0; k < r_o; ++k) {
                    const value_type abs_gamma_k = std::fabs(weights.additive_gamma[k]);
                    const value_type share_k =
                        gamma_l1_sum > value_type(0) ? abs_gamma_k / gamma_l1_sum : value_type(0);
                    weights.update_additive_gamma_ema_k(
                        k, abs_gamma_k, share_k, std::fabs(dgamma_by_k[k]) / grad_norm_divisor);
                }
            }
        }
    }
}

#endif
