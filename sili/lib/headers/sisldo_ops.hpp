#ifndef __SISLDO_OPS_HPP_
#define __SISLDO_OPS_HPP_

// Split out of sparse_struct.hpp (see conversation). Whole-structure memory
// operations (compact/expand_headroom -- opposite operations, see their own
// docstrings) and the actual forward/backward computation: sisldo_forward
// (SISLDO -- sparse input) and disldo_backward_sparse_grad (dense input,
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
 * pattern (sisldo_forward/backward/build_probes/synap_row_step).
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
    // Propagate the INPUT dc's own hard limits through -- without this,
    // the freshly-constructed result below starts with DEFAULT
    // (unbounded) limits, so it can grow arbitrarily past whatever
    // budget the caller originally set via set_limits(), regardless of
    // how small n*(1+blank_fraction) is relative to it. A real, measured
    // bug (see delta_csr_from_absolute's own comment): nnz reached 127x
    // the intended max_weights budget in a synaptogenesis stress test
    // before this fix, since repeated expand_headroom() calls each
    // silently re-based the cap on current content instead of the
    // original budget.
    return delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
        ptrs, idx, w, imp, dc.layout.rows, dc.layout.cols,
        n * (1.0 + blank_fraction) * (uleb128_max_bytes<COL_TYPE>() + 1) + 4096,
        static_cast<std::size_t>(n * (1.0 + blank_fraction)) + 64,
        blank_fraction,
        dc.max_indices_bytes, dc.max_values_bytes);
}

// Like expand_headroom() but sizes the total budget for at least
// min_nnz_per_row connections per row. Use before synaptogenesis on a
// freshly loaded layer, then call equalizer_step() for each row to
// redistribute the budget evenly. After a full equalization pass each row
// has total_budget/rows = min_nnz_per_row elements of reserved headroom.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> expand_headroom_to(
    const DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc,
    std::size_t min_nnz_per_row,
    float blank_fraction = 0.2f)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    std::vector<SIZE_TYPE>  ptrs, idx;
    std::vector<value_type> w, imp;
    delta_csr_to_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(dc, ptrs, idx, w, imp);
    const std::size_t rows   = dc.layout.rows;
    const std::size_t n      = idx.size();
    const std::size_t budget = std::max(n, rows * min_nnz_per_row);
    // See expand_headroom()'s identical comment -- propagate the INPUT
    // dc's own hard limits through. If min_nnz_per_row*rows genuinely
    // exceeds the layer's original max_weights budget, this now
    // correctly throws std::bad_alloc instead of silently granting more
    // than was ever configured.
    return delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
        ptrs, idx, w, imp, rows, dc.layout.cols,
        budget * (1.0 + blank_fraction) * (uleb128_max_bytes<COL_TYPE>() + 1) + 4096,
        static_cast<std::size_t>(budget * (1.0 + blank_fraction)) + 64,
        blank_fraction,
        dc.max_indices_bytes, dc.max_values_bytes);
}
// ── Forward pass ─────────────────────────────────────────────────────────────

// No learning_rate parameter -- matches disldo_forward's own fix (see
// linear_disldo.hpp's disldo_forward docstring for the full rationale):
// this used to run a gradient-free ADSP-style (Activity-Dependent
// Structural Plasticity) importance update whenever a nonzero
// learning_rate was passed, unconditionally on whether a matching
// backward call would ever follow. Real footgun, confirmed via direct
// tracing on the DISLDO sibling -- REMOVED here too, not just disabled.
// Importance updates only ever happen in a backward pass now, coupled to
// a real gradient.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void sisldo_forward(
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& input_tensor,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    typename ValueAccessor<VALUES_TYPE>::value_type* output,
    const int    num_cpus = 4,
    typename ValueAccessor<VALUES_TYPE>::value_type* original_contributions_output = nullptr)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;

    // NOTE: dc.empty() (zero scattered nnz) does NOT mean "nothing to do" --
    // a layer can be entirely block4-resident (dc.empty() == true) and still
    // have real work below in the block4 phase. Historically this function
    // returned here unconditionally, which silently skipped the block4
    // phase too for any all-block4 layer (found via a benchmark reporting
    // an implausible, density-independent ~0.0001ms for the sparse path --
    // it was hitting this return before ever reaching block4 code). L's
    // rows/cols come from dc.layout's shape, which stays valid even when
    // nnz is 0 (set at construction, e.g. delta_csr_from_absolute), so it's
    // safe to keep using L below regardless of dc.empty().
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

    if (!dc.empty()) {
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
                    const value_type  wval_stored = ValueAccessor<VALUES_TYPE>::get_w(dc.values, wptr);
                    // Scale lookups per-synapse, not hoisted: in_idx (the row) varies
                    // within this loop (work-offset iteration, not a simple per-row
                    // loop) -- unlike disldo_forward/backward, can't fix it once per
                    // outer iteration.
                    // BUG FIX: out_scale/output_importance_scale (per-column, e.g. from
                    // FoldedLayer.from_descriptor's value_scale_mode="rank1") were never
                    // read here, unlike disldo_forward's identical row*col combination --
                    // a rank-1-quantized layer run through forward_sparse silently
                    // dropped its column scale entirely, reconstructing only stored_w *
                    // val_scale instead of the true value.
                    // Rank-N scale (see scale_rank's own docstring,
                    // delta_csr_types.hpp) -- reduces to the exact
                    // original val_scale*out_scale at scale_rank==1,
                    // matching disldo_forward's identical replacement.
                    const value_type  wval      = wval_stored * weights.get_scale(in_idx, out_idx);   // -> true units
                    const value_type  contrib   = wval * in_val;

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
    }

    for (std::size_t i = 0; i < num_outputs; ++i)
        output[i] += all_outputs[i];
    if (original_contributions_output)
        for (std::size_t i = 0; i < num_inputs; ++i)
            original_contributions_output[i] += all_contributions[i];

    // ── block4 contribution ─────────────────────────────────────────────────
    //
    // Real, previously-silent bug this closes: everything above only ever
    // touches weights.connections (the scattered CSR side) -- block4-
    // resident synapses (created automatically by ordinary synaptogenesis,
    // see block4_maybe_promote) were NEVER read here at all, so a layer
    // with any block4 tiles gave silently wrong output through
    // forward_sparse(). block4 is FP4-specific (see block4.hpp), hence the
    // if constexpr guard -- this whole section compiles to nothing for any
    // other VALUES_TYPE.
    //
    // Read-only, same as disldo_forward's own block4 loop
    // (linear_disldo.hpp): forward does NOT update per-synapse block4
    // weight/importance inline (that's a documented, pre-existing gap --
    // see disldo_forward's "KNOWN GAP" comment -- not something this
    // change is expected to newly fix), so no learning_rate handling is
    // needed here, only decode + multiply + accumulate.
    //
    // Design A (see TODO_DUAL_BLOCK4.md / conversation): reuses the exact
    // work_offsets/chunk/w_start/w_end shape the scattered pass above
    // already uses, one level up -- over ACTIVE windows (block-rows with
    // >=1 nonzero input AND >=1 live block4 tile) instead of over
    // individual scattered synapses. A window's real "work" is its block4
    // tile count (weights.block4.block_layout.row_nnz(br)), mirroring how
    // the scattered pre-pass above sizes work by L.row_nnz(in_idx).
    // Explicitly a first, measured-not-assumed choice -- see the
    // investigation this same commit's benchmark records: if the serial
    // gather pre-pass turns out to dominate at realistic densities,
    // Design B (direct per-active-window binary search, no pre-pass) is
    // the documented fallback, not a hypothetical.
    //
    // PRECONDITION: input_tensor's indices, within each batch row, must be
    // ascending (standard CSR convention) -- required for the gather to
    // find a window's up-to-4 entries via one contiguous scan instead of a
    // search. top_k()'s own output is sorted by magnitude, not index --
    // callers must run sort_indices() (parallel.hpp) first if their input
    // came from there. Not re-checked/enforced here (same convention the
    // scattered pass above already silently assumes).
    if constexpr (std::is_same_v<VALUES_TYPE, FP4BiPacked>) {
        if (weights.block4.n_tiles() > 0) {
            const auto& BL4 = weights.block4.block_layout;

            std::vector<value_type> all_b4_outputs(
                static_cast<std::size_t>(num_cpus) * num_outputs, value_type(0));

            // Per-batch scratch, reused across batches (not reallocated
            // per batch) -- same reasoning as block4.hpp's own persistent
            // scratch buffers: batch=1 real-time calls can't amortize
            // repeated heap allocation.
            std::vector<SIZE_TYPE>   win_br;           // active window's block-row index
            std::vector<value_type>  win_vals;         // flat, 4 per window: win_vals[4*w + li]
            std::vector<SIZE_TYPE>   win_work_offsets;  // cumulative block4 tile count per window

            #pragma omp parallel num_threads(num_cpus)
            {
                const int tid      = omp_get_thread_num();
                const int nthreads = omp_get_num_threads();
                value_type* thread_output =
                    all_b4_outputs.data() + static_cast<std::size_t>(tid) * num_outputs;

                for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
                    const SIZE_TYPE batch_start  = (*input_tensor.ptrs[0])[batch];
                    const SIZE_TYPE batch_end    = (*input_tensor.ptrs[0])[batch + 1];
                    const SIZE_TYPE batch_nnz    = batch_end - batch_start;
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
                            const SIZE_TYPE window_hi = window_lo + static_cast<SIZE_TYPE>(BLOCK4_TILE);

                            // Gather every entry in [i, batch_nnz) that
                            // still falls in this SAME window -- sorted
                            // input means these are exactly the entries
                            // contiguous from i (see PRECONDITION above),
                            // so this is a single forward scan, not a
                            // search.
                            value_type local[4] = {value_type(0), value_type(0),
                                                    value_type(0), value_type(0)};
                            SIZE_TYPE j = i;
                            while (j < batch_nnz) {
                                const SIZE_TYPE idxj = (*input_tensor.indices[0])[batch_start + j];
                                if (idxj >= window_hi) break;
                                local[idxj - window_lo] = (*input_tensor.values[0])[batch_start + j];
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
                                win_work_offsets.push_back(
                                    win_work_offsets.back() + static_cast<SIZE_TYPE>(row_nnz_b4));
                            }
                            i = j;
                        }
                    }

                    const SIZE_TYPE n_windows  = static_cast<SIZE_TYPE>(win_br.size());
                    const SIZE_TYPE total_work = win_work_offsets.back();

                    if (total_work > 0) {
                        const SIZE_TYPE chunk   = (total_work + nthreads - 1) / nthreads;
                        const SIZE_TYPE w_start = std::min(static_cast<SIZE_TYPE>(tid) * chunk, total_work);
                        const SIZE_TYPE w_end   = std::min(w_start + chunk, total_work);

                        if (w_start < w_end) {
                            SIZE_TYPE wi = static_cast<SIZE_TYPE>(
                                std::upper_bound(win_work_offsets.begin(), win_work_offsets.end(), w_start)
                                - win_work_offsets.begin()) - 1;

                            SIZE_TYPE last_wi = std::numeric_limits<SIZE_TYPE>::max();
                            DeltaCSRRowCursor<uint32_t> bc_cursor;
                            std::size_t elem_pos = 0, byte_pos = 0;

                            for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                                while (wi + 1 < n_windows && win_work_offsets[wi + 1] <= w) ++wi;

                                const SIZE_TYPE br          = win_br[wi];
                                const SIZE_TYPE tile_offset = w - win_work_offsets[wi];
                                const value_type* local     = &win_vals[static_cast<std::size_t>(wi) * 4];

                                if (wi != last_wi) {
                                    // Walk from this row's start, tracking
                                    // elem_pos/byte_pos as we go (mirrors
                                    // disldo_forward's collection loop) --
                                    // avoids find()'s redundant O(row_nnz)
                                    // rescan below: it did the SAME walk
                                    // twice per tile (once here via the
                                    // cursor, once again inside find()'s
                                    // raw_find), measured as the dominant
                                    // real overhead vs. dense at high
                                    // density (2.7x slower than dense for
                                    // identical tile-work at density=0.9,
                                    // serial pre-pass itself <1% of total
                                    // time -- see conversation/benchmark).
                                    bc_cursor = weights.block4.row_cursor(static_cast<std::size_t>(br));
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

                                // const at_index(): the walk above already
                                // knows this tile's exact storage position
                                // -- skip find()'s redundant rescan (see
                                // comment above). Read-only, does not mark
                                // the handle dirty, same as disldo_forward's
                                // own block4 loop.
                                const auto tile = weights.block4.at_index(
                                    static_cast<uint32_t>(br), bc, elem_pos, byte_pos);
                                const uint8_t* tdata = tile.raw_data();

                                // Zero-skip (task: top_k4 sparsity work):
                                // `local[]` holds this window's 4 gathered
                                // input values -- until now, every li was
                                // decoded and multiplied regardless of
                                // whether local[li]==0, so sparsifying the
                                // input (top-k or otherwise) was a silent
                                // no-op on a block4-resident layer, only
                                // the OUTER window-admission check (line
                                // ~439, row_nnz_b4>0) ever skipped work.
                                // Window-level early-out: if every gathered
                                // value in this window is exactly zero
                                // (either because the input was already
                                // zero there, or a sparsification step
                                // zeroed all 4), skip the lj/li decode+
                                // accumulate loops below entirely -- the
                                // walk bookkeeping above (elem_pos/byte_pos/
                                // bc_cursor) has already run and must not be
                                // skipped (later windows' incremental walk
                                // depends on it), only the real per-slot
                                // decode work is saved here.
                                if (local[0] == value_type(0) && local[1] == value_type(0) &&
                                    local[2] == value_type(0) && local[3] == value_type(0)) {
                                    continue;
                                }

                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    const std::size_t col = static_cast<std::size_t>(bc) * BLOCK4_TILE + lj;
                                    if (col >= out_cols) continue;

                                    value_type acc = value_type(0);
                                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                                        const std::size_t row =
                                            static_cast<std::size_t>(br) * BLOCK4_TILE + li;
                                        if (row >= num_inputs) continue;
                                        // Per-li skip: a zeroed gathered input
                                        // contributes nothing regardless of the
                                        // weight -- skip its decode too, not
                                        // just the multiply-accumulate.
                                        if (local[li] == value_type(0)) continue;
                                        const uint8_t byte = tdata[Block4Tile::slot_index(li, lj)];
                                        if (byte == 0) continue;
                                        const value_type w_decoded = FP4_TABLE[byte & 0xFu];
                                        // Rank-N scale, matching disldo_forward's
                                        // block4 phase and the scattered fix above.
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

            // Sum EVERY thread's private slice, not just thread 0's --
            // mirrors disldo_forward's own identical final block4
            // reduction (linear_disldo.hpp).
            for (int t = 0; t < num_cpus; ++t) {
                const value_type* s = all_b4_outputs.data() + static_cast<std::size_t>(t) * num_outputs;
                for (std::size_t i = 0; i < num_outputs; ++i)
                    output[i] += s[i];
            }
        }
    }

    // ── AQRS additive branch ────────────────────────────────────────────────
    //
    // Direct port of disldo_forward's identical section (linear_disldo.hpp),
    // adapted for sparse input: the P[k] projection walks the CSR's nonzero
    // entries instead of a dense row scan -- functionally identical, since
    // the dense version already skips iv==0 entries. The second pass
    // (P -> output) is unchanged (dense over out_cols, no sparsity to
    // exploit there -- every output column can receive a nonzero additive
    // contribution regardless of which synapses are live). No-op at the
    // default additive_rank==0.
    if (weights.additive_rank > 0) {
        std::vector<value_type> proj(
            static_cast<std::size_t>(input_tensor.rows) * weights.additive_rank, value_type(0));
        for (SIZE_TYPE b = 0; b < input_tensor.rows; ++b) {
            const SIZE_TYPE batch_start = (*input_tensor.ptrs[0])[b];
            const SIZE_TYPE batch_end   = (*input_tensor.ptrs[0])[b + 1];
            value_type* p_row = proj.data() + static_cast<std::size_t>(b) * weights.additive_rank;
            for (SIZE_TYPE i = batch_start; i < batch_end; ++i) {
                const SIZE_TYPE  r  = (*input_tensor.indices[0])[i];
                const value_type iv = (*input_tensor.values[0])[i];
                if (iv == value_type(0)) continue;
                for (std::size_t k = 0; k < weights.additive_rank; ++k)
                    p_row[k] += weights.get_additive_u_k(static_cast<std::size_t>(r), k) * iv;
            }
        }
        for (SIZE_TYPE b = 0; b < input_tensor.rows; ++b) {
            const value_type* p_row = proj.data() + static_cast<std::size_t>(b) * weights.additive_rank;
            value_type* out_row = output + static_cast<std::size_t>(b) * out_cols;
            for (std::size_t c = 0; c < out_cols; ++c) {
                value_type acc = value_type(0);
                for (std::size_t k = 0; k < weights.additive_rank; ++k)
                    acc += weights.get_additive_gamma_k(k) * weights.get_additive_v_k(c, k) * p_row[k];
                out_row[c] += acc;
            }
        }
    }
}

// delta_csr_backward (sparse input + sparse gradient) removed here -- see
// conversation. Confirmed wrong design: sparse input in backward permanently
// loses the ability to correct "didn't fire & should have" (a row not in the
// sparse input representation has no computational path to receive gradient
// at all, regardless of how strong the signal is). Only "fired & shouldn't
// have" could ever be fixed. Replaced by disldo_backward_sparse_grad
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

// Template-parameter parity with linear_disldo.hpp::disldo_backward (task
// #100, "[Deferred] Apply same template params to sisldo_ops.hpp backward
// functions"): ScalePolicy/StochasticRounding/SynapsePolicyT, in the same
// order disldo_backward uses (DeferredScaleWrite deliberately scoped out --
// orthogonal to this function's actual callers). This closes two real gaps
// at once: (1) synapse_kwargs (max_abs_delta/max_ci/min_decay_frac/
// scale_invariant) previously could not reach backward_sparse at all, no
// matter what a caller passed; (2) the scattered AND block4 phases below
// were both keeping their weight-update math in TRUE-WEIGHT space while
// storing back into CODE space via `new_w / combined_scale` -- the same
// ~1/S^2-scale-direction bug class found and fixed in FP8's block4
// backward earlier (see project_fp8_block4_scale_bug). Porting to
// disldo_backward's own code-space convention (quant += update_cw(...),
// S properly threaded as a real per-synapse scale rather than implicitly
// inverted) fixes this identically here.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t,
          typename ScalePolicy = RMSpropScalePolicy<typename ValueAccessor<VALUES_TYPE>::value_type>,
          bool StochasticRounding = true,
          template <typename> class SynapsePolicyT = BoundedRMSpropSynapsePolicy>
void disldo_backward_sparse_grad(
    const typename ValueAccessor<VALUES_TYPE>::value_type* input,   // dense [batch, n_inputs]
    SIZE_TYPE batch,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& out_grad_sparse,
    typename ValueAccessor<VALUES_TYPE>::value_type* input_gradients,  // dense [batch, n_inputs], accumulated
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_input_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_grad_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type   learning_rate = 0.01f,
    const int    num_cpus = 4,
    bool         lr_per_row_nnz = false,
    bool         damp_by_importance = true,
    typename ValueAccessor<VALUES_TYPE>::value_type   beta2 = 0.999f,
    typename ValueAccessor<VALUES_TYPE>::value_type   eps = 1e-8f,
    typename ValueAccessor<VALUES_TYPE>::value_type   min_decay_frac = 0.0f,
    typename ValueAccessor<VALUES_TYPE>::value_type   max_abs_delta = 1e30f,
    typename ValueAccessor<VALUES_TYPE>::value_type   max_ci = 1e30f,
    bool         scale_invariant = false)
{
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
        for (SIZE_TYPE j = (*out_grad_sparse.ptrs[0])[i]; j < (*out_grad_sparse.ptrs[0])[i+1]; ++j)
            neuron_grad_accum[(*out_grad_sparse.indices[0])[j]] +=
                std::abs((*out_grad_sparse.values[0])[j]);

    // ── AQRS rank-N scale scaffolding ───────────────────────────────────────
    //
    // Direct port of disldo_backward's identical scaffolding
    // (linear_disldo.hpp) -- see its own comments for the full rationale.
    // Shared by both the scattered phase below and the block4 phase further
    // down, hence computed here rather than inside either guarded block.
    const std::size_t rank = weights.scale_rank;
    std::vector<value_type> t_col_grad(static_cast<std::size_t>(num_cpus) * out_cols * rank, value_type(0));
    std::vector<value_type> t_col_grad_contrib(static_cast<std::size_t>(num_cpus) * out_cols * rank, value_type(0));
    const bool output_scale_trainable = weights.output_scale_is_trainable;
    // AQRS gamma's own gradient (task #273/#283 parity) -- layer-wide, not
    // per-row/col, so sized num_cpus*rank (not num_cpus*out_cols*rank like
    // t_col_grad above).
    std::vector<value_type> t_gamma_grad(static_cast<std::size_t>(num_cpus) * rank, value_type(0));
    std::vector<value_type> t_gamma_grad_contrib(static_cast<std::size_t>(num_cpus) * rank, value_type(0));

    // Pre-size value_scale/output_scale to n_inputs*rank / out_cols*rank --
    // see disldo_backward's identical pre-sizing comment (linear_disldo.hpp)
    // for the real bug this exact backfill pattern fixes (a uniform 1.0
    // fill would put every new rank component in permanent lockstep with
    // k==0; only k==0 gets the transparent-default 1.0, k>=1 starts at the
    // neutral 0.0).
    if (weights.value_scale.size() < n_inputs * rank) {
        const std::size_t old_size = weights.value_scale.size();
        weights.value_scale.resize(n_inputs * rank, value_type(0));
        for (std::size_t idx = old_size; idx < weights.value_scale.size(); ++idx)
            if (idx % rank == 0) weights.value_scale[idx] = value_type(1);
    }
    if (weights.output_scale.size() < out_cols * rank) {
        const std::size_t old_size = weights.output_scale.size();
        weights.output_scale.resize(out_cols * rank, value_type(0));
        for (std::size_t idx = old_size; idx < weights.output_scale.size(); ++idx)
            if (idx % rank == 0) weights.output_scale[idx] = value_type(1);
    }
    if (weights.value_scale_importance.size() < n_inputs * rank)
        weights.value_scale_importance.resize(n_inputs * rank, value_type(0));
    if (weights.output_scale_importance.size() < out_cols * rank)
        weights.output_scale_importance.resize(out_cols * rank, value_type(0));
    if (weights.value_scale_step.size() < n_inputs * rank)
        weights.value_scale_step.resize(n_inputs * rank, 0);

    // NOTE: dc.empty() (zero scattered nnz) does NOT mean "nothing to do"
    // -- see sisldo_forward's identical fix/comment. A layer can be
    // entirely block4-resident and still have real work in the block4
    // phase below, appended after this guarded block instead of an
    // unconditional early return.
    if (!dc.empty()) {
    // Importance stats accumulators across batches -- each batch's
    // #pragma omp parallel for is a SEPARATE parallel region (re-created
    // every batch iteration), so reduction() handles within-one-batch
    // thread-safety and these accumulate each batch's reduced total for one
    // final call after the whole loop. Value stats (update_value_stats_
    // aggregate) are intentionally NOT tracked here -- see disldo_backward's
    // comment for the same reasoning.
    double total_sum_abs_new_i = 0.0, total_sum_abs_old_i = 0.0;
    double total_sum_sq_new_i  = 0.0, total_sum_sq_old_i  = 0.0;
    value_type total_max_new_i = value_type(0);

    // value_scale gradient: serial per-(row,k) vector accumulated across
    // batches (within each batch's parallel for, each r is unique per
    // thread, so += into scale_grad_sums_rank[r*rank+k] is race-free;
    // across batch iterations the outer loop is serial, so also race-free).
    // Applied once after all batches -- "sum first, then apply lr" per
    // conversation. NOTE: unlike disldo_backward (row-outer/batch-inner,
    // so its own scale_grad_sum_rank is a small per-row-local vector reset
    // every row), this function nests batch OUTER / row INNER -- a given
    // row is visited once per SEPARATE batch iteration, not all at once --
    // so this accumulator must persist across the whole `for (batch)` loop,
    // indexed by the full (row,k) pair, not just k.
    std::vector<double> scale_grad_sums_rank(n_inputs * rank, 0.0);
    // Parallel forward-contribution accumulator, mirroring
    // disldo_backward's scale_grad_sum_rank_contrib (linear_disldo.hpp) --
    // same additive (square-then-sum) combination, now applied here too
    // since this function shares the same value_scale/value_scale_importance
    // arrays and was previously the one path left using plain g^2.
    std::vector<double> scale_grad_sums_rank_contrib(n_inputs * rank, 0.0);

    for (SIZE_TYPE b = 0; b < batch; ++b) {
        const SIZE_TYPE og_start = (*out_grad_sparse.ptrs[0])[b];
        const SIZE_TYPE og_end   = (*out_grad_sparse.ptrs[0])[b + 1];

        double batch_sum_abs_new_i = 0.0, batch_sum_abs_old_i = 0.0;
        double batch_sum_sq_new_i  = 0.0, batch_sum_sq_old_i  = 0.0;
        value_type batch_max_new_i = value_type(0);

        #pragma omp parallel for num_threads(num_cpus) schedule(static) \
            reduction(+:batch_sum_abs_new_i, batch_sum_abs_old_i, batch_sum_sq_new_i, batch_sum_sq_old_i) \
            reduction(max:batch_max_new_i)
        for (std::size_t r = 0; r < n_inputs; ++r) {
            const std::size_t nnz_this_row = L.row_nnz(r);
            if (nnz_this_row == 0) continue;
            const value_type in_val = input[b * n_inputs + r];
            // lr_row/nnz_this_row -- see disldo_backward's comment for the
            // full reasoning (a row with more synapses gets more simultaneous
            // per-synapse nudges each pass, so dividing by nnz_this_row keeps
            // the aggregate weight update comparable across rows of different
            // connection counts).
            const value_type effective_lr = lr_per_row_nnz
                ? learning_rate / static_cast<value_type>(nnz_this_row)
                : learning_rate;
            // value_scale's own scale_eff_lr (= learning_rate/nnz_this_row,
            // ALWAYS, independent of lr_per_row_nnz -- see disldo_backward's
            // comment) is applied once after all batches now, not folded in
            // per-synapse -- see the final application loop below.

            auto cursor = dc.row_cursor(r);
            SIZE_TYPE  og_ptr   = og_start;   // fresh per row -- each row does its own merge
            value_type dx_accum = value_type(0);
            const value_type imp_scale = weights.get_importance_scale(r);

            // Per-thread rank-N accumulator lambdas -- mirrors
            // disldo_backward's mcol_at/mgamma_at exactly (linear_disldo.hpp).
            const int tid = omp_get_thread_num();
            value_type* mcol_base = t_col_grad.data() + static_cast<std::size_t>(tid) * out_cols * rank;
            auto mcol_at = [&](std::size_t col_, std::size_t k) -> value_type& { return mcol_base[col_ * rank + k]; };
            value_type* mcol_contrib_base = t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * out_cols * rank;
            auto mcol_at_contrib = [&](std::size_t col_, std::size_t k) -> value_type& { return mcol_contrib_base[col_ * rank + k]; };
            value_type* mgamma_base = t_gamma_grad.data() + static_cast<std::size_t>(tid) * rank;
            auto mgamma_at = [&](std::size_t k) -> value_type& { return mgamma_base[k]; };
            value_type* mgamma_contrib_base = t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * rank;
            auto mgamma_at_contrib = [&](std::size_t k) -> value_type& { return mgamma_contrib_base[k]; };

            for (std::size_t e = 0; e < nnz_this_row; ++e) {
                const COL_TYPE    col = cursor.advance();
                const std::size_t vb  = L.elem_start[r] + e;
                const value_type  cw_orig = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                // S: real per-synapse scale, rank-N (reduces to the exact
                // original val_scale*out_scale at scale_rank==1), matching
                // disldo_backward's own `S = weights.get_scale(r, col)`.
                // Threading S through SynapsePolicy::update_cw (instead of
                // the old true-weight-space `new_w / combined_scale` store)
                // is what fixes the ~1/S^2 bug noted above.
                const value_type  S = weights.get_scale(r, col);
                const value_type  w = cw_orig * S;   // -> true units, for dx only

                // Merge-advance (both this row's columns and the gradient's
                // columns are sorted ascending) -- O(nnz_this_row + grad_nnz)
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
                    const value_type out_scale = weights.get_output_scale(col);
                    const value_type out_imp_scale = weights.get_output_importance_scale(col);
                    const value_type combined_imp_scale = imp_scale * out_imp_scale;
                    const value_type grad = dy_val * in_val;   // scales with true input value
                    const value_type ci_orig = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                    value_type ci = ci_orig * combined_imp_scale;   // -> true units
                    // Additive contrib combination, mirroring disldo_backward's
                    // own `ci` update (linear_disldo.hpp) -- see its docstring
                    // for the full rationale (square-then-sum, not sum-then-
                    // square: a large-magnitude disagreement between grad
                    // and contrib must still damp the step, not collapse
                    // the denominator toward zero and explode it). w here
                    // is the true (pre-update) weight, same role as
                    // cw_orig there.
                    const value_type contrib = in_val * w;
                    ci = SynapsePolicy::update_ci(ci, grad, contrib, beta2, min_decay_frac, max_ci);
                    value_type quant = cw_orig;   // code-space accumulator, matches disldo_backward
                    quant += SynapsePolicy::update_cw(grad, ci, S, effective_lr, eps,
                                                       damp_by_importance, max_abs_delta, scale_invariant);
                    if constexpr (StochasticRounding) {
                        ValueAccessor<VALUES_TYPE>::set_stochastic_live(dc.values, vb, quant, ci / combined_imp_scale);
                    } else {
                        ValueAccessor<VALUES_TYPE>::set_live(dc.values, vb, quant, ci / combined_imp_scale);
                    }
                    const value_type actual_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                    batch_sum_abs_new_i += std::abs(static_cast<double>(actual_imp));
                    batch_sum_abs_old_i += std::abs(static_cast<double>(ci_orig));
                    batch_sum_sq_new_i  += static_cast<double>(actual_imp) * actual_imp;
                    batch_sum_sq_old_i  += static_cast<double>(ci_orig) * ci_orig;
                    batch_max_new_i = std::max(batch_max_new_i, std::abs(actual_imp));

                    // dL/d(value_scale_k(r,k)) = grad * quant_floor *
                    // output_scale_k(col,k) * gamma_k, for EACH component k
                    // -- direct port of disldo_backward's identical loop
                    // (linear_disldo.hpp:1025-1054), including quant_floor's
                    // zero-escape gating (only cw_orig==0 substitutes the
                    // small positive epsilon; every other value uses itself
                    // directly, exact and signed).
                    const value_type quant_floor = (cw_orig == value_type(0)) ? value_type(0.1f) : cw_orig;
                    for (std::size_t k = 0; k < rank; ++k) {
                        const value_type out_scale_k = weights.get_output_scale_k(col, k);
                        const value_type val_scale_k = weights.get_value_scale_k(r, k);
                        const value_type gamma_k = weights.get_scale_gamma_k(k);
                        scale_grad_sums_rank[r * rank + k] += static_cast<double>(quant_floor) * static_cast<double>(out_scale_k)
                                              * static_cast<double>(gamma_k) * grad;
                        mcol_at(col, k) += quant_floor * val_scale_k * gamma_k * grad;
                        mgamma_at(k) += quant_floor * val_scale_k * out_scale_k * grad;
                        scale_grad_sums_rank_contrib[r * rank + k] += static_cast<double>(quant_floor) * static_cast<double>(out_scale_k)
                                              * static_cast<double>(gamma_k) * static_cast<double>(contrib);
                        mcol_at_contrib(col, k) += quant_floor * val_scale_k * gamma_k * contrib;
                        mgamma_at_contrib(k) += quant_floor * val_scale_k * out_scale_k * contrib;
                    }
                }
            }
            input_gradients[b * n_inputs + r] += dx_accum;
        }

        total_sum_abs_new_i += batch_sum_abs_new_i; total_sum_abs_old_i += batch_sum_abs_old_i;
        total_sum_sq_new_i  += batch_sum_sq_new_i;  total_sum_sq_old_i  += batch_sum_sq_old_i;
        total_max_new_i = std::max(total_max_new_i, batch_max_new_i);
    }

    if (learning_rate != value_type(0)) {
        weights.update_importance_stats_aggregate(
            total_sum_abs_new_i, total_sum_abs_old_i,
            total_sum_sq_new_i,  total_sum_sq_old_i, total_max_new_i);

        // Apply value_scale gradient once per (row,k), after ALL batches.
        // scale_grad_sums_rank[r*rank+k] holds the RAW (pre-lr) gradient sum
        // now (see the accumulation site above) -- scale_eff_lr is
        // recomputed here and applied once, matching disldo_backward's
        // !DeferredScaleWrite (rank-N) branch exactly (linear_disldo.hpp:
        // 1093-1100).
        for (std::size_t r = 0; r < n_inputs; ++r) {
            const std::size_t nnz_this_row = L.row_nnz(r);
            if (nnz_this_row == 0) continue;
            const value_type scale_eff_lr = learning_rate / static_cast<value_type>(nnz_this_row);
            for (std::size_t k = 0; k < rank; ++k) {
                if (scale_grad_sums_rank[r * rank + k] == 0.0 && scale_grad_sums_rank_contrib[r * rank + k] == 0.0) continue;
                const value_type g_agg = static_cast<value_type>(scale_grad_sums_rank[r * rank + k]);
                const value_type contrib_agg = static_cast<value_type>(scale_grad_sums_rank_contrib[r * rank + k]);
                ScalePolicy::update(weights.value_scale[r * rank + k], weights.value_scale_importance[r * rank + k],
                                    g_agg, scale_eff_lr, beta2, eps, contrib_agg,
                                    &weights.get_value_scale_step_k(r, k), scale_invariant);
            }
        }
    }
    } // closes if (!dc.empty())

    // ── output_scale's own gradient reduction ───────────────────────────────
    //
    // Direct port of disldo_backward's identical block (linear_disldo.hpp)
    // -- reduces t_col_grad/t_col_grad_contrib across every thread, once per
    // (col,k), then applies via the same ScalePolicy. Placed OUTSIDE the
    // `if (!dc.empty())` guard (like the scaffolding above) since block4
    // synapses also contribute into t_col_grad, further down.
    if (learning_rate != value_type(0) && output_scale_trainable) {
        for (std::size_t c = 0; c < out_cols; ++c) {
            const std::size_t deg = c < weights.out_degree.size()
                ? static_cast<std::size_t>(weights.out_degree[c]) : 0;
            if (deg == 0) continue;
            const value_type col_eff_lr = learning_rate / static_cast<value_type>(deg);
            for (std::size_t k = 0; k < rank; ++k) {
                double col_grad_sum = 0.0, col_grad_sum_contrib = 0.0;
                for (int t = 0; t < num_cpus; ++t) {
                    col_grad_sum += t_col_grad[static_cast<std::size_t>(t) * out_cols * rank + c * rank + k];
                    col_grad_sum_contrib += t_col_grad_contrib[static_cast<std::size_t>(t) * out_cols * rank + c * rank + k];
                }
                const value_type g_agg = static_cast<value_type>(col_grad_sum);
                const value_type contrib_agg = static_cast<value_type>(col_grad_sum_contrib);
                ScalePolicy::update(weights.output_scale[c * rank + k], weights.output_scale_importance[c * rank + k],
                                    g_agg, col_eff_lr, beta2, eps, contrib_agg,
                                    &weights.get_output_scale_step_k(c, k), scale_invariant);
            }
        }
    }

    // ── AQRS scale_gamma's own update ───────────────────────────────────────
    //
    // Direct port of disldo_backward's identical block (linear_disldo.hpp),
    // including the EMA/dynamic-rank-control tracking call
    // (update_scale_gamma_ema_k) -- that method itself already lives on
    // `weights` (shared with the dense path), this just needs to feed it
    // the same per-k inputs disldo_backward does. l1_coef is not yet a
    // parameter of this function (matches its current scope -- scale_gamma
    // itself was never previously touched at all here); defaults to 0
    // (off) until threaded through, same as every other new caller of
    // this L1 mechanism starts.
    if (learning_rate != value_type(0) && weights.scale_gamma_is_trainable) {
        std::vector<value_type> g_agg_by_k(rank);
        for (std::size_t k = 0; k < rank; ++k) {
            double gamma_grad_sum = 0.0, gamma_grad_sum_contrib = 0.0;
            for (int t = 0; t < num_cpus; ++t) {
                gamma_grad_sum += t_gamma_grad[static_cast<std::size_t>(t) * rank + k];
                gamma_grad_sum_contrib += t_gamma_grad_contrib[static_cast<std::size_t>(t) * rank + k];
            }
            const value_type g_agg = static_cast<value_type>(gamma_grad_sum);
            const value_type contrib_agg = static_cast<value_type>(gamma_grad_sum_contrib);
            g_agg_by_k[k] = g_agg;
            weights.set_scale_gamma_raw_k(k, weights.get_scale_gamma_k(k));
            ScalePolicy::update(weights.scale_gamma[k], weights.get_scale_gamma_state_k(k),
                                g_agg, learning_rate, beta2, eps, contrib_agg,
                                &weights.get_scale_gamma_step_k(k), false);
        }
        value_type gamma_l1_sum = value_type(0);
        for (std::size_t k = 0; k < rank; ++k) gamma_l1_sum += std::fabs(weights.scale_gamma[k]);
        const value_type grad_norm_divisor = static_cast<value_type>(n_inputs) * static_cast<value_type>(out_cols);
        for (std::size_t k = 0; k < rank; ++k) {
            const value_type abs_gamma_k = std::fabs(weights.scale_gamma[k]);
            const value_type share_k = gamma_l1_sum > value_type(0) ? abs_gamma_k / gamma_l1_sum : value_type(0);
            weights.update_scale_gamma_ema_k(k, abs_gamma_k, share_k,
                                             std::fabs(g_agg_by_k[k]) / grad_norm_divisor);
        }
    }

    // ── block4 contribution ─────────────────────────────────────────────────
    //
    // Real, previously-silent bug this closes: same as sisldo_forward's
    // (see its comment) -- block4-resident synapses were never touched by
    // this function at all. FP4-specific (block4 is FP4-only), hence the
    // if constexpr guard.
    //
    // Gather design: mirrors the scattered loop above's own merge-scan
    // (row_cursor + og_ptr walking forward through sorted out_grad_sparse),
    // applied one level up -- per TILE (4 output columns) instead of per
    // synapse. PRECONDITION: out_grad_sparse's indices, within each batch
    // row, must be ascending (same convention sisldo_forward's input
    // requires -- see its comment).
    //
    // Parallelized by BLOCK-ROW (br), not by tile -- unlike disldo_backward
    // (which partitions by flat tile index and therefore needs cross-thread
    // accumulator buffers, since two different tiles can share a block-row
    // when they differ only in block-column), here each br owns exactly 4
    // unique input rows (br*4..br*4+3) that NO OTHER br ever touches, and a
    // thread processes one br's tiles serially within itself -- so
    // value_scale/dx/importance-stat writes for those 4 rows can go
    // straight into shared (non-per-thread) accumulators with no race,
    // simpler than disldo_backward's scheme. This also gives the same row-
    // exclusive-ownership safety a block4 tile resize needs (see
    // block4_resize_tile_in_row's comment) for free.
    //
    // Correctness-first scalar port (matches sisldo_forward's own choice
    // not to bring over disldo_forward/backward's Block4Vec SIMD machinery
    // immediately) -- revisit with real profiling if a benchmark shows this
    // is a bottleneck, same as forward's documented approach.
    //
    // KNOWN SIMPLIFICATION (documented, not a bug, mirrors disldo_backward's
    // identical one): a row with both scattered and block4 synapses gets
    // two sequential value_scale gradient steps (this section's own, after
    // the scattered section's own above) rather than one combined step.
    if constexpr (std::is_same_v<VALUES_TYPE, FP4BiPacked>) {
        if (weights.block4.n_tiles() > 0) {
            const auto& BL4 = weights.block4.block_layout;
            const std::size_t tiles_r = BL4.rows;

            // Safe to write directly (no per-thread buffer / final
            // reduction needed): each row is exclusively owned by one
            // br across the WHOLE function (br = row/4, no two br's
            // share a row), and different batches' parallel regions
            // never overlap in time (each is fully joined before the
            // next begins) -- see comment above.
            std::vector<double> row_scale_grad_sums(n_inputs, 0.0);
            // Parallel forward-contribution accumulator -- see the scattered
            // path's scale_grad_sums_contrib above for the full rationale.
            std::vector<double> row_scale_grad_sums_contrib(n_inputs, 0.0);

            double b4_total_sum_abs_new = 0.0, b4_total_sum_abs_old = 0.0;
            double b4_total_sum_sq_new  = 0.0, b4_total_sum_sq_old  = 0.0;
            value_type b4_total_max_new = value_type(0);

            for (SIZE_TYPE b = 0; b < batch; ++b) {
                const SIZE_TYPE og_start = (*out_grad_sparse.ptrs[0])[b];
                const SIZE_TYPE og_end   = (*out_grad_sparse.ptrs[0])[b + 1];

                double batch_sum_abs_new = 0.0, batch_sum_abs_old = 0.0;
                double batch_sum_sq_new  = 0.0, batch_sum_sq_old  = 0.0;
                value_type batch_max_new = value_type(0);

                #pragma omp parallel for num_threads(num_cpus) schedule(static) \
                    reduction(+:batch_sum_abs_new, batch_sum_abs_old, batch_sum_sq_new, batch_sum_sq_old) \
                    reduction(max:batch_max_new)
                for (std::size_t br = 0; br < tiles_r; ++br) {
                    const std::size_t row_nnz_b4 = BL4.row_nnz(br);
                    if (row_nnz_b4 == 0) continue;
                    // Total live slots across ALL of this row's tiles this
                    // call -- every tile contributes exactly BLOCK4_TILE
                    // slots per row it covers (dense, weight=0.0 included,
                    // see block4.hpp), mirrors disldo_backward's
                    // row_live_count.
                    const std::size_t nnz_row = row_nnz_b4 * BLOCK4_TILE;

                    value_type dx_accum[BLOCK4_TILE]                = {0, 0, 0, 0};
                    double     row_grad_local[BLOCK4_TILE]          = {0, 0, 0, 0};
                    double     row_grad_local_contrib[BLOCK4_TILE]  = {0, 0, 0, 0};

                    if (learning_rate == value_type(0)) {
                        // Read-only: no writes anywhere in this row, so no
                        // resize can ever happen -- the plain shared-store
                        // at_index() path has no concurrency hazard at all
                        // here (see Block4Store::RowWorkspace's comment on
                        // why that hazard is specifically about growth).
                        auto bc_cursor = weights.block4.row_cursor(br);
                        std::size_t elem_pos = BL4.elem_start[br];
                        std::size_t byte_pos = weights.block4.tile_byte_start[br];
                        SIZE_TYPE og_ptr = og_start;

                        for (std::size_t e = 0; e < row_nnz_b4; ++e) {
                            const uint32_t    bc        = bc_cursor.advance();
                            const std::size_t window_lo = static_cast<std::size_t>(bc) * BLOCK4_TILE;
                            const std::size_t window_hi = window_lo + BLOCK4_TILE;

                            while (og_ptr < og_end &&
                                   static_cast<std::size_t>((*out_grad_sparse.indices[0])[og_ptr]) < window_lo)
                                ++og_ptr;

                            value_type dy_local[BLOCK4_TILE] = {0, 0, 0, 0};
                            bool any = false;
                            for (SIZE_TYPE p = og_ptr;
                                 p < og_end && static_cast<std::size_t>((*out_grad_sparse.indices[0])[p]) < window_hi;
                                 ++p) {
                                const std::size_t col = static_cast<std::size_t>((*out_grad_sparse.indices[0])[p]);
                                dy_local[col - window_lo] = (*out_grad_sparse.values[0])[p];
                                any = true;
                            }

                            if (any) {
                                const auto tile = weights.block4.at_index(
                                    static_cast<uint32_t>(br), bc, elem_pos, byte_pos);
                                const uint8_t* tdata = tile.raw_data();
                                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                                    const std::size_t row = br * BLOCK4_TILE + li;
                                    if (row >= n_inputs) continue;
                                    const value_type val_scale = weights.get_value_scale(row);
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (dy_local[lj] == value_type(0)) continue;
                                        const std::size_t col = window_lo + lj;
                                        if (col >= out_cols) continue;
                                        const uint8_t byte = tdata[Block4Tile::slot_index(li, lj)];
                                        const value_type w_decoded = FP4_TABLE[byte & 0xFu];
                                        const value_type out_scale = weights.get_output_scale(col);
                                        const value_type w = w_decoded * val_scale * out_scale;
                                        dx_accum[li] += w * dy_local[lj];
                                    }
                                }
                            }
                            byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
                            ++elem_pos;
                        }
                    } else {
                        // Writing: use a row-local workspace so growth never
                        // touches shared tile_data/tbyte_start/tbyte_end
                        // until a single row-exclusive merge-back at the
                        // end -- see Block4Store::RowWorkspace's comment
                        // for why the plain shared-store path (find()-free
                        // or not) is NOT safe here under concurrent
                        // per-row-owning threads, confirmed via ASan.
                        auto ws = weights.block4.snapshot_row(br);
                        SIZE_TYPE og_ptr = og_start;
                        std::size_t local_pos = 0;

                        for (std::size_t e = 0; e < row_nnz_b4; ++e) {
                            const uint32_t    bc        = ws.bc[e];
                            const std::size_t window_lo = static_cast<std::size_t>(bc) * BLOCK4_TILE;
                            const std::size_t window_hi = window_lo + BLOCK4_TILE;

                            while (og_ptr < og_end &&
                                   static_cast<std::size_t>((*out_grad_sparse.indices[0])[og_ptr]) < window_lo)
                                ++og_ptr;

                            value_type dy_local[BLOCK4_TILE] = {0, 0, 0, 0};
                            bool any = false;
                            for (SIZE_TYPE p = og_ptr;
                                 p < og_end && static_cast<std::size_t>((*out_grad_sparse.indices[0])[p]) < window_hi;
                                 ++p) {
                                const std::size_t col = static_cast<std::size_t>((*out_grad_sparse.indices[0])[p]);
                                dy_local[col - window_lo] = (*out_grad_sparse.values[0])[p];
                                any = true;
                            }

                            const std::size_t this_local_pos = local_pos;
                            if (any) {
                                uint8_t scratch[BLOCK4_TILE_SLOTS];
                                weights.block4.unpack_workspace_tile(ws, e, this_local_pos, scratch);
                                bool dirty = false;

                                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                                    const std::size_t row = br * BLOCK4_TILE + li;
                                    if (row >= n_inputs) continue;
                                    const value_type in_val    = input[static_cast<std::size_t>(b) * n_inputs + row];
                                    const value_type val_scale = weights.get_value_scale(row);
                                    const value_type imp_scale = weights.get_importance_scale(row);
                                    const value_type effective_lr = lr_per_row_nnz
                                        ? learning_rate / static_cast<value_type>(nnz_row)
                                        : learning_rate;
                                    // value_scale's own scale_eff_lr is applied
                                    // once after all batches now, not folded in
                                    // per-synapse -- see the final application
                                    // loop below (recomputed there from nnz_row).

                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (dy_local[lj] == value_type(0)) continue;
                                        const std::size_t col = window_lo + lj;
                                        if (col >= out_cols) continue;

                                        const uint8_t byte = scratch[Block4Tile::slot_index(li, lj)];
                                        const value_type w_decoded      = FP4_TABLE[byte & 0xFu];
                                        const value_type out_scale      = weights.get_output_scale(col);
                                        const value_type combined_scale = val_scale * out_scale;
                                        // S: real per-synapse scale (Phase 2 swaps this one line
                                        // for the rank-N accessor) -- see the scattered path's
                                        // identical note on why threading S through
                                        // SynapsePolicy::update_cw (code-space) instead of the old
                                        // true-weight-space `new_w / combined_scale` store fixes
                                        // the ~1/S^2 bug here too.
                                        const value_type S               = combined_scale;
                                        const value_type w               = w_decoded * S;
                                        const value_type dy_val          = dy_local[lj];
                                        dx_accum[li] += w * dy_val;

                                        const value_type out_imp_scale = weights.get_output_importance_scale(col);
                                        const value_type combined_imp_scale = imp_scale * out_imp_scale;
                                        const value_type imp_decoded = FP4_TABLE[(byte >> 4) & 0xFu];
                                        const value_type grad = dy_val * in_val;
                                        value_type ci = imp_decoded * combined_imp_scale;
                                        // Additive contrib combination, matching
                                        // the scattered path above -- see its
                                        // comment for the full rationale. w is
                                        // the true (pre-update) weight decoded
                                        // just above.
                                        const value_type contrib = in_val * w;
                                        ci = SynapsePolicy::update_ci(ci, grad, contrib, beta2, min_decay_frac, max_ci);
                                        value_type quant = w_decoded;   // code-space accumulator, matches disldo_backward
                                        quant += SynapsePolicy::update_cw(grad, ci, S, effective_lr, eps,
                                                                           damp_by_importance, max_abs_delta, scale_invariant);
                                        // was_live gate -- see
                                        // linear_disldo.hpp's was_live4/
                                        // was_live4_8 declaration comments for
                                        // the full rationale: `byte` here is
                                        // the PRE-update packed byte, still
                                        // untouched at this point -- a cell
                                        // that was never a real synapse
                                        // (byte==0) must stay allowed to
                                        // round back to 0, not be forced
                                        // permanently live just because this
                                        // column happened to have gradient
                                        // signal this step.
                                        const bool was_live = (byte != 0);
                                        const value_type imp_ratio = ci / combined_imp_scale;
                                        uint8_t new_w_code, new_imp_code;
                                        if constexpr (StochasticRounding) {
                                            new_w_code   = was_live ? fp4_quantize_stochastic_live(quant)
                                                                    : fp4_quantize_stochastic(quant);
                                            new_imp_code = was_live ? fp4_quantize_stochastic_live_nonneg(imp_ratio)
                                                                    : fp4_quantize_stochastic(imp_ratio);
                                        } else {
                                            new_w_code   = was_live ? fp4_quantize_live(quant)     : fp4_quantize(quant);
                                            new_imp_code = was_live ? fp4_quantize_live(imp_ratio) : fp4_quantize(imp_ratio);
                                        }
                                        scratch[Block4Tile::slot_index(li, lj)] = uint8_t((new_imp_code << 4) | new_w_code);
                                        dirty = true;

                                        const value_type actual_imp = FP4_TABLE[new_imp_code] * combined_imp_scale;
                                        const value_type stored_imp = imp_decoded * combined_imp_scale;
                                        batch_sum_abs_new += std::abs(static_cast<double>(actual_imp));
                                        batch_sum_abs_old += std::abs(static_cast<double>(stored_imp));
                                        batch_sum_sq_new  += static_cast<double>(actual_imp) * actual_imp;
                                        batch_sum_sq_old  += static_cast<double>(stored_imp) * stored_imp;
                                        batch_max_new = std::max(batch_max_new, std::abs(actual_imp));

                                        row_grad_local[li] += static_cast<double>(w_decoded)
                                            * static_cast<double>(out_scale) * (dy_val * in_val);
                                        row_grad_local_contrib[li] += static_cast<double>(w_decoded)
                                            * static_cast<double>(out_scale) * static_cast<double>(contrib);
                                    }
                                }
                                if (dirty)
                                    weights.block4.commit_dirty_tile_in_workspace(ws, e, this_local_pos, scratch);
                            }
                            local_pos += block4_stored_tile_len(ws.is_sparse[e], &ws.bytes[this_local_pos]);
                        } // tiles in this row

                        // Merge back -- evicts lowest-|true-importance|
                        // synapses only if this row genuinely grew past its
                        // own current headroom (see merge_row_workspace's
                        // comment). True importance uses the SAME per-row/
                        // per-col scale lookups as above, per conversation
                        // (raw 4-bit codes alone aren't enough resolution
                        // to rank meaningfully).
                        weights.block4.merge_row_workspace(br, ws,
                            [&](std::size_t ev_row, std::size_t ev_col, uint8_t ev_imp_code) -> double {
                                const value_type imp_scale     = weights.get_importance_scale(ev_row);
                                const value_type out_imp_scale = weights.get_output_importance_scale(ev_col);
                                return static_cast<double>(FP4_TABLE[ev_imp_code & 0xFu])
                                     * static_cast<double>(imp_scale) * static_cast<double>(out_imp_scale);
                            });
                    }

                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = br * BLOCK4_TILE + li;
                        if (row >= n_inputs) continue;
                        input_gradients[static_cast<std::size_t>(b) * n_inputs + row] += dx_accum[li];
                        if (learning_rate != value_type(0)) {
                            row_scale_grad_sums[row] += row_grad_local[li];
                            row_scale_grad_sums_contrib[row] += row_grad_local_contrib[li];
                        }
                    }
                } // br

                b4_total_sum_abs_new += batch_sum_abs_new; b4_total_sum_abs_old += batch_sum_abs_old;
                b4_total_sum_sq_new  += batch_sum_sq_new;  b4_total_sum_sq_old  += batch_sum_sq_old;
                b4_total_max_new = std::max(b4_total_max_new, batch_max_new);
            } // batch

            if (learning_rate != value_type(0)) {
                weights.update_importance_stats_aggregate(
                    b4_total_sum_abs_new, b4_total_sum_abs_old,
                    b4_total_sum_sq_new,  b4_total_sum_sq_old, b4_total_max_new);

                for (std::size_t row = 0; row < n_inputs; ++row) {
                    if (row_scale_grad_sums[row] == 0.0 && row_scale_grad_sums_contrib[row] == 0.0) continue;
                    // nnz_row for this row's block-row -- same derivation as
                    // the accumulation loop above (row_nnz_b4 * BLOCK4_TILE).
                    const std::size_t br = row / BLOCK4_TILE;
                    const std::size_t nnz_row = (br < BL4.rows ? BL4.row_nnz(br) : 0) * BLOCK4_TILE;
                    if (nnz_row == 0) continue;
                    const value_type scale_eff_lr = learning_rate / static_cast<value_type>(nnz_row);
                    const value_type g_agg = static_cast<value_type>(row_scale_grad_sums[row]);
                    const value_type contrib_agg = static_cast<value_type>(row_scale_grad_sums_contrib[row]);
                    // Routed through the swappable ScalePolicy -- see the
                    // scattered path's identical call above. SAME
                    // value_scale_step counter (shared per-row across
                    // scattered and block4, matching disldo_backward's own
                    // shared value_scale/importance).
                    ScalePolicy::update(weights.value_scale[row], weights.value_scale_importance[row],
                                        g_agg, scale_eff_lr, beta2, eps, contrib_agg,
                                        &weights.get_value_scale_step_k(row, 0), scale_invariant);
                }
            }
        }
    }
}

#endif
