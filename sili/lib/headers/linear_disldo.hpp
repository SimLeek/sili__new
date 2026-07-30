#pragma once
// Benchmark-only escape hatch: force block4's disldo_backward onto its
// pre-SIMD scalar path (identical math, no Block4Vec) to measure the SIMD
// rewrite's real speedup against a same-commit, same-everything-else
// baseline -- e.g. `-DSILI_BLOCK4_FORCE_SCALAR_BACKWARD=1`. Defaults off;
// not a runtime knob, not meant to ship enabled.
#ifndef SILI_BLOCK4_FORCE_SCALAR_BACKWARD
#define SILI_BLOCK4_FORCE_SCALAR_BACKWARD 0
#endif
#include "csr.hpp"
#include "sparse_struct.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
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

    const std::size_t n_in  = L.rows;
    const std::size_t n_out = L.cols;
    const std::size_t ost   = static_cast<std::size_t>(batch) * n_out;

    // value_scale is gradient-trainable (disldo_backward), so -- same as
    // every per-synapse weight -- it needs a forward-side importance
    // update too (Hebbian activity correlation, not just the backward
    // gradient step). Pre-size for safe indexed writes inside the
    // per-row parallel loop below (each row is thread-exclusive, so no
    // race once sized).
    if (weights.value_scale_importance.size() < n_in)
        weights.value_scale_importance.resize(n_in, value_type(0));

    // dc.empty() no longer means "nothing to do": block4 (below) may hold
    // live synapses even when the scattered CSR is empty (e.g. everything
    // in a small/dense layer promoted). L.rows/L.cols stay valid either
    // way (set at construction, independent of nnz), so skipping just this
    // block is safe.
    if (!dc.empty()) {
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
                // Same row*col combination as the weight's own scale --
                // a synapse's stored importance lives at the same (row,
                // col) position as its stored weight, so its
                // representability scale needs the same two factors.
                const value_type  out_imp_scale   = weights.get_output_importance_scale(col);
                const value_type  combined_imp_scale = imp_scale * out_imp_scale;

                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                    if (iv == value_type(0)) continue;
                    const value_type contrib = w * iv;
                    mo[static_cast<std::size_t>(b) * n_out + col] += contrib;
                    row_contrib_sum += static_cast<double>(contrib);

                    if (learning_rate != value_type(0)) {
                        const value_type stored_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                        value_type imp = stored_imp * combined_imp_scale;   // -> true units
                        imp += contrib * learning_rate / (value_type(1) + std::abs(imp));
                        ValueAccessor<VALUES_TYPE>::set_stochastic(dc.values, vb, w_stored, imp / combined_imp_scale);
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
    }  // !dc.empty()

    // block4 contribution -- same shared per-row value_scale/output_scale
    // as the scattered path above (see block4.hpp: this is the whole point
    // of NOT giving block4 its own separate scale, unlike the prototype).
    // No gather here: within an active tile, position IS the column, a
    // fixed compile-time-known offset -- see block4.hpp / BLOCK4_NOTES.md
    // for the real, measured SIMD benefit this gets on this machine that
    // the scattered loop above never can (GCC: "complicated access
    // pattern", confirmed via -fopt-info-vec, never auto-vectorizes).
    //
    // KNOWN GAP, not yet addressed: unlike the scattered path above, this
    // does not yet update value_scale_importance/output_scale_importance's
    // forward-side Hebbian signal for block4-owned synapses specifically
    // -- those importance scales are updated only by whatever fraction of
    // a row/column still has scattered synapses. Fine for a first working,
    // speed-focused version; a real quality comparison against the
    // scattered-only baseline should check whether this matters before
    // being trusted for training quality claims, not just forward value
    // correctness (which does not depend on this).
    if (weights.block4.n_tiles() > 0) {
        // Row-major cursor walk isn't parallel-for-friendly directly (same
        // reason the old hash-map iteration wasn't); collect (br,bc,elem_pos)
        // TRIPLES once per call (not Block4Tile pointers/handles -- a
        // handle can't be pre-collected across the parallel region since
        // it's move-only, RAII, and per-tile compress/decompress decisions
        // must happen within ONE thread's ownership of ONE tile at a time;
        // each thread constructs its own handle fresh, inside the loop
        // body below, from these coordinates). elem_pos is the tile's
        // actual index into block4's tile_values (this walk already knows
        // it -- block_layout.elem_start[br]+bk -- for free), passed to
        // Block4Store::at_index() so the hot loop below doesn't redo an
        // O(row_nnz) coordinate re-scan per tile via find(). That redundant
        // second scan (discover a tile here, then re-discover it again via
        // find()'s own raw_find()) measured as the dominant real cost of
        // this loop at batch=1 -- see conversation: batch=1 has too little
        // per-tile compute (16 FLOPs) to amortize even one such scan, let
        // alone two.
        // Persistent scratch (Block4Store::scratch_tile_br/bc/elem), not a
        // fresh vector every call -- see block4.hpp: batch=1 real-time
        // calls can't amortize repeated heap allocation of these the way
        // a large training batch could.
        std::vector<uint32_t>&    tile_br   = weights.block4.scratch_tile_br;
        std::vector<uint32_t>&    tile_bc   = weights.block4.scratch_tile_bc;
        std::vector<std::size_t>& tile_elem = weights.block4.scratch_tile_elem;
        const std::size_t n_b4 = weights.block4.n_tiles();
        // resize()+direct indexing, not reserve()+push_back(): push_back's
        // per-call capacity check (branch + increment) is real, measured
        // exclusive cost at this scale (~49k push_back calls across the 3
        // vectors on a fully block4-resident 512x512 layer -- confirmed
        // via callgrind: switching to scratch buffers alone barely moved
        // this cost, since reused capacity still pays the per-push_back
        // check every call regardless of allocation). resize() is a single
        // capacity check for the whole vector; the fill loop below then
        // writes through plain indexed stores.
        tile_br.resize(n_b4);
        tile_bc.resize(n_b4);
        tile_elem.resize(n_b4);
        const auto& BL4 = weights.block4.block_layout;
        std::size_t ti = 0;
        for (std::size_t br = 0; br < BL4.rows; ++br) {
            const std::size_t n_bc = BL4.row_nnz(br);
            if (n_bc == 0) continue;
            auto bc_cursor = weights.block4.row_cursor(br);
            std::size_t elem_pos = BL4.elem_start[br];
            for (std::size_t bk = 0; bk < n_bc; ++bk, ++elem_pos, ++ti) {
                tile_br[ti] = uint32_t(br);
                tile_bc[ti] = bc_cursor.advance();
                tile_elem[ti] = elem_pos;
            }
        }
        // Per-thread private output buffers, same pattern as the scattered
        // path's t_out above -- necessary, not optional: two tiles that
        // share a block-COLUMN (different block-rows, i.e. different input
        // rows feeding the same output columns) write to the same output
        // positions, so parallelizing freely over tiles without this would
        // race exactly the way the scattered path's own scatter-write
        // would without t_out.
        std::vector<value_type> b4_out(static_cast<std::size_t>(num_cpus) * ost, value_type(0));
        // Hoisted out of the loop condition below -- measured, not
        // assumed: tile_br.size() was being re-evaluated on every single
        // loop iteration instead of once (confirmed via callgrind: 8.78%
        // of this function's total instruction count was spent purely
        // inside std::vector::size(), on a fully block4-resident
        // 512x512 layer -- the compiler apparently couldn't prove
        // tile_br's size is loop-invariant across the omp-outlined
        // function boundary, so it re-read _M_finish - _M_start on
        // every iteration instead of hoisting it). A plain local
        // variable is trivially provably invariant. NOTE: correct and a
        // real instruction-count win, but measured (isolated-process
        // methodology) as NOT moving wall-clock time noticeably --
        // apparently absorbed by the CPU's own execution resources
        // (same pattern already seen once for the vector-allocation-
        // churn fix). Kept anyway: real, harmless, zero-risk, and
        // instruction count reductions are not guaranteed irrelevant on
        // every CPU/compiler this code will ever run on.
        const int64_t n_tiles_local = int64_t(n_b4);
        #pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mo = b4_out.data() + static_cast<std::size_t>(tid) * ost;
            #pragma omp for schedule(static)
            for (int64_t ti = 0; ti < n_tiles_local; ++ti) {
                const uint32_t br = tile_br[std::size_t(ti)], bc = tile_bc[std::size_t(ti)];
                // const: routes .at() through the const overload, which
                // does NOT mark the handle dirty -- forward is read-only,
                // so a sparse tile's destructor should do nothing here
                // (no wasted re-pack of unchanged content). at_index(): the
                // collection loop above already knows this tile's exact
                // storage position, so skip find()'s redundant O(row_nnz)
                // re-scan (see collection loop's comment).
                const auto tile = weights.block4.at_index(br, bc, tile_elem[std::size_t(ti)]);
                // Resolved ONCE per tile instead of once per .at() call
                // (16 calls/tile below otherwise, each re-branching on
                // whether this tile is sparse-packed -- a property that
                // can't change mid-tile). See Block4TileHandle::raw_data().
                const uint8_t* tdata = tile.raw_data();
                // Decode this column's whole 4-wide weight vector via
                // block4_vec_decode_fp4 (fp4quant.hpp's bit-shift
                // formula, not FP4_TABLE[code]'s 4 separate gathers --
                // see fp4quant.hpp's header comment); the remaining
                // per-row scale multiply/clamp stays scalar
                // (get_value_scale(row) itself isn't a SIMD operation).
                //
                // NOTE (measured, not assumed): an isolated microbenchmark
                // suggested scalar FP4_TABLE decode should win here too
                // (same as backward's decode, below) -- but swapping it
                // in for THIS specific loop measurably regressed the
                // real disldo_forward benchmark at batch=1 (~1.71x ->
                // ~1.55x speedup vs scattered CSR at 100% density,
                // reproduced consistently across repeats), unlike
                // backward where the same swap was a real, consistent
                // win. Reverted here; kept for backward. Compiler
                // codegen interactions with the surrounding code
                // apparently differ enough between the two functions
                // that the isolated test's result didn't transfer --
                // trust the real benchmark over the isolated one. See
                // TODO_DUAL_BLOCK4.md's Part C.
                //
                // LJ templated (compile-time constant), not a runtime
                // `for (lj...)` loop: -fopt-info-vec confirmed GCC could
                // not vectorize the runtime version at all -- "loop nest
                // containing two or more consecutive inner loops cannot
                // be vectorized" (the li-decode loop followed by the
                // b-batch loop, both nested inside the lj loop). A
                // per-LJ templated lambda gives the compiler 4 SEPARATE,
                // independent instantiations instead of one loop nest it
                // has to reason about jointly -- each with a compile-time-
                // known column offset, matching the pattern already used
                // for the decode step's own 4-way unroll. See
                // TODO_DUAL_BLOCK4.md's Part C for the measured effect.
                auto process_col = [&]<uint32_t LJ>() {
                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + LJ;
                    if (col >= n_out) return;
                    const value_type out_scale = weights.get_output_scale(col);

                    const Block4VecU w_codes = {uint32_t(tdata[Block4Tile::slot_index(0, LJ)] & 0xFu), uint32_t(tdata[Block4Tile::slot_index(1, LJ)] & 0xFu),
                                                 uint32_t(tdata[Block4Tile::slot_index(2, LJ)] & 0xFu), uint32_t(tdata[Block4Tile::slot_index(3, LJ)] & 0xFu)};
                    const Block4Vec w_decoded = block4_vec_decode_fp4(w_codes);
                    const value_type w_decoded_arr[BLOCK4_TILE] = {value_type(w_decoded[0]), value_type(w_decoded[1]),
                                                                     value_type(w_decoded[2]), value_type(w_decoded[3])};

                    value_type w4[BLOCK4_TILE];
                    std::size_t row_idx[BLOCK4_TILE];
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                        if (row < n_in) {
                            w4[li] = w_decoded_arr[li] * weights.get_value_scale(row) * out_scale;
                            row_idx[li] = row;
                        } else {
                            w4[li] = value_type(0);
                            row_idx[li] = 0;
                        }
                    }

                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        value_type acc = value_type(0);
                        const value_type* in_row = input + static_cast<std::size_t>(b) * in_cols;
                        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
                            acc += w4[li] * in_row[row_idx[li]];
                        mo[static_cast<std::size_t>(b) * n_out + col] += acc;
                    }
                };
                process_col.template operator()<0>();
                process_col.template operator()<1>();
                process_col.template operator()<2>();
                process_col.template operator()<3>();
                static_assert(BLOCK4_TILE == 4, "process_col above is hand-unrolled for exactly 4 columns");
            }
        }
        for (int t = 0; t < num_cpus; ++t) {
            const value_type* s = b4_out.data() + static_cast<std::size_t>(t) * ost;
            for (std::size_t i = 0; i < ost; ++i) output[i] += s[i];
        }
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
 * @param damp_by_importance When true (default): the weight update is
 *        divided by (1+|ci|), so a synapse that's accumulated a lot of
 *        same-direction gradient pressure gets progressively smaller
 *        steps -- a per-synapse adaptive-learning-rate effect. When
 *        false: the raw (-effective_lr * g) step is applied directly,
 *        with no damping -- ci is still tracked/updated identically
 *        either way (importance stays meaningful for pruning/
 *        synaptogenesis decisions regardless), only its use as a
 *        WEIGHT-UPDATE damping factor is toggled. Exists specifically
 *        so a caller can A/B this mechanism against itself on the same
 *        kernel -- see sili_peridot's/sili__new's importance-damping-
 *        as-optimizer integration test.
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
    bool         lr_per_row_nnz = false,
    bool         damp_by_importance = true)
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

    if (dc.empty() && weights.block4.n_tiles() == 0) return;

    const std::size_t dst = static_cast<std::size_t>(batch) * in_cols;
    std::vector<value_type> t_dx(static_cast<std::size_t>(num_cpus) * dst, value_type(0));

    // output_scale's own gradient, symmetric to value_scale's: a column
    // can be touched by many rows, spread across threads by the outer
    // #pragma omp for (over rows) -- each thread accumulates into its own
    // [n_out]-sized slice, reduced after the parallel region (same
    // pattern as t_dx above). Only applied if output_scale_is_trainable
    // (set by set_output_scale_raw) -- not a size check, since the resize
    // below runs unconditionally for safe reads regardless of mode.
    // Shared between the scattered loop below AND block4's own loop
    // further down -- both write into the same buffer (per-thread-private
    // slices, indexed by their own tid), summed once at the very end.
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

    if (!dc.empty()) {
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
                // Same row*col combination as the weight's own scale --
                // see the matching comment in disldo_forward.
                const value_type  out_imp_scale      = weights.get_output_importance_scale(col);
                const value_type  combined_imp_scale = imp_scale * out_imp_scale;
                value_type ci  = ci_orig * combined_imp_scale;   // -> true units

                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type iv  = input[static_cast<std::size_t>(b) * in_cols + r];
                    const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col];
                    const value_type g   = dyv * iv;

                    if (learning_rate != value_type(0)) {
                        ci -= g * effective_lr;
                        cw += damp_by_importance
                            ? (-effective_lr * g) / (value_type(1) + std::abs(ci))
                            : (-effective_lr * g);
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
                    ValueAccessor<VALUES_TYPE>::set_stochastic(dc.values, vb, cw / combined_scale, ci / combined_imp_scale);
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
    }  // !dc.empty()

    // block4 backward: dx + inline weight/importance update, mirroring the
    // scattered loop above but keyed by tile instead of CSR row. Same
    // shared value_scale/output_scale as forward and the scattered path
    // (see block4.hpp) -- moving a synapse between representations stays
    // a lossless byte copy, so its gradient math must stay consistent too.
    //
    // Race note: the scattered loop above parallelizes over ROWS, so each
    // row (and thus each row's value_scale/value_scale_importance update)
    // is owned by exactly one thread. Here we parallelize over TILES, and
    // two different tiles can share the same block-ROW (different block-
    // columns) -- so a row's value_scale gradient can now be touched by
    // more than one thread concurrently. Fixed the same way output_scale's
    // gradient already is in this function: per-thread-private accumulator
    // buffers (t_row_grad, indexed like t_col_grad), reduced serially once
    // after the parallel region, instead of applying the update inline.
    //
    // KNOWN SIMPLIFICATION (documented, not a bug): if a row has BOTH
    // scattered and block4 synapses, its value_scale gets two sequential
    // gradient steps (the scattered loop's own step above, then block4's
    // step below) rather than one combined step over the true total nnz.
    // Mathematically this is just two successive descent steps, not an
    // incorrect one -- acceptable for a first working version; revisit if
    // the comparison script (TODO_DUAL_BLOCK4.md) shows it matters.
    if (weights.block4.n_tiles() > 0) {
        // Coordinates + storage index -- see forward's identical comment
        // above for why (handles are move-only RAII, constructed fresh
        // per-tile inside the parallel loop body below; elem_pos lets that
        // construction skip find()'s redundant O(row_nnz) re-scan via
        // at_index() instead).
        // Persistent scratch (Block4Store::scratch_*), not fresh vectors
        // every call -- see forward's identical comment above and
        // block4.hpp: batch=1 real-time calls can't amortize repeated
        // heap allocation of these the way a large training batch could.
        std::vector<uint32_t>&    tile_br   = weights.block4.scratch_tile_br;
        std::vector<uint32_t>&    tile_bc   = weights.block4.scratch_tile_bc;
        std::vector<std::size_t>& tile_elem = weights.block4.scratch_tile_elem;
        const std::size_t n_b4 = weights.block4.n_tiles();
        // resize()+direct indexing, not reserve()+push_back() -- see
        // forward's identical comment above (real, measured cost:
        // push_back's per-call capacity check across ~3*n_tiles calls,
        // independent of whether the backing allocation is fresh or
        // reused).
        tile_br.resize(n_b4);
        tile_bc.resize(n_b4);
        tile_elem.resize(n_b4);

        // Per-row slot count across ALL block4 tiles touching that row
        // (not just one tile), needed for both lr_per_row_nnz and the
        // unconditional scale_eff_lr normalization. Every tile contributes
        // exactly BLOCK4_TILE slots per row it covers -- dense, no
        // per-slot scan needed (see block4.hpp: a live tile's slots are
        // all real synapses, weight=0.0 included). Computed directly per
        // block-row here (n_bc * BLOCK4_TILE, known before the tile loop
        // even starts) instead of a separate second pass re-walking
        // tile_br after the fact -- same result, one fewer O(n_tiles) pass.
        std::vector<uint32_t>& row_live_count = weights.block4.scratch_row_live_count;
        row_live_count.assign(n_in, 0);
        const auto& BL4 = weights.block4.block_layout;
        std::size_t ti = 0;
        for (std::size_t br = 0; br < BL4.rows; ++br) {
            const std::size_t n_bc = BL4.row_nnz(br);
            if (n_bc == 0) continue;
            const uint32_t row_count = uint32_t(n_bc) * BLOCK4_TILE;
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                const std::size_t row = br * BLOCK4_TILE + li;
                if (row < n_in) row_live_count[row] = row_count;
            }
            auto bc_cursor = weights.block4.row_cursor(br);
            std::size_t elem_pos = BL4.elem_start[br];
            for (std::size_t bk = 0; bk < n_bc; ++bk, ++elem_pos, ++ti) {
                tile_br[ti] = uint32_t(br);
                tile_bc[ti] = bc_cursor.advance();
                tile_elem[ti] = elem_pos;
            }
        }

        std::vector<double>& t_row_grad = weights.block4.scratch_row_grad;
        t_row_grad.assign(static_cast<std::size_t>(num_cpus) * n_in, 0.0);

        // Hoisted out of the loop condition below -- see forward's
        // identical fix/comment (block4.hpp): re-evaluating
        // tile_br.size() every iteration instead of once measured as a
        // real, large cost (8.78% of forward's total instruction count
        // at 100% density). n_b4 (already computed above) is exactly
        // this same value, so no extra call needed either way.
        const int64_t n_tiles_local = int64_t(n_b4);
        #pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mdx  = t_dx.data() + static_cast<std::size_t>(tid) * dst;
            value_type* mcol = t_col_grad.data() + static_cast<std::size_t>(tid) * n_out;
            double* mrow = t_row_grad.data() + static_cast<std::size_t>(tid) * n_in;

            #pragma omp for schedule(static)
            for (int64_t ti = 0; ti < n_tiles_local; ++ti) {
                const uint32_t br = tile_br[std::size_t(ti)], bc = tile_bc[std::size_t(ti)];
                // Non-const: this tile is both read (precompute) and
                // written (writeback) below, spanning this whole
                // iteration -- one handle, unpacks once if sparse,
                // re-packs once (if touched) when it goes out of scope
                // at the end of this iteration. at_index(): skip find()'s
                // redundant re-scan, see collection loop's comment above.
                // Safe across a sparse->dense re-pack mid-loop (this
                // iteration's own tile, in its own destructor) because
                // Block4StoredTile is fixed-size -- repacking never moves
                // ANY tile's slot, so every OTHER iteration's elem_pos
                // stays valid regardless of what this one does.
                auto tile = weights.block4.at_index(br, bc, tile_elem[std::size_t(ti)]);
                // Resolved ONCE per tile for the read-only decode below
                // instead of once per .at() call (16 calls/tile
                // otherwise) -- see forward's identical hoist and
                // Block4TileHandle::raw_data()'s comment. Only valid for
                // READS; the write-back further down still goes through
                // tile.at(...) = ..., which marks the handle dirty.
                const uint8_t* tdata = tile.raw_data();

                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                    const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                    if (row >= n_in) continue;
                    const uint32_t nnz_row = row_live_count[row];
                    if (nnz_row == 0) continue;
                    const value_type val_scale = weights.get_value_scale(row);
                    const value_type imp_scale = weights.get_importance_scale(row);
                    const value_type effective_lr = lr_per_row_nnz
                        ? learning_rate / static_cast<value_type>(nnz_row)
                        : learning_rate;

                    // Decode this row's whole 4-wide column vector ONCE,
                    // outside the batch loop: table lookups and the
                    // get_output_scale()/get_output_importance_scale()
                    // calls (both branch on col bounds) block
                    // auto-vectorization if left inside it, and none
                    // depend on b. The batch loop itself can't vectorize
                    // across b -- cw4/ci4 carry a genuine sequential
                    // per-sample update across b (online SGD within the
                    // call, not a reduction) -- but each of the 4 COLUMNS
                    // is independent of the others, so the inner loop over
                    // lj at each b is a real, branch-free, gather-free
                    // 4-wide vectorization target instead.
                    // Decode both the weight and importance codes for all
                    // BLOCK4_TILE columns via plain FP4_TABLE[code] lookups,
                    // NOT block4_vec_decode_fp4's SIMD bit-shift formula.
                    // NOT a "SIMD loses to scalar" finding -- that was
                    // checked properly and is FALSE: fp4_decode_bits()
                    // (the true scalar equivalent of the SIMD formula,
                    // same bit-shift algorithm, just unvectorized) is
                    // measurably the WORST of the three options here
                    // (~1.6x slower than either alternative in the full
                    // backward benchmark), confirming SIMD genuinely beats
                    // scalar bit-shift decode, consistent with this
                    // codebase's earlier documented finding
                    // (TODO_DUAL_BLOCK4.md) and not contradicted by
                    // anything here. What DOES win, measured in the real
                    // disldo_backward benchmark (not just an isolated
                    // microbenchmark, which misleadingly suggested a
                    // bigger and differently-shaped effect than the real
                    // one -- see TODO_DUAL_BLOCK4.md's Part C) is
                    // FP4_TABLE[code] specifically -- a different decode
                    // ALGORITHM (branchless array lookup vs bit-field
                    // reconstruction), not a SIMD-vs-scalar swap. Real,
                    // reproducible ~6% win over the SIMD bit-shift version
                    // for backward specifically (3 repeats each, clean
                    // non-overlapping ranges); forward showed the OPPOSITE
                    // (SIMD bit-shift wins there, kept as-is above) --
                    // the two functions' surrounding code apparently
                    // interacts with this choice differently enough that
                    // the same swap doesn't transfer between them.
                    // Unconditional, before the per-lj bounds check below,
                    // since decoding an out-of-bounds column's byte is
                    // harmless (the branch below discards it).
                    const uint8_t byte0 = tdata[Block4Tile::slot_index(li, 0)], byte1 = tdata[Block4Tile::slot_index(li, 1)],
                                  byte2 = tdata[Block4Tile::slot_index(li, 2)], byte3 = tdata[Block4Tile::slot_index(li, 3)];
                    const value_type w_decoded_arr[BLOCK4_TILE]   = {FP4_TABLE[byte0 & 0xFu], FP4_TABLE[byte1 & 0xFu],
                                                                       FP4_TABLE[byte2 & 0xFu], FP4_TABLE[byte3 & 0xFu]};
                    const value_type imp_decoded_arr[BLOCK4_TILE] = {FP4_TABLE[(byte0 >> 4) & 0xFu], FP4_TABLE[(byte1 >> 4) & 0xFu],
                                                                       FP4_TABLE[(byte2 >> 4) & 0xFu], FP4_TABLE[(byte3 >> 4) & 0xFu]};

                    std::size_t col4[BLOCK4_TILE];
                    bool        col_valid4[BLOCK4_TILE];
                    value_type  out_scale4[BLOCK4_TILE];
                    value_type  combined_scale4[BLOCK4_TILE], combined_imp_scale4[BLOCK4_TILE];
                    value_type  cw4[BLOCK4_TILE], ci4[BLOCK4_TILE], cw_orig4[BLOCK4_TILE];
                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                        const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                        col_valid4[lj] = col < n_out;
                        if (!col_valid4[lj]) {
                            col4[lj] = 0;
                            out_scale4[lj] = combined_scale4[lj] = combined_imp_scale4[lj] = value_type(0);
                            cw4[lj] = ci4[lj] = cw_orig4[lj] = value_type(0);
                            continue;
                        }
                        col4[lj] = col;
                        // Every slot is a real synapse, weight=0.0 included
                        // -- see block4.hpp -- so every slot gets a real
                        // gradient update every call, no liveness check.
                        out_scale4[lj] = weights.get_output_scale(col);
                        const value_type out_imp_scale = weights.get_output_importance_scale(col);
                        combined_scale4[lj]     = val_scale * out_scale4[lj];
                        combined_imp_scale4[lj] = imp_scale * out_imp_scale;
                        cw_orig4[lj] = w_decoded_arr[lj];
                        cw4[lj] = cw_orig4[lj] * combined_scale4[lj];          // -> true units
                        ci4[lj] = imp_decoded_arr[lj] * combined_imp_scale4[lj];
                    }

                    // mcol[col4[lj]] is a real 4-way SCATTER (4 distinct
                    // output columns) if written every (b, lj) -- confirmed
                    // via -fopt-info-vec as the actual remaining blocker
                    // once the FP4_TABLE lookups above were already moved
                    // out of this loop (a gather was never the issue here
                    // once that happened). Accumulate into a small local
                    // array across the whole batch loop instead -- pure
                    // register/stack traffic, no memory scatter -- and
                    // flush the 4 (not 4*batch) real scatter writes once,
                    // after. mrow[row] isn't a scatter (same address for
                    // every lj already, a horizontal reduction), but gets
                    // the same local-accumulate-then-flush treatment for
                    // consistency and to keep it out of the hot loop too.
                    value_type mcol4[BLOCK4_TILE] = {0};
                    double     mrow_local = 0.0;
                    // col4[lj] is really just col_base+lj (contiguous), but
                    // reading it back OUT of the array hides that from GCC
                    // -- confirmed via -fopt-info-vec: it correctly proved
                    // the loop below vectorizable via SLP, then rejected it
                    // as "unprofitable" because indexing output_grad through
                    // col4[lj] looks like a 4-way gather instead of one
                    // contiguous load. Reading output_grad[...+col_base+lj]
                    // directly (a plain affine index in the loop variable)
                    // lets it see the load is contiguous. Only valid when
                    // the whole tile-column is in bounds (true for every
                    // tile except possibly the last, boundary one) --
                    // that's the split below, checked once per tile, not
                    // per batch element.
                    const std::size_t col_base = std::size_t(bc) * BLOCK4_TILE;
                    const bool full_tile_cols = (col_base + BLOCK4_TILE <= n_out);
                    // block4 is FP4-specific and value_type is float in
                    // every real instantiation (FP4BiPacked and
                    // DeltaCSRBiValues<float> both use it; nothing in this
                    // codebase ever instantiates DeltaCSRBiValues<double>)
                    // -- but this function isn't itself gated behind
                    // is_same_v<VALUES_TYPE, FP4BiPacked>, so it must still
                    // COMPILE generically. Block4Vec is float-only by
                    // design (see block4.hpp), so the real SIMD path is
                    // guarded here and a plain scalar fallback (identical
                    // math, matches the pre-SIMD version already verified
                    // correct) covers any hypothetical non-float
                    // instantiation instead of silently miscompiling one.
                    if constexpr (std::is_same_v<value_type, float> && !SILI_BLOCK4_FORCE_SCALAR_BACKWARD) {
                        if (full_tile_cols) {
                            const Block4Vec effective_lr_v = block4_vec_broadcast(effective_lr);
                            const Block4Vec val_scale_v    = block4_vec_broadcast(val_scale);
                            const Block4Vec one_v          = block4_vec_broadcast(1.0f);
                            Block4Vec cw_v        = block4_vec_load(cw4);
                            Block4Vec ci_v         = block4_vec_load(ci4);
                            const Block4Vec cw_orig_v      = block4_vec_load(cw_orig4);
                            const Block4Vec out_scale_v    = block4_vec_load(out_scale4);
                            Block4Vec mcol_acc_v   = block4_vec_broadcast(0.0f);
                            const bool training = (learning_rate != value_type(0));
                            for (SIZE_TYPE b = 0; b < batch; ++b) {
                                const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                                value_type* mdx_row = mdx + static_cast<std::size_t>(b) * in_cols + row;
                                const Block4Vec dyv_v = block4_vec_load(
                                    output_grad + static_cast<std::size_t>(b) * n_out + col_base);
                                const Block4Vec g_v = dyv_v * block4_vec_broadcast(iv);
                                if (training) {
                                    ci_v -= g_v * effective_lr_v;
                                    const Block4Vec neg_lr_g_v = -(effective_lr_v * g_v);
                                    const Block4Vec delta_v = damp_by_importance
                                        ? neg_lr_g_v / (one_v + block4_vec_abs(ci_v))
                                        : neg_lr_g_v;
                                    cw_v += delta_v;
                                    // mrow_local accumulates in DOUBLE, one
                                    // horizontal-sum per b -- matches the
                                    // pre-SIMD code's own double-precision
                                    // accumulation exactly (unlike mcol,
                                    // which was already float-precision in
                                    // the original scalar code, so
                                    // accumulating it as a Block4Vec across
                                    // the whole loop is not a regression).
                                    // A real, confirmed-not-hypothetical
                                    // issue: accumulating mrow_local in
                                    // float across a whole batch loop
                                    // measurably changed which growth/
                                    // pruning decisions this codebase's
                                    // stochastic FP4 rounding makes over
                                    // many cycles (0 tiles ended up promoted
                                    // in a real 512x512 growth run instead
                                    // of the ~250 expected) -- precision
                                    // here isn't cosmetic.
                                    mrow_local += static_cast<double>(block4_vec_hsum(cw_orig_v * out_scale_v * g_v));
                                    mcol_acc_v += cw_orig_v * val_scale_v * g_v;
                                }
                                *mdx_row += block4_vec_hsum(cw_v * dyv_v);
                            }
                            block4_vec_store(cw4, cw_v);
                            block4_vec_store(ci4, ci_v);
                            block4_vec_store(mcol4, mcol_acc_v);
                        } else {
                            // Boundary tile-column (rare -- only the last
                            // one, when n_out isn't a multiple of
                            // BLOCK4_TILE): scalar bounds-checked fallback,
                            // not on the fast path, doesn't need SIMD.
                            for (SIZE_TYPE b = 0; b < batch; ++b) {
                                const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                                value_type* mdx_row = mdx + static_cast<std::size_t>(b) * in_cols + row;
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    if (!col_valid4[lj]) continue;
                                    const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col4[lj]];
                                    const value_type g   = dyv * iv;
                                    if (learning_rate != value_type(0)) {
                                        ci4[lj] -= g * effective_lr;
                                        cw4[lj] += damp_by_importance
                                            ? (-effective_lr * g) / (value_type(1) + std::abs(ci4[lj]))
                                            : (-effective_lr * g);
                                        mrow_local += static_cast<double>(cw_orig4[lj]) * static_cast<double>(out_scale4[lj]) * g;
                                        mcol4[lj] += cw_orig4[lj] * val_scale * g;
                                    }
                                    *mdx_row += cw4[lj] * dyv;
                                }
                            }
                        }
                    } else {
                        // Hypothetical non-float value_type (never actually
                        // instantiated in this codebase -- see the comment
                        // above): the bounds-checked array form, correct
                        // for both the full-tile and boundary cases via
                        // col_valid4 either way, so no full_tile_cols split
                        // needed here at all.
                        for (SIZE_TYPE b = 0; b < batch; ++b) {
                            const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                            value_type* mdx_row = mdx + static_cast<std::size_t>(b) * in_cols + row;
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                if (!col_valid4[lj]) continue;
                                const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col4[lj]];
                                const value_type g   = dyv * iv;
                                if (learning_rate != value_type(0)) {
                                    ci4[lj] -= g * effective_lr;
                                    cw4[lj] += damp_by_importance
                                        ? (-effective_lr * g) / (value_type(1) + std::abs(ci4[lj]))
                                        : (-effective_lr * g);
                                    mrow_local += static_cast<double>(cw_orig4[lj]) * static_cast<double>(out_scale4[lj]) * g;
                                    mcol4[lj] += cw_orig4[lj] * val_scale * g;
                                }
                                *mdx_row += cw4[lj] * dyv;
                            }
                        }
                    }
                    if (learning_rate != value_type(0)) {
                        mrow[row] += mrow_local;   // mrow is double* already, no down/up-cast needed
                        if constexpr (std::is_same_v<value_type, float> && !SILI_BLOCK4_FORCE_SCALAR_BACKWARD) {
                            if (full_tile_cols) {
                                // 2 SIMD stochastic-quantize calls (4 lanes
                                // each) instead of 8 scalar
                                // fp4_quantize_stochastic() calls. Safe to
                                // divide unconditionally here (unlike the
                                // scalar fallback's per-lj guard) --
                                // full_tile_cols means every lane is valid,
                                // so combined_scale4/combined_imp_scale4
                                // are never the invalid-lane 0 that would
                                // make this a 0/0 division.
                                const Block4Vec w_to_encode = {
                                    cw4[0] / combined_scale4[0], cw4[1] / combined_scale4[1],
                                    cw4[2] / combined_scale4[2], cw4[3] / combined_scale4[3]};
                                const Block4Vec imp_to_encode = {
                                    ci4[0] / combined_imp_scale4[0], ci4[1] / combined_imp_scale4[1],
                                    ci4[2] / combined_imp_scale4[2], ci4[3] / combined_imp_scale4[3]};
                                const Block4VecU new_w_codes   = block4_vec_quantize_stochastic_fp4(w_to_encode);
                                const Block4VecU new_imp_codes = block4_vec_quantize_stochastic_fp4(imp_to_encode);
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    mcol[col4[lj]] += mcol4[lj];
                                    tile.at(li, lj) = uint8_t((new_imp_codes[lj] << 4) | new_w_codes[lj]);
                                }
                            } else {
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    if (!col_valid4[lj]) continue;
                                    mcol[col4[lj]] += mcol4[lj];
                                    const uint8_t new_w   = fp4_quantize_stochastic(cw4[lj] / combined_scale4[lj]);
                                    const uint8_t new_imp = fp4_quantize_stochastic(ci4[lj] / combined_imp_scale4[lj]);
                                    tile.at(li, lj) = uint8_t((new_imp << 4) | new_w);
                                }
                            }
                        } else {
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                if (!col_valid4[lj]) continue;
                                mcol[col4[lj]] += mcol4[lj];
                                const uint8_t new_w   = fp4_quantize_stochastic(cw4[lj] / combined_scale4[lj]);
                                const uint8_t new_imp = fp4_quantize_stochastic(ci4[lj] / combined_imp_scale4[lj]);
                                tile.at(li, lj) = uint8_t((new_imp << 4) | new_w);
                            }
                        }
                    }
                }
            }
        }

        if (learning_rate != value_type(0)) {
            for (std::size_t row = 0; row < n_in; ++row) {
                const uint32_t nnz_row = row_live_count[row];
                if (nnz_row == 0) continue;
                double sum = 0.0;
                for (int t = 0; t < num_cpus; ++t)
                    sum += t_row_grad[static_cast<std::size_t>(t) * n_in + row];
                if (sum == 0.0) continue;
                const value_type scale_eff_lr = learning_rate / static_cast<value_type>(nnz_row);
                const value_type raw_update = static_cast<value_type>(scale_eff_lr * sum);
                weights.value_scale_importance[row] -= raw_update;
                const value_type vs_imp = weights.value_scale_importance[row];
                weights.value_scale[row] -= raw_update / (value_type(1) + std::abs(vs_imp));
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
