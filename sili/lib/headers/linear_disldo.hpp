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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── DISLDO: Dense Input, Sparse Linear, Dense Output ─────────────────────────
//
// Generic over VALUES_TYPE via ValueAccessor -- works identically for
// FP4BiPacked (default, 4-bit) and DeltaCSRBiValues<float> (32-bit fallback),
// matching sisldo_forward (the SISLDO/sparse-input forward equivalent
// in sisldo_ops.hpp) and delta_csr_synap_row_step / delta_csr_build_probes,
// which already use this same pattern.
//
// Supersedes the previous float32/absolute-CSR disldo_forward/disldo_backward
// (which never used DeltaCSRLayout/FP4BiPacked at all -- see conversation).
// Dense-input walk is embarrassingly parallel by input row, unlike the
// sparse-input SISLDO path which needs a work-offset table to balance
// threads across a variable-density CSR batch.

// ── forward ───────────────────────────────────────────────────────────────────

/**
 * @brief Dense-input forward pass. Pure computation, no side effects.
 *
 * @param input          [batch x in_cols] row-major dense.
 * @param batch, in_cols Input dimensions.
 * @param weights        Layer state (read-only here).
 * @param output         [batch x out_cols] accumulated into (caller zeroes first).
 * @param num_cpus       OpenMP thread count.
 *
 * No learning_rate parameter -- forward used to run its own gradient-free
 * ADSP-style (Activity-Dependent Structural Plasticity) importance update
 * whenever a nonzero learning_rate was passed, independently of whether a
 * matching backward_dense() call would ever follow. Not Hebbian learning
 * (no VALUE changes here, only the importance/wiring-strength signal --
 * closer to activity-driven synaptic sprouting/pruning than to a weight
 * update). Confirmed as a real footgun (traced directly: fired on every
 * forward call including ones with no corresponding gradient, e.g. every
 * non-query tick of an online RNN, measurably corrupting training at low
 * learning rates independent of any real task signal). Importance is now
 * updated ONLY by disldo_backward(), coupled to a real gradient, same
 * principle as weight updates always having been backward-only. REMOVED,
 * not just disabled -- a caller that still wants an unconditional
 * activity-correlation signal should build that explicitly, not get it
 * silently bundled into every forward pass.
 *
 * NOTE (test): output must equal the dense matmul input @ W_dense where
 * W_dense[r,c] = weight of synapse (r->c). Same reference check used
 * for sisldo_forward and for this session's standalone disldo_ops.hpp
 * (see conversation) -- both passed it.
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void disldo_forward(
    const typename ValueAccessor<VALUES_TYPE>::value_type* input,
    SIZE_TYPE    batch,
    SIZE_TYPE    in_cols,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    typename ValueAccessor<VALUES_TYPE>::value_type* output,
    int          num_cpus = 4)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;
    const auto& L  = dc.layout;

    const std::size_t n_in  = L.rows;
    const std::size_t n_out = L.cols;
    const std::size_t ost   = static_cast<std::size_t>(batch) * n_out;

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

        #pragma omp for schedule(static)
        for (std::size_t r = 0; r < n_in; ++r) {
            const std::size_t n_row = L.row_nnz(r);
            if (n_row == 0) continue;

            auto cursor = dc.row_cursor(r);
            for (std::size_t e = 0; e < n_row; ++e) {
                const COL_TYPE    col = cursor.advance();
                const std::size_t vb  = L.elem_start[r] + e;
                const value_type  w_stored = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                // Rank-N scale (see scale_rank's own docstring) -- reduces
                // to the exact original val_scale*out_scale at scale_rank==1.
                const value_type  w = w_stored * weights.get_scale(r, col);   // -> true units

                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                    if (iv == value_type(0)) continue;
                    const value_type contrib = w * iv;
                    mo[static_cast<std::size_t>(b) * n_out + col] += contrib;
                }
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
    if (weights.block4.n_tiles() > 0) {
        // Row-major cursor walk isn't parallel-for-friendly directly (same
        // reason the old hash-map iteration wasn't); collect (br,bc,elem_pos)
        // TRIPLES once per call (not Block4Tile pointers/handles -- a
        // handle can't be pre-collected across the parallel region since
        // it's move-only, RAII, and per-tile compress/decompress decisions
        // must happen within ONE thread's ownership of ONE tile at a time;
        // each thread constructs its own handle fresh, inside the loop
        // body below, from these coordinates). elem_pos is the tile's
        // index into block4's own address space (this walk already knows
        // it -- block_layout.elem_start[br]+bk -- for free); byte_pos is
        // its position in block4's flat variable-length tile_data buffer
        // (see block4.hpp -- a tile's byte length varies with its live
        // count when sparse, so unlike elem_pos this can't be derived by
        // simple arithmetic, only by walking the row and summing each
        // preceding tile's real length, which this collection loop is
        // already doing). Both are passed to Block4Store::at_index() so
        // the hot loop below doesn't redo an O(row_nnz) coordinate
        // re-scan per tile via find(). That redundant second scan
        // (discover a tile here, then re-discover it again via find()'s
        // own raw_find()) measured as the dominant real cost of this loop
        // at batch=1 -- see conversation: batch=1 has too little per-tile
        // compute (16 FLOPs) to amortize even one such scan, let alone two.
        // Persistent scratch (Block4Store::scratch_tile_br/bc/elem/byte),
        // not a fresh vector every call -- see block4.hpp: batch=1
        // real-time calls can't amortize repeated heap allocation of
        // these the way a large training batch could.
        std::vector<uint32_t>&    tile_br   = weights.block4.scratch_tile_br;
        std::vector<uint32_t>&    tile_bc   = weights.block4.scratch_tile_bc;
        std::vector<std::size_t>& tile_elem = weights.block4.scratch_tile_elem;
        std::vector<std::size_t>& tile_byte = weights.block4.scratch_tile_byte;
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
        tile_byte.resize(n_b4);
        const auto& BL4 = weights.block4.block_layout;
        std::size_t ti = 0;
        for (std::size_t br = 0; br < BL4.rows; ++br) {
            const std::size_t n_bc = BL4.row_nnz(br);
            if (n_bc == 0) continue;
            auto bc_cursor = weights.block4.row_cursor(br);
            std::size_t elem_pos = BL4.elem_start[br];
            std::size_t byte_pos = weights.block4.tile_byte_start[br];
            for (std::size_t bk = 0; bk < n_bc; ++bk, ++elem_pos, ++ti) {
                tile_br[ti] = uint32_t(br);
                tile_bc[ti] = bc_cursor.advance();
                tile_elem[ti] = elem_pos;
                tile_byte[ti] = byte_pos;
                byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
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
                const auto tile = weights.block4.at_index(br, bc, tile_elem[std::size_t(ti)], tile_byte[std::size_t(ti)]);
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

                    // FP8 dispatch: Block4Tile8's layout is a full byte/
                    // slot (no nibble mask) and decodes via E4M3
                    // (fp8quant.hpp/block4_vec_decode_fp8), not FP4's
                    // table-driven bit-shift codec -- the ONLY thing that
                    // differs from the FP4 branch below; everything past
                    // this point (row-scale multiply, batch accumulation)
                    // is identical float32 math regardless of storage
                    // width, per direct instruction. FP4 branch is
                    // byte-for-byte the pre-existing code, untouched.
                    Block4Vec w_decoded;
                    if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                        const Block4VecU w_codes = {uint32_t(tdata[Block4Tile8::slot_index(0, LJ)]), uint32_t(tdata[Block4Tile8::slot_index(1, LJ)]),
                                                     uint32_t(tdata[Block4Tile8::slot_index(2, LJ)]), uint32_t(tdata[Block4Tile8::slot_index(3, LJ)])};
                        w_decoded = block4_vec_decode_fp8(w_codes);
                    } else {
                        const Block4VecU w_codes = {uint32_t(tdata[Block4Tile::slot_index(0, LJ)] & 0xFu), uint32_t(tdata[Block4Tile::slot_index(1, LJ)] & 0xFu),
                                                     uint32_t(tdata[Block4Tile::slot_index(2, LJ)] & 0xFu), uint32_t(tdata[Block4Tile::slot_index(3, LJ)] & 0xFu)};
                        w_decoded = block4_vec_decode_fp4(w_codes);
                    }
                    const value_type w_decoded_arr[BLOCK4_TILE] = {value_type(w_decoded[0]), value_type(w_decoded[1]),
                                                                     value_type(w_decoded[2]), value_type(w_decoded[3])};

                    value_type w4[BLOCK4_TILE];
                    std::size_t row_idx[BLOCK4_TILE];
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                        if (row < n_in) {
                            // Rank-N scale (see scale_rank's own
                            // docstring) -- reduces to the exact original
                            // val_scale*out_scale at scale_rank==1.
                            w4[li] = w_decoded_arr[li] * weights.get_scale(row, col);
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

    // output_scale/value_scale importance are no longer touched here --
    // see this function's own docstring: forward is pure now, importance
    // updates only ever happen in disldo_backward, coupled to a real
    // gradient.

    // ── AQRS additive branch (task #276, gamma wired in at task #289,
    // see sili_peridot/AQRS_DESIGN.md) ── A[row,col] = sum_k gamma_k *
    // additive_u_k(row,k)*additive_v_k(col,k), ADDED (not Hadamard
    // -multiplied against quant like the scale_rank branch above) to
    // the effective weight. Genuinely independent of the sparse/block4
    // structure above -- it's a dense low-rank correction that touches
    // every output regardless of which synapses happen to be live, so
    // it's computed as its own small pass rather than woven into either
    // per-synapse loop. Fused per Theorem 11 (never materializes the
    // n_in x n_out A matrix): project the input down to additive_rank
    // dimensions via additive_u, then back up to n_out via additive_v --
    // O(batch * additive_rank * (n_in + n_out)), cheap as long as
    // additive_rank << min(n_in, n_out). No-op (skipped entirely) at the
    // default additive_rank==0, matching value_scale/output_scale's own
    // "unconfigured component contributes nothing" convention. gamma_k
    // itself defaults to 1.0 (see get_additive_gamma_k's own docstring)
    // so this multiply is transparent for every caller that's never
    // touched gamma.
    if (weights.additive_rank > 0) {
        std::vector<value_type> proj(static_cast<std::size_t>(batch) * weights.additive_rank, value_type(0));
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* in_row = input + static_cast<std::size_t>(b) * in_cols;
            value_type* p_row = proj.data() + static_cast<std::size_t>(b) * weights.additive_rank;
            for (std::size_t r = 0; r < n_in; ++r) {
                const value_type iv = in_row[r];
                if (iv == value_type(0)) continue;
                for (std::size_t k = 0; k < weights.additive_rank; ++k)
                    p_row[k] += weights.get_additive_u_k(r, k) * iv;
            }
        }
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* p_row = proj.data() + static_cast<std::size_t>(b) * weights.additive_rank;
            value_type* out_row = output + static_cast<std::size_t>(b) * n_out;
            for (std::size_t c = 0; c < n_out; ++c) {
                value_type acc = value_type(0);
                for (std::size_t k = 0; k < weights.additive_rank; ++k)
                    acc += weights.get_additive_gamma_k(k) * weights.get_additive_v_k(c, k) * p_row[k];
                out_row[c] += acc;
            }
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
 *        divided by (sqrt(ci)+eps), where ci is an RMSprop-style
 *        exponential moving average of g^2 (decayed, magnitude-only) --
 *        a per-synapse adaptive-learning-rate effect, same one extra
 *        scalar of state per synapse this always used, just tracking a
 *        different quantity. When false: the raw (-effective_lr * g)
 *        step is applied directly, with no damping -- ci is still
 *        tracked/updated identically either way (importance stays
 *        meaningful for pruning/synaptogenesis decisions regardless),
 *        only its use as a WEIGHT-UPDATE damping factor is toggled.
 *        Exists specifically so a caller can A/B this mechanism against
 *        itself on the same kernel -- see sili_peridot's/sili__new's
 *        importance-damping-as-optimizer integration test.
 *
 *        REPLACED (see conversation/JOURNAL.md), not just retuned: the
 *        previous formula (ci -= g*effective_lr, an undecayed running
 *        SUM of SIGNED gradient, divide by 1+|ci|) was confirmed via a
 *        real ablation to converge no better than plain SGD -- it has a
 *        structural blind spot where sign-oscillating (noisy) gradient
 *        pressure CANCELS in the sum, so damping barely engages exactly
 *        when it should. A dense-Tensor RMSprop control (one scalar of
 *        state per parameter, decayed g^2, no momentum/second buffer)
 *        reached essentially full-Adam convergence quality on the same
 *        task at the same storage budget -- this formula ports that
 *        result in, still one scalar/synapse, no new storage.
 * @param beta2  Decay rate for ci's g^2 EMA (default 0.999, matching
 *        this project's own AdamOptimizer convention). Only used when
 *        damp_by_importance is true.
 * @param eps    Numerical floor added to sqrt(ci) so a synapse with zero
 *        accumulated gradient magnitude doesn't produce a divide-by-zero
 *        (matches Adam's own eps convention, default 1e-8).
 *
 * NOTE (test): with learning_rate=0, input_grad must equal W_dense^T @ output_grad
 * per batch sample, weights/importance unchanged. Same reference check as
 * delta_csr_backward.
 */
// ScalePolicy / DeferredScaleWrite: swappable value_scale/output_scale
// update, added to compare real update rules against a toy Python
// fake-quantize simulation's closed-form full-layer refit -- see
// ScalePolicy's own docstring (delta_csr_types.hpp) and
// sili_peridot/JOURNAL.md's 2026-08-09 tile-recurrence entries. Both
// parameters default to exactly today's behavior so every existing
// caller (SparseLinearLayer, SparseLinearLayer8, DISLDOLayerV, etc.)
// is unaffected without any changes on their part.
//
// SCOPE: covers the SCATTERED path only (this function's per-row loop
// below and the shared output_scale reduction at the end). Block4's
// own internal value_scale update (further down, inside
// `if (weights.block4.n_tiles() > 0)`) is intentionally left
// untouched by both parameters -- real future work, not started here;
// see [[project_hybrid_precision_plan]] (sili_peridot memory) /
// JOURNAL.md for why this scope was chosen (block4 promotion only
// fires from synaptogenesis, so scattered-only already covers every
// layer that hasn't triggered growth yet).
// StochasticRounding (default true, current behavior): scattered-path
// weight/importance stores use ValueAccessor::set_stochastic (unbiased
// dithered rounding, real per-step noise) when true, or the deterministic
// nearest-neighbour ValueAccessor::set (fp4_quantize/fp8 equivalent, zero
// added noise) when false. Scoped to the scattered path only, same as
// ScalePolicy/DeferredScaleWrite above -- block4's dense-tile SIMD
// quantize_stochastic calls are untouched (real follow-up, not yet needed:
// no toy config here triggers block4 promotion). Built to test whether
// real FP4's per-step dithered rounding, not value_scale staleness, is
// what makes it collapse to chance where a deterministic-rounding fp32
// shadow control (sili_peridot's fixed_digit_residual_quantize /
// TrueMultiDigitLayer's simulate_quantize) succeeds by a wide margin --
// see sili_peridot/JOURNAL.md.
// SynapsePolicy: swappable per-synapse `ci` update (update_ci) + weight-delta
// computation (update_cw) -- see PlainRMSpropSynapsePolicy/
// BoundedRMSpropSynapsePolicy's own docstrings (delta_csr_types.hpp) for
// the full root-cause story (late-training RMSprop divergence once ci
// decays down to match a shrinking residual gradient, found via direct
// trace on DISLDOLayer32; an lr-decay schedule "fixes" it by eventually
// freezing the whole network, incompatible with this project's lifelong-
// learning goal, hence a stationary per-synapse floor+clip instead).
// Defaults to BoundedRMSpropSynapsePolicy (see its own docstring,
// delta_csr_types.hpp, for the tuned production defaults: max_abs_delta,
// max_ci real; min_decay_frac a true no-op). PlainRMSpropSynapsePolicy
// remains available for explicit opt-in and as the bit-identical
// reference the fix was checked against. min_decay_frac/max_abs_delta/
// max_ci are all inert (ignored) under Plain; all three only matter once
// a caller is on BoundedRMSpropSynapsePolicy.
//
// Template-TEMPLATE parameter (not a fully-instantiated `typename
// SynapsePolicy = Policy<value_type>` like ScalePolicy above), because
// this function's SIMD sites need the SAME chosen policy (Plain or
// Bounded) instantiated at Block4Vec, not just value_type -- one caller-
// visible choice, two internal instantiations (`SynapsePolicy` and
// `SynapsePolicyVec` aliases below), rather than requiring every caller
// to specify the policy twice. Placed LAST in the template parameter
// list, not next to ScalePolicy -- existing callers across the test
// suite explicitly specify template args positionally up through
// DeferredScaleWrite/StochasticRounding (e.g.
// `disldo_backward<S, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>,
// false, false>`); inserting a new parameter anywhere before those would
// silently shift every one of those positional args onto the wrong
// parameter (confirmed directly: doing this broke the build with
// "expected a class template, got 'false'").
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t,
          typename ScalePolicy = RMSpropScalePolicy<typename ValueAccessor<VALUES_TYPE>::value_type>,
          bool DeferredScaleWrite = false, bool StochasticRounding = true,
          template <typename> class SynapsePolicyT = BoundedRMSpropSynapsePolicy>
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
    bool         damp_by_importance = true,
    typename ValueAccessor<VALUES_TYPE>::value_type  beta2 = 0.999f,
    typename ValueAccessor<VALUES_TYPE>::value_type  eps = 1e-8f,
    typename ValueAccessor<VALUES_TYPE>::value_type  beta1 = 0.9f,
    typename ValueAccessor<VALUES_TYPE>::value_type  min_decay_frac = 0.0f,
    typename ValueAccessor<VALUES_TYPE>::value_type  max_abs_delta = 1e30f,
    typename ValueAccessor<VALUES_TYPE>::value_type  max_ci = 1e30f,
    typename ValueAccessor<VALUES_TYPE>::value_type  zero_escape_eps = 0.1f,
    bool         scale_invariant = false,
    // AQRS gamma's L1 penalty coefficient (task #273/#283, Theorem 8) --
    // 0 (default) disables it entirely, matching every existing call
    // site's behavior unchanged. Appended LAST, after scale_invariant, so
    // no existing positional call site anywhere in the codebase shifts --
    // see this function's own template-parameter docstring above for why
    // that ordering discipline matters here specifically.
    typename ValueAccessor<VALUES_TYPE>::value_type  l1_coef = 0.0f)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    // Same chosen policy (SynapsePolicyT), instantiated at both widths --
    // scalar sites use SynapsePolicy, Block4Vec SIMD sites use
    // SynapsePolicyVec. See this function's own template-parameter
    // docstring above for why SynapsePolicyT is a template-template
    // parameter rather than a fully-instantiated type like ScalePolicy.
    using SynapsePolicy = SynapsePolicyT<value_type>;
    using SynapsePolicyVec = SynapsePolicyT<Block4Vec>;
    // Only meaningful when DeferredScaleWrite -- a touched scattered
    // entry's true-units (cw, ci) get cached here instead of stored
    // immediately, and are written out via set_stochastic only after
    // BOTH value_scale[row] and output_scale[col] are finalized for
    // this whole call (output_scale's own reduction runs last, after
    // every row AND after block4 -- see the end of this function).
    struct DeferredScaleWriteEntry {
        std::size_t vb;
        value_type  cw;
        value_type  ci;
        std::size_t row;
        COL_TYPE    col;
    };
    std::vector<std::vector<DeferredScaleWriteEntry>> t_deferred;
    if constexpr (DeferredScaleWrite) t_deferred.resize(static_cast<std::size_t>(num_cpus));
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

    // Previously an early return here when both storages were empty --
    // removed: the dead-row value_scale bootstrap pass (near the end of
    // this function) needs to run precisely in that case (a genuinely
    // fresh zero-synapse-init layer), so this can no longer be a pure
    // early-exit. Everything below is already independently guarded
    // (`if (!dc.empty())`, `if (weights.block4.n_tiles() > 0)`) and
    // degrades to safe no-ops when both are empty -- this was an
    // optimization for the doubly-empty case, not a correctness
    // requirement, so removing it only costs a little extra harmless
    // work in that case, not correctness.
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
    const std::size_t rank = weights.scale_rank;
    // Persistent per-instance heap scratch (task #295) backing block4's
    // SIMD backward path's per-rank-component accumulators -- grows
    // (never shrinks automatically) to fit `rank`/`num_cpus`, a cheap
    // no-op once already large enough. See ScaleRankScratch's own
    // docstring, delta_csr_types.hpp, for the full rationale (replaces
    // the old compile-time SCALE_RANK_MAX=4 stack-array cap).
    weights.scale_rank_scratch.ensure(static_cast<std::size_t>(num_cpus), rank, BLOCK4_TILE);
    // Thin 2D view over a flat scratch buffer -- op[k] returns a
    // value_type* row pointer, exactly matching the syntax (and the
    // block4_vec_load/store call convention) the old fixed-size 2D
    // stack arrays (out_scale_k4[SCALE_RANK_MAX][BLOCK4_TILE] etc.) used,
    // so every existing use site below is a drop-in replacement (task
    // #295).
    struct Flat2DView {
        value_type* base; std::size_t stride;
        inline value_type* operator[](std::size_t k) const { return base + k * stride; }
    };
    // AQRS gamma (task #273/#283): layer-wide, doesn't vary by row/col/
    // tile, so fetched ONCE here rather than per-row like value_scale_k/
    // output_scale_k -- shared by the scattered loop below AND every
    // block4 sub-path (FP4 SIMD/scalar, FP8 SIMD/scalar) further down.
    // Deliberately NOT baked into value_scale_k8/out_scale_k4 (etc.)'s own
    // local caches -- gamma's OWN gradient needs the PURE (un-multiplied)
    // value_direction_k*output_direction_k product, and baking gamma into
    // either side would make recovering that require dividing by gamma_k,
    // fragile exactly at gamma_k=0 (the most common case, since gamma
    // defaults to 0 for any channel beyond the always-on k=0). Applied as
    // an explicit extra factor at each of value_scale's/output_scale's own
    // gradient accumulation sites instead, matching the scattered path's
    // own style exactly (see its identical comment above).
    std::vector<value_type> gamma_k_arr(rank);
    for (std::size_t k = 0; k < rank; ++k) gamma_k_arr[k] = weights.get_scale_gamma_k(k);
    std::vector<value_type> t_col_grad(static_cast<std::size_t>(num_cpus) * n_out * rank, value_type(0));
    // Parallel forward-contribution accumulator, same shape/layout as
    // t_col_grad -- mirrors the per-synapse ci fix (contrib=x*w combined
    // additively with g into the RMSprop second moment, see
    // sili__new/lean_proofs/importance_signal_information_gain/
    // SiliImportanceProof/ImportanceSignalInformationGain.lean,
    // Joint.combined_signal_strictly_informative) up one level: value_scale/
    // output_scale's own importance (scale_state) is the SAME kind of
    // RMSprop second-moment accumulator as ci, just aggregated over a row/
    // column instead of a single synapse -- no principled reason for it to
    // skip the same combination.
    std::vector<value_type> t_col_grad_contrib(static_cast<std::size_t>(num_cpus) * n_out * rank, value_type(0));
    const bool output_scale_trainable = weights.output_scale_is_trainable;

    // AQRS gamma's own gradient (task #273/#283): dL/d(gamma_k) = sum over
    // every (row,col) touched this call of quant_floor * value_direction_k
    // (row) * output_direction_k(col) * g -- a LAYER-WIDE scalar per k, not
    // per-row/col like value_scale/output_scale's own gradients, so this is
    // sized num_cpus*rank (not num_cpus*n_out*rank like t_col_grad above).
    std::vector<value_type> t_gamma_grad(static_cast<std::size_t>(num_cpus) * rank, value_type(0));
    // Same combined-signal (g + contrib=x*w) treatment as value_scale/
    // output_scale's own importance above -- see t_col_grad_contrib's own
    // comment for the full rationale.
    std::vector<value_type> t_gamma_grad_contrib(static_cast<std::size_t>(num_cpus) * rank, value_type(0));

    // Pre-size value_scale/output_scale (now n_in*rank / n_out*rank, see
    // scale_rank's own docstring) so that direct indexed writes from
    // within the parallel region are safe (resize would race if called
    // per-thread).
    //
    // CORRECTED (real bug, found via writing test_aqrs_rank_growth_shrink.cpp
    // -- see conversation): a uniform `resize(..., value_type(1))` fill
    // backfills EVERY newly-appended slot with 1.0, not just the k==0 ones
    // -- but get_value_scale_k/get_output_scale_k's own documented default
    // is k==0 -> 1.0, k>=1 -> 0.0 (an untrained extra rank component must
    // be a pure no-op). Confirmed via direct probe: growing scale_rank from
    // 1 to 2 mid-training made the new k=1 channel start IDENTICAL to k=0
    // (both 1.0), so it received identical gradients every step by
    // symmetry and stayed in permanent lockstep with k=0 -- effectively a
    // scaled rank-1, not real rank-2 capacity. Fix: resize with a neutral
    // 0 fill, then explicitly set only the k==0 slots in the newly-added
    // range to 1.0 -- matches reshuffle_rank_array's own scale_default
    // lambda (set_scale_rank, delta_csr_types.hpp), just applied to a
    // straight append instead of a rank-changing reshuffle.
    if (weights.value_scale.size() < n_in * rank) {
        const std::size_t old_size = weights.value_scale.size();
        weights.value_scale.resize(n_in * rank, value_type(0));
        for (std::size_t idx = old_size; idx < weights.value_scale.size(); ++idx)
            if (idx % rank == 0) weights.value_scale[idx] = value_type(1);
    }
    if (weights.output_scale.size() < n_out * rank) {
        const std::size_t old_size = weights.output_scale.size();
        weights.output_scale.resize(n_out * rank, value_type(0));
        for (std::size_t idx = old_size; idx < weights.output_scale.size(); ++idx)
            if (idx % rank == 0) weights.output_scale[idx] = value_type(1);
    }
    if (weights.value_scale_importance.size() < n_in * rank)
        weights.value_scale_importance.resize(n_in * rank, value_type(0));
    if (weights.output_scale_importance.size() < n_out * rank)
        weights.output_scale_importance.resize(n_out * rank, value_type(0));
    if (weights.value_scale_momentum.size() < n_in * rank)
        weights.value_scale_momentum.resize(n_in * rank, value_type(0));
    // value_scale_step was missing from this pre-sizing list -- a real,
    // confirmed bug (found via AddressSanitizer, not guessed): get_value_
    // scale_step_k's own lazy .resize() runs completely unguarded (no
    // lock at all, unlike block4's tile_data), called directly from every
    // row inside the #pragma omp parallel region below. Two threads
    // touching a not-yet-grown index at the same time race on the SAME
    // vector's resize/reallocation -- confirmed as a genuine heap-use-
    // after-free (ASan: thread reading value_scale_step[idx] while
    // another thread's resize() had already freed the old buffer).
    // Pre-sizing here, same as value_scale_importance right above (same
    // n_in*rank shape, same reason), makes the lazy-resize branch in
    // get_value_scale_step_k dead in the normal case.
    if (weights.value_scale_step.size() < n_in * rank)
        weights.value_scale_step.resize(n_in * rank, 0);

    if (!dc.empty()) {
    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid = omp_get_thread_num();
        value_type* mdx  = t_dx.data() + static_cast<std::size_t>(tid) * dst;
        // Per-component now: mcol[col*rank+k]. Indexing helper local to
        // this thread, rank captured from the enclosing scope.
        value_type* mcol_base = t_col_grad.data() + static_cast<std::size_t>(tid) * n_out * rank;
        auto mcol_at = [&](std::size_t col, std::size_t k) -> value_type& { return mcol_base[col * rank + k]; };
        value_type* mcol_contrib_base = t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * n_out * rank;
        auto mcol_at_contrib = [&](std::size_t col, std::size_t k) -> value_type& { return mcol_contrib_base[col * rank + k]; };
        value_type* mgamma_base = t_gamma_grad.data() + static_cast<std::size_t>(tid) * rank;
        auto mgamma_at = [&](std::size_t k) -> value_type& { return mgamma_base[k]; };
        value_type* mgamma_contrib_base = t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * rank;
        auto mgamma_at_contrib = [&](std::size_t k) -> value_type& { return mgamma_contrib_base[k]; };
        [[maybe_unused]] std::vector<DeferredScaleWriteEntry>* mdeferred = nullptr;
        if constexpr (DeferredScaleWrite) mdeferred = &t_deferred[static_cast<std::size_t>(tid)];

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
            // application avoids that. scale_grad_sum (single double) is
            // the DeferredScaleWrite-only path's accumulator (component 0
            // only, see that branch's own scope note); scale_grad_sum_rank
            // is the non-deferred path's per-component accumulator.
            double scale_grad_sum = 0.0;
            std::vector<double> scale_grad_sum_rank(rank, 0.0);
            // Parallel forward-contribution accumulators, mirroring
            // scale_grad_sum/scale_grad_sum_rank -- see t_col_grad_contrib's
            // own comment above (same additive-combination rationale, one
            // level up from per-synapse ci).
            double scale_grad_sum_contrib = 0.0;
            std::vector<double> scale_grad_sum_rank_contrib(rank, 0.0);
            for (std::size_t e = 0; e < nnz_this_row; ++e) {
                const COL_TYPE    col = cursor.advance();
                const std::size_t vb  = L.elem_start[r] + e;
                const value_type  cw_orig = ValueAccessor<VALUES_TYPE>::get_w  (dc.values, vb);
                const value_type  ci_orig = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                const value_type  out_scale     = weights.get_output_scale(col);
                const value_type  combined_scale = val_scale * out_scale;
                // Same row*col combination as the weight's own scale --
                // see the matching comment in disldo_forward.
                const value_type  out_imp_scale      = weights.get_output_importance_scale(col);
                const value_type  combined_imp_scale = imp_scale * out_imp_scale;
                value_type ci  = ci_orig * combined_imp_scale;   // -> true units

                if constexpr (DeferredScaleWrite) {
                    // UNCHANGED, old true-units-round-trip formula --
                    // value_scale[r]/output_scale[col] finalize AFTER
                    // this loop (value_scale right below, output_scale in
                    // the shared reduction at the very end of this
                    // function), so eagerly multiplying the code's own
                    // step by a not-yet-finalized combined_scale (the
                    // "new" formula below) would reintroduce exactly the
                    // staleness DeferredScaleWrite exists to avoid. Needs
                    // its own separate treatment to get the zero-escape
                    // property too -- not done here, so a DeferredScaleWrite
                    // layer (SparseLinearLayerResync et al) does NOT yet
                    // get deterministic-rounding zero-escape; only the
                    // default (non-deferred) SparseLinearLayer/
                    // SparseLinearLayerDeterministic do.
                    // cw_start: FIXED for the whole batch loop below --
                    // matches a real mini-batch optimizer's own contract
                    // (gradient computed against ONE parameter snapshot,
                    // exactly one step taken after). Confirmed as a real
                    // bug this session (see conversation): the previous
                    // version mutated `cw`/`ci` INSIDE this loop, once per
                    // batch row, so row b's own contrib/dx used row
                    // (b-1)'s already-applied update instead of a
                    // consistent snapshot -- b sequential mini-steps
                    // instead of one real step, and even dx (the
                    // gradient flowing to the PREVIOUS layer) was computed
                    // against a moving target. g_agg/contrib_agg are
                    // per-synapse scalar locals (stack, not a matrix-sized
                    // buffer) summed across the batch, mirroring the
                    // torch reference's own `x_sum = x.sum(dim=0)`
                    // aggregation exactly (by linearity, sum_b(iv_b*cw) ==
                    // cw*sum_b(iv_b) since cw_start is constant here).
                    const value_type cw_start = cw_orig * combined_scale;
                    double g_agg = 0.0, contrib_agg = 0.0;
                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        const value_type iv  = input[static_cast<std::size_t>(b) * in_cols + r];
                        const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col];
                        const value_type g   = dyv * iv;
                        if (learning_rate != value_type(0)) {
                            // Additive combination of the backward
                            // sensitivity signal (g = dy*x) with a forward
                            // contribution signal (contrib = x*w): proved to
                            // strictly improve the importance estimate's
                            // information about the synapse's true
                            // importance whenever the two are not
                            // conditionally independent given it -- see
                            // sili__new/lean_proofs/importance_signal_information_gain/
                            // SiliImportanceProof/ImportanceSignalInformationGain.lean,
                            // theorem Joint.combined_signal_strictly_informative
                            // (built on Joint.entropy_le_condEntropy,
                            // H(Θ|X,Y) ≤ H(Θ|X)). Additive, not multiplicative,
                            // so ci still updates from contrib alone even
                            // when g=0 (e.g. a zero-gradient backward call),
                            // instead of reintroducing the old quant=0-forever
                            // deadlock a multiplicative gate would cause.
                            //
                            // SQUARE first, THEN sum -- g^2+contrib^2, not
                            // (g+contrib)^2. ci is the DIVISOR of the weight
                            // update's step size (see cw's own update just
                            // below), so its job is safety: never let the
                            // denominator collapse toward zero while the
                            // numerator (g) stays large. Sum-then-square
                            // fails exactly that: when g and contrib are
                            // large and near-opposite in sign (a real,
                            // common case -- a synapse whose current value
                            // and the task's error signal are pulling
                            // against each other), (g+contrib)^2 can
                            // collapse toward zero even though BOTH signals
                            // are individually large, making the step
                            // -effective_lr*g/(sqrt(ci)+eps) explode --
                            // the same class of instability the value_scale
                            // bias-correction fix elsewhere in this file
                            // closes, just triggered by cancellation
                            // instead of cold start (see conversation).
                            // Square-then-sum is bounded below by
                            // max(g,contrib)^2 regardless of sign, so a
                            // large-magnitude disagreement still damps the
                            // step (correctly -- a synapse under real
                            // contention needs protecting from a jumpy
                            // update, not less of it) instead of amplifying
                            // it. This still captures the additive-signal
                            // information-gain property the Lean proof
                            // establishes (Joint.combined_signal_strictly_
                            // informative, cited above) -- that proof is
                            // about combining the two signals being more
                            // informative than either alone in the
                            // abstract, it does not prescribe sum-before-
                            // square as the numeric encoding.
                            const value_type contrib = iv * cw_start;
                            g_agg += static_cast<double>(g);
                            contrib_agg += static_cast<double>(contrib);
                            scale_grad_sum += static_cast<double>(cw_orig) * static_cast<double>(out_scale) * g;
                            mcol_at(col, 0) += cw_orig * val_scale * g;
                            // Parallel forward-contribution accumulation --
                            // see scale_grad_sum_contrib's own comment above.
                            scale_grad_sum_contrib += static_cast<double>(cw_orig) * static_cast<double>(out_scale) * contrib;
                            mcol_at_contrib(col, 0) += cw_orig * val_scale * contrib;
                        }
                        mdx[static_cast<std::size_t>(b) * in_cols + r] += cw_start * dyv;
                    }
                    value_type cw = cw_start;
                    if (learning_rate != value_type(0)) {
                        // ONE update, using the batch-aggregated g/contrib
                        // -- see cw_start's own comment above for why.
                        ci = SynapsePolicy::update_ci(ci, static_cast<value_type>(g_agg),
                                                      static_cast<value_type>(contrib_agg),
                                                      beta2, min_decay_frac, max_ci);
                        cw += SynapsePolicy::update_cw(static_cast<value_type>(g_agg), ci, value_type(1),
                                                       effective_lr, eps, damp_by_importance, max_abs_delta,
                                                       scale_invariant);
                        // Defer the store until value_scale[r] AND
                        // output_scale[col] are BOTH finalized for this
                        // call -- storing now would use the stale
                        // pre-update scale, exactly the bug this parameter
                        // exists to fix. Stats use the pre-store
                        // true-units `ci` directly as a stand-in for the
                        // post-quantization readback (the real stored
                        // code doesn't exist yet when the write is
                        // deferred) -- a documented approximation,
                        // harmless since these stats are purely
                        // observational and unused by anything in a
                        // no-synaptogenesis training loop.
                        mdeferred->push_back(DeferredScaleWriteEntry{vb, cw, ci, r, col});
                        local_sum_abs_new_i += std::abs(static_cast<double>(ci));
                        local_sum_abs_old_i += std::abs(static_cast<double>(ci_orig));
                        local_sum_sq_new_i  += static_cast<double>(ci) * ci;
                        local_sum_sq_old_i  += static_cast<double>(ci_orig) * ci_orig;
                        local_max_new_i = std::max(local_max_new_i, std::abs(ci));
                    }
                } else {
                    // NEW formula: quant (the stored CODE) is the primary
                    // optimized quantity now, updated DIRECTLY via
                    // dL/d(quant) = g * S(row,col) -- proper chain rule on
                    // true_w = quant * S -- instead of the old true-units
                    // round-trip (cw += ...; new_code = cw / combined_scale),
                    // which DIVIDED by the scale, backwards: made a LARGER
                    // scale SHRINK the per-call code step instead of
                    // growing it. This version is unconditionally nonzero
                    // given S != 0, so quant can escape exactly 0 without
                    // needing stochastic rounding, no matter how small
                    // learning_rate is, given enough persistent-signal
                    // steps (each value_scale component keeps growing
                    // under the same signal, per its own update just
                    // below, giving quant ever more leverage over time).
                    // S = weights.get_scale(r,col), a sum over `rank`
                    // outer-product components (rank=1 reproduces the
                    // exact original val_scale*out_scale).
                    const value_type S = weights.get_scale(r, col);
                    // cw_start: FIXED for the whole batch loop -- see the
                    // DeferredScaleWrite branch's identical cw_start
                    // comment above for the full rationale (real bug this
                    // session: quant/ci/cw were mutated mid-loop, once per
                    // batch row, so g_agg/contrib_agg replace that with
                    // the standard "aggregate over the batch, take ONE
                    // step" contract every real optimizer expects).
                    // quant_floor is likewise computed ONCE from cw_orig
                    // (not the live-updating quant) for the same reason --
                    // still exact/signed for every already-escaped
                    // synapse, matching the zero_escape_eps rationale just
                    // below, now evaluated against the TRUE pre-call state
                    // instead of a value some earlier batch row already
                    // perturbed.
                    const value_type cw_start = cw_orig * S;
                    const value_type quant_floor = (cw_orig == value_type(0)) ? zero_escape_eps : cw_orig;
                    double g_agg = 0.0, contrib_agg = 0.0;
                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        const value_type iv  = input[static_cast<std::size_t>(b) * in_cols + r];
                        const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col];
                        const value_type g   = dyv * iv;
                        if (learning_rate != value_type(0)) {
                            // RMSprop-style: ci = decayed EMA of g^2 (magnitude-only,
                            // recency-weighted), damp by sqrt(ci)+eps instead of 1+|ci|
                            // -- see this function's own docstring for why.
                            //
                            // Additive combination with the forward
                            // contribution signal (contrib = x*w): see the
                            // DeferredScaleWrite branch above (same fix,
                            // same theorem) for the full rationale --
                            // sili__new/lean_proofs/importance_signal_information_gain/
                            // SiliImportanceProof/ImportanceSignalInformationGain.lean,
                            // Joint.combined_signal_strictly_informative.
                            const value_type contrib = iv * cw_start;
                            g_agg += static_cast<double>(g);
                            contrib_agg += static_cast<double>(contrib);
                            // dL/d(value_scale_k(r,k)) = g * quant *
                            // output_scale_k(col,k), for EACH component k
                            // (out_scale_k held fixed within its own term,
                            // holding every OTHER component fixed too --
                            // partial derivative of a sum of independent
                            // outer products). Vanishes EXACTLY at quant=0
                            // for every k simultaneously, which is exactly
                            // the case that needs to escape.
                            //
                            // CORRECTED (real bug, found via conversation --
                            // see conversation for the full trace): the
                            // floor must be GATED on quant==0, not applied
                            // unconditionally as `zero_escape_eps +
                            // |quant|`. The unconditional version silently
                            // discarded quant's sign on EVERY synapse, not
                            // just stuck-at-zero ones -- for any nonzero
                            // quant (the overwhelming majority once
                            // training gets going, including every
                            // non-zero-init synapse from the very first
                            // step), it fed the ALWAYS-POSITIVE quant_floor
                            // into a formula whose correct gradient is
                            // SIGNED. That's not a small epsilon bias, it's
                            // wholesale directional corruption of every
                            // trained synapse's contribution to
                            // value_scale/output_scale's gradient --
                            // confirmed as the root cause of a real,
                            // reproducible failure: zero-init models
                            // produced predictions and eval_acc BIT
                            // -IDENTICAL to a completely untrained model
                            // after 15000 real training steps, energy and
                            // rank-N included (see
                            // sili_peridot/scripts/zeroinit_minimal_repro.py).
                            // Only quant==0 (the genuinely-stuck case,
                            // where the correct gradient truly is zero and
                            // sign is legitimately undefined/free -- this
                            // part of the original reasoning was correct,
                            // just applied too broadly) substitutes the
                            // small positive epsilon; every other quant
                            // value uses itself directly, exact and signed
                            // (quant_floor itself now computed once, fixed,
                            // before this loop -- see its own comment above).
                            for (std::size_t k = 0; k < rank; ++k) {
                                // AQRS gamma (task #273/#283): out_scale_k/
                                // val_scale_k above are pure DIRECTION
                                // (value_scale_k/output_scale_k's getters
                                // are never gamma-baked -- only get_scale()
                                // combines direction*gamma, for the S/
                                // weight-update math elsewhere). Each of
                                // value_scale's and output_scale's OWN
                                // gradient needs an explicit gamma_k factor
                                // (dS/d(value_direction_k)=gamma_k*
                                // output_direction_k, and symmetrically for
                                // output_direction_k) since neither
                                // scale_grad_sum_rank[k] nor mcol_at(col,k)
                                // otherwise sees gamma at all. gamma_k's
                                // OWN gradient (dS/d(gamma_k)=value_
                                // direction_k*output_direction_k) is a NEW,
                                // layer-wide (not per-row/col) accumulator.
                                const value_type out_scale_k = weights.get_output_scale_k(col, k);
                                const value_type val_scale_k = weights.get_value_scale_k(r, k);
                                const value_type gamma_k = weights.get_scale_gamma_k(k);
                                scale_grad_sum_rank[k] += static_cast<double>(quant_floor) * static_cast<double>(out_scale_k) * static_cast<double>(gamma_k) * g;
                                mcol_at(col, k) += quant_floor * val_scale_k * gamma_k * g;
                                mgamma_at(k) += quant_floor * val_scale_k * out_scale_k * g;
                                // Parallel forward-contribution accumulation
                                // -- see scale_grad_sum_contrib's own
                                // comment above.
                                scale_grad_sum_rank_contrib[k] += static_cast<double>(quant_floor) * static_cast<double>(out_scale_k) * static_cast<double>(gamma_k) * contrib;
                                mcol_at_contrib(col, k) += quant_floor * val_scale_k * gamma_k * contrib;
                                mgamma_at_contrib(k) += quant_floor * val_scale_k * out_scale_k * contrib;
                            }
                        }
                        mdx[static_cast<std::size_t>(b) * in_cols + r] += cw_start * dyv;
                    }
                    // ONE update, using the batch-aggregated g/contrib --
                    // see cw_start's own comment above for why.
                    value_type quant = cw_orig;
                    value_type cw = cw_start;
                    if (learning_rate != value_type(0)) {
                        ci = SynapsePolicy::update_ci(ci, static_cast<value_type>(g_agg),
                                                      static_cast<value_type>(contrib_agg),
                                                      beta2, min_decay_frac, max_ci);
                        quant += SynapsePolicy::update_cw(static_cast<value_type>(g_agg), ci, S,
                                                          effective_lr, eps, damp_by_importance, max_abs_delta,
                                                          scale_invariant);
                        cw = quant * S;
                        if constexpr (StochasticRounding) {
                            ValueAccessor<VALUES_TYPE>::set_stochastic_live(dc.values, vb, quant, ci / combined_imp_scale);
                        } else {
                            ValueAccessor<VALUES_TYPE>::set_live(dc.values, vb, quant, ci / combined_imp_scale);
                        }
                        const value_type actual_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                        local_sum_abs_new_i += std::abs(static_cast<double>(actual_imp));
                        local_sum_abs_old_i += std::abs(static_cast<double>(ci_orig));
                        local_sum_sq_new_i  += static_cast<double>(actual_imp) * actual_imp;
                        local_sum_sq_old_i  += static_cast<double>(ci_orig) * ci_orig;
                        local_max_new_i = std::max(local_max_new_i, std::abs(actual_imp));
                    }
                }
            }
            if (learning_rate != value_type(0)) {
                // Scale update via the swappable policy (default
                // RMSpropScalePolicy reproduces this exact formula) --
                // see ScalePolicy's own docstring, delta_csr_types.hpp.
                // DeferredScaleWrite still only updates component 0
                // (scale_grad_sum, the old single-double accumulator) --
                // rank>1 is only meaningful for the non-deferred branch
                // above (scale_grad_sum_rank), matching that branch's own
                // scope note.
                if constexpr (!DeferredScaleWrite) {
                    for (std::size_t k = 0; k < rank; ++k) {
                        const value_type g_agg_k = static_cast<value_type>(scale_grad_sum_rank[k]);
                        const value_type contrib_agg_k = static_cast<value_type>(scale_grad_sum_rank_contrib[k]);
                        ScalePolicy::update(weights.value_scale[r * rank + k], weights.value_scale_importance[r * rank + k],
                                            g_agg_k, scale_eff_lr, beta2, eps, contrib_agg_k,
                                            &weights.get_value_scale_step_k(r, k), scale_invariant);
                    }
                } else {
                    const value_type g_agg = static_cast<value_type>(scale_grad_sum);
                    const value_type contrib_agg = static_cast<value_type>(scale_grad_sum_contrib);
                    ScalePolicy::update(weights.value_scale[r], weights.value_scale_importance[r],
                                        g_agg, scale_eff_lr, beta2, eps, contrib_agg,
                                        &weights.get_value_scale_step_k(r, 0), scale_invariant);
                }
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
        // row_ti_start: cumulative tile count per block-row, needed to
        // split the row-partitioned parallel loop below (each thread's
        // #pragma omp for chunk is a contiguous BLOCK-ROW range, but tile
        // COUNTS per row vary, so row_ti_start[br]..row_ti_start[br+1]
        // gives each row's tile-count window within that chunk -- NOT
        // storage offsets: since the row-workspace rewrite (see
        // conversation), tile byte/elem positions are tracked entirely
        // within each row's own RowWorkspace, snapshotted fresh per row,
        // not precomputed globally here (a global precompute was the
        // root of the byte_pos-staleness class of bugs this rewrite
        // closes -- see Block4Store::RowWorkspace's comment).
        std::vector<std::size_t>& row_ti_start = weights.block4.scratch_row_ti_start;
        const auto& BL4 = weights.block4.block_layout;
        row_ti_start.resize(BL4.rows + 1);

        // Per-row slot count across ALL block4 tiles touching that row
        // (not just one tile), needed for both lr_per_row_nnz and the
        // unconditional scale_eff_lr normalization. Every tile contributes
        // exactly BLOCK4_TILE slots per row it covers -- dense, no
        // per-slot scan needed (see block4.hpp: a live tile's slots are
        // all real synapses, weight=0.0 included).
        std::vector<uint32_t>& row_live_count = weights.block4.scratch_row_live_count;
        row_live_count.assign(n_in, 0);
        std::size_t ti = 0;
        for (std::size_t br = 0; br < BL4.rows; ++br) {
            row_ti_start[br] = ti; // written unconditionally -- an empty
            // row (n_bc==0) still needs a valid (empty) [start,start)
            // range for the parallel loop's row-partitioned split below.
            const std::size_t n_bc = BL4.row_nnz(br);
            ti += n_bc;
            if (n_bc == 0) continue;
            const uint32_t row_count = uint32_t(n_bc) * BLOCK4_TILE;
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                const std::size_t row = br * BLOCK4_TILE + li;
                if (row < n_in) row_live_count[row] = row_count;
            }
        }
        row_ti_start[BL4.rows] = ti;

        std::vector<double>& t_row_grad = weights.block4.scratch_row_grad;
        t_row_grad.assign(static_cast<std::size_t>(num_cpus) * n_in * rank, 0.0);
        // Parallel forward-contribution accumulator, same shape/layout as
        // t_row_grad -- mirrors t_col_grad_contrib's own rationale
        // (additive combination, one level up from per-synapse ci; see
        // that array's own comment above this function).
        std::vector<double> t_row_grad_contrib(static_cast<std::size_t>(num_cpus) * n_in * rank, 0.0);

        #pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mdx  = t_dx.data() + static_cast<std::size_t>(tid) * dst;
            // t_col_grad/t_row_grad are laid out [thread][col or row][k]
            // (see their sizing above/at this function's top). Both the
            // FP4 and FP8 block4 branches below are full rank-N -- they
            // loop k<rank and write every component through these same
            // accessors; there is no separate rank-1-strided pointer,
            // which would alias incorrectly against this same buffer once
            // rank>1.
            value_type* mcol_base = t_col_grad.data() + static_cast<std::size_t>(tid) * n_out * rank;
            auto mcol_at = [&](std::size_t col, std::size_t k) -> value_type& { return mcol_base[col * rank + k]; };
            double* mrow_base = t_row_grad.data() + static_cast<std::size_t>(tid) * n_in * rank;
            auto mrow_at = [&](std::size_t row, std::size_t k) -> double& { return mrow_base[row * rank + k]; };
            value_type* mcol_contrib_base = t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * n_out * rank;
            auto mcol_at_contrib = [&](std::size_t col, std::size_t k) -> value_type& { return mcol_contrib_base[col * rank + k]; };
            double* mrow_contrib_base = t_row_grad_contrib.data() + static_cast<std::size_t>(tid) * n_in * rank;
            auto mrow_at_contrib = [&](std::size_t row, std::size_t k) -> double& { return mrow_contrib_base[row * rank + k]; };
            // AQRS gamma (task #273/#283) -- block4 gets its OWN parallel
            // region (separate from the scattered path's, above), so
            // mgamma_at/mgamma_at_contrib need their own tid-scoped
            // closures here too, reading from the SAME t_gamma_grad/
            // t_gamma_grad_contrib buffers (declared once, outside both
            // parallel regions, at this function's top) that the scattered
            // path's own mgamma_at uses -- both regions' threads
            // accumulate into the same shared array, reduced together once
            // after both parallel regions close.
            value_type* mgamma_base = t_gamma_grad.data() + static_cast<std::size_t>(tid) * rank;
            auto mgamma_at = [&](std::size_t k) -> value_type& { return mgamma_base[k]; };
            value_type* mgamma_contrib_base = t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * rank;
            auto mgamma_at_contrib = [&](std::size_t k) -> value_type& { return mgamma_contrib_base[k]; };

            // Partitioned BY BLOCK-ROW, not by flat tile index -- each
            // thread exclusively owns every tile in the rows it's
            // assigned, processed sequentially within that row (the
            // inner `for (ti...)` below), so no two threads ever touch
            // the same row's tile_data concurrently. This is what makes
            // it safe for a tile's handle to resize (real sparse<->dense
            // transitions, driven by real content) inside this parallel
            // region at all -- see block4_resize_tile_in_row's comment:
            // resizing ONE tile shifts every LATER tile in ITS OWN row,
            // which is exactly the memory a same-row thread already owns
            // exclusively, never memory another thread could be reading
            // or writing. (Real budget exhaustion during that resize is
            // handled by declining the growth, not throwing -- see
            // ~Block4TileHandle()'s comment -- so there's no exception-
            // across-the-parallel-region-boundary concern here either.)
            // schedule(static): row widths CAN vary a lot (some rows have
            // far more live tiles than others), which in principle makes
            // static's flat contiguous split load-balance worse than
            // schedule(dynamic)/schedule(guided) once num_cpus > 1 --
            // tried both, measured worse in practice: their real
            // per-chunk dispatch overhead showed up as a real backward
            // slowdown at high tile density even at num_cpus=1
            // (scripts/bench_block4_vs_dense_fp4.cpp: dynamic/guided both
            // measured ~1.69x speedup over the dense floor at 100% fill,
            // vs ~1.97x for static -- worse than even the pre-this-fix
            // documented baseline of ~1.88x), where there's no actual
            // load-balancing benefit to buy that overhead with in the
            // first place. Revisit if a real, measured multi-thread
            // workload with genuinely lopsided row widths shows static's
            // imbalance actually costing more wall-clock time than this.
            #pragma omp for schedule(static)
            for (std::size_t br = 0; br < BL4.rows; ++br) {
            if (row_ti_start[br] == row_ti_start[br + 1]) continue; // empty row
            // BUG FIX (see conversation): the OLD version of this loop read
            // and wrote each tile directly through the shared store (via
            // at_index()'s fast path), relying only on row-exclusive thread
            // ownership for safety. That protects against two threads
            // touching the SAME row, but NOT against a DIFFERENT row's
            // growth: growing row X past its own current headroom shifts
            // tbyte_start/tbyte_end -- and physically MEMMOVES the tile
            // BYTES themselves -- for every row after X, including
            // whatever row this thread owns. Confirmed via ASan as a real,
            // reproducible heap-use-after-free / negative-size memmove
            // even with tile_data's capacity pre-reserved (which only
            // fixes buffer REALLOCATION, a separate hazard) and even with
            // a lock guarding concurrent growers (which only fixes two
            // growers racing each other, not a grower racing this row's
            // reader). See disldo_backward_sparse_grad's identical fix
            // in sisldo_ops.hpp for the full writeup.
            //
            // Fix: snapshot this row into a thread-private workspace
            // ONCE, do all reads/writes against that private copy, then
            // merge back in one row-exclusive step at the end (evicting
            // lowest-importance synapses if this row grew past its own
            // existing headroom rather than shifting anything else -- see
            // Block4Store::merge_row_workspace's comment). This makes the
            // cross-row-shift hazard structurally unreachable rather than
            // merely less likely: no row growing through this workspace
            // path ever calls the shared store's cross-row-shifting
            // resize at all. Used UNCONDITIONALLY (even when
            // learning_rate == 0, which never actually writes anything
            // back) rather than branching read-only vs writing: dx doesn't
            // depend on whether writes happen, this keeps a single tested
            // code path instead of two, and it stays correct even if some
            // OTHER concurrent caller on the same store is writing at the
            // same time.
            // process_tile: the per-tile gradient math (SIMD batch loop,
            // stochastic requantize) extracted into a lambda so the
            // read-only and writing branches below share EXACTLY one
            // implementation instead of two copies that could drift
            // apart -- only how `tdata` is SOURCED differs between them,
            // never the math. Every write inside is already gated by
            // `learning_rate != value_type(0)` (unchanged from before
            // this split), so calling this from the read-only branch is
            // provably a no-op write-wise, not just assumed safe.
            auto process_tile = [&](uint32_t bc, uint8_t* tdata) -> bool {
                bool tile_dirty = false;
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
                    if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                        // FP8 (E4M3) block4 weight+importance update.
                        // MEASURED (scripts/bench_block4_fp8_simd.cpp,
                        // 512x512 100%-dense block4, best-of-200,
                        // -O3 -ffast-math -march=native), not assumed:
                        //   batch=1:  full-SIMD 0.0060s vs plain-scalar
                        //             0.0048s -- SIMD LOST (~20% slower).
                        //   batch=1:  scalar decode+encode + SIMD
                        //             accumulate-only ~= plain-scalar
                        //             (0.0048s, no measurable win OR
                        //             loss -- batch=1 means the SIMD
                        //             accumulate loop's own inner `for
                        //             (b<batch)` only runs once, so it
                        //             never gets to amortize its setup
                        //             cost against reused decoded state).
                        //   batch=32: scalar decode+encode + SIMD
                        //             accumulate 0.0227s vs plain-scalar
                        //             0.0298s -- SIMD WON (~24% faster),
                        //             confirmed via objdump that this is
                        //             real 128-bit packed SIMD (vmulps/
                        //             vrsqrtps/vaddps on xmm registers)
                        //             inside this exact lambda, not
                        //             GCC auto-vectorizing scalar code.
                        // Conclusion: block4_vec_decode_fp8/
                        // block4_vec_quantize_stochastic_fp8 (the SIMD
                        // decode/encode built and tested alongside
                        // disldo_forward's own block4 section) measurably
                        // LOSE here -- E4M3's 256-code space makes their
                        // subnormal/NaN-lane scalar-correction fallback
                        // (block4.hpp) real, non-negligible overhead that
                        // FP4's simpler E2M1 (16 codes) never pays. So:
                        // scalar fp8_decode_bits/fp8_quantize_stochastic
                        // for decode/encode (matching FP4's own empirical
                        // finding for backward, now independently
                        // confirmed true for FP8 too, not just assumed to
                        // transfer), SIMD (Block4Vec) kept ONLY for the
                        // batch-loop accumulation math -- identical to
                        // FP4's own accumulate loop once weight/
                        // importance are decoded to float, and the one
                        // piece that measurably earns its complexity at
                        // realistic (>1) batch sizes.
                        if constexpr (std::is_same_v<value_type, float> && !SILI_BLOCK4_FORCE_SCALAR_BACKWARD) {
                            // bit-shift for backward specifically") against
                            // FP8's real measured numbers -- first SIMD-
                            // decode version measured slower than the plain
                            // scalar fallback (0.0060s vs 0.0048s/call,
                            // 512x512 100%-dense block4, 200-rep best-of),
                            // this is checking whether decode specifically
                            // is the cause before reverting the whole path.
                            const value_type w_decoded_arr8[BLOCK4_TILE] = {
                                fp8_decode_bits(tdata[Block4Tile8::slot_index(li, 0)]), fp8_decode_bits(tdata[Block4Tile8::slot_index(li, 1)]),
                                fp8_decode_bits(tdata[Block4Tile8::slot_index(li, 2)]), fp8_decode_bits(tdata[Block4Tile8::slot_index(li, 3)])};
                            const value_type imp_decoded_arr8[BLOCK4_TILE] = {
                                fp8_decode_bits(tdata[BLOCK4_TILE + Block4Tile8::slot_index(li, 0)]), fp8_decode_bits(tdata[BLOCK4_TILE + Block4Tile8::slot_index(li, 1)]),
                                fp8_decode_bits(tdata[BLOCK4_TILE + Block4Tile8::slot_index(li, 2)]), fp8_decode_bits(tdata[BLOCK4_TILE + Block4Tile8::slot_index(li, 3)])};

                            // value_scale_k(row,k) -- see FP4 branch's
                            // identical comment above (mirrors it exactly,
                            // now generalized over rank instead of the
                            // single rank-1-only val_scale local).
                            value_type* value_scale_k8 = weights.scale_rank_scratch.value_scale_k.data()
                                + static_cast<std::size_t>(tid) * rank;
                            for (std::size_t k = 0; k < rank; ++k) value_scale_k8[k] = weights.get_value_scale_k(row, k);

                            std::size_t col4_8[BLOCK4_TILE];
                            bool        col_valid4_8[BLOCK4_TILE];
                            // out_scale_k4_8[k][lj]: per-rank-component
                            // output_scale, needed for value_scale_k's own
                            // gradient below (mirrors FP4's out_scale_k4).
                            const Flat2DView out_scale_k4_8{
                                weights.scale_rank_scratch.out_scale_k.data() + static_cast<std::size_t>(tid) * rank * BLOCK4_TILE,
                                BLOCK4_TILE};
                            value_type  combined_scale4_8[BLOCK4_TILE], combined_imp_scale4_8[BLOCK4_TILE];
                            value_type  cw4_8[BLOCK4_TILE], ci4_8[BLOCK4_TILE], cw_orig4_8[BLOCK4_TILE];
                            // was_live4_8[lj]: TRUE only if this cell already
                            // held a genuine synapse (weight OR importance
                            // byte nonzero) BEFORE this backward call.
                            // col_valid4_8 is just col<n_out -- block4 tiles
                            // are DENSE 4x4 blocks touched at every in-bounds
                            // (li,lj), including cells that were never a real
                            // synapse (block4's own "no synapse here" IS
                            // weight==importance==0, unlike scattered CSR's
                            // structural absence -- see block4_count_live/
                            // block4_sparse_pack's identical `dense[i]!=0`
                            // criterion). The never-zero live quantizer must
                            // ONLY protect an ALREADY-established synapse
                            // from rounding back to the dead code -- applying
                            // it unconditionally to every col_valid4_8 cell
                            // would force every touched-but-never-connected
                            // cell permanently "live", corrupting block4's
                            // own sparse/dense repacking (confirmed directly:
                            // caused test_disldo_block4_backward's whole tile
                            // to zero out via a corrupted live-count/repack --
                            // see conversation).
                            bool was_live4_8[BLOCK4_TILE];
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                                col_valid4_8[lj] = col < n_out;
                                if (!col_valid4_8[lj]) {
                                    col4_8[lj] = 0;
                                    combined_scale4_8[lj] = combined_imp_scale4_8[lj] = value_type(0);
                                    cw4_8[lj] = ci4_8[lj] = cw_orig4_8[lj] = value_type(0);
                                    was_live4_8[lj] = false;
                                    for (std::size_t k = 0; k < rank; ++k) out_scale_k4_8[k][lj] = value_type(0);
                                    continue;
                                }
                                was_live4_8[lj] = (w_decoded_arr8[lj] != value_type(0)) || (imp_decoded_arr8[lj] != value_type(0));
                                col4_8[lj] = col;
                                const value_type out_imp_scale = weights.get_output_importance_scale(col);
                                // S(row,col) = sum_k value_scale_k*output_scale_k
                                // -- see FP4 branch's identical get_scale
                                // comment (mirrors it exactly). cw4_8 is now
                                // CODE-SPACE (same convention as FP4's
                                // quant4/quant_orig4, NOT the true-weight-
                                // scaled representation this used to hold --
                                // see conversation: keeping cw4_8 in true
                                // units while update_cw's RMSprop-normalized
                                // delta is ~S-independent (magnitude ~eff_lr
                                // regardless of S) meant the later `/
                                // combined_scale4_8` at write time amplified
                                // every step by 1/S instead of damping it by
                                // S the way FP4's convention does, blowing
                                // up explosively for any layer with small
                                // output_scale (e.g. a wide dense layer's
                                // fan-in-corrected scale). Every use of
                                // cw4_8/cw_start4_8 as a TRUE weight (contrib,
                                // dx, update_cw's S argument) below now
                                // multiplies by combined_scale4_8 explicitly,
                                // matching FP4's quant_start4[lj]*S pattern.
                                combined_scale4_8[lj]     = weights.get_scale(row, col);
                                combined_imp_scale4_8[lj] = imp_scale * out_imp_scale;
                                cw_orig4_8[lj] = w_decoded_arr8[lj];
                                cw4_8[lj] = cw_orig4_8[lj];
                                ci4_8[lj] = imp_decoded_arr8[lj] * combined_imp_scale4_8[lj];
                                for (std::size_t k = 0; k < rank; ++k) out_scale_k4_8[k][lj] = weights.get_output_scale_k(col, k);
                            }

                            // Per-rank versions of mcol4_8/mrow_local8 (see
                            // FP4's identical mcol4_rank/mrow_local_k) --
                            // needed to train EVERY rank component, not
                            // just component 0.
                            // Reused scratch memory (not fresh stack space
                            // this iteration) -- must be explicitly zeroed
                            // each tile visit, unlike the old `= {}`
                            // stack-array zero-init (task #295).
                            const std::size_t tid_rank = static_cast<std::size_t>(tid) * rank;
                            const std::size_t tid_rank_tile = tid_rank * BLOCK4_TILE;
                            auto& srs = weights.scale_rank_scratch;
                            const Flat2DView mcol4_8_rank{srs.mcol_rank.data() + tid_rank_tile, BLOCK4_TILE};
                            double* mrow_local8_k = srs.mrow_local_k.data() + tid_rank;
                            const Flat2DView mcol4_8_rank_contrib{srs.mcol_rank_contrib.data() + tid_rank_tile, BLOCK4_TILE};
                            double* mrow_local8_k_contrib = srs.mrow_local_k_contrib.data() + tid_rank;
                            // AQRS gamma's own gradient (task #273/#283) --
                            // per-tile-row local accumulator, folded into
                            // the shared mgamma_at/mgamma_at_contrib (same
                            // ones the scattered path uses) once this row
                            // finishes, matching mrow_local8_k's own
                            // fold-into-mrow_at pattern below.
                            double* mgamma_local8_k = srs.mgamma_local_k.data() + tid_rank;
                            double* mgamma_local8_k_contrib = srs.mgamma_local_k_contrib.data() + tid_rank;
                            std::fill(mcol4_8_rank[0], mcol4_8_rank[0] + rank * BLOCK4_TILE, value_type(0));
                            std::fill(mrow_local8_k, mrow_local8_k + rank, 0.0);
                            std::fill(mcol4_8_rank_contrib[0], mcol4_8_rank_contrib[0] + rank * BLOCK4_TILE, value_type(0));
                            std::fill(mrow_local8_k_contrib, mrow_local8_k_contrib + rank, 0.0);
                            std::fill(mgamma_local8_k, mgamma_local8_k + rank, 0.0);
                            std::fill(mgamma_local8_k_contrib, mgamma_local8_k_contrib + rank, 0.0);
                            const std::size_t col_base8 = std::size_t(bc) * BLOCK4_TILE;
                            const bool full_tile_cols8 = (col_base8 + BLOCK4_TILE <= n_out);

                            if (full_tile_cols8) {
                                const Block4Vec effective_lr_v = block4_vec_broadcast(effective_lr);
                                const Block4Vec beta2_v        = block4_vec_broadcast(beta2);
                                const Block4Vec eps_v          = block4_vec_broadcast(eps);
                                const Block4Vec min_decay_frac_v = block4_vec_broadcast(min_decay_frac);
                                const Block4Vec max_ci_v = block4_vec_broadcast(max_ci);
                                const Block4Vec max_abs_delta_v  = block4_vec_broadcast(max_abs_delta);
                                // cw_start_v: FIXED for the whole batch loop
                                // -- see the scattered path's identical
                                // cw_start comment for the full rationale
                                // (real bug this session: cw_v/ci_v were
                                // mutated mid-loop, once per batch row).
                                // g_agg_v/contrib_agg_v are Block4Vec
                                // (16-byte) stack locals, not a matrix
                                // -sized buffer.
                                const Block4Vec cw_start_v = block4_vec_load(cw4_8);
                                Block4Vec ci_v = block4_vec_load(ci4_8);
                                const Block4Vec cw_orig_v   = block4_vec_load(cw_orig4_8);
                                // S_v: code-space -> true-weight conversion
                                // factor, matching FP4's scalar S -- see
                                // this tile's decode-site comment above for
                                // the full rationale.
                                const Block4Vec S_v = block4_vec_load(combined_scale4_8);
                                // mcol_acc_v_k8/mcol_acc_v_k8_contrib are
                                // the only TRUE cross-batch accumulators
                                // here (value_scale_k_v8/out_scale_k_v8
                                // were plain caches of a broadcast/load
                                // already available from value_scale_k8/
                                // out_scale_k4_8 -- recomputed inline at
                                // each use below instead, cheap SIMD ops,
                                // no need to cache them). Backed by scratch
                                // (task #295), load-accumulate-store each
                                // batch iteration instead of a live
                                // Block4Vec register array across
                                // iterations.
                                value_type* mcol_acc_raw8 = srs.mcol_acc_raw.data() + tid_rank_tile;
                                value_type* mcol_acc_raw8_contrib = srs.mcol_acc_raw_contrib.data() + tid_rank_tile;
                                for (std::size_t k = 0; k < rank; ++k) {
                                    block4_vec_store(mcol_acc_raw8 + k * BLOCK4_TILE, block4_vec_broadcast(0.0f));
                                    block4_vec_store(mcol_acc_raw8_contrib + k * BLOCK4_TILE, block4_vec_broadcast(0.0f));
                                }
                                Block4Vec g_agg_v = block4_vec_broadcast(0.0f);
                                Block4Vec contrib_agg_v = block4_vec_broadcast(0.0f);
                                const bool training = (learning_rate != value_type(0));
                                for (SIZE_TYPE b = 0; b < batch; ++b) {
                                    const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                                    value_type* mdx_row = mdx + static_cast<std::size_t>(b) * in_cols + row;
                                    const Block4Vec dyv_v = block4_vec_load(
                                        output_grad + static_cast<std::size_t>(b) * n_out + col_base8);
                                    const Block4Vec g_v = dyv_v * block4_vec_broadcast(iv);
                                    if (training) {
                                        // Additive forward-contribution
                                        // combination -- see the scattered
                                        // path's identical fix for the full
                                        // rationale and the proved theorem
                                        // (Joint.combined_signal_strictly_informative).
                                        const Block4Vec contrib_v = cw_start_v * S_v * block4_vec_broadcast(iv);
                                        g_agg_v += g_v;
                                        contrib_agg_v += contrib_v;
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            // AQRS gamma (task #273/#283): out_scale_k_v8/value_scale_k_v8
                                            // are pure DIRECTION (never gamma-baked -- see gamma_k_arr's
                                            // own docstring above for why). value_scale's own gradient
                                            // (mrow_local8_k) and output_scale's own gradient
                                            // (mcol_acc_v_k8) each need an explicit gamma_k factor;
                                            // gamma's own gradient (mgamma_local8_k) uses the PURE
                                            // direction product, no gamma factor.
                                            const Block4Vec value_scale_k_v8 = block4_vec_broadcast(value_scale_k8[k]);
                                            const Block4Vec out_scale_k_v8 = block4_vec_load(out_scale_k4_8[k]);
                                            mrow_local8_k[k] += static_cast<double>(block4_vec_hsum(cw_orig_v * out_scale_k_v8 * g_v)) * static_cast<double>(gamma_k_arr[k]);
                                            mgamma_local8_k[k] += static_cast<double>(block4_vec_hsum(cw_orig_v * out_scale_k_v8 * value_scale_k_v8 * g_v));
                                            mrow_local8_k_contrib[k] += static_cast<double>(block4_vec_hsum(cw_orig_v * out_scale_k_v8 * contrib_v)) * static_cast<double>(gamma_k_arr[k]);
                                            mgamma_local8_k_contrib[k] += static_cast<double>(block4_vec_hsum(cw_orig_v * out_scale_k_v8 * value_scale_k_v8 * contrib_v));
                                            value_type* acc = mcol_acc_raw8 + k * BLOCK4_TILE;
                                            block4_vec_store(acc, block4_vec_load(acc) + cw_orig_v * value_scale_k_v8 * g_v * block4_vec_broadcast(gamma_k_arr[k]));
                                            value_type* acc_c = mcol_acc_raw8_contrib + k * BLOCK4_TILE;
                                            block4_vec_store(acc_c, block4_vec_load(acc_c) + cw_orig_v * value_scale_k_v8 * contrib_v * block4_vec_broadcast(gamma_k_arr[k]));
                                        }
                                    }
                                    *mdx_row += block4_vec_hsum(cw_start_v * S_v * dyv_v);
                                }
                                // ONE update, using the batch-aggregated g/contrib.
                                Block4Vec cw_v = cw_start_v;
                                if (training) {
                                    ci_v = SynapsePolicyVec::update_ci(ci_v, g_agg_v, contrib_agg_v, beta2_v, min_decay_frac_v, max_ci_v);
                                    const Block4Vec delta_v = SynapsePolicyVec::update_cw(
                                        g_agg_v, ci_v, S_v, effective_lr_v, eps_v,
                                        damp_by_importance, max_abs_delta_v, scale_invariant);
                                    cw_v += delta_v;
                                }
                                block4_vec_store(cw4_8, cw_v);
                                block4_vec_store(ci4_8, ci_v);
                                for (std::size_t k = 0; k < rank; ++k) {
                                    block4_vec_store(mcol4_8_rank[k], block4_vec_load(mcol_acc_raw8 + k * BLOCK4_TILE));
                                    block4_vec_store(mcol4_8_rank_contrib[k], block4_vec_load(mcol_acc_raw8_contrib + k * BLOCK4_TILE));
                                }
                            } else {
                                // Boundary tile-column: scalar bounds-checked
                                // fallback, matching the FP4 branch's own.
                                // cw_start4_8/g_agg4_8/contrib_agg4_8: FIXED
                                // per-lj snapshot + aggregators for the
                                // whole batch loop -- see the scattered
                                // path's identical cw_start comment for the
                                // full rationale. 3 small (BLOCK4_TILE=4
                                // element) stack arrays, not matrix-sized.
                                value_type cw_start4_8[BLOCK4_TILE];
                                double g_agg4_8[BLOCK4_TILE] = {0.0};
                                double contrib_agg4_8[BLOCK4_TILE] = {0.0};
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) cw_start4_8[lj] = cw4_8[lj];
                                for (SIZE_TYPE b = 0; b < batch; ++b) {
                                    const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                                    value_type* mdx_row = mdx + static_cast<std::size_t>(b) * in_cols + row;
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (!col_valid4_8[lj]) continue;
                                        const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col4_8[lj]];
                                        const value_type g   = dyv * iv;
                                        if (learning_rate != value_type(0)) {
                                            // Additive forward-contribution
                                            // combination -- see the
                                            // scattered path's identical fix
                                            // (Joint.combined_signal_strictly_informative).
                                            const value_type contrib = iv * (cw_start4_8[lj] * combined_scale4_8[lj]);
                                            g_agg4_8[lj] += static_cast<double>(g);
                                            contrib_agg4_8[lj] += static_cast<double>(contrib);
                                            for (std::size_t k = 0; k < rank; ++k) {
                                                // AQRS gamma (task #273/#283) -- see the SIMD branch's
                                                // identical comment above for the full rationale.
                                                mrow_local8_k[k] += static_cast<double>(cw_orig4_8[lj]) * static_cast<double>(out_scale_k4_8[k][lj]) * static_cast<double>(gamma_k_arr[k]) * g;
                                                mcol4_8_rank[k][lj] += cw_orig4_8[lj] * value_scale_k8[k] * gamma_k_arr[k] * g;
                                                mgamma_local8_k[k] += static_cast<double>(cw_orig4_8[lj]) * static_cast<double>(out_scale_k4_8[k][lj]) * static_cast<double>(value_scale_k8[k]) * g;
                                                mrow_local8_k_contrib[k] += static_cast<double>(cw_orig4_8[lj]) * static_cast<double>(out_scale_k4_8[k][lj]) * static_cast<double>(gamma_k_arr[k]) * contrib;
                                                mcol4_8_rank_contrib[k][lj] += cw_orig4_8[lj] * value_scale_k8[k] * gamma_k_arr[k] * contrib;
                                                mgamma_local8_k_contrib[k] += static_cast<double>(cw_orig4_8[lj]) * static_cast<double>(out_scale_k4_8[k][lj]) * static_cast<double>(value_scale_k8[k]) * contrib;
                                            }
                                        }
                                        *mdx_row += cw_start4_8[lj] * combined_scale4_8[lj] * dyv;
                                    }
                                }
                                // ONE update per lj, using the batch-aggregated g/contrib.
                                if (learning_rate != value_type(0)) {
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (!col_valid4_8[lj]) continue;
                                        const value_type g_agg = static_cast<value_type>(g_agg4_8[lj]);
                                        const value_type contrib_agg = static_cast<value_type>(contrib_agg4_8[lj]);
                                        ci4_8[lj] = SynapsePolicy::update_ci(ci4_8[lj], g_agg, contrib_agg, beta2, min_decay_frac, max_ci);
                                        cw4_8[lj] = cw_start4_8[lj] + SynapsePolicy::update_cw(
                                            g_agg, ci4_8[lj], combined_scale4_8[lj], effective_lr, eps, damp_by_importance, max_abs_delta,
                                            scale_invariant);
                                    }
                                }
                            }

                            if (learning_rate != value_type(0)) {
                                for (std::size_t k = 0; k < rank; ++k) {
                                    mrow_at(row, k) += mrow_local8_k[k];
                                    mrow_at_contrib(row, k) += mrow_local8_k_contrib[k];
                                    mgamma_at(k) += static_cast<value_type>(mgamma_local8_k[k]);
                                    mgamma_at_contrib(k) += static_cast<value_type>(mgamma_local8_k_contrib[k]);
                                }
                                if (full_tile_cols8) {
                                    // Scalar fp8_quantize_stochastic, not
                                    // block4_vec_quantize_stochastic_fp8
                                    // -- see decode's comment above this
                                    // whole branch for the measured
                                    // reasoning (same conclusion applies
                                    // to encode).
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            mcol_at(col4_8[lj], k) += mcol4_8_rank[k][lj];
                                            mcol_at_contrib(col4_8[lj], k) += mcol4_8_rank_contrib[k][lj];
                                        }
                                        const uint32_t slot = Block4Tile8::slot_index(li, lj);
                                        // was_live4_8[lj] gate -- see its own
                                        // declaration comment above: a cell
                                        // that was never a real synapse must
                                        // stay allowed to round to 0.
                                        if (was_live4_8[lj]) {
                                            tdata[slot]               = fp8_quantize_stochastic_live(cw4_8[lj]);
                                            tdata[BLOCK4_TILE + slot] = fp8_quantize_stochastic_live_nonneg(ci4_8[lj] / combined_imp_scale4_8[lj]);
                                        } else {
                                            tdata[slot]               = fp8_quantize_stochastic(cw4_8[lj]);
                                            tdata[BLOCK4_TILE + slot] = fp8_quantize_stochastic(ci4_8[lj] / combined_imp_scale4_8[lj]);
                                        }
                                    }
                                    tile_dirty = true;
                                } else {
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (!col_valid4_8[lj]) continue;
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            mcol_at(col4_8[lj], k) += mcol4_8_rank[k][lj];
                                            mcol_at_contrib(col4_8[lj], k) += mcol4_8_rank_contrib[k][lj];
                                        }
                                        const uint32_t slot = Block4Tile8::slot_index(li, lj);
                                        if (was_live4_8[lj]) {
                                            tdata[slot]               = fp8_quantize_stochastic_live(cw4_8[lj]);
                                            tdata[BLOCK4_TILE + slot] = fp8_quantize_stochastic_live_nonneg(ci4_8[lj] / combined_imp_scale4_8[lj]);
                                        } else {
                                            tdata[slot]               = fp8_quantize_stochastic(cw4_8[lj]);
                                            tdata[BLOCK4_TILE + slot] = fp8_quantize_stochastic(ci4_8[lj] / combined_imp_scale4_8[lj]);
                                        }
                                        tile_dirty = true;
                                    }
                                }
                            }
                        } else {
                            // Hypothetical non-float value_type (never
                            // actually instantiated -- see the FP4 branch's
                            // identical comment below): plain scalar,
                            // per-lj-sequential fallback, same math as the
                            // SIMD path above, previously the ONLY FP8
                            // implementation before this SIMD pass -- kept
                            // here rather than deleted, matching FP4's own
                            // precedent for this exact fallback slot.
                            std::vector<value_type> value_scale_k8_fb(rank);
                            for (std::size_t k = 0; k < rank; ++k) value_scale_k8_fb[k] = weights.get_value_scale_k(row, k);
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                                if (col >= n_out) continue;
                                const uint32_t slot = Block4Tile8::slot_index(li, lj);
                                std::vector<value_type> out_scale_k8_fb(rank);
                                for (std::size_t k = 0; k < rank; ++k) out_scale_k8_fb[k] = weights.get_output_scale_k(col, k);
                                const value_type out_imp_scale = weights.get_output_importance_scale(col);
                                const value_type combined_scale     = weights.get_scale(row, col);
                                const value_type combined_imp_scale = imp_scale * out_imp_scale;
                                const value_type cw_orig = fp8_decode_bits(tdata[slot]);
                                const value_type cw_start = cw_orig * combined_scale;
                                value_type ci = fp8_decode_bits(tdata[BLOCK4_TILE + slot]) * combined_imp_scale;

                                std::vector<value_type> mcol_local_k(rank, value_type(0));
                                std::vector<double>     mrow_local_k(rank, 0.0);
                                std::vector<value_type> mcol_local_k_contrib(rank, value_type(0));
                                std::vector<double>     mrow_local_k_contrib(rank, 0.0);
                                std::vector<double>     mgamma_local_k(rank, 0.0);
                                std::vector<double>     mgamma_local_k_contrib(rank, 0.0);
                                double     g_agg = 0.0, contrib_agg = 0.0;
                                for (SIZE_TYPE b = 0; b < batch; ++b) {
                                    const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                                    value_type* mdx_row = mdx + static_cast<std::size_t>(b) * in_cols + row;
                                    const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col];
                                    const value_type g   = dyv * iv;
                                    if (learning_rate != value_type(0)) {
                                        // Additive forward-contribution
                                        // combination -- see the scattered
                                        // path's identical fix
                                        // (Joint.combined_signal_strictly_informative).
                                        const value_type contrib = iv * cw_start;
                                        g_agg += static_cast<double>(g);
                                        contrib_agg += static_cast<double>(contrib);
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            // AQRS gamma -- see the SIMD branch's identical comment above.
                                            mrow_local_k[k] += static_cast<double>(cw_orig) * static_cast<double>(out_scale_k8_fb[k]) * static_cast<double>(gamma_k_arr[k]) * g;
                                            mcol_local_k[k] += cw_orig * value_scale_k8_fb[k] * gamma_k_arr[k] * g;
                                            mgamma_local_k[k] += static_cast<double>(cw_orig) * static_cast<double>(out_scale_k8_fb[k]) * static_cast<double>(value_scale_k8_fb[k]) * g;
                                            mrow_local_k_contrib[k] += static_cast<double>(cw_orig) * static_cast<double>(out_scale_k8_fb[k]) * static_cast<double>(gamma_k_arr[k]) * contrib;
                                            mcol_local_k_contrib[k] += cw_orig * value_scale_k8_fb[k] * gamma_k_arr[k] * contrib;
                                            mgamma_local_k_contrib[k] += static_cast<double>(cw_orig) * static_cast<double>(out_scale_k8_fb[k]) * static_cast<double>(value_scale_k8_fb[k]) * contrib;
                                        }
                                    }
                                    *mdx_row += cw_start * dyv;
                                }
                                // ONE update, using the batch-aggregated g/contrib.
                                value_type cw = cw_start;
                                if (learning_rate != value_type(0)) {
                                    ci = SynapsePolicy::update_ci(ci, static_cast<value_type>(g_agg),
                                                                  static_cast<value_type>(contrib_agg),
                                                                  beta2, min_decay_frac, max_ci);
                                    cw += SynapsePolicy::update_cw(static_cast<value_type>(g_agg), ci, value_type(1),
                                                                   effective_lr, eps, damp_by_importance, max_abs_delta,
                                                                   scale_invariant);
                                    for (std::size_t k = 0; k < rank; ++k) {
                                        mrow_at(row, k) += mrow_local_k[k];
                                        mcol_at(col, k) += mcol_local_k[k];
                                        mrow_at_contrib(row, k) += mrow_local_k_contrib[k];
                                        mcol_at_contrib(col, k) += mcol_local_k_contrib[k];
                                        mgamma_at(k) += static_cast<value_type>(mgamma_local_k[k]);
                                        mgamma_at_contrib(k) += static_cast<value_type>(mgamma_local_k_contrib[k]);
                                    }
                                    // was_live gate -- see was_live4_8's
                                    // declaration comment (SIMD branch
                                    // above) for the full rationale. tdata
                                    // still holds the PRE-update bytes here.
                                    const bool was_live = (cw_orig != value_type(0)) ||
                                        (fp8_decode_bits(tdata[BLOCK4_TILE + slot]) != value_type(0));
                                    if (was_live) {
                                        tdata[slot]               = fp8_quantize_stochastic_live(cw / combined_scale);
                                        tdata[BLOCK4_TILE + slot] = fp8_quantize_stochastic_live_nonneg(ci / combined_imp_scale);
                                    } else {
                                        tdata[slot]               = fp8_quantize_stochastic(cw / combined_scale);
                                        tdata[BLOCK4_TILE + slot] = fp8_quantize_stochastic(ci / combined_imp_scale);
                                    }
                                    tile_dirty = true;
                                }
                            }
                        }
                        continue; // this li done -- skip the FP4 branch entirely
                    }
                    const uint8_t byte0 = tdata[Block4Tile::slot_index(li, 0)], byte1 = tdata[Block4Tile::slot_index(li, 1)],
                                  byte2 = tdata[Block4Tile::slot_index(li, 2)], byte3 = tdata[Block4Tile::slot_index(li, 3)];
                    const value_type w_decoded_arr[BLOCK4_TILE]   = {FP4_TABLE[byte0 & 0xFu], FP4_TABLE[byte1 & 0xFu],
                                                                       FP4_TABLE[byte2 & 0xFu], FP4_TABLE[byte3 & 0xFu]};
                    const value_type imp_decoded_arr[BLOCK4_TILE] = {FP4_TABLE[(byte0 >> 4) & 0xFu], FP4_TABLE[(byte1 >> 4) & 0xFu],
                                                                       FP4_TABLE[(byte2 >> 4) & 0xFu], FP4_TABLE[(byte3 >> 4) & 0xFu]};

                    // value_scale_k(row,k), fetched once per row -- matches
                    // the scattered path's own once-per-row granularity
                    // (see disldo_backward's non-DeferredScaleWrite branch).
                    value_type* value_scale_k = weights.scale_rank_scratch.value_scale_k.data()
                        + static_cast<std::size_t>(tid) * rank;
                    for (std::size_t k = 0; k < rank; ++k) value_scale_k[k] = weights.get_value_scale_k(row, k);

                    std::size_t col4[BLOCK4_TILE];
                    bool        col_valid4[BLOCK4_TILE];
                    value_type  combined_scale4[BLOCK4_TILE], combined_imp_scale4[BLOCK4_TILE];
                    // quant4: the stored CODE itself -- now the primary
                    // optimized quantity (see the batch loop below), NOT
                    // true units like the old cw4 was. ci4 stays in true
                    // (importance) units, unrelated to this fix. quant_orig4
                    // is an immutable snapshot of quant4's call-entry value
                    // (quant4 itself mutates across the batch loop below),
                    // used for the forward-contribution signal
                    // (contrib = x*quant_orig) -- mirrors FP8's cw_orig4_8.
                    value_type  quant4[BLOCK4_TILE], ci4[BLOCK4_TILE], quant_orig4[BLOCK4_TILE];
                    // out_scale_k4[k][lj]: per-rank-component output_scale,
                    // needed for value_scale_k's own gradient below.
                    const Flat2DView out_scale_k4{
                        weights.scale_rank_scratch.out_scale_k.data() + static_cast<std::size_t>(tid) * rank * BLOCK4_TILE,
                        BLOCK4_TILE};
                    // was_live4[lj]: see was_live4_8's declaration comment
                    // (FP8 branch above) for the full rationale -- CORRECTED
                    // from this loop's own former comment ("every slot is a
                    // real synapse... no liveness check"), which was true
                    // and harmless under the OLD quantizer (an untouched
                    // slot's delta is 0, and plain fp4_quantize(0)==0
                    // preserved blank status) but is NOT harmless once the
                    // WRITE uses the never-zero live quantizer -- that would
                    // permanently "birth" a synapse at every col_valid4
                    // position regardless of whether gradient signal ever
                    // touched it, corrupting block4's own sparse/dense
                    // repacking (confirmed directly via
                    // test_disldo_block4_backward -- see conversation).
                    bool was_live4[BLOCK4_TILE];
                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                        const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                        col_valid4[lj] = col < n_out;
                        if (!col_valid4[lj]) {
                            col4[lj] = 0;
                            combined_scale4[lj] = combined_imp_scale4[lj] = value_type(0);
                            quant4[lj] = ci4[lj] = quant_orig4[lj] = value_type(0);
                            was_live4[lj] = false;
                            for (std::size_t k = 0; k < rank; ++k) out_scale_k4[k][lj] = value_type(0);
                            continue;
                        }
                        col4[lj] = col;
                        was_live4[lj] = (w_decoded_arr[lj] != value_type(0)) || (imp_decoded_arr[lj] != value_type(0));
                        const value_type out_imp_scale = weights.get_output_importance_scale(col);
                        // S(row,col) = sum_k value_scale_k(row,k)*
                        // output_scale_k(col,k) -- see disldo_backward's
                        // scattered-path comment on the chain-rule fix this
                        // mirrors (get_scale already sums over rank).
                        combined_scale4[lj]     = weights.get_scale(row, col);
                        combined_imp_scale4[lj] = imp_scale * out_imp_scale;
                        quant4[lj] = quant_orig4[lj] = w_decoded_arr[lj];
                        ci4[lj] = imp_decoded_arr[lj] * combined_imp_scale4[lj];
                        for (std::size_t k = 0; k < rank; ++k) out_scale_k4[k][lj] = weights.get_output_scale_k(col, k);
                    }

                    // mcol_at(col,k) is a real 4-way (times rank) SCATTER
                    // if written every (b, lj) -- confirmed via
                    // -fopt-info-vec as the actual remaining blocker once
                    // the FP4_TABLE lookups above were already moved out of
                    // this loop (a gather was never the issue here once
                    // that happened; same finding, now generalized over
                    // rank). Accumulate into small local arrays across the
                    // whole batch loop instead -- pure register/stack
                    // traffic, no memory scatter -- and flush the real
                    // scatter writes once, after. mrow_at(row,k) isn't a
                    // scatter (same address for every lj already, a
                    // horizontal reduction), but gets the same
                    // local-accumulate-then-flush treatment for
                    // consistency and to keep it out of the hot loop too.
                    // Reused scratch memory -- explicit zero each tile
                    // visit (task #295, see FP8 branch's identical
                    // comment above).
                    const std::size_t tid_rank = static_cast<std::size_t>(tid) * rank;
                    const std::size_t tid_rank_tile = tid_rank * BLOCK4_TILE;
                    auto& srs = weights.scale_rank_scratch;
                    const Flat2DView mcol4_rank{srs.mcol_rank.data() + tid_rank_tile, BLOCK4_TILE};
                    double* mrow_local_k = srs.mrow_local_k.data() + tid_rank;
                    const Flat2DView mcol4_rank_contrib{srs.mcol_rank_contrib.data() + tid_rank_tile, BLOCK4_TILE};
                    double* mrow_local_k_contrib = srs.mrow_local_k_contrib.data() + tid_rank;
                    // AQRS gamma's own gradient (task #273/#283) -- folded
                    // into the shared mgamma_at/mgamma_at_contrib once this
                    // row finishes, matching mrow_local_k's own
                    // fold-into-mrow_at pattern below.
                    double* mgamma_local_k = srs.mgamma_local_k.data() + tid_rank;
                    double* mgamma_local_k_contrib = srs.mgamma_local_k_contrib.data() + tid_rank;
                    std::fill(mcol4_rank[0], mcol4_rank[0] + rank * BLOCK4_TILE, value_type(0));
                    std::fill(mrow_local_k, mrow_local_k + rank, 0.0);
                    std::fill(mcol4_rank_contrib[0], mcol4_rank_contrib[0] + rank * BLOCK4_TILE, value_type(0));
                    std::fill(mrow_local_k_contrib, mrow_local_k_contrib + rank, 0.0);
                    std::fill(mgamma_local_k, mgamma_local_k + rank, 0.0);
                    std::fill(mgamma_local_k_contrib, mgamma_local_k_contrib + rank, 0.0);
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
                            const Block4Vec beta2_v        = block4_vec_broadcast(beta2);
                            const Block4Vec eps_v          = block4_vec_broadcast(eps);
                            const Block4Vec min_decay_frac_v = block4_vec_broadcast(min_decay_frac);
                                const Block4Vec max_ci_v = block4_vec_broadcast(max_ci);
                            const Block4Vec max_abs_delta_v  = block4_vec_broadcast(max_abs_delta);
                            const Block4Vec combined_scale_v = block4_vec_load(combined_scale4);
                            // quant_start_v: FIXED for the whole batch loop
                            // -- see the scattered path's identical
                            // cw_start comment for the full rationale.
                            // quant_floor_v computed once from this fixed
                            // snapshot too (still exact/signed for every
                            // already-escaped synapse, see its own comment
                            // below).
                            const Block4Vec quant_start_v = block4_vec_load(quant4);
                            Block4Vec ci_v    = block4_vec_load(ci4);
                            Block4Vec quant_floor_v;
                            for (int lane = 0; lane < BLOCK4_TILE; ++lane)
                                quant_floor_v[lane] = (quant_start_v[lane] == 0.0f) ? zero_escape_eps : quant_start_v[lane];
                            Block4Vec g_agg_v = block4_vec_broadcast(0.0f);
                            Block4Vec contrib_agg_v = block4_vec_broadcast(0.0f);
                            // Per-rank-component vectors -- rank is small
                            // (1-2 in practice, capped at SCALE_RANK_MAX),
                            // kept as real Block4Vec accumulators so the
                            // column dimension stays 4-wide SIMD regardless
                            // of rank; only `rank` itself is a small plain
                            // loop, orthogonal to the SIMD lane dimension.
                            // mcol_acc_v_k/mcol_acc_v_k_contrib are the only
                            // TRUE cross-batch accumulators -- see FP8
                            // branch's identical comment above for the
                            // full rationale (task #295).
                            value_type* mcol_acc_raw = srs.mcol_acc_raw.data() + tid_rank_tile;
                            value_type* mcol_acc_raw_contrib = srs.mcol_acc_raw_contrib.data() + tid_rank_tile;
                            for (std::size_t k = 0; k < rank; ++k) {
                                block4_vec_store(mcol_acc_raw + k * BLOCK4_TILE, block4_vec_broadcast(0.0f));
                                block4_vec_store(mcol_acc_raw_contrib + k * BLOCK4_TILE, block4_vec_broadcast(0.0f));
                            }
                            const bool training = (learning_rate != value_type(0));
                            for (SIZE_TYPE b = 0; b < batch; ++b) {
                                const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                                value_type* mdx_row = mdx + static_cast<std::size_t>(b) * in_cols + row;
                                const Block4Vec dyv_v = block4_vec_load(
                                    output_grad + static_cast<std::size_t>(b) * n_out + col_base);
                                const Block4Vec g_v = dyv_v * block4_vec_broadcast(iv);
                                if (training) {
                                    // RMSprop-style: decayed EMA of g^2, damp by
                                    // sqrt(ci)+eps -- see disldo_backward's own
                                    // docstring for why (matches the scattered
                                    // per-synapse path above exactly).
                                    //
                                    // Additive forward-contribution
                                    // combination (contrib = x*quant_orig) --
                                    // see the scattered path's identical fix
                                    // for the full rationale and the proved
                                    // theorem
                                    // (Joint.combined_signal_strictly_informative,
                                    // sili__new/lean_proofs/importance_signal_information_gain/
                                    // SiliImportanceProof/ImportanceSignalInformationGain.lean).
                                    // FIXED snapshot for the whole batch --
                                    // see quant_start_v's own comment above.
                                    const Block4Vec contrib_v = (quant_start_v * combined_scale_v) * block4_vec_broadcast(iv);
                                    g_agg_v += g_v;
                                    contrib_agg_v += contrib_v;
                                    // quant_floor: real signed quant for
                                    // value_scale_k/output_scale_k's own
                                    // gradient, EXCEPT exactly at quant==0
                                    // (genuinely stuck, correct gradient
                                    // truly is zero, sign legitimately
                                    // free) where zero_escape_eps
                                    // substitutes a floor instead -- see
                                    // disldo_backward's scattered-path
                                    // comment for the full correctness
                                    // trace (CORRECTED: the old
                                    // unconditional `eps+|quant|` discarded
                                    // sign on every nonzero synapse, not
                                    // just stuck ones -- confirmed as the
                                    // root cause of a real zero-init
                                    // training failure). No whole-vector
                                    // compare-and-select op available for
                                    // Block4Vec (same reason block4_vec_sqrt
                                    // above is a plain per-lane loop, not a
                                    // SIMD intrinsic) -- correctness first,
                                    // matches that precedent exactly.
                                    // quant's OWN update above needs no
                                    // floor at all (S is nonzero-driven,
                                    // never multiplies a vanishing quant
                                    // factor into itself).
                                    // mrow_local_k accumulates in DOUBLE, one
                                    // horizontal-sum per (b,k) -- matches the
                                    // pre-SIMD code's own double-precision
                                    // accumulation exactly (unlike mcol,
                                    // which is float-precision, matching the
                                    // original scalar code -- see this
                                    // block's own precision note below;
                                    // quant_floor_v itself now computed once,
                                    // fixed, before this loop.
                                    for (std::size_t k = 0; k < rank; ++k) {
                                        // AQRS gamma -- see the scattered path's identical comment (near
                                        // gamma_k_arr's own declaration) for the full rationale.
                                        const Block4Vec value_scale_k_v = block4_vec_broadcast(value_scale_k[k]);
                                        const Block4Vec out_scale_k_v = block4_vec_load(out_scale_k4[k]);
                                        mrow_local_k[k] += static_cast<double>(block4_vec_hsum(quant_floor_v * out_scale_k_v * g_v)) * static_cast<double>(gamma_k_arr[k]);
                                        mgamma_local_k[k] += static_cast<double>(block4_vec_hsum(quant_floor_v * out_scale_k_v * value_scale_k_v * g_v));
                                        mrow_local_k_contrib[k] += static_cast<double>(block4_vec_hsum(quant_floor_v * out_scale_k_v * contrib_v)) * static_cast<double>(gamma_k_arr[k]);
                                        mgamma_local_k_contrib[k] += static_cast<double>(block4_vec_hsum(quant_floor_v * out_scale_k_v * value_scale_k_v * contrib_v));
                                        value_type* acc = mcol_acc_raw + k * BLOCK4_TILE;
                                        block4_vec_store(acc, block4_vec_load(acc) + quant_floor_v * value_scale_k_v * g_v * block4_vec_broadcast(gamma_k_arr[k]));
                                        value_type* acc_c = mcol_acc_raw_contrib + k * BLOCK4_TILE;
                                        block4_vec_store(acc_c, block4_vec_load(acc_c) + quant_floor_v * value_scale_k_v * contrib_v * block4_vec_broadcast(gamma_k_arr[k]));
                                    }
                                }
                                // w = quant*S, FIXED for the whole batch --
                                // see quant_start_v's own comment above.
                                const Block4Vec w_v = quant_start_v * combined_scale_v;
                                *mdx_row += block4_vec_hsum(w_v * dyv_v);
                            }
                            // ONE update, using the batch-aggregated g/contrib.
                            Block4Vec quant_v = quant_start_v;
                            if (training) {
                                ci_v = SynapsePolicyVec::update_ci(ci_v, g_agg_v, contrib_agg_v, beta2_v, min_decay_frac_v, max_ci_v);
                                // dL/d(quant) = g*S -- proper chain rule
                                // on true_w=quant*S (multiply, not
                                // divide -- see disldo_backward's
                                // scattered-path comment this mirrors
                                // exactly, and mcol_acc_v's own comment
                                // 2 blocks up).
                                const Block4Vec delta_v = SynapsePolicyVec::update_cw(
                                    g_agg_v, ci_v, combined_scale_v, effective_lr_v, eps_v,
                                    damp_by_importance, max_abs_delta_v, scale_invariant);
                                quant_v += delta_v;
                            }
                            block4_vec_store(quant4, quant_v);
                            block4_vec_store(ci4, ci_v);
                            for (std::size_t k = 0; k < rank; ++k) {
                                block4_vec_store(mcol4_rank[k], block4_vec_load(mcol_acc_raw + k * BLOCK4_TILE));
                                block4_vec_store(mcol4_rank_contrib[k], block4_vec_load(mcol_acc_raw_contrib + k * BLOCK4_TILE));
                            }
                        } else {
                            // Boundary tile-column (rare -- only the last
                            // one, when n_out isn't a multiple of
                            // BLOCK4_TILE): scalar bounds-checked fallback,
                            // not on the fast path, doesn't need SIMD. Same
                            // math as the SIMD path above.
                            value_type quant_start4[BLOCK4_TILE], quant_floor4[BLOCK4_TILE];
                            double g_agg4[BLOCK4_TILE] = {0.0}, contrib_agg4[BLOCK4_TILE] = {0.0};
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                quant_start4[lj] = quant4[lj];
                                quant_floor4[lj] = (quant4[lj] == value_type(0)) ? zero_escape_eps : quant4[lj];
                            }
                            for (SIZE_TYPE b = 0; b < batch; ++b) {
                                const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                                value_type* mdx_row = mdx + static_cast<std::size_t>(b) * in_cols + row;
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    if (!col_valid4[lj]) continue;
                                    const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col4[lj]];
                                    const value_type g   = dyv * iv;
                                    const value_type S   = combined_scale4[lj];
                                    if (learning_rate != value_type(0)) {
                                        // Additive forward-contribution
                                        // combination -- see the scattered
                                        // path's identical fix
                                        // (Joint.combined_signal_strictly_informative).
                                        // contrib/quant_floor use FIXED
                                        // batch-start snapshots -- see the
                                        // scattered path's cw_start comment
                                        // for the full rationale.
                                        const value_type contrib = iv * (quant_start4[lj] * S);
                                        g_agg4[lj] += static_cast<double>(g);
                                        contrib_agg4[lj] += static_cast<double>(contrib);
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            // AQRS gamma -- see the SIMD branch's identical comment above.
                                            mrow_local_k[k] += static_cast<double>(quant_floor4[lj]) * static_cast<double>(out_scale_k4[k][lj]) * static_cast<double>(gamma_k_arr[k]) * g;
                                            mcol4_rank[k][lj] += quant_floor4[lj] * value_scale_k[k] * gamma_k_arr[k] * g;
                                            mgamma_local_k[k] += static_cast<double>(quant_floor4[lj]) * static_cast<double>(out_scale_k4[k][lj]) * static_cast<double>(value_scale_k[k]) * g;
                                            mrow_local_k_contrib[k] += static_cast<double>(quant_floor4[lj]) * static_cast<double>(out_scale_k4[k][lj]) * static_cast<double>(gamma_k_arr[k]) * contrib;
                                            mcol4_rank_contrib[k][lj] += quant_floor4[lj] * value_scale_k[k] * gamma_k_arr[k] * contrib;
                                            mgamma_local_k_contrib[k] += static_cast<double>(quant_floor4[lj]) * static_cast<double>(out_scale_k4[k][lj]) * static_cast<double>(value_scale_k[k]) * contrib;
                                        }
                                    }
                                    *mdx_row += quant_start4[lj] * S * dyv;
                                }
                            }
                            // ONE update per lj, using the batch-aggregated g/contrib.
                            if (learning_rate != value_type(0)) {
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    if (!col_valid4[lj]) continue;
                                    const value_type S = combined_scale4[lj];
                                    const value_type g_agg = static_cast<value_type>(g_agg4[lj]);
                                    const value_type contrib_agg = static_cast<value_type>(contrib_agg4[lj]);
                                    ci4[lj] = SynapsePolicy::update_ci(ci4[lj], g_agg, contrib_agg, beta2, min_decay_frac, max_ci);
                                    quant4[lj] = quant_start4[lj] + SynapsePolicy::update_cw(
                                        g_agg, ci4[lj], S, effective_lr, eps, damp_by_importance, max_abs_delta,
                                        scale_invariant);
                                }
                            }
                        }
                    } else {
                        // Hypothetical non-float value_type (never actually
                        // instantiated in this codebase -- see the comment
                        // above): the bounds-checked array form, correct
                        // for both the full-tile and boundary cases via
                        // col_valid4 either way, so no full_tile_cols split
                        // needed here at all. Identical math to the
                        // boundary branch above.
                        value_type quant_start4[BLOCK4_TILE], quant_floor4[BLOCK4_TILE];
                        double g_agg4[BLOCK4_TILE] = {0.0}, contrib_agg4[BLOCK4_TILE] = {0.0};
                        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                            quant_start4[lj] = quant4[lj];
                            quant_floor4[lj] = (quant4[lj] == value_type(0)) ? zero_escape_eps : quant4[lj];
                        }
                        for (SIZE_TYPE b = 0; b < batch; ++b) {
                            const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                            value_type* mdx_row = mdx + static_cast<std::size_t>(b) * in_cols + row;
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                if (!col_valid4[lj]) continue;
                                const value_type dyv = output_grad[static_cast<std::size_t>(b) * n_out + col4[lj]];
                                const value_type g   = dyv * iv;
                                const value_type S   = combined_scale4[lj];
                                if (learning_rate != value_type(0)) {
                                    // Additive forward-contribution
                                    // combination -- see the scattered
                                    // path's identical fix
                                    // (Joint.combined_signal_strictly_informative).
                                    // contrib/quant_floor use FIXED
                                    // batch-start snapshots -- see the
                                    // scattered path's cw_start comment
                                    // for the full rationale.
                                    const value_type contrib = iv * (quant_start4[lj] * S);
                                    g_agg4[lj] += static_cast<double>(g);
                                    contrib_agg4[lj] += static_cast<double>(contrib);
                                    for (std::size_t k = 0; k < rank; ++k) {
                                        mrow_local_k[k] += static_cast<double>(quant_floor4[lj]) * static_cast<double>(out_scale_k4[k][lj]) * g;
                                        mcol4_rank[k][lj] += quant_floor4[lj] * value_scale_k[k] * g;
                                        mrow_local_k_contrib[k] += static_cast<double>(quant_floor4[lj]) * static_cast<double>(out_scale_k4[k][lj]) * contrib;
                                        mcol4_rank_contrib[k][lj] += quant_floor4[lj] * value_scale_k[k] * contrib;
                                    }
                                }
                                *mdx_row += quant_start4[lj] * S * dyv;
                            }
                        }
                        // ONE update per lj, using the batch-aggregated g/contrib.
                        if (learning_rate != value_type(0)) {
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                if (!col_valid4[lj]) continue;
                                const value_type S = combined_scale4[lj];
                                const value_type g_agg = static_cast<value_type>(g_agg4[lj]);
                                const value_type contrib_agg = static_cast<value_type>(contrib_agg4[lj]);
                                ci4[lj] = SynapsePolicy::update_ci(ci4[lj], g_agg, contrib_agg, beta2, min_decay_frac, max_ci);
                                quant4[lj] = quant_start4[lj] + SynapsePolicy::update_cw(
                                    g_agg, ci4[lj], S, effective_lr, eps, damp_by_importance, max_abs_delta,
                                    scale_invariant);
                            }
                        }
                    }
                    if (learning_rate != value_type(0)) {
                        for (std::size_t k = 0; k < rank; ++k) {
                            mrow_at(row, k) += mrow_local_k[k];
                            mrow_at_contrib(row, k) += mrow_local_k_contrib[k];
                            mgamma_at(k) += static_cast<value_type>(mgamma_local_k[k]);
                            mgamma_at_contrib(k) += static_cast<value_type>(mgamma_local_k_contrib[k]);
                        }
                        if constexpr (!StochasticRounding) {
                            // Deterministic: NOT gated by the SIMD fast path above --
                            // no deterministic block4_vec_quantize_fp4 SIMD kernel
                            // exists yet (only block4_vec_quantize_stochastic_fp4
                            // does), so this always uses the scalar fp4_quantize()
                            // codec, mirroring the scattered-CSR path's own
                            // if constexpr (StochasticRounding) branch above.
                            // Correctness first -- a SIMD deterministic variant
                            // would be a reasonable follow-up perf optimization,
                            // not needed for SparseLinearLayerDeterministic to
                            // actually BE deterministic, which this fixes: before
                            // this, block4 (dense/promoted) synapses ALWAYS used
                            // fp4_quantize_stochastic() here regardless of the
                            // StochasticRounding template parameter, silently
                            // making "Deterministic" layers non-deterministic
                            // whenever they touched block4 storage (confirmed via
                            // a standalone C++ repro: back-to-back runs of the
                            // exact same binary gave different final nnz purely
                            // from this unseeded/uncontrolled stochastic rounding,
                            // with NO memory corruption or uninitialized reads
                            // involved -- valgrind memcheck came back clean).
                            // quant4[lj] is stored directly, no division --
                            // it's already the code (see the chain-rule fix
                            // above); importance still needs its own ratio,
                            // unaffected by this fix.
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                if (!col_valid4[lj]) continue;
                                for (std::size_t k = 0; k < rank; ++k) {
                                    mcol_at(col4[lj], k) += mcol4_rank[k][lj];
                                    mcol_at_contrib(col4[lj], k) += mcol4_rank_contrib[k][lj];
                                }
                                const value_type imp_ratio = ci4[lj] / combined_imp_scale4[lj];
                                // was_live4[lj] gate -- see its own
                                // declaration comment above for the full
                                // rationale.
                                const uint8_t new_w   = was_live4[lj] ? fp4_quantize_live(quant4[lj])  : fp4_quantize(quant4[lj]);
                                const uint8_t new_imp = was_live4[lj] ? fp4_quantize_live(imp_ratio)   : fp4_quantize(imp_ratio);
                                tdata[Block4Tile::slot_index(li, lj)] = uint8_t((new_imp << 4) | new_w);
                                tile_dirty = true;
                            }
                        } else if constexpr (std::is_same_v<value_type, float> && !SILI_BLOCK4_FORCE_SCALAR_BACKWARD) {
                            if (full_tile_cols) {
                                // Per-lane scalar quantize gated on
                                // was_live4[lj] (see its declaration comment
                                // above) -- NOT the SIMD
                                // block4_vec_quantize_stochastic_fp4_live
                                // kernel here, since that applies the same
                                // choice uniformly to all 4 lanes and can't
                                // express "some lanes live, some not" without
                                // a mask-blend this correctness fix doesn't
                                // need yet (accumulation above stays fully
                                // vectorized; only the encode step is
                                // scalarized). quant4 is already the code to
                                // encode (no division -- see the chain-rule
                                // fix above); importance still needs its own
                                // ratio. Safe to divide combined_imp_scale4
                                // unconditionally here (unlike the scalar
                                // fallback's per-lj guard) -- full_tile_cols
                                // means every lane is valid, so
                                // combined_imp_scale4 is never the
                                // invalid-lane 0 that would make this a 0/0
                                // division.
                                uint8_t new_w_codes[BLOCK4_TILE], new_imp_codes[BLOCK4_TILE];
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    const value_type imp_ratio = ci4[lj] / combined_imp_scale4[lj];
                                    if (was_live4[lj]) {
                                        new_w_codes[lj]   = fp4_quantize_stochastic_live(quant4[lj]);
                                        new_imp_codes[lj] = fp4_quantize_stochastic_live_nonneg(imp_ratio);
                                    } else {
                                        new_w_codes[lj]   = fp4_quantize_stochastic(quant4[lj]);
                                        new_imp_codes[lj] = fp4_quantize_stochastic(imp_ratio);
                                    }
                                }
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    for (std::size_t k = 0; k < rank; ++k) {
                                    mcol_at(col4[lj], k) += mcol4_rank[k][lj];
                                    mcol_at_contrib(col4[lj], k) += mcol4_rank_contrib[k][lj];
                                }
                                    tdata[Block4Tile::slot_index(li, lj)] = uint8_t((new_imp_codes[lj] << 4) | new_w_codes[lj]);
                                }
                                tile_dirty = true;
                            } else {
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    if (!col_valid4[lj]) continue;
                                    for (std::size_t k = 0; k < rank; ++k) {
                                    mcol_at(col4[lj], k) += mcol4_rank[k][lj];
                                    mcol_at_contrib(col4[lj], k) += mcol4_rank_contrib[k][lj];
                                }
                                    const value_type imp_ratio = ci4[lj] / combined_imp_scale4[lj];
                                    uint8_t new_w, new_imp;
                                    if (was_live4[lj]) {
                                        new_w   = fp4_quantize_stochastic_live(quant4[lj]);
                                        new_imp = fp4_quantize_stochastic_live_nonneg(imp_ratio);
                                    } else {
                                        new_w   = fp4_quantize_stochastic(quant4[lj]);
                                        new_imp = fp4_quantize_stochastic(imp_ratio);
                                    }
                                    tdata[Block4Tile::slot_index(li, lj)] = uint8_t((new_imp << 4) | new_w);
                                    tile_dirty = true;
                                }
                            }
                        } else {
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                if (!col_valid4[lj]) continue;
                                for (std::size_t k = 0; k < rank; ++k) {
                                    mcol_at(col4[lj], k) += mcol4_rank[k][lj];
                                    mcol_at_contrib(col4[lj], k) += mcol4_rank_contrib[k][lj];
                                }
                                const value_type imp_ratio = ci4[lj] / combined_imp_scale4[lj];
                                uint8_t new_w, new_imp;
                                if (was_live4[lj]) {
                                    new_w   = fp4_quantize_stochastic_live(quant4[lj]);
                                    new_imp = fp4_quantize_stochastic_live_nonneg(imp_ratio);
                                } else {
                                    new_w   = fp4_quantize_stochastic(quant4[lj]);
                                    new_imp = fp4_quantize_stochastic(imp_ratio);
                                }
                                tdata[Block4Tile::slot_index(li, lj)] = uint8_t((new_imp << 4) | new_w);
                                tile_dirty = true;
                            }
                        }
                    }
                } // closes for (li...)
                return tile_dirty;
            }; // closes process_tile

            if (learning_rate == value_type(0)) {
                // Read-only: process_tile structurally never writes here
                // (every write inside it is gated by learning_rate != 0),
                // so no resize can ever happen -- the plain shared-store
                // at_index() path has no concurrency hazard at all in
                // this case (see Block4Store::RowWorkspace's comment on
                // why that hazard is specifically about growth). Skips
                // the row-workspace snapshot+merge-back entirely, since
                // it would just copy the row's bytes in and back out
                // unchanged -- measured real overhead for zero benefit
                // when nothing is ever written (see conversation).
                auto bc_cursor = weights.block4.row_cursor(br);
                std::size_t elem_pos = BL4.elem_start[br];
                std::size_t byte_pos = weights.block4.tile_byte_start[br];
                for (std::size_t ti = row_ti_start[br]; ti < row_ti_start[br + 1]; ++ti, ++elem_pos) {
                    const uint32_t bc = bc_cursor.advance();
                    const auto tile = weights.block4.at_index(uint32_t(br), bc, elem_pos, byte_pos);
                    // const_cast: safe specifically because this branch
                    // only runs when learning_rate == value_type(0),
                    // under which process_tile provably never writes
                    // through tdata (every write is gated by the same
                    // condition inside it) -- not a general-purpose cast,
                    // just avoiding a second (const-parametrized) copy of
                    // process_tile for a write that can't happen here.
                    process_tile(bc, const_cast<uint8_t*>(tile.raw_data()));
                    byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
                }
            } else {
                // Writing: row-local workspace -- see the comment above
                // this whole block4 section (row_ti_start's comment) and
                // Block4Store::RowWorkspace's own comment for why this is
                // necessary under concurrency, not just an optimization.
                auto ws = weights.block4.snapshot_row(br);
                std::size_t live_byte_pos = 0;
                for (std::size_t ti = row_ti_start[br]; ti < row_ti_start[br + 1]; ++ti) {
                    const std::size_t e  = ti - row_ti_start[br];
                    const uint32_t    bc = ws.bc[e];
                    const std::size_t this_byte_pos = live_byte_pos;
                    // Sized per VALUES_TYPE: FP8's Block4Tile8 is 32
                    // bytes/tile (2/slot), not FP4's 16 (1/slot) --
                    // a real stack buffer overflow otherwise (caught via
                    // -fsanitize=address / gcc's own stringop-overflow
                    // warning when this was still hardcoded to
                    // BLOCK4_TILE_SLOTS for both types).
                    uint8_t scratch_buf[std::is_same_v<VALUES_TYPE, FP8BiValues> ? BLOCK4_TILE_SLOTS8_BYTES : BLOCK4_TILE_SLOTS];
                    weights.block4.unpack_workspace_tile(ws, e, this_byte_pos, scratch_buf);
                    const bool tile_dirty = process_tile(bc, scratch_buf);
                    if (tile_dirty)
                        weights.block4.commit_dirty_tile_in_workspace(ws, e, this_byte_pos, scratch_buf);
                    if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>)
                        live_byte_pos += block4_stored_tile_len8(ws.is_sparse[e], &ws.bytes[this_byte_pos]);
                    else
                        live_byte_pos += block4_stored_tile_len(ws.is_sparse[e], &ws.bytes[this_byte_pos]);
                } // closes for (ti...)
                // Merge back -- evicts lowest-|true-importance| synapses
                // only if this row genuinely grew past its own current
                // headroom (see Block4Store::merge_row_workspace's
                // comment).
                weights.block4.merge_row_workspace(br, ws,
                    [&](std::size_t ev_row, std::size_t ev_col, uint8_t ev_imp_code) -> double {
                        const value_type imp_scale     = weights.get_importance_scale(ev_row);
                        const value_type out_imp_scale = weights.get_output_importance_scale(ev_col);
                        // FP8's Block4Store8::merge_row_workspace passes the
                        // FULL importance byte (an E4M3 code), not FP4's
                        // 4-bit nibble -- see Block4Store8::merge_row_workspace's
                        // own docstring (block4.hpp).
                        const double decoded = std::is_same_v<VALUES_TYPE, FP8BiValues>
                            ? static_cast<double>(fp8_decode_bits(ev_imp_code))
                            : static_cast<double>(FP4_TABLE[ev_imp_code & 0xFu]);
                        return decoded * static_cast<double>(imp_scale) * static_cast<double>(out_imp_scale);
                    });
            }
            } // closes for (br...)
        } // closes #pragma omp parallel

        if (learning_rate != value_type(0)) {
            for (std::size_t row = 0; row < n_in; ++row) {
                const uint32_t nnz_row = row_live_count[row];
                if (nnz_row == 0) continue;
                const value_type scale_eff_lr = learning_rate / static_cast<value_type>(nnz_row);
                // t_row_grad is laid out [thread][row][k] (see its sizing
                // above, matching t_col_grad's own layout). rank=1
                // reproduces the exact original single-sum formula.
                for (std::size_t k = 0; k < rank; ++k) {
                    double sum = 0.0, sum_contrib = 0.0;
                    for (int t = 0; t < num_cpus; ++t) {
                        sum += t_row_grad[(static_cast<std::size_t>(t) * n_in + row) * rank + k];
                        sum_contrib += t_row_grad_contrib[(static_cast<std::size_t>(t) * n_in + row) * rank + k];
                    }
                    // Only skip when BOTH are zero -- a zero real gradient
                    // (sum==0) with nonzero forward-contribution signal
                    // still needs to update vs_imp (same zero-escape
                    // property as per-synapse ci, see linear_disldo.hpp's
                    // additive combination -- a dy=0 backward call should
                    // still move importance from activity alone).
                    if (sum == 0.0 && sum_contrib == 0.0) continue;
                    const value_type g_agg = static_cast<value_type>(sum);
                    const value_type contrib_agg = static_cast<value_type>(sum_contrib);
                    // Hand-inlined here rather than routed through ScalePolicy
                    // (block4 path predates that abstraction) -- same NaN/Inf
                    // guard as ScalePolicy::update and for the same reason:
                    // sum (double, accumulated across possibly many more live
                    // synapses per row under dense connectivity) can overflow
                    // to Inf when narrowed to value_type, and vs_imp's EMA
                    // never decays a stray Inf back down, so once-off overflow
                    // becomes permanent corruption. See delta_csr_types.hpp's
                    // RMSpropScalePolicy::update docstring for the full trace
                    // (sili_peridot, JOURNAL.md 2026-08-10).
                    if (!std::isfinite(g_agg) || !std::isfinite(contrib_agg)) continue;
                    const std::size_t idx = row * rank + k;
                    value_type& vs_imp = weights.value_scale_importance[idx];
                    // Square-then-sum (g_agg^2+contrib_agg^2), matching
                    // RMSpropScalePolicy::update's own combination -- see
                    // its docstring: sum-then-square lets a large-magnitude
                    // g_agg/contrib_agg disagreement collapse this toward
                    // zero, which would make the step below explode
                    // (dividing by a near-zero denominator) -- exactly the
                    // same class of instability the bias-correction fix
                    // above closes, just triggered by cancellation instead
                    // of cold start. Square-then-sum is bounded below by
                    // max(g_agg,contrib_agg)^2 regardless of sign, so a
                    // large disagreement still damps the step instead of
                    // amplifying it.
                    const value_type new_vs_imp = beta2 * vs_imp + (value_type(1) - beta2) * (g_agg * g_agg + contrib_agg * contrib_agg);
                    if (!std::isfinite(new_vs_imp)) continue;
                    // Same Adam-style bias correction as
                    // RMSpropScalePolicy::update (delta_csr_types.hpp) --
                    // this path updates the SAME value_scale_importance
                    // array via a separately hand-inlined formula (block4
                    // predates the ScalePolicy abstraction), so it needs
                    // the identical fix or it reintroduces the exact
                    // cold-start bug this whole change exists to close.
                    uint32_t& step = weights.get_value_scale_step_k(row, k);
                    ++step;
                    const value_type bias_correction = value_type(1) - std::pow(beta2, static_cast<value_type>(step));
                    const value_type vs_imp_hat = bias_correction > value_type(0) ? new_vs_imp / bias_correction : new_vs_imp;
                    if (!std::isfinite(vs_imp_hat)) continue;
                    const value_type new_vs = weights.value_scale[idx] - scale_eff_lr * g_agg / (std::sqrt(vs_imp_hat) + eps);
                    if (!std::isfinite(new_vs)) continue;
                    vs_imp = new_vs_imp;
                    weights.value_scale[idx] = new_vs;
                }
            }
        }
    }

    // Dead-row (combined scattered CSR + block4) value_scale bootstrap:
    // a row with ZERO live synapses in EITHER representation gets no
    // gradient through either path above (both skip it via their own
    // nnz==0 checks), so value_scale/value_scale_importance would stay
    // frozen forever -- blocking any hope of bootstrapping predictions
    // from a genuine zero-synapse init (not the same as all_zero_init,
    // which keeps full block4 connectivity with weight=0 -- a TRULY
    // zero-synapse row has no tile/synapse allocated at all). Checked
    // here, after both the scattered and block4 sections above, using
    // COMBINED liveness -- a row live in ONE representation but not the
    // other is already correctly handled by its own path and must not
    // be double-touched here. Deliberately NOT nested inside
    // `if (weights.block4.n_tiles() > 0)` -- a layer with zero block4
    // tiles (pure scattered CSR, or genuinely empty) is exactly the
    // fresh zero-synapse-init case this exists for, and must still run.
    //
    // g_agg = input[row] * S[b], S[b] = sum_col(output_scale[col]*
    // output_grad[b,col]) -- exact (distributive law), not an
    // approximation, and S[b] doesn't depend on row, so it costs
    // O(batch*n_out) ONCE regardless of how many rows are dead (only
    // paid at all if at least one row actually is), not an
    // O(n_in_dead*n_out) dense rescan per row -- real MiniCPM5-scale
    // layers can't afford the latter even while mostly zero-synapse.
    //
    // Drives value_scale via a standard two-moment Adam step
    // (value_scale_momentum = first moment, value_scale_importance
    // reused as second moment -- safe here since it's provably untouched
    // by both paths above whenever a row is dead in both) instead of a
    // single-moment RMSprop step -- linear in g_agg (not g_agg^2), so
    // E[update]=0 under zero-mean noise regardless of variance, letting
    // a genuinely inconsistent signal cancel out and stay at zero
    // (preserving sparsity) while a persistent bias still accumulates.
    if (learning_rate != value_type(0)) {
        auto block4_row_live = [&](std::size_t row) -> std::size_t {
            const auto& bl = weights.block4.block_layout;
            if (bl.rows == 0) return 0;
            const std::size_t br = row / BLOCK4_TILE;
            if (br >= bl.rows) return 0;
            return bl.row_nnz(br) * BLOCK4_TILE;
        };
        bool any_dead_row = false;
        for (std::size_t row = 0; row < n_in; ++row) {
            if (L.row_nnz(row) == 0 && block4_row_live(row) == 0) { any_dead_row = true; break; }
        }
        if (any_dead_row) {
            // dead_row_S[b][k] = sum_col(output_scale_k(col,k)*output_grad
            // [b,col]) -- one independent sum per rank component (rank=1
            // reproduces the exact original single-sum formula). Still
            // row-independent, so still O(batch*n_out*rank) ONCE, not
            // O(n_in_dead*n_out*rank) per row.
            std::vector<double> dead_row_S(static_cast<std::size_t>(batch) * rank, 0.0);
            for (SIZE_TYPE b = 0; b < batch; ++b) {
                for (std::size_t k = 0; k < rank; ++k) {
                    double s = 0.0;
                    for (std::size_t col = 0; col < n_out; ++col) {
                        s += static_cast<double>(weights.get_output_scale_k(col, k))
                           * static_cast<double>(output_grad[static_cast<std::size_t>(b) * n_out + col]);
                    }
                    dead_row_S[static_cast<std::size_t>(b) * rank + k] = s;
                }
            }
            for (std::size_t row = 0; row < n_in; ++row) {
                if (L.row_nnz(row) != 0 || block4_row_live(row) != 0) continue;
                for (std::size_t k = 0; k < rank; ++k) {
                    double sum = 0.0;
                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                        if (iv == value_type(0)) continue;
                        sum += static_cast<double>(iv) * dead_row_S[static_cast<std::size_t>(b) * rank + k];
                    }
                    if (sum == 0.0) continue;
                    const value_type g_agg = static_cast<value_type>(sum);
                    if (!std::isfinite(g_agg)) continue;
                    const std::size_t idx = row * rank + k;
                    value_type& m = weights.value_scale_momentum[idx];
                    value_type& v = weights.value_scale_importance[idx];
                    const value_type new_m = beta1 * m + (value_type(1) - beta1) * g_agg;
                    const value_type new_v = beta2 * v + (value_type(1) - beta2) * g_agg * g_agg;
                    if (!std::isfinite(new_m) || !std::isfinite(new_v)) continue;
                    const value_type dead_row_lr = learning_rate / static_cast<value_type>(n_out);
                    const value_type new_vs = weights.value_scale[idx]
                        - dead_row_lr * new_m / (std::sqrt(new_v) + eps);
                    if (!std::isfinite(new_vs)) continue;
                    m = new_m;
                    v = new_v;
                    weights.value_scale[idx] = new_vs;
                }
            }
        }
    }

    for (int t = 0; t < num_cpus; ++t) {
        const value_type* s = t_dx.data() + static_cast<std::size_t>(t) * dst;
        for (std::size_t i = 0; i < dst; ++i) {
            input_grad[i] += s[i];
        }
    }

    if (learning_rate != value_type(0) && output_scale_trainable) {
        // output_scale[c]'s gradient, reduced across threads then applied
        // once per column -- same "sum first, apply lr once" reasoning as
        // value_scale's own update. Normalizes by out_degree[c] (how many
        // rows feed this output), the column-axis equivalent of
        // nnz_this_row; a column with zero connections is skipped. Now
        // per-component (rank>1, see scale_rank's own docstring) --
        // t_col_grad is laid out [thread][col][k].
        for (std::size_t c = 0; c < n_out; ++c) {
            const std::size_t deg = c < weights.out_degree.size()
                ? static_cast<std::size_t>(weights.out_degree[c]) : 0;
            if (deg == 0) continue;
            const value_type col_eff_lr = learning_rate / static_cast<value_type>(deg);
            for (std::size_t k = 0; k < rank; ++k) {
                double col_grad_sum = 0.0;
                double col_grad_sum_contrib = 0.0;
                for (int t = 0; t < num_cpus; ++t) {
                    col_grad_sum += t_col_grad[static_cast<std::size_t>(t) * n_out * rank + c * rank + k];
                    col_grad_sum_contrib += t_col_grad_contrib[static_cast<std::size_t>(t) * n_out * rank + c * rank + k];
                }
                // Scale update via the swappable policy -- same as
                // value_scale's own update above.
                const value_type g_agg = static_cast<value_type>(col_grad_sum);
                const value_type contrib_agg = static_cast<value_type>(col_grad_sum_contrib);
                ScalePolicy::update(weights.output_scale[c * rank + k], weights.output_scale_importance[c * rank + k],
                                    g_agg, col_eff_lr, beta2, eps, contrib_agg,
                                    &weights.get_output_scale_step_k(c, k), scale_invariant);
            }
        }
    }

    if (learning_rate != value_type(0) && weights.scale_gamma_is_trainable) {
        // AQRS gamma's own update (task #273/#283, see sili_peridot/
        // AQRS_DESIGN.md Theorem 8): reduced across threads once per
        // channel k (gamma is layer-wide, no per-row/col normalization
        // needed, unlike value_scale/output_scale's deg-scaled eff_lr).
        // Gated on scale_gamma_is_trainable (same opt-in pattern as
        // output_scale_is_trainable) -- see scale_gamma_is_trainable's own
        // docstring, delta_csr_types.hpp, for why this gate is required,
        // not optional: without it every existing rank>=1 layer, including
        // ones that have never heard of gamma, would get an unsolicited
        // gradient-driven perturbation to gamma_s_k(0) every step.
        // Same ScalePolicy convention (RMSprop default) for the gradient
        // step, THEN a proximal L1 soft-threshold shrinkage on top --
        // that second step is what creates a genuine attracting fixed
        // point at exactly gamma=0 (soft-thresholding zeroes anything
        // within l1_coef*learning_rate of zero after the gradient step;
        // plain RMSprop/L2-style decay only asymptotically approaches
        // zero, never reaches it exactly, which is why L1 needs its own
        // explicit step here rather than folding into ScalePolicy's own
        // gradient-only update).
        // g_agg_by_k captured here for the EMA pass below (task #284) --
        // that pass needs every channel's raw gradient AND every
        // channel's just-updated |gamma| value simultaneously (C_k needs
        // ||gamma||_1 over ALL k), so it can't be folded into this same
        // loop. Plain vector (task #295) -- allocated once per call, same
        // frequency as t_gamma_grad's own vector above, not a hot-loop cost.
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
            // Force-size scale_gamma up to k (lazy default preserved) so a
            // direct reference is safe to hand to ScalePolicy::update,
            // matching value_scale/output_scale's own direct-array-access
            // convention above.
            weights.set_scale_gamma_raw_k(k, weights.get_scale_gamma_k(k));
            ScalePolicy::update(weights.scale_gamma[k], weights.get_scale_gamma_state_k(k),
                                g_agg, learning_rate, beta2, eps, contrib_agg,
                                &weights.get_scale_gamma_step_k(k), scale_invariant);
            // L1 only applies to k>=1 -- channel 0 is the always-on
            // baseline (set_scale_rank rejects rank==0, so channel 0 can
            // NEVER actually be pruned regardless of how small its gamma
            // gets); penalizing it anyway just fights the fit with no
            // possible payoff. Matches the k==0-is-special convention
            // already used everywhere else in this rank-N mechanism
            // (value_scale/output_scale's own default, gamma's own
            // lazy-transparent default).
            if (l1_coef > value_type(0) && k > 0) {
                const value_type shrink = l1_coef * learning_rate;
                value_type& gm = weights.scale_gamma[k];
                if (gm > shrink) gm -= shrink;
                else if (gm < -shrink) gm += shrink;
                else gm = value_type(0);
            }
        }

        // AQRS dynamic rank control (task #273/#284): EMA-smoothed
        // |gamma_k|/C_k/|grad_k| tracking, updated EVERY step -- see
        // AQRS_DESIGN.md's corrected noise-mitigation design (EMA every
        // step is the actual noise filter, periodic N-step checking is
        // rejected as a "luck filter"). Second pass, after every
        // channel's gamma value is finalized above -- C_k = |gamma_k| /
        // sum_j|gamma_j| needs every channel's CURRENT value first.
        {
            value_type gamma_l1_sum = value_type(0);
            for (std::size_t k = 0; k < rank; ++k) gamma_l1_sum += std::fabs(weights.scale_gamma[k]);
            // g_agg_by_k[k] is a RAW per-synapse-accumulated sum (mgamma_at
            // above, summed across every touched row/col pair) -- same
            // layer-width-dependent scaling issue as the additive branch's
            // own dgamma_by_k (see that block's own comment, task #294
            // fix); normalize by n_in*n_out here for the SAME reason, same
            // scope (trigger tracking only, not gamma's own ScalePolicy
            // step above).
            const value_type grad_norm_divisor = static_cast<value_type>(n_in) * static_cast<value_type>(n_out);
            for (std::size_t k = 0; k < rank; ++k) {
                const value_type abs_gamma_k = std::fabs(weights.scale_gamma[k]);
                const value_type share_k = gamma_l1_sum > value_type(0) ? abs_gamma_k / gamma_l1_sum : value_type(0);
                weights.update_scale_gamma_ema_k(k, abs_gamma_k, share_k, std::fabs(g_agg_by_k[k]) / grad_norm_divisor);
            }
        }
    }

    // Deferred-store replay: only the scattered-path entries buffered
    // above (block4 stays untouched by DeferredScaleWrite, see this
    // function's own docstring) -- now that value_scale[row] AND
    // output_scale[col] are BOTH fully finalized for this call (every
    // row's update above, block4's own update if it ran, and the
    // output_scale reduction just above), write each buffered entry's
    // true-units (cw, ci) back out under the scale that's actually in
    // effect now, not the stale one from when it was computed.
    if constexpr (DeferredScaleWrite) {
        if (learning_rate != value_type(0)) {
            for (int t = 0; t < num_cpus; ++t) {
                for (const auto& entry : t_deferred[static_cast<std::size_t>(t)]) {
                    const value_type final_val_scale = weights.get_value_scale(entry.row);
                    const value_type final_out_scale = weights.get_output_scale(entry.col);
                    const value_type final_imp_scale = weights.get_importance_scale(entry.row);
                    const value_type final_out_imp_scale = weights.get_output_importance_scale(entry.col);
                    const value_type final_combined_scale = final_val_scale * final_out_scale;
                    const value_type final_combined_imp_scale = final_imp_scale * final_out_imp_scale;
                    if constexpr (StochasticRounding) {
                        ValueAccessor<VALUES_TYPE>::set_stochastic_live(
                            dc.values, entry.vb,
                            entry.cw / final_combined_scale, entry.ci / final_combined_imp_scale);
                    } else {
                        ValueAccessor<VALUES_TYPE>::set_live(
                            dc.values, entry.vb,
                            entry.cw / final_combined_scale, entry.ci / final_combined_imp_scale);
                    }
                }
            }
        }
    }

    // ── AQRS additive branch backward (task #277, see sili_peridot/
    // AQRS_DESIGN.md) -- differentiates disldo_forward's own additive
    // -branch block. Genuinely independent of the sparse/block4 structure
    // above (same reasoning as forward), computed as its own self
    // -contained pass. No-op at the default additive_rank==0.
    //
    // Forward recap: P[b,k] = sum_r U[r,k]*X[b,r]; Y[b,c] += sum_k
    // V[c,k]*P[b,k]. Standard backprop through that:
    //   dV[c,k]  = sum_b dY[b,c]*P[b,k]
    //   dP[b,k]  = sum_c dY[b,c]*V[c,k]
    //   dU[r,k]  = sum_b dP[b,k]*X[b,r]
    //   dX[b,r] += sum_k dP[b,k]*U[r,k]
    // P is NOT cached from forward (this function has no access to
    // forward's locals, and this codebase's own convention elsewhere is
    // to recompute from `input` rather than carry hidden state across the
    // forward/backward call boundary) -- recomputed here from `input`
    // directly, cheap given additive_rank is small.
    //
    // Adam-style default (AdamScalePolicy, delta_csr_types.hpp) applied
    // per component AFTER the whole-batch gradient is aggregated (dU_rk/
    // dV_ck below sum over every sample in the batch first) -- matches
    // how every other per-scale update in this function works (one
    // update per call, not once per batch sample).
    if (weights.additive_rank > 0) {
        const std::size_t r_o = weights.additive_rank;
        std::vector<value_type> P(static_cast<std::size_t>(batch) * r_o, value_type(0));
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* in_row = input + static_cast<std::size_t>(b) * in_cols;
            value_type* p_row = P.data() + static_cast<std::size_t>(b) * r_o;
            for (std::size_t r = 0; r < n_in; ++r) {
                const value_type iv = in_row[r];
                if (iv == value_type(0)) continue;
                for (std::size_t k = 0; k < r_o; ++k)
                    p_row[k] += weights.get_additive_u_k(r, k) * iv;
            }
        }
        std::vector<value_type> dP(static_cast<std::size_t>(batch) * r_o, value_type(0));
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* dy_row = output_grad + static_cast<std::size_t>(b) * n_out;
            value_type* dp_row = dP.data() + static_cast<std::size_t>(b) * r_o;
            for (std::size_t c = 0; c < n_out; ++c) {
                const value_type dy = dy_row[c];
                if (dy == value_type(0)) continue;
                for (std::size_t k = 0; k < r_o; ++k)
                    dp_row[k] += weights.get_additive_v_k(c, k) * dy;
            }
        }
        // dP/P above are the RAW (un-gamma'd) projections -- real dL/dP_k
        // = gamma_k * dP_raw[b,k] (see task #289's derivation, sili_peridot/
        // conversation): Y_k[b,c] = gamma_k*V[c,k]*P_k[b], so
        //   dX          = sum_k U[r,k] * gamma_k * dP_raw[b,k]
        //   dU[r,k]     = gamma_k * sum_b dP_raw[b,k]*X[b,r]
        //   dV[c,k]     = gamma_k * sum_b dY[b,c]*P[b,k]
        //   dgamma_k    = sum_b P[b,k] * dP_raw[b,k]   (reuses P/dP as-is,
        //                 cheap -- no gamma factor here, gamma_k IS the
        //                 thing being differentiated)
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* dp_row = dP.data() + static_cast<std::size_t>(b) * r_o;
            value_type* dx_row = input_grad + static_cast<std::size_t>(b) * in_cols;
            for (std::size_t r = 0; r < n_in; ++r) {
                value_type acc = value_type(0);
                for (std::size_t k = 0; k < r_o; ++k)
                    acc += weights.get_additive_gamma_k(k) * weights.get_additive_u_k(r, k) * dp_row[k];
                dx_row[r] += acc;
            }
        }
        for (std::size_t r = 0; r < n_in; ++r) {
            for (std::size_t k = 0; k < r_o; ++k) {
                value_type dU_rk = value_type(0);
                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                    if (iv == value_type(0)) continue;
                    dU_rk += dP[static_cast<std::size_t>(b) * r_o + k] * iv;
                }
                dU_rk *= weights.get_additive_gamma_k(k);
                if (dU_rk == value_type(0)) continue;
                value_type u_val = weights.get_additive_u_k(r, k);
                AdamScalePolicy<value_type>::update(
                    u_val, weights.get_additive_u_state_k(r, k), weights.get_additive_u_momentum_k(r, k),
                    dU_rk, learning_rate, beta1, beta2, eps, &weights.get_additive_u_step_k(r, k));
                weights.set_additive_u_raw_k(r, k, u_val);
            }
        }
        for (std::size_t c = 0; c < n_out; ++c) {
            for (std::size_t k = 0; k < r_o; ++k) {
                value_type dV_ck = value_type(0);
                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type dy = output_grad[static_cast<std::size_t>(b) * n_out + c];
                    if (dy == value_type(0)) continue;
                    dV_ck += dy * P[static_cast<std::size_t>(b) * r_o + k];
                }
                dV_ck *= weights.get_additive_gamma_k(k);
                if (dV_ck == value_type(0)) continue;
                value_type v_val = weights.get_additive_v_k(c, k);
                AdamScalePolicy<value_type>::update(
                    v_val, weights.get_additive_v_state_k(c, k), weights.get_additive_v_momentum_k(c, k),
                    dV_ck, learning_rate, beta1, beta2, eps, &weights.get_additive_v_step_k(c, k));
                weights.set_additive_v_raw_k(c, k, v_val);
            }
        }

        // AQRS additive_gamma's own update (task #289, mirrors scale_
        // gamma's own update block above exactly -- same ScalePolicy
        // -then-L1-then-EMA structure, same reasoning throughout, see
        // that block's own comments for the full rationale). Gated on
        // additive_gamma_is_trainable (same opt-in pattern as scale_
        // gamma_is_trainable) so a caller that's never touched gamma
        // never gets an unsolicited perturbation to the transparent 1.0
        // default. Uses the function's generic `ScalePolicy` template
        // param (same RMSprop-style, momentum-free policy scale_gamma
        // itself uses) -- NOT AdamScalePolicy, despite additive_u/v using
        // Adam. Found via a real test failure (see conversation): Adam's
        // momentum overshoots the L1-created zero fixed point (Theorem
        // 8), driving gamma persistently negative instead of settling
        // exactly at 0. gamma needs the SAME exact-zero-fixed-point
        // property in both branches, so it uses the SAME policy in both;
        // only the direction vectors get their own independent optimizer
        // choice. log_space=false unconditionally (not threading this
        // function's own `scale_invariant` parameter here -- that flag's
        // meaning is specifically about the multiplicative branch's
        // coupling with the quantized weight, which the additive branch
        // doesn't have).
        //
        // UNLIKE scale_gamma's k>0 L1 exemption: additive_rank has NO
        // legacy always-on channel to protect (min_rank=0 in apply_
        // additive_dynamic_rank_control -- the branch can legitimately
        // shrink itself back to fully off), so L1 applies to every k
        // here, including k==0.
        if (learning_rate != value_type(0) && weights.additive_gamma_is_trainable) {
            std::vector<value_type> dgamma_by_k(r_o);
            for (std::size_t k = 0; k < r_o; ++k) {
                double dgamma_sum = 0.0;
                for (SIZE_TYPE b = 0; b < batch; ++b)
                    dgamma_sum += static_cast<double>(P[static_cast<std::size_t>(b) * r_o + k]) *
                                  static_cast<double>(dP[static_cast<std::size_t>(b) * r_o + k]);
                const value_type dgamma_k = static_cast<value_type>(dgamma_sum);
                dgamma_by_k[k] = dgamma_k;
                weights.set_additive_gamma_raw_k(k, weights.get_additive_gamma_k(k));
                ScalePolicy::update(weights.additive_gamma[k], weights.get_additive_gamma_state_k(k),
                                    dgamma_k, learning_rate, beta2, eps, value_type(0),
                                    &weights.get_additive_gamma_step_k(k), false);
                if (l1_coef > value_type(0)) {
                    const value_type shrink = l1_coef * learning_rate;
                    value_type& gm = weights.additive_gamma[k];
                    if (gm > shrink) gm -= shrink;
                    else if (gm < -shrink) gm += shrink;
                    else gm = value_type(0);
                }
            }
            value_type gamma_l1_sum = value_type(0);
            for (std::size_t k = 0; k < r_o; ++k) gamma_l1_sum += std::fabs(weights.additive_gamma[k]);
            // dgamma_by_k[k] = sum_b P[b,k]*dP[b,k] is a RAW, unnormalized
            // sum (P sums over n_in terms, dP sums over n_out terms) --
            // its magnitude scales with layer WIDTH, not just "how much
            // does this channel actually matter." A real MQAR run showed
            // grad_ema on a 128x128 layer ~9 orders of magnitude larger
            // than a 16x128 layer's, making theta (a single global
            // constant) meaningless across differently-shaped layers --
            // task #294 fix. Normalizing by n_in*n_out here (trigger
            // tracking ONLY -- gamma's own ScalePolicy step above is left
            // on the raw dgamma_k, since that update already self
            // -normalizes via its own second-moment estimate and existing
            // tests are tuned against it) makes theta comparable across
            // layer shapes.
            const value_type grad_norm_divisor = static_cast<value_type>(n_in) * static_cast<value_type>(n_out);
            for (std::size_t k = 0; k < r_o; ++k) {
                const value_type abs_gamma_k = std::fabs(weights.additive_gamma[k]);
                const value_type share_k = gamma_l1_sum > value_type(0) ? abs_gamma_k / gamma_l1_sum : value_type(0);
                weights.update_additive_gamma_ema_k(k, abs_gamma_k, share_k, std::fabs(dgamma_by_k[k]) / grad_norm_divisor);
            }
        }
    }
}
