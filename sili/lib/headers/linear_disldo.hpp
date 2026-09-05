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
// Generic over VALUES_TYPE via ValueAccessor (FP4BiPacked / 32-bit
// fallback). Dense-input walk is embarrassingly parallel by input row,
// unlike the sparse-input SISLDO path (sisldo_ops.hpp), which needs a
// work-offset table to balance threads across a variable-density CSR
// batch. See docs/research/linear_disldo.rst for background.

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
 * No learning_rate parameter -- forward is pure computation, no side
 * effects. Importance updates happen only in disldo_backward(), coupled
 * to a real gradient (see docs/research/linear_disldo.rst for why a
 * gradient-free forward-side importance update was removed).
 *
 * NOTE (test): output must equal the dense matmul input @ W_dense where
 * W_dense[r,c] = weight of synapse (r->c). Same reference check used
 * for sisldo_forward and disldo_ops.hpp.
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void disldo_forward(const typename ValueAccessor<VALUES_TYPE>::value_type* input, SIZE_TYPE batch,
                    SIZE_TYPE in_cols,
                    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
                    typename ValueAccessor<VALUES_TYPE>::value_type* output, int num_cpus = 4) {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;
    const auto& L = dc.layout;

    const std::size_t n_in = L.rows;
    const std::size_t n_out = L.cols;
    const std::size_t ost = static_cast<std::size_t>(batch) * n_out;

    // dc.empty() no longer means "nothing to do" -- block4 below may still
    // hold live synapses. See docs/research/linear_disldo.rst.
    if (!dc.empty()) {
        std::vector<value_type> t_out(static_cast<std::size_t>(num_cpus) * ost, value_type(0));

#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mo = t_out.data() + static_cast<std::size_t>(tid) * ost;

#pragma omp for schedule(static)
            for (std::size_t r = 0; r < n_in; ++r) {
                const std::size_t n_row = L.row_nnz(r);
                if (n_row == 0)
                    continue;

                auto cursor = dc.row_cursor(r);
                for (std::size_t e = 0; e < n_row; ++e) {
                    const COL_TYPE col = cursor.advance();
                    const std::size_t vb = L.elem_start[r] + e;
                    const value_type w_stored = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                    const value_type w =
                        w_stored * weights.get_scale(r, col); // rank-N scale -> true units

                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                        if (iv == value_type(0))
                            continue;
                        const value_type contrib = w * iv;
                        mo[static_cast<std::size_t>(b) * n_out + col] += contrib;
                    }
                }
            }
        }

        for (int t = 0; t < num_cpus; ++t) {
            const value_type* s = t_out.data() + static_cast<std::size_t>(t) * ost;
            for (std::size_t i = 0; i < ost; ++i)
                output[i] += s[i];
        }
    } // !dc.empty()

    // block4 contribution -- same shared per-row value_scale/output_scale
    // as the scattered path above. No gather: within an active tile,
    // position IS the column (fixed compile-time offset), which is what
    // lets this SIMD where the scattered loop above can't. See
    // docs/research/linear_disldo.rst.
    if (weights.block4.n_tiles() > 0) {
        // Collect (br,bc,elem_pos,byte_pos) tuples once per call before the
        // parallel region -- row-major cursor walk isn't parallel-for
        // friendly, and a Block4Tile handle is move-only/RAII so it can't
        // be pre-collected across threads. See docs/research/linear_disldo.rst.
        std::vector<uint32_t>& tile_br = weights.block4.scratch_tile_br;
        std::vector<uint32_t>& tile_bc = weights.block4.scratch_tile_bc;
        std::vector<std::size_t>& tile_elem = weights.block4.scratch_tile_elem;
        std::vector<std::size_t>& tile_byte = weights.block4.scratch_tile_byte;
        const std::size_t n_b4 = weights.block4.n_tiles();
        // resize()+direct indexing, not reserve()+push_back(): push_back's
        // per-call capacity check is measured exclusive cost at this scale.
        // See docs/research/linear_disldo.rst.
        tile_br.resize(n_b4);
        tile_bc.resize(n_b4);
        tile_elem.resize(n_b4);
        tile_byte.resize(n_b4);
        const auto& BL4 = weights.block4.block_layout;
        std::size_t ti = 0;
        for (std::size_t br = 0; br < BL4.rows; ++br) {
            const std::size_t n_bc = BL4.row_nnz(br);
            if (n_bc == 0)
                continue;
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
        // Per-thread private output buffers -- necessary, not optional:
        // two tiles sharing a block-column write to the same output
        // positions. See docs/research/linear_disldo.rst.
        std::vector<value_type> b4_out(static_cast<std::size_t>(num_cpus) * ost, value_type(0));
        // Hoisted loop bound -- measured instruction-count win (no
        // wall-clock effect). See docs/research/linear_disldo.rst.
        const int64_t n_tiles_local = int64_t(n_b4);
#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mo = b4_out.data() + static_cast<std::size_t>(tid) * ost;
#pragma omp for schedule(static)
            for (int64_t row_ti = 0; row_ti < n_tiles_local; ++row_ti) {
                const uint32_t br = tile_br[std::size_t(row_ti)], bc = tile_bc[std::size_t(row_ti)];
                // const: routes .at() through the const overload (no dirty
                // mark, forward is read-only). at_index() reuses the
                // coordinates the collection loop above already resolved.
                const auto tile = weights.block4.at_index(br, bc, tile_elem[std::size_t(row_ti)],
                                                          tile_byte[std::size_t(row_ti)]);
                // Resolved once per tile, not once per .at() call. See
                // Block4TileHandle::raw_data().
                const uint8_t* tdata = tile.raw_data();
                // Decode via block4_vec_decode_fp4 (bit-shift), not
                // FP4_TABLE[code] gathers -- and LJ is templated
                // (compile-time constant), not a runtime loop, since GCC
                // could not vectorize the runtime version at all. See
                // docs/research/linear_disldo.rst for the measured
                // rationale, including why the scalar-table result that
                // helped backward's decode did NOT transfer here.
                auto process_col = [&]<uint32_t LJ>() {
                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + LJ;
                    if (col >= n_out)
                        return;

                    // FP8 dispatch: Block4Tile8 is a full byte/slot (no
                    // nibble mask), decoded via E4M3 -- the only thing that
                    // differs from the FP4 branch below.
                    Block4Vec w_decoded;
                    if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                        const Block4VecU w_codes = {
                            uint32_t(tdata[Block4Tile8::slot_index(0, LJ)]),
                            uint32_t(tdata[Block4Tile8::slot_index(1, LJ)]),
                            uint32_t(tdata[Block4Tile8::slot_index(2, LJ)]),
                            uint32_t(tdata[Block4Tile8::slot_index(3, LJ)])};
                        w_decoded = block4_vec_decode_fp8(w_codes);
                    } else {
                        const Block4VecU w_codes = {
                            uint32_t(tdata[Block4Tile::slot_index(0, LJ)] & 0xFu),
                            uint32_t(tdata[Block4Tile::slot_index(1, LJ)] & 0xFu),
                            uint32_t(tdata[Block4Tile::slot_index(2, LJ)] & 0xFu),
                            uint32_t(tdata[Block4Tile::slot_index(3, LJ)] & 0xFu)};
                        w_decoded = block4_vec_decode_fp4(w_codes);
                    }
                    const value_type w_decoded_arr[BLOCK4_TILE] = {
                        value_type(w_decoded[0]), value_type(w_decoded[1]),
                        value_type(w_decoded[2]), value_type(w_decoded[3])};

                    value_type w4[BLOCK4_TILE];
                    std::size_t row_idx[BLOCK4_TILE];
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                        if (row < n_in) {
                            w4[li] =
                                w_decoded_arr[li] * weights.get_scale(row, col); // rank-N scale
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
                static_assert(BLOCK4_TILE == 4,
                              "process_col above is hand-unrolled for exactly 4 columns");
            }
        }
        for (int t = 0; t < num_cpus; ++t) {
            const value_type* s = b4_out.data() + static_cast<std::size_t>(t) * ost;
            for (std::size_t i = 0; i < ost; ++i)
                output[i] += s[i];
        }
    }

    // AQRS additive branch (task #276, gamma at #289): A[row,col] =
    // sum_k gamma_k * additive_u_k(row,k) * additive_v_k(col,k), summed
    // into the effective weight. Fused per Theorem 11 (never materializes
    // the n_in x n_out A matrix). No-op at additive_rank==0. See
    // docs/research/linear_disldo.rst.
    if (weights.additive_rank > 0) {
        std::vector<value_type> proj(static_cast<std::size_t>(batch) * weights.additive_rank,
                                     value_type(0));
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* in_row = input + static_cast<std::size_t>(b) * in_cols;
            value_type* p_row = proj.data() + static_cast<std::size_t>(b) * weights.additive_rank;
            for (std::size_t r = 0; r < n_in; ++r) {
                const value_type iv = in_row[r];
                if (iv == value_type(0))
                    continue;
                for (std::size_t k = 0; k < weights.additive_rank; ++k)
                    p_row[k] += weights.get_additive_u_k(r, k) * iv;
            }
        }
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* p_row =
                proj.data() + static_cast<std::size_t>(b) * weights.additive_rank;
            value_type* out_row = output + static_cast<std::size_t>(b) * n_out;
            for (std::size_t c = 0; c < n_out; ++c) {
                value_type acc = value_type(0);
                for (std::size_t k = 0; k < weights.additive_rank; ++k)
                    acc +=
                        weights.get_additive_gamma_k(k) * weights.get_additive_v_k(c, k) * p_row[k];
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
 *        divided by (sqrt(ci)+eps), an RMSprop-style EMA of g^2. When
 *        false: the raw (-effective_lr * g) step is applied, undamped;
 *        ci is still tracked either way. See docs/research/linear_disldo.rst
 *        for why this replaced an earlier signed-sum formula.
 * @param beta2  Decay rate for ci's g^2 EMA (default 0.999). Only used
 *        when damp_by_importance is true.
 * @param eps    Numerical floor added to sqrt(ci) (matches Adam's eps
 *        convention, default 1e-8).
 *
 * NOTE (test): with learning_rate=0, input_grad must equal W_dense^T @ output_grad
 * per batch sample, weights/importance unchanged. Same reference check as
 * delta_csr_backward.
 *
 * Template params ScalePolicy/DeferredScaleWrite/StochasticRounding/
 * SynapsePolicyT and the ordering of the trailing scalar params
 * (l1_coef last) are constrained by existing callers' positional template
 * args -- see docs/research/linear_disldo.rst for the full rationale
 * and why SynapsePolicyT specifically must be template-template.
 */
template <
    typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t,
    typename ScalePolicy = RMSpropScalePolicy<typename ValueAccessor<VALUES_TYPE>::value_type>,
    bool DeferredScaleWrite = false, bool StochasticRounding = true,
    template <typename> class SynapsePolicyT = BoundedRMSpropSynapsePolicy>
void disldo_backward(const typename ValueAccessor<VALUES_TYPE>::value_type* input, SIZE_TYPE batch,
                     SIZE_TYPE in_cols,
                     const typename ValueAccessor<VALUES_TYPE>::value_type* output_grad,
                     SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
                     typename ValueAccessor<VALUES_TYPE>::value_type* input_grad,
                     typename ValueAccessor<VALUES_TYPE>::value_type* neuron_input_accum,
                     typename ValueAccessor<VALUES_TYPE>::value_type* neuron_grad_accum,
                     typename ValueAccessor<VALUES_TYPE>::value_type learning_rate = 0.01f,
                     int num_cpus = 4, bool lr_per_row_nnz = false, bool damp_by_importance = true,
                     typename ValueAccessor<VALUES_TYPE>::value_type beta2 = 0.999f,
                     typename ValueAccessor<VALUES_TYPE>::value_type eps = 1e-8f,
                     typename ValueAccessor<VALUES_TYPE>::value_type beta1 = 0.9f,
                     typename ValueAccessor<VALUES_TYPE>::value_type min_decay_frac = 0.0f,
                     typename ValueAccessor<VALUES_TYPE>::value_type max_abs_delta = 1e30f,
                     typename ValueAccessor<VALUES_TYPE>::value_type max_ci = 1e30f,
                     typename ValueAccessor<VALUES_TYPE>::value_type zero_escape_eps = 0.1f,
                     bool scale_invariant = false,
                     // AQRS gamma's L1 penalty coefficient (task #273/#283, Theorem 8).
                     // 0 (default) disables it. Appended LAST for positional-arg safety
                     // -- see this function's own docstring above.
                     typename ValueAccessor<VALUES_TYPE>::value_type l1_coef = 0.0f) {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    using SynapsePolicy = SynapsePolicyT<value_type>;
    using SynapsePolicyVec = SynapsePolicyT<Block4Vec>;
    // Only meaningful when DeferredScaleWrite: a touched scattered entry's
    // true-units (cw, ci) get cached here instead of stored immediately,
    // written out only once value_scale/output_scale are finalized for
    // this whole call. See docs/research/linear_disldo.rst.
    struct DeferredScaleWriteEntry {
        // cppcheck-suppress uninitMemberVarNoCtor
        std::size_t vb;
        value_type cw;
        value_type ci;
        // cppcheck-suppress uninitMemberVarNoCtor
        std::size_t row;
        // cppcheck-suppress uninitMemberVarNoCtor
        COL_TYPE col;
    };
    std::vector<std::vector<DeferredScaleWriteEntry>> t_deferred;
    if constexpr (DeferredScaleWrite)
        t_deferred.resize(static_cast<std::size_t>(num_cpus));
    auto& dc = weights.connections;
    const auto& L = dc.layout;
    const std::size_t n_in = L.rows;
    const std::size_t n_out = L.cols;

    for (SIZE_TYPE b = 0; b < batch; ++b) {
        for (std::size_t r = 0; r < n_in; ++r)
            neuron_input_accum[r] += std::abs(input[static_cast<std::size_t>(b) * in_cols + r]);
        for (std::size_t c = 0; c < n_out; ++c)
            neuron_grad_accum[c] += std::abs(output_grad[static_cast<std::size_t>(b) * n_out + c]);
    }

    // No early return when both storages are empty: the dead-row
    // value_scale bootstrap pass (near the end) needs to run precisely
    // in that case. See docs/research/linear_disldo.rst.
    const std::size_t dst = static_cast<std::size_t>(batch) * in_cols;
    std::vector<value_type> t_dx(static_cast<std::size_t>(num_cpus) * dst, value_type(0));

    // output_scale's own gradient (symmetric to value_scale's), shared
    // between the scattered loop and block4's own loop, per-thread-private
    // slices reduced once at the end. Only applied if output_scale_is_trainable.
    const std::size_t rank = weights.scale_rank;
    // Persistent per-instance heap scratch (task #295) backing block4's
    // SIMD backward path's per-rank-component accumulators.
    weights.scale_rank_scratch.ensure(static_cast<std::size_t>(num_cpus), rank, BLOCK4_TILE);
    // Thin 2D view over a flat scratch buffer, drop-in for the old
    // fixed-size 2D stack arrays (task #295).
    struct Flat2DView {
        value_type* base;
        std::size_t stride;
        inline value_type* operator[](std::size_t k) const { return base + k * stride; }
    };
    // AQRS gamma (task #273/#283): layer-wide, fetched once rather than
    // per-row. See docs/research/linear_disldo.rst for why it's not baked
    // into the direction caches.
    std::vector<value_type> gamma_k_arr(rank);
    for (std::size_t k = 0; k < rank; ++k)
        gamma_k_arr[k] = weights.get_scale_gamma_k(k);
    std::vector<value_type> t_col_grad(static_cast<std::size_t>(num_cpus) * n_out * rank,
                                       value_type(0));
    // Parallel forward-contribution accumulator, same shape/layout as
    // t_col_grad -- mirrors the per-synapse ci fix (contrib=x*w combined
    // Additive-signal forward-contribution accumulator, same combination
    // rationale as per-synapse ci (Joint.combined_signal_strictly_informative)
    // applied one level up. See docs/research/linear_disldo.rst.
    std::vector<value_type> t_col_grad_contrib(static_cast<std::size_t>(num_cpus) * n_out * rank,
                                               value_type(0));
    const bool output_scale_trainable = weights.output_scale_is_trainable;

    // AQRS gamma's own gradient (task #273/#283): layer-wide scalar per k,
    // not per-row/col, so sized num_cpus*rank.
    std::vector<value_type> t_gamma_grad(static_cast<std::size_t>(num_cpus) * rank, value_type(0));
    std::vector<value_type> t_gamma_grad_contrib(static_cast<std::size_t>(num_cpus) * rank,
                                                 value_type(0));

    // Pre-size value_scale/output_scale (n_in*rank / n_out*rank) before the
    // parallel region so direct indexed writes are safe.
    //
    // A uniform resize(..., value_type(1)) fill would backfill EVERY
    // appended slot with 1.0, not just k==0, forcing any grown rank
    // component into permanent lockstep with k==0 instead of real extra
    // capacity -- see docs/research/linear_disldo.rst. Resize with 0
    // fill instead, then explicitly set only the k==0 slots to 1.0.
    if (weights.value_scale.size() < n_in * rank) {
        const std::size_t old_size = weights.value_scale.size();
        weights.value_scale.resize(n_in * rank, value_type(0));
        for (std::size_t idx = old_size; idx < weights.value_scale.size(); ++idx)
            if (idx % rank == 0)
                weights.value_scale[idx] = value_type(1);
    }
    if (weights.output_scale.size() < n_out * rank) {
        const std::size_t old_size = weights.output_scale.size();
        weights.output_scale.resize(n_out * rank, value_type(0));
        for (std::size_t idx = old_size; idx < weights.output_scale.size(); ++idx)
            if (idx % rank == 0)
                weights.output_scale[idx] = value_type(1);
    }
    if (weights.value_scale_importance.size() < n_in * rank)
        weights.value_scale_importance.resize(n_in * rank, value_type(0));
    if (weights.output_scale_importance.size() < n_out * rank)
        weights.output_scale_importance.resize(n_out * rank, value_type(0));
    if (weights.value_scale_momentum.size() < n_in * rank)
        weights.value_scale_momentum.resize(n_in * rank, value_type(0));
    // Bug fix, found via AddressSanitizer: value_scale_step was missing
    // from this pre-sizing list, so get_value_scale_step_k's own lazy
    // unguarded resize() raced across threads (real heap-use-after-free).
    // See docs/research/linear_disldo.rst.
    if (weights.value_scale_step.size() < n_in * rank)
        weights.value_scale_step.resize(n_in * rank, 0);

    if (!dc.empty()) {
#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mdx = t_dx.data() + static_cast<std::size_t>(tid) * dst;
            // mcol[col*rank+k]. False positive on all m*_base pointers
            // below: cppcheck can't trace mutation through the m*_at
            // lambdas' returned references.
            // cppcheck-suppress constVariablePointer
            value_type* mcol_base =
                t_col_grad.data() + static_cast<std::size_t>(tid) * n_out * rank;
            auto mcol_at = [&](std::size_t col, std::size_t k) -> value_type& {
                return mcol_base[col * rank + k];
            };
            // cppcheck-suppress constVariablePointer
            value_type* mcol_contrib_base =
                t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * n_out * rank;
            auto mcol_at_contrib = [&](std::size_t col, std::size_t k) -> value_type& {
                return mcol_contrib_base[col * rank + k];
            };
            // cppcheck-suppress constVariablePointer
            value_type* mgamma_base = t_gamma_grad.data() + static_cast<std::size_t>(tid) * rank;
            auto mgamma_at = [&](std::size_t k) -> value_type& { return mgamma_base[k]; };
            // cppcheck-suppress constVariablePointer
            value_type* mgamma_contrib_base =
                t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * rank;
            auto mgamma_at_contrib = [&](std::size_t k) -> value_type& {
                return mgamma_contrib_base[k];
            };
            [[maybe_unused]] std::vector<DeferredScaleWriteEntry>* mdeferred = nullptr;
            if constexpr (DeferredScaleWrite)
                mdeferred = &t_deferred[static_cast<std::size_t>(tid)];

            // Per-thread importance stats accumulators -- see
            // update_importance_stats()'s THREAD SAFETY note. Value stats
            // aren't tracked here since value_scale is learned via gradient
            // descent directly, not a Hoyer-based policy decision.
            double local_sum_abs_new_i = 0.0, local_sum_abs_old_i = 0.0;
            double local_sum_sq_new_i = 0.0, local_sum_sq_old_i = 0.0;
            value_type local_max_new_i = value_type(0);

#pragma omp for schedule(static)
            for (std::size_t r = 0; r < n_in; ++r) {
                const std::size_t nnz_this_row = L.row_nnz(r);
                if (nnz_this_row == 0)
                    continue;
                // lr_per_row_nnz: normalizes the aggregate per-row update
                // against synaptogenesis-driven nnz variation. See
                // docs/research/linear_disldo.rst.
                const value_type effective_lr =
                    lr_per_row_nnz ? learning_rate / static_cast<value_type>(nnz_this_row)
                                   : learning_rate;

                // value_scale gradient ALWAYS divides by nnz_this_row
                // (independent of lr_per_row_nnz), normalizing its
                // nnz_this_row*batch-term accumulator back to a single
                // scalar parameter's gradient semantics.
                const value_type scale_eff_lr =
                    learning_rate / static_cast<value_type>(nnz_this_row);

                auto cursor = dc.row_cursor(r);
                const value_type imp_scale = weights.get_importance_scale(r);
                const value_type val_scale = weights.get_value_scale(r);
                // Sum first across ALL (synapse, batch) pairs, apply lr
                // ONCE (avoids per-contribution increments disappearing
                // below float32 ULP). scale_grad_sum: DeferredScaleWrite's
                // component-0-only accumulator; scale_grad_sum_rank:
                // non-deferred path's per-component accumulator.
                double scale_grad_sum = 0.0;
                std::vector<double> scale_grad_sum_rank(rank, 0.0);
                double scale_grad_sum_contrib = 0.0;
                std::vector<double> scale_grad_sum_rank_contrib(rank, 0.0);
                for (std::size_t e = 0; e < nnz_this_row; ++e) {
                    const COL_TYPE col = cursor.advance();
                    const std::size_t vb = L.elem_start[r] + e;
                    const value_type cw_orig = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                    const value_type ci_orig = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                    const value_type out_scale = weights.get_output_scale(col);
                    const value_type combined_scale = val_scale * out_scale;
                    // Same row*col combination as the weight's own scale --
                    // see the matching comment in disldo_forward.
                    const value_type out_imp_scale = weights.get_output_importance_scale(col);
                    const value_type combined_imp_scale = imp_scale * out_imp_scale;
                    value_type ci = ci_orig * combined_imp_scale; // -> true units

                    if constexpr (DeferredScaleWrite) {
                        // True-units round-trip formula: this branch's
                        // value_scale/output_scale aren't finalized until
                        // after this loop, so it can't use the direct-quant
                        // chain-rule update below (would reintroduce the
                        // staleness DeferredScaleWrite exists to avoid) --
                        // and doesn't get deterministic-rounding zero-escape
                        // either. cw_start: FIXED
                        // snapshot for the whole batch loop, g_agg/contrib_agg
                        // aggregated across it, ONE update applied after --
                        // see docs/research/linear_disldo.rst for the real
                        // batch-aggregation bug this fixes.
                        const value_type cw_start = cw_orig * combined_scale;
                        double g_agg = 0.0, contrib_agg = 0.0;
                        for (SIZE_TYPE b = 0; b < batch; ++b) {
                            const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                            const value_type dyv =
                                output_grad[static_cast<std::size_t>(b) * n_out + col];
                            const value_type g = dyv * iv;
                            if (learning_rate != value_type(0)) {
                                // Additive g+contrib combination, square-then-sum
                                // -- see docs/research/linear_disldo.rst
                                // (Joint.combined_signal_strictly_informative).
                                const value_type contrib = iv * cw_start;
                                g_agg += static_cast<double>(g);
                                contrib_agg += static_cast<double>(contrib);
                                scale_grad_sum += static_cast<double>(cw_orig) *
                                                  static_cast<double>(out_scale) * g;
                                mcol_at(col, 0) += cw_orig * val_scale * g;
                                scale_grad_sum_contrib += static_cast<double>(cw_orig) *
                                                          static_cast<double>(out_scale) * contrib;
                                mcol_at_contrib(col, 0) += cw_orig * val_scale * contrib;
                            }
                            mdx[static_cast<std::size_t>(b) * in_cols + r] += cw_start * dyv;
                        }
                        value_type cw = cw_start;
                        if (learning_rate != value_type(0)) {
                            // ONE update, using the batch-aggregated g/contrib.
                            ci = SynapsePolicy::update_ci(ci, static_cast<value_type>(g_agg),
                                                          static_cast<value_type>(contrib_agg),
                                                          beta2, min_decay_frac, max_ci);
                            cw += SynapsePolicy::update_cw(
                                static_cast<value_type>(g_agg), ci, value_type(1), effective_lr,
                                eps, damp_by_importance, max_abs_delta, scale_invariant);
                            // Defer the store until value_scale[r] AND
                            // output_scale[col] are both finalized. See
                            // docs/research/linear_disldo.rst.
                            mdeferred->push_back(DeferredScaleWriteEntry{vb, cw, ci, r, col});
                            local_sum_abs_new_i += std::abs(static_cast<double>(ci));
                            local_sum_abs_old_i += std::abs(static_cast<double>(ci_orig));
                            local_sum_sq_new_i += static_cast<double>(ci) * ci;
                            local_sum_sq_old_i += static_cast<double>(ci_orig) * ci_orig;
                            local_max_new_i = std::max(local_max_new_i, std::abs(ci));
                        }
                    } else {
                        // Direct-quant chain rule: dL/d(quant) = g * S(row,col)
                        // on true_w = quant*S, replacing the old true-units
                        // round-trip that divided by S (backwards). S =
                        // weights.get_scale(r,col), summed over `rank`
                        // components. See docs/research/linear_disldo.rst.
                        const value_type S = weights.get_scale(r, col);
                        // cw_start/quant_floor: FIXED snapshots for the whole
                        // batch loop -- see the DeferredScaleWrite branch's
                        // identical cw_start comment above.
                        const value_type cw_start = cw_orig * S;
                        const value_type quant_floor =
                            (cw_orig == value_type(0)) ? zero_escape_eps : cw_orig;
                        double g_agg = 0.0, contrib_agg = 0.0;
                        for (SIZE_TYPE b = 0; b < batch; ++b) {
                            const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                            const value_type dyv =
                                output_grad[static_cast<std::size_t>(b) * n_out + col];
                            const value_type g = dyv * iv;
                            if (learning_rate != value_type(0)) {
                                // RMSprop-style ci, additive g+contrib
                                // combination -- see this function's own
                                // docstring and docs/research/linear_disldo.rst.
                                const value_type contrib = iv * cw_start;
                                g_agg += static_cast<double>(g);
                                contrib_agg += static_cast<double>(contrib);
                                // dL/d(value_scale_k(r,k)) = g*quant*
                                // output_scale_k(col,k), vanishing exactly at
                                // quant=0. quant_floor gates the zero-escape
                                // substitution on quant==0 specifically (not
                                // unconditionally) -- see
                                // docs/research/linear_disldo.rst for the real
                                // sign-corruption bug this fixes.
                                for (std::size_t k = 0; k < rank; ++k) {
                                    // AQRS gamma (task #273/#283): direction
                                    // caches are never gamma-baked, so each
                                    // gradient site needs an explicit gamma_k
                                    // factor; gamma's own gradient uses the
                                    // pure direction product. See
                                    // docs/research/linear_disldo.rst.
                                    const value_type out_scale_k =
                                        weights.get_output_scale_k(col, k);
                                    const value_type val_scale_k = weights.get_value_scale_k(r, k);
                                    const value_type gamma_k = weights.get_scale_gamma_k(k);
                                    scale_grad_sum_rank[k] += static_cast<double>(quant_floor) *
                                                              static_cast<double>(out_scale_k) *
                                                              static_cast<double>(gamma_k) * g;
                                    mcol_at(col, k) += quant_floor * val_scale_k * gamma_k * g;
                                    mgamma_at(k) += quant_floor * val_scale_k * out_scale_k * g;
                                    // Parallel forward-contribution accumulation
                                    // -- see scale_grad_sum_contrib's own
                                    // comment above.
                                    scale_grad_sum_rank_contrib[k] +=
                                        static_cast<double>(quant_floor) *
                                        static_cast<double>(out_scale_k) *
                                        static_cast<double>(gamma_k) * contrib;
                                    mcol_at_contrib(col, k) +=
                                        quant_floor * val_scale_k * gamma_k * contrib;
                                    mgamma_at_contrib(k) +=
                                        quant_floor * val_scale_k * out_scale_k * contrib;
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
                                                              effective_lr, eps, damp_by_importance,
                                                              max_abs_delta, scale_invariant);
                            cw = quant * S;
                            if constexpr (StochasticRounding) {
                                ValueAccessor<VALUES_TYPE>::set_stochastic_live(
                                    dc.values, vb, quant, ci / combined_imp_scale);
                            } else {
                                ValueAccessor<VALUES_TYPE>::set_live(dc.values, vb, quant,
                                                                     ci / combined_imp_scale);
                            }
                            const value_type actual_imp =
                                ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                            local_sum_abs_new_i += std::abs(static_cast<double>(actual_imp));
                            local_sum_abs_old_i += std::abs(static_cast<double>(ci_orig));
                            local_sum_sq_new_i += static_cast<double>(actual_imp) * actual_imp;
                            local_sum_sq_old_i += static_cast<double>(ci_orig) * ci_orig;
                            local_max_new_i = std::max(local_max_new_i, std::abs(actual_imp));
                        }
                    }
                }
                if (learning_rate != value_type(0)) {
                    // Scale update via the swappable policy (default
                    // RMSpropScalePolicy). DeferredScaleWrite only updates
                    // component 0 (rank>1 is non-deferred-only).
                    if constexpr (!DeferredScaleWrite) {
                        for (std::size_t k = 0; k < rank; ++k) {
                            const value_type g_agg_k =
                                static_cast<value_type>(scale_grad_sum_rank[k]);
                            const value_type contrib_agg_k =
                                static_cast<value_type>(scale_grad_sum_rank_contrib[k]);
                            ScalePolicy::update(weights.value_scale[r * rank + k],
                                                weights.value_scale_importance[r * rank + k],
                                                g_agg_k, scale_eff_lr, beta2, eps, contrib_agg_k,
                                                &weights.get_value_scale_step_k(r, k),
                                                scale_invariant);
                        }
                    } else {
                        const value_type g_agg = static_cast<value_type>(scale_grad_sum);
                        const value_type contrib_agg =
                            static_cast<value_type>(scale_grad_sum_contrib);
                        ScalePolicy::update(weights.value_scale[r],
                                            weights.value_scale_importance[r], g_agg, scale_eff_lr,
                                            beta2, eps, contrib_agg,
                                            &weights.get_value_scale_step_k(r, 0), scale_invariant);
                    }
                }
            }

            if (learning_rate != value_type(0)) {
#pragma omp critical
                {
                    weights.update_importance_stats_aggregate(
                        local_sum_abs_new_i, local_sum_abs_old_i, local_sum_sq_new_i,
                        local_sum_sq_old_i, local_max_new_i);
                }
            }
        }
    } // !dc.empty()

    // block4 backward: dx + inline weight/importance update, mirroring the
    // scattered loop but keyed by tile instead of CSR row, sharing the
    // same value_scale/output_scale. Parallelized over TILES (not rows),
    // so a row's value_scale gradient can be touched by more than one
    // thread -- fixed via per-thread-private accumulators (t_row_grad),
    // reduced once after the parallel region. Known, documented
    // simplification: a row live in BOTH representations gets two
    // sequential gradient steps, not a bug. See
    // docs/research/linear_disldo.rst.
    if (weights.block4.n_tiles() > 0) {
        // row_ti_start: cumulative tile count per block-row, needed to
        // split the row-partitioned parallel loop below. NOT storage
        // offsets -- since the row-workspace rewrite, tile byte/elem
        // positions live entirely within each row's own RowWorkspace,
        // snapshotted fresh per row. See docs/research/linear_disldo.rst.
        std::vector<std::size_t>& row_ti_start = weights.block4.scratch_row_ti_start;
        const auto& BL4 = weights.block4.block_layout;
        row_ti_start.resize(BL4.rows + 1);

        // Per-row slot count across ALL block4 tiles touching that row,
        // needed for lr_per_row_nnz and scale_eff_lr normalization. Every
        // tile contributes exactly BLOCK4_TILE slots per row it covers.
        std::vector<uint32_t>& row_live_count = weights.block4.scratch_row_live_count;
        row_live_count.assign(n_in, 0);
        std::size_t ti = 0;
        for (std::size_t br = 0; br < BL4.rows; ++br) {
            row_ti_start[br] = ti; // written unconditionally: an empty row
            // still needs a valid (empty) [start,start) range below.
            const std::size_t n_bc = BL4.row_nnz(br);
            ti += n_bc;
            if (n_bc == 0)
                continue;
            const uint32_t row_count = uint32_t(n_bc) * BLOCK4_TILE;
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                const std::size_t row = br * BLOCK4_TILE + li;
                if (row < n_in)
                    row_live_count[row] = row_count;
            }
        }
        row_ti_start[BL4.rows] = ti;

        std::vector<double>& t_row_grad = weights.block4.scratch_row_grad;
        t_row_grad.assign(static_cast<std::size_t>(num_cpus) * n_in * rank, 0.0);
        std::vector<double> t_row_grad_contrib(static_cast<std::size_t>(num_cpus) * n_in * rank,
                                               0.0);

#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mdx = t_dx.data() + static_cast<std::size_t>(tid) * dst;
            // t_col_grad/t_row_grad laid out [thread][col or row][k]; both
            // FP4/FP8 block4 branches below are full rank-N.
            // cppcheck-suppress constVariablePointer
            value_type* mcol_base =
                t_col_grad.data() + static_cast<std::size_t>(tid) * n_out * rank;
            auto mcol_at = [&](std::size_t col, std::size_t k) -> value_type& {
                return mcol_base[col * rank + k];
            };
            // cppcheck-suppress constVariablePointer
            double* mrow_base = t_row_grad.data() + static_cast<std::size_t>(tid) * n_in * rank;
            auto mrow_at = [&](std::size_t row, std::size_t k) -> double& {
                return mrow_base[row * rank + k];
            };
            // cppcheck-suppress constVariablePointer
            value_type* mcol_contrib_base =
                t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * n_out * rank;
            auto mcol_at_contrib = [&](std::size_t col, std::size_t k) -> value_type& {
                return mcol_contrib_base[col * rank + k];
            };
            // cppcheck-suppress constVariablePointer
            double* mrow_contrib_base =
                t_row_grad_contrib.data() + static_cast<std::size_t>(tid) * n_in * rank;
            auto mrow_at_contrib = [&](std::size_t row, std::size_t k) -> double& {
                return mrow_contrib_base[row * rank + k];
            };
            // AQRS gamma: block4 gets its OWN parallel region, so
            // mgamma_at/mgamma_at_contrib need their own tid-scoped
            // closures, reading from the SAME t_gamma_grad/
            // t_gamma_grad_contrib buffers the scattered path uses --
            // both regions accumulate into the same shared array, reduced
            // together once after both close.
            // cppcheck-suppress constVariablePointer
            value_type* mgamma_base = t_gamma_grad.data() + static_cast<std::size_t>(tid) * rank;
            auto mgamma_at = [&](std::size_t k) -> value_type& { return mgamma_base[k]; };
            // cppcheck-suppress constVariablePointer
            value_type* mgamma_contrib_base =
                t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * rank;
            auto mgamma_at_contrib = [&](std::size_t k) -> value_type& {
                return mgamma_contrib_base[k];
            };

// Partitioned BY BLOCK-ROW, not flat tile index: each thread
// exclusively owns every tile in its assigned rows, so a
// tile's handle can safely resize (real sparse<->dense
// transitions) inside this parallel region. schedule(static)
// measured to beat dynamic/guided here despite uneven row
// widths -- see docs/research/linear_disldo.rst.
#pragma omp for schedule(static)
            for (std::size_t br = 0; br < BL4.rows; ++br) {
                if (row_ti_start[br] == row_ti_start[br + 1])
                    continue; // empty row
                // Row-workspace snapshot fixes a real cross-row memmove
                // hazard (a different row's growth could memmove this
                // row's bytes mid-read) -- see
                // docs/research/linear_disldo.rst. process_tile: shared
                // between the read-only and writing branches below so
                // they can't drift apart; every write inside is gated by
                // learning_rate != 0.
                auto process_tile = [&](uint32_t bc, uint8_t* tdata) -> bool {
                    bool tile_dirty = false;
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                        if (row >= n_in)
                            continue;
                        const uint32_t nnz_row = row_live_count[row];
                        if (nnz_row == 0)
                            continue;
                        const value_type val_scale = weights.get_value_scale(row);
                        const value_type imp_scale = weights.get_importance_scale(row);
                        const value_type effective_lr =
                            lr_per_row_nnz ? learning_rate / static_cast<value_type>(nnz_row)
                                           : learning_rate;

                        // Decode this row's whole 4-wide column vector ONCE,
                        // outside the batch loop (avoids blocking
                        // auto-vectorization; the batch loop itself can't
                        // vectorize across b, but the 4 columns are
                        // independent, giving the inner lj loop a real
                        // 4-wide target). FP4_TABLE[code] lookup, not
                        // block4_vec_decode_fp4's SIMD bit-shift formula --
                        // measured ~6% win for backward specifically
                        // (opposite of forward's finding). See
                        // docs/research/linear_disldo.rst.
                        if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                            // FP8 (E4M3) block4 weight+importance update.
                            // Measured (scripts/bench_block4_fp8_simd.cpp):
                            // full SIMD LOSES at batch=1 (E4M3's subnormal/
                            // NaN-lane scalar-correction fallback is real
                            // overhead FP4 never pays), scalar decode/encode
                            // + SIMD accumulate-only WINS at batch=32. See
                            // docs/research/linear_disldo.rst.
                            if constexpr (std::is_same_v<value_type, float> &&
                                          !SILI_BLOCK4_FORCE_SCALAR_BACKWARD) {
                                const value_type w_decoded_arr8[BLOCK4_TILE] = {
                                    fp8_decode_bits(tdata[Block4Tile8::slot_index(li, 0)]),
                                    fp8_decode_bits(tdata[Block4Tile8::slot_index(li, 1)]),
                                    fp8_decode_bits(tdata[Block4Tile8::slot_index(li, 2)]),
                                    fp8_decode_bits(tdata[Block4Tile8::slot_index(li, 3)])};
                                const value_type imp_decoded_arr8[BLOCK4_TILE] = {
                                    fp8_decode_bits(
                                        tdata[BLOCK4_TILE + Block4Tile8::slot_index(li, 0)]),
                                    fp8_decode_bits(
                                        tdata[BLOCK4_TILE + Block4Tile8::slot_index(li, 1)]),
                                    fp8_decode_bits(
                                        tdata[BLOCK4_TILE + Block4Tile8::slot_index(li, 2)]),
                                    fp8_decode_bits(
                                        tdata[BLOCK4_TILE + Block4Tile8::slot_index(li, 3)])};

                                // value_scale_k(row,k) -- see FP4 branch's
                                // identical comment above (mirrors it exactly,
                                // now generalized over rank instead of the
                                // single rank-1-only val_scale local).
                                value_type* value_scale_k8 =
                                    weights.scale_rank_scratch.value_scale_k.data() +
                                    static_cast<std::size_t>(tid) * rank;
                                for (std::size_t k = 0; k < rank; ++k)
                                    value_scale_k8[k] = weights.get_value_scale_k(row, k);

                                std::size_t col4_8[BLOCK4_TILE];
                                bool col_valid4_8[BLOCK4_TILE];
                                // out_scale_k4_8[k][lj]: per-rank-component
                                // output_scale, needed for value_scale_k's own
                                // gradient below (mirrors FP4's out_scale_k4).
                                const Flat2DView out_scale_k4_8{
                                    weights.scale_rank_scratch.out_scale_k.data() +
                                        static_cast<std::size_t>(tid) * rank * BLOCK4_TILE,
                                    BLOCK4_TILE};
                                value_type combined_scale4_8[BLOCK4_TILE],
                                    combined_imp_scale4_8[BLOCK4_TILE];
                                value_type cw4_8[BLOCK4_TILE], ci4_8[BLOCK4_TILE],
                                    cw_orig4_8[BLOCK4_TILE];
                                // was_live4_8[lj]: TRUE only if this cell held a
                                // genuine synapse BEFORE this call -- gates the
                                // never-zero live quantizer so it never
                                // permanently "births" a never-connected cell.
                                // See docs/research/linear_disldo.rst.
                                bool was_live4_8[BLOCK4_TILE];
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                                    col_valid4_8[lj] = col < n_out;
                                    if (!col_valid4_8[lj]) {
                                        col4_8[lj] = 0;
                                        combined_scale4_8[lj] = combined_imp_scale4_8[lj] =
                                            value_type(0);
                                        cw4_8[lj] = ci4_8[lj] = cw_orig4_8[lj] = value_type(0);
                                        was_live4_8[lj] = false;
                                        for (std::size_t k = 0; k < rank; ++k)
                                            out_scale_k4_8[k][lj] = value_type(0);
                                        continue;
                                    }
                                    was_live4_8[lj] = (w_decoded_arr8[lj] != value_type(0)) ||
                                                      (imp_decoded_arr8[lj] != value_type(0));
                                    col4_8[lj] = col;
                                    const value_type out_imp_scale =
                                        weights.get_output_importance_scale(col);
                                    // S(row,col) = sum_k value_scale_k*output_scale_k.
                                    // cw4_8 is CODE-SPACE (matches FP4's quant4
                                    // convention) -- keeping it in true units
                                    // instead caused a real 1/S blowup at write
                                    // time for small output_scale. See
                                    // docs/research/linear_disldo.rst.
                                    combined_scale4_8[lj] = weights.get_scale(row, col);
                                    combined_imp_scale4_8[lj] = imp_scale * out_imp_scale;
                                    cw_orig4_8[lj] = w_decoded_arr8[lj];
                                    cw4_8[lj] = cw_orig4_8[lj];
                                    ci4_8[lj] = imp_decoded_arr8[lj] * combined_imp_scale4_8[lj];
                                    for (std::size_t k = 0; k < rank; ++k)
                                        out_scale_k4_8[k][lj] = weights.get_output_scale_k(col, k);
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
                                const Flat2DView mcol4_8_rank{srs.mcol_rank.data() + tid_rank_tile,
                                                              BLOCK4_TILE};
                                double* mrow_local8_k = srs.mrow_local_k.data() + tid_rank;
                                const Flat2DView mcol4_8_rank_contrib{
                                    srs.mcol_rank_contrib.data() + tid_rank_tile, BLOCK4_TILE};
                                double* mrow_local8_k_contrib =
                                    srs.mrow_local_k_contrib.data() + tid_rank;
                                // AQRS gamma's own gradient (task #273/#283) --
                                // per-tile-row local accumulator, folded into
                                // the shared mgamma_at/mgamma_at_contrib (same
                                // ones the scattered path uses) once this row
                                // finishes, matching mrow_local8_k's own
                                // fold-into-mrow_at pattern below.
                                double* mgamma_local8_k = srs.mgamma_local_k.data() + tid_rank;
                                double* mgamma_local8_k_contrib =
                                    srs.mgamma_local_k_contrib.data() + tid_rank;
                                std::fill(mcol4_8_rank[0], mcol4_8_rank[0] + rank * BLOCK4_TILE,
                                          value_type(0));
                                std::fill(mrow_local8_k, mrow_local8_k + rank, 0.0);
                                std::fill(mcol4_8_rank_contrib[0],
                                          mcol4_8_rank_contrib[0] + rank * BLOCK4_TILE,
                                          value_type(0));
                                std::fill(mrow_local8_k_contrib, mrow_local8_k_contrib + rank, 0.0);
                                std::fill(mgamma_local8_k, mgamma_local8_k + rank, 0.0);
                                std::fill(mgamma_local8_k_contrib, mgamma_local8_k_contrib + rank,
                                          0.0);
                                const std::size_t col_base8 = std::size_t(bc) * BLOCK4_TILE;
                                const bool full_tile_cols8 = (col_base8 + BLOCK4_TILE <= n_out);

                                if (full_tile_cols8) {
                                    const Block4Vec effective_lr_v =
                                        block4_vec_broadcast(effective_lr);
                                    const Block4Vec beta2_v = block4_vec_broadcast(beta2);
                                    const Block4Vec eps_v = block4_vec_broadcast(eps);
                                    const Block4Vec min_decay_frac_v =
                                        block4_vec_broadcast(min_decay_frac);
                                    const Block4Vec max_ci_v = block4_vec_broadcast(max_ci);
                                    const Block4Vec max_abs_delta_v =
                                        block4_vec_broadcast(max_abs_delta);
                                    // cw_start_v/S_v: FIXED snapshots for the
                                    // whole batch loop -- see the scattered
                                    // path's identical cw_start comment.
                                    const Block4Vec cw_start_v = block4_vec_load(cw4_8);
                                    Block4Vec ci_v = block4_vec_load(ci4_8);
                                    const Block4Vec cw_orig_v = block4_vec_load(cw_orig4_8);
                                    const Block4Vec S_v = block4_vec_load(combined_scale4_8);
                                    // mcol_acc_raw8 are the only TRUE
                                    // cross-batch accumulators here; backed by
                                    // scratch (task #295).
                                    value_type* mcol_acc_raw8 =
                                        srs.mcol_acc_raw.data() + tid_rank_tile;
                                    value_type* mcol_acc_raw8_contrib =
                                        srs.mcol_acc_raw_contrib.data() + tid_rank_tile;
                                    for (std::size_t k = 0; k < rank; ++k) {
                                        block4_vec_store(mcol_acc_raw8 + k * BLOCK4_TILE,
                                                         block4_vec_broadcast(0.0f));
                                        block4_vec_store(mcol_acc_raw8_contrib + k * BLOCK4_TILE,
                                                         block4_vec_broadcast(0.0f));
                                    }
                                    Block4Vec g_agg_v = block4_vec_broadcast(0.0f);
                                    Block4Vec contrib_agg_v = block4_vec_broadcast(0.0f);
                                    const bool training = (learning_rate != value_type(0));
                                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                                        const value_type iv =
                                            input[static_cast<std::size_t>(b) * in_cols + row];
                                        value_type* mdx_row =
                                            mdx + static_cast<std::size_t>(b) * in_cols + row;
                                        const Block4Vec dyv_v = block4_vec_load(
                                            output_grad + static_cast<std::size_t>(b) * n_out +
                                            col_base8);
                                        const Block4Vec g_v = dyv_v * block4_vec_broadcast(iv);
                                        if (training) {
                                            // Additive g+contrib combination --
                                            // see the scattered path's fix.
                                            const Block4Vec contrib_v =
                                                cw_start_v * S_v * block4_vec_broadcast(iv);
                                            g_agg_v += g_v;
                                            contrib_agg_v += contrib_v;
                                            for (std::size_t k = 0; k < rank; ++k) {
                                                // AQRS gamma: direction caches
                                                // never gamma-baked, explicit
                                                // gamma_k factor per site.
                                                const Block4Vec value_scale_k_v8 =
                                                    block4_vec_broadcast(value_scale_k8[k]);
                                                const Block4Vec out_scale_k_v8 =
                                                    block4_vec_load(out_scale_k4_8[k]);
                                                mrow_local8_k[k] +=
                                                    static_cast<double>(block4_vec_hsum(
                                                        cw_orig_v * out_scale_k_v8 * g_v)) *
                                                    static_cast<double>(gamma_k_arr[k]);
                                                mgamma_local8_k[k] += static_cast<double>(
                                                    block4_vec_hsum(cw_orig_v * out_scale_k_v8 *
                                                                    value_scale_k_v8 * g_v));
                                                mrow_local8_k_contrib[k] +=
                                                    static_cast<double>(block4_vec_hsum(
                                                        cw_orig_v * out_scale_k_v8 * contrib_v)) *
                                                    static_cast<double>(gamma_k_arr[k]);
                                                mgamma_local8_k_contrib[k] += static_cast<double>(
                                                    block4_vec_hsum(cw_orig_v * out_scale_k_v8 *
                                                                    value_scale_k_v8 * contrib_v));
                                                value_type* acc = mcol_acc_raw8 + k * BLOCK4_TILE;
                                                block4_vec_store(
                                                    acc,
                                                    block4_vec_load(acc) +
                                                        cw_orig_v * value_scale_k_v8 * g_v *
                                                            block4_vec_broadcast(gamma_k_arr[k]));
                                                value_type* acc_c =
                                                    mcol_acc_raw8_contrib + k * BLOCK4_TILE;
                                                block4_vec_store(
                                                    acc_c,
                                                    block4_vec_load(acc_c) +
                                                        cw_orig_v * value_scale_k_v8 * contrib_v *
                                                            block4_vec_broadcast(gamma_k_arr[k]));
                                            }
                                        }
                                        *mdx_row += block4_vec_hsum(cw_start_v * S_v * dyv_v);
                                    }
                                    // ONE update, using the batch-aggregated g/contrib.
                                    Block4Vec cw_v = cw_start_v;
                                    if (training) {
                                        ci_v = SynapsePolicyVec::update_ci(
                                            ci_v, g_agg_v, contrib_agg_v, beta2_v, min_decay_frac_v,
                                            max_ci_v);
                                        const Block4Vec delta_v = SynapsePolicyVec::update_cw(
                                            g_agg_v, ci_v, S_v, effective_lr_v, eps_v,
                                            damp_by_importance, max_abs_delta_v, scale_invariant);
                                        cw_v += delta_v;
                                    }
                                    block4_vec_store(cw4_8, cw_v);
                                    block4_vec_store(ci4_8, ci_v);
                                    for (std::size_t k = 0; k < rank; ++k) {
                                        block4_vec_store(
                                            mcol4_8_rank[k],
                                            block4_vec_load(mcol_acc_raw8 + k * BLOCK4_TILE));
                                        block4_vec_store(mcol4_8_rank_contrib[k],
                                                         block4_vec_load(mcol_acc_raw8_contrib +
                                                                         k * BLOCK4_TILE));
                                    }
                                } else {
                                    // Boundary tile-column: scalar bounds-checked
                                    // fallback. cw_start4_8/g_agg4_8/contrib_agg4_8:
                                    // FIXED per-lj snapshots for the whole batch
                                    // loop -- see the scattered path's cw_start
                                    // comment.
                                    value_type cw_start4_8[BLOCK4_TILE];
                                    double g_agg4_8[BLOCK4_TILE] = {0.0};
                                    double contrib_agg4_8[BLOCK4_TILE] = {0.0};
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj)
                                        cw_start4_8[lj] = cw4_8[lj];
                                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                                        const value_type iv =
                                            input[static_cast<std::size_t>(b) * in_cols + row];
                                        value_type* mdx_row =
                                            mdx + static_cast<std::size_t>(b) * in_cols + row;
                                        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                            if (!col_valid4_8[lj])
                                                continue;
                                            const value_type dyv =
                                                output_grad[static_cast<std::size_t>(b) * n_out +
                                                            col4_8[lj]];
                                            const value_type g = dyv * iv;
                                            if (learning_rate != value_type(0)) {
                                                // Additive g+contrib combination.
                                                const value_type contrib =
                                                    iv * (cw_start4_8[lj] * combined_scale4_8[lj]);
                                                g_agg4_8[lj] += static_cast<double>(g);
                                                contrib_agg4_8[lj] += static_cast<double>(contrib);
                                                for (std::size_t k = 0; k < rank; ++k) {
                                                    // AQRS gamma -- see above.
                                                    mrow_local8_k[k] +=
                                                        static_cast<double>(cw_orig4_8[lj]) *
                                                        static_cast<double>(out_scale_k4_8[k][lj]) *
                                                        static_cast<double>(gamma_k_arr[k]) * g;
                                                    mcol4_8_rank[k][lj] += cw_orig4_8[lj] *
                                                                           value_scale_k8[k] *
                                                                           gamma_k_arr[k] * g;
                                                    mgamma_local8_k[k] +=
                                                        static_cast<double>(cw_orig4_8[lj]) *
                                                        static_cast<double>(out_scale_k4_8[k][lj]) *
                                                        static_cast<double>(value_scale_k8[k]) * g;
                                                    mrow_local8_k_contrib[k] +=
                                                        static_cast<double>(cw_orig4_8[lj]) *
                                                        static_cast<double>(out_scale_k4_8[k][lj]) *
                                                        static_cast<double>(gamma_k_arr[k]) *
                                                        contrib;
                                                    mcol4_8_rank_contrib[k][lj] +=
                                                        cw_orig4_8[lj] * value_scale_k8[k] *
                                                        gamma_k_arr[k] * contrib;
                                                    mgamma_local8_k_contrib[k] +=
                                                        static_cast<double>(cw_orig4_8[lj]) *
                                                        static_cast<double>(out_scale_k4_8[k][lj]) *
                                                        static_cast<double>(value_scale_k8[k]) *
                                                        contrib;
                                                }
                                            }
                                            *mdx_row +=
                                                cw_start4_8[lj] * combined_scale4_8[lj] * dyv;
                                        }
                                    }
                                    // ONE update per lj, using the batch-aggregated g/contrib.
                                    if (learning_rate != value_type(0)) {
                                        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                            if (!col_valid4_8[lj])
                                                continue;
                                            const value_type g_agg =
                                                static_cast<value_type>(g_agg4_8[lj]);
                                            const value_type contrib_agg =
                                                static_cast<value_type>(contrib_agg4_8[lj]);
                                            ci4_8[lj] = SynapsePolicy::update_ci(
                                                ci4_8[lj], g_agg, contrib_agg, beta2,
                                                min_decay_frac, max_ci);
                                            cw4_8[lj] = cw_start4_8[lj] +
                                                        SynapsePolicy::update_cw(
                                                            g_agg, ci4_8[lj], combined_scale4_8[lj],
                                                            effective_lr, eps, damp_by_importance,
                                                            max_abs_delta, scale_invariant);
                                        }
                                    }
                                }

                                if (learning_rate != value_type(0)) {
                                    for (std::size_t k = 0; k < rank; ++k) {
                                        mrow_at(row, k) += mrow_local8_k[k];
                                        mrow_at_contrib(row, k) += mrow_local8_k_contrib[k];
                                        mgamma_at(k) += static_cast<value_type>(mgamma_local8_k[k]);
                                        mgamma_at_contrib(k) +=
                                            static_cast<value_type>(mgamma_local8_k_contrib[k]);
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
                                                mcol_at_contrib(col4_8[lj], k) +=
                                                    mcol4_8_rank_contrib[k][lj];
                                            }
                                            const uint32_t slot = Block4Tile8::slot_index(li, lj);
                                            // was_live4_8[lj] gate -- see its own
                                            // declaration comment above: a cell
                                            // that was never a real synapse must
                                            // stay allowed to round to 0.
                                            if (was_live4_8[lj]) {
                                                tdata[slot] =
                                                    fp8_quantize_stochastic_live(cw4_8[lj]);
                                                tdata[BLOCK4_TILE + slot] =
                                                    fp8_quantize_stochastic_live_nonneg(
                                                        ci4_8[lj] / combined_imp_scale4_8[lj]);
                                            } else {
                                                tdata[slot] = fp8_quantize_stochastic(cw4_8[lj]);
                                                tdata[BLOCK4_TILE + slot] = fp8_quantize_stochastic(
                                                    ci4_8[lj] / combined_imp_scale4_8[lj]);
                                            }
                                        }
                                        tile_dirty = true;
                                    } else {
                                        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                            if (!col_valid4_8[lj])
                                                continue;
                                            for (std::size_t k = 0; k < rank; ++k) {
                                                mcol_at(col4_8[lj], k) += mcol4_8_rank[k][lj];
                                                mcol_at_contrib(col4_8[lj], k) +=
                                                    mcol4_8_rank_contrib[k][lj];
                                            }
                                            const uint32_t slot = Block4Tile8::slot_index(li, lj);
                                            if (was_live4_8[lj]) {
                                                tdata[slot] =
                                                    fp8_quantize_stochastic_live(cw4_8[lj]);
                                                tdata[BLOCK4_TILE + slot] =
                                                    fp8_quantize_stochastic_live_nonneg(
                                                        ci4_8[lj] / combined_imp_scale4_8[lj]);
                                            } else {
                                                tdata[slot] = fp8_quantize_stochastic(cw4_8[lj]);
                                                tdata[BLOCK4_TILE + slot] = fp8_quantize_stochastic(
                                                    ci4_8[lj] / combined_imp_scale4_8[lj]);
                                            }
                                            tile_dirty = true;
                                        }
                                    }
                                }
                            } else {
                                // Hypothetical non-float value_type (never
                                // actually instantiated): plain scalar
                                // per-lj-sequential fallback, same math as
                                // the SIMD path above.
                                std::vector<value_type> value_scale_k8_fb(rank);
                                for (std::size_t k = 0; k < rank; ++k)
                                    value_scale_k8_fb[k] = weights.get_value_scale_k(row, k);
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                                    if (col >= n_out)
                                        continue;
                                    const uint32_t slot = Block4Tile8::slot_index(li, lj);
                                    std::vector<value_type> out_scale_k8_fb(rank);
                                    for (std::size_t k = 0; k < rank; ++k)
                                        out_scale_k8_fb[k] = weights.get_output_scale_k(col, k);
                                    const value_type out_imp_scale =
                                        weights.get_output_importance_scale(col);
                                    const value_type combined_scale = weights.get_scale(row, col);
                                    const value_type combined_imp_scale = imp_scale * out_imp_scale;
                                    const value_type cw_orig = fp8_decode_bits(tdata[slot]);
                                    const value_type cw_start = cw_orig * combined_scale;
                                    value_type ci = fp8_decode_bits(tdata[BLOCK4_TILE + slot]) *
                                                    combined_imp_scale;

                                    std::vector<value_type> mcol_local_k(rank, value_type(0));
                                    std::vector<double> mrow_local_k(rank, 0.0);
                                    std::vector<value_type> mcol_local_k_contrib(rank,
                                                                                 value_type(0));
                                    std::vector<double> mrow_local_k_contrib(rank, 0.0);
                                    std::vector<double> mgamma_local_k(rank, 0.0);
                                    std::vector<double> mgamma_local_k_contrib(rank, 0.0);
                                    double g_agg = 0.0, contrib_agg = 0.0;
                                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                                        const value_type iv =
                                            input[static_cast<std::size_t>(b) * in_cols + row];
                                        value_type* mdx_row =
                                            mdx + static_cast<std::size_t>(b) * in_cols + row;
                                        const value_type dyv =
                                            output_grad[static_cast<std::size_t>(b) * n_out + col];
                                        const value_type g = dyv * iv;
                                        if (learning_rate != value_type(0)) {
                                            // Additive g+contrib combination.
                                            const value_type contrib = iv * cw_start;
                                            g_agg += static_cast<double>(g);
                                            contrib_agg += static_cast<double>(contrib);
                                            for (std::size_t k = 0; k < rank; ++k) {
                                                // AQRS gamma -- see above.
                                                mrow_local_k[k] +=
                                                    static_cast<double>(cw_orig) *
                                                    static_cast<double>(out_scale_k8_fb[k]) *
                                                    static_cast<double>(gamma_k_arr[k]) * g;
                                                mcol_local_k[k] += cw_orig * value_scale_k8_fb[k] *
                                                                   gamma_k_arr[k] * g;
                                                mgamma_local_k[k] +=
                                                    static_cast<double>(cw_orig) *
                                                    static_cast<double>(out_scale_k8_fb[k]) *
                                                    static_cast<double>(value_scale_k8_fb[k]) * g;
                                                mrow_local_k_contrib[k] +=
                                                    static_cast<double>(cw_orig) *
                                                    static_cast<double>(out_scale_k8_fb[k]) *
                                                    static_cast<double>(gamma_k_arr[k]) * contrib;
                                                mcol_local_k_contrib[k] += cw_orig *
                                                                           value_scale_k8_fb[k] *
                                                                           gamma_k_arr[k] * contrib;
                                                mgamma_local_k_contrib[k] +=
                                                    static_cast<double>(cw_orig) *
                                                    static_cast<double>(out_scale_k8_fb[k]) *
                                                    static_cast<double>(value_scale_k8_fb[k]) *
                                                    contrib;
                                            }
                                        }
                                        *mdx_row += cw_start * dyv;
                                    }
                                    // ONE update, using the batch-aggregated g/contrib.
                                    value_type cw = cw_start;
                                    if (learning_rate != value_type(0)) {
                                        ci = SynapsePolicy::update_ci(
                                            ci, static_cast<value_type>(g_agg),
                                            static_cast<value_type>(contrib_agg), beta2,
                                            min_decay_frac, max_ci);
                                        cw += SynapsePolicy::update_cw(
                                            static_cast<value_type>(g_agg), ci, value_type(1),
                                            effective_lr, eps, damp_by_importance, max_abs_delta,
                                            scale_invariant);
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            mrow_at(row, k) += mrow_local_k[k];
                                            mcol_at(col, k) += mcol_local_k[k];
                                            mrow_at_contrib(row, k) += mrow_local_k_contrib[k];
                                            mcol_at_contrib(col, k) += mcol_local_k_contrib[k];
                                            mgamma_at(k) +=
                                                static_cast<value_type>(mgamma_local_k[k]);
                                            mgamma_at_contrib(k) +=
                                                static_cast<value_type>(mgamma_local_k_contrib[k]);
                                        }
                                        // was_live gate -- see was_live4_8's
                                        // declaration comment (SIMD branch
                                        // above) for the full rationale. tdata
                                        // still holds the PRE-update bytes here.
                                        const bool was_live =
                                            (cw_orig != value_type(0)) ||
                                            (fp8_decode_bits(tdata[BLOCK4_TILE + slot]) !=
                                             value_type(0));
                                        if (was_live) {
                                            tdata[slot] =
                                                fp8_quantize_stochastic_live(cw / combined_scale);
                                            tdata[BLOCK4_TILE + slot] =
                                                fp8_quantize_stochastic_live_nonneg(
                                                    ci / combined_imp_scale);
                                        } else {
                                            tdata[slot] =
                                                fp8_quantize_stochastic(cw / combined_scale);
                                            tdata[BLOCK4_TILE + slot] =
                                                fp8_quantize_stochastic(ci / combined_imp_scale);
                                        }
                                        tile_dirty = true;
                                    }
                                }
                            }
                            continue; // this li done -- skip the FP4 branch entirely
                        }
                        const uint8_t byte0 = tdata[Block4Tile::slot_index(li, 0)],
                                      byte1 = tdata[Block4Tile::slot_index(li, 1)],
                                      byte2 = tdata[Block4Tile::slot_index(li, 2)],
                                      byte3 = tdata[Block4Tile::slot_index(li, 3)];
                        const value_type w_decoded_arr[BLOCK4_TILE] = {
                            FP4_TABLE[byte0 & 0xFu], FP4_TABLE[byte1 & 0xFu],
                            FP4_TABLE[byte2 & 0xFu], FP4_TABLE[byte3 & 0xFu]};
                        const value_type imp_decoded_arr[BLOCK4_TILE] = {
                            FP4_TABLE[(byte0 >> 4) & 0xFu], FP4_TABLE[(byte1 >> 4) & 0xFu],
                            FP4_TABLE[(byte2 >> 4) & 0xFu], FP4_TABLE[(byte3 >> 4) & 0xFu]};

                        // value_scale_k(row,k), fetched once per row -- matches
                        // the scattered path's own once-per-row granularity
                        // (see disldo_backward's non-DeferredScaleWrite branch).
                        value_type* value_scale_k =
                            weights.scale_rank_scratch.value_scale_k.data() +
                            static_cast<std::size_t>(tid) * rank;
                        for (std::size_t k = 0; k < rank; ++k)
                            value_scale_k[k] = weights.get_value_scale_k(row, k);

                        std::size_t col4[BLOCK4_TILE];
                        bool col_valid4[BLOCK4_TILE];
                        value_type combined_scale4[BLOCK4_TILE], combined_imp_scale4[BLOCK4_TILE];
                        // quant4: the stored CODE, now the primary optimized
                        // quantity (not true units). quant_orig4: immutable
                        // snapshot of its call-entry value, for contrib.
                        value_type quant4[BLOCK4_TILE], ci4[BLOCK4_TILE], quant_orig4[BLOCK4_TILE];
                        const Flat2DView out_scale_k4{
                            weights.scale_rank_scratch.out_scale_k.data() +
                                static_cast<std::size_t>(tid) * rank * BLOCK4_TILE,
                            BLOCK4_TILE};
                        // was_live4[lj]: same gating as was_live4_8 (FP8
                        // branch above). See docs/research/linear_disldo.rst.
                        bool was_live4[BLOCK4_TILE];
                        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                            const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                            col_valid4[lj] = col < n_out;
                            if (!col_valid4[lj]) {
                                col4[lj] = 0;
                                combined_scale4[lj] = combined_imp_scale4[lj] = value_type(0);
                                quant4[lj] = ci4[lj] = quant_orig4[lj] = value_type(0);
                                was_live4[lj] = false;
                                for (std::size_t k = 0; k < rank; ++k)
                                    out_scale_k4[k][lj] = value_type(0);
                                continue;
                            }
                            col4[lj] = col;
                            was_live4[lj] = (w_decoded_arr[lj] != value_type(0)) ||
                                            (imp_decoded_arr[lj] != value_type(0));
                            const value_type out_imp_scale =
                                weights.get_output_importance_scale(col);
                            // S(row,col) = sum_k value_scale_k(row,k)*
                            // output_scale_k(col,k) -- see disldo_backward's
                            // scattered-path comment on the chain-rule fix this
                            // mirrors (get_scale already sums over rank).
                            combined_scale4[lj] = weights.get_scale(row, col);
                            combined_imp_scale4[lj] = imp_scale * out_imp_scale;
                            quant4[lj] = quant_orig4[lj] = w_decoded_arr[lj];
                            ci4[lj] = imp_decoded_arr[lj] * combined_imp_scale4[lj];
                            for (std::size_t k = 0; k < rank; ++k)
                                out_scale_k4[k][lj] = weights.get_output_scale_k(col, k);
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
                        const Flat2DView mcol4_rank{srs.mcol_rank.data() + tid_rank_tile,
                                                    BLOCK4_TILE};
                        double* mrow_local_k = srs.mrow_local_k.data() + tid_rank;
                        const Flat2DView mcol4_rank_contrib{
                            srs.mcol_rank_contrib.data() + tid_rank_tile, BLOCK4_TILE};
                        double* mrow_local_k_contrib = srs.mrow_local_k_contrib.data() + tid_rank;
                        // AQRS gamma's own gradient (task #273/#283) -- folded
                        // into the shared mgamma_at/mgamma_at_contrib once this
                        // row finishes, matching mrow_local_k's own
                        // fold-into-mrow_at pattern below.
                        double* mgamma_local_k = srs.mgamma_local_k.data() + tid_rank;
                        double* mgamma_local_k_contrib =
                            srs.mgamma_local_k_contrib.data() + tid_rank;
                        std::fill(mcol4_rank[0], mcol4_rank[0] + rank * BLOCK4_TILE, value_type(0));
                        std::fill(mrow_local_k, mrow_local_k + rank, 0.0);
                        std::fill(mcol4_rank_contrib[0], mcol4_rank_contrib[0] + rank * BLOCK4_TILE,
                                  value_type(0));
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
                        if constexpr (std::is_same_v<value_type, float> &&
                                      !SILI_BLOCK4_FORCE_SCALAR_BACKWARD) {
                            if (full_tile_cols) {
                                const Block4Vec effective_lr_v = block4_vec_broadcast(effective_lr);
                                const Block4Vec beta2_v = block4_vec_broadcast(beta2);
                                const Block4Vec eps_v = block4_vec_broadcast(eps);
                                const Block4Vec min_decay_frac_v =
                                    block4_vec_broadcast(min_decay_frac);
                                const Block4Vec max_ci_v = block4_vec_broadcast(max_ci);
                                const Block4Vec max_abs_delta_v =
                                    block4_vec_broadcast(max_abs_delta);
                                const Block4Vec combined_scale_v = block4_vec_load(combined_scale4);
                                // quant_start_v: FIXED for the whole batch loop
                                // -- see the scattered path's identical
                                // cw_start comment for the full rationale.
                                // quant_floor_v computed once from this fixed
                                // snapshot too (still exact/signed for every
                                // already-escaped synapse, see its own comment
                                // below).
                                const Block4Vec quant_start_v = block4_vec_load(quant4);
                                Block4Vec ci_v = block4_vec_load(ci4);
                                Block4Vec quant_floor_v;
                                for (int lane = 0; lane < BLOCK4_TILE; ++lane)
                                    quant_floor_v[lane] = (quant_start_v[lane] == 0.0f)
                                                              ? zero_escape_eps
                                                              : quant_start_v[lane];
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
                                value_type* mcol_acc_raw_contrib =
                                    srs.mcol_acc_raw_contrib.data() + tid_rank_tile;
                                for (std::size_t k = 0; k < rank; ++k) {
                                    block4_vec_store(mcol_acc_raw + k * BLOCK4_TILE,
                                                     block4_vec_broadcast(0.0f));
                                    block4_vec_store(mcol_acc_raw_contrib + k * BLOCK4_TILE,
                                                     block4_vec_broadcast(0.0f));
                                }
                                const bool training = (learning_rate != value_type(0));
                                for (SIZE_TYPE b = 0; b < batch; ++b) {
                                    const value_type iv =
                                        input[static_cast<std::size_t>(b) * in_cols + row];
                                    value_type* mdx_row =
                                        mdx + static_cast<std::size_t>(b) * in_cols + row;
                                    const Block4Vec dyv_v = block4_vec_load(
                                        output_grad + static_cast<std::size_t>(b) * n_out +
                                        col_base);
                                    const Block4Vec g_v = dyv_v * block4_vec_broadcast(iv);
                                    if (training) {
                                        // RMSprop-style ci, additive g+contrib
                                        // combination -- see disldo_backward's
                                        // own docstring and
                                        // docs/research/linear_disldo.rst.
                                        const Block4Vec contrib_v =
                                            (quant_start_v * combined_scale_v) *
                                            block4_vec_broadcast(iv);
                                        g_agg_v += g_v;
                                        contrib_agg_v += contrib_v;
                                        // quant_floor: signed quant, except at
                                        // quant==0 where zero_escape_eps
                                        // substitutes -- see
                                        // docs/research/linear_disldo.rst.
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            // AQRS gamma -- see above.
                                            const Block4Vec value_scale_k_v =
                                                block4_vec_broadcast(value_scale_k[k]);
                                            const Block4Vec out_scale_k_v =
                                                block4_vec_load(out_scale_k4[k]);
                                            mrow_local_k[k] +=
                                                static_cast<double>(block4_vec_hsum(
                                                    quant_floor_v * out_scale_k_v * g_v)) *
                                                static_cast<double>(gamma_k_arr[k]);
                                            mgamma_local_k[k] += static_cast<double>(
                                                block4_vec_hsum(quant_floor_v * out_scale_k_v *
                                                                value_scale_k_v * g_v));
                                            mrow_local_k_contrib[k] +=
                                                static_cast<double>(block4_vec_hsum(
                                                    quant_floor_v * out_scale_k_v * contrib_v)) *
                                                static_cast<double>(gamma_k_arr[k]);
                                            mgamma_local_k_contrib[k] += static_cast<double>(
                                                block4_vec_hsum(quant_floor_v * out_scale_k_v *
                                                                value_scale_k_v * contrib_v));
                                            value_type* acc = mcol_acc_raw + k * BLOCK4_TILE;
                                            block4_vec_store(
                                                acc, block4_vec_load(acc) +
                                                         quant_floor_v * value_scale_k_v * g_v *
                                                             block4_vec_broadcast(gamma_k_arr[k]));
                                            value_type* acc_c =
                                                mcol_acc_raw_contrib + k * BLOCK4_TILE;
                                            block4_vec_store(
                                                acc_c,
                                                block4_vec_load(acc_c) +
                                                    quant_floor_v * value_scale_k_v * contrib_v *
                                                        block4_vec_broadcast(gamma_k_arr[k]));
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
                                    ci_v = SynapsePolicyVec::update_ci(ci_v, g_agg_v, contrib_agg_v,
                                                                       beta2_v, min_decay_frac_v,
                                                                       max_ci_v);
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
                                    block4_vec_store(
                                        mcol4_rank[k],
                                        block4_vec_load(mcol_acc_raw + k * BLOCK4_TILE));
                                    block4_vec_store(
                                        mcol4_rank_contrib[k],
                                        block4_vec_load(mcol_acc_raw_contrib + k * BLOCK4_TILE));
                                }
                            } else {
                                // Boundary tile-column (rare -- only the last
                                // one, when n_out isn't a multiple of
                                // BLOCK4_TILE): scalar bounds-checked fallback,
                                // not on the fast path, doesn't need SIMD. Same
                                // math as the SIMD path above.
                                value_type quant_start4[BLOCK4_TILE], quant_floor4[BLOCK4_TILE];
                                double g_agg4[BLOCK4_TILE] = {0.0},
                                       contrib_agg4[BLOCK4_TILE] = {0.0};
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    quant_start4[lj] = quant4[lj];
                                    quant_floor4[lj] = (quant4[lj] == value_type(0))
                                                           ? zero_escape_eps
                                                           : quant4[lj];
                                }
                                for (SIZE_TYPE b = 0; b < batch; ++b) {
                                    const value_type iv =
                                        input[static_cast<std::size_t>(b) * in_cols + row];
                                    value_type* mdx_row =
                                        mdx + static_cast<std::size_t>(b) * in_cols + row;
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (!col_valid4[lj])
                                            continue;
                                        const value_type dyv =
                                            output_grad[static_cast<std::size_t>(b) * n_out +
                                                        col4[lj]];
                                        const value_type g = dyv * iv;
                                        const value_type S = combined_scale4[lj];
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
                                                // AQRS gamma -- see the SIMD branch's identical
                                                // comment above.
                                                mrow_local_k[k] +=
                                                    static_cast<double>(quant_floor4[lj]) *
                                                    static_cast<double>(out_scale_k4[k][lj]) *
                                                    static_cast<double>(gamma_k_arr[k]) * g;
                                                mcol4_rank[k][lj] += quant_floor4[lj] *
                                                                     value_scale_k[k] *
                                                                     gamma_k_arr[k] * g;
                                                mgamma_local_k[k] +=
                                                    static_cast<double>(quant_floor4[lj]) *
                                                    static_cast<double>(out_scale_k4[k][lj]) *
                                                    static_cast<double>(value_scale_k[k]) * g;
                                                mrow_local_k_contrib[k] +=
                                                    static_cast<double>(quant_floor4[lj]) *
                                                    static_cast<double>(out_scale_k4[k][lj]) *
                                                    static_cast<double>(gamma_k_arr[k]) * contrib;
                                                mcol4_rank_contrib[k][lj] +=
                                                    quant_floor4[lj] * value_scale_k[k] *
                                                    gamma_k_arr[k] * contrib;
                                                mgamma_local_k_contrib[k] +=
                                                    static_cast<double>(quant_floor4[lj]) *
                                                    static_cast<double>(out_scale_k4[k][lj]) *
                                                    static_cast<double>(value_scale_k[k]) * contrib;
                                            }
                                        }
                                        *mdx_row += quant_start4[lj] * S * dyv;
                                    }
                                }
                                // ONE update per lj, using the batch-aggregated g/contrib.
                                if (learning_rate != value_type(0)) {
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (!col_valid4[lj])
                                            continue;
                                        const value_type S = combined_scale4[lj];
                                        const value_type g_agg =
                                            static_cast<value_type>(g_agg4[lj]);
                                        const value_type contrib_agg =
                                            static_cast<value_type>(contrib_agg4[lj]);
                                        ci4[lj] =
                                            SynapsePolicy::update_ci(ci4[lj], g_agg, contrib_agg,
                                                                     beta2, min_decay_frac, max_ci);
                                        quant4[lj] =
                                            quant_start4[lj] + SynapsePolicy::update_cw(
                                                                   g_agg, ci4[lj], S, effective_lr,
                                                                   eps, damp_by_importance,
                                                                   max_abs_delta, scale_invariant);
                                    }
                                }
                            }
                        } else {
                            // Hypothetical non-float value_type (never
                            // actually instantiated): bounds-checked array
                            // form, identical math to the boundary branch
                            // above.
                            value_type quant_start4[BLOCK4_TILE], quant_floor4[BLOCK4_TILE];
                            double g_agg4[BLOCK4_TILE] = {0.0}, contrib_agg4[BLOCK4_TILE] = {0.0};
                            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                quant_start4[lj] = quant4[lj];
                                quant_floor4[lj] =
                                    (quant4[lj] == value_type(0)) ? zero_escape_eps : quant4[lj];
                            }
                            for (SIZE_TYPE b = 0; b < batch; ++b) {
                                const value_type iv =
                                    input[static_cast<std::size_t>(b) * in_cols + row];
                                value_type* mdx_row =
                                    mdx + static_cast<std::size_t>(b) * in_cols + row;
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    if (!col_valid4[lj])
                                        continue;
                                    const value_type dyv =
                                        output_grad[static_cast<std::size_t>(b) * n_out + col4[lj]];
                                    const value_type g = dyv * iv;
                                    const value_type S = combined_scale4[lj];
                                    if (learning_rate != value_type(0)) {
                                        // Additive g+contrib combination,
                                        // FIXED batch-start snapshots.
                                        const value_type contrib = iv * (quant_start4[lj] * S);
                                        g_agg4[lj] += static_cast<double>(g);
                                        contrib_agg4[lj] += static_cast<double>(contrib);
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            mrow_local_k[k] +=
                                                static_cast<double>(quant_floor4[lj]) *
                                                static_cast<double>(out_scale_k4[k][lj]) * g;
                                            mcol4_rank[k][lj] +=
                                                quant_floor4[lj] * value_scale_k[k] * g;
                                            mrow_local_k_contrib[k] +=
                                                static_cast<double>(quant_floor4[lj]) *
                                                static_cast<double>(out_scale_k4[k][lj]) * contrib;
                                            mcol4_rank_contrib[k][lj] +=
                                                quant_floor4[lj] * value_scale_k[k] * contrib;
                                        }
                                    }
                                    *mdx_row += quant_start4[lj] * S * dyv;
                                }
                            }
                            // ONE update per lj, using the batch-aggregated g/contrib.
                            if (learning_rate != value_type(0)) {
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    if (!col_valid4[lj])
                                        continue;
                                    const value_type S = combined_scale4[lj];
                                    const value_type g_agg = static_cast<value_type>(g_agg4[lj]);
                                    const value_type contrib_agg =
                                        static_cast<value_type>(contrib_agg4[lj]);
                                    ci4[lj] = SynapsePolicy::update_ci(
                                        ci4[lj], g_agg, contrib_agg, beta2, min_decay_frac, max_ci);
                                    quant4[lj] =
                                        quant_start4[lj] +
                                        SynapsePolicy::update_cw(g_agg, ci4[lj], S, effective_lr,
                                                                 eps, damp_by_importance,
                                                                 max_abs_delta, scale_invariant);
                                }
                            }
                        }
                        if (learning_rate != value_type(0)) {
                            for (std::size_t k = 0; k < rank; ++k) {
                                mrow_at(row, k) += mrow_local_k[k];
                                mrow_at_contrib(row, k) += mrow_local_k_contrib[k];
                                mgamma_at(k) += static_cast<value_type>(mgamma_local_k[k]);
                                mgamma_at_contrib(k) +=
                                    static_cast<value_type>(mgamma_local_k_contrib[k]);
                            }
                            if constexpr (!StochasticRounding) {
                                // Deterministic: no SIMD deterministic kernel
                                // exists yet, so always scalar fp4_quantize().
                                // Real non-determinism bug fixed here (block4
                                // used to always stochastic-round regardless
                                // of StochasticRounding) -- see
                                // docs/research/linear_disldo.rst.
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    if (!col_valid4[lj])
                                        continue;
                                    for (std::size_t k = 0; k < rank; ++k) {
                                        mcol_at(col4[lj], k) += mcol4_rank[k][lj];
                                        mcol_at_contrib(col4[lj], k) += mcol4_rank_contrib[k][lj];
                                    }
                                    const value_type imp_ratio = ci4[lj] / combined_imp_scale4[lj];
                                    // was_live4[lj] gate -- see its own
                                    // declaration comment above for the full
                                    // rationale.
                                    const uint8_t new_w = was_live4[lj]
                                                              ? fp4_quantize_live(quant4[lj])
                                                              : fp4_quantize(quant4[lj]);
                                    const uint8_t new_imp = was_live4[lj]
                                                                ? fp4_quantize_live(imp_ratio)
                                                                : fp4_quantize(imp_ratio);
                                    tdata[Block4Tile::slot_index(li, lj)] =
                                        uint8_t((new_imp << 4) | new_w);
                                    tile_dirty = true;
                                }
                            } else if constexpr (std::is_same_v<value_type, float> &&
                                                 !SILI_BLOCK4_FORCE_SCALAR_BACKWARD) {
                                if (full_tile_cols) {
                                    // Per-lane scalar quantize gated on
                                    // was_live4[lj] -- not the SIMD live-encode
                                    // kernel, which can't express "some lanes
                                    // live, some not" without a mask-blend.
                                    uint8_t new_w_codes[BLOCK4_TILE], new_imp_codes[BLOCK4_TILE];
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        const value_type imp_ratio =
                                            ci4[lj] / combined_imp_scale4[lj];
                                        if (was_live4[lj]) {
                                            new_w_codes[lj] =
                                                fp4_quantize_stochastic_live(quant4[lj]);
                                            new_imp_codes[lj] =
                                                fp4_quantize_stochastic_live_nonneg(imp_ratio);
                                        } else {
                                            new_w_codes[lj] = fp4_quantize_stochastic(quant4[lj]);
                                            new_imp_codes[lj] = fp4_quantize_stochastic(imp_ratio);
                                        }
                                    }
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            mcol_at(col4[lj], k) += mcol4_rank[k][lj];
                                            mcol_at_contrib(col4[lj], k) +=
                                                mcol4_rank_contrib[k][lj];
                                        }
                                        tdata[Block4Tile::slot_index(li, lj)] =
                                            uint8_t((new_imp_codes[lj] << 4) | new_w_codes[lj]);
                                    }
                                    tile_dirty = true;
                                } else {
                                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                        if (!col_valid4[lj])
                                            continue;
                                        for (std::size_t k = 0; k < rank; ++k) {
                                            mcol_at(col4[lj], k) += mcol4_rank[k][lj];
                                            mcol_at_contrib(col4[lj], k) +=
                                                mcol4_rank_contrib[k][lj];
                                        }
                                        const value_type imp_ratio =
                                            ci4[lj] / combined_imp_scale4[lj];
                                        uint8_t new_w, new_imp;
                                        if (was_live4[lj]) {
                                            new_w = fp4_quantize_stochastic_live(quant4[lj]);
                                            new_imp =
                                                fp4_quantize_stochastic_live_nonneg(imp_ratio);
                                        } else {
                                            new_w = fp4_quantize_stochastic(quant4[lj]);
                                            new_imp = fp4_quantize_stochastic(imp_ratio);
                                        }
                                        tdata[Block4Tile::slot_index(li, lj)] =
                                            uint8_t((new_imp << 4) | new_w);
                                        tile_dirty = true;
                                    }
                                }
                            } else {
                                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                                    if (!col_valid4[lj])
                                        continue;
                                    for (std::size_t k = 0; k < rank; ++k) {
                                        mcol_at(col4[lj], k) += mcol4_rank[k][lj];
                                        mcol_at_contrib(col4[lj], k) += mcol4_rank_contrib[k][lj];
                                    }
                                    const value_type imp_ratio = ci4[lj] / combined_imp_scale4[lj];
                                    uint8_t new_w, new_imp;
                                    if (was_live4[lj]) {
                                        new_w = fp4_quantize_stochastic_live(quant4[lj]);
                                        new_imp = fp4_quantize_stochastic_live_nonneg(imp_ratio);
                                    } else {
                                        new_w = fp4_quantize_stochastic(quant4[lj]);
                                        new_imp = fp4_quantize_stochastic(imp_ratio);
                                    }
                                    tdata[Block4Tile::slot_index(li, lj)] =
                                        uint8_t((new_imp << 4) | new_w);
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
                    // when nothing is ever written.
                    auto bc_cursor = weights.block4.row_cursor(br);
                    std::size_t elem_pos = BL4.elem_start[br];
                    std::size_t byte_pos = weights.block4.tile_byte_start[br];
                    for (std::size_t row_ti = row_ti_start[br]; row_ti < row_ti_start[br + 1];
                         ++row_ti, ++elem_pos) {
                        const uint32_t bc = bc_cursor.advance();
                        const auto tile =
                            weights.block4.at_index(uint32_t(br), bc, elem_pos, byte_pos);
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
                    for (std::size_t row_ti = row_ti_start[br]; row_ti < row_ti_start[br + 1];
                         ++row_ti) {
                        const std::size_t e = row_ti - row_ti_start[br];
                        const uint32_t bc = ws.bc[e];
                        const std::size_t this_byte_pos = live_byte_pos;
                        // Sized per VALUES_TYPE: FP8's Block4Tile8 is 32
                        // bytes/tile (2/slot), not FP4's 16 (1/slot) --
                        // a real stack buffer overflow otherwise (caught via
                        // -fsanitize=address / gcc's own stringop-overflow
                        // warning when this was still hardcoded to
                        // BLOCK4_TILE_SLOTS for both types).
                        uint8_t scratch_buf[std::is_same_v<VALUES_TYPE, FP8BiValues>
                                                ? BLOCK4_TILE_SLOTS8_BYTES
                                                : BLOCK4_TILE_SLOTS];
                        weights.block4.unpack_workspace_tile(ws, e, this_byte_pos, scratch_buf);
                        const bool tile_dirty = process_tile(bc, scratch_buf);
                        if (tile_dirty)
                            weights.block4.commit_dirty_tile_in_workspace(ws, e, this_byte_pos,
                                                                          scratch_buf);
                        if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>)
                            live_byte_pos +=
                                block4_stored_tile_len8(ws.is_sparse[e], &ws.bytes[this_byte_pos]);
                        else
                            live_byte_pos +=
                                block4_stored_tile_len(ws.is_sparse[e], &ws.bytes[this_byte_pos]);
                    } // closes for (row_ti...)
                    // Merge back -- evicts lowest-|true-importance| synapses
                    // only if this row genuinely grew past its own current
                    // headroom (see Block4Store::merge_row_workspace's
                    // comment).
                    weights.block4.merge_row_workspace(
                        br, ws,
                        [&](std::size_t ev_row, std::size_t ev_col, uint8_t ev_imp_code) -> double {
                            const value_type imp_scale = weights.get_importance_scale(ev_row);
                            const value_type out_imp_scale =
                                weights.get_output_importance_scale(ev_col);
                            // FP8's Block4Store8::merge_row_workspace passes the
                            // FULL importance byte (an E4M3 code), not FP4's
                            // 4-bit nibble -- see Block4Store8::merge_row_workspace's
                            // own docstring (block4.hpp).
                            const double decoded =
                                std::is_same_v<VALUES_TYPE, FP8BiValues>
                                    ? static_cast<double>(fp8_decode_bits(ev_imp_code))
                                    : static_cast<double>(FP4_TABLE[ev_imp_code & 0xFu]);
                            return decoded * static_cast<double>(imp_scale) *
                                   static_cast<double>(out_imp_scale);
                        });
                }
            } // closes for (br...)
        } // closes #pragma omp parallel

        if (learning_rate != value_type(0)) {
            for (std::size_t row = 0; row < n_in; ++row) {
                const uint32_t nnz_row = row_live_count[row];
                if (nnz_row == 0)
                    continue;
                const value_type scale_eff_lr = learning_rate / static_cast<value_type>(nnz_row);
                // t_row_grad is laid out [thread][row][k] (see its sizing
                // above, matching t_col_grad's own layout). rank=1
                // reproduces the exact original single-sum formula.
                for (std::size_t k = 0; k < rank; ++k) {
                    double sum = 0.0, sum_contrib = 0.0;
                    for (int t = 0; t < num_cpus; ++t) {
                        sum += t_row_grad[(static_cast<std::size_t>(t) * n_in + row) * rank + k];
                        sum_contrib +=
                            t_row_grad_contrib[(static_cast<std::size_t>(t) * n_in + row) * rank +
                                               k];
                    }
                    // Only skip when BOTH are zero -- a zero real gradient
                    // (sum==0) with nonzero forward-contribution signal
                    // still needs to update vs_imp (same zero-escape
                    // property as per-synapse ci, see linear_disldo.hpp's
                    // additive combination -- a dy=0 backward call should
                    // still move importance from activity alone).
                    if (sum == 0.0 && sum_contrib == 0.0)
                        continue;
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
                    if (!std::isfinite(g_agg) || !std::isfinite(contrib_agg))
                        continue;
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
                    const value_type new_vs_imp =
                        beta2 * vs_imp +
                        (value_type(1) - beta2) * (g_agg * g_agg + contrib_agg * contrib_agg);
                    if (!std::isfinite(new_vs_imp))
                        continue;
                    // Same Adam-style bias correction as
                    // RMSpropScalePolicy::update (delta_csr_types.hpp) --
                    // this path updates the SAME value_scale_importance
                    // array via a separately hand-inlined formula (block4
                    // predates the ScalePolicy abstraction), so it needs
                    // the identical fix or it reintroduces the exact
                    // cold-start bug this whole change exists to close.
                    uint32_t& step = weights.get_value_scale_step_k(row, k);
                    ++step;
                    const value_type bias_correction =
                        value_type(1) - std::pow(beta2, static_cast<value_type>(step));
                    const value_type vs_imp_hat =
                        bias_correction > value_type(0) ? new_vs_imp / bias_correction : new_vs_imp;
                    if (!std::isfinite(vs_imp_hat))
                        continue;
                    const value_type new_vs = weights.value_scale[idx] -
                                              scale_eff_lr * g_agg / (std::sqrt(vs_imp_hat) + eps);
                    if (!std::isfinite(new_vs))
                        continue;
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
            if (bl.rows == 0)
                return 0;
            const std::size_t br = row / BLOCK4_TILE;
            if (br >= bl.rows)
                return 0;
            return bl.row_nnz(br) * BLOCK4_TILE;
        };
        bool any_dead_row = false;
        for (std::size_t row = 0; row < n_in; ++row) {
            if (L.row_nnz(row) == 0 && block4_row_live(row) == 0) {
                any_dead_row = true;
                break;
            }
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
                        s += static_cast<double>(weights.get_output_scale_k(col, k)) *
                             static_cast<double>(
                                 output_grad[static_cast<std::size_t>(b) * n_out + col]);
                    }
                    dead_row_S[static_cast<std::size_t>(b) * rank + k] = s;
                }
            }
            for (std::size_t row = 0; row < n_in; ++row) {
                if (L.row_nnz(row) != 0 || block4_row_live(row) != 0)
                    continue;
                for (std::size_t k = 0; k < rank; ++k) {
                    double sum = 0.0;
                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        const value_type iv = input[static_cast<std::size_t>(b) * in_cols + row];
                        if (iv == value_type(0))
                            continue;
                        sum += static_cast<double>(iv) *
                               dead_row_S[static_cast<std::size_t>(b) * rank + k];
                    }
                    if (sum == 0.0)
                        continue;
                    const value_type g_agg = static_cast<value_type>(sum);
                    if (!std::isfinite(g_agg))
                        continue;
                    const std::size_t idx = row * rank + k;
                    value_type& m = weights.value_scale_momentum[idx];
                    value_type& v = weights.value_scale_importance[idx];
                    const value_type new_m = beta1 * m + (value_type(1) - beta1) * g_agg;
                    const value_type new_v = beta2 * v + (value_type(1) - beta2) * g_agg * g_agg;
                    if (!std::isfinite(new_m) || !std::isfinite(new_v))
                        continue;
                    const value_type dead_row_lr = learning_rate / static_cast<value_type>(n_out);
                    const value_type new_vs =
                        weights.value_scale[idx] - dead_row_lr * new_m / (std::sqrt(new_v) + eps);
                    if (!std::isfinite(new_vs))
                        continue;
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
            const std::size_t deg =
                c < weights.out_degree.size() ? static_cast<std::size_t>(weights.out_degree[c]) : 0;
            if (deg == 0)
                continue;
            const value_type col_eff_lr = learning_rate / static_cast<value_type>(deg);
            for (std::size_t k = 0; k < rank; ++k) {
                double col_grad_sum = 0.0;
                double col_grad_sum_contrib = 0.0;
                for (int t = 0; t < num_cpus; ++t) {
                    col_grad_sum +=
                        t_col_grad[static_cast<std::size_t>(t) * n_out * rank + c * rank + k];
                    col_grad_sum_contrib +=
                        t_col_grad_contrib[static_cast<std::size_t>(t) * n_out * rank + c * rank +
                                           k];
                }
                // Scale update via the swappable policy -- same as
                // value_scale's own update above.
                const value_type g_agg = static_cast<value_type>(col_grad_sum);
                const value_type contrib_agg = static_cast<value_type>(col_grad_sum_contrib);
                ScalePolicy::update(weights.output_scale[c * rank + k],
                                    weights.output_scale_importance[c * rank + k], g_agg,
                                    col_eff_lr, beta2, eps, contrib_agg,
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
                gamma_grad_sum_contrib +=
                    t_gamma_grad_contrib[static_cast<std::size_t>(t) * rank + k];
            }
            const value_type g_agg = static_cast<value_type>(gamma_grad_sum);
            const value_type contrib_agg = static_cast<value_type>(gamma_grad_sum_contrib);
            g_agg_by_k[k] = g_agg;
            // Force-size scale_gamma up to k (lazy default preserved) so a
            // direct reference is safe to hand to ScalePolicy::update,
            // matching value_scale/output_scale's own direct-array-access
            // convention above.
            weights.set_scale_gamma_raw_k(k, weights.get_scale_gamma_k(k));
            ScalePolicy::update(weights.scale_gamma[k], weights.get_scale_gamma_state_k(k), g_agg,
                                learning_rate, beta2, eps, contrib_agg,
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
                if (gm > shrink)
                    gm -= shrink;
                else if (gm < -shrink)
                    gm += shrink;
                else
                    gm = value_type(0);
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
            for (std::size_t k = 0; k < rank; ++k)
                gamma_l1_sum += std::fabs(weights.scale_gamma[k]);
            // g_agg_by_k[k] is a RAW per-synapse-accumulated sum (mgamma_at
            // above, summed across every touched row/col pair) -- same
            // layer-width-dependent scaling issue as the additive branch's
            // own dgamma_by_k (see that block's own comment, task #294
            // fix); normalize by n_in*n_out here for the SAME reason, same
            // scope (trigger tracking only, not gamma's own ScalePolicy
            // step above).
            const value_type grad_norm_divisor =
                static_cast<value_type>(n_in) * static_cast<value_type>(n_out);
            for (std::size_t k = 0; k < rank; ++k) {
                const value_type abs_gamma_k = std::fabs(weights.scale_gamma[k]);
                const value_type share_k =
                    gamma_l1_sum > value_type(0) ? abs_gamma_k / gamma_l1_sum : value_type(0);
                weights.update_scale_gamma_ema_k(k, abs_gamma_k, share_k,
                                                 std::fabs(g_agg_by_k[k]) / grad_norm_divisor);
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
                    const value_type final_out_imp_scale =
                        weights.get_output_importance_scale(entry.col);
                    const value_type final_combined_scale = final_val_scale * final_out_scale;
                    const value_type final_combined_imp_scale =
                        final_imp_scale * final_out_imp_scale;
                    if constexpr (StochasticRounding) {
                        ValueAccessor<VALUES_TYPE>::set_stochastic_live(
                            dc.values, entry.vb, entry.cw / final_combined_scale,
                            entry.ci / final_combined_imp_scale);
                    } else {
                        ValueAccessor<VALUES_TYPE>::set_live(dc.values, entry.vb,
                                                             entry.cw / final_combined_scale,
                                                             entry.ci / final_combined_imp_scale);
                    }
                }
            }
        }
    }

    // AQRS additive branch backward (task #277): differentiates
    // disldo_forward's additive-branch block, its own self-contained pass.
    // No-op at additive_rank==0. P is NOT cached from forward -- recomputed
    // from `input` directly. See docs/research/linear_disldo.rst.
    if (weights.additive_rank > 0) {
        const std::size_t r_o = weights.additive_rank;
        std::vector<value_type> P(static_cast<std::size_t>(batch) * r_o, value_type(0));
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type* in_row = input + static_cast<std::size_t>(b) * in_cols;
            value_type* p_row = P.data() + static_cast<std::size_t>(b) * r_o;
            for (std::size_t r = 0; r < n_in; ++r) {
                const value_type iv = in_row[r];
                if (iv == value_type(0))
                    continue;
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
                if (dy == value_type(0))
                    continue;
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
                    acc += weights.get_additive_gamma_k(k) * weights.get_additive_u_k(r, k) *
                           dp_row[k];
                dx_row[r] += acc;
            }
        }
        for (std::size_t r = 0; r < n_in; ++r) {
            for (std::size_t k = 0; k < r_o; ++k) {
                value_type dU_rk = value_type(0);
                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                    if (iv == value_type(0))
                        continue;
                    dU_rk += dP[static_cast<std::size_t>(b) * r_o + k] * iv;
                }
                dU_rk *= weights.get_additive_gamma_k(k);
                if (dU_rk == value_type(0))
                    continue;
                value_type u_val = weights.get_additive_u_k(r, k);
                AdamScalePolicy<value_type>::update(u_val, weights.get_additive_u_state_k(r, k),
                                                    weights.get_additive_u_momentum_k(r, k), dU_rk,
                                                    learning_rate, beta1, beta2, eps,
                                                    &weights.get_additive_u_step_k(r, k));
                weights.set_additive_u_raw_k(r, k, u_val);
            }
        }
        for (std::size_t c = 0; c < n_out; ++c) {
            for (std::size_t k = 0; k < r_o; ++k) {
                value_type dV_ck = value_type(0);
                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type dy = output_grad[static_cast<std::size_t>(b) * n_out + c];
                    if (dy == value_type(0))
                        continue;
                    dV_ck += dy * P[static_cast<std::size_t>(b) * r_o + k];
                }
                dV_ck *= weights.get_additive_gamma_k(k);
                if (dV_ck == value_type(0))
                    continue;
                value_type v_val = weights.get_additive_v_k(c, k);
                AdamScalePolicy<value_type>::update(v_val, weights.get_additive_v_state_k(c, k),
                                                    weights.get_additive_v_momentum_k(c, k), dV_ck,
                                                    learning_rate, beta1, beta2, eps,
                                                    &weights.get_additive_v_step_k(c, k));
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
        // Adam. Found via a real test failure: Adam's
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
                ScalePolicy::update(weights.additive_gamma[k],
                                    weights.get_additive_gamma_state_k(k), dgamma_k, learning_rate,
                                    beta2, eps, value_type(0),
                                    &weights.get_additive_gamma_step_k(k), false);
                if (l1_coef > value_type(0)) {
                    const value_type shrink = l1_coef * learning_rate;
                    value_type& gm = weights.additive_gamma[k];
                    if (gm > shrink)
                        gm -= shrink;
                    else if (gm < -shrink)
                        gm += shrink;
                    else
                        gm = value_type(0);
                }
            }
            value_type gamma_l1_sum = value_type(0);
            for (std::size_t k = 0; k < r_o; ++k)
                gamma_l1_sum += std::fabs(weights.additive_gamma[k]);
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
            const value_type grad_norm_divisor =
                static_cast<value_type>(n_in) * static_cast<value_type>(n_out);
            for (std::size_t k = 0; k < r_o; ++k) {
                const value_type abs_gamma_k = std::fabs(weights.additive_gamma[k]);
                const value_type share_k =
                    gamma_l1_sum > value_type(0) ? abs_gamma_k / gamma_l1_sum : value_type(0);
                weights.update_additive_gamma_ema_k(k, abs_gamma_k, share_k,
                                                    std::fabs(dgamma_by_k[k]) / grad_norm_divisor);
            }
        }
    }
}
