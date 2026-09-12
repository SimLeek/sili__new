#pragma once
// Benchmark-only escape hatch: force block4's disldo_backward onto its
// pre-SIMD scalar path (identical math, no Block4Vec) to measure the SIMD
// rewrite's real speedup against a same-commit, same-everything-else
// baseline -- e.g. `-DSILI_BLOCK4_FORCE_SCALAR_BACKWARD=1`. Defaults off;
// not a runtime knob, not meant to ship enabled.
#ifndef SILI_BLOCK4_FORCE_SCALAR_BACKWARD
#define SILI_BLOCK4_FORCE_SCALAR_BACKWARD 0
#endif
#include "block4_codec.hpp"
#include "cpu_topology.hpp"
#include "csr.hpp"
#include "sparse_struct.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// disldo_backward.block4_extract_function_refactor: the per-row block4 weight
// +importance update, extracted from what used to be an inline block inside
// disldo_backward's nested process_tile lambda (see block4_codec.hpp's own
// comment on Block4BackwardParams/Accumulators for the full GPU-portability
// rationale behind this shape). Handles decode, the AQRS gradient-accumulation
// terms, the SIMD (full-tile-column) vs scalar (boundary-column) update paths,
// and the write-back -- everything Block4Codec<VALUES_TYPE> didn't already
// collapse across precisions in the first refactor pass (PR #51). `br` is
// accepted for signature consistency with the other three extracted block4
// backward functions (all take the same row-block/tile coordinate prefix)
// even though this particular function doesn't need it.
template <typename SIZE_TYPE, typename VALUES_TYPE, typename COL_TYPE, typename ScalePolicy,
          bool DeferredScaleWrite, bool StochasticRounding,
          template <typename> class SynapsePolicyT>
void block4_backward_process_single_row(
    const Block4BackwardParams<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& params,
    Block4BackwardAccumulators<typename ValueAccessor<VALUES_TYPE>::value_type>& block4_accum,
    std::size_t /*br*/, uint32_t bc, uint32_t li, std::size_t row, uint32_t nnz_row, uint8_t* tdata,
    bool& tile_dirty) {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    using SynapsePolicy = SynapsePolicyT<value_type>;
    using SynapsePolicyVec = SynapsePolicyT<Block4Vec>;
    auto& weights = params.weights;
    const std::size_t rank = params.rank;
    const value_type learning_rate = params.learning_rate;
    const bool lr_per_row_nnz = params.lr_per_row_nnz;
    const std::size_t n_out = params.n_out;
    const value_type* input_T = params.input_T;
    const value_type* output_grad_T = params.output_grad_T;
    const auto batch = params.batch;
    const auto in_cols = params.in_cols; // still needed for mdx (dx accum), unchanged layout
    const value_type beta2 = params.beta2;
    const value_type eps = params.eps;
    const value_type min_decay_frac = params.min_decay_frac;
    const value_type max_ci = params.max_ci;
    const value_type max_abs_delta = params.max_abs_delta;
    const bool damp_by_importance = params.damp_by_importance;
    const bool scale_invariant = params.scale_invariant;
    const value_type zero_escape_eps = params.zero_escape_eps;
    const value_type* gamma_k_arr = params.gamma_k_arr;
    const int tid = block4_accum.tid;
    // disldo_backward.batch_stride_transpose: input_T is [n_in, batch]
    // (contiguous per row across batch); output_grad_T is [n_col_tiles,
    // batch, BLOCK4_TILE] (contiguous per output-tile across batch, still
    // 4-wide-SIMD-loadable per sample) -- see Block4BackwardParams's
    // comment (block4_codec.hpp) and disldo_backward's construction site.
    const value_type* input_row_T = input_T + row * static_cast<std::size_t>(batch);
    const value_type* dy_tile_T =
        output_grad_T + static_cast<std::size_t>(bc) * batch * BLOCK4_TILE;

    const value_type val_scale = weights.get_value_scale(row);
    const value_type imp_scale = weights.get_importance_scale(row);
    const value_type effective_lr =
        lr_per_row_nnz ? learning_rate / static_cast<value_type>(nnz_row) : learning_rate;

    // Decode this row's whole 4-wide column vector ONCE,
    // outside the batch loop (avoids blocking
    // auto-vectorization; the batch loop itself can't
    // vectorize across b, but the 4 columns are
    // independent, giving the inner lj loop a real
    // 4-wide target). FP4_TABLE[code] lookup, not
    // block4_vec_decode_fp4's SIMD bit-shift formula --
    // measured ~6% win for backward specifically
    // (opposite of forward's finding). See
    // disldo_backward.fp4_table_decode in
    // docs/research/linear_disldo.rst.
    // disldo_backward.block4_codec_refactor: single
    // generic implementation replacing what was 3
    // near-identical fp32/fp8/fp4 copies (~1500 lines),
    // differing only in decode/encode/quant_floor calls
    // -- now routed through Block4Codec<VALUES_TYPE>. See
    // block4_codec.hpp and
    // docs/research/linear_disldo.rst:block4_codec_refactor.
    using Codec = Block4Codec<VALUES_TYPE>;
    value_type w_decoded_arr[BLOCK4_TILE], imp_decoded_arr[BLOCK4_TILE];
    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
        w_decoded_arr[lj] = Codec::decode_weight(tdata, li, lj);
        imp_decoded_arr[lj] = Codec::decode_importance(tdata, li, lj);
    }

    // value_scale_k(row,k), fetched once per row -- matches
    // the scattered path's own once-per-row granularity
    // (see disldo_backward's non-DeferredScaleWrite branch).
    value_type* value_scale_k =
        weights.scale_rank_scratch.value_scale_k.data() + static_cast<std::size_t>(tid) * rank;
    for (std::size_t k = 0; k < rank; ++k)
        value_scale_k[k] = weights.get_value_scale_k(row, k);

    std::size_t col4[BLOCK4_TILE];
    bool col_valid4[BLOCK4_TILE];
    value_type combined_scale4[BLOCK4_TILE], combined_imp_scale4[BLOCK4_TILE];
    // quant4: the stored CODE-space quantity (decoded
    // weight, used as-is -- for fp4/fp8 this is a point on
    // their fixed grid, for fp32 it's just the raw float).
    // quant_orig4: immutable snapshot of its call-entry
    // value, for contrib.
    value_type quant4[BLOCK4_TILE], ci4[BLOCK4_TILE], quant_orig4[BLOCK4_TILE];
    const Flat2DView<value_type> out_scale_k4{weights.scale_rank_scratch.out_scale_k.data() +
                                                  static_cast<std::size_t>(tid) * rank *
                                                      BLOCK4_TILE,
                                              BLOCK4_TILE};
    // was_live4[lj]: TRUE only if this cell held a genuine
    // synapse BEFORE this call -- gates the never-zero
    // live quantizer so it never permanently "births" a
    // never-connected cell. Codec::encode ignores this for
    // fp32 (no quantization floor to escape there). See
    // disldo_backward.was_live_gating in
    // docs/research/linear_disldo.rst.
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
        was_live4[lj] =
            (w_decoded_arr[lj] != value_type(0)) || (imp_decoded_arr[lj] != value_type(0));
        const value_type out_imp_scale = weights.get_output_importance_scale(col);
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

    // block4_accum.mcol_at(col,k) is a real 4-way (times rank) SCATTER
    // if written every (b, lj) -- accumulate into small
    // local arrays across the whole batch loop instead
    // (pure register/stack traffic), flush once after.
    // Reused scratch memory -- explicit zero each tile
    // visit (task #295).
    const std::size_t tid_rank = static_cast<std::size_t>(tid) * rank;
    const std::size_t tid_rank_tile = tid_rank * BLOCK4_TILE;
    auto& srs = weights.scale_rank_scratch;
    const Flat2DView<value_type> mcol4_rank{srs.mcol_rank.data() + tid_rank_tile, BLOCK4_TILE};
    double* mrow_local_k = srs.mrow_local_k.data() + tid_rank;
    const Flat2DView<value_type> mcol4_rank_contrib{srs.mcol_rank_contrib.data() + tid_rank_tile,
                                                    BLOCK4_TILE};
    double* mrow_local_k_contrib = srs.mrow_local_k_contrib.data() + tid_rank;
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
    // -- see disldo_backward.fp4_table_decode's identical
    // finding. Only valid when the whole tile-column is in
    // bounds (true for every tile except possibly the
    // last, boundary one).
    const std::size_t col_base = std::size_t(bc) * BLOCK4_TILE;
    const bool full_tile_cols = (col_base + BLOCK4_TILE <= n_out);

    // Scalar per-lj fallback: used both for the (rare)
    // boundary tile-column and for any hypothetical
    // non-float value_type instantiation (Block4Vec is
    // float-only by design; nothing in this codebase ever
    // instantiates value_type != float, but the function
    // must still compile generically). Identical math to
    // the SIMD path below, just scalar per lane.
    auto scalar_row_update = [&]() {
        value_type quant_start4[BLOCK4_TILE], quant_floor4[BLOCK4_TILE];
        double g_agg4[BLOCK4_TILE] = {0.0}, contrib_agg4[BLOCK4_TILE] = {0.0};
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            quant_start4[lj] = quant4[lj];
            quant_floor4[lj] = Codec::quant_floor(quant4[lj], zero_escape_eps);
        }
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type iv = input_row_T[b];
            value_type* mdx_row = block4_accum.mdx + row * static_cast<std::size_t>(batch) + b;
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                if (!col_valid4[lj])
                    continue;
                const value_type dyv = dy_tile_T[static_cast<std::size_t>(b) * BLOCK4_TILE + lj];
                const value_type g = dyv * iv;
                const value_type S = combined_scale4[lj];
                if (learning_rate != value_type(0)) {
                    // Additive g+contrib combination,
                    // FIXED batch-start snapshots -- see
                    // the scattered path's cw_start
                    // comment for the full rationale.
                    const value_type contrib = iv * (quant_start4[lj] * S);
                    g_agg4[lj] += static_cast<double>(g);
                    contrib_agg4[lj] += static_cast<double>(contrib);
                    for (std::size_t k = 0; k < rank; ++k) {
                        mrow_local_k[k] += static_cast<double>(quant_floor4[lj]) *
                                           static_cast<double>(out_scale_k4[k][lj]) *
                                           static_cast<double>(gamma_k_arr[k]) * g;
                        mcol4_rank[k][lj] +=
                            quant_floor4[lj] * value_scale_k[k] * gamma_k_arr[k] * g;
                        mgamma_local_k[k] += static_cast<double>(quant_floor4[lj]) *
                                             static_cast<double>(out_scale_k4[k][lj]) *
                                             static_cast<double>(value_scale_k[k]) * g;
                        mrow_local_k_contrib[k] += static_cast<double>(quant_floor4[lj]) *
                                                   static_cast<double>(out_scale_k4[k][lj]) *
                                                   static_cast<double>(gamma_k_arr[k]) * contrib;
                        mcol4_rank_contrib[k][lj] +=
                            quant_floor4[lj] * value_scale_k[k] * gamma_k_arr[k] * contrib;
                        mgamma_local_k_contrib[k] += static_cast<double>(quant_floor4[lj]) *
                                                     static_cast<double>(out_scale_k4[k][lj]) *
                                                     static_cast<double>(value_scale_k[k]) *
                                                     contrib;
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
                const value_type contrib_agg = static_cast<value_type>(contrib_agg4[lj]);
                ci4[lj] = SynapsePolicy::update_ci(ci4[lj], g_agg, contrib_agg, beta2,
                                                   min_decay_frac, max_ci);
                quant4[lj] =
                    quant_start4[lj] + SynapsePolicy::update_cw(g_agg, ci4[lj], S, effective_lr,
                                                                eps, damp_by_importance,
                                                                max_abs_delta, scale_invariant);
            }
        }
    };

    if constexpr (std::is_same_v<value_type, float> && !SILI_BLOCK4_FORCE_SCALAR_BACKWARD) {
        if (full_tile_cols) {
            const Block4Vec effective_lr_v = block4_vec_broadcast(effective_lr);
            const Block4Vec beta2_v = block4_vec_broadcast(beta2);
            const Block4Vec eps_v = block4_vec_broadcast(eps);
            const Block4Vec min_decay_frac_v = block4_vec_broadcast(min_decay_frac);
            const Block4Vec max_ci_v = block4_vec_broadcast(max_ci);
            const Block4Vec max_abs_delta_v = block4_vec_broadcast(max_abs_delta);
            const Block4Vec combined_scale_v = block4_vec_load(combined_scale4);
            // quant_start_v: FIXED for the whole batch loop
            // -- see the scattered path's identical
            // cw_start comment. quant_floor_v computed once
            // from this fixed snapshot too (identity for
            // fp8/fp32, epsilon-substituted for fp4 -- see
            // Block4Codec::quant_floor).
            const Block4Vec quant_start_v = block4_vec_load(quant4);
            Block4Vec ci_v = block4_vec_load(ci4);
            Block4Vec quant_floor_v;
            for (int lane = 0; lane < BLOCK4_TILE; ++lane)
                quant_floor_v[lane] = Codec::quant_floor(quant_start_v[lane], zero_escape_eps);
            Block4Vec g_agg_v = block4_vec_broadcast(0.0f);
            // mcol_acc_raw/_contrib are the only TRUE
            // cross-batch accumulators here; backed by
            // scratch (task #295).
            value_type* mcol_acc_raw = srs.mcol_acc_raw.data() + tid_rank_tile;
            value_type* mcol_acc_raw_contrib = srs.mcol_acc_raw_contrib.data() + tid_rank_tile;
            for (std::size_t k = 0; k < rank; ++k) {
                block4_vec_store(mcol_acc_raw + k * BLOCK4_TILE, block4_vec_broadcast(0.0f));
                block4_vec_store(mcol_acc_raw_contrib + k * BLOCK4_TILE,
                                 block4_vec_broadcast(0.0f));
            }
            const bool training = (learning_rate != value_type(0));
            // disldo_backward.contrib_batch_invariant: contrib = (quant_start*S)*iv
            // is a CONSTANT (quant_start*S) times iv, so sum_b(contrib) ==
            // (quant_start*S)*sum_b(iv) -- tracking a running scalar iv sum here
            // and multiplying once after the loop replaces a per-sample Block4Vec
            // multiply+add with a per-sample scalar add. See
            // docs/research/linear_disldo.rst.
            double input_sum = 0.0;
            for (SIZE_TYPE b = 0; b < batch; ++b) {
                const value_type iv = input_row_T[b];
                input_sum += static_cast<double>(iv);
                value_type* mdx_row = block4_accum.mdx + row * static_cast<std::size_t>(batch) + b;
                const Block4Vec dyv_v =
                    block4_vec_load(dy_tile_T + static_cast<std::size_t>(b) * BLOCK4_TILE);
                if (training)
                    g_agg_v += dyv_v * block4_vec_broadcast(iv);
                // w = quant*S, FIXED for the whole batch --
                // see quant_start_v's own comment above.
                const Block4Vec w_v = quant_start_v * combined_scale_v;
                *mdx_row += block4_vec_hsum(w_v * dyv_v);
            }
            const Block4Vec contrib_agg_v =
                (quant_start_v * combined_scale_v) *
                block4_vec_broadcast(static_cast<value_type>(input_sum));
            // disldo_backward.batch_hsum_deferral: mrow_local_k/mgamma_local_k
            // (and contrib variants) are LINEAR in g_v/contrib_v, and every
            // other factor (quant_floor_v, out_scale_k_v, value_scale_k[k],
            // gamma_k_arr[k]) is batch-invariant -- so
            // sum_b(hsum(C*g_v_b)) == hsum(C*sum_b(g_v_b)) == hsum(C*g_agg_v).
            // Computing this ONCE here (using the g_agg_v/contrib_agg_v this
            // loop already accumulates) instead of once per batch sample
            // eliminates up to 4*rank horizontal-SIMD-reduction calls per
            // sample -- see disldo_backward.batch_stride_transpose in
            // docs/research/linear_disldo.rst for the profiling that found
            // this (transpose/merge fixes measured negligible; this per-
            // sample hsum work was ~99% of the real cost at batch=256).
            if (training) {
                for (std::size_t k = 0; k < rank; ++k) {
                    const Block4Vec value_scale_k_v = block4_vec_broadcast(value_scale_k[k]);
                    const Block4Vec out_scale_k_v = block4_vec_load(out_scale_k4[k]);
                    mrow_local_k[k] += static_cast<double>(block4_vec_hsum(
                                           quant_floor_v * out_scale_k_v * g_agg_v)) *
                                       static_cast<double>(gamma_k_arr[k]);
                    mgamma_local_k[k] += static_cast<double>(
                        block4_vec_hsum(quant_floor_v * out_scale_k_v * value_scale_k_v * g_agg_v));
                    mrow_local_k_contrib[k] += static_cast<double>(block4_vec_hsum(
                                                   quant_floor_v * out_scale_k_v * contrib_agg_v)) *
                                               static_cast<double>(gamma_k_arr[k]);
                    mgamma_local_k_contrib[k] += static_cast<double>(block4_vec_hsum(
                        quant_floor_v * out_scale_k_v * value_scale_k_v * contrib_agg_v));
                    value_type* acc = mcol_acc_raw + k * BLOCK4_TILE;
                    block4_vec_store(acc, block4_vec_load(acc) +
                                              quant_floor_v * value_scale_k_v * g_agg_v *
                                                  block4_vec_broadcast(gamma_k_arr[k]));
                    value_type* acc_c = mcol_acc_raw_contrib + k * BLOCK4_TILE;
                    block4_vec_store(acc_c, block4_vec_load(acc_c) +
                                                quant_floor_v * value_scale_k_v * contrib_agg_v *
                                                    block4_vec_broadcast(gamma_k_arr[k]));
                }
            }
            // ONE update, using the batch-aggregated g/contrib.
            Block4Vec quant_v = quant_start_v;
            if (training) {
                ci_v = SynapsePolicyVec::update_ci(ci_v, g_agg_v, contrib_agg_v, beta2_v,
                                                   min_decay_frac_v, max_ci_v);
                // dL/d(quant) = g*S -- proper chain rule on
                // true_w=quant*S (multiply, not divide --
                // see disldo_backward's scattered-path
                // comment this mirrors exactly).
                const Block4Vec delta_v = SynapsePolicyVec::update_cw(
                    g_agg_v, ci_v, combined_scale_v, effective_lr_v, eps_v, damp_by_importance,
                    max_abs_delta_v, scale_invariant);
                quant_v += delta_v;
            }
            block4_vec_store(quant4, quant_v);
            block4_vec_store(ci4, ci_v);
            for (std::size_t k = 0; k < rank; ++k) {
                block4_vec_store(mcol4_rank[k], block4_vec_load(mcol_acc_raw + k * BLOCK4_TILE));
                block4_vec_store(mcol4_rank_contrib[k],
                                 block4_vec_load(mcol_acc_raw_contrib + k * BLOCK4_TILE));
            }
        } else {
            scalar_row_update();
        }
    } else {
        scalar_row_update();
    }

    // Write-back: was_live4[lj] gate + StochasticRounding
    // both handled inside Codec::encode (see block4_codec.hpp)
    // -- collapses what used to be a 4-way
    // StochasticRounding x was_live dispatch, tripled per
    // precision, into one call.
    if (learning_rate != value_type(0)) {
        for (std::size_t k = 0; k < rank; ++k) {
            block4_accum.mrow_at(row, k) += mrow_local_k[k];
            block4_accum.mrow_at_contrib(row, k) += mrow_local_k_contrib[k];
            block4_accum.mgamma_at(k) += static_cast<value_type>(mgamma_local_k[k]);
            block4_accum.mgamma_at_contrib(k) += static_cast<value_type>(mgamma_local_k_contrib[k]);
        }
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            if (!col_valid4[lj])
                continue;
            for (std::size_t k = 0; k < rank; ++k) {
                block4_accum.mcol_at(col4[lj], k) += mcol4_rank[k][lj];
                block4_accum.mcol_at_contrib(col4[lj], k) += mcol4_rank_contrib[k][lj];
            }
            const value_type imp_ratio = ci4[lj] / combined_imp_scale4[lj];
            Codec::template encode<StochasticRounding>(tdata, li, lj, quant4[lj], imp_ratio,
                                                       was_live4[lj]);
            tile_dirty = true;
        }
    }
}

// disldo_backward.block4_extract_function_refactor: AVX2 row-pairing (combines
// rows li0/li0+1 of the SAME tile into one 8-wide op instead of two 4-wide
// ones) -- extracted from what used to be a lambda nested inside
// process_tile/block4_backward_process_single_tile. Only called when the
// whole tile-column is in bounds (checked by the caller before invoking this
// -- see disldo_backward.fp32_block4_avx2_row_pairing in
// docs/research/linear_disldo.rst for the full rationale, and
// block4_codec.hpp's Block4BackwardParams/Accumulators comment for why this
// takes a parameter-object struct instead of capturing by reference).
template <typename SIZE_TYPE, typename VALUES_TYPE, typename COL_TYPE, typename ScalePolicy,
          bool DeferredScaleWrite, bool StochasticRounding,
          template <typename> class SynapsePolicyT>
void block4_backward_process_row_pair(
    const Block4BackwardParams<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& params,
    Block4BackwardAccumulators<typename ValueAccessor<VALUES_TYPE>::value_type>& block4_accum,
    std::size_t /*br*/, uint32_t bc, uint32_t li0, std::size_t row0, uint32_t nnz_row0,
    std::size_t row1, uint32_t nnz_row1, uint8_t* tdata, bool& tile_dirty) {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    using SynapsePolicyVec = SynapsePolicyT<Block4Vec>;
    auto& weights = params.weights;
    const std::size_t rank = params.rank;
    const value_type learning_rate = params.learning_rate;
    const bool lr_per_row_nnz = params.lr_per_row_nnz;
    const std::size_t n_out = params.n_out;
    const value_type* input_T = params.input_T;
    const value_type* output_grad_T = params.output_grad_T;
    const auto batch = params.batch;
    const auto in_cols = params.in_cols; // still needed for mdx (dx accum), unchanged layout
    const value_type beta2 = params.beta2;
    const value_type eps = params.eps;
    const value_type min_decay_frac = params.min_decay_frac;
    const value_type max_ci = params.max_ci;
    const value_type max_abs_delta = params.max_abs_delta;
    const bool damp_by_importance = params.damp_by_importance;
    const bool scale_invariant = params.scale_invariant;
    const value_type zero_escape_eps = params.zero_escape_eps;
    const value_type* gamma_k_arr = params.gamma_k_arr;
    const int tid = block4_accum.tid;
    // disldo_backward.batch_stride_transpose: see block4_backward_process_
    // single_row's identical comment.
    const value_type* input_row0_T = input_T + row0 * static_cast<std::size_t>(batch);
    const value_type* input_row1_T = input_T + row1 * static_cast<std::size_t>(batch);
    const value_type* dy_tile_T =
        output_grad_T + static_cast<std::size_t>(bc) * batch * BLOCK4_TILE;

    using Codec = Block4Codec<VALUES_TYPE>;
    const uint32_t li1 = li0 + 1;
    const value_type imp_scale0 = weights.get_importance_scale(row0);
    const value_type effective_lr0 =
        lr_per_row_nnz ? learning_rate / static_cast<value_type>(nnz_row0) : learning_rate;
    const value_type imp_scale1 = weights.get_importance_scale(row1);
    const value_type effective_lr1 =
        lr_per_row_nnz ? learning_rate / static_cast<value_type>(nnz_row1) : learning_rate;

    // Decode both rows' 4-wide column vectors. No single
    // contiguous load across rows -- fixed-li,
    // varying-lj is a stride-4 gather for every
    // precision's slot_index -- same total loads as two
    // separate single-row passes; the win is the 8-wide
    // ARITHMETIC that follows.
    value_type w_decoded_arr8[2 * BLOCK4_TILE], imp_decoded_arr8[2 * BLOCK4_TILE];
    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
        w_decoded_arr8[lj] = Codec::decode_weight(tdata, li0, lj);
        w_decoded_arr8[lj + BLOCK4_TILE] = Codec::decode_weight(tdata, li1, lj);
        imp_decoded_arr8[lj] = Codec::decode_importance(tdata, li0, lj);
        imp_decoded_arr8[lj + BLOCK4_TILE] = Codec::decode_importance(tdata, li1, lj);
    }

    // Columns are shared between both rows (col only
    // depends on bc/lj, not li) -- computed once,
    // 4-wide, reused for both halves below.
    std::size_t col4[BLOCK4_TILE];
    value_type out_imp_scale4[BLOCK4_TILE];
    const std::size_t tid_rank = static_cast<std::size_t>(tid) * rank;
    const std::size_t tid_rank_tile = tid_rank * BLOCK4_TILE;
    auto& srs = weights.scale_rank_scratch;
    const Flat2DView<value_type> out_scale_k4{srs.out_scale_k.data() + tid_rank_tile, BLOCK4_TILE};
    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
        col4[lj] = std::size_t(bc) * BLOCK4_TILE + lj;
        out_imp_scale4[lj] = weights.get_output_importance_scale(col4[lj]);
        for (std::size_t k = 0; k < rank; ++k)
            out_scale_k4[k][lj] = weights.get_output_scale_k(col4[lj], k);
    }

    std::vector<value_type> value_scale_k_row0(rank), value_scale_k_row1(rank);
    for (std::size_t k = 0; k < rank; ++k) {
        value_scale_k_row0[k] = weights.get_value_scale_k(row0, k);
        value_scale_k_row1[k] = weights.get_value_scale_k(row1, k);
    }

    // was_live8[idx]: TRUE only if this cell held a
    // genuine synapse BEFORE this call -- see was_live4's
    // declaration comment (single-row branch below) for
    // the full rationale. Codec::encode ignores this for
    // fp32 (no quantization floor to escape there).
    bool was_live8[2 * BLOCK4_TILE];
    value_type combined_scale8[2 * BLOCK4_TILE], combined_imp_scale8[2 * BLOCK4_TILE];
    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
        was_live8[lj] =
            (w_decoded_arr8[lj] != value_type(0)) || (imp_decoded_arr8[lj] != value_type(0));
        was_live8[lj + BLOCK4_TILE] = (w_decoded_arr8[lj + BLOCK4_TILE] != value_type(0)) ||
                                      (imp_decoded_arr8[lj + BLOCK4_TILE] != value_type(0));
        combined_scale8[lj] = weights.get_scale(row0, col4[lj]);
        combined_scale8[lj + BLOCK4_TILE] = weights.get_scale(row1, col4[lj]);
        combined_imp_scale8[lj] = imp_scale0 * out_imp_scale4[lj];
        combined_imp_scale8[lj + BLOCK4_TILE] = imp_scale1 * out_imp_scale4[lj];
    }
    // quant8: the stored CODE-space quantity (decoded
    // weight). quant_floor8: quant8 with fp4's zero-
    // escape substitution -- see Codec::quant_floor,
    // identity for fp8/fp32 (matches what the original
    // fp32/fp8 lambdas called cw_orig).
    value_type quant8[2 * BLOCK4_TILE], ci8[2 * BLOCK4_TILE], quant_floor8[2 * BLOCK4_TILE];
    for (uint32_t i = 0; i < 2 * BLOCK4_TILE; ++i) {
        quant8[i] = w_decoded_arr8[i];
        ci8[i] = imp_decoded_arr8[i] * combined_imp_scale8[i];
        quant_floor8[i] = Codec::quant_floor(quant8[i], zero_escape_eps);
    }

    const Block8Vec quant_start_v = block8_vec_load(quant8);
    Block8Vec ci_v = block8_vec_load(ci8);
    const Block8Vec quant_floor_v = block8_vec_load(quant_floor8);
    const Block8Vec S_v = block8_vec_load(combined_scale8);

    value_type* mcol_acc = srs.mcol_acc_raw.data() + tid_rank_tile;
    value_type* mcol_acc_contrib = srs.mcol_acc_raw_contrib.data() + tid_rank_tile;
    for (std::size_t k = 0; k < rank; ++k) {
        block4_vec_store(mcol_acc + k * BLOCK4_TILE, block4_vec_broadcast(0.0f));
        block4_vec_store(mcol_acc_contrib + k * BLOCK4_TILE, block4_vec_broadcast(0.0f));
    }
    std::vector<double> mrow_local0_k(rank, 0.0), mrow_local1_k(rank, 0.0),
        mrow_local0_k_contrib(rank, 0.0), mrow_local1_k_contrib(rank, 0.0);
    double* mgamma_local_k = srs.mgamma_local_k.data() + tid_rank;
    double* mgamma_local_k_contrib = srs.mgamma_local_k_contrib.data() + tid_rank;
    std::fill(mgamma_local_k, mgamma_local_k + rank, 0.0);
    std::fill(mgamma_local_k_contrib, mgamma_local_k_contrib + rank, 0.0);

    Block8Vec g_agg_v = block8_vec_broadcast(0.0f);
    const bool training = (learning_rate != value_type(0));
    // disldo_backward.contrib_batch_invariant: see block4_backward_process_
    // single_row's identical comment -- contrib = (quant_start*S)*iv is a
    // CONSTANT times iv per row-half, so summing it over the batch collapses
    // to a constant times sum_b(iv). Track running row0/row1 input sums
    // instead of a per-sample Block8Vec multiply+add.
    double iv0_sum = 0.0, iv1_sum = 0.0;
    for (SIZE_TYPE b = 0; b < batch; ++b) {
        const value_type iv0 = input_row0_T[b];
        const value_type iv1 = input_row1_T[b];
        iv0_sum += static_cast<double>(iv0);
        iv1_sum += static_cast<double>(iv1);
        value_type* mdx_row0 = block4_accum.mdx + row0 * static_cast<std::size_t>(batch) + b;
        value_type* mdx_row1 = block4_accum.mdx + row1 * static_cast<std::size_t>(batch) + b;
        const Block4Vec dyv4 =
            block4_vec_load(dy_tile_T + static_cast<std::size_t>(b) * BLOCK4_TILE);
        const Block8Vec dyv_v = block8_vec_dup4(dyv4);
        if (training)
            g_agg_v += dyv_v * block8_vec_broadcast_pair(iv0, iv1);
        const Block8Vec mdx_term = quant_start_v * S_v * dyv_v;
        *mdx_row0 += block4_vec_hsum(block8_vec_lo4(mdx_term));
        *mdx_row1 += block4_vec_hsum(block8_vec_hi4(mdx_term));
    }
    const Block8Vec contrib_agg_v = quant_start_v * S_v *
                                    block8_vec_broadcast_pair(static_cast<value_type>(iv0_sum),
                                                              static_cast<value_type>(iv1_sum));

    // disldo_backward.batch_hsum_deferral: see block4_backward_process_
    // single_row's identical comment -- every factor here besides g_agg_v/
    // contrib_agg_v is batch-invariant, so this collapses what was up to
    // 6*rank horizontal-SIMD-reductions per batch sample into 6*rank total.
    if (training) {
        for (std::size_t k = 0; k < rank; ++k) {
            const Block8Vec value_scale_k_v =
                block8_vec_broadcast_pair(value_scale_k_row0[k], value_scale_k_row1[k]);
            const Block8Vec out_scale_k_v = block8_vec_dup4(block4_vec_load(out_scale_k4[k]));
            const Block8Vec prod_g = quant_floor_v * out_scale_k_v * g_agg_v;
            mrow_local0_k[k] += static_cast<double>(block4_vec_hsum(block8_vec_lo4(prod_g))) *
                                static_cast<double>(gamma_k_arr[k]);
            mrow_local1_k[k] += static_cast<double>(block4_vec_hsum(block8_vec_hi4(prod_g))) *
                                static_cast<double>(gamma_k_arr[k]);
            mgamma_local_k[k] +=
                static_cast<double>(block8_vec_hsum(out_scale_k_v * value_scale_k_v * prod_g));
            const Block8Vec prod_contrib = quant_floor_v * out_scale_k_v * contrib_agg_v;
            mrow_local0_k_contrib[k] +=
                static_cast<double>(block4_vec_hsum(block8_vec_lo4(prod_contrib))) *
                static_cast<double>(gamma_k_arr[k]);
            mrow_local1_k_contrib[k] +=
                static_cast<double>(block4_vec_hsum(block8_vec_hi4(prod_contrib))) *
                static_cast<double>(gamma_k_arr[k]);
            mgamma_local_k_contrib[k] += static_cast<double>(
                block8_vec_hsum(out_scale_k_v * value_scale_k_v * prod_contrib));
            value_type* acc = mcol_acc + k * BLOCK4_TILE;
            const Block8Vec term =
                quant_floor_v * value_scale_k_v * g_agg_v * block8_vec_broadcast(gamma_k_arr[k]);
            block4_vec_store(acc, block4_vec_load(acc) + block8_vec_fold4(term));
            value_type* acc_c = mcol_acc_contrib + k * BLOCK4_TILE;
            const Block8Vec term_c = quant_floor_v * value_scale_k_v * contrib_agg_v *
                                     block8_vec_broadcast(gamma_k_arr[k]);
            block4_vec_store(acc_c, block4_vec_load(acc_c) + block8_vec_fold4(term_c));
        }
    }

    Block8Vec quant_v = quant_start_v;
    if (training) {
        // ci/cw update is genuinely per-(row,col) cell
        // (no shared/duplicated lanes to exploit here),
        // and no Block8Vec specialization of
        // SynapsePolicy exists -- split into the two
        // EXISTING Block4Vec-specialized calls. This is
        // a once-per-row-pair cost, not a per-batch
        // one, so it doesn't undo the batch loop's win.
        const Block4Vec g_agg_v0 = block8_vec_lo4(g_agg_v);
        const Block4Vec g_agg_v1 = block8_vec_hi4(g_agg_v);
        const Block4Vec contrib_agg_v0 = block8_vec_lo4(contrib_agg_v);
        const Block4Vec contrib_agg_v1 = block8_vec_hi4(contrib_agg_v);
        Block4Vec ci_v0 = block8_vec_lo4(ci_v);
        Block4Vec ci_v1 = block8_vec_hi4(ci_v);
        const Block4Vec S_v0 = block8_vec_lo4(S_v);
        const Block4Vec S_v1 = block8_vec_hi4(S_v);
        const Block4Vec effective_lr_v0 = block4_vec_broadcast(effective_lr0);
        const Block4Vec effective_lr_v1 = block4_vec_broadcast(effective_lr1);
        const Block4Vec beta2_v4 = block4_vec_broadcast(beta2);
        const Block4Vec eps_v4 = block4_vec_broadcast(eps);
        const Block4Vec min_decay_frac_v4 = block4_vec_broadcast(min_decay_frac);
        const Block4Vec max_ci_v4 = block4_vec_broadcast(max_ci);
        const Block4Vec max_abs_delta_v4 = block4_vec_broadcast(max_abs_delta);
        ci_v0 = SynapsePolicyVec::update_ci(ci_v0, g_agg_v0, contrib_agg_v0, beta2_v4,
                                            min_decay_frac_v4, max_ci_v4);
        ci_v1 = SynapsePolicyVec::update_ci(ci_v1, g_agg_v1, contrib_agg_v1, beta2_v4,
                                            min_decay_frac_v4, max_ci_v4);
        const Block4Vec delta_v0 =
            SynapsePolicyVec::update_cw(g_agg_v0, ci_v0, S_v0, effective_lr_v0, eps_v4,
                                        damp_by_importance, max_abs_delta_v4, scale_invariant);
        const Block4Vec delta_v1 =
            SynapsePolicyVec::update_cw(g_agg_v1, ci_v1, S_v1, effective_lr_v1, eps_v4,
                                        damp_by_importance, max_abs_delta_v4, scale_invariant);
        quant_v = block8_vec_from_lo_hi(block8_vec_lo4(quant_start_v) + delta_v0,
                                        block8_vec_hi4(quant_start_v) + delta_v1);
        ci_v = block8_vec_from_lo_hi(ci_v0, ci_v1);
        block8_vec_store(quant8, quant_v);
        block8_vec_store(ci8, ci_v);

        for (std::size_t k = 0; k < rank; ++k) {
            block4_accum.mrow_at(row0, k) += mrow_local0_k[k];
            block4_accum.mrow_at(row1, k) += mrow_local1_k[k];
            block4_accum.mrow_at_contrib(row0, k) += mrow_local0_k_contrib[k];
            block4_accum.mrow_at_contrib(row1, k) += mrow_local1_k_contrib[k];
            block4_accum.mgamma_at(k) += static_cast<value_type>(mgamma_local_k[k]);
            block4_accum.mgamma_at_contrib(k) += static_cast<value_type>(mgamma_local_k_contrib[k]);
        }
        // Write-back: was_live8[idx] gate +
        // StochasticRounding both handled inside
        // Codec::encode -- collapses what used to be a
        // 4-way dispatch (or plain memcpy for fp32),
        // tripled per precision, into one call per half.
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            for (std::size_t k = 0; k < rank; ++k) {
                block4_accum.mcol_at(col4[lj], k) += (mcol_acc + k * BLOCK4_TILE)[lj];
                block4_accum.mcol_at_contrib(col4[lj], k) +=
                    (mcol_acc_contrib + k * BLOCK4_TILE)[lj];
            }
            for (int half = 0; half < 2; ++half) {
                const uint32_t li_h = half == 0 ? li0 : li1;
                const uint32_t idx = half == 0 ? lj : lj + BLOCK4_TILE;
                const value_type imp_ratio = ci8[idx] / combined_imp_scale8[idx];
                Codec::template encode<StochasticRounding>(tdata, li_h, lj, quant8[idx], imp_ratio,
                                                           was_live8[idx]);
            }
        }
        tile_dirty = true;
    }
}

// disldo_backward.block4_extract_function_refactor: per-tile orchestration --
// was the process_tile lambda, now a real free function. Loops over this
// tile's 4 rows, dispatching each either to block4_backward_process_row_pair
// (two adjacent live rows, full tile-column) or
// block4_backward_process_single_row. Shared between the read-only and
// writing call sites in disldo_backward, so they can't drift apart; every
// write inside the two functions above is gated by learning_rate != 0.
template <typename SIZE_TYPE, typename VALUES_TYPE, typename COL_TYPE, typename ScalePolicy,
          bool DeferredScaleWrite, bool StochasticRounding,
          template <typename> class SynapsePolicyT>
bool block4_backward_process_single_tile(
    const Block4BackwardParams<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& block4_params,
    Block4BackwardAccumulators<typename ValueAccessor<VALUES_TYPE>::value_type>& block4_accum,
    std::size_t br, uint32_t bc, uint8_t* tdata) {
    bool tile_dirty = false;
    const std::size_t n_in = block4_params.n_in;
    const std::size_t n_out = block4_params.n_out;
    const uint32_t* row_live_count = block4_params.row_live_count;

    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
        const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
        if (row >= n_in)
            continue;
        const uint32_t nnz_row = row_live_count[row];
        if (nnz_row == 0)
            continue;

        if ((li % 2) == 0) {
            const std::size_t row1 = row + 1;
            const std::size_t col_base_pair = std::size_t(bc) * BLOCK4_TILE;
            if (row1 < n_in && (col_base_pair + BLOCK4_TILE <= n_out)) {
                const uint32_t nnz_row1 = row_live_count[row1];
                if (nnz_row1 > 0) {
                    block4_backward_process_row_pair<SIZE_TYPE, VALUES_TYPE, COL_TYPE, ScalePolicy,
                                                     DeferredScaleWrite, StochasticRounding,
                                                     SynapsePolicyT>(block4_params, block4_accum,
                                                                     br, bc, li, row, nnz_row, row1,
                                                                     nnz_row1, tdata, tile_dirty);
                    ++li; // consumes li+1 too -- for's own ++li then makes
                          // the net advance +2, safely skipping li+1
                    continue;
                }
            }
        }

        block4_backward_process_single_row<SIZE_TYPE, VALUES_TYPE, COL_TYPE, ScalePolicy,
                                           DeferredScaleWrite, StochasticRounding, SynapsePolicyT>(
            block4_params, block4_accum, br, bc, li, row, nnz_row, tdata, tile_dirty);
    } // closes for (li...)
    return tile_dirty;
}

// disldo_backward.block4_extract_function_refactor: cross-tile pairing --
// was the process_tile_pair lambda, now a real free function. Pairs TWO
// DIFFERENT tiles (bcA, bcB) sharing this br into one op per row li, the
// mirror image of block4_backward_process_row_pair (which pairs two ROWS
// of the SAME tile). Row-level quantities (value_scale_k, effective_lr)
// are shared between both halves; column-level quantities (out_scale_k,
// combined_scale) are per-half, since A and B are different columns;
// mdx/mrow/mgamma fold across both halves into the same row, mcol stays
// per-half. FP32-only, rank==1 -- see the call site's own comment on why
// that gate is kept. Returns a plain struct (not std::pair) since GLSL,
// this kernel's eventual Kompute/Vulkan target, has no std::pair
// analogue.
template <typename SIZE_TYPE, typename VALUES_TYPE, typename COL_TYPE, typename ScalePolicy,
          bool DeferredScaleWrite, bool StochasticRounding,
          template <typename> class SynapsePolicyT>
Block4TileDirtyPair block4_backward_process_tile_pair(
    const Block4BackwardParams<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& params,
    Block4BackwardAccumulators<typename ValueAccessor<VALUES_TYPE>::value_type>& block4_accum,
    std::size_t br, uint32_t bcA, uint32_t bcB, uint8_t* tdataA, uint8_t* tdataB) {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    using SynapsePolicyVec = SynapsePolicyT<Block4Vec>;
    auto& weights = params.weights;
    const std::size_t n_in = params.n_in;
    const std::size_t n_out = params.n_out;
    const std::size_t rank = params.rank;
    const value_type learning_rate = params.learning_rate;
    const bool lr_per_row_nnz = params.lr_per_row_nnz;
    const value_type* input_T = params.input_T;
    const value_type* output_grad_T = params.output_grad_T;
    const auto batch = params.batch;
    const auto in_cols = params.in_cols; // still needed for mdx (dx accum), unchanged layout
    const value_type beta2 = params.beta2;
    const value_type eps = params.eps;
    const value_type min_decay_frac = params.min_decay_frac;
    const value_type max_ci = params.max_ci;
    const value_type max_abs_delta = params.max_abs_delta;
    const bool damp_by_importance = params.damp_by_importance;
    const bool scale_invariant = params.scale_invariant;
    const value_type zero_escape_eps = params.zero_escape_eps;
    const value_type* gamma_k_arr = params.gamma_k_arr;
    const uint32_t* row_live_count = params.row_live_count;

    using Codec = Block4Codec<VALUES_TYPE>;
    bool dirtyA = false, dirtyB = false;
    const std::size_t col_baseA = std::size_t(bcA) * BLOCK4_TILE;
    const std::size_t col_baseB = std::size_t(bcB) * BLOCK4_TILE;
    const bool training = (learning_rate != value_type(0));
    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
        const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
        if (row >= n_in)
            continue;
        const uint32_t nnz_row = row_live_count[row];
        if (nnz_row == 0)
            continue;
        const value_type imp_scale = weights.get_importance_scale(row);
        const value_type effective_lr =
            lr_per_row_nnz ? learning_rate / static_cast<value_type>(nnz_row) : learning_rate;
        // Shared across both halves (same row).
        std::vector<value_type> value_scale_k_row(rank);
        for (std::size_t k = 0; k < rank; ++k)
            value_scale_k_row[k] = weights.get_value_scale_k(row, k);

        std::size_t colA[BLOCK4_TILE], colB[BLOCK4_TILE];
        value_type out_imp_scaleA[BLOCK4_TILE], out_imp_scaleB[BLOCK4_TILE];
        // Per-half, per-k -- A and B are DIFFERENT columns.
        std::vector<value_type> out_scale_kA(rank * BLOCK4_TILE), out_scale_kB(rank * BLOCK4_TILE);
        value_type combined_scale8[2 * BLOCK4_TILE];
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            colA[lj] = col_baseA + lj;
            colB[lj] = col_baseB + lj;
            const bool haveA = colA[lj] < n_out;
            const bool haveB = colB[lj] < n_out;
            out_imp_scaleA[lj] =
                haveA ? weights.get_output_importance_scale(colA[lj]) : value_type(0);
            out_imp_scaleB[lj] =
                haveB ? weights.get_output_importance_scale(colB[lj]) : value_type(0);
            for (std::size_t k = 0; k < rank; ++k) {
                out_scale_kA[k * BLOCK4_TILE + lj] =
                    haveA ? weights.get_output_scale_k(colA[lj], k) : value_type(0);
                out_scale_kB[k * BLOCK4_TILE + lj] =
                    haveB ? weights.get_output_scale_k(colB[lj], k) : value_type(0);
            }
            combined_scale8[lj] = haveA ? weights.get_scale(row, colA[lj]) : value_type(0);
            combined_scale8[lj + BLOCK4_TILE] =
                haveB ? weights.get_scale(row, colB[lj]) : value_type(0);
        }

        value_type w_decoded8[2 * BLOCK4_TILE], imp_decoded8[2 * BLOCK4_TILE];
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            w_decoded8[lj] = Codec::decode_weight(tdataA, li, lj);
            w_decoded8[lj + BLOCK4_TILE] = Codec::decode_weight(tdataB, li, lj);
            imp_decoded8[lj] = Codec::decode_importance(tdataA, li, lj);
            imp_decoded8[lj + BLOCK4_TILE] = Codec::decode_importance(tdataB, li, lj);
        }
        // was_live8[idx]: see block4_backward_process_row_pair's identical
        // declaration comment -- ignored by Codec::encode for fp32.
        bool was_live8[2 * BLOCK4_TILE];
        for (uint32_t i = 0; i < 2 * BLOCK4_TILE; ++i)
            was_live8[i] = (w_decoded8[i] != value_type(0)) || (imp_decoded8[i] != value_type(0));

        value_type combined_imp_scale8[2 * BLOCK4_TILE];
        // quant8: CODE-space decoded weight. quant_floor8:
        // quant8 with fp4's zero-escape substitution -- see
        // Codec::quant_floor, identity for fp8/fp32.
        value_type quant8[2 * BLOCK4_TILE], ci8[2 * BLOCK4_TILE], quant_floor8[2 * BLOCK4_TILE];
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            combined_imp_scale8[lj] = imp_scale * out_imp_scaleA[lj];
            combined_imp_scale8[lj + BLOCK4_TILE] = imp_scale * out_imp_scaleB[lj];
        }
        for (uint32_t i = 0; i < 2 * BLOCK4_TILE; ++i) {
            quant8[i] = w_decoded8[i];
            ci8[i] = imp_decoded8[i] * combined_imp_scale8[i];
            quant_floor8[i] = Codec::quant_floor(quant8[i], zero_escape_eps);
        }

        double g_agg8[2 * BLOCK4_TILE] = {0};
        // Single accumulator per k (not split A/B) -- both
        // halves feed the SAME row's mrow/mgamma gradient.
        std::vector<double> mrow_local_k(rank, 0.0), mrow_local_k_contrib(rank, 0.0);
        std::vector<double> mgamma_local_k(rank, 0.0), mgamma_local_k_contrib(rank, 0.0);
        std::vector<value_type> mcol_local8(rank * 2 * BLOCK4_TILE, value_type(0)),
            mcol_local_contrib8(rank * 2 * BLOCK4_TILE, value_type(0));
        // disldo_backward.batch_stride_transpose: see block4_backward_process_
        // single_row's identical comment. bcA/bcB tile-blocks are already
        // zero-padded past n_out, so no extra bounds check is needed here.
        const value_type* input_row_T = input_T + row * static_cast<std::size_t>(batch);
        const value_type* dy_tileA_T =
            output_grad_T + static_cast<std::size_t>(bcA) * batch * BLOCK4_TILE;
        const value_type* dy_tileB_T =
            output_grad_T + static_cast<std::size_t>(bcB) * batch * BLOCK4_TILE;
        // disldo_backward.contrib_batch_invariant: see block4_backward_process_
        // single_row's identical comment -- contrib[i] = quant8[i]*
        // combined_scale8[i]*iv is a CONSTANT (per lane i) times iv, so
        // sum_b(contrib[i]) == quant8[i]*combined_scale8[i]*sum_b(iv). Track a
        // single running input_sum instead of 8 separate per-lane accumulators.
        double input_sum = 0.0;
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            const value_type iv = input_row_T[b];
            input_sum += static_cast<double>(iv);
            value_type dyv8[2 * BLOCK4_TILE];
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                dyv8[lj] = dy_tileA_T[static_cast<std::size_t>(b) * BLOCK4_TILE + lj];
                dyv8[lj + BLOCK4_TILE] = dy_tileB_T[static_cast<std::size_t>(b) * BLOCK4_TILE + lj];
            }
            value_type mdx_term = value_type(0);
            for (uint32_t i = 0; i < 2 * BLOCK4_TILE; ++i) {
                mdx_term += quant8[i] * combined_scale8[i] * dyv8[i];
                if (!training)
                    continue;
                g_agg8[i] += dyv8[i] * iv;
            }
            block4_accum.mdx[row * static_cast<std::size_t>(batch) + b] += mdx_term;
        }
        double contrib_agg8[2 * BLOCK4_TILE] = {0};
        if (training)
            for (uint32_t i = 0; i < 2 * BLOCK4_TILE; ++i)
                contrib_agg8[i] = static_cast<double>(quant8[i]) *
                                  static_cast<double>(combined_scale8[i]) * input_sum;

        // disldo_backward.batch_hsum_deferral: out_scale_k_i/value_scale_k_row/
        // gamma_k_arr are all batch-invariant here, so this k-loop (rank * 8
        // work) is deferred to run ONCE per tile against g_agg8/contrib_agg8
        // instead of once per batch sample -- same algebraic identity as
        // single_row/row_pair's deferral, just without an hsum in the middle.
        if (training) {
            for (uint32_t i = 0; i < 2 * BLOCK4_TILE; ++i) {
                for (std::size_t k = 0; k < rank; ++k) {
                    const value_type out_scale_k_i =
                        i < BLOCK4_TILE ? out_scale_kA[k * BLOCK4_TILE + i]
                                        : out_scale_kB[k * BLOCK4_TILE + (i - BLOCK4_TILE)];
                    const value_type prod_g = quant_floor8[i] * out_scale_k_i * g_agg8[i];
                    const value_type prod_contrib =
                        quant_floor8[i] * out_scale_k_i * contrib_agg8[i];
                    mrow_local_k[k] += static_cast<double>(prod_g) * gamma_k_arr[k];
                    mrow_local_k_contrib[k] += static_cast<double>(prod_contrib) * gamma_k_arr[k];
                    mgamma_local_k[k] +=
                        static_cast<double>(out_scale_k_i * value_scale_k_row[k] * prod_g);
                    mgamma_local_k_contrib[k] +=
                        static_cast<double>(out_scale_k_i * value_scale_k_row[k] * prod_contrib);
                    mcol_local8[k * 2 * BLOCK4_TILE + i] +=
                        quant_floor8[i] * value_scale_k_row[k] * g_agg8[i] * gamma_k_arr[k];
                    mcol_local_contrib8[k * 2 * BLOCK4_TILE + i] +=
                        quant_floor8[i] * value_scale_k_row[k] * contrib_agg8[i] * gamma_k_arr[k];
                }
            }
        }

        if (training) {
            for (std::size_t k = 0; k < rank; ++k) {
                block4_accum.mrow_at(row, k) += mrow_local_k[k];
                block4_accum.mrow_at_contrib(row, k) += mrow_local_k_contrib[k];
                block4_accum.mgamma_at(k) += static_cast<value_type>(mgamma_local_k[k]);
                block4_accum.mgamma_at_contrib(k) +=
                    static_cast<value_type>(mgamma_local_k_contrib[k]);
            }
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                for (std::size_t k = 0; k < rank; ++k) {
                    if (colA[lj] < n_out) {
                        block4_accum.mcol_at(colA[lj], k) += mcol_local8[k * 2 * BLOCK4_TILE + lj];
                        block4_accum.mcol_at_contrib(colA[lj], k) +=
                            mcol_local_contrib8[k * 2 * BLOCK4_TILE + lj];
                    }
                    if (colB[lj] < n_out) {
                        block4_accum.mcol_at(colB[lj], k) +=
                            mcol_local8[k * 2 * BLOCK4_TILE + lj + BLOCK4_TILE];
                        block4_accum.mcol_at_contrib(colB[lj], k) +=
                            mcol_local_contrib8[k * 2 * BLOCK4_TILE + lj + BLOCK4_TILE];
                    }
                }
            }
            // No Block8Vec specialization of SynapsePolicy
            // exists (mirrors block4_backward_process_row_pair's own
            // rationale) -- split into two 4-wide
            // Block4Vec-specialized calls.
            Block4Vec g_agg_v0, g_agg_v1, contrib_agg_v0, contrib_agg_v1, ci_v0, ci_v1, S_v0, S_v1;
            for (uint32_t i = 0; i < BLOCK4_TILE; ++i) {
                g_agg_v0[i] = static_cast<value_type>(g_agg8[i]);
                g_agg_v1[i] = static_cast<value_type>(g_agg8[i + BLOCK4_TILE]);
                contrib_agg_v0[i] = static_cast<value_type>(contrib_agg8[i]);
                contrib_agg_v1[i] = static_cast<value_type>(contrib_agg8[i + BLOCK4_TILE]);
                ci_v0[i] = ci8[i];
                ci_v1[i] = ci8[i + BLOCK4_TILE];
                S_v0[i] = combined_scale8[i];
                S_v1[i] = combined_scale8[i + BLOCK4_TILE];
            }
            const Block4Vec effective_lr_v = block4_vec_broadcast(effective_lr);
            const Block4Vec beta2_v4 = block4_vec_broadcast(beta2);
            const Block4Vec eps_v4 = block4_vec_broadcast(eps);
            const Block4Vec min_decay_frac_v4 = block4_vec_broadcast(min_decay_frac);
            const Block4Vec max_ci_v4 = block4_vec_broadcast(max_ci);
            const Block4Vec max_abs_delta_v4 = block4_vec_broadcast(max_abs_delta);
            ci_v0 = SynapsePolicyVec::update_ci(ci_v0, g_agg_v0, contrib_agg_v0, beta2_v4,
                                                min_decay_frac_v4, max_ci_v4);
            ci_v1 = SynapsePolicyVec::update_ci(ci_v1, g_agg_v1, contrib_agg_v1, beta2_v4,
                                                min_decay_frac_v4, max_ci_v4);
            const Block4Vec delta_v0 =
                SynapsePolicyVec::update_cw(g_agg_v0, ci_v0, S_v0, effective_lr_v, eps_v4,
                                            damp_by_importance, max_abs_delta_v4, scale_invariant);
            const Block4Vec delta_v1 =
                SynapsePolicyVec::update_cw(g_agg_v1, ci_v1, S_v1, effective_lr_v, eps_v4,
                                            damp_by_importance, max_abs_delta_v4, scale_invariant);
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                quant8[lj] += delta_v0[lj];
                quant8[lj + BLOCK4_TILE] += delta_v1[lj];
                ci8[lj] = ci_v0[lj];
                ci8[lj + BLOCK4_TILE] = ci_v1[lj];
            }
            // Write-back: was_live8[idx] gate + StochasticRounding
            // both handled inside Codec::encode.
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                if (colA[lj] < n_out) {
                    const value_type imp_ratio = ci8[lj] / combined_imp_scale8[lj];
                    Codec::template encode<StochasticRounding>(tdataA, li, lj, quant8[lj],
                                                               imp_ratio, was_live8[lj]);
                    dirtyA = true;
                }
                if (colB[lj] < n_out) {
                    const uint32_t idx = lj + BLOCK4_TILE;
                    const value_type imp_ratio = ci8[idx] / combined_imp_scale8[idx];
                    Codec::template encode<StochasticRounding>(tdataB, li, lj, quant8[idx],
                                                               imp_ratio, was_live8[idx]);
                    dirtyB = true;
                }
            }
        }
    } // closes for (li...)
    return {dirtyA, dirtyB};
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
 *        ci is still tracked either way. See
 *        disldo_backward.rmsprop_floor_rationale in docs/research/linear_disldo.rst
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
 * args -- see disldo_backward.rmsprop_floor_rationale in
 * docs/research/linear_disldo.rst for the full rationale and why
 * SynapsePolicyT specifically must be template-template.
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
    // this whole call. See disldo_backward.setup_and_presizing_bugs in
    // docs/research/linear_disldo.rst.
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
    // in that case. See disldo_backward.setup_and_presizing_bugs in
    // docs/research/linear_disldo.rst.
    const std::size_t dst = static_cast<std::size_t>(batch) * in_cols;

    // output_scale's own gradient (symmetric to value_scale's), shared
    // between the scattered loop and block4's own loop, per-thread-private
    // slices reduced once at the end. Only applied if output_scale_is_trainable.
    const std::size_t rank = weights.scale_rank;
    // Persistent per-instance heap scratch (task #295) backing block4's
    // SIMD backward path's per-rank-component accumulators.
    weights.scale_rank_scratch.ensure(static_cast<std::size_t>(num_cpus), rank, BLOCK4_TILE);
    // AQRS gamma (task #273/#283): layer-wide, fetched once rather than
    // per-row. See disldo_backward.gamma_fetched_once in
    // docs/research/linear_disldo.rst for why it's not baked into the
    // direction caches.
    std::vector<value_type> gamma_k_arr(rank);
    for (std::size_t k = 0; k < rank; ++k)
        gamma_k_arr[k] = weights.get_scale_gamma_k(k);
    const bool output_scale_trainable = weights.output_scale_is_trainable;

    // t_dx/t_col_grad/t_col_grad_contrib/t_gamma_grad/t_gamma_grad_contrib:
    // persistent per-instance heap scratch (this session's fix, mirrors
    // ScaleRankScratch's task #295 pattern above), replacing a fresh
    // num_cpus*(...)-sized heap allocation + full zero-fill on EVERY
    // disldo_backward() call. Measured as a real cost: at width=288
    // (82944-element layer), backward at num_cpus=8 measured slower than
    // num_cpus=1 -- these buffers scale with num_cpus, so more threads
    // meant more to allocate+zero serially before the parallel region
    // even opened. Addressing below ALWAYS uses the stored cap_* stride
    // (t_dx_stride/t_out_rank_stride/t_rank_stride), never this call's
    // own (possibly smaller) dst/n_out*rank/rank, so a later call with
    // smaller dimensions than a historical max can never alias two
    // threads' slices together.
    weights.disldo_backward_scratch.ensure(static_cast<std::size_t>(num_cpus), dst, n_out * rank,
                                           rank);
    const std::size_t t_dx_stride = weights.disldo_backward_scratch.cap_dst;
    const std::size_t t_out_rank_stride = weights.disldo_backward_scratch.cap_out_rank;
    const std::size_t t_rank_stride = weights.disldo_backward_scratch.cap_rank;
    std::vector<value_type>& t_dx = weights.disldo_backward_scratch.t_dx;
    std::vector<value_type>& t_dx_T = weights.disldo_backward_scratch.t_dx_T;
    std::vector<value_type>& t_col_grad = weights.disldo_backward_scratch.t_col_grad;
    std::vector<value_type>& t_col_grad_contrib =
        weights.disldo_backward_scratch.t_col_grad_contrib;
    std::vector<value_type>& t_gamma_grad = weights.disldo_backward_scratch.t_gamma_grad;
    std::vector<value_type>& t_gamma_grad_contrib =
        weights.disldo_backward_scratch.t_gamma_grad_contrib;

    // Zero exactly the range THIS call will use, in its own dedicated
    // parallel region (one thread per slice, so the zero-fill itself is
    // parallelized instead of one thread serially zeroing everything
    // before the real work starts). Deliberately its OWN region, separate
    // from the scattered loop's and block4's parallel regions below: both
    // of those ACCUMULATE (+=) into these same buffers and run
    // sequentially one after the other, so zeroing inside either of them
    // would wipe out whichever one ran first.
    //
    // group_of_tid[tid]: this call's cache-domain ("CCX") group per
    // thread, feeding the group-aware t_dx/t_col_grad reduction far
    // below. See disldo_backward.ccx_aware_reduction in
    // docs/research/linear_disldo.rst.
    std::vector<int> group_of_tid(static_cast<std::size_t>(num_cpus), 0);
#pragma omp parallel num_threads(num_cpus)
    {
        const int tid = omp_get_thread_num();
        group_of_tid[static_cast<std::size_t>(tid)] = sili_topology::current_thread_group();
        std::fill_n(t_dx.data() + static_cast<std::size_t>(tid) * t_dx_stride, dst, value_type(0));
        std::fill_n(t_dx_T.data() + static_cast<std::size_t>(tid) * t_dx_stride, dst,
                    value_type(0));
        std::fill_n(t_col_grad.data() + static_cast<std::size_t>(tid) * t_out_rank_stride,
                    n_out * rank, value_type(0));
        std::fill_n(t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * t_out_rank_stride,
                    n_out * rank, value_type(0));
        std::fill_n(t_gamma_grad.data() + static_cast<std::size_t>(tid) * t_rank_stride, rank,
                    value_type(0));
        std::fill_n(t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * t_rank_stride,
                    rank, value_type(0));
    }
    // Distinct groups actually present among this call's num_cpus
    // threads, and each group's leader (lowest tid) -- computed once,
    // serially (num_cpus is small), reused by the group-aware reduction
    // after both accumulate regions (scattered + block4) below.
    std::vector<int> group_ids;
    std::vector<int> group_leader_tid;
    std::vector<int> tid_to_g(static_cast<std::size_t>(num_cpus), 0);
    for (int t = 0; t < num_cpus; ++t) {
        const int gid = group_of_tid[static_cast<std::size_t>(t)];
        const auto it = std::find(group_ids.begin(), group_ids.end(), gid);
        if (it == group_ids.end()) {
            tid_to_g[static_cast<std::size_t>(t)] = static_cast<int>(group_ids.size());
            group_ids.push_back(gid);
            group_leader_tid.push_back(t);
        } else {
            tid_to_g[static_cast<std::size_t>(t)] = static_cast<int>(it - group_ids.begin());
        }
    }
    const std::size_t num_groups = group_ids.size();

    // Pre-size value_scale/output_scale (n_in*rank / n_out*rank) before the
    // parallel region so direct indexed writes are safe.
    //
    // A uniform resize(..., value_type(1)) fill would backfill EVERY
    // appended slot with 1.0, not just k==0, forcing any grown rank
    // component into permanent lockstep with k==0 instead of real extra
    // capacity -- see disldo_backward.setup_and_presizing_bugs in
    // docs/research/linear_disldo.rst. Resize with 0
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
    // See disldo_backward.setup_and_presizing_bugs in
    // docs/research/linear_disldo.rst.
    if (weights.value_scale_step.size() < n_in * rank)
        weights.value_scale_step.resize(n_in * rank, 0);

    // Per-thread importance-stats staging (replaces a #pragma omp critical
    // call to update_importance_stats_aggregate() inside the parallel
    // region below): that function's own THREAD SAFETY comment already
    // documents the intended pattern -- accumulate locally, call ONCE per
    // thread AFTER the parallel region, not from inside one under a lock.
    // Sized num_cpus regardless of dc.empty() so the post-region reduction
    // loop below is unconditional and branch-free either way.
    std::vector<double> t_stats_sum_abs_new(static_cast<std::size_t>(num_cpus), 0.0);
    std::vector<double> t_stats_sum_abs_old(static_cast<std::size_t>(num_cpus), 0.0);
    std::vector<double> t_stats_sum_sq_new(static_cast<std::size_t>(num_cpus), 0.0);
    std::vector<double> t_stats_sum_sq_old(static_cast<std::size_t>(num_cpus), 0.0);
    std::vector<value_type> t_stats_max_new(static_cast<std::size_t>(num_cpus), value_type(0));

    if (!dc.empty()) {
#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mdx = t_dx.data() + static_cast<std::size_t>(tid) * t_dx_stride;
            // mcol[col*rank+k]. False positive on all m*_base pointers
            // below: cppcheck can't trace mutation through the m*_at
            // lambdas' returned references.
            // cppcheck-suppress constVariablePointer
            value_type* mcol_base =
                t_col_grad.data() + static_cast<std::size_t>(tid) * t_out_rank_stride;
            auto mcol_at = [&](std::size_t col, std::size_t k) -> value_type& {
                return mcol_base[col * rank + k];
            };
            // cppcheck-suppress constVariablePointer
            value_type* mcol_contrib_base =
                t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * t_out_rank_stride;
            auto mcol_at_contrib = [&](std::size_t col, std::size_t k) -> value_type& {
                return mcol_contrib_base[col * rank + k];
            };
            // cppcheck-suppress constVariablePointer
            value_type* mgamma_base =
                t_gamma_grad.data() + static_cast<std::size_t>(tid) * t_rank_stride;
            auto mgamma_at = [&](std::size_t k) -> value_type& { return mgamma_base[k]; };
            // cppcheck-suppress constVariablePointer
            value_type* mgamma_contrib_base =
                t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * t_rank_stride;
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
                // disldo_backward.scattered_lr_and_signal in
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
                        // see disldo_backward.scattered_lr_and_signal in
                        // docs/research/linear_disldo.rst for the real
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
                                // -- see disldo_backward.scattered_lr_and_signal in
                                // docs/research/linear_disldo.rst
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
                            // disldo_backward.deferred_vs_direct_quant in
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
                        // components. See disldo_backward.deferred_vs_direct_quant
                        // in docs/research/linear_disldo.rst.
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
                                // docstring and disldo_backward.deferred_vs_direct_quant
                                // in docs/research/linear_disldo.rst.
                                const value_type contrib = iv * cw_start;
                                g_agg += static_cast<double>(g);
                                contrib_agg += static_cast<double>(contrib);
                                // dL/d(value_scale_k(r,k)) = g*quant*
                                // output_scale_k(col,k), vanishing exactly at
                                // quant=0. quant_floor gates the zero-escape
                                // substitution on quant==0 specifically (not
                                // unconditionally) -- see
                                // disldo_backward.deferred_vs_direct_quant in
                                // docs/research/linear_disldo.rst for the real
                                // sign-corruption bug this fixes.
                                for (std::size_t k = 0; k < rank; ++k) {
                                    // AQRS gamma (task #273/#283): direction
                                    // caches are never gamma-baked, so each
                                    // gradient site needs an explicit gamma_k
                                    // factor; gamma's own gradient uses the
                                    // pure direction product. See
                                    // disldo_backward.deferred_vs_direct_quant in
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
                t_stats_sum_abs_new[tid] = local_sum_abs_new_i;
                t_stats_sum_abs_old[tid] = local_sum_abs_old_i;
                t_stats_sum_sq_new[tid] = local_sum_sq_new_i;
                t_stats_sum_sq_old[tid] = local_sum_sq_old_i;
                t_stats_max_new[tid] = local_max_new_i;
            }
        }
        // Serial reduction, single-threaded (the parallel region above has
        // already closed) -- no lock needed, matching
        // update_importance_stats_aggregate()'s own documented contract.
        if (learning_rate != value_type(0)) {
            for (int t = 0; t < num_cpus; ++t) {
                weights.update_importance_stats_aggregate(
                    t_stats_sum_abs_new[static_cast<std::size_t>(t)],
                    t_stats_sum_abs_old[static_cast<std::size_t>(t)],
                    t_stats_sum_sq_new[static_cast<std::size_t>(t)],
                    t_stats_sum_sq_old[static_cast<std::size_t>(t)],
                    t_stats_max_new[static_cast<std::size_t>(t)]);
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
    // disldo_backward.block4_overview_races in docs/research/linear_disldo.rst.
    if (weights.block4.n_tiles() > 0) {
        // row_ti_start: cumulative tile count per block-row, needed to
        // split the row-partitioned parallel loop below. NOT storage
        // offsets -- since the row-workspace rewrite, tile byte/elem
        // positions live entirely within each row's own RowWorkspace,
        // snapshotted fresh per row. See disldo_backward.row_ti_start_workspace
        // in docs/research/linear_disldo.rst.
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

        // disldo_backward.batch_stride_transpose: input/output_grad are
        // [batch, features] (batch-major), but every row/tile visited below
        // needs its own scan across the batch dimension -- for a dense
        // layer that's O(rows*cols) separate batch-major scans, each
        // striding through memory at in_cols/n_out floats per step (one
        // cache line fetched, 4 bytes used). Pre-transposing ONCE here
        // (O(n_in*batch + n_out*batch), negligible next to the O(nnz*batch)
        // main loop) turns every one of those repeated scans contiguous.
        // output_grad_T is a BLOCK transpose (grouped by output tile, not a
        // flat transpose) so the existing 4-wide SIMD load stays contiguous
        // too -- see Block4BackwardParams's own comment (block4_codec.hpp).
        // TEMPORARY diagnostic timing (not for commit) -- gated behind an
        // env var so it's a no-op unless explicitly requested.
        const bool sili_timing_on = std::getenv("SILI_DISLDO_TIMING") != nullptr;
        const auto sili_t_transpose_start = std::chrono::steady_clock::now();
        std::vector<value_type>& input_T = weights.block4.scratch_input_T;
        input_T.assign(n_in * static_cast<std::size_t>(batch), value_type(0));
        for (std::size_t row = 0; row < n_in; ++row)
            for (SIZE_TYPE b = 0; b < batch; ++b)
                input_T[row * static_cast<std::size_t>(batch) + b] =
                    input[static_cast<std::size_t>(b) * in_cols + row];

        const std::size_t n_col_tiles = (n_out + BLOCK4_TILE - 1) / BLOCK4_TILE;
        std::vector<value_type>& output_grad_T = weights.block4.scratch_output_grad_T;
        output_grad_T.assign(n_col_tiles * static_cast<std::size_t>(batch) * BLOCK4_TILE,
                             value_type(0));
        for (std::size_t col = 0; col < n_out; ++col) {
            const std::size_t tile_idx = col / BLOCK4_TILE;
            const std::size_t lj = col % BLOCK4_TILE;
            for (SIZE_TYPE b = 0; b < batch; ++b)
                output_grad_T[(tile_idx * static_cast<std::size_t>(batch) + b) * BLOCK4_TILE + lj] =
                    output_grad[static_cast<std::size_t>(b) * n_out + col];
        }

        const auto sili_t_transpose_end = std::chrono::steady_clock::now();
        if (sili_timing_on)
            std::fprintf(stderr, "[disldo_backward] transpose_build: %.4f ms\n",
                         std::chrono::duration<double, std::milli>(sili_t_transpose_end -
                                                                   sili_t_transpose_start)
                             .count());

        // disldo_backward.block4_extract_function_refactor: broadcast/read-only
        // state for the whole block4 parallel region, constructed once here --
        // see Block4BackwardParams's own comment (block4_codec.hpp) for why this
        // is split from the per-thread accumulators below, and the GPU-
        // portability rationale for the split shape.
        Block4BackwardParams<SIZE_TYPE, VALUES_TYPE, COL_TYPE> block4_params{weights,
                                                                             input,
                                                                             output_grad,
                                                                             batch,
                                                                             in_cols,
                                                                             n_in,
                                                                             n_out,
                                                                             rank,
                                                                             learning_rate,
                                                                             beta2,
                                                                             eps,
                                                                             min_decay_frac,
                                                                             max_abs_delta,
                                                                             max_ci,
                                                                             zero_escape_eps,
                                                                             damp_by_importance,
                                                                             scale_invariant,
                                                                             lr_per_row_nnz,
                                                                             gamma_k_arr.data(),
                                                                             row_live_count.data(),
                                                                             input_T.data(),
                                                                             output_grad_T.data()};

        const auto sili_t_tileloop_start = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            // t_col_grad/t_row_grad laid out [thread][col or row][k]; both
            // FP4/FP8 block4 branches below are full rank-N.
            Block4BackwardAccumulators<value_type> block4_accum{
                tid, t_dx_T.data() + static_cast<std::size_t>(tid) * t_dx_stride,
                t_col_grad.data() + static_cast<std::size_t>(tid) * t_out_rank_stride,
                t_col_grad_contrib.data() + static_cast<std::size_t>(tid) * t_out_rank_stride,
                t_row_grad.data() + static_cast<std::size_t>(tid) * n_in * rank,
                t_row_grad_contrib.data() + static_cast<std::size_t>(tid) * n_in * rank,
                // AQRS gamma: block4 gets its OWN parallel region, so this
                // thread's mgamma_base/mgamma_contrib_base need their own
                // tid-scoped offset into the SAME t_gamma_grad/
                // t_gamma_grad_contrib buffers the scattered path uses --
                // both regions accumulate into the same shared array,
                // reduced together once after both close.
                t_gamma_grad.data() + static_cast<std::size_t>(tid) * t_rank_stride,
                t_gamma_grad_contrib.data() + static_cast<std::size_t>(tid) * t_rank_stride, rank};

// Partitioned BY BLOCK-ROW, not flat tile index: each thread
// exclusively owns every tile in its assigned rows, so a
// tile's handle can safely resize (real sparse<->dense
// transitions) inside this parallel region. schedule(static)
// measured to beat dynamic/guided here despite uneven row
// widths -- see disldo_backward.schedule_static_measured in
// docs/research/linear_disldo.rst.
#pragma omp for schedule(static)
            for (std::size_t br = 0; br < BL4.rows; ++br) {
                if (row_ti_start[br] == row_ti_start[br + 1])
                    continue; // empty row
                // Row-workspace snapshot fixes a real cross-row memmove
                // hazard (a different row's growth could memmove this
                // row's bytes mid-read) -- see
                // disldo_backward.row_workspace_snapshot_fix in
                // docs/research/linear_disldo.rst.
                // block4_backward_process_single_tile is shared between the
                // read-only and writing branches below so they can't drift
                // apart; every write inside is gated by learning_rate != 0.

                // block4_backward_process_tile_pair (Phase 7e extraction)
                // is shared between the read-only and writing branches below
                // so they can't drift apart; every write inside is gated by
                // learning_rate != 0. FP32-only, rank==1 -- see
                // disldo_forward.fp32_block4_cross_tile_pairing in
                // docs/research/linear_disldo.rst and
                // tests/unit/test_disldo_block4_fp32_crosstile_backward.cpp for
                // why that gate is kept (rank>1/AQRS layers fall back to the
                // existing per-tile block4_backward_process_single_tile path,
                // unchanged, below).

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
                        // under which block4_backward_process_single_tile
                        // provably never writes through tdata (every write is
                        // gated by the same condition inside it) -- not a
                        // general-purpose cast, just avoiding a second
                        // (const-parametrized) copy of that function for a
                        // write that can't happen here.
                        block4_backward_process_single_tile<SIZE_TYPE, VALUES_TYPE, COL_TYPE,
                                                            ScalePolicy, DeferredScaleWrite,
                                                            StochasticRounding, SynapsePolicyT>(
                            block4_params, block4_accum, br, bc,
                            const_cast<uint8_t*>(tile.raw_data()));
                        byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
                    }
                } else {
                    // Writing: row-local workspace -- see the comment above
                    // this whole block4 section (row_ti_start's comment) and
                    // Block4Store::RowWorkspace's own comment for why this is
                    // necessary under concurrency, not just an optimization.
                    auto ws = weights.block4.snapshot_row(br);
                    std::size_t live_byte_pos = 0;
                    using Codec = Block4Codec<VALUES_TYPE>;
                    // disldo_backward.block4_codec_refactor: process_tile_pair
                    // pairs two DIFFERENT tiles per iteration; everything
                    // else falls back to the original one-tile-per-iteration
                    // loop below. Sequencing within a pair matters: B's
                    // ORIGINAL byte position depends on A's ORIGINAL
                    // (pre-write) length (both tiles must be unpacked before
                    // either is committed, since a commit can resize a tile
                    // and shift every subsequent tile's bytes in the row's
                    // shared buffer -- see
                    // disldo_backward.row_workspace_snapshot_fix), so A is
                    // committed FIRST and its length is RE-READ afterward to
                    // find B's true CURRENT position before B is committed
                    // -- exactly mirroring how the single-tile loop below
                    // re-reads its own tile's post-commit length before
                    // advancing to the next tile, just applied twice per
                    // pair instead of once per tile.
                    //
                    // FP32 historically only paired at rank==1 (its
                    // process_tile_pair_fp32 predated AQRS rank-N support);
                    // process_tile_pair is now rank-N generic for every
                    // precision including fp32 (Codec::quant_floor/encode
                    // handle it uniformly), but the rank==1 gate is kept
                    // here deliberately -- fp32+rank>1 block4 backward is
                    // untested/unused in this codebase, so this refactor
                    // does not silently widen fp32's reachable code path,
                    // only removes the fp8/fp8/fp4 duplication around it.
                    bool row_handled_as_pairs;
                    if constexpr (std::is_same_v<VALUES_TYPE, DeltaCSRBiValues<float>>)
                        row_handled_as_pairs = (rank == 1);
                    else
                        row_handled_as_pairs = true;
                    if (row_handled_as_pairs) {
                        std::size_t row_ti = row_ti_start[br];
                        while (row_ti < row_ti_start[br + 1]) {
                            const std::size_t eA = row_ti - row_ti_start[br];
                            const uint32_t bcA = ws.bc[eA];
                            const std::size_t byte_posA = live_byte_pos;
                            const bool has_partner = (row_ti + 1 < row_ti_start[br + 1]);
                            if (has_partner) {
                                const std::size_t eB = eA + 1;
                                const uint32_t bcB = ws.bc[eB];
                                uint8_t scratch_bufA[Codec::scratch_bytes];
                                uint8_t scratch_bufB[Codec::scratch_bytes];
                                weights.block4.unpack_workspace_tile(ws, eA, byte_posA,
                                                                     scratch_bufA);
                                const std::size_t lenA_before =
                                    Codec::stored_tile_len(ws.is_sparse[eA], &ws.bytes[byte_posA]);
                                const std::size_t byte_posB_before = byte_posA + lenA_before;
                                weights.block4.unpack_workspace_tile(ws, eB, byte_posB_before,
                                                                     scratch_bufB);
                                const auto [dirtyA, dirtyB] = block4_backward_process_tile_pair<
                                    SIZE_TYPE, VALUES_TYPE, COL_TYPE, ScalePolicy,
                                    DeferredScaleWrite, StochasticRounding, SynapsePolicyT>(
                                    block4_params, block4_accum, br, bcA, bcB, scratch_bufA,
                                    scratch_bufB);
                                if (dirtyA)
                                    weights.block4.commit_dirty_tile_in_workspace(ws, eA, byte_posA,
                                                                                  scratch_bufA);
                                const std::size_t lenA_after =
                                    Codec::stored_tile_len(ws.is_sparse[eA], &ws.bytes[byte_posA]);
                                const std::size_t byte_posB_after = byte_posA + lenA_after;
                                if (dirtyB)
                                    weights.block4.commit_dirty_tile_in_workspace(
                                        ws, eB, byte_posB_after, scratch_bufB);
                                const std::size_t lenB_after = Codec::stored_tile_len(
                                    ws.is_sparse[eB], &ws.bytes[byte_posB_after]);
                                live_byte_pos = byte_posB_after + lenB_after;
                                row_ti += 2;
                            } else {
                                uint8_t scratch_buf[Codec::scratch_bytes];
                                weights.block4.unpack_workspace_tile(ws, eA, byte_posA,
                                                                     scratch_buf);
                                const bool tile_dirty = block4_backward_process_single_tile<
                                    SIZE_TYPE, VALUES_TYPE, COL_TYPE, ScalePolicy,
                                    DeferredScaleWrite, StochasticRounding, SynapsePolicyT>(
                                    block4_params, block4_accum, br, bcA, scratch_buf);
                                if (tile_dirty)
                                    weights.block4.commit_dirty_tile_in_workspace(ws, eA, byte_posA,
                                                                                  scratch_buf);
                                live_byte_pos +=
                                    Codec::stored_tile_len(ws.is_sparse[eA], &ws.bytes[byte_posA]);
                                ++row_ti;
                            }
                        } // closes while (row_ti...)
                    } else {
                        for (std::size_t row_ti = row_ti_start[br]; row_ti < row_ti_start[br + 1];
                             ++row_ti) {
                            const std::size_t e = row_ti - row_ti_start[br];
                            const uint32_t bc = ws.bc[e];
                            const std::size_t this_byte_pos = live_byte_pos;
                            uint8_t scratch_buf[Codec::scratch_bytes];
                            weights.block4.unpack_workspace_tile(ws, e, this_byte_pos, scratch_buf);
                            const bool tile_dirty = block4_backward_process_single_tile<
                                SIZE_TYPE, VALUES_TYPE, COL_TYPE, ScalePolicy, DeferredScaleWrite,
                                StochasticRounding, SynapsePolicyT>(block4_params, block4_accum, br,
                                                                    bc, scratch_buf);
                            if (tile_dirty)
                                weights.block4.commit_dirty_tile_in_workspace(ws, e, this_byte_pos,
                                                                              scratch_buf);
                            live_byte_pos +=
                                Codec::stored_tile_len(ws.is_sparse[e], &ws.bytes[this_byte_pos]);
                        } // closes for (row_ti...)
                    }
                    // Merge back -- evicts lowest-|true-importance| synapses
                    // only if this row genuinely grew past its own current
                    // headroom (see Block4Store::merge_row_workspace's
                    // comment).
                    weights.block4.merge_row_workspace(
                        br, ws,
                        // Generic lambda (auto param, not uint8_t): FP4/FP8's
                        // Block4Store(8)::merge_row_workspace pass an integer
                        // CODE to decode, but Block4Store32's passes the raw
                        // STORED FLOAT directly (float32 has no code to
                        // decode) -- a fixed uint8_t parameter here would
                        // silently narrow that float via an implicit
                        // conversion instead of a real decode. See
                        // Block4Store8::merge_row_workspace's own docstring
                        // (block4.hpp) for FP8's full-byte-not-nibble
                        // convention, and Block4Store32's for FP32's.
                        [&](std::size_t ev_row, std::size_t ev_col, auto ev_imp_raw) -> double {
                            const value_type imp_scale = weights.get_importance_scale(ev_row);
                            const value_type out_imp_scale =
                                weights.get_output_importance_scale(ev_col);
                            double decoded;
                            if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                                decoded = static_cast<double>(fp8_decode_bits(ev_imp_raw));
                            } else if constexpr (std::is_same_v<VALUES_TYPE,
                                                                DeltaCSRBiValues<float>>) {
                                decoded = static_cast<double>(ev_imp_raw);
                            } else {
                                decoded = static_cast<double>(FP4_TABLE[ev_imp_raw & 0xFu]);
                            }
                            return decoded * static_cast<double>(imp_scale) *
                                   static_cast<double>(out_imp_scale);
                        });
                }
            } // closes for (br...)
        } // closes #pragma omp parallel
        const auto sili_t_tileloop_end = std::chrono::steady_clock::now();
        if (sili_timing_on)
            std::fprintf(stderr, "[disldo_backward] tile_loop: %.4f ms\n",
                         std::chrono::duration<double, std::milli>(sili_t_tileloop_end -
                                                                   sili_t_tileloop_start)
                             .count());

        // disldo_backward.batch_stride_transpose: merge block4's transposed
        // dx accumulator (t_dx_T) into the shared batch-major t_dx once,
        // before the group-aware t_dx reduction further below -- see
        // Block4BackwardParams's own comment (block4_codec.hpp).
        const auto sili_t_merge_start = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* dst_dx = t_dx.data() + static_cast<std::size_t>(tid) * t_dx_stride;
            const value_type* src_dx_t =
                t_dx_T.data() + static_cast<std::size_t>(tid) * t_dx_stride;
            for (std::size_t row = 0; row < n_in; ++row) {
                const value_type* row_t = src_dx_t + row * static_cast<std::size_t>(batch);
                for (SIZE_TYPE b = 0; b < batch; ++b)
                    dst_dx[static_cast<std::size_t>(b) * in_cols + row] += row_t[b];
            }
        }
        if (sili_timing_on) {
            const auto sili_t_merge_end = std::chrono::steady_clock::now();
            std::fprintf(
                stderr, "[disldo_backward] mdx_merge: %.4f ms\n",
                std::chrono::duration<double, std::milli>(sili_t_merge_end - sili_t_merge_start)
                    .count());
        }

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

    // Group-aware (CCX) reduction of t_dx/t_col_grad/t_col_grad_contrib,
    // reusing group_of_tid captured above. See
    // disldo_backward.ccx_aware_reduction in
    // docs/research/linear_disldo.rst for the full design rationale and
    // measured before/after numbers -- this REQUIRES OMP_PROC_BIND/
    // OMP_PLACES pinning (set in sili/__init__.py) to be a net win rather
    // than a regression. Gamma's own reduction further down is left flat
    // (rank is small enough that the group-aware version isn't worth the
    // added complexity), and t_row_grad's reduction above is deliberately
    // out of scope (see its own comment).
    std::vector<value_type> group_dx(num_groups * dst, value_type(0));
    std::vector<value_type> group_col_grad(num_groups * n_out * rank, value_type(0));
    std::vector<value_type> group_col_grad_contrib(num_groups * n_out * rank, value_type(0));
    if (num_groups <= 1) {
        value_type* gdx = group_dx.data();
        value_type* gcol = group_col_grad.data();
        value_type* gcol_c = group_col_grad_contrib.data();
        for (int t = 0; t < num_cpus; ++t) {
            const value_type* sdx = t_dx.data() + static_cast<std::size_t>(t) * t_dx_stride;
            for (std::size_t i = 0; i < dst; ++i)
                gdx[i] += sdx[i];
            const value_type* scol =
                t_col_grad.data() + static_cast<std::size_t>(t) * t_out_rank_stride;
            const value_type* scol_c =
                t_col_grad_contrib.data() + static_cast<std::size_t>(t) * t_out_rank_stride;
            for (std::size_t i = 0; i < n_out * rank; ++i) {
                gcol[i] += scol[i];
                gcol_c[i] += scol_c[i];
            }
        }
    } else {
#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            const int g = tid_to_g[static_cast<std::size_t>(tid)];
            if (tid == group_leader_tid[static_cast<std::size_t>(g)]) {
                value_type* gdx = group_dx.data() + static_cast<std::size_t>(g) * dst;
                value_type* gcol =
                    group_col_grad.data() + static_cast<std::size_t>(g) * n_out * rank;
                value_type* gcol_c =
                    group_col_grad_contrib.data() + static_cast<std::size_t>(g) * n_out * rank;
                for (int t = 0; t < num_cpus; ++t) {
                    if (tid_to_g[static_cast<std::size_t>(t)] != g)
                        continue;
                    const value_type* sdx = t_dx.data() + static_cast<std::size_t>(t) * t_dx_stride;
                    for (std::size_t i = 0; i < dst; ++i)
                        gdx[i] += sdx[i];
                    const value_type* scol =
                        t_col_grad.data() + static_cast<std::size_t>(t) * t_out_rank_stride;
                    const value_type* scol_c =
                        t_col_grad_contrib.data() + static_cast<std::size_t>(t) * t_out_rank_stride;
                    for (std::size_t i = 0; i < n_out * rank; ++i) {
                        gcol[i] += scol[i];
                        gcol_c[i] += scol_c[i];
                    }
                }
            }
        }
    }

    for (std::size_t g = 0; g < num_groups; ++g) {
        const value_type* s = group_dx.data() + g * dst;
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
                // Reads group_col_grad/group_col_grad_contrib (already
                // fully reduced across every thread within its group,
                // above) rather than t_col_grad directly -- num_groups is
                // typically 1-2, not num_cpus.
                for (std::size_t g = 0; g < num_groups; ++g) {
                    col_grad_sum += group_col_grad[g * n_out * rank + c * rank + k];
                    col_grad_sum_contrib += group_col_grad_contrib[g * n_out * rank + c * rank + k];
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
                gamma_grad_sum += t_gamma_grad[static_cast<std::size_t>(t) * t_rank_stride + k];
                gamma_grad_sum_contrib +=
                    t_gamma_grad_contrib[static_cast<std::size_t>(t) * t_rank_stride + k];
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
    // from `input` directly. See disldo_backward.aqrs_additive_backward in
    // docs/research/linear_disldo.rst.
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
