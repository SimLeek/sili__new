#pragma once
#include "block4_codec.hpp"
#include "cpu_topology.hpp"
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
// batch. See docs/research/linear_disldo.rst.

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
 * to a real gradient (see disldo_forward.pure_computation in
 * docs/research/linear_disldo.rst for why a gradient-free forward-side
 * importance update was removed).
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
    // hold live synapses. See disldo_forward.dc_empty_check in
    // docs/research/linear_disldo.rst.
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
    // disldo_forward.tile_coord_collection in docs/research/linear_disldo.rst.
    if (weights.block4.n_tiles() > 0) {
        // Collect (br,bc,elem_pos,byte_pos) tuples once per call before the
        // parallel region -- row-major cursor walk isn't parallel-for
        // friendly, and a Block4Tile handle is move-only/RAII so it can't
        // be pre-collected across threads. See
        // disldo_forward.tile_coord_collection in docs/research/linear_disldo.rst.
        std::vector<uint32_t>& tile_br = weights.block4.scratch_tile_br;
        std::vector<uint32_t>& tile_bc = weights.block4.scratch_tile_bc;
        std::vector<std::size_t>& tile_elem = weights.block4.scratch_tile_elem;
        std::vector<std::size_t>& tile_byte = weights.block4.scratch_tile_byte;
        // disldo_forward.fp32_block4_cross_tile_pairing: FP32-only, marks
        // every SECOND tile of a same-br run (bk odd) as a "follower" --
        // already consumed by its leader (ti-1) once the parallel loop
        // below pairs two DIFFERENT tiles sharing a br. Computed for free
        // during this same sequential collection walk (bk is already the
        // per-row loop counter); unused (but harmlessly populated) for
        // FP8/FP4, which share this collection loop but keep their
        // existing per-tile (not cross-tile) processing.
        std::vector<uint8_t>& tile_is_follower = weights.block4.scratch_tile_is_follower;
        const std::size_t n_b4 = weights.block4.n_tiles();
        // resize()+direct indexing, not reserve()+push_back(): push_back's
        // per-call capacity check is measured exclusive cost at this scale.
        // See disldo_forward.tile_coord_collection in docs/research/linear_disldo.rst.
        tile_br.resize(n_b4);
        tile_bc.resize(n_b4);
        tile_elem.resize(n_b4);
        tile_byte.resize(n_b4);
        tile_is_follower.resize(n_b4);
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
                tile_is_follower[ti] = uint8_t(bk % 2 == 1);
                byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
            }
        }
        // Per-thread private output buffers -- necessary, not optional:
        // two tiles sharing a block-column write to the same output
        // positions. See disldo_forward.per_thread_output_buffers in
        // docs/research/linear_disldo.rst.
        std::vector<value_type> b4_out(static_cast<std::size_t>(num_cpus) * ost, value_type(0));
        // Hoisted loop bound -- measured instruction-count win (no
        // wall-clock effect). See disldo_forward.hoisted_tile_count in
        // docs/research/linear_disldo.rst.
        const int64_t n_tiles_local = int64_t(n_b4);
#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mo = b4_out.data() + static_cast<std::size_t>(tid) * ost;
#pragma omp for schedule(static)
            for (int64_t row_ti = 0; row_ti < n_tiles_local; ++row_ti) {
                // disldo_forward.block4_cross_tile_pairing (Phase 2 of the
                // block4_codec refactor, see docs/research/linear_disldo.
                // rst): pair this tile (the "leader") with its consecutive
                // same-br partner if one exists (tile_is_follower[row_ti+1],
                // set by the collection loop above), instead of always
                // falling back to within-tile column pairing. Forward is
                // read-only, so there's no live-handle-aliasing hazard from
                // holding two tile handles at once (unlike backward's port
                // of this same axis). ONE generic implementation for all
                // three precisions now -- the only per-precision difference
                // was ever the weight decode, which is now
                // Block4Codec<VALUES_TYPE>::decode_weight_column4 instead of
                // three copy-pasted ~140-line blocks (raw memcpy / fp8
                // gather+decode / fp4 mask+decode). Correctness and the real
                // speedup (measured, not assumed) were both validated first
                // via a standalone PoC -- see disldo_forward.
                // fp32_block4_cross_tile_pairing in docs/research/
                // linear_disldo.rst and
                // tests/unit/test_disldo_block4_fp32_crosstile_forward.cpp.
                using Codec = Block4Codec<VALUES_TYPE>;
                if (tile_is_follower[std::size_t(row_ti)])
                    continue; // already consumed by row_ti-1 as its partner.
                const uint32_t br = tile_br[std::size_t(row_ti)],
                               bcA = tile_bc[std::size_t(row_ti)];
                const auto tileA = weights.block4.at_index(br, bcA, tile_elem[std::size_t(row_ti)],
                                                           tile_byte[std::size_t(row_ti)]);
                const uint8_t* tdataA = tileA.raw_data();
                const bool has_partner =
                    (row_ti + 1 < n_tiles_local) && tile_is_follower[std::size_t(row_ti + 1)];
                if (has_partner) {
                    const uint32_t bcB = tile_bc[std::size_t(row_ti + 1)];
                    const auto tileB =
                        weights.block4.at_index(br, bcB, tile_elem[std::size_t(row_ti + 1)],
                                                tile_byte[std::size_t(row_ti + 1)]);
                    const uint8_t* tdataB = tileB.raw_data();
                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                        const std::size_t colA = std::size_t(bcA) * BLOCK4_TILE + lj;
                        const std::size_t colB = std::size_t(bcB) * BLOCK4_TILE + lj;
                        const bool haveA = colA < n_out;
                        const bool haveB = colB < n_out;
                        if (!haveA)
                            continue;

                        const Block4Vec wA = Codec::decode_weight_column4(tdataA, lj);
                        const Block4Vec wB = haveB ? Codec::decode_weight_column4(tdataB, lj)
                                                   : block4_vec_broadcast(0.0f);
                        const Block8Vec w_decoded8 = block8_vec_from_lo_hi(wA, wB);

                        std::size_t row_idx[BLOCK4_TILE];
                        Block8Vec s8;
                        for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                            const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                            row_idx[li] = row < n_in ? row : 0;
                            s8[li] = row < n_in ? weights.get_scale(row, colA) : value_type(0);
                            s8[li + 4] = (row < n_in && haveB) ? weights.get_scale(row, colB)
                                                               : value_type(0);
                        }
                        const Block8Vec w8 = w_decoded8 * s8;

                        for (SIZE_TYPE b = 0; b < batch; ++b) {
                            const value_type* in_row =
                                input + static_cast<std::size_t>(b) * in_cols;
                            Block8Vec in8;
                            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                                const value_type iv = in_row[row_idx[li]];
                                in8[li] = iv;
                                in8[li + 4] = iv;
                            }
                            const Block8Vec prod = w8 * in8;
                            mo[static_cast<std::size_t>(b) * n_out + colA] +=
                                prod[0] + prod[1] + prod[2] + prod[3];
                            if (haveB)
                                mo[static_cast<std::size_t>(b) * n_out + colB] +=
                                    prod[4] + prod[5] + prod[6] + prod[7];
                        }
                    }
                    continue;
                }
                // solo: no partner tile in this br -- fall back to the
                // existing within-tile column pairing, unchanged math
                // (disldo_forward.block4_avx2_column_pairing).
                static_assert(BLOCK4_TILE == 4,
                              "column pairing below assumes exactly 4 columns (2 pairs)");
                const uint32_t bc = bcA;
                const uint8_t* tdata = tdataA;
                const uint32_t br_ = br;
                auto process_pair = [&, br_]<uint32_t LJ0>() {
                    constexpr uint32_t LJ1 = LJ0 + 1;
                    const std::size_t col0 = std::size_t(bc) * BLOCK4_TILE + LJ0;
                    const std::size_t col1 = std::size_t(bc) * BLOCK4_TILE + LJ1;
                    const bool have0 = col0 < n_out;
                    const bool have1 =
                        col1 < n_out; // have0==false implies have1==false (col1>col0)
                    if (!have0)
                        return;

                    const Block4Vec w_decoded0 = Codec::decode_weight_column4(tdata, LJ0);
                    const Block4Vec w_decoded1 = have1 ? Codec::decode_weight_column4(tdata, LJ1)
                                                       : block4_vec_broadcast(0.0f);
                    const Block8Vec w_decoded8 = block8_vec_from_lo_hi(w_decoded0, w_decoded1);

                    std::size_t row_idx[BLOCK4_TILE];
                    Block8Vec s8;
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = std::size_t(br_) * BLOCK4_TILE + li;
                        row_idx[li] = row < n_in ? row : 0;
                        s8[li] = row < n_in ? weights.get_scale(row, col0) : value_type(0);
                        s8[li + 4] =
                            (row < n_in && have1) ? weights.get_scale(row, col1) : value_type(0);
                    }
                    const Block8Vec w8 = w_decoded8 * s8;

                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        const value_type* in_row = input + static_cast<std::size_t>(b) * in_cols;
                        Block8Vec in8;
                        for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                            const value_type iv = in_row[row_idx[li]];
                            in8[li] = iv;
                            in8[li + 4] = iv;
                        }
                        const Block8Vec prod = w8 * in8;
                        mo[static_cast<std::size_t>(b) * n_out + col0] +=
                            prod[0] + prod[1] + prod[2] + prod[3];
                        if (have1)
                            mo[static_cast<std::size_t>(b) * n_out + col1] +=
                                prod[4] + prod[5] + prod[6] + prod[7];
                    }
                };
                process_pair.template operator()<0>();
                process_pair.template operator()<2>();
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
    // disldo_forward.aqrs_additive_branch in docs/research/linear_disldo.rst.
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
