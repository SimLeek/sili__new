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
        // Same persistent-scratch + static-per-thread-range reduction
        // already proven for the block4 paths -- see
        // disldo_forward.persistent_scratch_buffers in
        // docs/research/linear_disldo.rst (measured NOT to matter here,
        // see the scattered-path note at the end of that section --
        // kept anyway since it's correct and harmless).
        std::vector<value_type>& t_out = weights.block4.scratch_scattered_out;
        t_out.resize(static_cast<std::size_t>(num_cpus) * ost);

#pragma omp parallel num_threads(num_cpus)
        {
            const int tid = omp_get_thread_num();
            value_type* mo = t_out.data() + static_cast<std::size_t>(tid) * ost;
            std::fill(mo, mo + ost, value_type(0));

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

                    // No per-sample zero-skip here -- DISLDO is Dense Input
                    // by design (see file header), so input is almost
                    // never exactly zero in the intended use case, making
                    // this branch nearly-always-false overhead on every
                    // (synapse, batch-sample) pair rather than a real skip
                    // (measured via callgrind as 18% of this function's
                    // total instructions at batch=256). w*0=0 either way,
                    // so dropping the check is a pure perf change, not a
                    // behavior change. SISLDO (sparse input, CSRInput) is
                    // the actual sparse-input path and has no analogous
                    // check to remove -- its input format only ever
                    // contains nonzero entries in the first place.
                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        const value_type iv = input[static_cast<std::size_t>(b) * in_cols + r];
                        mo[static_cast<std::size_t>(b) * n_out + col] += w * iv;
                    }
                }
            }
            // Parallel tree reduction + flush, static per-thread index
            // range, one barrier (implicit, end of the `#pragma omp for`
            // above) -- see disldo_forward.narrow_path_barrier_reduction
            // in docs/research/linear_disldo.rst for why this needs no
            // further barriers.
            const std::size_t i_lo =
                (static_cast<std::size_t>(tid) * ost) / static_cast<std::size_t>(num_cpus);
            const std::size_t i_hi =
                (static_cast<std::size_t>(tid + 1) * ost) / static_cast<std::size_t>(num_cpus);
            for (int stride = 1; stride < num_cpus; stride *= 2) {
                const int step = stride * 2;
                for (std::size_t i = i_lo; i < i_hi; ++i) {
                    for (int base = 0; base + stride < num_cpus; base += step)
                        t_out[static_cast<std::size_t>(base) * ost + i] +=
                            t_out[static_cast<std::size_t>(base + stride) * ost + i];
                }
            }
            for (std::size_t i = i_lo; i < i_hi; ++i)
                output[i] += t_out[i];
        }
    } // !dc.empty()

    // block4 contribution -- same shared per-row value_scale/output_scale
    // as the scattered path above. No gather: within an active tile,
    // position IS the column (fixed compile-time offset), which is what
    // lets this SIMD where the scattered loop above can't. See
    // disldo_forward.tile_coord_collection in docs/research/linear_disldo.rst.
    if (weights.block4.n_tiles() > 0) {
        // Two threading strategies chosen by layer width -- see
        // disldo_forward.column_partitioned_threading and
        // disldo_forward.narrow_layer_tree_reduction in
        // docs/research/linear_disldo.rst.
        const auto& BL4 = weights.block4.block_layout;
        const std::size_t n_bc_total = BL4.cols;

        // Per-row lookahead (this row's tiles, gathered before deciding
        // pairing) -- reused/cleared every row instead of reallocated.
        std::vector<uint32_t>& row_bc = weights.block4.scratch_row_bc_lookahead;
        std::vector<std::size_t>& row_elem = weights.block4.scratch_row_elem_lookahead;
        std::vector<std::size_t>& row_byte = weights.block4.scratch_row_byte_lookahead;
        std::vector<Block4WorkItem>& flat_items = weights.block4.scratch_flat_items;
        flat_items.clear();

        for (std::size_t br = 0; br < BL4.rows; ++br) {
            const std::size_t n_bc = BL4.row_nnz(br);
            if (n_bc == 0)
                continue;
            row_bc.clear();
            row_elem.clear();
            row_byte.clear();
            auto bc_cursor = weights.block4.row_cursor(br);
            std::size_t elem_pos = BL4.elem_start[br];
            std::size_t byte_pos = weights.block4.tile_byte_start[br];
            for (std::size_t bk = 0; bk < n_bc; ++bk, ++elem_pos) {
                row_bc.push_back(bc_cursor.advance());
                row_elem.push_back(elem_pos);
                row_byte.push_back(byte_pos);
                byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
            }
            // disldo_forward.fp32_block4_cross_tile_pairing: pair this
            // tile with the next one IN THIS ROW (bk parity, unconditional
            // -- ownership isn't decided until/unless the wide path
            // re-buckets this list below).
            for (std::size_t bk = 0; bk < n_bc; bk += 2) {
                Block4WorkItem item;
                item.br = uint32_t(br);
                item.bcA = row_bc[bk];
                item.elemA = row_elem[bk];
                item.byteA = row_byte[bk];
                if (bk + 1 < n_bc) {
                    item.has_partner = true;
                    item.bcB = row_bc[bk + 1];
                    item.elemB = row_elem[bk + 1];
                    item.byteB = row_byte[bk + 1];
                } else {
                    item.has_partner = false;
                }
                flat_items.push_back(item);
            }
        }

        // See the big comment above: measured threshold (arch-sandbox,
        // CCX-pinned) -- 4 cols/thread was a severe regression, 9 was
        // decent-but-not-great, 18 was a clean win. See docs/research/
        // linear_disldo.rst for the actual sweep this was tuned from.
        constexpr std::size_t COLUMN_PARTITION_MIN_COLS_PER_THREAD = 12;
        const std::size_t cols_per_thread = n_bc_total / static_cast<std::size_t>(num_cpus);
        const bool use_column_partition = cols_per_thread >= COLUMN_PARTITION_MIN_COLS_PER_THREAD;

        std::vector<std::vector<Block4WorkItem>>& thread_items =
            weights.block4.scratch_thread_items;
        if (use_column_partition) {
            if (thread_items.size() != static_cast<std::size_t>(num_cpus))
                thread_items.resize(static_cast<std::size_t>(num_cpus));
            for (auto& items : thread_items)
                items.clear();

            // Contiguous, disjoint column-block ranges, split as evenly as
            // possible: thread t owns [thread_bc_start[t], thread_bc_start[t+1]).
            std::vector<std::size_t>& thread_bc_start = weights.block4.scratch_thread_bc_start;
            thread_bc_start.resize(static_cast<std::size_t>(num_cpus) + 1);
            for (int t = 0; t <= num_cpus; ++t)
                thread_bc_start[static_cast<std::size_t>(t)] =
                    (static_cast<std::size_t>(t) * n_bc_total) / static_cast<std::size_t>(num_cpus);
            // Flat O(1)-lookup owner table instead of a per-tile binary
            // search -- measured directly: a std::upper_bound() per tile
            // was a real, mostly-fixed per-call tax (~50-90us at
            // n_out=288 regardless of batch), the dominant cost of this
            // whole restructure at low batch before this table existed.
            std::vector<int32_t>& bc_owner_table = weights.block4.scratch_bc_owner_table;
            bc_owner_table.resize(n_bc_total);
            for (int t = 0; t < num_cpus; ++t) {
                const std::size_t lo = thread_bc_start[static_cast<std::size_t>(t)];
                const std::size_t hi = thread_bc_start[static_cast<std::size_t>(t) + 1];
                std::fill(bc_owner_table.begin() + static_cast<std::ptrdiff_t>(lo),
                          bc_owner_table.begin() + static_cast<std::ptrdiff_t>(hi), t);
            }

            // Re-bucket the flat list: a pair straddling a thread
            // boundary would need cross-thread writes, exactly what this
            // path exists to avoid, so it's demoted to two solo items
            // instead (each still cheap: solo processing never leaves its
            // own tile's 4 columns, always inside one thread's range by
            // construction).
            for (const Block4WorkItem& src : flat_items) {
                const int ownerA = bc_owner_table[src.bcA];
                if (src.has_partner && bc_owner_table[src.bcB] == ownerA) {
                    thread_items[static_cast<std::size_t>(ownerA)].push_back(src);
                } else {
                    Block4WorkItem solo = src;
                    solo.has_partner = false;
                    thread_items[static_cast<std::size_t>(ownerA)].push_back(solo);
                    if (src.has_partner) {
                        Block4WorkItem partner;
                        partner.br = src.br;
                        partner.bcA = src.bcB;
                        partner.elemA = src.elemB;
                        partner.byteA = src.byteB;
                        partner.has_partner = false;
                        thread_items[static_cast<std::size_t>(bc_owner_table[src.bcB])].push_back(
                            partner);
                    }
                }
            }
        }
        // gamma_k is layer-wide (not row/col-dependent) -- hoisted OUTSIDE
        // the parallel region entirely, computed once per call and shared
        // read-only across every thread, instead of get_scale() re-deriving
        // it from get_scale_gamma_k()'s own lazy-default check on every
        // single (row,col) cell. See disldo_forward.get_scale_row_col_cache
        // in docs/research/linear_disldo.rst.
        //
        // weights.scale_rank is 0 by default for full-precision layers
        // (fp32/fp64 -- see is_full_precision_values in
        // delta_csr_types.hpp): every row_scale_cache/col_scale_cache fill
        // loop below becomes a 0-iteration no-op, and the dot-product
        // loops start their accumulator at scale_identity (1) instead of
        // 0 and never execute either -- so the accumulator is left at
        // exactly 1, matching get_scale()'s own rank==0 short-circuit,
        // with NO extra branching inside the hot per-cell loops
        // themselves.
        const std::size_t scale_rank = weights.scale_rank;
        const value_type scale_identity = scale_rank == 0 ? value_type(1) : value_type(0);
        std::vector<value_type> gamma_k_arr(scale_rank);
        for (std::size_t k = 0; k < scale_rank; ++k)
            gamma_k_arr[k] = weights.get_scale_gamma_k(k);
        // rank==0 fast-path constants (see below): S is then always
        // exactly 1, so the whole per-lane sA/sB dot-product collapses to
        // pure boundary masking -- for a full (non-boundary) tile-pair
        // that's just a constant vector, no per-lane loop at all.
        // Measured via callgrind as ~23% of disldo_forward's total
        // instructions at batch=1 (over-lane construction dominating the
        // actual SIMD math by more than 10x) -- see
        // disldo_forward.rank0_s8_fast_path in docs/research/linear_disldo.rst.
        const Block4Vec ones4 = block4_vec_broadcast(1.0f);
        const Block4Vec zeros4 = block4_vec_broadcast(0.0f);
        // disldo_forward.block4_item_processor: the actual per-item SIMD
        // math, factored out into one callable shared by BOTH strategies
        // below -- only the OUTPUT POINTER (`mo`: `output` directly for
        // the wide path, or a private per-thread slice for the narrow
        // path's pre-reduction buffer) and per-thread scratch differ.
        // Captures only read-only, thread-safe state by reference
        // (weights, input dims, scale_rank/gamma_k_arr/ones4/zeros4); the
        // genuinely per-thread state (row_scale_cache/col_scale_cache) is
        // passed in explicitly since a captured reference would alias
        // across threads. See disldo_forward.block4_cross_tile_pairing
        // (Phase 2 of the block4_codec refactor, docs/research/
        // linear_disldo.rst) for why leader+follower tile-pairs get
        // processed together via 8-wide SIMD instead of always falling
        // back to within-tile column pairing, and
        // tests/unit/test_disldo_block4_fp32_crosstile_forward.cpp for
        // the standalone PoC that validated the real (measured, not
        // assumed) speedup first. ONE generic implementation for all
        // three precisions -- the only per-precision difference is the
        // weight decode, Block4Codec<VALUES_TYPE>::decode_weight_column4.
        // disldo_forward.batch_blocked_wide_path (Phase 1, see
        // TODO_BATCH_BLOCKING.md): when `input_T` is non-null, `mo` is
        // reinterpreted as a per-thread PRIVATE column-major accumulator
        // (thread's own columns only, offset by `col_lo`) instead of the
        // real row-major `output` -- see block4_batch_accumulate's own
        // docstring in block4.hpp for the full rationale. `input_T` and
        // `col_lo` default to "off" (nullptr / 0) so every other call site
        // (narrow path, Phase 2+) is untouched.
        auto process_block4_item = [&](const Block4WorkItem& item, value_type* mo,
                                       value_type* row_scale_cache, value_type* col_scale_cache,
                                       const value_type* input_T = nullptr,
                                       std::size_t col_lo = 0) {
            using Codec = Block4Codec<VALUES_TYPE>;
            const uint32_t br = item.br, bcA = item.bcA;
            const auto tileA = weights.block4.at_index(br, bcA, item.elemA, item.byteA);
            const uint8_t* tdataA = tileA.raw_data();
            if (item.has_partner) {
                const uint32_t bcB = item.bcB;
                const auto tileB = weights.block4.at_index(br, bcB, item.elemB, item.byteB);
                const uint8_t* tdataB = tileB.raw_data();
                // Row-level value_scale_k -- doesn't depend on lj at
                // all (br is fixed for this whole tile-pair), so cache
                // it ONCE here instead of once per lj (was 4x
                // redundant).
                std::size_t row_idx[BLOCK4_TILE];
                // True iff all 4 rows of this br block are in bounds --
                // false only for the last (possibly partial) row block.
                // See the rank==0 s8 fast path below.
                const bool full_rows = std::size_t(br) * BLOCK4_TILE + BLOCK4_TILE <= n_in;
                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                    const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                    row_idx[li] = row < n_in ? row : 0;
                    for (std::size_t k = 0; k < scale_rank; ++k)
                        row_scale_cache[li * scale_rank + k] =
                            row < n_in ? weights.get_value_scale_k(row, k) : value_type(0);
                }
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

                    // Column-level output_scale_k -- once per lj, not
                    // once per li (was 4x redundant).
                    for (std::size_t k = 0; k < scale_rank; ++k) {
                        col_scale_cache[k] = weights.get_output_scale_k(colA, k);
                        col_scale_cache[scale_rank + k] =
                            haveB ? weights.get_output_scale_k(colB, k) : value_type(0);
                    }
                    Block8Vec s8;
                    if (scale_rank == 0 && full_rows) {
                        // S is always exactly 1 here (see scale_identity
                        // above) and every row is in bounds -- s8 is a
                        // pure constant, no per-lane loop needed.
                        s8 = haveB ? block8_vec_from_lo_hi(ones4, ones4)
                                   : block8_vec_from_lo_hi(ones4, zeros4);
                    } else if (scale_rank == 0) {
                        // Last (partial) row block -- still no scale
                        // lookup needed, just per-lane row masking.
                        for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                            const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                            s8[li] = row < n_in ? value_type(1) : value_type(0);
                            s8[li + 4] = (row < n_in && haveB) ? value_type(1) : value_type(0);
                        }
                    } else {
                        for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                            const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                            value_type sA = scale_identity, sB = scale_identity;
                            for (std::size_t k = 0; k < scale_rank; ++k) {
                                const value_type vsk = row_scale_cache[li * scale_rank + k];
                                sA += gamma_k_arr[k] * vsk * col_scale_cache[k];
                                sB += gamma_k_arr[k] * vsk * col_scale_cache[scale_rank + k];
                            }
                            s8[li] = row < n_in ? sA : value_type(0);
                            s8[li + 4] = (row < n_in && haveB) ? sB : value_type(0);
                        }
                    }
                    const Block8Vec w8 = w_decoded8 * s8;

                    if (input_T != nullptr) {
                        // disldo_forward.batch_blocked_wide_path: mo is
                        // this thread's private column-major accumulator
                        // here, not `output` -- see block4_batch_accumulate.
                        const value_type* row_ptrs[BLOCK4_TILE];
                        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
                            row_ptrs[li] = input_T + row_idx[li] * static_cast<std::size_t>(batch);
                        float wA[BLOCK4_TILE], wB[BLOCK4_TILE];
                        for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                            wA[li] = w8[li];
                            wB[li] = w8[li + 4];
                        }
                        block4_batch_accumulate(wA, row_ptrs, batch,
                                                mo + (colA - col_lo) *
                                                         static_cast<std::size_t>(batch));
                        if (haveB)
                            block4_batch_accumulate(wB, row_ptrs, batch,
                                                    mo + (colB - col_lo) *
                                                             static_cast<std::size_t>(batch));
                        continue;
                    }

                    // full_rows: row_idx[0..3] are 4 consecutive rows,
                    // one contiguous wide load instead of 4 scalar reads.
                    for (SIZE_TYPE b = 0; b < batch; ++b) {
                        const value_type* in_row = input + static_cast<std::size_t>(b) * in_cols;
                        Block8Vec in8;
                        if (full_rows) {
                            in8 = block8_vec_dup4(block4_vec_load(in_row + row_idx[0]));
                        } else {
                            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                                const value_type iv = in_row[row_idx[li]];
                                in8[li] = iv;
                                in8[li + 4] = iv;
                            }
                        }
                        const Block8Vec prod = w8 * in8;
                        mo[static_cast<std::size_t>(b) * n_out + colA] +=
                            prod[0] + prod[1] + prod[2] + prod[3];
                        if (haveB)
                            mo[static_cast<std::size_t>(b) * n_out + colB] +=
                                prod[4] + prod[5] + prod[6] + prod[7];
                    }
                }
                return;
            }
            // solo: no partner tile -- fall back to the existing
            // within-tile column pairing, unchanged math
            // (disldo_forward.block4_avx2_column_pairing).
            static_assert(BLOCK4_TILE == 4,
                          "column pairing below assumes exactly 4 columns (2 pairs)");
            const uint32_t bc = bcA;
            const uint8_t* tdata = tdataA;
            const uint32_t br_ = br;
            std::size_t row_idx[BLOCK4_TILE];
            const bool full_rows = std::size_t(br_) * BLOCK4_TILE + BLOCK4_TILE <= n_in;
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                const std::size_t row = std::size_t(br_) * BLOCK4_TILE + li;
                row_idx[li] = row < n_in ? row : 0;
                for (std::size_t k = 0; k < scale_rank; ++k)
                    row_scale_cache[li * scale_rank + k] =
                        row < n_in ? weights.get_value_scale_k(row, k) : value_type(0);
            }
            auto process_pair = [&, br_]<uint32_t LJ0>() {
                constexpr uint32_t LJ1 = LJ0 + 1;
                const std::size_t col0 = std::size_t(bc) * BLOCK4_TILE + LJ0;
                const std::size_t col1 = std::size_t(bc) * BLOCK4_TILE + LJ1;
                const bool have0 = col0 < n_out;
                const bool have1 = col1 < n_out; // have0==false implies have1==false
                if (!have0)
                    return;

                const Block4Vec w_decoded0 = Codec::decode_weight_column4(tdata, LJ0);
                const Block4Vec w_decoded1 =
                    have1 ? Codec::decode_weight_column4(tdata, LJ1) : block4_vec_broadcast(0.0f);
                const Block8Vec w_decoded8 = block8_vec_from_lo_hi(w_decoded0, w_decoded1);

                for (std::size_t k = 0; k < scale_rank; ++k) {
                    col_scale_cache[k] = weights.get_output_scale_k(col0, k);
                    col_scale_cache[scale_rank + k] =
                        have1 ? weights.get_output_scale_k(col1, k) : value_type(0);
                }
                Block8Vec s8;
                if (scale_rank == 0 && full_rows) {
                    s8 = have1 ? block8_vec_from_lo_hi(ones4, ones4)
                               : block8_vec_from_lo_hi(ones4, zeros4);
                } else if (scale_rank == 0) {
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = std::size_t(br_) * BLOCK4_TILE + li;
                        s8[li] = row < n_in ? value_type(1) : value_type(0);
                        s8[li + 4] = (row < n_in && have1) ? value_type(1) : value_type(0);
                    }
                } else {
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = std::size_t(br_) * BLOCK4_TILE + li;
                        value_type s0 = scale_identity, s1 = scale_identity;
                        for (std::size_t k = 0; k < scale_rank; ++k) {
                            const value_type vsk = row_scale_cache[li * scale_rank + k];
                            s0 += gamma_k_arr[k] * vsk * col_scale_cache[k];
                            s1 += gamma_k_arr[k] * vsk * col_scale_cache[scale_rank + k];
                        }
                        s8[li] = row < n_in ? s0 : value_type(0);
                        s8[li + 4] = (row < n_in && have1) ? s1 : value_type(0);
                    }
                }
                const Block8Vec w8 = w_decoded8 * s8;

                if (input_T != nullptr) {
                    const value_type* row_ptrs[BLOCK4_TILE];
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
                        row_ptrs[li] = input_T + row_idx[li] * static_cast<std::size_t>(batch);
                    float w0[BLOCK4_TILE], w1[BLOCK4_TILE];
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        w0[li] = w8[li];
                        w1[li] = w8[li + 4];
                    }
                    block4_batch_accumulate(w0, row_ptrs, batch,
                                            mo + (col0 - col_lo) * static_cast<std::size_t>(batch));
                    if (have1)
                        block4_batch_accumulate(w1, row_ptrs, batch,
                                                mo + (col1 - col_lo) *
                                                         static_cast<std::size_t>(batch));
                    return;
                }

                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const value_type* in_row = input + static_cast<std::size_t>(b) * in_cols;
                    Block8Vec in8;
                    if (full_rows) {
                        in8 = block8_vec_dup4(block4_vec_load(in_row + row_idx[0]));
                    } else {
                        for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                            const value_type iv = in_row[row_idx[li]];
                            in8[li] = iv;
                            in8[li + 4] = iv;
                        }
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
        };

        // See disldo_forward.batch_blocked_threshold in
        // docs/research/linear_disldo.rst -- shared by the wide and
        // narrow dispatch below.
        constexpr SIZE_TYPE BLOCK4_BATCH_BLOCK_THRESHOLD = 8;

        if (use_column_partition) {
            // Wide path: thread_items[tid] was already assigned to this
            // exact thread during collection, by output column-block
            // ownership -- that's what makes writing directly into
            // `output` safe with zero cross-thread writes, zero
            // reduction. At/above threshold, see
            // disldo_forward.batch_blocked_threshold in
            // docs/research/linear_disldo.rst.
            if (batch >= BLOCK4_BATCH_BLOCK_THRESHOLD) {
                std::vector<value_type>& input_T = weights.block4.scratch_input_T;
                input_T.resize(n_in * static_cast<std::size_t>(batch));
#pragma omp parallel for num_threads(num_cpus) schedule(static)
                for (std::size_t r = 0; r < n_in; ++r) {
                    value_type* dst = input_T.data() + r * static_cast<std::size_t>(batch);
                    for (SIZE_TYPE b = 0; b < batch; ++b)
                        dst[b] = input[static_cast<std::size_t>(b) * in_cols + r];
                }

                // See disldo_forward.persistent_scratch_buffers in
                // docs/research/linear_disldo.rst.
                std::vector<value_type>& thread_buf_all = weights.block4.scratch_thread_buf;
                thread_buf_all.resize(n_out * static_cast<std::size_t>(batch));
#pragma omp parallel num_threads(num_cpus)
                {
                    const int tid = omp_get_thread_num();
                    std::vector<value_type> row_scale_cache(BLOCK4_TILE * scale_rank);
                    std::vector<value_type> col_scale_cache(2 * scale_rank);
                    const std::vector<std::size_t>& thread_bc_start_ =
                        weights.block4.scratch_thread_bc_start;
                    const std::size_t col_lo = std::min(
                        thread_bc_start_[static_cast<std::size_t>(tid)] * BLOCK4_TILE, n_out);
                    const std::size_t col_hi = std::min(
                        thread_bc_start_[static_cast<std::size_t>(tid) + 1] * BLOCK4_TILE, n_out);
                    if (col_hi > col_lo) {
                        const std::size_t thread_ncols = col_hi - col_lo;
                        value_type* thread_buf =
                            thread_buf_all.data() + col_lo * static_cast<std::size_t>(batch);
                        std::fill(thread_buf,
                                  thread_buf + thread_ncols * static_cast<std::size_t>(batch),
                                  value_type(0));
                        for (const Block4WorkItem& item :
                             thread_items[static_cast<std::size_t>(tid)])
                            process_block4_item(item, thread_buf, row_scale_cache.data(),
                                                col_scale_cache.data(), input_T.data(), col_lo);
                        // Transpose-back flush: this thread's columns
                        // only, so no cross-thread writes here either.
                        for (std::size_t col = col_lo; col < col_hi; ++col) {
                            const value_type* src =
                                thread_buf + (col - col_lo) * static_cast<std::size_t>(batch);
                            for (SIZE_TYPE b = 0; b < batch; ++b)
                                output[static_cast<std::size_t>(b) * n_out + col] += src[b];
                        }
                    }
                }
            } else {
#pragma omp parallel num_threads(num_cpus)
                {
                    const int tid = omp_get_thread_num();
                    std::vector<value_type> row_scale_cache(BLOCK4_TILE * scale_rank);
                    std::vector<value_type> col_scale_cache(2 * scale_rank);
                    for (const Block4WorkItem& item : thread_items[static_cast<std::size_t>(tid)])
                        process_block4_item(item, output, row_scale_cache.data(),
                                            col_scale_cache.data());
                }
            }
        } else {
            // Narrow path: process the flat, row-major item list via a
            // real `#pragma omp for` (contiguous chunk per thread -- same
            // locality the old row-partitioned scheme had) into a private
            // per-thread buffer, then combine with a PARALLEL tree
            // reduction inside this same region instead of one thread
            // summing num_cpus * ost elements serially afterward: each of
            // the log2(num_cpus) rounds halves the number of "active"
            // buffers, and every round's own combine work is itself split
            // across all num_cpus threads via `#pragma omp for` (not just
            // done by whichever thread owns that round's pair) -- total
            // combine work is still O(ost * num_cpus), same as a serial
            // reduction, but wall-clock drops to O(ost * log2(num_cpus))
            // since it's parallelized every round instead of serialized
            // once. See disldo_forward.per_thread_output_buffers in
            // docs/research/linear_disldo.rst.
            //
            // Unlike the wide path, items here are NOT column-
            // partitioned, so at/above threshold the per-thread buffer
            // is column-major [n_out, batch] (full range, not just an
            // owned slice) -- see disldo_forward.batch_blocked_threshold
            // in docs/research/linear_disldo.rst.
            if (batch >= BLOCK4_BATCH_BLOCK_THRESHOLD) {
                std::vector<value_type>& input_T = weights.block4.scratch_input_T;
                input_T.resize(n_in * static_cast<std::size_t>(batch));
#pragma omp parallel for num_threads(num_cpus) schedule(static)
                for (std::size_t r = 0; r < n_in; ++r) {
                    value_type* dst = input_T.data() + r * static_cast<std::size_t>(batch);
                    for (SIZE_TYPE b = 0; b < batch; ++b)
                        dst[b] = input[static_cast<std::size_t>(b) * in_cols + r];
                }

                // See disldo_forward.persistent_scratch_buffers in
                // docs/research/linear_disldo.rst.
                std::vector<value_type>& b4_out_T = weights.block4.scratch_b4_out;
                b4_out_T.resize(static_cast<std::size_t>(num_cpus) * ost);
                const int64_t n_items_local = int64_t(flat_items.size());
#pragma omp parallel num_threads(num_cpus)
                {
                    const int tid = omp_get_thread_num();
                    // Column-major [n_out, batch] for this thread's slice.
                    value_type* mo = b4_out_T.data() + static_cast<std::size_t>(tid) * ost;
                    std::fill(mo, mo + ost, value_type(0));
                    std::vector<value_type> row_scale_cache(BLOCK4_TILE * scale_rank);
                    std::vector<value_type> col_scale_cache(2 * scale_rank);
#pragma omp for schedule(static)
                    for (int64_t idx = 0; idx < n_items_local; ++idx)
                        process_block4_item(flat_items[static_cast<std::size_t>(idx)], mo,
                                            row_scale_cache.data(), col_scale_cache.data(),
                                            input_T.data(), std::size_t(0));
                    // The ONE barrier this reduction genuinely needs: a
                    // thread's own column range below may hold data
                    // written by ANY other thread's items above (item
                    // ownership above is by flat-list position, not by
                    // column) -- so every write above must be visible
                    // before any thread starts reducing. Implicit at the
                    // end of the `#pragma omp for` above; kept. See
                    // disldo_forward.narrow_path_barrier_reduction in
                    // docs/research/linear_disldo.rst: below, each thread
                    // claims a FIXED static column range (GPU-global-id
                    // style) and does ALL reduction rounds + the flush
                    // for it sequentially, no further barriers -- correct
                    // because round N+1 for a column only ever depends on
                    // a value THIS thread wrote in round N. Loop order
                    // matters: stride MUST stay outermost (not
                    // column/batch), or every element thrashes between
                    // widely-separated buffer copies every iteration --
                    // see the docs section for the regression that
                    // caused before it was caught.
                    const std::size_t col_lo = (static_cast<std::size_t>(tid) * n_out) /
                                               static_cast<std::size_t>(num_cpus);
                    const std::size_t col_hi = (static_cast<std::size_t>(tid + 1) * n_out) /
                                               static_cast<std::size_t>(num_cpus);
                    const std::size_t i_lo = col_lo * static_cast<std::size_t>(batch);
                    const std::size_t i_hi = col_hi * static_cast<std::size_t>(batch);
                    for (int stride = 1; stride < num_cpus; stride *= 2) {
                        const int step = stride * 2;
                        for (std::size_t i = i_lo; i < i_hi; ++i) {
                            for (int base = 0; base + stride < num_cpus; base += step)
                                b4_out_T[static_cast<std::size_t>(base) * ost + i] +=
                                    b4_out_T[static_cast<std::size_t>(base + stride) * ost + i];
                        }
                    }
                    // Transpose-back flush: b4_out_T[0*ost + col*batch + b]
                    // now holds the fully-reduced value for (col, b) --
                    // same column range this thread already owns above,
                    // still zero cross-thread reads.
                    for (std::size_t col = col_lo; col < col_hi; ++col) {
                        const value_type* src =
                            b4_out_T.data() + col * static_cast<std::size_t>(batch);
                        for (SIZE_TYPE b = 0; b < batch; ++b)
                            output[static_cast<std::size_t>(b) * n_out + col] += src[b];
                    }
                }
            } else {
                // Same persistent-scratch + barrier-reduction treatment
                // as the blocked branch above -- see
                // disldo_forward.persistent_scratch_buffers and
                // disldo_forward.narrow_path_barrier_reduction in
                // docs/research/linear_disldo.rst. Row-major layout
                // here means any contiguous flat-index range works (no
                // column alignment needed) and the flush is a trivial
                // 1:1 copy, no transpose.
                std::vector<value_type>& b4_out = weights.block4.scratch_b4_out;
                b4_out.resize(static_cast<std::size_t>(num_cpus) * ost);
                const int64_t n_items_local = int64_t(flat_items.size());
#pragma omp parallel num_threads(num_cpus)
                {
                    const int tid = omp_get_thread_num();
                    value_type* mo = b4_out.data() + static_cast<std::size_t>(tid) * ost;
                    std::fill(mo, mo + ost, value_type(0));
                    std::vector<value_type> row_scale_cache(BLOCK4_TILE * scale_rank);
                    std::vector<value_type> col_scale_cache(2 * scale_rank);
#pragma omp for schedule(static)
                    for (int64_t idx = 0; idx < n_items_local; ++idx)
                        process_block4_item(flat_items[static_cast<std::size_t>(idx)], mo,
                                            row_scale_cache.data(), col_scale_cache.data());
                    const std::size_t i_lo =
                        (static_cast<std::size_t>(tid) * ost) / static_cast<std::size_t>(num_cpus);
                    const std::size_t i_hi = (static_cast<std::size_t>(tid + 1) * ost) /
                                             static_cast<std::size_t>(num_cpus);
                    for (int stride = 1; stride < num_cpus; stride *= 2) {
                        const int step = stride * 2;
                        for (std::size_t i = i_lo; i < i_hi; ++i) {
                            for (int base = 0; base + stride < num_cpus; base += step)
                                b4_out[static_cast<std::size_t>(base) * ost + i] +=
                                    b4_out[static_cast<std::size_t>(base + stride) * ost + i];
                        }
                    }
                    for (std::size_t i = i_lo; i < i_hi; ++i)
                        output[i] += b4_out[i];
                }
            }
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
