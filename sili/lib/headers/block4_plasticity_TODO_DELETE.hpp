#pragma once
#include "delta_csr_types.hpp" // pulls in block4.hpp; PlasticityState/PlasticityStats/plasticity_select_cycle_boundary live here

// TODO: DELETE ME LATER. Same rationale as block4_decay_TODO_DELETE.hpp
// (this file's direct sibling): the block4 forward/backward kernels are
// not yet decomposed into a reusable "walk every populated tile" +
// pluggable per-cell operation, so this is a SEPARATE, deliberately
// simple (single-threaded, no SIMD) traversal built ONLY on
// already-tested primitives (Block4Store32::row_cursor/tile_len_at/
// at_index, Block4TileHandle32::get_weight/set_weight/get_importance/
// set_importance) -- it does NOT re-derive sparse/dense tile packing
// itself; Block4TileHandle32 already handles that correctly. Once the
// block4 kernels ARE split into composable functions with a pluggable
// per-cell callback, this file should be deleted and the plasticity
// mechanism should become a callback swap into that shared traversal.
//
// See docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design
// for the full mechanism derivation (top-K-by-importance FROZEN pool
// gated by a local per-column gradient-activity deviation -- the ONLY
// pool now, the dead pool was pruned after being identified as the
// likely cause of a real-run performance regression; see
// delta_csr_types.hpp's own file-header comment for the full
// derivation) -- this file mirrors
// tests/unit/test_amortized_plasticity_reset.cpp's (scattered)
// SAME per-cell logic and cycle-boundary selection
// (plasticity_select_cycle_boundary, delta_csr_types.hpp -- SHARED
// between scattered and block4, since it only touches the per-column
// PlasticityState, not either storage format's own cursor shape) --
// only the CELL TRAVERSAL differs, matching
// block4_decay_TODO_DELETE.hpp's own established split.
//
// chunk_size here counts TILES (block4's natural atomic unit, matching
// apply_amortized_block4_decay_stats's own convention), not individual
// cells -- each touched tile yields up to 16 per-cell updates. Only fp32
// (Block4Store32/Block4TileHandle32), same scope as the decay work.

struct Block4PlasticityCursor {
    std::size_t row = 0;
    std::size_t elem_pos = 0;
    std::size_t byte_pos = 0;
    bool initialized = false;
};

inline PlasticityStats apply_amortized_block4_plasticity_step(
    Block4Store32& store, std::size_t n_out, PlasticityState& state, Block4PlasticityCursor& cursor,
    std::size_t chunk_size, float eta, float eta_slow, float eta_slow_catchup, float eta_fast,
    float blend, float reset_fraction, float k, float eta_var = 0.9f, float l2_decay_lambda = 0.0f,
    float l2_decay_threshold = 0.9f, float l2_decay_temperature = 0.05f, float max_ci = 100.0f,
    bool select_by_deviation = false) {
    state.ensure_sized(n_out);
    const auto& BL = store.block_layout;
    const std::size_t n_rows = BL.rows;
    PlasticityStats out;

    if (n_rows == 0 || store.n_tiles() == 0) {
        out.cycle_complete = true;
        return out;
    }

    if (!cursor.initialized) {
        cursor.row = 0;
        cursor.elem_pos = 0;
        cursor.byte_pos = 0;
        cursor.initialized = true;
    }
    // Skip forward past any empty rows before starting -- same
    // block4_decay_TODO_DELETE.hpp precedent (handles the cursor landing
    // on a row with row_nnz==0, e.g. right after a wrap to row 0 if row
    // 0 itself happens to be empty).
    while (cursor.row < n_rows && BL.row_nnz(cursor.row) == 0) {
        ++cursor.row;
        if (cursor.row < n_rows) {
            cursor.elem_pos = BL.elem_start[cursor.row];
            cursor.byte_pos = store.tile_byte_start[cursor.row];
        }
    }
    if (cursor.row >= n_rows) {
        cursor.row = 0;
        cursor.elem_pos = 0;
        cursor.byte_pos = 0;
        while (cursor.row < n_rows && BL.row_nnz(cursor.row) == 0)
            ++cursor.row;
        if (cursor.row >= n_rows) {
            out.cycle_complete = true;
            return out;
        }
    }

    bool cycle_complete = false;
    for (std::size_t touched = 0; touched < chunk_size; ++touched) {
        auto row_cur = store.row_cursor(cursor.row);
        // Advance the row's own column cursor up to this elem_pos's
        // position within the row -- re-created fresh each tile touch
        // rather than persisted, same convention block4_decay_TODO_DELETE.hpp
        // already established.
        const std::size_t row_elem_start = BL.elem_start[cursor.row];
        uint32_t bc = 0;
        for (std::size_t pos = row_elem_start; pos <= cursor.elem_pos; ++pos)
            bc = row_cur.advance();

        {
            Block4TileHandle32 handle =
                store.at_index(uint32_t(cursor.row), bc, cursor.elem_pos, cursor.byte_pos);
            const std::size_t n_in_this_row = n_rows * BLOCK4_TILE; // fan-in scale reference
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const float w = handle.get_weight(li, lj);
                    const float imp = handle.get_importance(li, lj);
                    if (w == 0.0f && imp == 0.0f)
                        continue; // not a live cell

                    const std::size_t j = static_cast<std::size_t>(bc) * BLOCK4_TILE + lj;
                    if (j >= n_out)
                        continue; // padding column beyond the real n_out (last tile may overhang)

                    state.col_importance[j] = eta * state.col_importance[j] + (1.0f - eta) * imp;

                    float new_w = w;
                    float new_imp = imp;
                    bool cell_written = false;

                    if (state.col_reset_active[j]) {
                        const float strength = blend * state.col_plasticity_boost[j];
                        const float fan_in_scale =
                            1.0f /
                            std::sqrt(static_cast<float>(std::max<std::size_t>(1, n_in_this_row)));
                        const float fresh = fp4_stochastic_normal01() * fan_in_scale;
                        new_w = (1.0f - strength) * w + strength * fresh;
                        new_imp = (1.0f - strength) * imp;
                        cell_written = true;
                    }

                    // L2-saturation-gated decay -- see
                    // delta_csr_types.hpp's plasticity_select_cycle_boundary
                    // and test_amortized_plasticity_reset.cpp's header
                    // comment for the full derivation. Same population-
                    // level, current-state-only signal as the scattered
                    // path; l2_decay_lambda=0.0 (default) is an exact
                    // no-op.
                    if (l2_decay_lambda > 0.0f && state.l2_decay_strength > 0.0f) {
                        new_imp *= (1.0f - l2_decay_lambda * state.l2_decay_strength);
                        cell_written = true;
                    }

                    if (cell_written) {
                        handle.set_weight(li, lj, new_w);
                        handle.set_importance(li, lj, new_imp);
                    }
                }
            }
            // handle destructs here -- commits the dirty sparse tile (if
            // it was sparse) via Block4Store32::commit_dirty_sparse_tile,
            // same path every real weight update already goes through.
        }

        // Advance to the next tile, same byte-offset bookkeeping
        // block4_decay_TODO_DELETE.hpp's own row walk uses.
        cursor.byte_pos += store.tile_len_at(cursor.elem_pos, cursor.byte_pos);
        ++cursor.elem_pos;
        if (cursor.elem_pos >= BL.elem_start[cursor.row] + BL.row_nnz(cursor.row)) {
            ++cursor.row;
            while (cursor.row < n_rows && BL.row_nnz(cursor.row) == 0)
                ++cursor.row;
            if (cursor.row >= n_rows) {
                cycle_complete = true;
                cursor.row = 0;
                while (cursor.row < n_rows && BL.row_nnz(cursor.row) == 0)
                    ++cursor.row;
                if (cursor.row >= n_rows) {
                    cursor.elem_pos = 0;
                    cursor.byte_pos = 0;
                    break;
                }
            }
            cursor.elem_pos = BL.elem_start[cursor.row];
            cursor.byte_pos = store.tile_byte_start[cursor.row];
            if (cycle_complete)
                break;
        }
    }

    out.cycle_complete = cycle_complete;
    if (cycle_complete)
        plasticity_select_cycle_boundary(state, n_out, eta_slow, eta_slow_catchup, eta_fast,
                                         reset_fraction, k, eta_var, blend, out, l2_decay_threshold,
                                         l2_decay_temperature, max_ci, select_by_deviation);
    return out;
}
