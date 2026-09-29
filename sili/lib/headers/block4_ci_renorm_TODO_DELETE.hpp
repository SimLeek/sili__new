#pragma once
#include "delta_csr_types.hpp" // pulls in block4.hpp; CiRenormState/CiRenormStats live here

// TODO: DELETE ME LATER. Same rationale as block4_decay_TODO_DELETE.hpp/
// block4_plasticity_TODO_DELETE.hpp (this file's siblings): the block4
// forward/backward kernels are not yet decomposed into a reusable "walk
// every populated tile" + pluggable per-cell operation, so this is a
// SEPARATE, deliberately simple (single-threaded, no SIMD) traversal
// built ONLY on already-tested primitives (Block4Store32::row_cursor/
// tile_len_at/at_index, Block4TileHandle32::get_weight/get_importance/
// set_importance). Once the block4 kernels ARE split into composable
// functions with a pluggable per-cell callback, this file should be
// deleted and CiRenorm should become a callback swap into that shared
// traversal.
//
// See docs/research/delta_csr_types.rst:ci_renorm.design_and_v3_lesson
// for the full mechanism derivation. chunk_size here counts TILES
// (block4's natural atomic unit, matching the decay/plasticity work's
// own convention), not individual cells -- each touched tile yields up
// to 16 per-cell corrections.
//
// This traversal's own cursor uses the SAME pattern as
// block4_plasticity_TODO_DELETE.hpp's Block4PlasticityCursor, which
// does NOT have the scattered-side skip_empty_rows bug flagged in
// delta_csr_types.hpp (its elem_pos/byte_pos reassignment already lives
// inside the while loop, only firing on an actual row advance) -- kept
// that way here deliberately, not by luck.

struct Block4CiRenormCursor {
    std::size_t row = 0;
    std::size_t elem_pos = 0;
    std::size_t byte_pos = 0;
    bool initialized = false;
};

inline CiRenormStats apply_amortized_block4_ci_renorm_step(
    Block4Store32& store, std::size_t n_out, CiRenormState& state, Block4CiRenormCursor& cursor,
    std::size_t chunk_size, CiRenormMode mode, const float* target_mean, const float* target_std,
    float eff_lr, float max_ci_ref = 100.0f, float trust_ratio_min = 0.1f,
    float trust_ratio_max = 10.0f) {
    state.ensure_sized(n_out);
    const auto& BL = store.block_layout;
    const std::size_t n_rows = BL.rows;
    CiRenormStats out;

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
    // Skip forward past any empty rows -- elem_pos/byte_pos only
    // reassigned INSIDE the while loop, i.e. only when row actually
    // advances (not the scattered-side bug: see
    // delta_csr_types.hpp's apply_amortized_ci_renorm_step comment).
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
        const std::size_t row_elem_start = BL.elem_start[cursor.row];
        uint32_t bc = 0;
        for (std::size_t pos = row_elem_start; pos <= cursor.elem_pos; ++pos)
            bc = row_cur.advance();

        {
            Block4TileHandle32 handle =
                store.at_index(uint32_t(cursor.row), bc, cursor.elem_pos, cursor.byte_pos);
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const float w = handle.get_weight(li, lj);
                    const float imp = handle.get_importance(li, lj);
                    if (w == 0.0f && imp == 0.0f)
                        continue; // not a live cell

                    const std::size_t j = static_cast<std::size_t>(bc) * BLOCK4_TILE + lj;
                    if (j >= n_out)
                        continue; // padding column beyond the real n_out

                    const float new_imp = ci_renorm_touch_cell(
                        state, j, imp, w, mode, target_mean ? target_mean[j] : 0.0f,
                        target_std ? target_std[j] : 0.0f, eff_lr, max_ci_ref, trust_ratio_min,
                        trust_ratio_max);
                    if (new_imp != imp)
                        handle.set_importance(li, lj, new_imp);
                }
            }
            // handle destructs here -- commits the dirty sparse tile (if
            // it was sparse), same path every real weight update goes
            // through.
        }

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

    if (cycle_complete)
        ci_renorm_cycle_boundary(state, n_out, out);
    else
        out.cycle_complete = false;
    return out;
}

// WeightRenorm block4 half -- see synapse_policy.weight_renorm and
// apply_amortized_weight_renorm_step (delta_csr_types.hpp) for the full
// derivation. Reuses Block4CiRenormCursor (cursor shape is identical --
// only which field gets touched differs). clamp_nonnegative=false
// (weight is signed); no eff_lr/max_ci_ref/trust_ratio_* -- only
// StableRegion makes sense for weight (TrustRatio's own formula is
// derived FROM weight norm, so applying it to weight itself is
// circular).
inline CiRenormStats apply_amortized_block4_weight_renorm_step(
    Block4Store32& store, std::size_t n_out, CiRenormState& state, Block4CiRenormCursor& cursor,
    std::size_t chunk_size, CiRenormMode mode, const float* target_mean, const float* target_std) {
    state.ensure_sized(n_out);
    const auto& BL = store.block_layout;
    const std::size_t n_rows = BL.rows;
    CiRenormStats out;

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
        const std::size_t row_elem_start = BL.elem_start[cursor.row];
        uint32_t bc = 0;
        for (std::size_t pos = row_elem_start; pos <= cursor.elem_pos; ++pos)
            bc = row_cur.advance();

        {
            Block4TileHandle32 handle =
                store.at_index(uint32_t(cursor.row), bc, cursor.elem_pos, cursor.byte_pos);
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const float w = handle.get_weight(li, lj);
                    const float imp = handle.get_importance(li, lj);
                    if (w == 0.0f && imp == 0.0f)
                        continue; // not a live cell

                    const std::size_t j = static_cast<std::size_t>(bc) * BLOCK4_TILE + lj;
                    if (j >= n_out)
                        continue; // padding column beyond the real n_out

                    const float new_w = ci_renorm_touch_cell(
                        state, j, w, 0.0f, mode, target_mean ? target_mean[j] : 0.0f,
                        target_std ? target_std[j] : 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false);
                    if (new_w != w)
                        handle.set_weight(li, lj, new_w);
                }
            }
        }

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

    if (cycle_complete)
        ci_renorm_cycle_boundary(state, n_out, out);
    else
        out.cycle_complete = false;
    return out;
}
