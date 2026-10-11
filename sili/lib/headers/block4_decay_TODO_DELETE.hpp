#pragma once
#include "delta_csr_types.hpp" // pulls in block4.hpp; AmortizedDecayStats lives here
#include <cmath>
#include <cstddef>

// TODO: DELETE ME LATER. This file exists because the block4 forward/
// backward kernels (linear_disldo_forward.hpp/linear_disldo_backward.hpp)
// are not yet decomposed into a reusable "walk every populated tile" +
// pluggable per-cell operation -- the row/tile traversal and the actual
// arithmetic are entangled in one large, thread-partitioned, SIMD-tuned
// function. Rather than either (a) guessing at a new traversal and
// risking getting the sparse/dense tile duality wrong, or (b) blocking
// this feature on refactoring that kernel first, this is a SEPARATE,
// deliberately simple (single-threaded, no SIMD) traversal built ONLY on
// already-tested primitives (Block4Store32::row_cursor/tile_len_at/
// at_index, Block4TileHandle32::get_weight/set_weight/get_importance/
// set_importance) -- it does NOT re-derive sparse/dense tile packing
// itself; Block4TileHandle32 already handles that correctly (unpacks a
// sparse tile into a scratch buffer on construction, repacks via
// Block4Store32::commit_dirty_sparse_tile on destruction if dirty, same
// path every real weight update already goes through). Once the block4
// kernels ARE split into composable functions with a pluggable per-cell
// callback, this file should be deleted and decay should become a
// simple callback swap into that shared traversal instead of its own
// copy.
//
// Only fp32 (Block4Store32/Block4TileHandle32) -- see
// docs/research/delta_csr_types.rst:amortized_decay.chunked_cursor's
// decay_importance update for why (this project's actually-used
// precision throughout the investigation that needed this).

struct Block4DecayCursor {
    std::size_t row = 0;
    std::size_t elem_pos = 0;
    std::size_t byte_pos = 0;
    bool initialized = false;
};

// Same amortized/chunked-cursor shape and AmortizedDecayStats return type
// as apply_amortized_decay_stats/apply_amortized_flat_decay_stats
// (delta_csr_types.hpp), for interface consistency -- touches chunk_size
// TILES per call (not individual weights: a tile is block4's natural
// atomic unit, touching one is roughly constant cost regardless of how
// many of its 16 cells are populated). decay_importance selects which
// channel decays; the other passes through unchanged, same convention as
// the scattered/flat-array versions. Floors an exactly-zero result away
// from 0 (copysign a tiny epsilon) -- NOT required for correctness here
// (Block4TileHandle32/commit_dirty_sparse_tile already handle a tile's
// live-count and packed length changing safely), but avoids silently
// changing which cells count as "live" as a side effect of decay alone,
// which should stay a deliberate pruning decision, not an accident of a
// mostly-unrelated feature.
inline AmortizedDecayStats
apply_amortized_block4_decay_stats(Block4Store32& store, Block4DecayCursor& cursor, double& sum_abs,
                                   double& sum_sq, double& max_abs, std::size_t& n,
                                   std::size_t chunk_size, float decay_factor,
                                   bool decay_importance) {
    const auto& BL = store.block_layout;
    const std::size_t n_rows = BL.rows;
    bool cycle_complete = false;

    if (n_rows == 0 || store.n_tiles() == 0) {
        AmortizedDecayStats out;
        out.cycle_complete = true;
        return out;
    }

    if (!cursor.initialized) {
        cursor.row = 0;
        cursor.elem_pos = 0;
        cursor.byte_pos = 0;
        cursor.initialized = true;
    }
    // Skip forward past any empty rows before starting (handles the
    // cursor landing on a row with row_nnz==0, e.g. right after a wrap
    // to row 0 if row 0 itself happens to be empty).
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
            AmortizedDecayStats out;
            out.cycle_complete = true;
            return out;
        }
    }

    for (std::size_t touched = 0; touched < chunk_size; ++touched) {
        auto row_cur = store.row_cursor(cursor.row);
        // Advance the row's own column cursor up to this elem_pos's
        // position within the row (DeltaCSRRowCursor only walks forward
        // from its own start, same constraint raw_find works within --
        // re-created fresh each tile touch rather than persisted, since
        // Block4DecayCursor only stores elem_pos/byte_pos, not cursor
        // state; cheap relative to the tile decode/encode work itself).
        const std::size_t row_elem_start = BL.elem_start[cursor.row];
        uint32_t bc = 0;
        for (std::size_t k = row_elem_start; k <= cursor.elem_pos; ++k)
            bc = row_cur.advance();

        {
            Block4TileHandle32 handle =
                store.at_index(uint32_t(cursor.row), bc, cursor.elem_pos, cursor.byte_pos);
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const float w = handle.get_weight(li, lj);
                    const float imp = handle.get_importance(li, lj);
                    if (w == 0.0f && imp == 0.0f)
                        continue; // not a live cell, nothing to decay
                    if (decay_importance) {
                        float new_imp = imp * decay_factor;
                        if (new_imp == 0.0f && imp != 0.0f)
                            new_imp = std::copysign(std::numeric_limits<float>::min(), imp);
                        handle.set_importance(li, lj, new_imp);
                        const double av = std::abs(double(new_imp));
                        sum_abs += av;
                        sum_sq += av * av;
                        if (av > max_abs)
                            max_abs = av;
                    } else {
                        float new_w = w * decay_factor;
                        if (new_w == 0.0f && w != 0.0f)
                            new_w = std::copysign(std::numeric_limits<float>::min(), w);
                        handle.set_weight(li, lj, new_w);
                        const double av = std::abs(double(new_w));
                        sum_abs += av;
                        sum_sq += av * av;
                        if (av > max_abs)
                            max_abs = av;
                    }
                    ++n;
                }
            }
            // handle destructs here -- commits the dirty sparse tile (if
            // it was sparse) via Block4Store32::commit_dirty_sparse_tile,
            // same path every real weight update already goes through.
        }

        // Advance to the next tile, same byte-offset bookkeeping
        // disldo_forward's own row walk uses (tile_len_at BEFORE moving
        // past this tile, since it needs THIS tile's own stored length).
        cursor.byte_pos += store.tile_len_at(cursor.elem_pos, cursor.byte_pos);
        ++cursor.elem_pos;
        if (cursor.elem_pos >= BL.elem_start[cursor.row] + BL.row_nnz(cursor.row)) {
            // Move to the next non-empty row.
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

    AmortizedDecayStats out;
    out.cycle_complete = cycle_complete;
    if (cycle_complete && n > 0) {
        out.mean_abs = sum_abs / double(n);
        out.rms = std::sqrt(sum_sq / double(n));
        out.max_abs = max_abs;
        out.n = n;
        sum_abs = 0.0;
        sum_sq = 0.0;
        max_abs = 0.0;
        n = 0;
    }
    return out;
}
