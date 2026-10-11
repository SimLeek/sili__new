#pragma once
#include "delta_csr_types.hpp" // pulls in block4.hpp; AmortizedL2InitStats lives here
#include <cmath>
#include <cstddef>

// Block4 mirror of apply_amortized_l2_init -- same "separate, simple,
// single-threaded traversal built only on already-tested Block4
// primitives" convention as block4_decay_TODO_DELETE.hpp (see that
// file's own header comment for why this isn't threaded through the
// real forward/backward kernels). initial_weight/captured are indexed
// by a stable per-tile visit order (tile_index * 16 + li*4+lj) --
// deterministic across cycles as long as the store's own tile layout
// doesn't change mid-run (true for a dense-loaded, non-synaptogenesis
// run, which is what this is built for).

struct Block4L2InitCursor {
    std::size_t row = 0;
    std::size_t elem_pos = 0;
    std::size_t byte_pos = 0;
    std::size_t tile_index = 0;
    bool initialized = false;
};

template <typename V>
inline AmortizedL2InitStats
apply_amortized_block4_l2_init(Block4Store32& store, Block4L2InitCursor& cursor,
                               std::vector<V>& initial_weight, std::vector<uint8_t>& captured,
                               std::size_t chunk_size, V rate) {
    const auto& BL = store.block_layout;
    const std::size_t n_rows = BL.rows;
    const std::size_t n_tiles = store.n_tiles();
    const std::size_t total_cells = n_tiles * BLOCK4_TILE * BLOCK4_TILE;
    if (initial_weight.size() != total_cells) {
        initial_weight.assign(total_cells, V(0));
        captured.assign(total_cells, 0);
    }

    if (n_rows == 0 || n_tiles == 0) {
        AmortizedL2InitStats out;
        out.cycle_complete = true;
        return out;
    }

    if (!cursor.initialized) {
        cursor.row = 0;
        cursor.elem_pos = 0;
        cursor.byte_pos = 0;
        cursor.tile_index = 0;
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
            AmortizedL2InitStats out;
            out.cycle_complete = true;
            return out;
        }
    }

    bool cycle_complete = false;
    std::size_t n_touched = 0;

    for (std::size_t touched = 0; touched < chunk_size; ++touched) {
        auto row_cur = store.row_cursor(cursor.row);
        const std::size_t row_elem_start = BL.elem_start[cursor.row];
        uint32_t bc = 0;
        for (std::size_t k = row_elem_start; k <= cursor.elem_pos; ++k)
            bc = row_cur.advance();

        {
            Block4TileHandle32 handle =
                store.at_index(uint32_t(cursor.row), bc, cursor.elem_pos, cursor.byte_pos);
            const std::size_t tile_base = (cursor.tile_index % n_tiles) * BLOCK4_TILE * BLOCK4_TILE;
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const std::size_t idx = tile_base + li * BLOCK4_TILE + lj;
                    const float w = handle.get_weight(li, lj);
                    if (!captured[idx]) {
                        initial_weight[idx] = static_cast<V>(w);
                        captured[idx] = 1;
                    } else {
                        const V new_w =
                            static_cast<V>(w) + rate * (initial_weight[idx] - static_cast<V>(w));
                        handle.set_weight(li, lj, static_cast<float>(new_w));
                    }
                    ++n_touched;
                }
            }
            // handle destructs here -- commits the dirty sparse tile (if
            // it was sparse), same path every real weight update uses.
        }
        ++cursor.tile_index;

        cursor.byte_pos += store.tile_len_at(cursor.elem_pos, cursor.byte_pos);
        ++cursor.elem_pos;
        if (cursor.elem_pos >= BL.elem_start[cursor.row] + BL.row_nnz(cursor.row)) {
            ++cursor.row;
            while (cursor.row < n_rows && BL.row_nnz(cursor.row) == 0)
                ++cursor.row;
            if (cursor.row >= n_rows) {
                cycle_complete = true;
                cursor.row = 0;
                cursor.tile_index = 0;
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

    AmortizedL2InitStats out;
    out.cycle_complete = cycle_complete;
    out.n_touched = n_touched;
    return out;
}
