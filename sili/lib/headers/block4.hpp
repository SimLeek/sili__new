#pragma once
#include "fp4quant.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

// ── block4: dense NxN tiles, integrated into SparseLinearWeightsDelta ──────
//
// Production version of the prototype validated on feature/sili-ell-benchmark
// (PR #22, kept as reference, not merged): a real speedup (2-13x on the real
// MiniCPM5-1B-Base checkpoint, scaling with density) with verified no quality
// cost vs. disldo's own FP4 loss, from storing the LOCALLY-DENSE part of a
// sparse matrix as small dense tiles (no per-synapse gather -- position IS
// the column once a tile is active) alongside the existing scattered
// ULEB128-delta CSR (disldo) for everything else.
//
// Differences from the prototype, both real fixes, not stylistic:
//   - Shares the OWNING SparseLinearWeightsDelta's existing per-row
//     value_scale/importance_scale (not a separate scale array) -- the
//     prototype's own leftover path needed a late fix for exactly this
//     (a real 2.07x quality regression on one real tensor, traced to a
//     duplicated/unsynced scale). One authoritative scale per row,
//     regardless of which representation currently holds a given synapse,
//     makes promotion/demotion a lossless BYTE copy (re-quantizing an
//     already-exact FP4_TABLE value via fp4_quantize() round-trips exactly,
//     no requantization error introduced by moving between representations).
//   - Tile presence is a hash map (br,bc) -> tile, not a full block-CSR
//     rebuild -- needed for O(1)-ish promotion/demotion of a SINGLE tile at
//     a time (see delta_csr_memory.hpp's synap_row_step hook), not a
//     whole-matrix batch split the way the prototype's split_for_block4 was.
//
// Compile-time constants, overridable at build time (both setup.py's
// extra_compile_args and CMakeLists.txt need `-DSILI_BLOCK4_TILE_SIZE=N`/
// `-DSILI_BLOCK4_PROMOTE_MIN_LIVE=N` wired through for this to actually be
// build-configurable, not just in-principle). Real ceiling for TILE_SIZE is
// hardware-specific -- see prototypes/sili_ell/BLOCK4_NOTES.md: this
// machine's real SIMD width is ~4 (AVX2 runs double-pumped on this CPU
// generation), 2x2 is a measured REGRESSION not just a non-improvement.
// PROMOTE_MIN_LIVE=2 for TILE_SIZE=4 matches ceil(0.10 * 16) = 2, the
// measured real breakeven on this machine (also hardware/data-distribution
// specific -- re-verify before trusting a different default elsewhere).
#ifndef SILI_BLOCK4_TILE_SIZE
#define SILI_BLOCK4_TILE_SIZE 4
#endif
#ifndef SILI_BLOCK4_PROMOTE_MIN_LIVE
#define SILI_BLOCK4_PROMOTE_MIN_LIVE 2
#endif

constexpr uint32_t BLOCK4_TILE = SILI_BLOCK4_TILE_SIZE;
constexpr uint32_t BLOCK4_TILE_SLOTS = SILI_BLOCK4_TILE_SIZE * SILI_BLOCK4_TILE_SIZE;
constexpr uint32_t BLOCK4_PROMOTE_MIN_LIVE = SILI_BLOCK4_PROMOTE_MIN_LIVE;

// One dense tile: BLOCK4_TILE_SLOTS bytes, (imp4<<4|w4) per slot, stored
// [local_j * BLOCK4_TILE + local_i] (row=local_i, col=local_j within the
// tile -- matches disldo's own row=input/col=output CSR orientation, see
// linear_disldo.hpp's own orientation note, so NO transpose is needed
// between block4 and disldo's shared row indexing here, unlike the
// prototype which used the opposite (row=output) convention and needed an
// explicit orientation fix -- done differently and more simply here from
// the start).
struct Block4Tile {
    uint8_t data[BLOCK4_TILE_SLOTS] = {0};
    uint32_t live_count = 0;

    uint8_t& at(uint32_t local_i, uint32_t local_j) { return data[local_j * BLOCK4_TILE + local_i]; }
    uint8_t  at(uint32_t local_i, uint32_t local_j) const { return data[local_j * BLOCK4_TILE + local_i]; }
};

// Tiles are keyed by (block_row, block_col) = (row/BLOCK4_TILE, col/BLOCK4_TILE)
// in the SAME row/col space as the owning DeltaCSRWeights (row=input,
// col=output, disldo's convention). A hash map, not a dense array: real
// layers are still globally sparse even after locally-dense tiles are
// promoted (e.g. embed_tokens' real 20% density leaves most possible tile
// coordinates empty) -- a dense presence array would waste memory
// proportional to n_block_rows*n_block_cols, not to actual tile count.
struct Block4Store {
    std::unordered_map<uint64_t, Block4Tile> tiles;

    static uint64_t key(uint32_t br, uint32_t bc) {
        return (uint64_t(br) << 32) | uint64_t(bc);
    }
    Block4Tile* find(uint32_t br, uint32_t bc) {
        auto it = tiles.find(key(br, bc));
        return it != tiles.end() ? &it->second : nullptr;
    }
    const Block4Tile* find(uint32_t br, uint32_t bc) const {
        auto it = tiles.find(key(br, bc));
        return it != tiles.end() ? &it->second : nullptr;
    }
    Block4Tile& get_or_create(uint32_t br, uint32_t bc) {
        return tiles[key(br, bc)];
    }
    void erase(uint32_t br, uint32_t bc) {
        tiles.erase(key(br, bc));
    }
    std::size_t n_tiles() const { return tiles.size(); }
    std::size_t live_synapses() const {
        std::size_t n = 0;
        for (auto& kv : tiles) n += kv.second.live_count;
        return n;
    }
};
