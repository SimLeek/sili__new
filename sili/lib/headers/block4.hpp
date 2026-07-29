#pragma once
#include "fp4quant.hpp"
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ── block4: dense 4x4 tiles, integrated into SparseLinearWeightsDelta ──────
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
// This is a 4x4 dense tile, not a generic NxN one -- BLOCK4_TILE_SIZE exists
// only as a compile-time override for re-tuning that one fixed number (see
// TODO_DUAL_BLOCK4.md's settled design decisions), not as a hook for runtime
// or templated generality nothing here actually needs. Don't widen anything
// in this file "to be safe" for a size this isn't and was never asked to be.
#ifndef SILI_BLOCK4_TILE_SIZE
#define SILI_BLOCK4_TILE_SIZE 4
#endif
#ifndef SILI_BLOCK4_PROMOTE_MIN_LIVE
#define SILI_BLOCK4_PROMOTE_MIN_LIVE 2
#endif

constexpr uint32_t BLOCK4_TILE = SILI_BLOCK4_TILE_SIZE;
constexpr uint32_t BLOCK4_TILE_SLOTS = SILI_BLOCK4_TILE_SIZE * SILI_BLOCK4_TILE_SIZE;
constexpr uint32_t BLOCK4_PROMOTE_MIN_LIVE = SILI_BLOCK4_PROMOTE_MIN_LIVE;

// ── block4 4-wide SIMD helpers ───────────────────────────────────────────────
//
// float only -- this codebase's only real VALUES_TYPE::value_type, and
// deliberately not templated: block4 is a concrete 4x4 tile, not a generic
// abstraction (see the header comment above), so this doesn't pretend to
// support a hypothetical double path nothing here actually uses.
//
// GCC/Clang's vector_size extension, not raw target intrinsics: portable
// across whatever SIMD width the build target actually has (SSE/AVX/NEON),
// and it's what actually gets the compiler to emit real SIMD for block4's
// per-column state in disldo_backward -- confirmed via -fopt-info-vec that
// the auto-vectorizer's own SLP pass, working from plain scalar arrays,
// could PROVE the loop vectorizable but rejected it as unprofitable (cost
// model didn't like an array-indexed load it couldn't prove was contiguous).
// An explicit vector type doesn't leave that judgment call to the
// auto-vectorizer's cost heuristics.
using Block4Vec  = float    __attribute__((__vector_size__(SILI_BLOCK4_TILE_SIZE * sizeof(float))));
using Block4VecU = uint32_t __attribute__((__vector_size__(SILI_BLOCK4_TILE_SIZE * sizeof(uint32_t))));
static_assert(sizeof(Block4Vec) == BLOCK4_TILE * sizeof(float), "Block4Vec width must match BLOCK4_TILE");

inline Block4Vec block4_vec_load(const float* p) {
    Block4Vec v;
    std::memcpy(&v, p, sizeof(v));   // unaligned-safe load, no UB regardless of p's alignment
    return v;
}
inline void block4_vec_store(float* p, Block4Vec v) {
    std::memcpy(p, &v, sizeof(v));
}
inline Block4Vec block4_vec_broadcast(float x) {
    Block4Vec v;
    for (uint32_t i = 0; i < BLOCK4_TILE; ++i) v[i] = x;
    return v;
}
inline Block4VecU block4_vecu_broadcast(uint32_t x) {
    Block4VecU v;
    for (uint32_t i = 0; i < BLOCK4_TILE; ++i) v[i] = x;
    return v;
}
// Elementwise |x| via an IEEE-754 sign-bit clear -- std::abs isn't defined
// for GCC vector-extension types, and this avoids a per-lane branch.
inline Block4Vec block4_vec_abs(Block4Vec x) {
    Block4VecU bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits &= block4_vecu_broadcast(0x7FFFFFFFu);
    Block4Vec result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
inline float block4_vec_hsum(Block4Vec x) {
    float s = 0.0f;
    for (uint32_t i = 0; i < BLOCK4_TILE; ++i) s += x[i];
    return s;
}

// One dense tile: BLOCK4_TILE_SLOTS bytes, (imp4<<4|w4) per slot, stored
// [local_j * BLOCK4_TILE + local_i] (row=local_i, col=local_j within the
// tile -- matches disldo's own row=input/col=output CSR orientation, see
// linear_disldo.hpp's own orientation note, so NO transpose is needed
// between block4 and disldo's shared row indexing here, unlike the
// prototype which used the opposite (row=output) convention and needed an
// explicit orientation fix -- done differently and more simply here from
// the start).
//
// No liveness bit, no live_count. Every slot in a promoted tile is a real
// synapse, weight=0.0 included -- that's what "dense" means; a value-based
// liveness signal (data[slot] == 0) is a sparse-domain idea that doesn't
// belong here, and turned into a real bug the first time it was tried: a
// genuinely-live synapse whose weight AND importance both round to
// FP4_TABLE's zero entry (common -- 0.0 is nearest for anything near the
// origin, and a freshly-grown synapse starts at weight=0.0 exactly) was
// indistinguishable from an empty slot, silently dropping it from forward,
// from backward's gradient update (permanently -- it could never train away
// from zero), and from every read path. Forward/backward now process all
// BLOCK4_TILE_SLOTS slots of a live tile unconditionally -- see
// linear_disldo.hpp. (This didn't cost any SIMD either way: confirmed via
// -fopt-info-vec that these loops don't auto-vectorize regardless, blocked
// by the FP4_TABLE gather and the get_value_scale/get_output_scale calls,
// not by any liveness branch.)
//
// count_live() is a cold-path, on-demand O(16) byte scan (weight nibble
// nonzero), used only by promotion/demotion/reporting to decide whether a
// tile still holds enough real data to justify staying block4 -- not a
// structural oracle, just a cheap heuristic. A slot whose value happens to
// be exactly (0.0 weight, 0.0 importance) at the moment of that scan reads
// as "not live" here, same as before -- but the consequence is now minor
// (it just doesn't count toward this cycle's demotion/reporting tally,
// nothing more) instead of being permanently unreachable, since forward and
// backward no longer consult this at all.
struct Block4Tile {
    uint8_t data[BLOCK4_TILE_SLOTS] = {0};

    static uint32_t slot_index(uint32_t local_i, uint32_t local_j) { return local_j * BLOCK4_TILE + local_i; }

    uint8_t& at(uint32_t local_i, uint32_t local_j) { return data[slot_index(local_i, local_j)]; }
    uint8_t  at(uint32_t local_i, uint32_t local_j) const { return data[slot_index(local_i, local_j)]; }

    uint32_t count_live() const {
        // Whole byte, not just the weight nibble: a freshly-grown synapse
        // starts at weight=0.0 by convention (see delta_csr_memory.hpp's
        // Step 6) with only a nonzero importance -- checking the weight
        // nibble alone would miss it here the same way it did before this
        // was ever tracked as a separate liveness bit.
        uint32_t n = 0;
        for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i)
            if (data[i] != 0) ++n;
        return n;
    }
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

    // Reverse index: block-row -> set of block-columns with a live tile
    // there. delta_csr_synap_row_step's pruning path is triggered per CSR
    // ROW, and needs to know "does this row's block-row participate in any
    // block4 tile" -- without this, that check would be an O(n_tiles) scan
    // of the whole store on EVERY row-step call (the function is meant to
    // be cheap and called very frequently, one row per call). Kept in sync
    // by get_or_create/erase below -- never mutate `tiles` directly.
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> by_block_row;

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
        const uint64_t k = key(br, bc);
        auto it = tiles.find(k);
        if (it != tiles.end()) return it->second;
        by_block_row[br].insert(bc);
        return tiles[k];
    }
    void erase(uint32_t br, uint32_t bc) {
        tiles.erase(key(br, bc));
        auto it = by_block_row.find(br);
        if (it != by_block_row.end()) {
            it->second.erase(bc);
            if (it->second.empty()) by_block_row.erase(it);
        }
    }
    std::size_t n_tiles() const { return tiles.size(); }
    // Cold-path reporting only (nnz(), diagnostics) -- O(n_tiles * 16), not
    // called from forward/backward.
    std::size_t live_synapses() const {
        std::size_t n = 0;
        for (auto& kv : tiles) n += kv.second.count_live();
        return n;
    }
};
