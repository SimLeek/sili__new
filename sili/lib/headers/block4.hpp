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
// Hardcoded to 4 elements, not a BLOCK4_TILE-driven loop -- a real,
// measured bug found via callgrind: a runtime-indexed loop over a
// vector-extension type's lanes (`for (i...) v[i]=x`) forces GCC to treat
// the whole vector as memory instead of a register, compiling to a real
// scalar loop, NOT a single broadcast instruction -- 7.6% of all
// instructions in a profiled disldo_backward run were this "vectorized"
// broadcast helper alone. A 4-element brace-init list is a genuine single
// broadcast op; this file already commits to exactly 4 (see the header
// comment -- BLOCK4_TILE_SIZE is a compile-time override knob for
// re-tuning, not a hook for a generic-width loop nothing here needs).
inline Block4Vec block4_vec_broadcast(float x) {
    return Block4Vec{x, x, x, x};
}
inline Block4VecU block4_vecu_broadcast(uint32_t x) {
    return Block4VecU{x, x, x, x};
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
// Hardcoded 4-term sum, not a BLOCK4_TILE-driven loop -- same real,
// measured bug as block4_vec_broadcast above (runtime-indexed access
// forces scalar memory traffic instead of a real horizontal-add): this
// one function alone was 12.16% of all instructions in a profiled
// disldo_backward run. Constant-index element extraction (x[0] etc, index
// known at compile time) is what actually lets GCC treat x as a register
// and emit a real extract+add sequence instead of a loop.
inline float block4_vec_hsum(Block4Vec x) {
    return x[0] + x[1] + x[2] + x[3];
}
// 4-wide fp4_decode_bits (fp4quant.hpp) -- decodes BLOCK4_TILE codes in one
// shot instead of BLOCK4_TILE separate FP4_TABLE[code] lookups. Same
// branchless bit formula as the scalar version (see fp4quant.hpp's header
// comment on why it's exact, not an approximation), with the two branches
// combined via a compare-mask blend instead of an if/else -- GCC/Clang
// vector-extension comparison operators already return an all-ones/all-
// zero mask per lane, so `(a & mask) | (b & ~mask)` is the whole blend,
// no scalar branching anywhere. Verified bit-exact against
// fp4_decode_bits() (equivalently FP4_TABLE) for all 65536 4-tuples of
// codes -- see test_fp4_bitshift.cpp.
inline Block4Vec block4_vec_decode_fp4(Block4VecU codes) {
    const Block4VecU one_u  = block4_vecu_broadcast(1u);
    const Block4VecU zero_u = block4_vecu_broadcast(0u);
    const Block4VecU s = (codes >> 3) & one_u;
    const Block4VecU e = (codes >> 1) & block4_vecu_broadcast(3u);
    const Block4VecU m = codes & one_u;

    const Block4VecU bits_normal =
        (s << 31) | ((e + block4_vecu_broadcast(126u)) << 23) | (m << 22);

    const Block4VecU half_bits = block4_vecu_broadcast(0x3F000000u) | (s << 31);
    const Block4VecU nan_bits  = block4_vecu_broadcast(0x7FC00000u);
    const Block4VecU s_mask    = (s != zero_u);
    const Block4VecU bits_m0   = (nan_bits & s_mask) | (zero_u & ~s_mask);
    const Block4VecU m_mask    = (m != zero_u);
    const Block4VecU bits_special = (half_bits & m_mask) | (bits_m0 & ~m_mask);

    const Block4VecU e_mask = (e != zero_u);
    const Block4VecU bits = (bits_normal & e_mask) | (bits_special & ~e_mask);

    Block4Vec result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// 4-wide fp4_quantize_stochastic (fp4quant.hpp) -- stochastically quantizes
// BLOCK4_TILE values in one shot instead of BLOCK4_TILE separate scalar
// calls. Same dithered-rounding formula as the scalar version (see
// fp4quant.hpp's header comment for why the |v|>=1.0 branch's bit-add
// dithering is exact, not approximate), branches combined via
// compare-mask blends like block4_vec_decode_fp4 above. The 4 per-lane
// RNG draws stay genuinely scalar (fp4_stochastic_next_u64() x4) -- the
// SIMD win here is in the branch/arithmetic, not the RNG itself, which is
// a handful of xorshift ops regardless of lane count. Verified against
// fp4_quantize_stochastic() via matched-seed statistical (mean-converges-
// to-v) and saturation/exact-value checks -- see test_fp4_stochastic.cpp.
inline Block4VecU block4_vec_quantize_stochastic_fp4(Block4Vec v) {
    static constexpr uint32_t HALF_BITS = 0x3F000000u;  // bits_of(0.5f)
    static constexpr uint32_t ONE_BITS  = 0x3F800000u;  // bits_of(1.0f)
    static constexpr uint32_t SIX_BITS  = 0x40C00000u;  // bits_of(6.0f)

    Block4VecU bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const Block4VecU sign  = bits & block4_vecu_broadcast(0x80000000u);
    const Block4VecU abits = bits & block4_vecu_broadcast(0x7FFFFFFFu);

    const uint64_t r0 = fp4_stochastic_next_u64();
    const uint64_t r1 = fp4_stochastic_next_u64();
    const uint64_t r2 = fp4_stochastic_next_u64();
    const uint64_t r3 = fp4_stochastic_next_u64();
    const Block4Vec uniform01 = {
        float((r0 >> 40) * (1.0 / 16777216.0)), float((r1 >> 40) * (1.0 / 16777216.0)),
        float((r2 >> 40) * (1.0 / 16777216.0)), float((r3 >> 40) * (1.0 / 16777216.0))};
    const Block4VecU dither22 = {uint32_t(r0 & 0x3FFFFFu), uint32_t(r1 & 0x3FFFFFu),
                                  uint32_t(r2 & 0x3FFFFFu), uint32_t(r3 & 0x3FFFFFu)};

    Block4Vec av;  // abits reinterpreted as float -- always >= 0, a valid magnitude
    std::memcpy(&av, &abits, sizeof(av));
    const Block4Vec two_v = block4_vec_broadcast(2.0f);
    const Block4Vec p_low = av * two_v;                              // (v-0.0)/(0.5-0.0)
    const Block4Vec p_mid = av * two_v - block4_vec_broadcast(1.0f);  // (v-0.5)/(1.0-0.5)

    Block4VecU up_low_mask, up_mid_mask;
    {
        const auto cmp_low = (uniform01 < p_low);
        const auto cmp_mid = (uniform01 < p_mid);
        std::memcpy(&up_low_mask, &cmp_low, sizeof(up_low_mask));
        std::memcpy(&up_mid_mask, &cmp_mid, sizeof(up_mid_mask));
    }

    const Block4VecU sat_mask = (abits >= block4_vecu_broadcast(SIX_BITS));
    const Block4VecU low_mask = (abits < block4_vecu_broadcast(HALF_BITS));
    const Block4VecU mid_mask = ~sat_mask & ~low_mask & (abits < block4_vecu_broadcast(ONE_BITS));
    const Block4VecU norm_mask = ~sat_mask & ~low_mask & ~mid_mask;

    const Block4VecU mag_sat = block4_vecu_broadcast(7u);
    const Block4VecU mag_low = up_low_mask & block4_vecu_broadcast(1u);  // 1 if round up else 0
    const Block4VecU mag_mid = (up_mid_mask & block4_vecu_broadcast(2u)) | (~up_mid_mask & block4_vecu_broadcast(1u));

    Block4VecU rounded = abits + dither22;
    const Block4VecU six_mask = (rounded > block4_vecu_broadcast(SIX_BITS));
    rounded = (rounded & ~six_mask) | (block4_vecu_broadcast(SIX_BITS) & six_mask);
    const Block4VecU exp_field = (rounded >> 23) & block4_vecu_broadcast(0xFFu);
    const Block4VecU m_bit     = (rounded >> 22) & block4_vecu_broadcast(1u);
    const Block4VecU mag_norm  = ((exp_field - block4_vecu_broadcast(126u)) << 1) | m_bit;

    Block4VecU mag_code =
        (mag_sat & sat_mask) | (mag_low & low_mask) | (mag_mid & mid_mask) | (mag_norm & norm_mask);

    // Never the repurposed NaN slot (8) for a genuinely near-zero lane --
    // same rule as fp4_encode_bits/fp4_quantize_stochastic.
    const Block4VecU zero_mag_mask = (mag_code == block4_vecu_broadcast(0u));
    // sign is bit 31 (0x80000000) or 0; >>28 moves it to bit 3 (0x8 or 0),
    // exactly the code's sign-bit position -- no separate mask needed.
    const Block4VecU sign_bit = sign >> 28;
    return (sign_bit | mag_code) & ~zero_mag_mask;
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
