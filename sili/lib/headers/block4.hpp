#pragma once
#include "fp4quant.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
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

// ── Tile index: ULEB128 delta-CSR over BLOCK coordinates ────────────────────
//
// Previously an unordered_map<uint64_t, Block4Tile> (64-bit key -- 128 tile
// + 64 key = 192 bits/tile) + a separate unordered_map<uint32_t,
// unordered_set<uint32_t>> reverse index for the per-block-row liveness
// check. Per direction: reuse the SAME ULEB128 delta-encoding the scattered
// path already uses for its own column indices (confirmed this session:
// 100% single-byte at realistic density), just at BLOCK granularity
// (block_row = row/BLOCK4_TILE, block_col = col/BLOCK4_TILE) -- 128 tile +
// 8-16 uleb128 index = 136-144 bits/tile, and the per-block-row liveness
// check becomes "row_nnz(br) > 0" on this same layout instead of a second
// data structure to keep in sync.
//
// DeltaCSRLayout and DeltaCSRRowCursor (delta_csr_types.hpp, just above --
// this file is now included AFTER both) are reused directly, unchanged --
// neither ever referenced VALUES_TYPE/ValueAccessor, so nothing about them
// needed to change for a Block4Tile "value" instead of a float pair.
//
// block4_row_insert_tile/remove_tile below are NOT calls to the existing
// delta_csr_row_insert_col/remove_col (delta_csr_memory.hpp) -- those are
// shaped around ValueAccessor<VALUES_TYPE>'s weight+importance FLOAT PAIR
// interface (`insert_col(..., value_type weight, value_type importance)`),
// and a Block4Tile is one opaque 128-bit blob, not a (weight, importance)
// pair -- forcing it through that signature would be a worse fit than a
// parallel set of functions with the same shifting algorithm.

// Grows row `row`'s byte/elem allocation to at least the given targets,
// for any row EXCEPT the last -- mirrors delta_csr_shift_row's exact
// algorithm (byte_start/byte_end/elem_start/elem_end bookkeeping,
// memmove-based shift of every row after this one) but operates on
// `values` (std::vector<Block4Tile>) directly, not through ValueAccessor.
// Deliberately does nothing for the LAST row (row+1 == L.rows) -- same
// real subtlety as delta_csr_shift_row: there's no "next row" to shift out
// of the way, so growing the last row needs a different, simpler path
// (see block4_grow_last_row below) -- this isn't a bug carried over
// blindly, it's the same shape as the scattered path's own
// equalize_to_capacity, which special-cases the last row for exactly this
// reason.
inline void block4_row_shift(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4Tile>& values,
    std::size_t row,
    std::size_t target_byte_alloc,
    std::size_t target_elem_alloc)
{
    const std::size_t cur_byte_alloc = L.row_alloc_bytes(row);
    if (cur_byte_alloc != target_byte_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = L.byte_start[row + 1];
        const std::size_t move_len  = L.byte_start[L.rows] - move_src;
        const std::size_t new_start = L.byte_start[row] + target_byte_alloc;

        if (target_byte_alloc > cur_byte_alloc)
            ibuf.resize(ibuf.size() + (target_byte_alloc - cur_byte_alloc));
        if (move_len > 0)
            std::memmove(ibuf.data() + new_start, ibuf.data() + move_src, move_len);
        if (target_byte_alloc < cur_byte_alloc)
            ibuf.resize(ibuf.size() - (cur_byte_alloc - target_byte_alloc));

        const std::ptrdiff_t byte_delta =
            std::ptrdiff_t(target_byte_alloc) - std::ptrdiff_t(cur_byte_alloc);
        for (std::size_t r = row + 1; r <= L.rows; ++r)
            L.byte_start[r] = std::size_t(std::ptrdiff_t(L.byte_start[r]) + byte_delta);
        for (std::size_t r = row + 1; r < L.rows; ++r)
            L.byte_end[r] = std::size_t(std::ptrdiff_t(L.byte_end[r]) + byte_delta);
    }

    const std::size_t cur_elem_alloc = L.row_alloc_elems(row);
    if (cur_elem_alloc != target_elem_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = L.elem_start[row + 1];
        const std::size_t move_len  = L.elem_start[L.rows] - move_src;
        const std::size_t new_start = L.elem_start[row] + target_elem_alloc;
        const std::size_t current_total = L.total_alloc_elems();

        if (target_elem_alloc > cur_elem_alloc)
            values.resize(current_total + (target_elem_alloc - cur_elem_alloc));
        if (move_len > 0)
            // Block4Tile is trivially copyable (a plain uint8_t[16]), so a
            // raw memmove here is exactly as safe as std::vector's own
            // internal moves would be -- same reasoning delta_csr_shift_row
            // relies on via ValueAccessor::move for FP4BiPacked.
            std::memmove(values.data() + new_start, values.data() + move_src,
                         move_len * sizeof(Block4Tile));
        if (target_elem_alloc < cur_elem_alloc)
            values.resize(current_total - (cur_elem_alloc - target_elem_alloc));

        const std::ptrdiff_t elem_delta =
            std::ptrdiff_t(target_elem_alloc) - std::ptrdiff_t(cur_elem_alloc);
        for (std::size_t r = row + 1; r <= L.rows; ++r)
            L.elem_start[r] = std::size_t(std::ptrdiff_t(L.elem_start[r]) + elem_delta);
        for (std::size_t r = row + 1; r < L.rows; ++r)
            L.elem_end[r] = std::size_t(std::ptrdiff_t(L.elem_end[r]) + elem_delta);
    }
}

// Last-row growth: no row follows it, so no memmove is needed -- just
// extend the flat buffers and the row's own end marker (L.byte_start[rows]/
// L.elem_start[rows] double as "end of the last row's allocation" since
// there's no row `rows` to have its own start). Mirrors
// SparseLinearLayer::equalize_to_capacity's identical last-row special
// case (cpu_backend.cpp).
inline void block4_grow_last_row(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4Tile>& values,
    std::size_t target_byte_alloc,
    std::size_t target_elem_alloc)
{
    if (L.rows == 0) return;
    const std::size_t r = L.rows - 1;
    const std::size_t cur_b = L.row_alloc_bytes(r);
    const std::size_t cur_e = L.row_alloc_elems(r);
    if (target_byte_alloc > cur_b) {
        ibuf.resize(ibuf.size() + (target_byte_alloc - cur_b), uint8_t(0));
        L.byte_start[L.rows] = L.byte_start[r] + target_byte_alloc;
    }
    if (target_elem_alloc > cur_e) {
        const std::size_t new_total = L.elem_start[r] + target_elem_alloc;
        values.resize(new_total);
        L.elem_start[L.rows] = new_total;
    }
}

// Grows row `row` by exactly enough for ONE more tile (uleb128_max_bytes
// worst case on the byte side, +1 element) -- not amortized/doubling
// growth: block4's own population is inherently small (collision-limited
// via growth, confirmed empirically this session), so a fixed
// exactly-enough-for-one-more increment avoids overreserving without a
// meaningful cost in practice. Dispatches to block4_row_shift or
// block4_grow_last_row depending on whether `row` is the last one.
inline void block4_ensure_row_headroom(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4Tile>& values,
    std::size_t row)
{
    const std::size_t target_b = L.row_alloc_bytes(row) + uleb128_max_bytes<uint32_t>();
    const std::size_t target_e = L.row_alloc_elems(row) + 1;
    if (row + 1 < L.rows)
        block4_row_shift(L, ibuf, values, row, target_b, target_e);
    else
        block4_grow_last_row(L, ibuf, values, target_b, target_e);
}

// Insert `tile` at block-column `new_col` in block-row `row`, sorted order,
// in place. Returns true on success, false if the row has insufficient
// blank space (caller should grow via block4_ensure_row_headroom and
// retry -- Block4Store::get_or_create below does this automatically).
// Mirrors delta_csr_row_insert_col's exact algorithm (delta_csr_memory.hpp)
// -- same uleb128 re-encoding/byte-shift/headroom-check shape -- but writes
// a whole Block4Tile directly instead of going through
// ValueAccessor<VALUES_TYPE>::set(weight, importance).
inline bool block4_row_insert_tile(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4Tile>& values,
    std::size_t row,
    uint32_t new_col,
    const Block4Tile& tile)
{
    const std::size_t n = L.row_nnz(row);

    std::size_t byte_pos      = L.byte_start[row];
    std::size_t elem_pos      = L.elem_start[row];
    uint32_t    prev_col      = 0;
    std::size_t ins_byte_pos  = L.byte_end[row];
    std::size_t ins_elem_pos  = L.elem_end[row];
    bool        has_next      = false;
    uint32_t    next_col      = 0;
    std::size_t next_dlen     = 0;

    for (std::size_t e = 0; e < n; ++e) {
        std::size_t dlen = 0;
        const uint32_t delta = uleb128_decode<uint32_t>(ibuf.data() + byte_pos, dlen);
        const uint32_t col   = prev_col + delta;
        if (col == new_col) return false; // duplicate, skip (tile already exists)
        if (col > new_col) {
            ins_byte_pos = byte_pos;
            ins_elem_pos = elem_pos;
            has_next     = true;
            next_col     = col;
            next_dlen    = dlen;
            break;
        }
        prev_col  = col;
        byte_pos += dlen;
        elem_pos++;
    }

    uint8_t new_d_buf[uleb128_max_bytes<uint32_t>()];
    const std::size_t new_d_len = uleb128_encode<uint32_t>(new_col - prev_col, new_d_buf);

    uint8_t upd_d_buf[uleb128_max_bytes<uint32_t>()];
    std::size_t upd_d_len = 0;
    if (has_next)
        upd_d_len = uleb128_encode<uint32_t>(next_col - new_col, upd_d_buf);

    const std::ptrdiff_t idx_delta =
        std::ptrdiff_t(new_d_len + upd_d_len) - std::ptrdiff_t(next_dlen);

    const std::size_t used_bytes = L.byte_end[row] - L.byte_start[row];
    if (idx_delta > 0 && std::size_t(idx_delta) > L.row_alloc_bytes(row) - used_bytes)
        return false;
    if (L.row_nnz(row) >= L.row_alloc_elems(row))
        return false;

    if (idx_delta != 0) {
        const std::size_t shift_from = ins_byte_pos;
        const std::size_t shift_len  = L.byte_end[row] - shift_from;
        if (shift_len > 0)
            std::memmove(ibuf.data() + shift_from + idx_delta, ibuf.data() + shift_from, shift_len);
        L.byte_end[row] = std::size_t(std::ptrdiff_t(L.byte_end[row]) + idx_delta);
    }

    std::memcpy(ibuf.data() + ins_byte_pos, new_d_buf, new_d_len);
    if (has_next)
        std::memcpy(ibuf.data() + ins_byte_pos + new_d_len, upd_d_buf, upd_d_len);

    if (ins_elem_pos < L.elem_end[row])
        std::memmove(values.data() + ins_elem_pos + 1, values.data() + ins_elem_pos,
                     (L.elem_end[row] - ins_elem_pos) * sizeof(Block4Tile));
    values[ins_elem_pos] = tile;
    L.elem_end[row]++;
    L.total_nnz++;
    return true;
}

// Remove the tile at block-column `target_col` in block-row `row`, if
// present. Mirrors delta_csr_row_remove_col's exact algorithm
// (delta_csr_memory.hpp) -- same delta-merge/byte-shift-left shape -- for
// `std::vector<Block4Tile>` directly.
inline bool block4_row_remove_tile(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4Tile>& values,
    std::size_t row,
    uint32_t target_col)
{
    const std::size_t n = L.row_nnz(row);
    if (n == 0) return false;

    std::size_t byte_pos = L.byte_start[row];
    std::size_t elem_pos = L.elem_start[row];
    uint32_t    prev_col = 0;

    for (std::size_t e = 0; e < n; ++e) {
        std::size_t delta_len = 0;
        const uint32_t delta = uleb128_decode<uint32_t>(ibuf.data() + byte_pos, delta_len);
        const uint32_t col   = prev_col + delta;

        if (col == target_col) {
            const std::size_t next_byte_pos = byte_pos + delta_len;

            if (e + 1 < n) {
                std::size_t next_delta_len = 0;
                const uint32_t next_delta =
                    uleb128_decode<uint32_t>(ibuf.data() + next_byte_pos, next_delta_len);
                const uint32_t merged_delta = delta + next_delta;

                uint8_t merged_buf[uleb128_max_bytes<uint32_t>()];
                const std::size_t merged_len = uleb128_encode<uint32_t>(merged_delta, merged_buf);

                std::memcpy(ibuf.data() + byte_pos, merged_buf, merged_len);

                const std::size_t shift_from = next_byte_pos + next_delta_len;
                const std::size_t shift_len  = L.byte_end[row] - shift_from;
                const std::size_t freed      = delta_len + next_delta_len - merged_len;
                if (shift_len > 0)
                    std::memmove(ibuf.data() + byte_pos + merged_len, ibuf.data() + shift_from, shift_len);
                L.byte_end[row] -= freed;
            } else {
                L.byte_end[row] -= delta_len;
            }

            const std::size_t row_end = L.elem_end[row];
            if (elem_pos + 1 < row_end)
                std::memmove(values.data() + elem_pos, values.data() + elem_pos + 1,
                             (row_end - elem_pos - 1) * sizeof(Block4Tile));
            L.elem_end[row]--;
            L.total_nnz--;
            return true;
        }

        prev_col  = col;
        byte_pos += delta_len;
        elem_pos++;
    }
    return false;
}

struct Block4Store {
    DeltaCSRLayout           block_layout;   // rows/cols are BLOCK-granularity (ceil(n_in/4), ceil(n_out/4))
    std::vector<uint8_t>     indices_buf;    // uleb128-encoded block-col deltas
    std::vector<Block4Tile>  tile_values;    // parallel to block_layout's elem_start/elem_end

    // Sizes an EMPTY store for a layer of n_in x n_out real (not block)
    // dimensions. Zero initial per-row headroom -- growth is lazy, on
    // first insert into a given block-row (see get_or_create), matching
    // the whole point of this redesign (don't waste memory reserving
    // space most block-rows will never use, since block4 population is
    // inherently sparse even after promotion).
    void init(std::size_t n_in, std::size_t n_out) {
        block_layout = DeltaCSRLayout{};
        block_layout.rows = (n_in + BLOCK4_TILE - 1) / BLOCK4_TILE;
        block_layout.cols = (n_out + BLOCK4_TILE - 1) / BLOCK4_TILE;
        block_layout.byte_start.assign(block_layout.rows + 1, 0);
        block_layout.byte_end.assign(block_layout.rows, 0);
        block_layout.elem_start.assign(block_layout.rows + 1, 0);
        block_layout.elem_end.assign(block_layout.rows, 0);
        block_layout.total_nnz = 0;
        indices_buf.clear();
        tile_values.clear();
    }

    DeltaCSRRowCursor<uint32_t> row_cursor(std::size_t br) const {
        return DeltaCSRRowCursor<uint32_t>(indices_buf.data(), block_layout, br);
    }

    Block4Tile* find(uint32_t br, uint32_t bc) {
        if (br >= block_layout.rows) return nullptr;
        auto cur = row_cursor(br);
        const std::size_t n = block_layout.row_nnz(br);
        std::size_t elem_pos = block_layout.elem_start[br];
        for (std::size_t e = 0; e < n; ++e, ++elem_pos) {
            const uint32_t col = cur.advance();
            if (col == bc) return &tile_values[elem_pos];
            if (col > bc) break; // sorted ascending -- can't appear later
        }
        return nullptr;
    }
    const Block4Tile* find(uint32_t br, uint32_t bc) const {
        return const_cast<Block4Store*>(this)->find(br, bc);
    }

    Block4Tile& get_or_create(uint32_t br, uint32_t bc) {
        if (Block4Tile* existing = find(br, bc)) return *existing;
        // A real, easy-to-hit mistake, not hypothetical (caught via ASan
        // testing this session's own ULEB128 tile-indexing redesign): a
        // Block4Store that was never sized via init(n_in, n_out) has
        // block_layout.rows == 0, so block_layout.byte_start/elem_start
        // are both empty -- silently proceeding to
        // block4_row_insert_tile below would index them out of bounds
        // (a real SEGV, not just UB in theory). block4_maybe_promote
        // (delta_csr_memory.hpp) lazily self-inits before ever reaching
        // here; a caller invoking get_or_create() directly (bypassing
        // promotion, e.g. a hand-built test fixture) doesn't have that
        // safety net, since this function has no n_in/n_out to lazily
        // size to on its own -- fail loud instead of corrupting memory.
        if (br >= block_layout.rows)
            throw std::out_of_range(
                "Block4Store::get_or_create: block_row out of range -- "
                "was Block4Store::init(n_in, n_out) called?");
        if (!block4_row_insert_tile(block_layout, indices_buf, tile_values, br, bc, Block4Tile{})) {
            block4_ensure_row_headroom(block_layout, indices_buf, tile_values, br);
            const bool ok = block4_row_insert_tile(block_layout, indices_buf, tile_values, br, bc, Block4Tile{});
            (void)ok; // block4_ensure_row_headroom grows by exactly enough for one more tile -- this must succeed
        }
        Block4Tile* inserted = find(br, bc);
        return *inserted;
    }

    void erase(uint32_t br, uint32_t bc) {
        if (br >= block_layout.rows) return;
        block4_row_remove_tile(block_layout, indices_buf, tile_values, br, bc);
    }

    std::size_t n_tiles() const { return block_layout.total_nnz; }

    // Cold-path reporting only (nnz(), diagnostics) -- O(n_tiles * 16), not
    // called from forward/backward. Must walk per-row (elem_start[r]..
    // elem_end[r]), not tile_values[0..total_nnz) -- rows have blank
    // (unused) element slots between them, same as the scattered path's
    // own values array.
    std::size_t live_synapses() const {
        std::size_t n = 0;
        for (std::size_t r = 0; r < block_layout.rows; ++r) {
            const std::size_t start = block_layout.elem_start[r];
            const std::size_t end   = block_layout.elem_end[r];
            for (std::size_t i = start; i < end; ++i) n += tile_values[i].count_live();
        }
        return n;
    }
};
