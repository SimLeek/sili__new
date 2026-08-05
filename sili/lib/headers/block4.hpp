#pragma once
#include "fp4quant.hpp"
#include "fp8quant.hpp"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// block4: dense 4x4 tiles in a larger sparse matrix for SIMD optimization  

// Block4 = 4. Changing this might break everything for obvious reasons.
#ifndef SILI_BLOCK4_TILE_SIZE
#define SILI_BLOCK4_TILE_SIZE 4 
#endif

// 2 is chosen for speed & memory compared to pure csr after testing
#ifndef SILI_BLOCK4_PROMOTE_MIN_LIVE
#define SILI_BLOCK4_PROMOTE_MIN_LIVE 2
#endif

constexpr uint32_t BLOCK4_TILE = SILI_BLOCK4_TILE_SIZE;
constexpr uint32_t BLOCK4_TILE_SLOTS = SILI_BLOCK4_TILE_SIZE * SILI_BLOCK4_TILE_SIZE;
constexpr uint32_t BLOCK4_PROMOTE_MIN_LIVE = SILI_BLOCK4_PROMOTE_MIN_LIVE;

// block4 4-wide SIMD helpers
//
// Currently the only supported activation and backprop value is float32.
//
// GCC/Clang's vector_size extension is portable across SSE/AVX/NEON/etc.,
// and got the compiler to emit real SIMD for block4's for backprop, 
// confirmed via -fopt-info-vec that it vectorizes, and while plain scalar
// arrays could vectorize, they weren't as fast since gcc couldn't prove they
// array indexed load was contiguous.
using Block4Vec  = float    __attribute__((__vector_size__(SILI_BLOCK4_TILE_SIZE * sizeof(float))));
using Block4VecU = uint32_t __attribute__((__vector_size__(SILI_BLOCK4_TILE_SIZE * sizeof(uint32_t))));
static_assert(sizeof(Block4Vec) == BLOCK4_TILE * sizeof(float), "Block4Vec width must match BLOCK4_TILE");

inline Block4Vec block4_vec_load(const float* p) {
    Block4Vec v;
    std::memcpy(&v, p, sizeof(v));   // unaligned-safe load
    return v;
}
inline void block4_vec_store(float* p, Block4Vec v) {
    std::memcpy(p, &v, sizeof(v));
}

// `for (i...) v[i]=x` forces GCC to treat the whole vector as memory 
// instead of a register, and 7.6% of all instructions in a disldo_backward 
// run were this "vectorized" broadcast helper alone. This turns multiple
// load/store ops into a single op.
inline Block4Vec block4_vec_broadcast(float x) {
    return Block4Vec{x, x, x, x};
}
inline Block4VecU block4_vecu_broadcast(uint32_t x) {
    return Block4VecU{x, x, x, x};
}

// Elementwise |x| via an IEEE-754 sign-bit clear -- std::abs isn't defined
// for GCC vector-extension types.
inline Block4Vec block4_vec_abs(Block4Vec x) {
    Block4VecU bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits &= block4_vecu_broadcast(0x7FFFFFFFu);
    Block4Vec result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// Hardcoded 4-term sum. Similar fix as block4_vec_broadcast above.
// This one function was 12.16% of all instructions in a disldo_backward run.
inline float block4_vec_hsum(Block4Vec x) {
    return x[0] + x[1] + x[2] + x[3];
}

// Elementwise sqrt -- GCC vector-extension types have no built-in sqrt.
// Used by the RMSprop-style importance damping (decayed mean-of-g^2,
// see linear_disldo.hpp's disldo_backward) instead of block4_vec_abs.
inline Block4Vec block4_vec_sqrt(Block4Vec x) {
    Block4Vec result;
    for (int i = 0; i < SILI_BLOCK4_TILE_SIZE; ++i) result[i] = std::sqrt(x[i]);
    return result;
}

// 4-wide fp4_decode_bits (fp4quant.hpp) -- decodes 4 codes in one
// shot instead of 4 separate FP4_TABLE[code] lookups. 
// Verified bit-exact against fp4_decode_bits()
// (see test_fp4_bitshift.cpp.)
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
// 4 values in one shot. 
// See fp4quant.hpp's header comment for why the |v|>=1.0 branch's bit-add
// dithering is exact, not approximate. Verified against
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

    // Never the repurposed NaN slot (8) for a genuinely near-zero lane
    const Block4VecU zero_mag_mask = (mag_code == block4_vecu_broadcast(0u));
    // sign is bit 31 (0x80000000) or 0; >>28 moves it to bit 3 (0x8 or 0),
    const Block4VecU sign_bit = sign >> 28;
    return (sign_bit | mag_code) & ~zero_mag_mask;
}

// 4-wide fp8_decode_bits (fp8quant.hpp) -- decodes 4 E4M3 codes in one
// shot. Fast path handles the common normal-code case via the same
// field-placement bit trick as block4_vec_decode_fp4; the rare
// subnormal/NaN-slot lanes (no single shared exponent bracket to
// exploit, same reason fp8_decode_bits' own scalar version branches
// there) are corrected via the exact scalar reference per-lane, matching
// this file's own established precedent (block4_sparse_get_pos/
// block4_sparse_set_pos's comment: "every SIMD optimization we tried
// here was slower than these scalar versions") rather than forcing a
// full 8-way vectorized subnormal blend for a path block4-promoted
// tiles rarely hit (promotion favours the more-important, less-often-
// near-zero synapses in the first place).
inline Block4Vec block4_vec_decode_fp8(Block4VecU codes) {
    const Block4VecU one_u = block4_vecu_broadcast(1u);
    const Block4VecU s = (codes >> 7) & one_u;
    const Block4VecU e = (codes >> 3) & block4_vecu_broadcast(0xFu);
    const Block4VecU m = codes & block4_vecu_broadcast(7u);

    const Block4VecU bits_normal =
        (s << 31) | ((e + block4_vecu_broadcast(120u)) << 23) | (m << 20);

    Block4Vec result;
    std::memcpy(&result, &bits_normal, sizeof(result));

    uint32_t codes_arr[SILI_BLOCK4_TILE_SIZE];
    std::memcpy(codes_arr, &codes, sizeof(codes_arr));
    float result_arr[SILI_BLOCK4_TILE_SIZE];
    std::memcpy(result_arr, &result, sizeof(result_arr));
    for (int i = 0; i < SILI_BLOCK4_TILE_SIZE; ++i) {
        const uint32_t e_i = (codes_arr[i] >> 3) & 0xFu;
        const uint32_t m_i = codes_arr[i] & 0x7u;
        if (e_i == 0u || (e_i == 15u && m_i == 7u))
            result_arr[i] = fp8_decode_bits(uint8_t(codes_arr[i]));
    }
    std::memcpy(&result, result_arr, sizeof(result));
    return result;
}

// 4-wide fp8_quantize_stochastic (fp8quant.hpp) -- same split as
// block4_vec_decode_fp8 above: the common normal-range case uses the
// exact same dithered-carry technique as block4_vec_quantize_stochastic_fp4
// (just E4M3's bit widths), rare subnormal/saturating/NaN lanes fall
// back to the scalar reference.
inline Block4VecU block4_vec_quantize_stochastic_fp8(Block4Vec v) {
    Block4VecU bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const Block4VecU sign  = bits & block4_vecu_broadcast(0x80000000u);
    const Block4VecU abits = bits & block4_vecu_broadcast(0x7FFFFFFFu);

    const uint64_t r0 = fp4_stochastic_next_u64();
    const uint64_t r1 = fp4_stochastic_next_u64();
    const uint64_t r2 = fp4_stochastic_next_u64();
    const uint64_t r3 = fp4_stochastic_next_u64();
    const Block4VecU dither20 = {uint32_t(r0 & 0xFFFFFu), uint32_t(r1 & 0xFFFFFu),
                                  uint32_t(r2 & 0xFFFFFu), uint32_t(r3 & 0xFFFFFu)};

    Block4VecU rounded = abits + dither20;
    const Block4VecU nan_slot_u = block4_vecu_broadcast(FP8_NAN_SLOT_BITS);
    const Block4VecU over_mask = (rounded >= nan_slot_u);
    rounded = (rounded & ~over_mask) | (block4_vecu_broadcast(FP8_MAX_BITS) & over_mask);
    const Block4VecU exp_field = (rounded >> 23) & block4_vecu_broadcast(0xFFu);
    const Block4VecU m3        = (rounded >> 20) & block4_vecu_broadcast(0x7u);
    const Block4VecU e4        = exp_field - block4_vecu_broadcast(120u);
    const Block4VecU sign_bit  = sign >> 24;  // bit31 -> bit7
    Block4VecU result = sign_bit | (e4 << 3) | m3;

    uint32_t bits_arr[SILI_BLOCK4_TILE_SIZE];
    std::memcpy(bits_arr, &bits, sizeof(bits_arr));
    uint32_t result_arr[SILI_BLOCK4_TILE_SIZE];
    std::memcpy(result_arr, &result, sizeof(result_arr));
    for (int i = 0; i < SILI_BLOCK4_TILE_SIZE; ++i) {
        float vi;
        std::memcpy(&vi, &bits_arr[i], sizeof(vi));
        const uint32_t ab = bits_arr[i] & 0x7FFFFFFFu;
        if (ab < FP8_MIN_NORMAL_BITS || ab >= FP8_NAN_SLOT_BITS || ab > 0x7F800000u)
            result_arr[i] = fp8_quantize_stochastic(vi);
    }
    Block4VecU final_result;
    std::memcpy(&final_result, result_arr, sizeof(final_result));
    return final_result;
}

// One dense tile: 16 bytes, (4-bit importance<<4|4-bit weight) per slot,
// stored [local_j * 4 + local_i] (Matches disldo's CSR orientation,
// so no transpose is needed)
// 
// count_live() is a full O(16) byte scan, but used only by 
// promotion/demotion/reporting 
struct Block4Tile {
    uint8_t data[BLOCK4_TILE_SLOTS] = {0};

    static uint32_t slot_index(uint32_t local_i, uint32_t local_j) { return local_j * BLOCK4_TILE + local_i; }

    uint8_t& at(uint32_t local_i, uint32_t local_j) { return data[slot_index(local_i, local_j)]; }
    uint8_t  at(uint32_t local_i, uint32_t local_j) const { return data[slot_index(local_i, local_j)]; }

    uint32_t count_live() const {
        // todo: check if (data[i]>>4 != 0) or (data[i]>>4 > min_importance)
	// performs better, because low importance synapses aren't important
	// enough to count.
	uint32_t n = 0;
        for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i)
            if (data[i] != 0) ++n;
        return n;
    }
};

// Sparse-packed tile encoding for better compression
//
// Note: dense is strictly faster (direct access, no pack/unpack, no
// resize), sparse is strictly smaller for BLOCK4_SPARSE_MAX_COUNT (10)
// or fewer live synapses (11*12=132>128).
//
// Default 10, settable down to 0 to disable compression entirely.
#ifndef SILI_BLOCK4_SPARSE_MAX_COUNT
#define SILI_BLOCK4_SPARSE_MAX_COUNT 10
#endif
static_assert(SILI_BLOCK4_SPARSE_MAX_COUNT * 12 + 8 <= 128,
    "sparse tile encoding must fit in the same 128 bits as the dense one");
constexpr uint32_t BLOCK4_SPARSE_MAX_COUNT = SILI_BLOCK4_SPARSE_MAX_COUNT;

// Total byte length of a sparse-packed tile holding `count` live synapses 
inline std::size_t block4_sparse_packed_len(uint8_t count) {
    return std::size_t(1) + (std::size_t(count) + 1) / 2 + std::size_t(count);
}

// Every SIMD optimization we tried here was slower than these scalar versions.
inline uint8_t block4_sparse_get_pos(const uint8_t* packed, uint32_t i) {
    const uint8_t b = packed[1 + i / 2];
    return (i % 2 == 0) ? uint8_t(b >> 4) : uint8_t(b & 0xFu);
}
inline void block4_sparse_set_pos(uint8_t* packed, uint32_t i, uint8_t pos) {
    uint8_t& b = packed[1 + i / 2];
    if (i % 2 == 0) b = uint8_t((b & 0x0Fu) | (pos << 4));
    else            b = uint8_t((b & 0xF0u) | (pos & 0xFu));
}

inline void block4_sparse_unpack(const uint8_t* packed, uint8_t dense[BLOCK4_TILE_SLOTS]) {
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i) dense[i] = 0;
    const uint8_t count = packed[0];
    const std::size_t value_off = 1 + (std::size_t(count) + 1) / 2;
    for (uint8_t i = 0; i < count; ++i)
        dense[block4_sparse_get_pos(packed, i)] = packed[value_off + i];
}

// Packs `dense` into `packed`, writing exactly block4_sparse_packed_len(count)
// bytes (the caller must ensure that much room exists) -- returns the count,
// so the caller can compute the exact byte length written without a second
// pass over `dense`.
inline uint8_t block4_sparse_pack(const uint8_t dense[BLOCK4_TILE_SLOTS], uint8_t* packed) {
    uint8_t count = 0;
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i)
        if (dense[i] != 0) ++count;
    packed[0] = count;
    const std::size_t nib_bytes = (std::size_t(count) + 1) / 2;
    for (std::size_t k = 0; k < nib_bytes; ++k) packed[1 + k] = 0; // set_pos does read-modify-write
    const std::size_t value_off = 1 + nib_bytes;
    uint8_t idx = 0;
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i) {
        if (dense[i] == 0) continue;
        block4_sparse_set_pos(packed, idx, uint8_t(i));
        packed[value_off + idx] = dense[i];
        ++idx;
    }
    return count;
}

// Byte length of the tile currently stored at `tile_bytes`
inline std::size_t block4_stored_tile_len(bool is_sparse, const uint8_t* tile_bytes) {
    return is_sparse ? block4_sparse_packed_len(tile_bytes[0]) : std::size_t(BLOCK4_TILE_SLOTS);
}

// ── FP8 dense-tile encoding (Block4Tile8) ────────────────────────────────────
// "Alt, not replace": a full, separate tile-encoding layer for E4M3
// (fp8quant.hpp), added alongside Block4Tile/the FP4 sparse-pack functions
// above WITHOUT modifying any of them. E4M3 needs a full byte per value (no
// nibble-sharing like FP4's weight<<4|importance), so a slot is 2 bytes, not
// 1 -- two contiguous HALVES instead of interleaved pairs (data[0..15] =
// weight per slot, data[16..31] = importance per slot), matching
// FP8BiValues' own two-separate-arrays convention and keeping 4 consecutive
// weight (or importance) bytes contiguous for block4_vec_decode_fp8's SIMD load.
//
// block4_row_shift/block4_grow_last_row/block4_ensure_row_headroom/
// block4_row_insert_tile/block4_row_remove_tile/block4_resize_tile_in_row
// (below) are pure byte-buffer plumbing -- confirmed by reading them
// directly, no assumption anywhere about how many bytes make up one slot's
// value(s) -- so Block4Store8 (further below) reuses every one of them
// UNCHANGED; only the tile-encoding layer here is FP8-specific.

constexpr uint32_t BLOCK4_TILE_SLOTS8_BYTES = BLOCK4_TILE_SLOTS * 2;  // 32: weight half + importance half

// One dense FP8 tile: 32 bytes, weight byte per slot at [0..15],
// importance byte per slot at [16..31], same [local_j*4+local_i] slot
// order as Block4Tile (matches disldo's CSR orientation).
struct Block4Tile8 {
    uint8_t data[BLOCK4_TILE_SLOTS8_BYTES] = {0};

    static uint32_t slot_index(uint32_t local_i, uint32_t local_j) { return local_j * BLOCK4_TILE + local_i; }

    uint8_t& at_weight(uint32_t local_i, uint32_t local_j) { return data[slot_index(local_i, local_j)]; }
    uint8_t  at_weight(uint32_t local_i, uint32_t local_j) const { return data[slot_index(local_i, local_j)]; }
    uint8_t& at_importance(uint32_t local_i, uint32_t local_j) { return data[BLOCK4_TILE_SLOTS + slot_index(local_i, local_j)]; }
    uint8_t  at_importance(uint32_t local_i, uint32_t local_j) const { return data[BLOCK4_TILE_SLOTS + slot_index(local_i, local_j)]; }

    uint32_t count_live() const;  // defined after block4_count_live8 below
};

// A slot is live iff EITHER its weight or importance byte is nonzero --
// same "not both zero" convention as block4_count_live's `data[i] != 0`
// (FP4's single packed byte is 0 iff both nibbles decode to 0.0).
inline uint32_t block4_count_live8(const uint8_t dense[BLOCK4_TILE_SLOTS8_BYTES]) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i)
        if (dense[i] != 0 || dense[BLOCK4_TILE_SLOTS + i] != 0) ++n;
    return n;
}

inline uint32_t Block4Tile8::count_live() const { return block4_count_live8(data); }

// Sparse-packed FP8 tile encoding -- same position-nibble scheme as FP4's
// block4_sparse_pack (4-bit slot position, 2 packed per byte), but 2
// VALUE bytes per live entry (weight, then importance -- two contiguous
// runs, same halves convention as the dense layout above) instead of 1.
//
// Budget: count*20+8 <= 256 (32 bytes = 256 bits, matching
// Block4Tile8's own dense size, same relationship FP4's
// count*12+8<=128 has to its 16-byte dense tile) -- 20 bits/entry (4
// position + 16 value) vs FP4's 12 (4 position + 8 value). Max count:
// 12 (12*20+8=248<=256; 13*20+8=268>256) -- a real, correctly-derived
// difference from FP4's 10, not a copy-paste of FP4's constant.
#ifndef SILI_BLOCK4_SPARSE_MAX_COUNT8
#define SILI_BLOCK4_SPARSE_MAX_COUNT8 12
#endif
static_assert(SILI_BLOCK4_SPARSE_MAX_COUNT8 * 20 + 8 <= 256,
    "sparse FP8 tile encoding must fit in the same 256 bits as the dense one");
constexpr uint32_t BLOCK4_SPARSE_MAX_COUNT8 = SILI_BLOCK4_SPARSE_MAX_COUNT8;

inline std::size_t block4_sparse_packed_len8(uint8_t count) {
    return std::size_t(1) + (std::size_t(count) + 1) / 2 + std::size_t(count) * 2;
}

// Position nibble packing is byte-identical to FP4's -- reuses
// block4_sparse_get_pos/block4_sparse_set_pos directly (position
// encoding has nothing to do with value width).

inline void block4_sparse_unpack8(const uint8_t* packed, uint8_t dense[BLOCK4_TILE_SLOTS8_BYTES]) {
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS8_BYTES; ++i) dense[i] = 0;
    const uint8_t count = packed[0];
    const std::size_t value_off = 1 + (std::size_t(count) + 1) / 2;
    for (uint8_t i = 0; i < count; ++i) {
        const uint8_t pos = block4_sparse_get_pos(packed, i);
        dense[pos] = packed[value_off + i];                                    // weight
        dense[BLOCK4_TILE_SLOTS + pos] = packed[value_off + std::size_t(count) + i]; // importance
    }
}

inline uint8_t block4_sparse_pack8(const uint8_t dense[BLOCK4_TILE_SLOTS8_BYTES], uint8_t* packed) {
    uint8_t count = 0;
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i)
        if (dense[i] != 0 || dense[BLOCK4_TILE_SLOTS + i] != 0) ++count;
    packed[0] = count;
    const std::size_t nib_bytes = (std::size_t(count) + 1) / 2;
    for (std::size_t k = 0; k < nib_bytes; ++k) packed[1 + k] = 0;
    const std::size_t value_off = 1 + nib_bytes;
    uint8_t idx = 0;
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i) {
        if (dense[i] == 0 && dense[BLOCK4_TILE_SLOTS + i] == 0) continue;
        block4_sparse_set_pos(packed, idx, uint8_t(i));
        packed[value_off + idx] = dense[i];
        packed[value_off + std::size_t(count) + idx] = dense[BLOCK4_TILE_SLOTS + i];
        ++idx;
    }
    return count;
}

inline std::size_t block4_stored_tile_len8(bool is_sparse, const uint8_t* tile_bytes) {
    return is_sparse ? block4_sparse_packed_len8(tile_bytes[0]) : std::size_t(BLOCK4_TILE_SLOTS8_BYTES);
}

inline void block4_row_shift(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<std::size_t>& tbyte_start,
    std::vector<std::size_t>& tbyte_end,
    std::vector<uint8_t>& tile_data,
    std::vector<uint8_t>& tile_is_sparse, // todo: should be 1 bit per tile not 1 byte
    std::size_t row,
    std::size_t target_idx_byte_alloc,
    std::size_t target_tile_byte_alloc,
    std::size_t target_elem_alloc,
    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max(),
    std::size_t max_tile_bytes = std::numeric_limits<std::size_t>::max())
{
    // indices_buf byte side (uleb128 column deltas)
    const std::size_t cur_idx_alloc = L.row_alloc_bytes(row);
    if (cur_idx_alloc != target_idx_byte_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = L.byte_start[row + 1];
        const std::size_t move_len  = L.byte_start[L.rows] - move_src;
        const std::size_t new_start = L.byte_start[row] + target_idx_byte_alloc;

        if (target_idx_byte_alloc > cur_idx_alloc) {
            const std::size_t new_total = ibuf.size() + (target_idx_byte_alloc - cur_idx_alloc);
            if (new_total > max_indices_bytes) throw std::bad_alloc();
            ibuf.resize(new_total);
        }
        if (move_len > 0)
            std::memmove(ibuf.data() + new_start, ibuf.data() + move_src, move_len);
        if (target_idx_byte_alloc < cur_idx_alloc)
            ibuf.resize(ibuf.size() - (cur_idx_alloc - target_idx_byte_alloc));

        const std::ptrdiff_t d = std::ptrdiff_t(target_idx_byte_alloc) - std::ptrdiff_t(cur_idx_alloc);
        for (std::size_t r = row + 1; r <= L.rows; ++r)
            L.byte_start[r] = std::size_t(std::ptrdiff_t(L.byte_start[r]) + d);
        for (std::size_t r = row + 1; r < L.rows; ++r)
            L.byte_end[r] = std::size_t(std::ptrdiff_t(L.byte_end[r]) + d);
    }

    // tile_data byte side (variable-length packed/dense tile bytes 
    const std::size_t cur_tile_alloc = tbyte_start[row + 1] - tbyte_start[row];
    if (cur_tile_alloc != target_tile_byte_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = tbyte_start[row + 1];
        const std::size_t move_len  = tbyte_start[L.rows] - move_src;
        const std::size_t new_start = tbyte_start[row] + target_tile_byte_alloc;

        if (target_tile_byte_alloc > cur_tile_alloc) {
            const std::size_t new_total = tile_data.size() + (target_tile_byte_alloc - cur_tile_alloc);
            if (new_total > max_tile_bytes) throw std::bad_alloc();
            tile_data.resize(new_total);
        }
        if (move_len > 0)
            std::memmove(tile_data.data() + new_start, tile_data.data() + move_src, move_len);
        if (target_tile_byte_alloc < cur_tile_alloc)
            tile_data.resize(tile_data.size() - (cur_tile_alloc - target_tile_byte_alloc));

        const std::ptrdiff_t d = std::ptrdiff_t(target_tile_byte_alloc) - std::ptrdiff_t(cur_tile_alloc);
        for (std::size_t r = row + 1; r <= L.rows; ++r)
            tbyte_start[r] = std::size_t(std::ptrdiff_t(tbyte_start[r]) + d);
        for (std::size_t r = row + 1; r < L.rows; ++r)
            tbyte_end[r] = std::size_t(std::ptrdiff_t(tbyte_end[r]) + d);
    }

    // tile_is_sparse elem side (1 byte/slot flag)
    const std::size_t cur_elem_alloc = L.row_alloc_elems(row);
    if (cur_elem_alloc != target_elem_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = L.elem_start[row + 1];
        const std::size_t move_len  = L.elem_start[L.rows] - move_src;
        const std::size_t new_start = L.elem_start[row] + target_elem_alloc;
        const std::size_t current_total = L.total_alloc_elems();

        if (target_elem_alloc > cur_elem_alloc) {
            // Not separately budget-checked against max_tile_bytes -- an
            // elem slot can't exist without real tile_data bytes to back
            // it, so tile_data's own check above is always the binding
            // one; this is 1 byte/slot, real but small next to it.
            const std::size_t new_total_elems = current_total + (target_elem_alloc - cur_elem_alloc);
            tile_is_sparse.resize(new_total_elems);
        }
        if (move_len > 0)
            std::memmove(tile_is_sparse.data() + new_start, tile_is_sparse.data() + move_src, move_len);
        if (target_elem_alloc < cur_elem_alloc)
            tile_is_sparse.resize(current_total - (cur_elem_alloc - target_elem_alloc));

        const std::ptrdiff_t d = std::ptrdiff_t(target_elem_alloc) - std::ptrdiff_t(cur_elem_alloc);
        for (std::size_t r = row + 1; r <= L.rows; ++r)
            L.elem_start[r] = std::size_t(std::ptrdiff_t(L.elem_start[r]) + d);
        for (std::size_t r = row + 1; r < L.rows; ++r)
            L.elem_end[r] = std::size_t(std::ptrdiff_t(L.elem_end[r]) + d);
    }
}

inline void block4_grow_last_row(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<std::size_t>& tbyte_start,
    std::vector<std::size_t>& tbyte_end,
    std::vector<uint8_t>& tile_data,
    std::vector<uint8_t>& tile_is_sparse,
    std::size_t target_idx_byte_alloc,
    std::size_t target_tile_byte_alloc,
    std::size_t target_elem_alloc,
    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max(),
    std::size_t max_tile_bytes = std::numeric_limits<std::size_t>::max())
{
    if (L.rows == 0) return;
    const std::size_t r = L.rows - 1;
    const std::size_t cur_idx  = L.row_alloc_bytes(r);
    const std::size_t cur_tile = tbyte_start[r + 1] - tbyte_start[r];
    const std::size_t cur_elem = L.row_alloc_elems(r);
    if (target_idx_byte_alloc > cur_idx) {
        const std::size_t new_total = ibuf.size() + (target_idx_byte_alloc - cur_idx);
        if (new_total > max_indices_bytes) throw std::bad_alloc();
        ibuf.resize(new_total, uint8_t(0));
        L.byte_start[L.rows] = L.byte_start[r] + target_idx_byte_alloc;
    }
    if (target_tile_byte_alloc > cur_tile) {
        const std::size_t new_total = tile_data.size() + (target_tile_byte_alloc - cur_tile);
        if (new_total > max_tile_bytes) throw std::bad_alloc();
        tile_data.resize(new_total, uint8_t(0));
        tbyte_start[r + 1] = tbyte_start[r] + target_tile_byte_alloc;
    }
    if (target_elem_alloc > cur_elem) {
        const std::size_t new_total = L.elem_start[r] + target_elem_alloc;
        tile_is_sparse.resize(new_total);
        L.elem_start[L.rows] = new_total;
    }
}

inline void block4_ensure_row_headroom(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<std::size_t>& tbyte_start,
    std::vector<std::size_t>& tbyte_end,
    std::vector<uint8_t>& tile_data,
    std::vector<uint8_t>& tile_is_sparse,
    std::size_t row,
    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max(),
    std::size_t max_tile_bytes = std::numeric_limits<std::size_t>::max())
{
    const std::size_t target_idx  = L.row_alloc_bytes(row) + uleb128_max_bytes<uint32_t>();
    const std::size_t target_tile = (tbyte_start[row + 1] - tbyte_start[row]) + block4_sparse_packed_len(0);
    const std::size_t target_elem = L.row_alloc_elems(row) + 1;
    if (row + 1 < L.rows)
        block4_row_shift(L, ibuf, tbyte_start, tbyte_end, tile_data, tile_is_sparse, row,
                          target_idx, target_tile, target_elem, max_indices_bytes, max_tile_bytes);
    else
        block4_grow_last_row(L, ibuf, tbyte_start, tbyte_end, tile_data, tile_is_sparse,
                              target_idx, target_tile, target_elem, max_indices_bytes, max_tile_bytes);
}

inline bool block4_row_insert_tile(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<std::size_t>& tbyte_start,
    std::vector<std::size_t>& tbyte_end,
    std::vector<uint8_t>& tile_data,
    std::vector<uint8_t>& tile_is_sparse,
    std::size_t row,
    uint32_t new_col)
{
    const std::size_t n = L.row_nnz(row);

    std::size_t byte_pos      = L.byte_start[row];
    std::size_t elem_pos      = L.elem_start[row];
    std::size_t tbyte_pos     = tbyte_start[row];
    uint32_t    prev_col      = 0;
    std::size_t ins_byte_pos  = L.byte_end[row];
    std::size_t ins_elem_pos  = L.elem_end[row];
    std::size_t ins_tbyte_pos = tbyte_end[row];
    bool        has_next      = false;
    uint32_t    next_col      = 0;
    std::size_t next_dlen     = 0;

    for (std::size_t e = 0; e < n; ++e) {
        std::size_t dlen = 0;
        const uint32_t delta = uleb128_decode<uint32_t>(ibuf.data() + byte_pos, dlen);
        const uint32_t col   = prev_col + delta;
        if (col == new_col) return false; // duplicate, skip (tile already exists)
        if (col > new_col) {
            ins_byte_pos  = byte_pos;
            ins_elem_pos  = elem_pos;
            ins_tbyte_pos = tbyte_pos;
            has_next      = true;
            next_col      = col;
            next_dlen     = dlen;
            break;
        }
        const std::size_t tlen = block4_stored_tile_len(tile_is_sparse[elem_pos], tile_data.data() + tbyte_pos);
        prev_col   = col;
        byte_pos  += dlen;
        elem_pos++;
        tbyte_pos += tlen;
    }

    uint8_t new_d_buf[uleb128_max_bytes<uint32_t>()];
    const std::size_t new_d_len = uleb128_encode<uint32_t>(new_col - prev_col, new_d_buf);

    uint8_t upd_d_buf[uleb128_max_bytes<uint32_t>()];
    std::size_t upd_d_len = 0;
    if (has_next)
        upd_d_len = uleb128_encode<uint32_t>(next_col - new_col, upd_d_buf);

    const std::ptrdiff_t idx_delta =
        std::ptrdiff_t(new_d_len + upd_d_len) - std::ptrdiff_t(next_dlen);

    const std::size_t used_idx_bytes = L.byte_end[row] - L.byte_start[row];
    if (idx_delta > 0 && std::size_t(idx_delta) > L.row_alloc_bytes(row) - used_idx_bytes)
        return false;
    if (L.row_nnz(row) >= L.row_alloc_elems(row))
        return false;
    const std::size_t used_tile_bytes = tbyte_end[row] - tbyte_start[row];
    const std::size_t new_tile_len = block4_sparse_packed_len(0); // new tiles start empty (1 byte)
    if (new_tile_len > (tbyte_start[row + 1] - tbyte_start[row]) - used_tile_bytes)
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

    {
        const std::size_t shift_from = ins_tbyte_pos;
        const std::size_t shift_len  = tbyte_end[row] - shift_from;
        if (shift_len > 0)
            std::memmove(tile_data.data() + shift_from + new_tile_len, tile_data.data() + shift_from, shift_len);
        // new_tile_len is 1. writes just the sparse-tile count byte (0 = empty)
        std::memset(tile_data.data() + ins_tbyte_pos, 0, new_tile_len);
        tbyte_end[row] += new_tile_len;
    }

    if (ins_elem_pos < L.elem_end[row])
        std::memmove(tile_is_sparse.data() + ins_elem_pos + 1, tile_is_sparse.data() + ins_elem_pos,
                     L.elem_end[row] - ins_elem_pos);
    tile_is_sparse[ins_elem_pos] = 1; // true = sparse (empty, count=0) -- see comment above

    L.elem_end[row]++;
    L.total_nnz++;
    return true;
}

inline bool block4_row_remove_tile(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<std::size_t>& tbyte_start,
    std::vector<std::size_t>& tbyte_end,
    std::vector<uint8_t>& tile_data,
    std::vector<uint8_t>& tile_is_sparse,
    std::size_t row,
    uint32_t target_col)
{
    const std::size_t n = L.row_nnz(row);
    if (n == 0) return false;

    std::size_t byte_pos  = L.byte_start[row];
    std::size_t elem_pos  = L.elem_start[row];
    std::size_t tbyte_pos = tbyte_start[row];
    uint32_t    prev_col  = 0;

    for (std::size_t e = 0; e < n; ++e) {
        std::size_t delta_len = 0;
        const uint32_t delta = uleb128_decode<uint32_t>(ibuf.data() + byte_pos, delta_len);
        const uint32_t col   = prev_col + delta;
        const std::size_t tlen = block4_stored_tile_len(tile_is_sparse[elem_pos], tile_data.data() + tbyte_pos);

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

            {
                const std::size_t shift_from = tbyte_pos + tlen;
                const std::size_t shift_len  = tbyte_end[row] - shift_from;
                if (shift_len > 0)
                    std::memmove(tile_data.data() + tbyte_pos, tile_data.data() + shift_from, shift_len);
                tbyte_end[row] -= tlen;
            }

            const std::size_t row_end = L.elem_end[row];
            if (elem_pos + 1 < row_end)
                std::memmove(tile_is_sparse.data() + elem_pos, tile_is_sparse.data() + elem_pos + 1,
                             row_end - elem_pos - 1);
            L.elem_end[row]--;
            L.total_nnz--;
            return true;
        }

        prev_col   = col;
        byte_pos  += delta_len;
        elem_pos++;
        tbyte_pos += tlen;
    }
    return false;
}

inline void block4_resize_tile_in_row(
    DeltaCSRLayout& L,
    std::vector<std::size_t>& tbyte_start,
    std::vector<std::size_t>& tbyte_end,
    std::vector<uint8_t>& tile_data,
    std::size_t row,
    std::size_t tbyte_pos,
    std::size_t old_len,
    const uint8_t* new_bytes,
    std::size_t new_len,
    std::size_t max_tile_bytes = std::numeric_limits<std::size_t>::max())
{
    if (new_len == old_len) {
        std::memcpy(tile_data.data() + tbyte_pos, new_bytes, new_len);
        return;
    }
    const std::ptrdiff_t delta = std::ptrdiff_t(new_len) - std::ptrdiff_t(old_len);
    if (delta > 0) {
        const std::size_t alloc    = tbyte_start[row + 1] - tbyte_start[row];
        const std::size_t used     = tbyte_end[row] - tbyte_start[row];
        const std::size_t headroom = alloc - used;
        if (std::size_t(delta) > headroom) {
            const std::size_t shortfall = std::size_t(delta) - headroom;
            const std::size_t new_total = tile_data.size() + shortfall;
            if (new_total > max_tile_bytes) throw std::bad_alloc();
            if (row + 1 < L.rows) {
                const std::size_t move_src = tbyte_start[row + 1];
                const std::size_t move_len = tbyte_start[L.rows] - move_src;
                tile_data.resize(new_total);
                std::memmove(tile_data.data() + move_src + shortfall, tile_data.data() + move_src, move_len);
                for (std::size_t r = row + 1; r <= L.rows; ++r) tbyte_start[r] += shortfall;
                for (std::size_t r = row + 1; r < L.rows; ++r)  tbyte_end[r]   += shortfall;
            } else {
                tile_data.resize(new_total);
                tbyte_start[L.rows] += shortfall;
            }
        }
    }
    const std::size_t shift_from = tbyte_pos + old_len;
    const std::size_t shift_len  = tbyte_end[row] - shift_from;
    if (shift_len > 0)
        std::memmove(tile_data.data() + tbyte_pos + new_len, tile_data.data() + shift_from, shift_len);
    std::memcpy(tile_data.data() + tbyte_pos, new_bytes, new_len);
    tbyte_end[row] = std::size_t(std::ptrdiff_t(tbyte_end[row]) + delta);
}

// Same as Block4Tile::count_live(), but
// callable on a raw pointer since Block4TileHandle's scratch buffer isn't
// always wrapped in a full Block4Tile object.
//
// todo: check if (data[i]>>4 != 0) or (data[i]>>4 > min_importance)
// performs better, because low importance synapses aren't important
// enough to count.
inline uint32_t block4_count_live(const uint8_t dense[BLOCK4_TILE_SLOTS]) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i)
        if (dense[i] != 0) ++n;
    return n;
}

struct Block4Store;

// RAII accessor for one logical tile's synapse data. 
class Block4TileHandle {
    Block4Store* store_ = nullptr;
    uint32_t br_ = 0, bc_ = 0;
    std::size_t byte_pos_ = 0;  // position in store_->tile_data
    uint8_t scratch_[BLOCK4_TILE_SLOTS] = {0};
    bool was_sparse_ = false;
    bool dirty_ = false;
    bool valid_ = false;

public:
    Block4TileHandle() = default;
    Block4TileHandle(Block4Store& store, uint32_t br, uint32_t bc);
    // Fast-path constructor
    Block4TileHandle(Block4Store& store, uint32_t br, uint32_t bc, std::size_t elem_pos, std::size_t byte_pos);
    ~Block4TileHandle();

    Block4TileHandle(Block4TileHandle&& other) noexcept {
        store_ = other.store_; br_ = other.br_; bc_ = other.bc_; byte_pos_ = other.byte_pos_;
        std::memcpy(scratch_, other.scratch_, sizeof(scratch_));
        was_sparse_ = other.was_sparse_; dirty_ = other.dirty_; valid_ = other.valid_;
        other.valid_ = false; other.dirty_ = false; // moved-from: destructor becomes a no-op
    }
    Block4TileHandle& operator=(Block4TileHandle&& other) noexcept {
        if (this == &other) return *this;
        // Flush any pending write of THIS handle before taking over
        // other's state -- moving must not silently drop a re-pack.
        this->~Block4TileHandle();
        store_ = other.store_; br_ = other.br_; bc_ = other.bc_; byte_pos_ = other.byte_pos_;
        std::memcpy(scratch_, other.scratch_, sizeof(scratch_));
        was_sparse_ = other.was_sparse_; dirty_ = other.dirty_; valid_ = other.valid_;
        other.valid_ = false; other.dirty_ = false;
        return *this;
    }
    Block4TileHandle(const Block4TileHandle&) = delete;
    Block4TileHandle& operator=(const Block4TileHandle&) = delete;

    explicit operator bool() const { return valid_; }

    uint8_t& at(uint32_t li, uint32_t lj);
    uint8_t at(uint32_t li, uint32_t lj) const;
    uint32_t count_live() const;

    // Raw read-only pointer to this tile's 16 bytes (scratch_ if sparse,
    // tile_data+byte_pos_ if dense), for faster reads
    const uint8_t* raw_data() const;
};

struct Block4Store {
    DeltaCSRLayout            block_layout;    // rows/cols are block-granularity (ceil(n_in/4), ceil(n_out/4))
    std::vector<uint8_t>      indices_buf;     // uleb128-encoded block-col deltas

    // Variable-length tile storage
    std::vector<uint8_t>      tile_data;
    std::vector<std::size_t>  tile_byte_start; // size rows+1
    std::vector<std::size_t>  tile_byte_end;   // size rows
    std::vector<uint8_t>      tile_is_sparse;  // parallel to block_layout's elem_start/elem_end

    // Default BLOCK4_SPARSE_MAX_COUNT (10, the exact 10*12+8=128
    // arithmetic) saves the most memory, but lowering may increase speed.
    uint32_t switch_point = BLOCK4_SPARSE_MAX_COUNT;

    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max();
    std::size_t max_tile_bytes    = std::numeric_limits<std::size_t>::max();

    /**
     * @brief Sets the memory limits for indices and tile data.
     *
     * Pre-reserves `tile_data`'s capacity up to the configured cap so every
     * subsequent `tile_data.resize()` call within budget avoids allocation overhead.
     *
     * @param indices_limit_bytes Maximum allowable memory in bytes for indices.
     * @param tile_limit_bytes    Maximum allowable memory in bytes for tile data.
     */
    void set_limits(std::size_t indices_limit_bytes, std::size_t tile_limit_bytes) {
        max_indices_bytes = indices_limit_bytes;
        max_tile_bytes    = tile_limit_bytes;   
	if (tile_limit_bytes != std::numeric_limits<std::size_t>::max()) {
            try {
                tile_data.reserve(tile_limit_bytes);
            } catch (const std::length_error&) {
            } catch (const std::bad_alloc&) {
            }
        }
    }

    std::uint64_t dropped_growth_events = 0;

    std::uint32_t tile_data_grow_lock = 0;

    // Persistent scratch buffers for disldo_forward/disldo_backward
    std::vector<uint32_t>    scratch_tile_br, scratch_tile_bc;
    std::vector<std::size_t> scratch_tile_elem, scratch_tile_byte;
    std::vector<uint32_t>    scratch_row_live_count; // backward only
    std::vector<double>      scratch_row_grad;        // backward only
    // backward only: 
    std::vector<std::size_t> scratch_row_ti_start;

    // Sizes an empty store for a layer of n_in x n_out real (not block) dimensions.
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
        tile_byte_start.assign(block_layout.rows + 1, 0);
        tile_byte_end.assign(block_layout.rows, 0);
        tile_data.clear();
        tile_is_sparse.clear();
    }

    DeltaCSRRowCursor<uint32_t> row_cursor(std::size_t br) const {
        return DeltaCSRRowCursor<uint32_t>(indices_buf.data(), block_layout, br);
    }

    // Byte length of the tile at (elem_pos, byte_pos) 
    std::size_t tile_len_at(std::size_t elem_pos, std::size_t byte_pos) const {
        return block4_stored_tile_len(tile_is_sparse[elem_pos], &tile_data[byte_pos]);
    }

    // Raw storage lookup. Returns this tile's byte offset into tile_data 
    // (SIZE_MAX if (br,bc) has no live tile), and writes its elem_pos to 
    // *out_elem_pos if non-null.
    std::size_t raw_find(uint32_t br, uint32_t bc, std::size_t* out_elem_pos = nullptr) const {
        if (br >= block_layout.rows) return std::numeric_limits<std::size_t>::max();
        auto cur = row_cursor(br);
        const std::size_t n = block_layout.row_nnz(br);
        std::size_t elem_pos = block_layout.elem_start[br];
        std::size_t byte_pos = tile_byte_start[br];
        for (std::size_t e = 0; e < n; ++e, ++elem_pos) {
            const uint32_t col = cur.advance();
            if (col == bc) {
                if (out_elem_pos) *out_elem_pos = elem_pos;
                return byte_pos;
            }
            if (col > bc) break; // sorted ascending -- can't appear later
            byte_pos += tile_len_at(elem_pos, byte_pos);
        }
        return std::numeric_limits<std::size_t>::max();
    }

    bool is_sparse(uint32_t br, uint32_t bc) const {
        std::size_t elem_pos = 0;
        return raw_find(br, bc, &elem_pos) != std::numeric_limits<std::size_t>::max()
            && bool(tile_is_sparse[elem_pos]);
    }

    Block4TileHandle find(uint32_t br, uint32_t bc) { return Block4TileHandle(*this, br, bc); }
    Block4TileHandle find(uint32_t br, uint32_t bc) const {
        return const_cast<Block4Store*>(this)->find(br, bc);
    }

    // Fast path for a caller that already walked row_cursor(br) itself
    Block4TileHandle at_index(uint32_t br, uint32_t bc, std::size_t elem_pos, std::size_t byte_pos) {
        return Block4TileHandle(*this, br, bc, elem_pos, byte_pos);
    }
    Block4TileHandle at_index(uint32_t br, uint32_t bc, std::size_t elem_pos, std::size_t byte_pos) const {
        return const_cast<Block4Store*>(this)->at_index(br, bc, elem_pos, byte_pos);
    }

    Block4TileHandle get_or_create(uint32_t br, uint32_t bc) {
        if (raw_find(br, bc) == std::numeric_limits<std::size_t>::max()) {
            if (br >= block_layout.rows)
                throw std::out_of_range(
                    "Block4Store::get_or_create: block_row out of range -- "
                    "was Block4Store::init(n_in, n_out) called?");
            if (!block4_row_insert_tile(block_layout, indices_buf, tile_byte_start, tile_byte_end,
                                         tile_data, tile_is_sparse, br, bc)) {
                block4_ensure_row_headroom(block_layout, indices_buf, tile_byte_start, tile_byte_end,
                                            tile_data, tile_is_sparse, br, max_indices_bytes, max_tile_bytes);
                const bool ok = block4_row_insert_tile(block_layout, indices_buf, tile_byte_start, tile_byte_end,
                                                        tile_data, tile_is_sparse, br, bc);
                (void)ok; // block4_ensure_row_headroom grows by exactly enough for one more tile -- this must succeed
            }
        }
        return Block4TileHandle(*this, br, bc);
    }

    void erase(uint32_t br, uint32_t bc) {
        if (br >= block_layout.rows) return;
        block4_row_remove_tile(block_layout, indices_buf, tile_byte_start, tile_byte_end,
                                tile_data, tile_is_sparse, br, bc);
    }

    // Explicit compression check called at synaptogenesis/pruning checkpoints
    void maybe_compress(uint32_t br, uint32_t bc) {
        std::size_t elem_pos = 0;
        const std::size_t byte_pos = raw_find(br, bc, &elem_pos);
        if (byte_pos == std::numeric_limits<std::size_t>::max() || tile_is_sparse[elem_pos]) return;
        const uint32_t n = block4_count_live(&tile_data[byte_pos]);
        if (n > switch_point) return;
        uint8_t packed[BLOCK4_TILE_SLOTS];
        block4_sparse_pack(&tile_data[byte_pos], packed);
        const std::size_t new_len = block4_sparse_packed_len(uint8_t(n));
        block4_resize_tile_in_row(block_layout, tile_byte_start, tile_byte_end, tile_data,
                                   br, byte_pos, BLOCK4_TILE_SLOTS, packed, new_len, max_tile_bytes);
        tile_is_sparse[elem_pos] = 1;
    }

    // Redistributes row-level headroom
    void equalize_step(std::size_t& current_row) {
        if (block_layout.rows == 0) return;
        const std::size_t row = current_row % block_layout.rows;
        const std::size_t target_idx = block_layout.rows > 0
            ? (block_layout.total_alloc_bytes() + block_layout.rows - 1) / block_layout.rows : 0;
        const std::size_t target_tile = block_layout.rows > 0
            ? (tile_data.size() + block_layout.rows - 1) / block_layout.rows : 0;
        const std::size_t target_elem = block_layout.rows > 0
            ? (block_layout.total_alloc_elems() + block_layout.rows - 1) / block_layout.rows : 0;
        block4_row_shift(block_layout, indices_buf, tile_byte_start, tile_byte_end, tile_data, tile_is_sparse,
                          row, target_idx, target_tile, target_elem, max_indices_bytes, max_tile_bytes);
        current_row = (current_row + 1) % block_layout.rows;
    }

    // called from ~Block4TileHandle()'s was_sparse_ branch (rare)
    void commit_dirty_sparse_tile(uint32_t br, uint32_t bc, const uint8_t scratch[BLOCK4_TILE_SLOTS]) {
        // Another call could have inserted/removed a different tile in this same
        // row in between, so we need to fetch by coordinate and not a cached pointer
	std::size_t elem_pos = 0;
        const std::size_t fresh_byte_pos = raw_find(br, bc, &elem_pos);
        if (fresh_byte_pos == std::numeric_limits<std::size_t>::max()) return; // was already erased

        const uint32_t n = block4_count_live(scratch);
        uint8_t packed[BLOCK4_TILE_SLOTS];
        std::size_t new_len;
        bool now_sparse;
        if (n <= switch_point) {
            block4_sparse_pack(scratch, packed);
            new_len = block4_sparse_packed_len(uint8_t(n));
            now_sparse = true;
        } else {
            std::memcpy(packed, scratch, BLOCK4_TILE_SLOTS);
            new_len = BLOCK4_TILE_SLOTS;
            now_sparse = false;
        }
        const std::size_t cur_len = tile_len_at(elem_pos, fresh_byte_pos);
        // threading/size growth issues simply fail to grow rather than throwing
	// since that allows the neural net to continue online learning at high speed
	if (new_len > cur_len) {
            std::atomic_ref<std::uint32_t> grow_lock(tile_data_grow_lock);
            std::uint32_t expected = 0;
            if (!grow_lock.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
                // Another thread is currently in this same growing branch
                // (for any row) -- decline rather than risk two
                // concurrent reallocations of the same shared buffer.
                std::atomic_ref<std::uint64_t>(dropped_growth_events).fetch_add(1, std::memory_order_relaxed);
                return;
            }
            try {
                block4_resize_tile_in_row(block_layout, tile_byte_start, tile_byte_end, tile_data,
                                           br, fresh_byte_pos, cur_len, packed, new_len, max_tile_bytes);
                tile_is_sparse[elem_pos] = now_sparse ? 1 : 0;
            } catch (const std::bad_alloc&) {
                std::atomic_ref<std::uint64_t>(dropped_growth_events).fetch_add(1, std::memory_order_relaxed);
            }
            grow_lock.store(0, std::memory_order_release);
            return;
        }
        try {
            block4_resize_tile_in_row(block_layout, tile_byte_start, tile_byte_end, tile_data,
                                       br, fresh_byte_pos, cur_len, packed, new_len, max_tile_bytes);
            tile_is_sparse[elem_pos] = now_sparse ? 1 : 0;
        } catch (const std::bad_alloc&) {
            std::atomic_ref<std::uint64_t>(dropped_growth_events).fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Row-local workspace (concurrency-safe growth path)
    struct RowWorkspace {
        std::vector<uint8_t>  bytes;       // this row's tile bytes, private copy
        std::vector<uint8_t>  is_sparse;   // one flag per tile, 0-based within this row
        std::vector<uint32_t> bc;          // block-column of each tile, 0-based within this row --
                                            // needed by merge_row_workspace's eviction scan to
                                            // resolve (row, col) for importance-scale lookups
        std::size_t elem_start = 0;        // == block_layout.elem_start[br] (global elem index base)
        std::size_t row_nnz    = 0;
        // Fake single-row layout so block4_resize_tile_in_row's cross-row
        // shift branch (`row + 1 < L.rows`) is always false here
	DeltaCSRLayout local_L;
        std::vector<std::size_t> local_tbyte_start; // {0, capacity}
        std::vector<std::size_t> local_tbyte_end;   // {used}
    };

    RowWorkspace snapshot_row(std::size_t br) const {
        RowWorkspace ws;
        ws.row_nnz    = block_layout.row_nnz(br);
        ws.elem_start = block_layout.elem_start[br];
        const std::size_t used = tile_byte_end[br] - tile_byte_start[br];
        ws.bytes.assign(tile_data.begin() + std::ptrdiff_t(tile_byte_start[br]),
                         tile_data.begin() + std::ptrdiff_t(tile_byte_end[br]));
        ws.is_sparse.assign(tile_is_sparse.begin() + std::ptrdiff_t(ws.elem_start),
                             tile_is_sparse.begin() + std::ptrdiff_t(ws.elem_start + ws.row_nnz));
        ws.bc.resize(ws.row_nnz);
        {
            auto cur = row_cursor(br);
            for (std::size_t e = 0; e < ws.row_nnz; ++e) ws.bc[e] = cur.advance();
        }
        ws.local_L.rows      = 1;
        ws.local_tbyte_start = {0, used};
        ws.local_tbyte_end   = {used};
        return ws;
    }

    // Returns the byte offset of local tile index `e`'s start
    std::size_t workspace_tile_byte_pos(const RowWorkspace& ws, std::size_t e) const {
        std::size_t pos = 0;
        for (std::size_t i = 0; i < e; ++i)
            pos += block4_stored_tile_len(ws.is_sparse[i], &ws.bytes[pos]);
        return pos;
    }

    // Packs `scratch` into the workspace at local tile index `e`
    void commit_dirty_tile_in_workspace(RowWorkspace& ws, std::size_t e,
                                         std::size_t local_byte_pos,
                                         const uint8_t scratch[BLOCK4_TILE_SLOTS]) const {
        const uint32_t n = block4_count_live(scratch);
        uint8_t packed[BLOCK4_TILE_SLOTS];
        std::size_t new_len;
        bool now_sparse;
        if (n <= switch_point) {
            block4_sparse_pack(scratch, packed);
            new_len = block4_sparse_packed_len(uint8_t(n));
            now_sparse = true;
        } else {
            std::memcpy(packed, scratch, BLOCK4_TILE_SLOTS);
            new_len = BLOCK4_TILE_SLOTS;
            now_sparse = false;
        }
        const std::size_t cur_len = block4_stored_tile_len(ws.is_sparse[e], &ws.bytes[local_byte_pos]);
        block4_resize_tile_in_row(ws.local_L, ws.local_tbyte_start, ws.local_tbyte_end, ws.bytes,
                                   0, local_byte_pos, cur_len, packed, new_len,
                                   std::numeric_limits<std::size_t>::max());
        ws.is_sparse[e] = now_sparse ? 1 : 0;
    }

    // Unpacks local tile `e` into `scratch` 
    void unpack_workspace_tile(const RowWorkspace& ws, std::size_t e,
                                std::size_t local_byte_pos,
                                uint8_t scratch[BLOCK4_TILE_SLOTS]) const {
        if (ws.is_sparse[e]) block4_sparse_unpack(&ws.bytes[local_byte_pos], scratch);
        else std::memcpy(scratch, &ws.bytes[local_byte_pos], BLOCK4_TILE_SLOTS);
    }

    // Writes the workspace back into the shared store
    template <typename TrueImportanceFn>
    std::size_t merge_row_workspace(std::size_t br, RowWorkspace& ws, TrueImportanceFn&& true_importance) {
        const std::size_t global_headroom = tile_byte_start[br + 1] - tile_byte_start[br];
        std::size_t evicted = 0;
        while (ws.local_tbyte_end[0] > global_headroom) {
            // Find the globally lowest-|true-importance| LIVE slot across
            // every tile in this row.
	    // todo: ensure this is SIMD optimized instead of 3 for loops
	    //   with if statements in the deepest loop
	    double best_abs_imp = -1.0;
            std::size_t best_e = 0, best_local_pos = 0;
            uint32_t best_li = 0, best_lj = 0;
            std::size_t pos = 0;
            for (std::size_t e = 0; e < ws.row_nnz; ++e) {
                uint8_t scratch[BLOCK4_TILE_SLOTS];
                unpack_workspace_tile(ws, e, pos, scratch);
                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                        const uint8_t byte = scratch[Block4Tile::slot_index(li, lj)];
                        if (byte == 0) continue; // not live
                        const uint8_t imp_code = (byte >> 4) & 0xFu;
                        const std::size_t row = br * BLOCK4_TILE + li;
                        const std::size_t col = static_cast<std::size_t>(ws.bc[e]) * BLOCK4_TILE + lj;
                        const double abs_imp = std::abs(static_cast<double>(
                            true_importance(row, col, imp_code)));
                        if (best_abs_imp < 0.0 || abs_imp < best_abs_imp) {
                            best_abs_imp = abs_imp;
                            best_e = e; best_local_pos = pos;
                            best_li = li; best_lj = lj;
                        }
                    }
                }
                pos += block4_stored_tile_len(ws.is_sparse[e], &ws.bytes[pos]);
            }
            if (best_abs_imp < 0.0) break; // nothing live left to evict (shouldn't happen -- see caller)
            uint8_t scratch[BLOCK4_TILE_SLOTS];
            unpack_workspace_tile(ws, best_e, best_local_pos, scratch);
            scratch[Block4Tile::slot_index(best_li, best_lj)] = 0;
            commit_dirty_tile_in_workspace(ws, best_e, best_local_pos, scratch);
            ++evicted;
        }
        std::copy(ws.bytes.begin(), ws.bytes.begin() + std::ptrdiff_t(ws.local_tbyte_end[0]),
                   tile_data.begin() + std::ptrdiff_t(tile_byte_start[br]));
        tile_byte_end[br] = tile_byte_start[br] + ws.local_tbyte_end[0];
        std::copy(ws.is_sparse.begin(), ws.is_sparse.end(),
                   tile_is_sparse.begin() + std::ptrdiff_t(ws.elem_start));
        return evicted;
    }

    std::size_t n_tiles() const { return block_layout.total_nnz; }

    std::size_t total_tile_alloc_bytes() const { return tile_data.size(); }
    std::size_t total_tile_used_bytes() const {
        std::size_t n = 0;
        for (std::size_t r = 0; r < block_layout.rows; ++r) n += tile_byte_end[r] - tile_byte_start[r];
        return n;
    }

    // Cold-path reporting only (nnz(), diagnostics)
    std::size_t live_synapses() const {
        std::size_t n = 0;
        for (std::size_t r = 0; r < block_layout.rows; ++r) {
            const std::size_t start = block_layout.elem_start[r];
            const std::size_t end   = block_layout.elem_end[r];
            std::size_t byte_pos = tile_byte_start[r];
            for (std::size_t i = start; i < end; ++i) {
                if (tile_is_sparse[i]) {
                    const uint8_t count = tile_data[byte_pos];
                    n += count;
                    byte_pos += block4_sparse_packed_len(count);
                } else {
                    n += block4_count_live(&tile_data[byte_pos]);
                    byte_pos += BLOCK4_TILE_SLOTS;
                }
            }
        }
        return n;
    }
};

inline uint8_t& Block4TileHandle::at(uint32_t li, uint32_t lj) {
    dirty_ = true;
    return was_sparse_ ? scratch_[Block4Tile::slot_index(li, lj)]
                        : store_->tile_data[byte_pos_ + Block4Tile::slot_index(li, lj)];
}
inline uint8_t Block4TileHandle::at(uint32_t li, uint32_t lj) const {
    return was_sparse_ ? scratch_[Block4Tile::slot_index(li, lj)]
                        : store_->tile_data[byte_pos_ + Block4Tile::slot_index(li, lj)];
}
inline uint32_t Block4TileHandle::count_live() const {
    return was_sparse_ ? block4_count_live(scratch_) : block4_count_live(&store_->tile_data[byte_pos_]);
}
inline const uint8_t* Block4TileHandle::raw_data() const {
    return was_sparse_ ? scratch_ : &store_->tile_data[byte_pos_];
}

// ── Block4Store8 ─────────────────────────────────────────────────────────────
// FP8 (E4M3) counterpart to Block4Store/Block4TileHandle -- "alt, not
// replace": a fully separate struct, zero modification to Block4Store/
// Block4TileHandle/Block4Tile above. Reuses block4_row_shift/
// block4_grow_last_row/block4_ensure_row_headroom/block4_row_insert_tile/
// block4_row_remove_tile/block4_resize_tile_in_row UNCHANGED (confirmed,
// by reading each one directly, to be pure byte-buffer plumbing with no
// assumption about how many bytes make up one slot's value) -- only the
// tile-encoding calls (block4_sparse_pack8/unpack8/block4_stored_tile_len8/
// block4_count_live8, BLOCK4_TILE_SLOTS8_BYTES instead of BLOCK4_TILE_SLOTS)
// and the importance-byte extraction in merge_row_workspace8 (a full byte,
// not a 4-bit nibble shift) differ from Block4Store/Block4TileHandle's
// FP4-specific versions -- every method here is otherwise a mechanical,
// checked-line-by-line mirror of its Block4Store counterpart.

struct Block4Store8;

class Block4TileHandle8 {
    Block4Store8* store_ = nullptr;
    uint32_t br_ = 0, bc_ = 0;
    std::size_t byte_pos_ = 0;
    uint8_t scratch_[BLOCK4_TILE_SLOTS8_BYTES] = {0};
    bool was_sparse_ = false;
    bool dirty_ = false;
    bool valid_ = false;

public:
    Block4TileHandle8() = default;
    Block4TileHandle8(Block4Store8& store, uint32_t br, uint32_t bc);
    Block4TileHandle8(Block4Store8& store, uint32_t br, uint32_t bc, std::size_t elem_pos, std::size_t byte_pos);
    ~Block4TileHandle8();

    Block4TileHandle8(Block4TileHandle8&& other) noexcept {
        store_ = other.store_; br_ = other.br_; bc_ = other.bc_; byte_pos_ = other.byte_pos_;
        std::memcpy(scratch_, other.scratch_, sizeof(scratch_));
        was_sparse_ = other.was_sparse_; dirty_ = other.dirty_; valid_ = other.valid_;
        other.valid_ = false; other.dirty_ = false;
    }
    Block4TileHandle8& operator=(Block4TileHandle8&& other) noexcept {
        if (this == &other) return *this;
        this->~Block4TileHandle8();
        store_ = other.store_; br_ = other.br_; bc_ = other.bc_; byte_pos_ = other.byte_pos_;
        std::memcpy(scratch_, other.scratch_, sizeof(scratch_));
        was_sparse_ = other.was_sparse_; dirty_ = other.dirty_; valid_ = other.valid_;
        other.valid_ = false; other.dirty_ = false;
        return *this;
    }
    Block4TileHandle8(const Block4TileHandle8&) = delete;
    Block4TileHandle8& operator=(const Block4TileHandle8&) = delete;

    explicit operator bool() const { return valid_; }

    uint8_t& at_weight(uint32_t li, uint32_t lj);
    uint8_t  at_weight(uint32_t li, uint32_t lj) const;
    uint8_t& at_importance(uint32_t li, uint32_t lj);
    uint8_t  at_importance(uint32_t li, uint32_t lj) const;
    uint32_t count_live() const;

    // Raw read-only pointer to this tile's 32 bytes (scratch_ if sparse,
    // tile_data+byte_pos_ if dense) -- weight half [0..15], importance
    // half [16..31], matching Block4Tile8's own layout.
    const uint8_t* raw_data() const;
};

struct Block4Store8 {
    DeltaCSRLayout            block_layout;
    std::vector<uint8_t>      indices_buf;

    std::vector<uint8_t>      tile_data;
    std::vector<std::size_t>  tile_byte_start;
    std::vector<std::size_t>  tile_byte_end;
    std::vector<uint8_t>      tile_is_sparse;

    // Default BLOCK4_SPARSE_MAX_COUNT8 (12, the exact 12*20+8=248<=256
    // arithmetic -- see block4_sparse_packed_len8's own comment).
    uint32_t switch_point = BLOCK4_SPARSE_MAX_COUNT8;

    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max();
    std::size_t max_tile_bytes    = std::numeric_limits<std::size_t>::max();

    void set_limits(std::size_t indices_limit_bytes, std::size_t tile_limit_bytes) {
        max_indices_bytes = indices_limit_bytes;
        max_tile_bytes    = tile_limit_bytes;
        if (tile_limit_bytes != std::numeric_limits<std::size_t>::max()) {
            try {
                tile_data.reserve(tile_limit_bytes);
            } catch (const std::length_error&) {
            } catch (const std::bad_alloc&) {
            }
        }
    }

    std::uint64_t dropped_growth_events = 0;
    std::uint32_t tile_data_grow_lock = 0;

    std::vector<uint32_t>    scratch_tile_br, scratch_tile_bc;
    std::vector<std::size_t> scratch_tile_elem, scratch_tile_byte;
    std::vector<uint32_t>    scratch_row_live_count;
    std::vector<double>      scratch_row_grad;
    std::vector<std::size_t> scratch_row_ti_start;

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
        tile_byte_start.assign(block_layout.rows + 1, 0);
        tile_byte_end.assign(block_layout.rows, 0);
        tile_data.clear();
        tile_is_sparse.clear();
    }

    DeltaCSRRowCursor<uint32_t> row_cursor(std::size_t br) const {
        return DeltaCSRRowCursor<uint32_t>(indices_buf.data(), block_layout, br);
    }

    std::size_t tile_len_at(std::size_t elem_pos, std::size_t byte_pos) const {
        return block4_stored_tile_len8(tile_is_sparse[elem_pos], &tile_data[byte_pos]);
    }

    std::size_t raw_find(uint32_t br, uint32_t bc, std::size_t* out_elem_pos = nullptr) const {
        if (br >= block_layout.rows) return std::numeric_limits<std::size_t>::max();
        auto cur = row_cursor(br);
        const std::size_t n = block_layout.row_nnz(br);
        std::size_t elem_pos = block_layout.elem_start[br];
        std::size_t byte_pos = tile_byte_start[br];
        for (std::size_t e = 0; e < n; ++e, ++elem_pos) {
            const uint32_t col = cur.advance();
            if (col == bc) {
                if (out_elem_pos) *out_elem_pos = elem_pos;
                return byte_pos;
            }
            if (col > bc) break;
            byte_pos += tile_len_at(elem_pos, byte_pos);
        }
        return std::numeric_limits<std::size_t>::max();
    }

    bool is_sparse(uint32_t br, uint32_t bc) const {
        std::size_t elem_pos = 0;
        return raw_find(br, bc, &elem_pos) != std::numeric_limits<std::size_t>::max()
            && bool(tile_is_sparse[elem_pos]);
    }

    Block4TileHandle8 find(uint32_t br, uint32_t bc) { return Block4TileHandle8(*this, br, bc); }
    Block4TileHandle8 find(uint32_t br, uint32_t bc) const {
        return const_cast<Block4Store8*>(this)->find(br, bc);
    }

    Block4TileHandle8 at_index(uint32_t br, uint32_t bc, std::size_t elem_pos, std::size_t byte_pos) {
        return Block4TileHandle8(*this, br, bc, elem_pos, byte_pos);
    }
    Block4TileHandle8 at_index(uint32_t br, uint32_t bc, std::size_t elem_pos, std::size_t byte_pos) const {
        return const_cast<Block4Store8*>(this)->at_index(br, bc, elem_pos, byte_pos);
    }

    Block4TileHandle8 get_or_create(uint32_t br, uint32_t bc) {
        if (raw_find(br, bc) == std::numeric_limits<std::size_t>::max()) {
            if (br >= block_layout.rows)
                throw std::out_of_range(
                    "Block4Store8::get_or_create: block_row out of range -- "
                    "was Block4Store8::init(n_in, n_out) called?");
            if (!block4_row_insert_tile(block_layout, indices_buf, tile_byte_start, tile_byte_end,
                                         tile_data, tile_is_sparse, br, bc)) {
                block4_ensure_row_headroom(block_layout, indices_buf, tile_byte_start, tile_byte_end,
                                            tile_data, tile_is_sparse, br, max_indices_bytes, max_tile_bytes);
                const bool ok = block4_row_insert_tile(block_layout, indices_buf, tile_byte_start, tile_byte_end,
                                                        tile_data, tile_is_sparse, br, bc);
                (void)ok;
            }
        }
        return Block4TileHandle8(*this, br, bc);
    }

    void erase(uint32_t br, uint32_t bc) {
        if (br >= block_layout.rows) return;
        block4_row_remove_tile(block_layout, indices_buf, tile_byte_start, tile_byte_end,
                                tile_data, tile_is_sparse, br, bc);
    }

    void maybe_compress(uint32_t br, uint32_t bc) {
        std::size_t elem_pos = 0;
        const std::size_t byte_pos = raw_find(br, bc, &elem_pos);
        if (byte_pos == std::numeric_limits<std::size_t>::max() || tile_is_sparse[elem_pos]) return;
        const uint32_t n = block4_count_live8(&tile_data[byte_pos]);
        if (n > switch_point) return;
        uint8_t packed[BLOCK4_TILE_SLOTS8_BYTES];
        block4_sparse_pack8(&tile_data[byte_pos], packed);
        const std::size_t new_len = block4_sparse_packed_len8(uint8_t(n));
        block4_resize_tile_in_row(block_layout, tile_byte_start, tile_byte_end, tile_data,
                                   br, byte_pos, BLOCK4_TILE_SLOTS8_BYTES, packed, new_len, max_tile_bytes);
        tile_is_sparse[elem_pos] = 1;
    }

    void equalize_step(std::size_t& current_row) {
        if (block_layout.rows == 0) return;
        const std::size_t row = current_row % block_layout.rows;
        const std::size_t target_idx = block_layout.rows > 0
            ? (block_layout.total_alloc_bytes() + block_layout.rows - 1) / block_layout.rows : 0;
        const std::size_t target_tile = block_layout.rows > 0
            ? (tile_data.size() + block_layout.rows - 1) / block_layout.rows : 0;
        const std::size_t target_elem = block_layout.rows > 0
            ? (block_layout.total_alloc_elems() + block_layout.rows - 1) / block_layout.rows : 0;
        block4_row_shift(block_layout, indices_buf, tile_byte_start, tile_byte_end, tile_data, tile_is_sparse,
                          row, target_idx, target_tile, target_elem, max_indices_bytes, max_tile_bytes);
        current_row = (current_row + 1) % block_layout.rows;
    }

    void commit_dirty_sparse_tile(uint32_t br, uint32_t bc, const uint8_t scratch[BLOCK4_TILE_SLOTS8_BYTES]) {
        std::size_t elem_pos = 0;
        const std::size_t fresh_byte_pos = raw_find(br, bc, &elem_pos);
        if (fresh_byte_pos == std::numeric_limits<std::size_t>::max()) return;

        const uint32_t n = block4_count_live8(scratch);
        uint8_t packed[BLOCK4_TILE_SLOTS8_BYTES];
        std::size_t new_len;
        bool now_sparse;
        if (n <= switch_point) {
            block4_sparse_pack8(scratch, packed);
            new_len = block4_sparse_packed_len8(uint8_t(n));
            now_sparse = true;
        } else {
            std::memcpy(packed, scratch, BLOCK4_TILE_SLOTS8_BYTES);
            new_len = BLOCK4_TILE_SLOTS8_BYTES;
            now_sparse = false;
        }
        const std::size_t cur_len = tile_len_at(elem_pos, fresh_byte_pos);
        if (new_len > cur_len) {
            std::atomic_ref<std::uint32_t> grow_lock(tile_data_grow_lock);
            std::uint32_t expected = 0;
            if (!grow_lock.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
                std::atomic_ref<std::uint64_t>(dropped_growth_events).fetch_add(1, std::memory_order_relaxed);
                return;
            }
            try {
                block4_resize_tile_in_row(block_layout, tile_byte_start, tile_byte_end, tile_data,
                                           br, fresh_byte_pos, cur_len, packed, new_len, max_tile_bytes);
                tile_is_sparse[elem_pos] = now_sparse ? 1 : 0;
            } catch (const std::bad_alloc&) {
                std::atomic_ref<std::uint64_t>(dropped_growth_events).fetch_add(1, std::memory_order_relaxed);
            }
            grow_lock.store(0, std::memory_order_release);
            return;
        }
        try {
            block4_resize_tile_in_row(block_layout, tile_byte_start, tile_byte_end, tile_data,
                                       br, fresh_byte_pos, cur_len, packed, new_len, max_tile_bytes);
            tile_is_sparse[elem_pos] = now_sparse ? 1 : 0;
        } catch (const std::bad_alloc&) {
            std::atomic_ref<std::uint64_t>(dropped_growth_events).fetch_add(1, std::memory_order_relaxed);
        }
    }

    struct RowWorkspace {
        std::vector<uint8_t>  bytes;
        std::vector<uint8_t>  is_sparse;
        std::vector<uint32_t> bc;
        std::size_t elem_start = 0;
        std::size_t row_nnz    = 0;
        DeltaCSRLayout local_L;
        std::vector<std::size_t> local_tbyte_start;
        std::vector<std::size_t> local_tbyte_end;
    };

    RowWorkspace snapshot_row(std::size_t br) const {
        RowWorkspace ws;
        ws.row_nnz    = block_layout.row_nnz(br);
        ws.elem_start = block_layout.elem_start[br];
        const std::size_t used = tile_byte_end[br] - tile_byte_start[br];
        ws.bytes.assign(tile_data.begin() + std::ptrdiff_t(tile_byte_start[br]),
                         tile_data.begin() + std::ptrdiff_t(tile_byte_end[br]));
        ws.is_sparse.assign(tile_is_sparse.begin() + std::ptrdiff_t(ws.elem_start),
                             tile_is_sparse.begin() + std::ptrdiff_t(ws.elem_start + ws.row_nnz));
        ws.bc.resize(ws.row_nnz);
        {
            auto cur = row_cursor(br);
            for (std::size_t e = 0; e < ws.row_nnz; ++e) ws.bc[e] = cur.advance();
        }
        ws.local_L.rows      = 1;
        ws.local_tbyte_start = {0, used};
        ws.local_tbyte_end   = {used};
        return ws;
    }

    std::size_t workspace_tile_byte_pos(const RowWorkspace& ws, std::size_t e) const {
        std::size_t pos = 0;
        for (std::size_t i = 0; i < e; ++i)
            pos += block4_stored_tile_len8(ws.is_sparse[i], &ws.bytes[pos]);
        return pos;
    }

    void commit_dirty_tile_in_workspace(RowWorkspace& ws, std::size_t e,
                                         std::size_t local_byte_pos,
                                         const uint8_t scratch[BLOCK4_TILE_SLOTS8_BYTES]) const {
        const uint32_t n = block4_count_live8(scratch);
        uint8_t packed[BLOCK4_TILE_SLOTS8_BYTES];
        std::size_t new_len;
        bool now_sparse;
        if (n <= switch_point) {
            block4_sparse_pack8(scratch, packed);
            new_len = block4_sparse_packed_len8(uint8_t(n));
            now_sparse = true;
        } else {
            std::memcpy(packed, scratch, BLOCK4_TILE_SLOTS8_BYTES);
            new_len = BLOCK4_TILE_SLOTS8_BYTES;
            now_sparse = false;
        }
        const std::size_t cur_len = block4_stored_tile_len8(ws.is_sparse[e], &ws.bytes[local_byte_pos]);
        block4_resize_tile_in_row(ws.local_L, ws.local_tbyte_start, ws.local_tbyte_end, ws.bytes,
                                   0, local_byte_pos, cur_len, packed, new_len,
                                   std::numeric_limits<std::size_t>::max());
        ws.is_sparse[e] = now_sparse ? 1 : 0;
    }

    void unpack_workspace_tile(const RowWorkspace& ws, std::size_t e,
                                std::size_t local_byte_pos,
                                uint8_t scratch[BLOCK4_TILE_SLOTS8_BYTES]) const {
        if (ws.is_sparse[e]) block4_sparse_unpack8(&ws.bytes[local_byte_pos], scratch);
        else std::memcpy(scratch, &ws.bytes[local_byte_pos], BLOCK4_TILE_SLOTS8_BYTES);
    }

    // Writes the workspace back into the shared store. `true_importance`
    // receives the FULL importance BYTE (an E4M3 code, 0-255) -- unlike
    // Block4Store::merge_row_workspace's 4-bit nibble, since FP8 stores
    // a whole byte per value; the caller's own lambda decodes it via
    // fp8_decode_bits(imp_code), not FP4_TABLE[imp_code & 0xF].
    template <typename TrueImportanceFn>
    std::size_t merge_row_workspace(std::size_t br, RowWorkspace& ws, TrueImportanceFn&& true_importance) {
        const std::size_t global_headroom = tile_byte_start[br + 1] - tile_byte_start[br];
        std::size_t evicted = 0;
        while (ws.local_tbyte_end[0] > global_headroom) {
            double best_abs_imp = -1.0;
            std::size_t best_e = 0, best_local_pos = 0;
            uint32_t best_li = 0, best_lj = 0;
            std::size_t pos = 0;
            for (std::size_t e = 0; e < ws.row_nnz; ++e) {
                uint8_t scratch[BLOCK4_TILE_SLOTS8_BYTES];
                unpack_workspace_tile(ws, e, pos, scratch);
                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                        const uint32_t slot = Block4Tile8::slot_index(li, lj);
                        const uint8_t w_byte   = scratch[slot];
                        const uint8_t imp_code = scratch[BLOCK4_TILE_SLOTS + slot];
                        if (w_byte == 0 && imp_code == 0) continue; // not live
                        const std::size_t row = br * BLOCK4_TILE + li;
                        const std::size_t col = static_cast<std::size_t>(ws.bc[e]) * BLOCK4_TILE + lj;
                        const double abs_imp = std::abs(static_cast<double>(
                            true_importance(row, col, imp_code)));
                        if (best_abs_imp < 0.0 || abs_imp < best_abs_imp) {
                            best_abs_imp = abs_imp;
                            best_e = e; best_local_pos = pos;
                            best_li = li; best_lj = lj;
                        }
                    }
                }
                pos += block4_stored_tile_len8(ws.is_sparse[e], &ws.bytes[pos]);
            }
            if (best_abs_imp < 0.0) break;
            uint8_t scratch[BLOCK4_TILE_SLOTS8_BYTES];
            unpack_workspace_tile(ws, best_e, best_local_pos, scratch);
            const uint32_t best_slot = Block4Tile8::slot_index(best_li, best_lj);
            scratch[best_slot] = 0;
            scratch[BLOCK4_TILE_SLOTS + best_slot] = 0;
            commit_dirty_tile_in_workspace(ws, best_e, best_local_pos, scratch);
            ++evicted;
        }
        std::copy(ws.bytes.begin(), ws.bytes.begin() + std::ptrdiff_t(ws.local_tbyte_end[0]),
                   tile_data.begin() + std::ptrdiff_t(tile_byte_start[br]));
        tile_byte_end[br] = tile_byte_start[br] + ws.local_tbyte_end[0];
        std::copy(ws.is_sparse.begin(), ws.is_sparse.end(),
                   tile_is_sparse.begin() + std::ptrdiff_t(ws.elem_start));
        return evicted;
    }

    std::size_t n_tiles() const { return block_layout.total_nnz; }

    std::size_t total_tile_alloc_bytes() const { return tile_data.size(); }
    std::size_t total_tile_used_bytes() const {
        std::size_t n = 0;
        for (std::size_t r = 0; r < block_layout.rows; ++r) n += tile_byte_end[r] - tile_byte_start[r];
        return n;
    }

    std::size_t live_synapses() const {
        std::size_t n = 0;
        for (std::size_t r = 0; r < block_layout.rows; ++r) {
            const std::size_t start = block_layout.elem_start[r];
            const std::size_t end   = block_layout.elem_end[r];
            std::size_t byte_pos = tile_byte_start[r];
            for (std::size_t i = start; i < end; ++i) {
                if (tile_is_sparse[i]) {
                    const uint8_t count = tile_data[byte_pos];
                    n += count;
                    byte_pos += block4_sparse_packed_len8(count);
                } else {
                    n += block4_count_live8(&tile_data[byte_pos]);
                    byte_pos += BLOCK4_TILE_SLOTS8_BYTES;
                }
            }
        }
        return n;
    }
};

inline Block4TileHandle8::Block4TileHandle8(Block4Store8& store, uint32_t br, uint32_t bc)
    : store_(&store), br_(br), bc_(bc)
{
    const std::size_t bp = store_->raw_find(br_, bc_);
    if (bp == std::numeric_limits<std::size_t>::max()) { valid_ = false; return; }
    valid_ = true;
    byte_pos_ = bp;
    std::size_t elem_pos = 0;
    store_->raw_find(br_, bc_, &elem_pos);
    was_sparse_ = bool(store_->tile_is_sparse[elem_pos]);
    if (was_sparse_) block4_sparse_unpack8(&store_->tile_data[byte_pos_], scratch_);
}

inline Block4TileHandle8::Block4TileHandle8(Block4Store8& store, uint32_t br, uint32_t bc, std::size_t elem_pos, std::size_t byte_pos)
    : store_(&store), br_(br), bc_(bc), byte_pos_(byte_pos)
{
    valid_ = true;
    was_sparse_ = bool(store_->tile_is_sparse[elem_pos]);
    if (was_sparse_) block4_sparse_unpack8(&store_->tile_data[byte_pos_], scratch_);
}

inline Block4TileHandle8::~Block4TileHandle8() {
    if (!dirty_ || !valid_ || !was_sparse_) return;
    store_->commit_dirty_sparse_tile(br_, bc_, scratch_);
}

inline uint8_t& Block4TileHandle8::at_weight(uint32_t li, uint32_t lj) {
    dirty_ = true;
    const uint32_t slot = Block4Tile8::slot_index(li, lj);
    return was_sparse_ ? scratch_[slot] : store_->tile_data[byte_pos_ + slot];
}
inline uint8_t Block4TileHandle8::at_weight(uint32_t li, uint32_t lj) const {
    const uint32_t slot = Block4Tile8::slot_index(li, lj);
    return was_sparse_ ? scratch_[slot] : store_->tile_data[byte_pos_ + slot];
}
inline uint8_t& Block4TileHandle8::at_importance(uint32_t li, uint32_t lj) {
    dirty_ = true;
    const uint32_t slot = BLOCK4_TILE_SLOTS + Block4Tile8::slot_index(li, lj);
    return was_sparse_ ? scratch_[slot] : store_->tile_data[byte_pos_ + slot];
}
inline uint8_t Block4TileHandle8::at_importance(uint32_t li, uint32_t lj) const {
    const uint32_t slot = BLOCK4_TILE_SLOTS + Block4Tile8::slot_index(li, lj);
    return was_sparse_ ? scratch_[slot] : store_->tile_data[byte_pos_ + slot];
}
inline uint32_t Block4TileHandle8::count_live() const {
    return was_sparse_ ? block4_count_live8(scratch_) : block4_count_live8(&store_->tile_data[byte_pos_]);
}
inline const uint8_t* Block4TileHandle8::raw_data() const {
    return was_sparse_ ? scratch_ : &store_->tile_data[byte_pos_];
}

inline Block4TileHandle::Block4TileHandle(Block4Store& store, uint32_t br, uint32_t bc)
    : store_(&store), br_(br), bc_(bc)
{
    const std::size_t bp = store_->raw_find(br_, bc_);
    if (bp == std::numeric_limits<std::size_t>::max()) { valid_ = false; return; }
    valid_ = true;
    byte_pos_ = bp;
    std::size_t elem_pos = 0;
    store_->raw_find(br_, bc_, &elem_pos);
    was_sparse_ = bool(store_->tile_is_sparse[elem_pos]);
    if (was_sparse_) block4_sparse_unpack(&store_->tile_data[byte_pos_], scratch_);
}

inline Block4TileHandle::Block4TileHandle(Block4Store& store, uint32_t br, uint32_t bc, std::size_t elem_pos, std::size_t byte_pos)
    : store_(&store), br_(br), bc_(bc), byte_pos_(byte_pos)
{
    valid_ = true;
    was_sparse_ = bool(store_->tile_is_sparse[elem_pos]);
    if (was_sparse_) block4_sparse_unpack(&store_->tile_data[byte_pos_], scratch_);
}

// Not noexcept: a dirty sparse tile that grew past switch_point 
// can throw std::bad_alloc under budget exhaustion
//
// No try/catch: it slows the code path down and every dense tile 
// touched by disldo_backward destructs through this function.
inline Block4TileHandle::~Block4TileHandle() {
    if (!dirty_ || !valid_ || !was_sparse_) return;
    store_->commit_dirty_sparse_tile(br_, bc_, scratch_);
}
