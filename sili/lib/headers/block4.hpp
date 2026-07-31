#pragma once
#include "fp4quant.hpp"
#include <atomic>
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
// or fewer live synapses (11*12=132>128, the same bound this codebase
// already used to size the old fixed-16-byte slot).
//
// Layout, self-describing (no separate length table needed to skip
// past one during a row walk): byte[0] = count. Then ceil(count/2)
// nibble-packed position bytes (4 bits/position -- a 4x4=16-slot tile
// needs 4 bits/slot-index). Then `count` value bytes (importance<<4|
// weight, same byte format as a dense slot). Total bytes =
// 1 + ceil(count/2) + count -- e.g. count=2 is 1+1+2 = 4 bytes (32
// bits), count=10 is 1+5+10 = 16 bytes (matches dense exactly, the
// historical boundary where compression stops paying off).
//
// Default 10, settable down to 0 to disable compression entirely.
#ifndef SILI_BLOCK4_SPARSE_MAX_COUNT
#define SILI_BLOCK4_SPARSE_MAX_COUNT 10
#endif
static_assert(SILI_BLOCK4_SPARSE_MAX_COUNT * 12 + 8 <= 128,
    "sparse tile encoding must fit in the same 128 bits as the dense one");
constexpr uint32_t BLOCK4_SPARSE_MAX_COUNT = SILI_BLOCK4_SPARSE_MAX_COUNT;

// Total byte length of a sparse-packed tile holding `count` live
// synapses -- self-describing from the count byte alone (Block4Store
// reads this to advance a row walk past a tile without unpacking it).
inline std::size_t block4_sparse_packed_len(uint8_t count) {
    return std::size_t(1) + (std::size_t(count) + 1) / 2 + std::size_t(count);
}

// Every SIMD optimization we tried here was slower than these scalar versions.
// `packed` is a POINTER, not a fixed-length array -- this is the whole point
// of the redesign: a low-occupancy tile's storage (see Block4Store::tile_data)
// is genuinely smaller than BLOCK4_TILE_SLOTS bytes, not repacked within an
// already-dense-sized slot.
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

// Byte length of the tile currently stored at `tile_bytes` -- self-
// describing when sparse (its own count byte), fixed when dense.
inline std::size_t block4_stored_tile_len(bool is_sparse, const uint8_t* tile_bytes) {
    return is_sparse ? block4_sparse_packed_len(tile_bytes[0]) : std::size_t(BLOCK4_TILE_SLOTS);
}

// Real, enforced cap -- max_indices_bytes/max_tile_bytes default to
// SIZE_MAX (unbounded, matching every existing caller that doesn't
// pass them) but Block4Store::get_or_create/ensure_row_headroom always
// pass its OWN configured budget. Fixes a real, measured bug: this
// function (called every synaptogenesis cycle a tile gets promoted or
// grows) used to call ibuf.resize()/values.resize() unconditionally,
// with NO check against any budget at all -- confirmed via a stress
// test showing block4 growing to ~1.9x a layer's max_weights with zero
// resistance, entirely through this path (the scattered CSR side's
// OWN equivalent bug -- delta_csr_shift_row -- was fixed separately;
// see conversation / TODO_DUAL_BLOCK4.md).
//
// Three independently-growable regions per row, mirrored here exactly
// like the scattered CSR side's own byte-vs-elem split:
//   - ibuf: uleb128-encoded block-column deltas (unchanged from before).
//   - tile_data: the actual tile bytes -- variable length per tile now
//     (see block4_stored_tile_len), so this needs its OWN row byte
//     layout (tbyte_start/tbyte_end), separate from L's (which is for
//     ibuf only).
//   - tile_is_sparse: 1 byte/tile flag, shares L's elem address space
//     with ibuf's entry count (1 index = 1 tile, always).
inline void block4_row_shift(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<std::size_t>& tbyte_start,
    std::vector<std::size_t>& tbyte_end,
    std::vector<uint8_t>& tile_data,
    std::vector<uint8_t>& tile_is_sparse,
    std::size_t row,
    std::size_t target_idx_byte_alloc,
    std::size_t target_tile_byte_alloc,
    std::size_t target_elem_alloc,
    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max(),
    std::size_t max_tile_bytes = std::numeric_limits<std::size_t>::max())
{
    // ── indices_buf byte side (uleb128 column deltas) ──────────────────────
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

    // ── tile_data byte side (variable-length packed/dense tile bytes) ──────
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

    // ── tile_is_sparse elem side (1 byte/slot flag) ─────────────────────────
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

// See block4_row_shift's identical comment -- same real bug, same fix.
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
    // New tiles always start dense (BLOCK4_TILE_SLOTS bytes) -- see
    // block4_row_insert_tile.
    const std::size_t target_tile = (tbyte_start[row + 1] - tbyte_start[row]) + BLOCK4_TILE_SLOTS;
    const std::size_t target_elem = L.row_alloc_elems(row) + 1;
    if (row + 1 < L.rows)
        block4_row_shift(L, ibuf, tbyte_start, tbyte_end, tile_data, tile_is_sparse, row,
                          target_idx, target_tile, target_elem, max_indices_bytes, max_tile_bytes);
    else
        block4_grow_last_row(L, ibuf, tbyte_start, tbyte_end, tile_data, tile_is_sparse,
                              target_idx, target_tile, target_elem, max_indices_bytes, max_tile_bytes);
}

// Mirrors delta_csr_row_insert_col's exact algorithm (delta_csr_memory.hpp).
// New tiles always start dense and all-zero (BLOCK4_TILE_SLOTS bytes) --
// the caller writes into it afterward via Block4TileHandle.
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
    const std::size_t new_tile_len = BLOCK4_TILE_SLOTS; // new tiles always start dense
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
        std::memset(tile_data.data() + ins_tbyte_pos, 0, new_tile_len);
        tbyte_end[row] += new_tile_len;
    }

    if (ins_elem_pos < L.elem_end[row])
        std::memmove(tile_is_sparse.data() + ins_elem_pos + 1, tile_is_sparse.data() + ins_elem_pos,
                     L.elem_end[row] - ins_elem_pos);
    tile_is_sparse[ins_elem_pos] = 0; // false = dense

    L.elem_end[row]++;
    L.total_nnz++;
    return true;
}

// Mirrors delta_csr_row_remove_col's exact algorithm (delta_csr_memory.hpp)
// -- same delta-merge/byte-shift-left shape -- now also shrinking tile_data
// by the removed tile's own (variable) byte length.
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

// Replaces the tile currently occupying `old_len` bytes at tile_data[tbyte_pos]
// (within `row`) with `new_bytes` (length `new_len`), growing the row's
// tile_data allocation if needed (checked against max_tile_bytes -- throws
// std::bad_alloc on real budget exhaustion, same convention as every other
// block4/scattered-CSR growth path in this codebase) and shifting every
// LATER tile in the same row (and, if the row itself had to grow, every
// later ROW's tbyte_start/tbyte_end) to make room. Shrinking never needs
// new room and never throws. The throw check happens before any mutation,
// so a thrown call leaves tile_data/tbyte_start/tbyte_end untouched.
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

// RAII accessor for one logical tile's synapse data. A dense tile mutates
// tile_data in place (always exactly BLOCK4_TILE_SLOTS bytes -- never
// resizes). A sparse tile unpacks into scratch_ at construction and, if
// dirtied, re-packs (or promotes to dense) at destruction -- see
// ~Block4TileHandle()'s comment for what happens when that genuinely
// needs to grow this tile's footprint but the store's budget won't allow
// it (declined, not thrown -- see Block4Store::dropped_growth_events).
class Block4TileHandle {
    Block4Store* store_ = nullptr;
    uint32_t br_ = 0, bc_ = 0;
    std::size_t byte_pos_ = 0;  // position in store_->tile_data, valid at construction
    uint8_t scratch_[BLOCK4_TILE_SLOTS] = {0};
    bool was_sparse_ = false;
    bool dirty_ = false;
    bool valid_ = false;

public:
    Block4TileHandle() = default;
    Block4TileHandle(Block4Store& store, uint32_t br, uint32_t bc);
    // Fast-path constructor: caller already knows both this tile's
    // elem_pos (shared index space with indices_buf) and its byte_pos in
    // tile_data (from its own sequential row walk -- see
    // linear_disldo.hpp's collection loop) -- skips raw_find()'s redundant
    // O(row_nnz) re-scan.
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
    DeltaCSRLayout            block_layout;    // rows/cols are BLOCK-granularity (ceil(n_in/4), ceil(n_out/4))
    std::vector<uint8_t>      indices_buf;     // uleb128-encoded block-col deltas

    // Real variable-length tile storage: tile_data is a flat byte buffer,
    // tbyte_start/tbyte_end give each block-row's OWN byte range within it
    // (mirrors block_layout's byte_start/byte_end, which is for
    // indices_buf only -- a tile's byte length varies with its live count
    // when sparse, so it needs its own row byte accounting, separate from
    // the index side's). tile_is_sparse is 1 byte/tile, sharing
    // block_layout's elem address space with indices_buf's own entry
    // count (1 index = 1 tile, always) -- explicit, not inferred from
    // byte length (a sparse tile at count==BLOCK4_SPARSE_MAX_COUNT is the
    // same length as dense).
    std::vector<uint8_t>      tile_data;
    std::vector<std::size_t>  tile_byte_start; // size rows+1
    std::vector<std::size_t>  tile_byte_end;   // size rows
    std::vector<uint8_t>      tile_is_sparse;  // parallel to block_layout's elem_start/elem_end

    // Default BLOCK4_SPARSE_MAX_COUNT (10, the exact 10*12+8=128
    // arithmetic) saves the most memory, but lowering may increase speed.
    uint32_t switch_point = BLOCK4_SPARSE_MAX_COUNT;

    // Real, enforced growth cap -- a SEPARATE, independent budget from
    // the scattered CSR side's own max_indices_bytes/max_values_bytes
    // (DeltaCSRWeights), per direction: block4 and scattered represent
    // the same underlying weights, but sharing one combined budget
    // between the two representations would need real cross-structure
    // accounting (which one currently "owns" how many bytes of a
    // shared pool) -- deferred as a real, acknowledged simplification,
    // not the safest but simplest correct choice for now. Defaults to
    // unbounded (SIZE_MAX) for any Block4Store that never calls
    // set_limits() -- e.g. hand-built test fixtures -- matching
    // DeltaCSRWeights's own default-unbounded convention.
    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max();
    std::size_t max_tile_bytes    = std::numeric_limits<std::size_t>::max();

    void set_limits(std::size_t indices_limit_bytes, std::size_t tile_limit_bytes) {
        max_indices_bytes = indices_limit_bytes;
        max_tile_bytes    = tile_limit_bytes;
    }

    // Incremented (via std::atomic_ref -- see ~Block4TileHandle()) every
    // time a VALUE update to an EXISTING tile can't be persisted because
    // growing its storage would exceed max_tile_bytes. By design this is
    // the ONLY signal for that condition -- no exception is thrown (see
    // ~Block4TileHandle()'s comment for why: a backward/value-update op
    // dropping one write and continuing is far better for a system meant
    // to run continuously than aborting the whole call). Plain
    // std::uint64_t, not std::atomic<std::uint64_t>, so Block4Store stays
    // trivially copyable -- multiple threads in disldo_backward's row-
    // partitioned parallel loop can each increment this independently via
    // std::atomic_ref, same guarantee as a real atomic field without
    // giving up copyability. A caller doing its own memory management
    // (equalizer_step()/expand_headroom(), or just deciding the budget
    // needs to grow) can poll this after a call to detect and react to
    // dropped updates -- structural growth (get_or_create/
    // block4_maybe_promote inserting a genuinely NEW tile, or the
    // scattered CSR side's own equivalent) is NOT covered by this and
    // still throws std::bad_alloc as before -- that's a "handler decides
    // what to do" case, not a "keep training running regardless" one.
    std::uint64_t dropped_growth_events = 0;

    // Persistent scratch buffers for disldo_forward/disldo_backward
    std::vector<uint32_t>    scratch_tile_br, scratch_tile_bc;
    std::vector<std::size_t> scratch_tile_elem, scratch_tile_byte;
    std::vector<uint32_t>    scratch_row_live_count; // backward only
    std::vector<double>      scratch_row_grad;        // backward only
    // backward only: scratch_tile_br/bc/elem/byte are filled in row-major
    // order (see linear_disldo.hpp's collection loop), so row br's tiles
    // occupy [scratch_row_ti_start[br], scratch_row_ti_start[br+1]) --
    // lets backward's parallel loop partition work BY ROW instead of by
    // flat tile index, so two threads can never touch the same row's
    // tile_data concurrently (real tile resizing shifts every later tile
    // in its row -- see block4_resize_tile_in_row -- so row-exclusive
    // ownership, not a flat static split, is what makes it safe to let
    // tiles genuinely compress/decompress based on real content inside
    // the parallel region, instead of forcing every touched tile dense
    // beforehand regardless of what the loss actually did to it).
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

    // Byte length of the tile at (elem_pos, byte_pos) -- exposed so
    // callers walking a row themselves (linear_disldo.hpp) can advance
    // their own byte_pos cursor without duplicating block4_stored_tile_len.
    std::size_t tile_len_at(std::size_t elem_pos, std::size_t byte_pos) const {
        return block4_stored_tile_len(tile_is_sparse[elem_pos], &tile_data[byte_pos]);
    }

    // Raw storage lookup -- used internally by Block4TileHandle. Returns
    // this tile's byte offset into tile_data (SIZE_MAX if (br,bc) has no
    // live tile), and writes its elem_pos to *out_elem_pos if non-null.
    // Exposed as a regular public member because Block4TileHandle's
    // out-of-line methods need it.
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
            // A Block4Store that was never sized via
            // init(n_in, n_out) has block_layout.rows == 0, so
            // block_layout.byte_start/elem_start are both empty --
            // silently proceeding to block4_row_insert_tile below would
            // index them out of bounds.
	    // block4_maybe_promote (delta_csr_memory.hpp)
            // lazily self-inits before ever reaching here; a caller
            // invoking get_or_create() directly doesn't have that safety
            // net, since this function has no n_in/n_out to lazily size
            // to on its own -- fail loud instead of corrupting memory.
            if (br >= block_layout.rows)
                throw std::out_of_range(
                    "Block4Store::get_or_create: block_row out of range -- "
                    "was Block4Store::init(n_in, n_out) called?");
            if (!block4_row_insert_tile(block_layout, indices_buf, tile_byte_start, tile_byte_end,
                                         tile_data, tile_is_sparse, br, bc)) {
                // Real cap enforcement -- see block4_row_shift's comment.
                // Throws std::bad_alloc if this tile would push block4's
                // OWN budget (set_limits(), independent from the
                // scattered CSR side's) past its configured max.
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

    // Explicit compression check -- never automatic on every write, only
    // called at synaptogenesis/pruning checkpoints (see call sites in
    // delta_csr_memory.hpp).
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

    // Out-of-line, called only from ~Block4TileHandle()'s was_sparse_
    // branch (rare -- most handles are the dense fast path, which never
    // reaches this at all). Kept SEPARATE from the destructor itself,
    // not inlined into it, specifically so the try/catch this needs
    // (see its own comment) can't affect codegen for the destructor's
    // own hot, common early-return path -- measured, not assumed: with
    // the try/catch inlined directly into ~Block4TileHandle(), backward's
    // speedup over the dense floor at 100% fill dropped from ~1.97x to
    // ~1.82x even though every tile in that benchmark is dense and never
    // reaches the try block at all (scripts/bench_block4_vs_dense_fp4.cpp).
    void commit_dirty_sparse_tile(uint32_t br, uint32_t bc, const uint8_t scratch[BLOCK4_TILE_SLOTS]) {
        // Re-fetch by COORDINATE, not a cached pointer/offset -- another
        // call could have inserted/removed a DIFFERENT tile in this same
        // row in between, shifting every later tile's byte position (see
        // block4_row_insert_tile/remove_tile). Concurrent erasure of THIS
        // SAME tile is the other real hazard this guards (see
        // test_erase_while_handle_alive).
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
        // Real budget exhaustion during this resize is handled by
        // DECLINING the growth, not throwing: per direction, a value-
        // update op (this represents applying an already-computed
        // backward/promotion write) dropping just that one write and
        // letting the caller keep running is far better for a system
        // meant to run continuously than aborting the whole call over
        // one tile out of possibly thousands. The tile keeps its OLD
        // stored bytes/size exactly as they were before this touch --
        // block4_resize_tile_in_row's budget check runs before any
        // mutation, so a caught std::bad_alloc means nothing was written
        // at all, never a partial write. Indicated via
        // dropped_growth_events so a caller doing its own memory
        // management can still detect and react to this -- structural
        // growth (get_or_create/block4_maybe_promote inserting a
        // genuinely NEW tile) is a DIFFERENT code path and still throws
        // std::bad_alloc as before; that's a "handler decides what to do
        // before proceeding" case (synaptogenesis), not a "value update,
        // keep going regardless" one.
        try {
            block4_resize_tile_in_row(block_layout, tile_byte_start, tile_byte_end, tile_data,
                                       br, fresh_byte_pos, cur_len, packed, new_len, max_tile_bytes);
            tile_is_sparse[elem_pos] = now_sparse ? 1 : 0;
        } catch (const std::bad_alloc&) {
            std::atomic_ref<std::uint64_t>(dropped_growth_events).fetch_add(1, std::memory_order_relaxed);
        }
    }

    std::size_t n_tiles() const { return block_layout.total_nnz; }

    // Real total bytes currently allocated for tile storage (across every
    // row's own headroom) -- the actual memory footprint relevant to
    // max_tile_bytes, for reporting/benchmarking.
    std::size_t total_tile_alloc_bytes() const { return tile_data.size(); }
    // Real total bytes currently USED (no row headroom slack) -- the
    // number that should shrink when compression genuinely helps.
    std::size_t total_tile_used_bytes() const {
        std::size_t n = 0;
        for (std::size_t r = 0; r < block_layout.rows; ++r) n += tile_byte_end[r] - tile_byte_start[r];
        return n;
    }

    // Cold-path reporting only (nnz(), diagnostics). Must walk per-row
    // (elem_start[r]..elem_end[r]) -- rows have blank (unused) element
    // slots between them, same as the scattered path's own values array.
    // Sparse-mode tiles' count is free (the stored count byte) -- no
    // unpack needed just to report it.
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

// NOT noexcept: a dirty sparse tile that grew past switch_point (or just
// changed byte length while staying sparse -- see block4_sparse_packed_len,
// which scales with the tile's ACTUAL live count, not a fixed per-store
// size) needs block4_resize_tile_in_row to potentially grow this row's
// tile_data allocation, which can throw std::bad_alloc under real budget
// exhaustion -- same convention as every other growth path in this
// codebase (get_or_create, equalizer_step, ...). A dense handle's
// destructor never reaches this (dense tiles mutate in place, always
// exactly BLOCK4_TILE_SLOTS bytes, never resized), so this only fires for
// the was_sparse_ case.
//
// A resize here physically shifts every LATER tile in this row (see
// block4_resize_tile_in_row) -- unsafe if another thread could be
// reading/writing that same row concurrently. disldo_backward
// (linear_disldo.hpp) handles this by partitioning its parallel loop BY
// BLOCK-ROW (each thread exclusively owns whole rows, never sharing one
// with another thread) instead of forcing every touched tile dense
// beforehand regardless of real content -- an earlier version did the
// latter (see TODO_DUAL_BLOCK4.md's "force_dense_at" writeup for why
// that was wrong: it discarded genuine, persistent compression from
// sparsity-encouraging losses like L1/KL on every single touch).
//
// Kept tiny and try/catch-free on purpose -- see
// Block4Store::commit_dirty_sparse_tile's comment (block4.hpp) for why
// the actual repack/resize/decline logic lives in a separate, out-of-
// line function instead of inline here: a try/catch anywhere in this
// destructor's body measurably hurt codegen for its own common,
// early-return (dense/clean/invalid) path, which is by far the hottest
// one (every dense tile touched by disldo_backward destructs through
// exactly this function).
inline Block4TileHandle::~Block4TileHandle() {
    if (!dirty_ || !valid_ || !was_sparse_) return;
    store_->commit_dirty_sparse_tile(br_, bc_, scratch_);
}
