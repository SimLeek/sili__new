#pragma once
#include "fp4quant.hpp"
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
// Note: dense is strictly faster (direct access, no pack/unpack), 
// sparse is strictly smaller for 10 or fewer synapses (11*12=132>128)
// since 2 bits are needed for each x,y index in the 4-block, 
// adding 4 bits to the 8 bits per param we already have.
//
// Default 10, settable down to 0 to disable compression entirely.
#ifndef SILI_BLOCK4_SPARSE_MAX_COUNT
#define SILI_BLOCK4_SPARSE_MAX_COUNT 10
#endif
static_assert(SILI_BLOCK4_SPARSE_MAX_COUNT * 12 + 8 <= 128,
    "sparse tile encoding must fit in the same 128 bits as the dense one");
constexpr uint32_t BLOCK4_SPARSE_MAX_COUNT = SILI_BLOCK4_SPARSE_MAX_COUNT;

// Every SIMD optimization we tried here was slower than these scalar versions
inline uint8_t block4_sparse_get_pos(const uint8_t packed[16], uint32_t i) {
    const uint8_t b = packed[1 + i / 2];
    return (i % 2 == 0) ? uint8_t(b >> 4) : uint8_t(b & 0xFu);
}
inline void block4_sparse_set_pos(uint8_t packed[16], uint32_t i, uint8_t pos) {
    uint8_t& b = packed[1 + i / 2];
    if (i % 2 == 0) b = uint8_t((b & 0x0Fu) | (pos << 4));
    else            b = uint8_t((b & 0xF0u) | (pos & 0xFu));
}

// todo: this is a fucking joke. There's nothing "packed" here if the packed size, 16, is equal
//  to the desne size, BLOCK4_TILE_SLOTS, or 16. 'packed' MUST be a pointer, NOT a fixed length array.
inline void block4_sparse_unpack(const uint8_t packed[16], uint8_t dense[BLOCK4_TILE_SLOTS]) {
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i) dense[i] = 0;
    const uint8_t count = packed[0];
    for (uint8_t i = 0; i < count; ++i)
        dense[block4_sparse_get_pos(packed, i)] = packed[6 + i];
}

// todo: fix this shit too. Both of these ops save ZERO memory and are just a waste of resources as is.
//  They should take as much or less time by shrinking the for loop 
//  and the measured used memory should be smaller.
inline void block4_sparse_pack(const uint8_t dense[BLOCK4_TILE_SLOTS], uint8_t packed[16]) {
    uint8_t count = 0;
    for (uint32_t i = 0; i < BLOCK4_TILE_SLOTS; ++i) {
        if (dense[i] == 0) continue;
        block4_sparse_set_pos(packed, count, uint8_t(i));
        packed[6 + count] = dense[i];
        ++count;
    }
    packed[0] = count;
}

// todo: this stores in a fixed size-16 array when 2-10 may be compressed
// Use a new pointer instead when is_sparse==True, so a uint8_t pointer
// Since we already use DeltaCSRLayout we're often doing near random accessor
// anyway and not always sequential, and if we actually use the database system
// like we're supposed to, the compressed sizes still should be contiguous.
// So is_sparse==false could advance BLOCK4_TILE_SLOTS on an array while 
// is_sparse==true could advance 2-10 positions on an array
struct Block4StoredTile {
    uint8_t data[BLOCK4_TILE_SLOTS] = {0};
    bool    is_sparse = false;
};

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
inline void block4_row_shift(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4StoredTile>& values, // needs vector not ValueAccessor
    std::size_t row,
    std::size_t target_byte_alloc,
    std::size_t target_elem_alloc,
    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max(),
    std::size_t max_tile_bytes = std::numeric_limits<std::size_t>::max())
{
    const std::size_t cur_byte_alloc = L.row_alloc_bytes(row);
    if (cur_byte_alloc != target_byte_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = L.byte_start[row + 1];
        const std::size_t move_len  = L.byte_start[L.rows] - move_src;
        const std::size_t new_start = L.byte_start[row] + target_byte_alloc;

        if (target_byte_alloc > cur_byte_alloc) {
            const std::size_t new_total = ibuf.size() + (target_byte_alloc - cur_byte_alloc);
            if (new_total > max_indices_bytes) throw std::bad_alloc();
            ibuf.resize(new_total);
        }
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

        if (target_elem_alloc > cur_elem_alloc) {
            const std::size_t new_total_elems = current_total + (target_elem_alloc - cur_elem_alloc);
            if (new_total_elems * sizeof(Block4StoredTile) > max_tile_bytes) throw std::bad_alloc();
            values.resize(new_total_elems);
        }
        if (move_len > 0)
            // An uncompressed Block4Tile is trivially copyable (a plain uint8_t[16]), so a
            // raw memmove here is exactly as safe as std::vector's own
            // internal moves would be
            std::memmove(values.data() + new_start, values.data() + move_src,
                         move_len * sizeof(Block4StoredTile));
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

// See block4_row_shift's identical comment -- same real bug, same fix.
inline void block4_grow_last_row(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4StoredTile>& values,
    std::size_t target_byte_alloc,
    std::size_t target_elem_alloc,
    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max(),
    std::size_t max_tile_bytes = std::numeric_limits<std::size_t>::max())
{
    if (L.rows == 0) return;
    const std::size_t r = L.rows - 1;
    const std::size_t cur_b = L.row_alloc_bytes(r);
    const std::size_t cur_e = L.row_alloc_elems(r);
    if (target_byte_alloc > cur_b) {
        const std::size_t new_total = ibuf.size() + (target_byte_alloc - cur_b);
        if (new_total > max_indices_bytes) throw std::bad_alloc();
        ibuf.resize(new_total, uint8_t(0));
        L.byte_start[L.rows] = L.byte_start[r] + target_byte_alloc;
    }
    if (target_elem_alloc > cur_e) {
        const std::size_t new_total = L.elem_start[r] + target_elem_alloc;
        if (new_total * sizeof(Block4StoredTile) > max_tile_bytes) throw std::bad_alloc();
        values.resize(new_total);
        L.elem_start[L.rows] = new_total;
    }
}

inline void block4_ensure_row_headroom(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4StoredTile>& values,
    std::size_t row,
    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max(),
    std::size_t max_tile_bytes = std::numeric_limits<std::size_t>::max())
{
    const std::size_t target_b = L.row_alloc_bytes(row) + uleb128_max_bytes<uint32_t>();
    const std::size_t target_e = L.row_alloc_elems(row) + 1;
    if (row + 1 < L.rows)
        block4_row_shift(L, ibuf, values, row, target_b, target_e, max_indices_bytes, max_tile_bytes);
    else
        block4_grow_last_row(L, ibuf, values, target_b, target_e, max_indices_bytes, max_tile_bytes);
}

// Mirrors delta_csr_row_insert_col's exact algorithm (delta_csr_memory.hpp)
// but writes a whole Block4Tile directly instead of going through
// ValueAccessor<VALUES_TYPE>::set(weight, importance).
inline bool block4_row_insert_tile(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4StoredTile>& values,
    std::size_t row,
    uint32_t new_col,
    const Block4StoredTile& tile)
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
                     (L.elem_end[row] - ins_elem_pos) * sizeof(Block4StoredTile));
    values[ins_elem_pos] = tile;
    L.elem_end[row]++;
    L.total_nnz++;
    return true;
}

// Mirrors delta_csr_row_remove_col's exact algorithm
// (delta_csr_memory.hpp) -- same delta-merge/byte-shift-left shape -- for
// `std::vector<Block4StoredTile>` directly.
inline bool block4_row_remove_tile(
    DeltaCSRLayout& L,
    std::vector<uint8_t>& ibuf,
    std::vector<Block4StoredTile>& values,
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
                             (row_end - elem_pos - 1) * sizeof(Block4StoredTile));
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

// RAII accessor for one logical tile's synapse data,
class Block4TileHandle {
    Block4Store* store_ = nullptr;
    // Cached at construction for optimization
    Block4StoredTile* stored_ = nullptr;
    uint32_t br_ = 0, bc_ = 0;
    uint8_t scratch_[BLOCK4_TILE_SLOTS] = {0};
    bool was_sparse_ = false;
    bool dirty_ = false;
    bool valid_ = false;

public:
    Block4TileHandle() = default;
    Block4TileHandle(Block4Store& store, uint32_t br, uint32_t bc);
    // Fast-path constructor
    Block4TileHandle(Block4Store& store, uint32_t br, uint32_t bc, std::size_t elem_pos);
    ~Block4TileHandle();

    Block4TileHandle(Block4TileHandle&& other) noexcept { *this = std::move(other); }
    Block4TileHandle& operator=(Block4TileHandle&& other) noexcept {
        if (this == &other) return *this;
        // Flush any pending write of THIS handle before taking over
        // other's state -- moving must not silently drop a re-pack.
        this->~Block4TileHandle();
        store_ = other.store_; stored_ = other.stored_; br_ = other.br_; bc_ = other.bc_;
        std::memcpy(scratch_, other.scratch_, sizeof(scratch_));
        was_sparse_ = other.was_sparse_; dirty_ = other.dirty_; valid_ = other.valid_;
        other.valid_ = false; other.dirty_ = false; // moved-from: destructor becomes a no-op
        return *this;
    }
    Block4TileHandle(const Block4TileHandle&) = delete;
    Block4TileHandle& operator=(const Block4TileHandle&) = delete;

    explicit operator bool() const { return valid_; }

    uint8_t& at(uint32_t li, uint32_t lj) {
        dirty_ = true;
        return was_sparse_ ? scratch_[Block4Tile::slot_index(li, lj)]
                            : stored_->data[Block4Tile::slot_index(li, lj)];
    }
    uint8_t at(uint32_t li, uint32_t lj) const {
        return was_sparse_ ? scratch_[Block4Tile::slot_index(li, lj)]
                            : stored_->data[Block4Tile::slot_index(li, lj)];
    }
    uint32_t count_live() const {
        return was_sparse_ ? block4_count_live(scratch_) : block4_count_live(stored_->data);
    }

    // Raw read-only pointer to this tile's 16 bytes (scratch_ if sparse,
    // stored_->data if dense), for faster reads
    const uint8_t* raw_data() const { return was_sparse_ ? scratch_ : stored_->data; }
};

struct Block4Store {
    DeltaCSRLayout                 block_layout;   // rows/cols are BLOCK-granularity (ceil(n_in/4), ceil(n_out/4))
    std::vector<uint8_t>           indices_buf;    // uleb128-encoded block-col deltas
    std::vector<Block4StoredTile>  tile_values;    // parallel to block_layout's elem_start/elem_end

    // default BLOCK4_SPARSE_MAX_COUNT (10, the exact 10*12+8=128 arithmetic)
    // saves the most memory, but lowering may increase speed
    // todo: fix the other parts so this ACTUALLY saves memory
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

    // Persistent scratch buffers for disldo_forward/disldo_backward
    std::vector<uint32_t>    scratch_tile_br, scratch_tile_bc;
    std::vector<std::size_t> scratch_tile_elem;
    std::vector<uint32_t>    scratch_row_live_count; // backward only
    std::vector<double>      scratch_row_grad;        // backward only

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
        tile_values.clear();
    }

    DeltaCSRRowCursor<uint32_t> row_cursor(std::size_t br) const {
        return DeltaCSRRowCursor<uint32_t>(indices_buf.data(), block_layout, br);
    }

    // Raw storage access --  used internally by Block4TileHandle itself, 
    // which needs the actual storage slot to read its is_sparse flag 
    // and either unpack it or hand out a direct pointer). 
    // Exposed as a regular public member because
    // Block4TileHandle's out-of-line methods need it
    Block4StoredTile* raw_find(uint32_t br, uint32_t bc) {
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

    Block4TileHandle find(uint32_t br, uint32_t bc) { return Block4TileHandle(*this, br, bc); }
    Block4TileHandle find(uint32_t br, uint32_t bc) const {
        return const_cast<Block4Store*>(this)->find(br, bc);
    }

    // Fast path for a caller that already walked row_cursor(br) itself
    Block4TileHandle at_index(uint32_t br, uint32_t bc, std::size_t elem_pos) {
        return Block4TileHandle(*this, br, bc, elem_pos);
    }
    Block4TileHandle at_index(uint32_t br, uint32_t bc, std::size_t elem_pos) const {
        return const_cast<Block4Store*>(this)->at_index(br, bc, elem_pos);
    }

    Block4TileHandle get_or_create(uint32_t br, uint32_t bc) {
        if (raw_find(br, bc) == nullptr) {
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
            if (!block4_row_insert_tile(block_layout, indices_buf, tile_values, br, bc, Block4StoredTile{})) {
                // Real cap enforcement -- see block4_row_shift's comment.
                // Throws std::bad_alloc if this tile would push block4's
                // OWN budget (set_limits(), independent from the
                // scattered CSR side's) past its configured max.
                block4_ensure_row_headroom(block_layout, indices_buf, tile_values, br, max_indices_bytes, max_tile_bytes);
                const bool ok = block4_row_insert_tile(block_layout, indices_buf, tile_values, br, bc, Block4StoredTile{});
                (void)ok; // block4_ensure_row_headroom grows by exactly enough for one more tile -- this must succeed
            }
        }
        return Block4TileHandle(*this, br, bc);
    }

    void erase(uint32_t br, uint32_t bc) {
        if (br >= block_layout.rows) return;
        block4_row_remove_tile(block_layout, indices_buf, tile_values, br, bc);
    }

    // Explicit compression check
    // todo: if the is_sparse size ever changed, make a pointer to 
    // 'current_location' and call this on each memory management step,
    // not synaptogenesis/pruning step.
    void maybe_compress(uint32_t br, uint32_t bc) {
        Block4StoredTile* stored = raw_find(br, bc);
        if (!stored || stored->is_sparse) return;
        const uint32_t n = block4_count_live(stored->data);
        if (n > switch_point) return;
        uint8_t packed[BLOCK4_TILE_SLOTS];
        block4_sparse_pack(stored->data, packed);
        std::memcpy(stored->data, packed, BLOCK4_TILE_SLOTS);
        stored->is_sparse = true;
    }

    std::size_t n_tiles() const { return block_layout.total_nnz; }

    // Cold-path reporting only (nnz(), diagnostics) -- O(n_tiles * 16),
    // Must walk per-row (elem_start[r].. elem_end[r]) -- rows have blank
    // (unused) element slots between them, same as the scattered path's
    // own values array. Sparse-mode tiles' count is free (the stored
    // count byte, packed[0]) -- no unpack needed just to report it.
    std::size_t live_synapses() const {
        std::size_t n = 0;
        for (std::size_t r = 0; r < block_layout.rows; ++r) {
            const std::size_t start = block_layout.elem_start[r];
            const std::size_t end   = block_layout.elem_end[r];
            for (std::size_t i = start; i < end; ++i) {
                const Block4StoredTile& t = tile_values[i];
                n += t.is_sparse ? std::size_t(t.data[0]) : block4_count_live(t.data);
            }
        }
        return n;
    }
};

inline Block4TileHandle::Block4TileHandle(Block4Store& store, uint32_t br, uint32_t bc)
    : store_(&store), br_(br), bc_(bc)
{
    stored_ = store_->raw_find(br_, bc_);
    if (!stored_) { valid_ = false; return; }
    valid_ = true;
    was_sparse_ = stored_->is_sparse;
    if (was_sparse_) block4_sparse_unpack(stored_->data, scratch_);
}

inline Block4TileHandle::Block4TileHandle(Block4Store& store, uint32_t br, uint32_t bc, std::size_t elem_pos)
    : store_(&store), br_(br), bc_(bc)
{
    stored_ = &store_->tile_values[elem_pos];
    valid_ = true;
    was_sparse_ = stored_->is_sparse;
    if (was_sparse_) block4_sparse_unpack(stored_->data, scratch_);
}

inline Block4TileHandle::~Block4TileHandle() {
    if (!dirty_ || !valid_ || !was_sparse_) return;
    Block4StoredTile* stored = store_->raw_find(br_, bc_);
    if (!stored) return; // was already erased
    const uint32_t n = block4_count_live(scratch_);
    if (n <= store_->switch_point) {
        block4_sparse_pack(scratch_, stored->data);
        stored->is_sparse = true;
    } else {
        std::memcpy(stored->data, scratch_, BLOCK4_TILE_SLOTS);
        stored->is_sparse = false;
    }
}
