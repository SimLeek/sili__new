#ifndef __DELTA_CSR_FOR_HPP_
#define __DELTA_CSR_FOR_HPP_

// Frame-of-reference (FOR) delta encoding for DeltaCSRWeights column
// indices -- an alternative to delta_csr_types.hpp's sequential ULEB128
// encoding (DeltaCSRRowCursor), decode-focused for now (see SCOPE below).
//
// ULEB128 encodes each column index as a delta from the PREVIOUS index,
// so reconstructing absolute values requires a running cumulative sum --
// a genuinely sequential dependency chain (confirmed via
// `-fopt-info-vec-missed` that this specific loop cannot auto-vectorize).
// Seven SIMD strategies were tried against that encoding; six converged
// on the same ~1.6-2.5x ceiling because every one of them was still
// fighting that same dependency, just with progressively cleverer SIMD
// around it (see sili_peridot's JOURNAL.md for the full trail). This
// file implements the one that actually worked: group values into
// blocks of G, encode each value as an OFFSET FROM ITS GROUP'S OWN
// STARTING VALUE (not from the previous value). Since column indices
// are monotonic (deltas are always positive), every value in a group is
// independently computable as `group_start + offset[i]`, with NO
// dependency on any other value in the group. Decode: widen G
// fixed-width offsets, ONE broadcast-add of group_start, done -- no
// shift-add prefix-sum tree at all. The group's own last (already-
// computed) output value doubles as the next group's group_start for
// free -- the cross-group dependency that a two-pass design needs a
// whole separate pass for costs nothing extra here.
//
// Real, correctness-verified benchmark: 4-12x (mostly 5-7x at G=32/64)
// vs. the current ULEB128 decode, at real per-row nnz scale -- see
// prototypes/for_delta_encoding/ for the standalone reference this was
// validated with before landing here, including the size-overhead
// tradeoff table across G (bigger groups amortize the per-group tier
// descriptor better at low multi-byte-delta rates, but pay more when a
// single large delta forces the WHOLE group to a wider tier).
//
// Group size is deliberately a per-call CHOICE (ForGroupSize below),
// not a single hardcoded constant -- real sparsity/locality patterns
// (e.g. whether synapses cluster near the diagonal) aren't established
// yet for this library's actual workloads, so different weight tensors
// may want different G. for_encode_row/for_decode_row take G as a
// template parameter for full compile-time optimization of the hot
// decode loop; for_*_row_dispatch below wraps a runtime ForGroupSize
// choice over a curated set of pre-compiled instantiations, for callers
// that pick G per-tensor at model-build time rather than per-compile.
//
// SCOPE (read before using): this covers ENCODE-once/DECODE-many,
// matching a batched forward pass over an already-built row. It does
// NOT YET address synaptogenesis-driven row GROWTH (inserting a new
// synapse mid-row, which may force re-encoding a whole group at a wider
// tier or shifting later groups) -- delta_csr_memory.hpp's row-rebuild/
// insert/remove machinery is untouched, and DeltaCSRRowCursor/
// uleb128_encode/uleb128_decode remain exactly as they were. Wiring
// this into the live, growable weight storage
// (disldo_forward/delta_csr_forward's hot paths) is a separate,
// not-yet-done follow-up.

#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// ── Width tiers ────────────────────────────────────────────────────────────

/// 0/1/2 = 1/2/4 bytes per offset -- the smallest tier that can hold
/// every offset in a group. Always determined by that group's own LAST
/// offset (offsets are monotonic increasing within a group, since the
/// underlying deltas are all positive).
enum class ForWidthTier : uint8_t { Byte1 = 0, Byte2 = 1, Byte4 = 2 };

inline int for_tier_width(ForWidthTier tier) {
    switch (tier) {
        case ForWidthTier::Byte1: return 1;
        case ForWidthTier::Byte2: return 2;
        default:                  return 4;
    }
}

inline ForWidthTier for_tier_for_max(uint32_t max_offset) {
    if (max_offset < 256)   return ForWidthTier::Byte1;
    if (max_offset < 65536) return ForWidthTier::Byte2;
    return ForWidthTier::Byte4;
}

// ── Encode ─────────────────────────────────────────────────────────────────

/// Encode one row's raw per-synapse deltas (NOT cumulative -- same input
/// convention as uleb128_encode's callers, e.g. delta_csr_memory.hpp's
/// delta_csr_from_absolute) into FOR format, appending to out_bytes.
/// G must be a multiple of 8 (AVX2 lane width). The last (partial) group
/// is padded by repeating its own final offset -- harmless, decode never
/// writes padding past the caller-supplied row length n.
template <std::size_t G, typename COL_TYPE = uint32_t>
inline void for_encode_row(const COL_TYPE* deltas, std::size_t n, std::vector<uint8_t>& out_bytes) {
    static_assert(G % 8 == 0, "group size must be a multiple of 8 (AVX2 lane width)");
    std::size_t i = 0;
    while (i < n) {
        const std::size_t g = std::min(G, n - i);
        uint32_t offsets[G];
        uint64_t running = 0;
        for (std::size_t k = 0; k < g; ++k) {
            running += deltas[i + k];
            offsets[k] = static_cast<uint32_t>(running);
        }
        for (std::size_t k = g; k < G; ++k) offsets[k] = offsets[g - 1];

        const ForWidthTier tier = for_tier_for_max(offsets[G - 1]);
        const int width = for_tier_width(tier);
        out_bytes.push_back(static_cast<uint8_t>(tier));
        for (std::size_t k = 0; k < G; ++k)
            for (int b = 0; b < width; ++b)
                out_bytes.push_back(static_cast<uint8_t>(offsets[k] >> (8 * b)));

        i += g;
    }
}

// ── Decode ─────────────────────────────────────────────────────────────────

namespace for_detail {

inline __m256i widen_tier(const uint8_t* p, ForWidthTier tier, std::size_t chunk_idx) {
    switch (tier) {
        case ForWidthTier::Byte1: {
            __m128i chunk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + chunk_idx * 8));
            return _mm256_cvtepu8_epi32(chunk);
        }
        case ForWidthTier::Byte2: {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + chunk_idx * 16));
            return _mm256_cvtepu16_epi32(chunk);
        }
        default:
            return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + chunk_idx * 32));
    }
}

} // namespace for_detail

/// Decode an entire row (n values) from FOR-encoded bytes at buf into
/// out_cols (caller-allocated, >= n entries). Bulk API -- fills the
/// whole row in one call, not a per-element cursor like
/// DeltaCSRRowCursor::advance() -- matches how callers actually consume
/// a row (linear_disldo.hpp/delta_csr_ops.hpp loop over every element
/// regardless), and is what makes the zero-prefix-sum decode possible:
/// each 8-wide chunk is one widen + one broadcast-add, no per-element
/// work at all, no dependency between chunks within a group.
template <std::size_t G, typename COL_TYPE = uint32_t>
inline void for_decode_row(const uint8_t* buf, std::size_t n, COL_TYPE* out_cols) {
    static_assert(G % 8 == 0, "group size must be a multiple of 8 (AVX2 lane width)");
    constexpr std::size_t chunks_per_group = G / 8;
    std::size_t pos = 0;
    std::size_t written = 0;
    uint32_t carry = 0;
    while (written < n) {
        const ForWidthTier tier = static_cast<ForWidthTier>(buf[pos++]);
        const int width = for_tier_width(tier);
        const __m256i v_carry = _mm256_set1_epi32(static_cast<int>(carry));
        alignas(32) uint32_t group_out[G];
        for (std::size_t c = 0; c < chunks_per_group; ++c) {
            const __m256i offsets = for_detail::widen_tier(buf + pos, tier, c);
            const __m256i result  = _mm256_add_epi32(offsets, v_carry);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(group_out + c * 8), result);
        }
        pos += G * width;
        carry = group_out[G - 1];

        const std::size_t g = std::min(G, n - written);
        for (std::size_t k = 0; k < g; ++k) out_cols[written + k] = static_cast<COL_TYPE>(group_out[k]);
        written += g;
    }
}

// ── Runtime group-size dispatch ─────────────────────────────────────────────

/// Curated set of pre-compiled group sizes -- callers pick one per
/// tensor at model-build time (e.g. after measuring that tensor's own
/// delta distribution) without needing G at THEIR compile time. Each
/// case still calls a fully compile-time-G-specialized instantiation,
/// so this dispatch costs one switch per ROW (not per element).
enum class ForGroupSize : uint8_t { G8 = 8, G16 = 16, G32 = 32, G64 = 64 };

template <typename COL_TYPE = uint32_t>
inline void for_encode_row_dispatch(const COL_TYPE* deltas, std::size_t n,
                                     std::vector<uint8_t>& out_bytes, ForGroupSize g) {
    switch (g) {
        case ForGroupSize::G8:  for_encode_row<8,  COL_TYPE>(deltas, n, out_bytes); return;
        case ForGroupSize::G16: for_encode_row<16, COL_TYPE>(deltas, n, out_bytes); return;
        case ForGroupSize::G32: for_encode_row<32, COL_TYPE>(deltas, n, out_bytes); return;
        case ForGroupSize::G64: for_encode_row<64, COL_TYPE>(deltas, n, out_bytes); return;
    }
}

template <typename COL_TYPE = uint32_t>
inline void for_decode_row_dispatch(const uint8_t* buf, std::size_t n, COL_TYPE* out_cols, ForGroupSize g) {
    switch (g) {
        case ForGroupSize::G8:  for_decode_row<8,  COL_TYPE>(buf, n, out_cols); return;
        case ForGroupSize::G16: for_decode_row<16, COL_TYPE>(buf, n, out_cols); return;
        case ForGroupSize::G32: for_decode_row<32, COL_TYPE>(buf, n, out_cols); return;
        case ForGroupSize::G64: for_decode_row<64, COL_TYPE>(buf, n, out_cols); return;
    }
}

#endif
