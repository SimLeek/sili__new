#ifndef __BLOCK4_CODEC_HPP_
#define __BLOCK4_CODEC_HPP_

// Block4Codec<VALUES_TYPE>: a thin per-precision wrapper trait, NOT a
// reimplementation of fp4quant.hpp/fp8quant.hpp's bit-quantization logic.
// linear_disldo.hpp/sisldo_ops.hpp currently branch on
// `if constexpr (std::is_same_v<VALUES_TYPE, ...>)` three times per kernel
// to do near-identical math, differing only in which decode/encode call
// gets made -- this trait lets that math be written ONCE as a generic
// template function parameterized on `Block4Codec<VALUES_TYPE>`, instead
// of being copy-pasted per precision. See
// docs/research/linear_disldo.rst:block4_codec_refactor for the full
// design rationale and the phased rollout this is Phase 1 of.
#include "delta_csr_types.hpp" // pulls in block4.hpp (Block4Tile/8/32), which
                               // itself pulls in fp4quant.hpp/fp8quant.hpp

// Primary template intentionally undefined -- only the three specializations
// below are valid VALUES_TYPE instantiations, matching the set linear_disldo.hpp
// and sisldo_ops.hpp already dispatch on.
template <typename VALUES_TYPE> struct Block4Codec;

// ── FP4 ──────────────────────────────────────────────────────────────────
// Packed single-byte-per-slot storage: low nibble = weight code, high
// nibble = importance code (Block4Tile::at, block4.hpp). has_zero_escape
// is true here ONLY -- see quant_floor's own comment.
template <> struct Block4Codec<FP4BiPacked> {
    using tile_type = Block4Tile;
    static constexpr std::size_t scratch_bytes = BLOCK4_TILE_SLOTS;
    static constexpr bool has_zero_escape = true;
    static constexpr bool has_was_live_gate = true;

    static float decode_weight(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        return FP4_TABLE[tdata[Block4Tile::slot_index(li, lj)] & 0xFu];
    }
    static float decode_importance(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        return FP4_TABLE[(tdata[Block4Tile::slot_index(li, lj)] >> 4) & 0xFu];
    }
    // FP4-specific: the AQRS gradient-accumulation terms (mrow/mgamma/mcol)
    // substitute a small epsilon for an exactly-zero quantized weight, so a
    // permanently-dead synapse can still escape zero via its importance
    // gradient -- see disldo_backward.deferred_vs_direct_quant in
    // docs/research/linear_disldo.rst. FP8/FP32 have no such floor (their
    // specializations below return `quant` unchanged).
    static float quant_floor(float quant, float zero_escape_eps) {
        return quant == 0.0f ? zero_escape_eps : quant;
    }
    template <bool StochasticRounding>
    static void encode(uint8_t* tdata, uint32_t li, uint32_t lj, float w, float imp,
                       bool was_live) {
        const uint32_t slot = Block4Tile::slot_index(li, lj);
        uint8_t new_w, new_imp;
        if constexpr (StochasticRounding) {
            if (was_live) {
                new_w = fp4_quantize_stochastic_live(w);
                new_imp = fp4_quantize_stochastic_live_nonneg(imp);
            } else {
                new_w = fp4_quantize_stochastic(w);
                new_imp = fp4_quantize_stochastic(imp);
            }
        } else {
            if (was_live) {
                new_w = fp4_quantize_live(w);
                new_imp = fp4_quantize_live(imp);
            } else {
                new_w = fp4_quantize(w);
                new_imp = fp4_quantize(imp);
            }
        }
        tdata[slot] = uint8_t((new_imp << 4) | new_w);
    }
    static std::size_t stored_tile_len(bool is_sparse, const uint8_t* tile_bytes) {
        return block4_stored_tile_len(is_sparse, tile_bytes);
    }
};

// ── FP8 (E4M3) ───────────────────────────────────────────────────────────
// Two separate byte ranges: weight byte at slot_index, importance byte at
// BLOCK4_TILE_SLOTS + slot_index (Block4Tile8, block4.hpp).
template <> struct Block4Codec<FP8BiValues> {
    using tile_type = Block4Tile8;
    static constexpr std::size_t scratch_bytes = BLOCK4_TILE_SLOTS8_BYTES;
    static constexpr bool has_zero_escape = false;
    static constexpr bool has_was_live_gate = true;

    static float decode_weight(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        return fp8_decode_bits(tdata[Block4Tile8::slot_index(li, lj)]);
    }
    static float decode_importance(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        return fp8_decode_bits(tdata[BLOCK4_TILE_SLOTS + Block4Tile8::slot_index(li, lj)]);
    }
    static float quant_floor(float quant, float /*zero_escape_eps*/) { return quant; }
    template <bool StochasticRounding>
    static void encode(uint8_t* tdata, uint32_t li, uint32_t lj, float w, float imp,
                       bool was_live) {
        const uint32_t slot = Block4Tile8::slot_index(li, lj);
        uint8_t new_w, new_imp;
        if constexpr (StochasticRounding) {
            if (was_live) {
                new_w = fp8_quantize_stochastic_live(w);
                new_imp = fp8_quantize_stochastic_live_nonneg(imp);
            } else {
                new_w = fp8_quantize_stochastic(w);
                new_imp = fp8_quantize_stochastic(imp);
            }
        } else {
            if (was_live) {
                new_w = fp8_quantize_live(w);
                new_imp = fp8_quantize_live(imp);
            } else {
                new_w = fp8_quantize(w);
                new_imp = fp8_quantize(imp);
            }
        }
        tdata[slot] = new_w;
        tdata[BLOCK4_TILE_SLOTS + slot] = new_imp;
    }
    static std::size_t stored_tile_len(bool is_sparse, const uint8_t* tile_bytes) {
        return block4_stored_tile_len8(is_sparse, tile_bytes);
    }
};

// ── FP32 ─────────────────────────────────────────────────────────────────
// Raw float storage, two separate ranges (Block4Tile32, block4.hpp). No
// quantization at all -- no was_live gate, no zero-escape floor, no
// stochastic/deterministic distinction (StochasticRounding is accepted
// but unused, matching disldo_backward's existing FP32 write-back, which
// is a plain memcpy either way).
template <> struct Block4Codec<DeltaCSRBiValues<float>> {
    using tile_type = Block4Tile32;
    static constexpr std::size_t scratch_bytes = BLOCK4_TILE_SLOTS32_BYTES;
    static constexpr bool has_zero_escape = false;
    static constexpr bool has_was_live_gate = false;

    static float decode_weight(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        float v;
        std::memcpy(&v, tdata + sizeof(float) * Block4Tile32::slot_index(li, lj), sizeof(v));
        return v;
    }
    static float decode_importance(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        float v;
        std::memcpy(&v,
                    tdata + sizeof(float) * (BLOCK4_TILE_SLOTS + Block4Tile32::slot_index(li, lj)),
                    sizeof(v));
        return v;
    }
    static float quant_floor(float quant, float /*zero_escape_eps*/) { return quant; }
    template <bool StochasticRounding>
    static void encode(uint8_t* tdata, uint32_t li, uint32_t lj, float w, float imp,
                       bool /*was_live*/) {
        const uint32_t slot = Block4Tile32::slot_index(li, lj);
        std::memcpy(tdata + sizeof(float) * slot, &w, sizeof(w));
        std::memcpy(tdata + sizeof(float) * (BLOCK4_TILE_SLOTS + slot), &imp, sizeof(imp));
    }
    static std::size_t stored_tile_len(bool is_sparse, const uint8_t* tile_bytes) {
        return block4_stored_tile_len32(is_sparse, tile_bytes);
    }
};

#endif
