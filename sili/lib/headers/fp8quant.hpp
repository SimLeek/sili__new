#ifndef __FP8_HPP_
#define __FP8_HPP_

#include <cstdint>
#include <cstring>
#include <vector>

#include "fp4quant.hpp" // reuses the shared thread-local stochastic-rounding RNG

// ── FP8 (OCP MX E4M3) ─────────────────────────────────────────────────────
// 1 sign, 4 exponent (bias 7), 3 mantissa bits. NOT plain IEEE-754 -- see
// docs/research/fp8quant.rst:fp8_format.e4m3_design for the repurposed-NaN-
// slot rationale and the Python cross-format validation results.

constexpr uint32_t FP8_MIN_NORMAL_BITS = 0x3C800000u; // bits_of(2^-6)
constexpr uint32_t FP8_MAX_BITS = 0x43E00000u;        // bits_of(448.0f), e_field=15,m=6
constexpr uint32_t FP8_NAN_SLOT_BITS = 0x43F00000u; // bits_of(480.0f) -- e_field=15,m=7 in float32
                                                    // terms, the reserved NaN-adjacent pattern
constexpr float FP8_E4M3_MAX = 448.0f;

inline float fp8_decode_bits(uint8_t code) {
    const uint32_t s = (uint32_t(code) >> 7) & 1u;
    const uint32_t e4 = (uint32_t(code) >> 3) & 0xFu;
    const uint32_t m = uint32_t(code) & 0x7u;

    if (e4 == 0u) {
        // Subnormal slot: value = m * 2^-9, computed directly (no shared
        // float32 exponent bracket to place bits into), same spirit as
        // fp4_decode_bits' e==0 case.
        const float mag = float(m) * (1.0f / 512.0f);
        return s ? -mag : mag;
    }
    if (e4 == 15u && m == 7u) {
        uint32_t bits = 0x7FC00000u | (s << 31);
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }
    const uint32_t exp_field = e4 + 120u; // e4m3 bias 7 -> float32 bias 127: +120
    const uint32_t bits = (s << 31) | (exp_field << 23) | (m << 20);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline uint8_t fp8_encode_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;
    const uint32_t s = sign ? 1u : 0u;

    if (abits > 0x7F800000u) {
        // NaN input -- matches fp4_encode_bits' NaN convention.
        return 0x7Fu;
    }
    if (abits < FP8_MIN_NORMAL_BITS) {
        // Subnormal encode. See
        // docs/research/fp8quant.rst:fp8_encode_bits.subnormal_normal_boundary
        // for the carry-across-boundary edge case handled below.
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const uint32_t m = uint32_t(av * 512.0f + 0.5f);
        if (m > 7u) {
            return uint8_t((s << 7) | 0x08u);
        }
        return uint8_t((s << 7) | m);
    }
    uint32_t rounded = abits + (1u << 19); // half-ULP bias: 3 mantissa bits kept, 20 discarded
    if (rounded >= FP8_NAN_SLOT_BITS)
        rounded = FP8_MAX_BITS; // saturate, never land on the reserved NaN code
    const uint32_t exp_field_f32 = (rounded >> 23) & 0xFFu;
    const uint32_t m3 = (rounded >> 20) & 0x7u;
    const uint32_t e4 = exp_field_f32 - 120u;
    return uint8_t((s << 7) | (e4 << 3) | m3);
}

/// Nearest-neighbour quantize @p v to an 8-bit E4M3 code.
inline uint8_t fp8_quantize(float v) {
    return fp8_encode_bits(v);
}

// ── Never-zero ("live") variants ─────────────────────────────────────────────
// See docs/research/fp8quant.rst:fp8_quantize_live.dual_zero_rationale --
// E4M3 has distinct +0/-0 codes, unlike FP4's single shared zero.
inline uint8_t fp8_encode_bits_live(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;
    const uint32_t s = sign ? 1u : 0u;

    if (abits > 0x7F800000u) {
        return 0x7Fu; // NaN input -- already nonzero, no change needed
    }
    if (abits < FP8_MIN_NORMAL_BITS) {
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        uint32_t m = uint32_t(av * 512.0f + 0.5f);
        if (m > 7u) {
            return uint8_t((s << 7) | 0x08u);
        }
        if (m == 0u)
            m = 1u; // live: never round to +0 (0x00) or -0 (0x80)
        return uint8_t((s << 7) | m);
    }
    uint32_t rounded = abits + (1u << 19);
    if (rounded >= FP8_NAN_SLOT_BITS)
        rounded = FP8_MAX_BITS;
    const uint32_t exp_field_f32 = (rounded >> 23) & 0xFFu;
    const uint32_t m3 = (rounded >> 20) & 0x7u;
    const uint32_t e4 = exp_field_f32 - 120u;
    return uint8_t((s << 7) | (e4 << 3) | m3);
}

/// Never-zero deterministic quantize for a LIVE synapse's weight.
inline uint8_t fp8_quantize_live(float v) {
    return fp8_encode_bits_live(v);
}

/// Stochastic quantize -- unbiased (E[result] == v within [-448,448]). See
/// docs/research/fp8quant.rst:fp8_quantize_stochastic.dithered_rounding.
/// Gradient-driven update sites only (construction/loading/compact stay
/// deterministic).
inline uint8_t fp8_quantize_stochastic(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;
    const uint32_t s = sign ? 1u : 0u;

    if (abits > 0x7F800000u)
        return 0x7Fu; // NaN input

    if (abits >= FP8_NAN_SLOT_BITS) {
        return uint8_t((s << 7) | 0x7Eu); // deterministic saturate at 448 (e4=15,m=6)
    }
    if (abits < FP8_MIN_NORMAL_BITS) {
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const float pos = av * 512.0f; // continuous position in [0,8)
        const uint32_t m_lo = uint32_t(pos);
        const float frac = pos - float(m_lo);
        const uint32_t m = (fp4_stochastic_uniform01() < frac) ? (m_lo + 1u) : m_lo;
        if (m > 7u)
            return uint8_t((s << 7) | 0x08u); // carried into first normal code, see fp8_encode_bits
        return uint8_t((s << 7) | m);
    }
    const uint32_t dither = uint32_t(fp4_stochastic_next_u64() & 0xFFFFFu); // uniform in [0, 2^20)
    uint32_t rounded = abits + dither;
    if (rounded >= FP8_NAN_SLOT_BITS)
        rounded = FP8_MAX_BITS;
    const uint32_t exp_field_f32 = (rounded >> 23) & 0xFFu;
    const uint32_t m3 = (rounded >> 20) & 0x7u;
    const uint32_t e4 = exp_field_f32 - 120u;
    return uint8_t((s << 7) | (e4 << 3) | m3);
}

/// Never-zero STOCHASTIC quantize for a LIVE synapse's weight. See
/// docs/research/fp8quant.rst:fp8_quantize_stochastic_live.subnormal_signed_redirect.
inline uint8_t fp8_quantize_stochastic_live(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;
    const uint32_t s = sign ? 1u : 0u;

    if (abits > 0x7F800000u)
        return 0x7Fu; // NaN input, already nonzero

    if (abits >= FP8_NAN_SLOT_BITS) {
        return uint8_t((s << 7) | 0x7Eu); // deterministic saturate at 448
    }
    if (abits < FP8_MIN_NORMAL_BITS) {
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const float pos = av * 512.0f; // continuous position in [0,8)
        const uint32_t m_lo = uint32_t(pos);
        if (m_lo == 0u) {
            // Full signed bracket [-2^-9,+2^-9), see subnormal_signed_redirect above.
            const float signed_pos = sign ? -pos : pos;     // pos in [0,1) here
            const float p_pos = (signed_pos + 1.0f) * 0.5f; // (v-(-1))/((+1)-(-1)), units of 2^-9
            const bool pick_pos = fp4_stochastic_uniform01() < p_pos;
            return uint8_t(((pick_pos ? 0u : 1u) << 7) | 1u);
        }
        const float frac = pos - float(m_lo);
        const uint32_t m = (fp4_stochastic_uniform01() < frac) ? (m_lo + 1u) : m_lo;
        if (m > 7u)
            return uint8_t((s << 7) | 0x08u); // carried into first normal code
        return uint8_t((s << 7) | m);
    }
    const uint32_t dither = uint32_t(fp4_stochastic_next_u64() & 0xFFFFFu); // uniform in [0, 2^20)
    uint32_t rounded = abits + dither;
    if (rounded >= FP8_NAN_SLOT_BITS)
        rounded = FP8_MAX_BITS;
    const uint32_t exp_field_f32 = (rounded >> 23) & 0xFFu;
    const uint32_t m3 = (rounded >> 20) & 0x7u;
    const uint32_t e4 = exp_field_f32 - 120u;
    return uint8_t((s << 7) | (e4 << 3) | m3);
}

/// Never-zero STOCHASTIC quantize for a LIVE synapse's IMPORTANCE. See
/// docs/research/fp8quant.rst:fp8_quantize_stochastic_live_nonneg.sign_never_flipped.
inline uint8_t fp8_quantize_stochastic_live_nonneg(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;
    const uint32_t s = sign ? 1u : 0u;

    if (abits > 0x7F800000u)
        return 0x7Fu; // NaN input, already nonzero

    if (abits >= FP8_NAN_SLOT_BITS) {
        return uint8_t((s << 7) | 0x7Eu); // deterministic saturate at 448
    }
    if (abits < FP8_MIN_NORMAL_BITS) {
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const float pos = av * 512.0f; // continuous position in [0,8)
        const uint32_t m_lo = uint32_t(pos);
        if (m_lo == 0u) {
            return uint8_t((s << 7) | 1u); // deterministic -- see docstring above
        }
        const float frac = pos - float(m_lo);
        const uint32_t m = (fp4_stochastic_uniform01() < frac) ? (m_lo + 1u) : m_lo;
        if (m > 7u)
            return uint8_t((s << 7) | 0x08u); // carried into first normal code
        return uint8_t((s << 7) | m);
    }
    const uint32_t dither = uint32_t(fp4_stochastic_next_u64() & 0xFFFFFu); // uniform in [0, 2^20)
    uint32_t rounded = abits + dither;
    if (rounded >= FP8_NAN_SLOT_BITS)
        rounded = FP8_MAX_BITS;
    const uint32_t exp_field_f32 = (rounded >> 23) & 0xFFu;
    const uint32_t m3 = (rounded >> 20) & 0x7u;
    const uint32_t e4 = exp_field_f32 - 120u;
    return uint8_t((s << 7) | (e4 << 3) | m3);
}

// ── FP8BiValues ────────────────────────────────────────────────────────────
// Two full byte arrays (weight, importance), not nibble-packed. See
// docs/research/fp8quant.rst:fp8_bivalues.two_array_shape.

struct FP8BiValues {
    std::vector<uint8_t> weights;
    std::vector<uint8_t> importance;
};

#endif
