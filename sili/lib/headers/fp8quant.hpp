#ifndef __FP8_HPP_
#define __FP8_HPP_

#include <cstdint>
#include <cstring>
#include <vector>

#include "fp4quant.hpp"  // reuses the shared thread-local stochastic-rounding RNG

// ── FP8 (OCP MX E4M3) ─────────────────────────────────────────────────────
// 1 sign, 4 exponent (bias 7), 3 mantissa bits. NOT plain IEEE-754: like
// this codebase's own FP4 (E2M1 with one repurposed NaN slot), the
// exponent field 1111 is NOT entirely reserved for Inf/NaN -- only the
// single code S.1111.111 is NaN, every other e=1111 code is still a valid
// finite normal value, extending max representable magnitude to 448
// (matches the OCP Microscaling FP8 spec / NVIDIA Transformer Engine's
// E4M3, not a from-scratch format). Subnormals at e=0000: value = m/8 *
// 2^-6 (min positive normal 2^-6, subnormal step 2^-9).
//
// Validated in Python (sili_peridot's toy-model quantization sweep,
// July-Aug 2026 session) with a per-row/rank-1 (row*col envelope) scale
// on top of this raw code, exactly like FP4's existing value_scale/
// output_scale mechanism: beat both a plain-int8 rank-1 scheme AND real
// FP4 across every task family tested (single-cell tanh-RNN, deep
// causal-attention transformer, tile-recurrence). See sili_peridot's
// JOURNAL.md for the full validation writeup.
//
// Same bit-shift technique as fp4_encode_bits/fp4_decode_bits (no table,
// no per-candidate branch -- carry-propagating rounding bias, direct
// IEEE-754 float32 field placement with a rebias) -- one full byte per
// value means no nibble-packing/masking is needed at all, simpler than
// FP4's ElemRef/Lane machinery.

constexpr uint32_t FP8_MIN_NORMAL_BITS = 0x3C800000u;  // bits_of(2^-6)
constexpr uint32_t FP8_MAX_BITS        = 0x43E00000u;  // bits_of(448.0f), e_field=15,m=6
constexpr uint32_t FP8_NAN_SLOT_BITS   = 0x43F00000u;  // bits_of(480.0f) -- e_field=15,m=7 in float32 terms, the reserved NaN-adjacent pattern
constexpr float     FP8_E4M3_MAX       = 448.0f;

inline float fp8_decode_bits(uint8_t code) {
    const uint32_t s  = (uint32_t(code) >> 7) & 1u;
    const uint32_t e4 = (uint32_t(code) >> 3) & 0xFu;
    const uint32_t m  = uint32_t(code) & 0x7u;

    if (e4 == 0u) {
        // Subnormal slot: value = m * 2^-9 -- straddles E4M3's own
        // subnormal boundary (no single shared float32 exponent bracket
        // across all 8 codes), so computed directly rather than via
        // field placement, same spirit as fp4_decode_bits' e==0 case.
        const float mag = float(m) * (1.0f / 512.0f);
        return s ? -mag : mag;
    }
    if (e4 == 15u && m == 7u) {
        uint32_t bits = 0x7FC00000u | (s << 31);
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }
    const uint32_t exp_field = e4 + 120u;  // e4m3 bias 7 -> float32 bias 127: +120
    const uint32_t bits = (s << 31) | (exp_field << 23) | (m << 20);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline uint8_t fp8_encode_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign  = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;
    const uint32_t s     = sign ? 1u : 0u;

    if (abits > 0x7F800000u) {
        // NaN input -- matches fp4_encode_bits' choice to map NaN
        // deterministically rather than propagate it into storage.
        return 0x7Fu;
    }
    if (abits < FP8_MIN_NORMAL_BITS) {
        // Subnormal encode: nearest of {0, 2^-9, ..., 7*2^-9}. Each
        // sub-bracket here is linear in v directly (no shared exponent
        // to exploit for a bit trick), so a plain round suffices --
        // same reasoning fp4_encode_bits uses for its own <1.0 region.
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const uint32_t m = uint32_t(av * 512.0f + 0.5f);
        if (m > 7u) {
            // Rounding carried past the largest subnormal (7*2^-9) into
            // the first normal code (2^-6 = min normal, e4=1, m=0) --
            // e.g. 0.015367 is nearer to 0.015625 (min normal) than to
            // 0.013672 (max subnormal); clamping to 7 here would pick
            // the wrong neighbour instead of carrying across the
            // subnormal/normal boundary, same class of bug the carry
            // trick in the normal-range branch below handles for free.
            return uint8_t((s << 7) | 0x08u);
        }
        return uint8_t((s << 7) | m);
    }
    uint32_t rounded = abits + (1u << 19);  // half-ULP bias: 3 mantissa bits kept, 20 discarded
    if (rounded >= FP8_NAN_SLOT_BITS) rounded = FP8_MAX_BITS;  // saturate, never land on the reserved NaN code
    const uint32_t exp_field_f32 = (rounded >> 23) & 0xFFu;
    const uint32_t m3            = (rounded >> 20) & 0x7u;
    const uint32_t e4 = exp_field_f32 - 120u;
    return uint8_t((s << 7) | (e4 << 3) | m3);
}

/// Nearest-neighbour quantize @p v to an 8-bit E4M3 code.
inline uint8_t fp8_quantize(float v) {
    return fp8_encode_bits(v);
}

/// Stochastic quantize -- unbiased (E[result] == v within [-448,448]),
/// same dithered-rounding technique as fp4_quantize_stochastic() (see
/// fp4quant.hpp for the full derivation of why this is exact in the
/// normal region): a uniform random dither spanning the discarded
/// mantissa bits, added before truncation, lets integer carry
/// propagate a probabilistic round exactly the way the deterministic
/// version's fixed bias propagates a round-to-nearest. Shares the
/// caller's thread-local RNG state with FP4 (fp4_stochastic_next_u64)
/// -- gradient-driven updates only, same convention as FP4's own
/// set_stochastic (construction/loading/compact must stay deterministic).
inline uint8_t fp8_quantize_stochastic(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign  = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;
    const uint32_t s     = sign ? 1u : 0u;

    if (abits > 0x7F800000u) return 0x7Fu;  // NaN input

    if (abits >= FP8_NAN_SLOT_BITS) {
        return uint8_t((s << 7) | 0x7Eu);  // deterministic saturate at 448 (e4=15,m=6)
    }
    if (abits < FP8_MIN_NORMAL_BITS) {
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const float pos    = av * 512.0f;              // continuous position in [0,8)
        const uint32_t m_lo = uint32_t(pos);
        const float frac   = pos - float(m_lo);
        const uint32_t m = (fp4_stochastic_uniform01() < frac) ? (m_lo + 1u) : m_lo;
        if (m > 7u) return uint8_t((s << 7) | 0x08u);  // carried into first normal code, see fp8_encode_bits
        return uint8_t((s << 7) | m);
    }
    const uint32_t dither = uint32_t(fp4_stochastic_next_u64() & 0xFFFFFu);  // uniform in [0, 2^20)
    uint32_t rounded = abits + dither;
    if (rounded >= FP8_NAN_SLOT_BITS) rounded = FP8_MAX_BITS;
    const uint32_t exp_field_f32 = (rounded >> 23) & 0xFFu;
    const uint32_t m3            = (rounded >> 20) & 0x7u;
    const uint32_t e4 = exp_field_f32 - 120u;
    return uint8_t((s << 7) | (e4 << 3) | m3);
}

// ── FP8BiValues ────────────────────────────────────────────────────────────
// Two full byte arrays (weight, importance) -- unlike FP4BiPacked's
// nibble-interleaved single array, E4M3 needs a whole byte per value, so
// this mirrors DeltaCSRBiValues<T>'s plain two-array shape exactly
// (per direct instruction: "the sparse part can take pretty close to
// the fp32 version" -- same structure, narrower per-element storage,
// no bit-spanning/masking machinery needed). ValueAccessor<FP8BiValues>
// (delta_csr_types.hpp, alongside ValueAccessor<FP4BiPacked>/
// ValueAccessor<DeltaCSRBiValues<T>>) is what makes this a drop-in
// VALUES_TYPE for SparseLinearWeightsDelta/disldo_forward/disldo_backward.

struct FP8BiValues {
    std::vector<uint8_t> weights;
    std::vector<uint8_t> importance;
};

#endif
