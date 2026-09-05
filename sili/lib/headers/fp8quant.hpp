#ifndef __FP8_HPP_
#define __FP8_HPP_

#include <cstdint>
#include <cstring>
#include <vector>

#include "fp4quant.hpp" // reuses the shared thread-local stochastic-rounding RNG

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
// See fp4quant.hpp's identical block comment above fp4_encode_bits_live for
// the full rationale (dead-synapse-collapse bug, deliberate near-zero bias,
// empirical validation) -- same invariant here, applied to E4M3's own
// subnormal region. Unlike FP4_TABLE (a single shared zero code regardless
// of sign), E4M3 is a real floating-point format with DISTINCT +0 (byte
// 0x00, the actual block4/scattered blank-slot sentinel) and -0 (byte 0x80,
// decodes to the same 0.0f value but is a different byte) codes -- both
// must become unreachable for a live weight, not just 0x00, or a synapse
// could still silently collapse to a zero CONTRIBUTION via -0 even though
// it wouldn't trip the byte==0 liveness check.
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

/// Never-zero STOCHASTIC quantize for a LIVE synapse's weight -- see
/// fp8_encode_bits_live's block comment for the +0/-0 rationale and
/// fp4_quantize_stochastic_live's docstring for the probability-weighted
/// redirect this mirrors. Only the m_lo==0 sub-bracket of the subnormal
/// region (the one that could otherwise produce +0 or -0) is widened to a
/// full-signed-bracket probability-weighted choice between +2^-9 and
/// -2^-9; every other bracket (including the rest of the subnormal region
/// and the whole normal region) is untouched and stays exactly as
/// unbiased as fp8_quantize_stochastic's own.
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
            // Full signed bracket [-2^-9,+2^-9) -- see this function's own
            // docstring above.
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

/// Never-zero STOCHASTIC quantize for a LIVE synapse's IMPORTANCE -- see
/// fp4_quantize_stochastic_live_nonneg's docstring (fp4quant.hpp) for the
/// full rationale (same bug, same fix, mirrored for E4M3): the m_lo==0
/// cross-sign redirect in fp8_quantize_stochastic_live is correct for
/// weight but would let a nonnegative quantity like importance land on a
/// NEGATIVE code near zero, and a negative decoded ci makes every
/// downstream sqrt(ci) call NaN. Sign is never flipped here; the m_lo==0
/// bracket is deterministic (always m=1, matching fp8_encode_bits_live's
/// own safe behavior) since there's no legal nonzero code below 2^-9 to
/// interpolate toward besides 2^-9 itself. Every other bracket
/// (m_lo>0 subnormal region, and the whole normal region) is untouched.
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
