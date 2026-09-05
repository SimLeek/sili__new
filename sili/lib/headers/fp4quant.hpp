#ifndef __FP4_HPP_
#define __FP4_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

// ── FP4 lookup table — "FP4 All the Way" ─────────────────────────────────────
// Index layout: 0=zero, 1-7=positive, 8=NaN (treated as 0), 9-15=negative.
// Fits in 4 bits; two values pack exactly into one byte with no spanning.

static constexpr float FP4_TABLE[16] = {
    0.0f,  // 0000
    0.5f,  // 0001
    1.0f,  // 0010
    1.5f,  // 0011
    2.0f,  // 0100
    3.0f,  // 0101
    4.0f,  // 0110
    6.0f,  // 0111
    NAN,   // 1000
    -0.5f, // 1001
    -1.0f, // 1010
    -1.5f, // 1011
    -2.0f, // 1100
    -3.0f, // 1101
    -4.0f, // 1110
    -6.0f, // 1111
};

// ── Bit-shift codec (no table, no branch-per-candidate) ─────────────────────
//
// FP4_TABLE's 16 entries are exactly OCP MXFP4 E2M1 (2 exponent bits, 1
// mantissa bit, bias 1) with one repurposing: slot 8 (sign=1,exp=00,mant=0,
// which E2M1 itself would read as -0.0) stores NaN instead. That's a real,
// deliberate difference from raw E2M1 -- not a bug to route around here.
//
// decode: E2M1's exponent/mantissa fields slot directly into an IEEE-754
// float32's fields with a re-bias (E2M1 bias 1 -> float32 bias 127, so
// exp_field = e2m1_exp + 126) and the single mantissa bit placed at
// float32's top mantissa bit -- exact, no rounding, verified bit-for-bit
// against FP4_TABLE for all 16 codes (see conversation). Only exp==0 (the
// subnormal slot, values 0/NaN/-0.5 depending on sign+mantissa) needs
// separate handling; every other code is one shift+mask+bitcast.
//
// encode: nearest-value quantization via the standard low-precision-ML
// technique (also how real E2M1 hardware casts work) -- add a rounding
// bias at the mantissa truncation point and let integer addition's carry
// propagate the rounding through the exponent (1.75 -> 2.0 falls out of
// the carry automatically, no separate case needed), then saturate at the
// max representable magnitude (6.0). The one further special case is the
// [0.25, 1.0) magnitude range, which straddles the subnormal/normal
// boundary the generic carry trick doesn't span on its own.
//
// Tie-breaking here (nearest with exact float ties resolved by whichever
// way the bit arithmetic naturally falls, e.g. midpoints round up in
// magnitude) is NOT required to match fp4_quantize()'s old linear-scan
// tie convention (which favoured lower table index on a tie) -- the table
// isn't a frozen external format, just this codebase's own choice, and
// exact-tie floats essentially never occur in real gradient-driven data.
// fp4_quantize() below is now defined IN TERMS OF this encoder, not the
// other way around.

inline float fp4_decode_bits(uint8_t code) {
    const uint32_t s = (uint32_t(code) >> 3) & 1u;
    const uint32_t e = (uint32_t(code) >> 1) & 3u;
    const uint32_t m = uint32_t(code) & 1u;
    uint32_t bits;
    if (e == 0) {
        // Subnormal slot: code 0/1 -> 0.0/0.5, code 8/9 -> NaN/-0.5 (see
        // header comment -- 8 is the repurposed slot, not IEEE -0.0).
        bits = m ? (0x3F000000u | (s << 31)) : (s ? 0x7FC00000u : 0u);
    } else {
        const uint32_t exp_field = e + 126u;
        bits = (s << 31) | (exp_field << 23) | (m << 22);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

inline uint8_t fp4_encode_bits(float v) {
    static constexpr uint32_t TH_025 = 0x3E800000u; // bits_of(0.25f)
    static constexpr uint32_t TH_075 = 0x3F400000u; // bits_of(0.75f)
    static constexpr uint32_t TH_1 = 0x3F800000u;   // bits_of(1.0f)
    static constexpr uint32_t SIX = 0x40C00000u;    // bits_of(6.0f)

    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;

    uint32_t mag_code;
    if (abits > 0x7F800000u) {
        // NaN input: matches the old linear-scan's fp4_quantize(NaN) == 0
        // (every |NaN - table[i]| comparison is false, so `best` never
        // moves off its 0 initial value).
        return 0;
    } else if (abits < TH_025) {
        mag_code = 0;
    } else if (abits < TH_075) {
        mag_code = 1;
    } else if (abits < TH_1) {
        mag_code = 2;
    } else {
        uint32_t rounded = abits + (1u << 21);
        if (rounded > SIX)
            rounded = SIX;
        const uint32_t exp_field = (rounded >> 23) & 0xFFu;
        const uint32_t m = (rounded >> 22) & 1u;
        mag_code = ((exp_field - 126u) << 1) | m;
    }
    // A near-zero input must land on code 0 regardless of sign -- never
    // the repurposed NaN slot (8 = sign 1, magnitude 0).
    if (mag_code == 0)
        return 0;
    const uint32_t s = sign ? 1u : 0u;
    return uint8_t((s << 3) | mag_code);
}

/// Nearest-neighbour quantize @p v to a 4-bit FP4 index. NaN slot (8) is skipped.
inline uint8_t fp4_quantize(float v) {
    return fp4_encode_bits(v);
}

// ── Never-zero ("live") variants ─────────────────────────────────────────────
//
// A LIVE synapse's weight (never importance, never a genuinely blank/
// unallocated storage slot -- see block4.hpp/delta_csr_memory.hpp's own
// "byte==0 means blank" convention, which these functions must never be
// used to write) must never quantize to code 0: once a synapse's weight
// AND importance both land on 0, disldo_backward treats the whole row/
// tile as having zero live connections and excludes it from the normal
// per-synapse gradient path entirely, recoverable only via a much slower
// separate dead-row bootstrap fallback -- confirmed directly via
// tests/unit/test_scale_handling.cpp's "near-autapse" tests (see
// conversation). These variants redirect the near-zero region to the
// smallest nonzero magnitude (code 1 = 0.5) instead, skipping code 0
// entirely -- codes go directly from -0.5 to +0.5 with nothing between.
//
// DELIBERATELY BIASED near zero: true values very close to 0 no longer
// average to 0 (E[quantized] != v in that region, unlike fp4_quantize's
// own unbiased-elsewhere convention). This is an intentional tradeoff,
// not an oversight -- confirmed empirically (see conversation: a
// near-autapse harness across 6 seeds x 4 targets including exactly 0.0)
// that the resulting sign settles quickly and stays put rather than
// thrashing, and that overall accuracy is comparable to or better than
// allowing code 0, because the baseline's occasional multi-hundred-step
// "stuck dead synapse" episodes cost far more than this variant's small
// steady-state discretization floor.
inline uint8_t fp4_encode_bits_live(float v) {
    static constexpr uint32_t TH_025 = 0x3E800000u; // bits_of(0.25f)
    static constexpr uint32_t TH_075 = 0x3F400000u; // bits_of(0.75f)
    static constexpr uint32_t TH_1 = 0x3F800000u;   // bits_of(1.0f)
    static constexpr uint32_t SIX = 0x40C00000u;    // bits_of(6.0f)

    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;

    uint32_t mag_code;
    if (abits > 0x7F800000u) {
        // NaN input: a NaN weight reaching quantization is itself a distinct
        // bug class (upstream gradient corruption) -- the never-0 invariant
        // shouldn't carve out an exception for it, so redirect to a signed
        // code 1 rather than fp4_quantize's own code-0 convention.
        return uint8_t(((sign ? 1u : 0u) << 3) | 1u);
    } else if (abits < TH_025) {
        mag_code = 1; // was 0 in fp4_encode_bits -- the whole point of "live"
    } else if (abits < TH_075) {
        mag_code = 1;
    } else if (abits < TH_1) {
        mag_code = 2;
    } else {
        uint32_t rounded = abits + (1u << 21);
        if (rounded > SIX)
            rounded = SIX;
        const uint32_t exp_field = (rounded >> 23) & 0xFFu;
        const uint32_t m = (rounded >> 22) & 1u;
        mag_code = ((exp_field - 126u) << 1) | m;
    }
    // Defensive fallback -- every branch above already guarantees mag_code
    // >= 1, this should be unreachable; kept cheap in case a future edit to
    // the carry-propagation path above reintroduces a 0.
    if (mag_code == 0)
        mag_code = 1;
    const uint32_t s = sign ? 1u : 0u;
    return uint8_t((s << 3) | mag_code);
}

/// Never-zero deterministic quantize for a LIVE synapse's weight -- see the
/// block comment above fp4_encode_bits_live for the full rationale.
inline uint8_t fp4_quantize_live(float v) {
    return fp4_encode_bits_live(v);
}

// ── Stochastic rounding ───────────────────────────────────────────────────────
//
// fp4_quantize() above is deterministic nearest-neighbour: a gradient-driven
// update too small to cross the midpoint between two FP4_TABLE entries is
// silently discarded, every single time, with no memory of the near-miss --
// there's no persistent float32 "master weight" anywhere in this storage (see
// linear_disldo.hpp's disldo_forward/disldo_backward, which dequantize the
// CURRENTLY-STORED code, add the update, and requantize immediately). That's
// fine for one-shot construction/conversion (nearest-neighbour is the best
// single-shot approximation) but makes small/gradual training updates
// impossible: real per-synapse gradients here are routinely too small to
// cross even the finest FP4 gap (0->0.5) in one step, and with round-to-
// nearest they never accumulate toward doing so on a later step either.
//
// Standard fix from low-precision training literature (Gupta et al. 2015,
// "Deep Learning with Limited Numerical Precision", and used by essentially
// every training scheme with quantized weights and no master-weight copy,
// including modern large-model FP4 training): STOCHASTIC rounding. Round up
// or down with probability proportional to how close v is to each neighbour,
// so E[quantized value] == v exactly. A single small update then has a small
// but real, unbiased CHANCE of flipping the stored code, and the expected
// drift over many steps correctly tracks the true (unquantized) gradient
// signal -- unlike round-to-nearest, which has no expected drift at all for
// sub-half-gap updates.
//
// Deliberately NOT a replacement for fp4_quantize() -- only the gradient-
// driven update sites (disldo_forward's importance update, disldo_backward's
// weight+importance update) should use this; construction/loading/compact/
// synaptogenesis-insert must stay exactly deterministic (repacking or
// reloading the same content must not randomly perturb existing values).

// FP4_TABLE's 15 non-NaN indices, sorted ascending by VALUE (not by index --
// the raw table interleaves positive codes 0-7 and negative codes 9-15).
static constexpr uint8_t FP4_SORTED_IDX[15] = {
    15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7,
    // -6  -4  -3  -2 -1.5 -1 -.5 0 .5  1 1.5  2  3  4  6
};

/// Fast, thread-local, non-cryptographic PRNG (xorshift64*) for stochastic
/// rounding -- called from within OpenMP-parallelized per-synapse loops, so
/// a shared/global generator would mean either a data race or lock
/// contention on every single synapse update. Seeded once per thread from
/// its id by default -- explicitly reseedable via fp4_seed_stochastic_rng()
/// for tests that need reproducibility, same precedent as EnergyDynamics'
/// own unseeded-by-default exploration noise (np.random.seed(0) pinned by
/// callers that need it, see test_column_averaging_predictive.py).
inline uint64_t& fp4_stochastic_rng_state() {
    thread_local uint64_t state =
        (std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ 0x9E3779B97F4A7C15ULL) | 1ULL;
    return state;
}

/// Reseeds the CALLING thread's stochastic-rounding RNG. Single-threaded
/// callers (tests, small examples) get full reproducibility this way; a
/// multi-threaded (OpenMP) caller would need to call this once per worker
/// thread to pin all of them, which no caller currently needs -- training
/// runs are meant to be stochastic across threads too, this is for
/// unit-test determinism, not for controlling a real training run's outcome.
inline void fp4_seed_stochastic_rng(uint64_t seed) {
    fp4_stochastic_rng_state() = (seed ^ 0x9E3779B97F4A7C15ULL) | 1ULL;
}

/// One xorshift64* step, raw 64-bit output -- shared by fp4_stochastic_uniform01()
/// (top 24 bits) and fp4_quantize_stochastic()'s dithered rounding (different
/// bit slices of the SAME draw, not a second RNG step -- xorshift64*'s bits
/// are well-mixed enough that slicing different ranges for different
/// purposes within one draw is fine, and matters here: it keeps stochastic
/// quantize at exactly one RNG step per call, same as before this split).
inline uint64_t fp4_stochastic_next_u64() {
    uint64_t& s = fp4_stochastic_rng_state();
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

inline float fp4_stochastic_uniform01() {
    const uint64_t r = fp4_stochastic_next_u64();
    return static_cast<float>((r >> 40) * (1.0 / 16777216.0)); // top 24 bits -> [0,1)
}

/// Stochastic quantize @p v to a 4-bit FP4 index -- unbiased (E[result] == v
/// for v within the representable range [-6,6]; clamps deterministically
/// outside it, same as fp4_quantize() would).
///
/// Bit-shift/dithered-rounding implementation, not FP4_SORTED_IDX's linear
/// bracket scan (kept, for GPU/other-device use per direction, but no
/// longer this function's own CPU path -- same relationship as
/// fp4_quantize()/FP4_TABLE above). Two regimes, matching fp4_encode_bits'
/// own split:
///   - |v| >= 1.0 ("normal" E2M1 region): the interpolation fraction
///     between the two neighbouring representable values is EXACTLY the
///     value's own discarded IEEE mantissa bits (below the one bit E2M1
///     keeps), normalized to [0,1) -- provably, algebraically, not an
///     approximation (both endpoints of any such bracket share the same
///     IEEE exponent, so the bracket's linear interpolation fraction and
///     the mantissa's fractional position within that exponent coincide
///     exactly). That makes the classic dithered-rounding trick exact
///     here: add a UNIFORM RANDOM integer spanning the discarded bits'
///     full range (not the deterministic encoder's fixed 1<<21 bias),
///     then truncate -- integer addition's carry propagates a rounding
///     UP through the exponent as needed, precisely like the
///     deterministic version, just probabilistically instead of via a
///     fixed round-to-nearest bias.
///   - |v| < 1.0 (the subnormal/normal-transition region, magnitudes
///     0/0.5/1.0 only): straddles E2M1's own subnormal boundary the same
///     way fp4_encode_bits' 3-way split does, and doesn't have the
///     "discarded mantissa bits = interpolation fraction" property (no
///     shared exponent bracket to exploit) -- but the two sub-brackets
///     here ([0,0.5) and [0.5,1.0)) are each linear in v directly, so
///     the interpolation fraction is just a plain multiply (2v, or
///     2v-1), no bit tricks needed.
inline uint8_t fp4_quantize_stochastic(float v) {
    static constexpr uint32_t HALF_BITS = 0x3F000000u; // bits_of(0.5f)
    static constexpr uint32_t ONE_BITS = 0x3F800000u;  // bits_of(1.0f)
    static constexpr uint32_t SIX_BITS = 0x40C00000u;  // bits_of(6.0f)

    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;

    if (abits > 0x7F800000u)
        return 0; // NaN input -- matches fp4_quantize(NaN) == 0

    uint32_t mag_code;
    if (abits >= SIX_BITS) {
        mag_code = 7; // deterministic saturate, matches the old scan's v >= hi_bound clamp
    } else if (abits < HALF_BITS) {
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const float p_up = av * 2.0f; // (v - 0.0) / (0.5 - 0.0)
        mag_code = (fp4_stochastic_uniform01() < p_up) ? 1u : 0u;
    } else if (abits < ONE_BITS) {
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const float p_up = av * 2.0f - 1.0f; // (v - 0.5) / (1.0 - 0.5)
        mag_code = (fp4_stochastic_uniform01() < p_up) ? 2u : 1u;
    } else {
        const uint32_t dither =
            uint32_t(fp4_stochastic_next_u64() & 0x3FFFFFu); // uniform in [0, 2^22)
        uint32_t rounded = abits + dither;
        if (rounded > SIX_BITS)
            rounded = SIX_BITS;
        const uint32_t exp_field = (rounded >> 23) & 0xFFu;
        const uint32_t m = (rounded >> 22) & 1u;
        mag_code = ((exp_field - 126u) << 1) | m;
    }
    if (mag_code == 0)
        return 0; // never the repurposed NaN slot, see fp4_encode_bits
    return uint8_t(((sign ? 1u : 0u) << 3) | mag_code);
}

/// Never-zero STOCHASTIC quantize for a LIVE synapse's weight -- see the
/// block comment above fp4_encode_bits_live for the never-0 rationale.
///
/// NOT a collapse to a fixed "always code 1" outcome (an earlier draft of
/// this design got this wrong -- see conversation): the existing
/// fp4_quantize_stochastic's `abits < HALF_BITS` bracket ([0,0.5)) already
/// does genuine probability-weighted rounding for v>=0 (draws p_up=v*2,
/// picks code 0 vs 1 with probability continuously varying by where v
/// sits in the bracket). The live version generalizes that SAME mechanism
/// across the full SIGNED bracket [-0.5,+0.5) instead of collapsing it:
/// picks -0.5 vs +0.5 with probability proportional to where the signed v
/// falls across the doubled range (p_pos = (v+0.5)/1.0) -- near v=0 this
/// is close to a fair coin flip between signs; near +0.5 almost always
/// +0.5; near -0.5 almost always -0.5. Every bracket further from zero is
/// untouched and stays exactly as unbiased as fp4_quantize_stochastic's
/// own.
inline uint8_t fp4_quantize_stochastic_live(float v) {
    static constexpr uint32_t HALF_BITS = 0x3F000000u; // bits_of(0.5f)
    static constexpr uint32_t ONE_BITS = 0x3F800000u;  // bits_of(1.0f)
    static constexpr uint32_t SIX_BITS = 0x40C00000u;  // bits_of(6.0f)

    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;

    if (abits > 0x7F800000u) {
        // NaN input -- see fp4_encode_bits_live's identical rationale.
        return uint8_t(((sign ? 1u : 0u) << 3) | 1u);
    }

    uint32_t mag_code;
    uint32_t s_out = sign ? 1u : 0u;
    if (abits >= SIX_BITS) {
        mag_code = 7; // deterministic saturate, matches fp4_quantize_stochastic
    } else if (abits < HALF_BITS) {
        // Full signed bracket [-0.5,+0.5) -- see this function's own
        // docstring above for the probability-weighted redirect.
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const float signed_v = sign ? -av : av;
        const float p_pos = signed_v + 0.5f; // (v - (-0.5)) / ((+0.5) - (-0.5))
        const bool pick_pos = fp4_stochastic_uniform01() < p_pos;
        mag_code = 1u;
        s_out = pick_pos ? 0u : 1u;
    } else if (abits < ONE_BITS) {
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const float p_up = av * 2.0f - 1.0f; // (v - 0.5) / (1.0 - 0.5)
        mag_code = (fp4_stochastic_uniform01() < p_up) ? 2u : 1u;
    } else {
        const uint32_t dither =
            uint32_t(fp4_stochastic_next_u64() & 0x3FFFFFu); // uniform in [0, 2^22)
        uint32_t rounded = abits + dither;
        if (rounded > SIX_BITS)
            rounded = SIX_BITS;
        const uint32_t exp_field = (rounded >> 23) & 0xFFu;
        const uint32_t m = (rounded >> 22) & 1u;
        mag_code = ((exp_field - 126u) << 1) | m;
    }
    // Defensive fallback, should be unreachable -- see fp4_encode_bits_live.
    if (mag_code == 0)
        mag_code = 1u;
    return uint8_t((s_out << 3) | mag_code);
}

/// Never-zero STOCHASTIC quantize for a LIVE synapse's IMPORTANCE (or any
/// other quantity that is mathematically always >= 0, e.g. the ci
/// accumulator fed into sqrt(ci)+eps damping throughout disldo_backward/
/// sisldo_ops.hpp). fp4_quantize_stochastic_live's cross-sign redirect
/// (see its own docstring) is correct for WEIGHT, which can legitimately
/// be positive or negative -- but applying that SAME redirect to
/// importance is a bug: importance near 0 would then have up to a 50%
/// chance of landing on the NEGATIVE code, and a negative decoded
/// importance makes every downstream sqrt(ci) call return NaN, silently
/// poisoning training. Found via direct question/regression during this
/// session -- see conversation.
///
/// Sign is NEVER flipped here (matches fp4_encode_bits_live's own
/// deterministic behavior, which never had this bug). In the [0,0.5)
/// bracket there is no legal nonzero magnitude to interpolate toward
/// besides 0.5 itself (0 is forbidden, negative is forbidden), so that
/// bracket is deterministic: always mag_code=1 -- the same deliberate
/// near-zero bias already accepted for the live design as a whole, just
/// without weight's cross-sign freedom that a nonnegative quantity must
/// not have. Every bracket above HALF_BITS is untouched, identical to
/// fp4_quantize_stochastic_live's own (those never touch sign or produce
/// mag_code 0 anyway).
inline uint8_t fp4_quantize_stochastic_live_nonneg(float v) {
    static constexpr uint32_t HALF_BITS = 0x3F000000u; // bits_of(0.5f)
    static constexpr uint32_t ONE_BITS = 0x3F800000u;  // bits_of(1.0f)
    static constexpr uint32_t SIX_BITS = 0x40C00000u;  // bits_of(6.0f)

    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abits = bits & 0x7FFFFFFFu;
    const uint32_t s_out = sign ? 1u : 0u;

    if (abits > 0x7F800000u) {
        return uint8_t((s_out << 3) | 1u); // NaN input
    }

    uint32_t mag_code;
    if (abits >= SIX_BITS) {
        mag_code = 7;
    } else if (abits < HALF_BITS) {
        mag_code = 1u; // deterministic -- see docstring above
    } else if (abits < ONE_BITS) {
        float av;
        std::memcpy(&av, &abits, sizeof(av));
        const float p_up = av * 2.0f - 1.0f;
        mag_code = (fp4_stochastic_uniform01() < p_up) ? 2u : 1u;
    } else {
        const uint32_t dither = uint32_t(fp4_stochastic_next_u64() & 0x3FFFFFu);
        uint32_t rounded = abits + dither;
        if (rounded > SIX_BITS)
            rounded = SIX_BITS;
        const uint32_t exp_field = (rounded >> 23) & 0xFFu;
        const uint32_t m = (rounded >> 22) & 1u;
        mag_code = ((exp_field - 126u) << 1) | m;
    }
    if (mag_code == 0)
        mag_code = 1u;
    return uint8_t((s_out << 3) | mag_code);
}

// ── FP4BiPacked ───────────────────────────────────────────────────────────────
// Two FP4 values per byte: high nibble = values[0] (weight),
//                           low nibble = values[1] (importance/strength).
// One byte per connection — no bit-spanning, no cross-byte logic, no masking beyond 0xF.

struct FP4BiPacked {
    using storage_type = std::vector<uint8_t>;
    using size_type = std::size_t;

    struct ElemRef {
        uint8_t& _byte;
        bool _hi;

        operator float() const { return FP4_TABLE[_hi ? (_byte >> 4) : (_byte & 0xF)]; }
        ElemRef& operator=(float v) {
            const uint8_t idx = fp4_quantize(v);
            if (_hi)
                _byte = (_byte & 0x0F) | (idx << 4);
            else
                _byte = (_byte & 0xF0) | idx;
            return *this;
        }
        ElemRef& operator=(const ElemRef& other) { return *this = float(other); }
        ElemRef& operator+=(float v) { return *this = (float)*this + v; }
        ElemRef& operator-=(float v) { return *this = (float)*this - v; }
        ElemRef& operator*=(float v) { return *this = (float)*this * v; }
        ElemRef& operator/=(float v) { return *this = (float)*this / v; }
    };

    struct Lane {
        std::shared_ptr<std::vector<uint8_t>>* _dp;
        bool _hi;

        // ── shared_ptr-like interface ─────────────────────────────────────────

        explicit operator bool() const { return bool(*_dp); }

        /// operator* and operator-> return *this — Lane is both ptr and vector.
        Lane& operator*() { return *this; }
        const Lane& operator*() const { return *this; }
        Lane* operator->() { return this; }
        const Lane* operator->() const { return this; }

        /// Assignment from make_shared<vector<T>> ensures the backing store exists.
        /// Both lanes always share one array; the assigned ptr's contents are ignored.
        template <class T> Lane& operator=(std::shared_ptr<std::vector<T>>) {
            if (!*_dp)
                *_dp = std::make_shared<std::vector<uint8_t>>();
            return *this;
        }

        // ── vector-like interface ─────────────────────────────────────────────

        ElemRef operator[](std::size_t i) { return ElemRef{(**_dp)[i], _hi}; }
        float operator[](std::size_t i) const {
            const uint8_t b = (**_dp)[i];
            return FP4_TABLE[_hi ? (b >> 4) : (b & 0xF)];
        }

        std::size_t size() const { return *_dp ? (*_dp)->size() : 0; }
        std::size_t capacity() const { return *_dp ? (*_dp)->capacity() : 0; }
        bool empty() const { return !*_dp || (*_dp)->empty(); }

        void reserve(std::size_t n) {
            if (!*_dp)
                *_dp = std::make_shared<std::vector<uint8_t>>();
            (*_dp)->reserve(n);
        }
        void resize(std::size_t n, float v = 0.0f) {
            if (!*_dp)
                *_dp = std::make_shared<std::vector<uint8_t>>();
            const std::size_t old = (*_dp)->size();
            (*_dp)->resize(n, uint8_t(0));
            if (v != 0.0f) {
                const uint8_t idx = fp4_quantize(v);
                for (std::size_t i = old; i < n; ++i) {
                    auto& b = (**_dp)[i];
                    if (_hi)
                        b = (b & 0x0F) | (idx << 4);
                    else
                        b = (b & 0xF0) | idx;
                }
            }
        }
        /// push_back allocates one new byte (both nibbles share it).
        /// Call on one lane per element; set the other nibble via operator[].
        void push_back(float v) {
            if (!*_dp)
                *_dp = std::make_shared<std::vector<uint8_t>>();
            const uint8_t idx = fp4_quantize(v);
            const uint8_t byte = _hi ? uint8_t(idx << 4) : idx;
            (*_dp)->push_back(byte);
        }
        void clear() {
            if (*_dp)
                (*_dp)->clear();
        }
    };

    // ── state ─────────────────────────────────────────────────────────────────
    // _data declared before _lanes so &_data is valid in initializer lists.

    std::shared_ptr<std::vector<uint8_t>> _data;
    Lane _lanes[2];

    // ── constructors ──────────────────────────────────────────────────────────
    // Each ctor rebuilds _lanes pointing to this->_data.
    // Copy/move assignment only touches _data; _lanes already point to it.

    FP4BiPacked() : _data(), _lanes{Lane{&_data, true}, Lane{&_data, false}} {}

    FP4BiPacked(std::vector<uint8_t>&& data)
        : _data(std::make_shared<std::vector<uint8_t>>(std::move(data))),
          _lanes{Lane{&_data, true}, Lane{&_data, false}} {}

    FP4BiPacked(const FP4BiPacked& o)
        : _data(o._data), _lanes{Lane{&_data, true}, Lane{&_data, false}} {}

    FP4BiPacked(FP4BiPacked&& o) noexcept
        : _data(std::move(o._data)), _lanes{Lane{&_data, true}, Lane{&_data, false}} {}

    // (false positive: _lanes are views holding &this->_data, not
    // independent state -- correct already after copying _data, see
    // the ctors' own comment above)
    // cppcheck-suppress operatorEqVarError
    FP4BiPacked& operator=(const FP4BiPacked& o) {
        if (this != &o)
            _data = o._data; // _lanes._dp = &_data already correct
        return *this;
    }
    // cppcheck-suppress operatorEqVarError
    FP4BiPacked& operator=(FP4BiPacked&& o) noexcept {
        if (this != &o)
            _data = std::move(o._data);
        return *this;
    }

    // ── class-level vector interface ─────────────────────────────────────────────
    // Mirrors Lane's interface but operates on both nibbles simultaneously.
    // Prefer these over lane[0]/lane[1] calls when weight and importance
    // are known together — avoids touching the backing store twice.

    void reserve(std::size_t n) {
        if (!_data)
            _data = std::make_shared<std::vector<uint8_t>>();
        _data->reserve(n);
    }

    void resize(std::size_t n, float weight = 0.0f, float importance = 0.0f) {
        if (!_data)
            _data = std::make_shared<std::vector<uint8_t>>();
        const std::size_t old = _data->size();
        _data->resize(n, uint8_t(0));
        if (weight != 0.0f || importance != 0.0f) {
            const uint8_t fill = uint8_t((fp4_quantize(weight) << 4) | fp4_quantize(importance));
            for (std::size_t j = old; j < n; ++j)
                (*_data)[j] = fill;
        }
    }

    void push_back(float weight, float importance) {
        if (!_data)
            _data = std::make_shared<std::vector<uint8_t>>();
        _data->push_back(uint8_t((fp4_quantize(weight) << 4) | fp4_quantize(importance)));
    }

    /// Gradient-driven update only -- see fp4_quantize_stochastic()'s own
    /// docstring for why this exists separately from operator[]=/resize/
    /// push_back (all of which stay deterministic on purpose).
    void set_stochastic(std::size_t i, float weight, float importance) {
        if (!_data)
            _data = std::make_shared<std::vector<uint8_t>>();
        (*_data)[i] =
            uint8_t((fp4_quantize_stochastic(weight) << 4) | fp4_quantize_stochastic(importance));
    }

    /// Same as set() but for a LIVE synapse -- see fp4_encode_bits_live's
    /// own docstring for the never-0 rationale. Applies to BOTH weight
    /// and importance: a live synapse's importance quantizing to the
    /// blank-slot sentinel is the same failure mode as weight doing so
    /// (nnz_row==0-style dead-row checks, and pruning decisions that
    /// read importance as the significance signal), not a separate
    /// concern -- see conversation.
    void set_live(std::size_t i, float weight, float importance) {
        if (!_data)
            _data = std::make_shared<std::vector<uint8_t>>();
        (*_data)[i] = uint8_t((fp4_quantize_live(weight) << 4) | fp4_quantize_live(importance));
    }

    /// Same as set_stochastic() but for a LIVE synapse. Importance uses
    /// the _nonneg variant, NOT fp4_quantize_stochastic_live -- see
    /// fp4_quantize_stochastic_live_nonneg's own docstring: importance is
    /// mathematically always >= 0 (fed into sqrt(ci) throughout
    /// disldo_backward), and weight's cross-sign redirect would let it
    /// land on a negative code near zero, NaN-ing every downstream
    /// sqrt(ci) call.
    void set_stochastic_live(std::size_t i, float weight, float importance) {
        if (!_data)
            _data = std::make_shared<std::vector<uint8_t>>();
        (*_data)[i] = uint8_t((fp4_quantize_stochastic_live(weight) << 4) |
                              fp4_quantize_stochastic_live_nonneg(importance));
    }

    void clear() {
        if (_data)
            _data->clear();
    }

    /// Serializes the internal data into a standard byte vector.
    std::vector<uint8_t> serialize() const {
        if (_data) {
            // Returns a copy of the underlying shared storage
            return *_data;
        }
        return std::vector<uint8_t>{};
    }

    /// Deserializes from a standard byte vector (moves the buffer to avoid a copy).
    static FP4BiPacked deserialize(std::vector<uint8_t>&& buffer) {
        return FP4BiPacked(std::move(buffer));
    }

    Lane& operator[](std::size_t i) { return _lanes[i]; }
    const Lane& operator[](std::size_t i) const { return _lanes[i]; }

    std::size_t size() const { return _data ? _data->size() : 0; }
    bool empty() const { return !_data || _data->empty(); }
};

// tuple_size specialization so sparse_struct::n_value_arrays constexpr compiles.
namespace std {
template <> struct tuple_size<FP4BiPacked> : integral_constant<size_t, 2> {};
} // namespace std

#endif
