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
// OCP MXFP4 E2M1 with slot 8 repurposed for NaN; fp4_quantize() is defined
// in terms of this encoder. See
// docs/research/fp4quant.rst:fp4_codec.bitshift_design for the decode/encode
// derivation and the tie-breaking rationale.

inline float fp4_decode_bits(uint8_t code) {
    const uint32_t s = (uint32_t(code) >> 3) & 1u;
    const uint32_t e = (uint32_t(code) >> 1) & 3u;
    const uint32_t m = uint32_t(code) & 1u;
    uint32_t bits;
    if (e == 0) {
        // Subnormal slot: code 0/1 -> 0.0/0.5, code 8/9 -> NaN/-0.5 (8 is the
        // repurposed slot, not IEEE -0.0 -- see fp4_codec.bitshift_design).
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
        // NaN input: matches the old linear-scan's fp4_quantize(NaN) == 0.
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
// A live synapse's weight must never quantize to code 0 (0 doubles as "no
// synapse here" in block4/delta_csr's blank-slot convention). These
// variants redirect the near-zero region to code 1 (0.5) instead,
// deliberately biased near zero. See
// docs/research/fp4quant.rst:fp4_quantize_live.never_zero_rationale.
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
        // NaN input: redirect to a signed code 1 rather than fp4_quantize's
        // code-0 convention -- see fp4_quantize_live.never_zero_rationale.
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
    // Defensive fallback -- should be unreachable, every branch above
    // already guarantees mag_code >= 1.
    if (mag_code == 0)
        mag_code = 1;
    const uint32_t s = sign ? 1u : 0u;
    return uint8_t((s << 3) | mag_code);
}

/// Never-zero deterministic quantize for a LIVE synapse's weight. See
/// docs/research/fp4quant.rst:fp4_quantize_live.never_zero_rationale.
inline uint8_t fp4_quantize_live(float v) {
    return fp4_encode_bits_live(v);
}

// ── Stochastic rounding ───────────────────────────────────────────────────────
// fp4_quantize() is deterministic nearest-neighbour, which silently discards
// any gradient update too small to cross a table midpoint -- with no master-
// weight float32 copy anywhere in this storage, that makes small/gradual
// training updates impossible. Stochastic rounding (Gupta et al. 2015) fixes
// this: E[quantized value] == v exactly, so small updates have a real,
// unbiased chance of flipping the stored code. NOT a replacement for
// fp4_quantize() -- only gradient-driven update sites should use it;
// construction/loading/compact/synaptogenesis-insert stay deterministic. See
// docs/research/fp4quant.rst:fp4_quantize_stochastic.unbiased_rounding_rationale.

// FP4_TABLE's 15 non-NaN indices, sorted ascending by VALUE (not by index --
// the raw table interleaves positive codes 0-7 and negative codes 9-15).
static constexpr uint8_t FP4_SORTED_IDX[15] = {
    15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7,
    // -6  -4  -3  -2 -1.5 -1 -.5 0 .5  1 1.5  2  3  4  6
};

/// Fast, thread-local, non-cryptographic PRNG (xorshift64*) for stochastic
/// rounding -- avoids a data race/lock on a shared generator across OpenMP
/// per-synapse loops. See
/// docs/research/fp4quant.rst:fp4_stochastic_rng.thread_local_design.
inline uint64_t& fp4_stochastic_rng_state() {
    thread_local uint64_t state =
        (std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ 0x9E3779B97F4A7C15ULL) | 1ULL;
    return state;
}

/// Reseeds the CALLING thread's stochastic-rounding RNG only -- full
/// reproducibility for single-threaded callers (tests); see
/// docs/research/fp4quant.rst:fp4_stochastic_rng.thread_local_design.
inline void fp4_seed_stochastic_rng(uint64_t seed) {
    fp4_stochastic_rng_state() = (seed ^ 0x9E3779B97F4A7C15ULL) | 1ULL;
}

/// One xorshift64* step, raw 64-bit output -- shared by fp4_stochastic_uniform01()
/// and fp4_quantize_stochastic()'s dithered rounding (different bit slices
/// of the SAME draw, not a second RNG step). See
/// docs/research/fp4quant.rst:fp4_stochastic_rng.thread_local_design.
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
/// for v within [-6,6]; clamps deterministically outside it). Bit-shift/
/// dithered-rounding, not FP4_SORTED_IDX's linear bracket scan (kept for
/// GPU/other-device use, no longer this function's CPU path). Two regimes
/// (|v|>=1.0 exploits an exact mantissa-bits/interpolation-fraction
/// identity; |v|<1.0 is a plain linear multiply) -- see
/// docs/research/fp4quant.rst:fp4_quantize_stochastic.dithered_rounding_design.
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

/// Never-zero STOCHASTIC quantize for a LIVE synapse's weight -- a genuine
/// probability-weighted redirect across the signed [-0.5,+0.5) bracket, NOT
/// a collapse to a fixed "always code 1" outcome. See
/// docs/research/fp4quant.rst:fp4_quantize_stochastic_live.probability_weighted_redirect.
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
        // Full signed bracket [-0.5,+0.5), see probability_weighted_redirect.
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
/// other quantity mathematically always >= 0, e.g. ci, fed into
/// sqrt(ci)+eps damping). Unlike fp4_quantize_stochastic_live, sign is
/// NEVER flipped here -- its cross-sign redirect would give importance up
/// to a 50% chance of a NEGATIVE code near zero, NaN-ing every downstream
/// sqrt(ci) call. See
/// docs/research/fp4quant.rst:fp4_quantize_stochastic_live_nonneg.sign_never_flipped_bug.
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

    explicit FP4BiPacked(std::vector<uint8_t>&& data)
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

    /// Gradient-driven update only -- deterministic elsewhere (operator[]=/
    /// resize/push_back) on purpose. See fp4_quantize_stochastic.unbiased_rounding_rationale.
    void set_stochastic(std::size_t i, float weight, float importance) {
        if (!_data)
            _data = std::make_shared<std::vector<uint8_t>>();
        (*_data)[i] =
            uint8_t((fp4_quantize_stochastic(weight) << 4) | fp4_quantize_stochastic(importance));
    }

    /// Same as set() but for a LIVE synapse -- applies to BOTH weight and
    /// importance (the never-0 invariant matters equally for both). See
    /// fp4_quantize_live.never_zero_rationale.
    void set_live(std::size_t i, float weight, float importance) {
        if (!_data)
            _data = std::make_shared<std::vector<uint8_t>>();
        (*_data)[i] = uint8_t((fp4_quantize_live(weight) << 4) | fp4_quantize_live(importance));
    }

    /// Same as set_stochastic() but for a LIVE synapse. Importance uses the
    /// _nonneg variant, NOT fp4_quantize_stochastic_live -- see
    /// fp4_quantize_stochastic_live_nonneg.sign_never_flipped_bug.
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
