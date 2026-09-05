// Correctness check for fp4quant.hpp's bit-shift codec (fp4_decode_bits/
// fp4_encode_bits), which replaced FP4_TABLE's linear scan inside
// fp4_quantize() -- see conversation. decode is checked bit-exact against
// FP4_TABLE for all 16 codes (no rounding, so no tolerance needed); encode
// is checked to always land on a nearest-or-tied table entry (NOT required
// to match the old linear scan's specific tie-break, which favoured lower
// table index arbitrarily -- see fp4quant.hpp's header comment on the new
// encoder).
#include "../../sili/lib/headers/fp4quant.hpp"
#include <cmath>
#include <cstdio>
#include <random>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

static bool is_nearest_or_tied(float v, uint8_t code) {
    float best_err = INFINITY;
    for (uint8_t i = 0; i < 16; ++i) {
        if (i == 8)
            continue;
        const float err = std::abs(v - FP4_TABLE[i]);
        if (err < best_err)
            best_err = err;
    }
    const float got_err = std::abs(v - FP4_TABLE[code]);
    return std::abs(got_err - best_err) < 1e-6f;
}

int main() {
    // decode: bit-exact against FP4_TABLE for every one of the 16 codes,
    // including the NaN slot (code 8).
    for (int code = 0; code < 16; ++code) {
        const float got = fp4_decode_bits(uint8_t(code));
        const float exp = FP4_TABLE[code];
        const bool match = (std::isnan(got) && std::isnan(exp)) || got == exp;
        CHECK(match, "decode(%d): got %f, expected %f", code, double(got), double(exp));
    }

    // encode: never produces the NaN slot (8) for any finite input.
    for (int code = 0; code < 16; ++code)
        CHECK(code != 8 || true, "sanity"); // placeholder, real check below

    // encode: always lands nearest-or-tied, for a dense sweep + random +
    // exact boundary values (the ones that previously exposed real bugs:
    // the 0.75-1.0 subnormal/normal straddle region, and the near-zero
    // NaN-slot collision for negative magnitudes).
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    int tested = 0;
    for (int i = -8000; i <= 8000; ++i) {
        const float v = float(i) / 1000.0f;
        const uint8_t code = fp4_encode_bits(v);
        CHECK(code != 8, "encode(%f) produced the NaN slot (8) for a finite input", double(v));
        CHECK(is_nearest_or_tied(v, code),
              "encode(%f) = %d (%f) is not nearest (nearest err differs)", double(v), int(code),
              double(FP4_TABLE[code]));
        ++tested;
    }
    for (int i = 0; i < 500000; ++i) {
        const float v = dist(rng);
        const uint8_t code = fp4_encode_bits(v);
        CHECK(code != 8, "encode(%f) produced the NaN slot (8) for a finite input", double(v));
        CHECK(is_nearest_or_tied(v, code), "encode(%f) = %d (%f) is not nearest", double(v),
              int(code), double(FP4_TABLE[code]));
        ++tested;
    }
    const float boundaries[] = {0.25f, -0.25f, 0.75f, -0.75f, 1.25f,  -1.25f,
                                1.75f, -1.75f, 2.5f,  -2.5f,  3.5f,   -3.5f,
                                5.0f,  -5.0f,  0.0f,  -0.0f,  100.0f, -100.0f};
    for (float v : boundaries) {
        const uint8_t code = fp4_encode_bits(v);
        CHECK(code != 8, "encode(%f) produced the NaN slot (8) for a finite input", double(v));
        CHECK(is_nearest_or_tied(v, code), "encode(%f) = %d (%f) is not nearest", double(v),
              int(code), double(FP4_TABLE[code]));
    }
    CHECK(fp4_encode_bits(NAN) == 0,
          "encode(NaN) should be code 0, matching the old linear scan's fallback");

    std::printf("%s (%d failures, %d values swept)\n", g_fail ? "FAIL" : "PASS", g_fail, tested);
    return g_fail ? 1 : 0;
}
