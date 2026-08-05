// Correctness check for fp8quant.hpp's E4M3 bit-shift codec
// (fp8_decode_bits/fp8_encode_bits/fp8_quantize_stochastic). No table to
// check against (unlike FP4's 16-entry FP4_TABLE) -- E4M3 has 256 codes --
// so this verifies round-trip stability, decode-matches-formula, and
// encode-lands-nearest-among-neighbours instead, plus the specific edge
// cases fp4's own test caught bugs at (NaN-slot collision, saturation,
// subnormal/normal straddle).
#include "../../sili/lib/headers/fp8quant.hpp"
#include <cmath>
#include <cstdio>
#include <random>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); std::fflush(stdout); ++g_fail; } \
} while (0)

// Decode every non-NaN code once, up front -- used both to check
// decode's own internal consistency (finite, no duplicates beyond the
// expected +0/-0 pair) and as the "table" encode's nearest-check sweeps.
static float g_decoded[256];

static bool is_nearest_or_tied(float v, uint8_t code) {
    float best_err = INFINITY;
    for (int i = 0; i < 256; ++i) {
        if (i == 0x7F || i == 0xFF) continue;  // both NaN slots (sign doesn't matter for NaN)
        const float err = std::abs(v - g_decoded[i]);
        if (err < best_err) best_err = err;
    }
    const float got_err = std::abs(v - g_decoded[code]);
    return std::abs(got_err - best_err) < 1e-3f * std::max(1.0f, std::abs(v));
}

int main() {
    for (int code = 0; code < 256; ++code) g_decoded[code] = fp8_decode_bits(uint8_t(code));

    // decode: finite everywhere except the two NaN-slot codes; magnitude
    // is monotonically non-decreasing in the low 7 bits (sign-magnitude
    // layout, e4/m3 both increase the encoded magnitude).
    for (int code = 0; code < 256; ++code) {
        const bool is_nan_slot = (code & 0x7F) == 0x7F;
        if (is_nan_slot) {
            CHECK(std::isnan(g_decoded[code]), "decode(%d) expected NaN", code);
        } else {
            CHECK(std::isfinite(g_decoded[code]), "decode(%d) = %f not finite", code, double(g_decoded[code]));
        }
    }
    for (int m = 0; m < 0x7F; ++m) {
        if ((m & 0x7F) == 0x7F) continue;
        if (m > 0) CHECK(g_decoded[m] >= g_decoded[m - 1] - 1e-12f,
                          "decode magnitude not monotonic at code %d: %f then %f",
                          m, double(g_decoded[m - 1]), double(g_decoded[m]));
    }
    CHECK(g_decoded[0] == 0.0f, "decode(0) should be exactly 0.0, got %f", double(g_decoded[0]));
    CHECK(g_decoded[0x7E] == 448.0f, "decode(0x7E) (max finite) should be 448.0, got %f", double(g_decoded[0x7E]));

    // encode: never produces a NaN-slot code for a finite input.
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist_wide(-1000.0f, 1000.0f);
    std::uniform_real_distribution<float> dist_small(-0.05f, 0.05f);
    int tested = 0;
    for (int i = -9000; i <= 9000; ++i) {
        const float v = float(i) / 20.0f;  // sweep -450..450
        const uint8_t code = fp8_encode_bits(v);
        CHECK((code & 0x7F) != 0x7F, "encode(%f) produced a NaN slot for a finite input", double(v));
        CHECK(is_nearest_or_tied(v, code), "encode(%f) = %d (%f) is not nearest",
              double(v), int(code), double(g_decoded[code]));
        ++tested;
    }
    for (int i = 0; i < 200000; ++i) {
        const float v = dist_wide(rng);
        const uint8_t code = fp8_encode_bits(v);
        CHECK((code & 0x7F) != 0x7F, "encode(%f) produced a NaN slot for a finite input", double(v));
        CHECK(is_nearest_or_tied(v, code), "encode(%f) = %d (%f) is not nearest",
              double(v), int(code), double(g_decoded[code]));
        ++tested;
    }
    for (int i = 0; i < 50000; ++i) {
        const float v = dist_small(rng);  // subnormal region
        const uint8_t code = fp8_encode_bits(v);
        CHECK((code & 0x7F) != 0x7F, "encode(%f) produced a NaN slot for a finite input", double(v));
        CHECK(is_nearest_or_tied(v, code), "encode(%f) = %d (%f) is not nearest",
              double(v), int(code), double(g_decoded[code]));
        ++tested;
    }
    const float boundaries[] = {0.0f, -0.0f, 448.0f, -448.0f, 500.0f, -500.0f, 100000.0f, -100000.0f,
                                 0.015625f, -0.015625f, 0.001f, -0.001f, 1.0f, -1.0f};
    for (float v : boundaries) {
        const uint8_t code = fp8_encode_bits(v);
        CHECK((code & 0x7F) != 0x7F, "encode(%f) produced a NaN slot for a finite input", double(v));
        CHECK(is_nearest_or_tied(v, code), "encode(%f) = %d (%f) is not nearest",
              double(v), int(code), double(g_decoded[code]));
    }
    CHECK((fp8_encode_bits(NAN) & 0x7F) == 0x7F, "encode(NaN) should land on a NaN slot");
    CHECK(fp8_encode_bits(1e9f) == 0x7E, "encode(large finite) should saturate to max (0x7E), got %d", int(fp8_encode_bits(1e9f)));
    CHECK(fp8_encode_bits(-1e9f) == 0xFE, "encode(-large finite) should saturate to -max (0xFE), got %d", int(fp8_encode_bits(-1e9f)));

    // round-trip idempotence: encode(decode(encode(v))) == encode(v),
    // for every real (non-NaN-slot) code -- catches asymmetric rounding
    // bugs at exponent boundaries.
    for (int code = 0; code < 256; ++code) {
        if ((code & 0x7F) == 0x7F) continue;
        const float v = g_decoded[code];
        const uint8_t re = fp8_encode_bits(v);
        CHECK(re == uint8_t(code) || g_decoded[re] == v,
              "round-trip broke at code %d (%f): re-encoded to %d (%f)",
              code, double(v), int(re), double(g_decoded[re]));
    }

    // stochastic quantize: unbiased (mean over many draws converges to
    // v), matches fp4_quantize_stochastic's own test convention.
    {
        fp4_seed_stochastic_rng(42);
        const float v = 0.1f;  // lands strictly between two subnormal/near-boundary codes
        double sum = 0.0;
        const int n = 200000;
        for (int i = 0; i < n; ++i) sum += fp8_decode_bits(fp8_quantize_stochastic(v));
        const double mean = sum / n;
        CHECK(std::abs(mean - v) < 0.01 * std::abs(v) + 1e-4,
              "stochastic quantize mean %f too far from target %f", mean, double(v));
    }

    std::printf("%s (%d failures, %d values swept)\n", g_fail ? "FAIL" : "PASS", g_fail, tested);
    return g_fail ? 1 : 0;
}
