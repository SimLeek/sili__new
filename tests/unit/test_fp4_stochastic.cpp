// Correctness check for fp4quant.hpp's dithered-rounding fp4_quantize_stochastic()
// (replaced the old FP4_SORTED_IDX linear bracket scan -- see fp4quant.hpp's
// header comment on why the new bit-shift version is exact, not an
// approximation, for the |v|>=1.0 "normal" E2M1 region, and why the
// |v|<1.0 region uses a plain linear formula instead). Statistical, not
// exact-value, checks throughout -- stochastic rounding is only required
// to be UNBIASED (E[result] == v), not to reproduce any specific old
// per-draw sequence; no test anywhere in this repo pins exact stochastic
// codes for a given seed (checked before making this change).
#include "../../sili/lib/headers/fp4quant.hpp"
#include <cmath>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); std::fflush(stdout); ++g_fail; } \
} while (0)

int main() {
    fp4_seed_stochastic_rng(0);
    const int N = 200000;

    const float test_vals[] = {0.05f, 0.1f, 0.2f, 0.3f, 0.4f, 0.6f, 0.7f, 0.8f, 0.9f,
                                1.1f, 1.25f, 1.4f, 1.6f, 1.9f, 2.2f, 2.6f, 2.9f,
                                3.2f, 3.7f, 4.5f, 5.0f, 5.5f, 5.9f,
                                -0.3f, -1.7f, -3.3f, -5.8f};
    for (float v : test_vals) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += double(fp4_decode_bits(fp4_quantize_stochastic(v)));
        const double mean = sum / N;
        // Worst-case single-gap width is 2.0 (the 4<->6 gap); N=200000
        // easily clears 0.02 unless something is systematically biased.
        CHECK(std::abs(mean - v) <= 0.02, "E[stochastic(%f)] = %f, off by %f", double(v), mean, mean - v);
    }

    for (int i = 0; i < 100; ++i) {
        CHECK(fp4_decode_bits(fp4_quantize_stochastic(6.0f)) == 6.0f, "saturate +6 failed");
        CHECK(fp4_decode_bits(fp4_quantize_stochastic(100.0f)) == 6.0f, "saturate +100 failed");
        CHECK(fp4_decode_bits(fp4_quantize_stochastic(-6.0f)) == -6.0f, "saturate -6 failed");
        CHECK(fp4_decode_bits(fp4_quantize_stochastic(-100.0f)) == -6.0f, "saturate -100 failed");
    }

    const float exact_vals[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    for (float v : exact_vals)
        for (int i = 0; i < 1000; ++i)
            CHECK(fp4_decode_bits(fp4_quantize_stochastic(v)) == v,
                  "exactly-representable v=%f didn't round to itself", double(v));

    CHECK(fp4_quantize_stochastic(NAN) == 0, "NaN input should encode to code 0");

    for (int i = -1200; i <= 1200; ++i) {
        const float v = float(i) / 100.0f;
        CHECK(fp4_quantize_stochastic(v) != 8, "produced the NaN slot (8) for finite v=%f", double(v));
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
