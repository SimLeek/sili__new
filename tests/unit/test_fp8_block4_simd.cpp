// Correctness check for block4.hpp's 4-wide FP8 (E4M3) SIMD helpers
// (block4_vec_decode_fp8/block4_vec_quantize_stochastic_fp8) against
// fp8quant.hpp's scalar reference (fp8_decode_bits/fp8_quantize_stochastic)
// -- same structure as this repo's own test_fp4_bitshift.cpp/
// test_fp4_stochastic.cpp, adapted to a 4-lane batch instead of one
// value at a time.
#include "../../sili/lib/headers/delta_csr_types.hpp"
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

static Block4VecU codes_from(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    Block4VecU v;
    uint32_t arr[4] = {a, b, c, d};
    std::memcpy(&v, arr, sizeof(v));
    return v;
}

int main() {
    // decode: every code 0..255, batched 4 at a time, bit-exact against
    // the scalar reference (including the two NaN-slot codes).
    for (int base = 0; base < 256; base += 4) {
        const Block4VecU codes =
            codes_from(uint8_t(base), uint8_t(base + 1), uint8_t(base + 2), uint8_t(base + 3));
        const Block4Vec got = block4_vec_decode_fp8(codes);
        float got_arr[4];
        std::memcpy(got_arr, &got, sizeof(got_arr));
        for (int lane = 0; lane < 4; ++lane) {
            const uint8_t code = uint8_t(base + lane);
            const float exp = fp8_decode_bits(code);
            const bool match =
                (std::isnan(got_arr[lane]) && std::isnan(exp)) || got_arr[lane] == exp;
            CHECK(match, "decode(%d) lane %d: got %f, expected %f", code, lane,
                  double(got_arr[lane]), double(exp));
        }
    }

    // stochastic quantize: statistical mean check (same convention as
    // test_fp4_stochastic.cpp) across a batch of 4 DIFFERENT values at once.
    fp4_seed_stochastic_rng(0);
    const float test_vals[4] = {0.1f, 3.0f, -50.0f, 0.0001f};
    const Block4Vec v_batch = Block4Vec{test_vals[0], test_vals[1], test_vals[2], test_vals[3]};
    double sums[4] = {0, 0, 0, 0};
    const int N = 200000;
    for (int i = 0; i < N; ++i) {
        const Block4VecU codes = block4_vec_quantize_stochastic_fp8(v_batch);
        uint32_t codes_arr[4];
        std::memcpy(codes_arr, &codes, sizeof(codes_arr));
        for (int lane = 0; lane < 4; ++lane)
            sums[lane] += double(fp8_decode_bits(uint8_t(codes_arr[lane])));
    }
    for (int lane = 0; lane < 4; ++lane) {
        const double mean = sums[lane] / N;
        const double tol = 0.02 * std::max(1.0f, std::abs(test_vals[lane])) + 1e-4;
        CHECK(std::abs(mean - test_vals[lane]) <= tol, "E[stochastic(%f)] = %f, off by %f (tol %f)",
              double(test_vals[lane]), mean, mean - test_vals[lane], tol);
    }

    // saturation: large-magnitude batch always lands at +-448.
    const Block4Vec big = Block4Vec{1e6f, -1e6f, 448.0f, -448.0f};
    for (int i = 0; i < 100; ++i) {
        const Block4VecU codes = block4_vec_quantize_stochastic_fp8(big);
        uint32_t codes_arr[4];
        std::memcpy(codes_arr, &codes, sizeof(codes_arr));
        CHECK(fp8_decode_bits(uint8_t(codes_arr[0])) == 448.0f, "saturate +big failed");
        CHECK(fp8_decode_bits(uint8_t(codes_arr[1])) == -448.0f, "saturate -big failed");
        CHECK(fp8_decode_bits(uint8_t(codes_arr[2])) == 448.0f, "saturate +448 failed");
        CHECK(fp8_decode_bits(uint8_t(codes_arr[3])) == -448.0f, "saturate -448 failed");
    }

    // random batches: decode(stochastic_quantize(v)) never hits a NaN slot.
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-500.0f, 500.0f);
    for (int i = 0; i < 50000; ++i) {
        const Block4Vec v = Block4Vec{dist(rng), dist(rng), dist(rng), dist(rng)};
        const Block4VecU codes = block4_vec_quantize_stochastic_fp8(v);
        uint32_t codes_arr[4];
        std::memcpy(codes_arr, &codes, sizeof(codes_arr));
        for (int lane = 0; lane < 4; ++lane)
            CHECK((codes_arr[lane] & 0x7Fu) != 0x7Fu, "produced a NaN slot for finite input");
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
