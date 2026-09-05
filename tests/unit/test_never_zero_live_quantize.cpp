// Correctness check for fp4quant.hpp/fp8quant.hpp's never-zero "_live"
// quantize variants (fp4_quantize_live, fp4_quantize_stochastic_live,
// fp8_quantize_live, fp8_quantize_stochastic_live). These exist so a LIVE
// synapse's weight can never quantize to the byte value block4/scattered
// storage uses as its "this slot is blank/unallocated" sentinel -- see
// fp4quant.hpp's block comment above fp4_encode_bits_live for the full
// dead-synapse-collapse rationale (found via tests/unit/test_scale_handling.cpp's
// "near-autapse" tests) and the design conversation for why the stochastic
// variants are a probability-weighted widening of the SAME mechanism the
// non-live functions already use, not a collapse to a fixed outcome.
#include "../../sili/lib/headers/fp4quant.hpp"
#include "../../sili/lib/headers/fp8quant.hpp"
#include "../../sili/lib/headers/delta_csr_types.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

int main() {
    fp4_seed_stochastic_rng(0);

    // ── (1) Never code 0 / byte 0x00 -- deterministic, dense sweep ──────────
    for (int i = -10000; i <= 10000; ++i) {
        const float v = float(i) / 10000.0f; // [-1, 1]
        CHECK(fp4_quantize_live(v) != 0, "fp4_quantize_live(%f) returned code 0", double(v));
        CHECK(fp8_quantize_live(v) != 0, "fp8_quantize_live(%f) returned code 0", double(v));
    }
    CHECK(fp4_quantize_live(0.0f) != 0, "fp4_quantize_live(0.0) returned code 0");
    CHECK(fp8_quantize_live(0.0f) != 0, "fp8_quantize_live(0.0) returned code 0");
    CHECK(fp4_quantize_live(-0.0f) != 0, "fp4_quantize_live(-0.0) returned code 0");
    CHECK(fp8_quantize_live(-0.0f) != 0, "fp8_quantize_live(-0.0) returned code 0");
    CHECK(fp4_quantize_live(NAN) != 0, "fp4_quantize_live(NaN) returned code 0");
    CHECK(fp8_quantize_live(NAN) != 0, "fp8_quantize_live(NaN) returned code 0");
    // FP8 also has a distinct -0 byte (0x80) that decodes to the same 0.0f
    // -- must be unreachable too, not just 0x00.
    CHECK(fp8_quantize_live(0.0f) != 0x80, "fp8_quantize_live(0.0) returned -0 byte (0x80)");
    CHECK(fp8_quantize_live(-0.0f) != 0x80, "fp8_quantize_live(-0.0) returned -0 byte (0x80)");

    // ── (2) Never code 0 -- stochastic, hard floor over many draws ──────────
    const float fp4_test_vals[] = {0.0f,  -0.0f,  0.01f, -0.01f, 0.1f,   -0.1f,  0.2f, -0.2f,
                                   0.24f, -0.24f, 0.4f,  -0.4f,  0.5f,   -0.5f,  1.0f, -1.0f,
                                   2.0f,  -2.0f,  5.9f,  -5.9f,  100.0f, -100.0f};
    for (float v : fp4_test_vals) {
        for (int i = 0; i < 100000; ++i) {
            CHECK(fp4_quantize_stochastic_live(v) != 0,
                  "fp4_quantize_stochastic_live(%f) returned code 0 on draw %d", double(v), i);
        }
    }
    const float fp8_test_vals[] = {0.0f,   -0.0f, 1e-4f, -1e-4f, 0.001f,  -0.001f, 0.01f,
                                   -0.01f, 1.0f,  -1.0f, 100.0f, -100.0f, 500.0f,  -500.0f};
    for (float v : fp8_test_vals) {
        for (int i = 0; i < 100000; ++i) {
            const uint8_t code = fp8_quantize_stochastic_live(v);
            CHECK(code != 0x00, "fp8_quantize_stochastic_live(%f) returned +0 byte on draw %d",
                  double(v), i);
            CHECK(code != 0x80, "fp8_quantize_stochastic_live(%f) returned -0 byte on draw %d",
                  double(v), i);
        }
    }
    CHECK(fp4_quantize_stochastic_live(NAN) != 0,
          "fp4_quantize_stochastic_live(NaN) returned code 0");
    CHECK(fp8_quantize_stochastic_live(NAN) != 0,
          "fp8_quantize_stochastic_live(NaN) returned code 0");

    // ── (3) Sign-swing probability curve matches the design's intended ──────
    //        p_pos = (v + HALF_BITS) / (2*HALF_BITS) for FP4's [-0.5,0.5)
    //        bracket -- direct proof the redirect is probability-weighted,
    //        not a deterministic pin (an earlier draft of this design got
    //        this wrong -- see conversation).
    {
        const int N = 200000;
        const float grid[] = {-0.45f, -0.3f, -0.15f, -0.05f, 0.0f, 0.05f, 0.15f, 0.3f, 0.45f};
        for (float v : grid) {
            int pos_count = 0;
            for (int i = 0; i < N; ++i) {
                if (fp4_decode_bits(fp4_quantize_stochastic_live(v)) > 0.0f)
                    ++pos_count;
            }
            const double p_pos_measured = double(pos_count) / N;
            const double p_pos_expected = (double(v) + 0.5) / 1.0;
            CHECK(std::abs(p_pos_measured - p_pos_expected) < 0.01,
                  "fp4 sign-swing P(+0.5) for v=%f: measured=%f expected=%f", double(v),
                  p_pos_measured, p_pos_expected);
        }
        // v=0.0 must be close to a fair coin flip, NOT near-0 or near-1
        // (that would mean the redirect collapsed to a fixed sign again).
        int pos_count_zero = 0;
        for (int i = 0; i < N; ++i)
            if (fp4_decode_bits(fp4_quantize_stochastic_live(0.0f)) > 0.0f)
                ++pos_count_zero;
        const double p_zero = double(pos_count_zero) / N;
        CHECK(p_zero > 0.45 && p_zero < 0.55,
              "fp4 sign-swing at v=0.0 should be ~50/50, measured P(+0.5)=%f", p_zero);
    }
    {
        // Same curve check for FP8's [-2^-9,+2^-9) bracket.
        const int N = 200000;
        const float half = 1.0f / 512.0f;
        const float grid[] = {-0.9f * 1.0f / 512.0f, -0.5f / 512.0f, -0.1f / 512.0f, 0.0f,
                              0.1f / 512.0f,         0.5f / 512.0f,  0.9f / 512.0f};
        for (float v : grid) {
            int pos_count = 0;
            for (int i = 0; i < N; ++i) {
                if (fp8_decode_bits(fp8_quantize_stochastic_live(v)) > 0.0f)
                    ++pos_count;
            }
            const double p_pos_measured = double(pos_count) / N;
            const double p_pos_expected = (double(v) + double(half)) / (2.0 * double(half));
            CHECK(std::abs(p_pos_measured - p_pos_expected) < 0.01,
                  "fp8 sign-swing P(+2^-9) for v=%f: measured=%f expected=%f", double(v),
                  p_pos_measured, p_pos_expected);
        }
    }

    // ── (3c) Importance never-negative: fp4_quantize_stochastic_live_nonneg /
    //        fp8_quantize_stochastic_live_nonneg -- these exist because
    //        naively reusing weight's cross-sign SIGNED redirect for
    //        importance (a magnitude/energy quantity that is always >= 0
    //        and feeds sqrt(ci) throughout disldo_backward/sisldo_ops.hpp)
    //        gave importance up to a 50% chance of decoding to a NEGATIVE
    //        float near zero -- sqrt(negative) = NaN, silently poisoning
    //        training. Found and fixed via direct question during this
    //        session's implementation, confirmed by 3 real test
    //        regressions before the fix landed -- see conversation.
    {
        const float nonneg_vals[] = {0.0f, 1e-6f, 0.001f, 0.01f, 0.1f, 0.24f,
                                     0.4f, 0.49f, 0.5f,   1.0f,  5.9f, 100.0f};
        for (float v : nonneg_vals) {
            for (int i = 0; i < 50000; ++i) {
                CHECK(fp4_decode_bits(fp4_quantize_stochastic_live_nonneg(v)) >= 0.0f,
                      "fp4_quantize_stochastic_live_nonneg(%f) went negative on draw %d", double(v),
                      i);
                CHECK(fp8_decode_bits(fp8_quantize_stochastic_live_nonneg(v)) >= 0.0f,
                      "fp8_quantize_stochastic_live_nonneg(%f) went negative on draw %d", double(v),
                      i);
            }
        }
        // Deterministic live functions were already safe (never had this
        // bug) -- guard that stays true too.
        CHECK(fp4_decode_bits(fp4_quantize_live(0.0f)) >= 0.0f,
              "fp4_quantize_live(0.0) went negative");
        CHECK(fp8_decode_bits(fp8_quantize_live(0.0f)) >= 0.0f,
              "fp8_quantize_live(0.0) went negative");
    }

    // ── (3b) SIMD wrapper: same never-0 guarantee, all 4 lanes ──────────────
    {
        const float lane_vals[][4] = {
            {0.0f, -0.0f, 0.01f, -0.01f},
            {0.1f, -0.1f, 0.24f, -0.24f},
            {0.4f, -0.4f, 6.0f, -6.0f},
            {2.5f, -2.5f, NAN, 100.0f},
        };
        for (const auto& lv : lane_vals) {
            Block4Vec v = {lv[0], lv[1], lv[2], lv[3]};
            for (int i = 0; i < 20000; ++i) {
                const Block4VecU codes = block4_vec_quantize_stochastic_fp4_live(v);
                uint32_t code_arr[4];
                std::memcpy(code_arr, &codes, sizeof(code_arr));
                for (int lane = 0; lane < 4; ++lane) {
                    CHECK((code_arr[lane] & 0xFu) != 0,
                          "block4_vec_quantize_stochastic_fp4_live lane %d (v=%f) returned code 0 "
                          "on draw %d",
                          lane, double(lv[lane]), i);
                }
            }
        }
    }

    // ── (4) Regression guard: non-live functions must be completely untouched ──
    CHECK(fp4_quantize(0.0f) == 0,
          "fp4_quantize(0.0) should still be code 0 (non-live unaffected)");
    CHECK(fp8_quantize(0.0f) == 0,
          "fp8_quantize(0.0) should still be code 0 (non-live unaffected)");
    {
        // Confirm the non-live stochastic function can still land on 0 for
        // near-zero values (statistical -- at least one 0 across many draws).
        bool saw_zero_fp4 = false, saw_zero_fp8 = false;
        for (int i = 0; i < 10000; ++i) {
            if (fp4_quantize_stochastic(0.1f) == 0)
                saw_zero_fp4 = true;
            if (fp8_quantize_stochastic(0.001f) == 0)
                saw_zero_fp8 = true;
        }
        CHECK(saw_zero_fp4, "fp4_quantize_stochastic(0.1) never landed on code 0 in 10000 draws -- "
                            "non-live behavior changed?");
        CHECK(saw_zero_fp8, "fp8_quantize_stochastic(0.001) never landed on code 0 in 10000 draws "
                            "-- non-live behavior changed?");
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
