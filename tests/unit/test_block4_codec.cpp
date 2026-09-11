// Phase 1 correctness gate for Block4Codec<VALUES_TYPE> (block4_codec.hpp):
// proves the trait is a real passthrough to the EXISTING fp4quant.hpp/
// fp8quant.hpp/raw-float logic, not a reimplementation with its own bugs.
// No kernel (disldo_forward/disldo_backward) calls the trait yet -- this
// test exists purely to gate that the wrapper itself is correct BEFORE
// any real kernel is rewired onto it in later phases. See
// docs/research/linear_disldo.rst:block4_codec_refactor.
#include "../../sili/lib/headers/block4_codec.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Both fp4/fp8 have a NaN-decoding slot; a byte that lands there makes both
// sides of a "same computation twice" comparison independently produce NaN,
// and NaN != NaN in IEEE754 even when it's the identical bit pattern -- so
// equality checks on decoded values must treat NaN==NaN as a pass.
static bool float_eq(float a, float b) {
    return (a == b) || (std::isnan(a) && std::isnan(b));
}

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

static void test_fp4() {
    using Codec = Block4Codec<FP4BiPacked>;
    CHECK(Codec::scratch_bytes == BLOCK4_TILE_SLOTS, "fp4 scratch_bytes mismatch");
    CHECK(Codec::has_zero_escape, "fp4 must have_zero_escape");
    CHECK(Codec::has_was_live_gate, "fp4 must have_was_live_gate");

    uint8_t tdata[BLOCK4_TILE_SLOTS];
    std::mt19937 rng(112233);
    for (auto& b : tdata)
        b = uint8_t(rng() & 0xFFu);

    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            const uint8_t byte = tdata[Block4Tile::slot_index(li, lj)];
            const float expect_w = FP4_TABLE[byte & 0xFu];
            const float expect_i = FP4_TABLE[(byte >> 4) & 0xFu];
            CHECK(float_eq(Codec::decode_weight(tdata, li, lj), expect_w),
                  "fp4 decode_weight mismatch");
            CHECK(float_eq(Codec::decode_importance(tdata, li, lj), expect_i),
                  "fp4 decode_importance mismatch");
        }
    }

    CHECK(Codec::quant_floor(0.0f, 0.1f) == 0.1f, "fp4 quant_floor(0) should substitute eps");
    CHECK(Codec::quant_floor(2.0f, 0.1f) == 2.0f, "fp4 quant_floor(nonzero) should pass through");

    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
        const Block4Vec col = Codec::decode_weight_column4(tdata, lj);
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
            CHECK(float_eq(col[li], Codec::decode_weight(tdata, li, lj)),
                  "fp4 decode_weight_column4 mismatch at li=%u lj=%u", li, lj);
    }

    // Encode round-trip: for each was_live x StochasticRounding=false combo,
    // the trait's write-back byte must match calling fp4_quantize{,_live}
    // directly and packing the same way process_tile_pair_fp4 does.
    for (int was_live = 0; was_live <= 1; ++was_live) {
        for (float w : {-3.0f, 0.0f, 1.5f, 6.0f}) {
            for (float imp : {0.0f, 0.5f, 4.0f}) {
                uint8_t buf[BLOCK4_TILE_SLOTS] = {0};
                Codec::encode<false>(buf, 1, 2, w, imp, was_live != 0);
                const uint8_t got = buf[Block4Tile::slot_index(1, 2)];
                const uint8_t expect_w = was_live ? fp4_quantize_live(w) : fp4_quantize(w);
                const uint8_t expect_i = was_live ? fp4_quantize_live(imp) : fp4_quantize(imp);
                const uint8_t expect = uint8_t((expect_i << 4) | expect_w);
                CHECK(got == expect, "fp4 encode<false> mismatch (was_live=%d w=%g imp=%g)",
                      was_live, w, imp);
            }
        }
    }

    CHECK(Codec::stored_tile_len(false, tdata) == BLOCK4_TILE_SLOTS,
          "fp4 stored_tile_len dense mismatch");
}

static void test_fp8() {
    using Codec = Block4Codec<FP8BiValues>;
    CHECK(Codec::scratch_bytes == BLOCK4_TILE_SLOTS8_BYTES, "fp8 scratch_bytes mismatch");
    CHECK(!Codec::has_zero_escape, "fp8 must NOT have_zero_escape");
    CHECK(Codec::has_was_live_gate, "fp8 must have_was_live_gate");

    uint8_t tdata[BLOCK4_TILE_SLOTS8_BYTES];
    std::mt19937 rng(445566);
    for (auto& b : tdata)
        b = uint8_t(rng() & 0xFFu);

    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            const float expect_w = fp8_decode_bits(tdata[Block4Tile8::slot_index(li, lj)]);
            const float expect_i =
                fp8_decode_bits(tdata[BLOCK4_TILE_SLOTS + Block4Tile8::slot_index(li, lj)]);
            CHECK(float_eq(Codec::decode_weight(tdata, li, lj), expect_w),
                  "fp8 decode_weight mismatch");
            CHECK(float_eq(Codec::decode_importance(tdata, li, lj), expect_i),
                  "fp8 decode_importance mismatch");
        }
    }

    CHECK(Codec::quant_floor(0.0f, 0.1f) == 0.0f, "fp8 quant_floor must be identity");
    CHECK(Codec::quant_floor(3.0f, 0.1f) == 3.0f, "fp8 quant_floor must be identity");

    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
        const Block4Vec col = Codec::decode_weight_column4(tdata, lj);
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
            CHECK(float_eq(col[li], Codec::decode_weight(tdata, li, lj)),
                  "fp8 decode_weight_column4 mismatch at li=%u lj=%u", li, lj);
    }

    for (int was_live = 0; was_live <= 1; ++was_live) {
        for (float w : {-100.0f, 0.0f, 12.5f}) {
            for (float imp : {0.0f, 0.25f, 8.0f}) {
                uint8_t buf[BLOCK4_TILE_SLOTS8_BYTES] = {0};
                Codec::encode<false>(buf, 0, 3, w, imp, was_live != 0);
                const uint32_t slot = Block4Tile8::slot_index(0, 3);
                const uint8_t got_w = buf[slot];
                const uint8_t got_i = buf[BLOCK4_TILE_SLOTS + slot];
                const uint8_t expect_w = was_live ? fp8_quantize_live(w) : fp8_quantize(w);
                const uint8_t expect_i = was_live ? fp8_quantize_live(imp) : fp8_quantize(imp);
                CHECK(got_w == expect_w, "fp8 encode<false> weight mismatch (was_live=%d w=%g)",
                      was_live, w);
                CHECK(got_i == expect_i,
                      "fp8 encode<false> importance mismatch (was_live=%d imp=%g)", was_live, imp);
            }
        }
    }

    CHECK(Codec::stored_tile_len(false, tdata) == BLOCK4_TILE_SLOTS8_BYTES,
          "fp8 stored_tile_len dense mismatch");
}

static void test_fp32() {
    using Codec = Block4Codec<DeltaCSRBiValues<float>>;
    CHECK(Codec::scratch_bytes == BLOCK4_TILE_SLOTS32_BYTES, "fp32 scratch_bytes mismatch");
    CHECK(!Codec::has_zero_escape, "fp32 must NOT have_zero_escape");
    CHECK(!Codec::has_was_live_gate, "fp32 must NOT have_was_live_gate");

    uint8_t tdata[BLOCK4_TILE_SLOTS32_BYTES];
    std::mt19937 rng(778899);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            const float w = dist(rng), imp = dist(rng);
            std::memcpy(tdata + sizeof(float) * Block4Tile32::slot_index(li, lj), &w, sizeof(w));
            std::memcpy(tdata +
                            sizeof(float) * (BLOCK4_TILE_SLOTS + Block4Tile32::slot_index(li, lj)),
                        &imp, sizeof(imp));
            CHECK(Codec::decode_weight(tdata, li, lj) == w, "fp32 decode_weight mismatch");
            CHECK(Codec::decode_importance(tdata, li, lj) == imp,
                  "fp32 decode_importance mismatch");
        }
    }

    CHECK(Codec::quant_floor(0.0f, 0.1f) == 0.0f, "fp32 quant_floor must be identity");

    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
        const Block4Vec col = Codec::decode_weight_column4(tdata, lj);
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
            CHECK(col[li] == Codec::decode_weight(tdata, li, lj),
                  "fp32 decode_weight_column4 mismatch at li=%u lj=%u", li, lj);
    }

    uint8_t buf[BLOCK4_TILE_SLOTS32_BYTES] = {0};
    Codec::encode<false>(buf, 2, 1, 3.5f, -1.25f, true);
    CHECK(Codec::decode_weight(buf, 2, 1) == 3.5f, "fp32 encode weight round-trip mismatch");
    CHECK(Codec::decode_importance(buf, 2, 1) == -1.25f,
          "fp32 encode importance round-trip mismatch");

    CHECK(Codec::stored_tile_len(false, tdata) == BLOCK4_TILE_SLOTS32_BYTES,
          "fp32 stored_tile_len dense mismatch");
}

int main() {
    test_fp4();
    test_fp8();
    test_fp32();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
