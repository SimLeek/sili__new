// Correctness check for block4.hpp's FP32 dense/sparse tile encoding
// (Block4Tile32, block4_sparse_pack32/unpack32/packed_len32,
// block4_count_live32) -- same structure as test_fp8_block4_sparse_tile.cpp,
// adapted for 8-byte (4-byte weight + 4-byte importance) float32 slots
// instead of FP8's 2-byte codes. No quantization anywhere in this file --
// every check here is bit-exact, not tolerance-based.
#include "../../sili/lib/headers/delta_csr_types.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <algorithm>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

static Block4Tile32 make_random_dense32(std::mt19937& rng, int n) {
    Block4Tile32 t;
    std::vector<int> positions(16);
    for (int i = 0; i < 16; ++i)
        positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    // Nonzero, exactly-representable floats -- no rounding concerns since
    // there's no codec here at all.
    std::uniform_real_distribution<float> val_dist(1.0f, 1000.0f);
    for (int i = 0; i < n; ++i) {
        const int pos = positions[i];
        const float w = val_dist(rng);
        const float imp = val_dist(rng);
        std::memcpy(t.data + sizeof(float) * pos, &w, sizeof(w));
        std::memcpy(t.data + sizeof(float) * (BLOCK4_TILE_SLOTS + pos), &imp, sizeof(imp));
    }
    return t;
}

static void test_pack_unpack_roundtrip() {
    std::mt19937 rng(0);
    int tested = 0;
    for (int n = 0; n <= 14; ++n) {
        for (int trial = 0; trial < 2000; ++trial) {
            Block4Tile32 dense = make_random_dense32(rng, n);
            uint8_t packed[BLOCK4_TILE_SLOTS32_BYTES] = {0};
            const uint8_t written_count = block4_sparse_pack32(dense.data, packed);
            CHECK(packed[0] == n, "pack32: count byte should be %d, got %d", n, packed[0]);
            CHECK(written_count == n, "pack32: returned count should be %d, got %d", n,
                  written_count);
            const std::size_t expect_len = block4_sparse_packed_len32(uint8_t(n));
            CHECK(expect_len == std::size_t(1 + (n + 1) / 2 + n * 2 * sizeof(float)),
                  "block4_sparse_packed_len32(%d) should be 1+ceil(n/2)+8n = %zu, got %zu", n,
                  std::size_t(1 + (n + 1) / 2 + n * 2 * sizeof(float)), expect_len);
            CHECK(expect_len <= BLOCK4_TILE_SLOTS32_BYTES,
                  "packed length must never exceed the dense size at n=%d", n);
            uint8_t roundtrip[BLOCK4_TILE_SLOTS32_BYTES];
            block4_sparse_unpack32(packed, roundtrip);
            CHECK(std::memcmp(dense.data, roundtrip, BLOCK4_TILE_SLOTS32_BYTES) == 0,
                  "pack32/unpack32 round trip mismatch at n=%d trial=%d", n, trial);
            ++tested;
        }
    }
    std::printf("test_pack_unpack_roundtrip32: %d cases tested\n", tested);
}

static void test_count_live_matches_pack_count() {
    std::mt19937 rng(1);
    for (int n = 0; n <= 14; ++n) {
        Block4Tile32 dense = make_random_dense32(rng, n);
        CHECK(dense.count_live() == uint32_t(n), "count_live() = %u, expected %d",
              dense.count_live(), n);
        CHECK(block4_count_live32(dense.data) == uint32_t(n),
              "block4_count_live32() = %u, expected %d", block4_count_live32(dense.data), n);
    }
}

static void test_max_sparse_count_fits_budget() {
    // The whole point of BLOCK4_SPARSE_MAX_COUNT32=14 (vs FP4's 10, FP8's
    // 12): pack exactly that many live slots and confirm it still fits.
    std::mt19937 rng(2);
    Block4Tile32 dense = make_random_dense32(rng, int(BLOCK4_SPARSE_MAX_COUNT32));
    uint8_t packed[BLOCK4_TILE_SLOTS32_BYTES];
    block4_sparse_pack32(dense.data, packed);
    const std::size_t len = block4_sparse_packed_len32(uint8_t(BLOCK4_SPARSE_MAX_COUNT32));
    CHECK(len <= BLOCK4_TILE_SLOTS32_BYTES,
          "max sparse count %u packs to %zu bytes, exceeds dense size %u",
          BLOCK4_SPARSE_MAX_COUNT32, len, BLOCK4_TILE_SLOTS32_BYTES);
}

static void test_weight_and_importance_independently_addressable() {
    Block4Tile32 t;
    t.set_weight(1, 2, 42.5f);
    t.set_importance(1, 2, 99.25f);
    CHECK(t.get_weight(1, 2) == 42.5f, "get_weight readback failed");
    CHECK(t.get_importance(1, 2) == 99.25f, "get_importance readback failed");
    // Different slots must not alias.
    t.set_weight(0, 0, 11.0f);
    CHECK(t.get_weight(1, 2) == 42.5f, "set_weight(0,0) write aliased slot (1,2)");
    CHECK(t.get_importance(1, 2) == 99.25f, "set_weight(0,0) write aliased importance half");
}

int main() {
    test_pack_unpack_roundtrip();
    test_count_live_matches_pack_count();
    test_max_sparse_count_fits_budget();
    test_weight_and_importance_independently_addressable();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
