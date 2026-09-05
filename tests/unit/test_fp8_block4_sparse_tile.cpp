// Correctness check for block4.hpp's FP8 dense/sparse tile encoding
// (Block4Tile8, block4_sparse_pack8/unpack8/packed_len8,
// block4_count_live8) -- same structure as this repo's own
// test_block4_sparse_tile.cpp, adapted for 2-byte (weight+importance)
// slots instead of FP4's 1-byte nibble-packed slots.
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

static Block4Tile8 make_random_dense8(std::mt19937& rng, int n) {
    Block4Tile8 t;
    std::vector<int> positions(16);
    for (int i = 0; i < 16; ++i)
        positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    std::uniform_int_distribution<int> val_dist(1, 255);
    for (int i = 0; i < n; ++i) {
        const int pos = positions[i];
        t.data[pos] = uint8_t(val_dist(rng));      // weight
        t.data[16 + pos] = uint8_t(val_dist(rng)); // importance
    }
    return t;
}

static void test_pack_unpack_roundtrip() {
    std::mt19937 rng(0);
    int tested = 0;
    for (int n = 0; n <= 12; ++n) {
        for (int trial = 0; trial < 2000; ++trial) {
            Block4Tile8 dense = make_random_dense8(rng, n);
            uint8_t packed[32] = {0};
            const uint8_t written_count = block4_sparse_pack8(dense.data, packed);
            CHECK(packed[0] == n, "pack8: count byte should be %d, got %d", n, packed[0]);
            CHECK(written_count == n, "pack8: returned count should be %d, got %d", n,
                  written_count);
            const std::size_t expect_len = block4_sparse_packed_len8(uint8_t(n));
            CHECK(expect_len == std::size_t(1 + (n + 1) / 2 + n * 2),
                  "block4_sparse_packed_len8(%d) should be 1+ceil(n/2)+2n = %zu, got %zu", n,
                  std::size_t(1 + (n + 1) / 2 + n * 2), expect_len);
            CHECK(expect_len <= 32, "packed length must never exceed the dense size at n=%d", n);
            uint8_t roundtrip[32];
            block4_sparse_unpack8(packed, roundtrip);
            CHECK(std::memcmp(dense.data, roundtrip, 32) == 0,
                  "pack8/unpack8 round trip mismatch at n=%d trial=%d", n, trial);
            ++tested;
        }
    }
    std::printf("test_pack_unpack_roundtrip8: %d cases tested\n", tested);
}

static void test_count_live_matches_pack_count() {
    std::mt19937 rng(1);
    for (int n = 0; n <= 12; ++n) {
        Block4Tile8 dense = make_random_dense8(rng, n);
        CHECK(dense.count_live() == uint32_t(n), "count_live() = %u, expected %d",
              dense.count_live(), n);
        CHECK(block4_count_live8(dense.data) == uint32_t(n),
              "block4_count_live8() = %u, expected %d", block4_count_live8(dense.data), n);
    }
}

static void test_max_sparse_count_fits_budget() {
    // The whole point of BLOCK4_SPARSE_MAX_COUNT8=12 (vs FP4's 10): pack
    // exactly that many live slots and confirm it still fits in <= 32 bytes.
    std::mt19937 rng(2);
    Block4Tile8 dense = make_random_dense8(rng, int(BLOCK4_SPARSE_MAX_COUNT8));
    uint8_t packed[32];
    block4_sparse_pack8(dense.data, packed);
    const std::size_t len = block4_sparse_packed_len8(uint8_t(BLOCK4_SPARSE_MAX_COUNT8));
    CHECK(len <= 32, "max sparse count %u packs to %zu bytes, exceeds dense size 32",
          BLOCK4_SPARSE_MAX_COUNT8, len);
}

static void test_weight_and_importance_independently_addressable() {
    Block4Tile8 t;
    t.at_weight(1, 2) = 0x42;
    t.at_importance(1, 2) = 0x99;
    CHECK(t.at_weight(1, 2) == 0x42, "at_weight readback failed");
    CHECK(t.at_importance(1, 2) == 0x99, "at_importance readback failed");
    // Different slots must not alias.
    t.at_weight(0, 0) = 0x11;
    CHECK(t.at_weight(1, 2) == 0x42, "at_weight(0,0) write aliased slot (1,2)");
    CHECK(t.at_importance(1, 2) == 0x99, "at_weight(0,0) write aliased importance half");
}

int main() {
    test_pack_unpack_roundtrip();
    test_count_live_matches_pack_count();
    test_max_sparse_count_fits_budget();
    test_weight_and_importance_independently_addressable();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
