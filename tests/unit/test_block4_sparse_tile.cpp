// Correctness check for block4's sparse-packed tile encoding
// (block4_sparse_pack/unpack, Block4StoredTile, Block4TileHandle) --
// see block4.hpp and TODO_DUAL_BLOCK4.md's "Part B" writeup.
//
// SIMD unpack was prototyped and speed-tested separately (three variants,
// all measurably slower than plain scalar at every count 0-10, see
// TODO_DUAL_BLOCK4.md) -- only the winning scalar approach is here.
#include "../../sili/lib/headers/delta_csr_types.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <algorithm>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); std::fflush(stdout); ++g_fail; } \
} while (0)

static Block4Tile make_random_dense(std::mt19937& rng, int n) {
    Block4Tile t;
    std::vector<int> positions(16);
    for (int i = 0; i < 16; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    std::uniform_int_distribution<int> val_dist(1, 255);
    for (int i = 0; i < n; ++i) t.data[positions[i]] = uint8_t(val_dist(rng));
    return t;
}

static void test_pack_unpack_roundtrip() {
    std::mt19937 rng(0);
    int tested = 0;
    for (int n = 0; n <= 10; ++n) {
        for (int trial = 0; trial < 2000; ++trial) {
            Block4Tile dense = make_random_dense(rng, n);
            uint8_t packed[16] = {0};
            block4_sparse_pack(dense.data, packed);
            CHECK(packed[0] == n, "pack: count byte should be %d, got %d", n, packed[0]);
            uint8_t roundtrip[16];
            block4_sparse_unpack(packed, roundtrip);
            CHECK(std::memcmp(dense.data, roundtrip, 16) == 0,
                  "pack/unpack round trip mismatch at n=%d trial=%d", n, trial);
            ++tested;
        }
    }
    std::printf("test_pack_unpack_roundtrip: %d cases tested\n", tested);
}

static void test_handle_dense_passthrough() {
    Block4Store store;
    store.init(16, 16);
    {
        auto h = store.get_or_create(0, 0);
        CHECK(bool(h), "get_or_create should return a valid handle");
        h.at(0, 0) = 0x12;
        h.at(1, 1) = 0x34;
    }
    {
        auto h = store.find(0, 0);
        CHECK(bool(h), "find() should locate the tile just created");
        CHECK(h.at(0, 0) == 0x12, "at(0,0) should read back 0x12, got %d", h.at(0, 0));
        CHECK(h.at(1, 1) == 0x34, "at(1,1) should read back 0x34, got %d", h.at(1, 1));
        CHECK(h.at(2, 2) == 0, "untouched slot should be 0");
    }
    CHECK(!bool(store.find(1, 1)), "find() on a nonexistent tile should be invalid");
}

static void test_switch_point_zero_disables_compression() {
    Block4Store store;
    store.init(16, 16);
    store.switch_point = 0;
    {
        auto h = store.get_or_create(0, 0);
        h.at(0, 0) = 0x11;
    }
    store.maybe_compress(0, 0);
    CHECK(store.tile_values[0].is_sparse == false, "switch_point=0 should never compress, even at count=1");
}

static void test_maybe_compress_explicit_check() {
    Block4Store store;
    store.init(16, 16);
    store.switch_point = 2;
    {
        auto h = store.get_or_create(0, 0);
        h.at(0, 0) = 0x11;
        h.at(0, 1) = 0x22;
    }
    CHECK(store.tile_values[0].is_sparse == false, "fresh tile stays dense until maybe_compress -- not automatic");
    store.maybe_compress(0, 0);
    CHECK(store.tile_values[0].is_sparse == true, "2 active <= switch_point=2, maybe_compress should compress");
    {
        auto h = store.find(0, 0);
        CHECK(h.at(0, 0) == 0x11 && h.at(0, 1) == 0x22, "values survive compression");
    }

    Block4Store store2;
    store2.init(16, 16);
    store2.switch_point = 1;
    {
        auto h = store2.get_or_create(0, 0);
        h.at(0, 0) = 0x11;
        h.at(0, 1) = 0x22; // 2 active > switch_point=1
    }
    store2.maybe_compress(0, 0);
    CHECK(store2.tile_values[0].is_sparse == false, "2 active > switch_point=1, maybe_compress should NOT compress");
    store2.maybe_compress(5, 5); // nonexistent tile: no-op, no crash
}

static void test_sparse_to_dense_promotion_on_write() {
    Block4Store store;
    store.init(16, 16);
    store.switch_point = 2;
    {
        auto h = store.get_or_create(0, 0);
        h.at(0, 0) = 0x11;
        h.at(0, 1) = 0x22;
    }
    store.maybe_compress(0, 0);
    CHECK(store.tile_values[0].is_sparse == true, "setup: should be sparse before the test");
    {
        auto h = store.find(0, 0);
        h.at(0, 2) = 0x33; // now 3 active, > switch_point=2
    } // destructor should promote back to dense
    CHECK(store.tile_values[0].is_sparse == false, "3 active > switch_point=2 should decompress to dense");
    {
        auto h = store.find(0, 0);
        CHECK(h.at(0, 0) == 0x11 && h.at(0, 1) == 0x22 && h.at(0, 2) == 0x33,
              "values must survive the sparse->dense transition intact");
    }
}

static void test_erase_while_handle_alive() {
    // The real hazard this design has to handle correctly: a handle for
    // (br,bc) still logically alive when erase() removes that SAME
    // (br,bc) -- must not crash (ASan) or write through a stale pointer.
    // See Block4TileHandle's class comment (block4.hpp) for why the
    // destructor re-fetches by coordinate instead of a cached pointer.
    {
        Block4Store store;
        store.init(16, 16);
        {
            auto h = store.get_or_create(0, 0);
            h.at(0, 0) = 0x11;
        }
        store.maybe_compress(0, 0); // now sparse
        CHECK(store.tile_values[0].is_sparse == true, "setup: should be sparse");
        {
            auto h = store.find(0, 0); // sparse handle, unpacks
            h.at(0, 1) = 0x22;         // dirty write into scratch
            store.erase(0, 0);         // erase the SAME tile while h is still alive
        } // h's destructor runs here, AFTER erase()
        CHECK(store.n_tiles() == 0, "tile should be gone after erase(); handle destructor must not resurrect it");
    }
    {
        // Same hazard, dense tile (destructor is a no-op regardless, but
        // verify no crash either way).
        Block4Store store;
        store.init(16, 16);
        store.switch_point = 0; // force dense
        {
            auto h = store.get_or_create(0, 0);
            h.at(0, 0) = 0x11;
        }
        {
            auto h = store.find(0, 0);
            h.at(0, 1) = 0x22;
            store.erase(0, 0);
        }
        CHECK(store.n_tiles() == 0, "tile should be gone (dense case)");
    }
}

static void test_move_semantics() {
    Block4Store store;
    store.init(16, 16);
    auto h1 = store.get_or_create(0, 0);
    h1.at(0, 0) = 0x11;
    auto h2 = std::move(h1);
    CHECK(!bool(h1), "moved-from handle should be invalid");
    CHECK(bool(h2), "moved-to handle should be valid");
    h2.at(0, 1) = 0x22;
}

int main() {
    test_pack_unpack_roundtrip();
    test_handle_dense_passthrough();
    test_switch_point_zero_disables_compression();
    test_maybe_compress_explicit_check();
    test_sparse_to_dense_promotion_on_write();
    test_erase_while_handle_alive();
    test_move_semantics();

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
