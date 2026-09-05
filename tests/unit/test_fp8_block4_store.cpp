// Correctness check for block4.hpp's Block4Store8/Block4TileHandle8 --
// same test structure as this repo's own test_block4_sparse_tile.cpp's
// Block4Store-specific tests, adapted for FP8's at_weight()/
// at_importance() (two byte accessors per slot) instead of FP4's
// single at().
#include "../../sili/lib/headers/delta_csr_types.hpp"
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

static void test_handle_dense_passthrough() {
    Block4Store8 store;
    store.init(16, 16);
    {
        auto h = store.get_or_create(0, 0);
        CHECK(bool(h), "get_or_create should return a valid handle");
        h.at_weight(0, 0) = 0x12;
        h.at_importance(0, 0) = 0x99;
        h.at_weight(1, 1) = 0x34;
    }
    {
        auto h = store.find(0, 0);
        CHECK(bool(h), "find() should locate the tile just created");
        CHECK(h.at_weight(0, 0) == 0x12, "at_weight(0,0) should read back 0x12, got %d",
              h.at_weight(0, 0));
        CHECK(h.at_importance(0, 0) == 0x99, "at_importance(0,0) should read back 0x99, got %d",
              h.at_importance(0, 0));
        CHECK(h.at_weight(1, 1) == 0x34, "at_weight(1,1) should read back 0x34, got %d",
              h.at_weight(1, 1));
        CHECK(h.at_weight(2, 2) == 0, "untouched slot weight should be 0");
        CHECK(h.at_importance(2, 2) == 0, "untouched slot importance should be 0");
    }
    CHECK(!bool(store.find(1, 1)), "find() on a nonexistent tile should be invalid");
}

static void test_switch_point_zero_disables_compression() {
    Block4Store8 store;
    store.init(16, 16);
    store.switch_point = 0;
    {
        auto h = store.get_or_create(0, 0);
        h.at_weight(0, 0) = 0x11;
    }
    store.maybe_compress(0, 0);
    CHECK(store.is_sparse(0, 0) == false, "switch_point=0 should never compress, even at count=1");
}

static void test_maybe_compress_explicit_check() {
    Block4Store8 store;
    store.init(16, 16);
    store.switch_point = 2;
    {
        auto h = store.get_or_create(0, 0);
        h.at_weight(0, 0) = 0x11;
        h.at_weight(0, 1) = 0x22;
        h.at_weight(0, 2) = 0x33; // 3 active > switch_point=2 -> dense
    }
    CHECK(store.is_sparse(0, 0) == false, "3 active > switch_point=2 should be dense");
    {
        auto h = store.find(0, 0);
        h.at_weight(0, 2) = 0x00;
    }
    CHECK(store.is_sparse(0, 0) == false,
          "dropping to 2 active must not auto-compress -- not automatic");
    store.maybe_compress(0, 0);
    CHECK(store.is_sparse(0, 0) == true,
          "2 active <= switch_point=2, explicit maybe_compress should compress");
    {
        auto h = store.find(0, 0);
        CHECK(h.at_weight(0, 0) == 0x11 && h.at_weight(0, 1) == 0x22 && h.at_weight(0, 2) == 0x00,
              "values survive compression");
    }

    Block4Store8 store2;
    store2.init(16, 16);
    store2.switch_point = 1;
    {
        auto h = store2.get_or_create(0, 0);
        h.at_weight(0, 0) = 0x11;
        h.at_weight(0, 1) = 0x22; // 2 active > switch_point=1
    }
    CHECK(store2.is_sparse(0, 0) == false, "2 active > switch_point=1 should be dense");
    store2.maybe_compress(0, 0);
    CHECK(store2.is_sparse(0, 0) == false,
          "2 active > switch_point=1, maybe_compress should NOT compress");
    store2.maybe_compress(5, 5); // nonexistent tile: no-op, no crash
}

static void test_sparse_to_dense_promotion_on_write() {
    Block4Store8 store;
    store.init(16, 16);
    store.switch_point = 2;
    {
        auto h = store.get_or_create(0, 0);
        h.at_weight(0, 0) = 0x11;
        h.at_weight(0, 1) = 0x22;
    }
    store.maybe_compress(0, 0);
    CHECK(store.is_sparse(0, 0) == true, "setup: should be sparse before the test");
    {
        auto h = store.find(0, 0);
        h.at_weight(0, 2) = 0x33; // now 3 active, > switch_point=2
    } // destructor should promote back to dense
    CHECK(store.is_sparse(0, 0) == false, "3 active > switch_point=2 should decompress to dense");
    {
        auto h = store.find(0, 0);
        CHECK(h.at_weight(0, 0) == 0x11 && h.at_weight(0, 1) == 0x22 && h.at_weight(0, 2) == 0x33,
              "values must survive the sparse->dense transition intact");
    }
}

static void test_weight_only_or_importance_only_slot_is_still_live() {
    // A slot is live iff EITHER weight or importance is nonzero (not
    // "both must be nonzero") -- e.g. a freshly-grown synapse with
    // weight=0 but nonzero importance must still count as live.
    Block4Store8 store;
    store.init(16, 16);
    {
        auto h = store.get_or_create(0, 0);
        h.at_weight(0, 0) = 0x00;
        h.at_importance(0, 0) = 0x42;
    }
    {
        auto h = store.find(0, 0);
        CHECK(h.count_live() == 1,
              "weight=0,importance!=0 slot should count as live, got count_live=%u",
              h.count_live());
    }
}

static void test_erase_while_handle_alive() {
    {
        Block4Store8 store;
        store.init(16, 16);
        {
            auto h = store.get_or_create(0, 0);
            h.at_weight(0, 0) = 0x11;
        }
        store.maybe_compress(0, 0); // now sparse
        CHECK(store.is_sparse(0, 0) == true, "setup: should be sparse");
        {
            auto h = store.find(0, 0);
            h.at_weight(0, 1) = 0x22;
            store.erase(0, 0);
        }
        CHECK(store.n_tiles() == 0,
              "tile should be gone after erase(); handle destructor must not resurrect it");
    }
    {
        Block4Store8 store;
        store.init(16, 16);
        store.switch_point = 0; // force dense
        {
            auto h = store.get_or_create(0, 0);
            h.at_weight(0, 0) = 0x11;
        }
        {
            auto h = store.find(0, 0);
            h.at_weight(0, 1) = 0x22;
            store.erase(0, 0);
        }
        CHECK(store.n_tiles() == 0, "tile should be gone (dense case)");
    }
}

static void test_live_synapses_matches_manual_count() {
    Block4Store8 store;
    store.init(16, 16);
    {
        auto h = store.get_or_create(0, 0);
        h.at_weight(0, 0) = 0x11;
        h.at_weight(0, 1) = 0x22;
        h.at_importance(1, 1) = 0x33; // weight=0, importance!=0 -- still live
    }
    CHECK(store.live_synapses() == 3, "live_synapses() = %zu, expected 3", store.live_synapses());
    store.maybe_compress(0, 0); // force sparse (3 <= default switch_point=12)
    CHECK(store.is_sparse(0, 0) == true,
          "setup: should compress at count=3 with default switch_point");
    CHECK(store.live_synapses() == 3, "live_synapses() after compression = %zu, expected 3",
          store.live_synapses());
}

static void test_multi_row_and_multi_tile_promotion_growth() {
    // Larger, more realistic scenario: many rows/tiles, forcing real
    // synaptogenesis-style growth (block4_ensure_row_headroom /
    // block4_row_insert_tile -- the shared, reused-unchanged functions)
    // across several block-rows, not just a single tile.
    Block4Store8 store;
    store.init(64, 64); // 16x16 block-grid
    for (uint32_t br = 0; br < 16; ++br) {
        for (uint32_t bc = 0; bc < 4; ++bc) {
            auto h = store.get_or_create(br, bc);
            h.at_weight(0, 0) = uint8_t(br + 1);
            h.at_importance(0, 0) = uint8_t(bc + 1);
        }
    }
    CHECK(store.n_tiles() == 64, "n_tiles() = %zu, expected 64", store.n_tiles());
    for (uint32_t br = 0; br < 16; ++br) {
        for (uint32_t bc = 0; bc < 4; ++bc) {
            auto h = store.find(br, bc);
            CHECK(bool(h), "tile (%u,%u) should be findable", br, bc);
            CHECK(h.at_weight(0, 0) == uint8_t(br + 1), "tile (%u,%u) weight corrupted", br, bc);
            CHECK(h.at_importance(0, 0) == uint8_t(bc + 1), "tile (%u,%u) importance corrupted", br,
                  bc);
        }
    }
}

int main() {
    test_handle_dense_passthrough();
    test_switch_point_zero_disables_compression();
    test_maybe_compress_explicit_check();
    test_sparse_to_dense_promotion_on_write();
    test_weight_only_or_importance_only_slot_is_still_live();
    test_erase_while_handle_alive();
    test_live_synapses_matches_manual_count();
    test_multi_row_and_multi_tile_promotion_growth();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
