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
            const uint8_t written_count = block4_sparse_pack(dense.data, packed);
            CHECK(packed[0] == n, "pack: count byte should be %d, got %d", n, packed[0]);
            CHECK(written_count == n, "pack: returned count should be %d, got %d", n, written_count);
            const std::size_t expect_len = block4_sparse_packed_len(uint8_t(n));
            CHECK(expect_len == std::size_t(1 + (n + 1) / 2 + n),
                  "block4_sparse_packed_len(%d) should be 1+ceil(n/2)+n = %zu, got %zu",
                  n, std::size_t(1 + (n + 1) / 2 + n), expect_len);
            CHECK(expect_len <= 16, "packed length must never exceed the dense size at n=%d", n);
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
    CHECK(store.is_sparse(0, 0) == false, "switch_point=0 should never compress, even at count=1");
}

static void test_maybe_compress_explicit_check() {
    // A FRESH tile sizes itself sparse immediately if it qualifies (see
    // block4.hpp: new tiles start empty, not dense) -- maybe_compress's
    // real remaining job is a tile that's ALREADY dense (forced there by
    // exceeding switch_point) and later drops back to low occupancy
    // (e.g. via pruning zeroing some slots): that shrink is NOT
    // automatic, only an explicit maybe_compress() call re-evaluates it.
    Block4Store store;
    store.init(16, 16);
    store.switch_point = 2;
    {
        auto h = store.get_or_create(0, 0);
        h.at(0, 0) = 0x11;
        h.at(0, 1) = 0x22;
        h.at(0, 2) = 0x33; // 3 active > switch_point=2 -> dense
    }
    CHECK(store.is_sparse(0, 0) == false, "3 active > switch_point=2 should be dense");
    {
        // Simulate pruning: zero one slot, bringing live count to 2
        // (<=switch_point). A dense tile's destructor never re-evaluates
        // this on its own.
        auto h = store.find(0, 0);
        h.at(0, 2) = 0x00;
    }
    CHECK(store.is_sparse(0, 0) == false, "dropping to 2 active must not auto-compress -- not automatic");
    store.maybe_compress(0, 0);
    CHECK(store.is_sparse(0, 0) == true, "2 active <= switch_point=2, explicit maybe_compress should compress");
    {
        auto h = store.find(0, 0);
        CHECK(h.at(0, 0) == 0x11 && h.at(0, 1) == 0x22 && h.at(0, 2) == 0x00, "values survive compression");
    }

    Block4Store store2;
    store2.init(16, 16);
    store2.switch_point = 1;
    {
        auto h = store2.get_or_create(0, 0);
        h.at(0, 0) = 0x11;
        h.at(0, 1) = 0x22; // 2 active > switch_point=1
    }
    CHECK(store2.is_sparse(0, 0) == false, "2 active > switch_point=1 should be dense");
    store2.maybe_compress(0, 0);
    CHECK(store2.is_sparse(0, 0) == false, "2 active > switch_point=1, maybe_compress should NOT compress");
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
    CHECK(store.is_sparse(0, 0) == true, "setup: should be sparse before the test");
    {
        auto h = store.find(0, 0);
        h.at(0, 2) = 0x33; // now 3 active, > switch_point=2
    } // destructor should promote back to dense
    CHECK(store.is_sparse(0, 0) == false, "3 active > switch_point=2 should decompress to dense");
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
        CHECK(store.is_sparse(0, 0) == true, "setup: should be sparse");
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

static void test_real_compression_shrinks_footprint() {
    // The actual point of this redesign: a low-occupancy tile must use
    // FEWER real bytes in tile_data than the old fixed dense-worst-case
    // reservation, not just get a repacked-in-place is_sparse flag
    // inside an already-dense-sized slot (see block4.hpp's old todo
    // comments -- that was the bug). New tiles size themselves
    // immediately based on real content (see block4.hpp: new tiles
    // start empty, not dense), so a fresh 2-live tile is ALREADY at its
    // compressed size the moment its creating handle destructs -- no
    // separate maybe_compress() call needed for a brand new tile.
    Block4Store store;
    store.init(16, 16);
    {
        auto h = store.get_or_create(0, 0);
        h.at(0, 0) = 0x11;
        h.at(0, 1) = 0x22; // 2 live synapses
    }
    CHECK(store.is_sparse(0, 0), "2 <= default switch_point (10) should size itself sparse immediately");
    const std::size_t compressed_bytes = store.total_tile_used_bytes();
    const std::size_t expect_bytes = block4_sparse_packed_len(2); // 1 + 1 + 2 = 4
    CHECK(compressed_bytes == expect_bytes,
          "2-live tile should use exactly %zu bytes, got %zu", expect_bytes, compressed_bytes);
    CHECK(compressed_bytes < BLOCK4_TILE_SLOTS,
          "real per-tile footprint (%zu bytes) must genuinely beat the old fixed dense-worst-case"
          " reservation (%u bytes)", compressed_bytes, BLOCK4_TILE_SLOTS);
    {
        auto h = store.find(0, 0);
        CHECK(h.at(0, 0) == 0x11 && h.at(0, 1) == 0x22, "values survive real compression");
    }

    // maybe_compress's real remaining job: a tile that's ALREADY dense
    // (forced there by exceeding switch_point) and later drops back to
    // low occupancy through pruning does NOT automatically re-shrink --
    // only an explicit maybe_compress() call does that.
    Block4Store store2;
    store2.init(16, 16);
    store2.switch_point = 2;
    {
        auto h = store2.get_or_create(0, 1);
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj)
                h.at(li, lj) = 0x11; // fully dense, 16 live > switch_point=2
    }
    CHECK(!store2.is_sparse(0, 1), "setup: fully dense tile should stay dense");
    const std::size_t dense_bytes = store2.total_tile_used_bytes();
    CHECK(dense_bytes == BLOCK4_TILE_SLOTS,
          "fully dense tile should use exactly %u bytes, got %zu", BLOCK4_TILE_SLOTS, dense_bytes);
    {
        auto h = store2.find(0, 1);
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj)
                if (!(li == 0 && (lj == 0 || lj == 1))) h.at(li, lj) = 0x00; // prune down to 2 live
    }
    CHECK(!store2.is_sparse(0, 1), "dropping to 2 live must not auto-compress -- not automatic");
    store2.maybe_compress(0, 1);
    CHECK(store2.is_sparse(0, 1), "explicit maybe_compress should now compress the pruned-down tile");
    const std::size_t shrunk_bytes = store2.total_tile_used_bytes();
    CHECK(shrunk_bytes == block4_sparse_packed_len(2),
          "pruned-then-compressed tile should use exactly %zu bytes, got %zu",
          block4_sparse_packed_len(2), shrunk_bytes);
    CHECK(shrunk_bytes < dense_bytes,
          "compression must genuinely shrink used bytes (%zu -> %zu), not just flip a flag in a fixed slot",
          dense_bytes, shrunk_bytes);
    // Note: total_tile_alloc_bytes() (row headroom) doesn't necessarily
    // shrink to match total_tile_used_bytes() immediately -- like the
    // scattered CSR side's own blank space, freed bytes stay as reusable
    // row-local slack until Block4Store::equalize_step() redistributes
    // it (see test_block4_equalize_step_redistributes_headroom).
}

static void test_equalize_step_redistributes_block4_row_headroom() {
    // Block4Store::equalize_step mirrors the scattered CSR side's own
    // delta_csr_equalize_step: the ENTIRE store's budget (max_tile_bytes)
    // stays fixed, but how much of it each ROW currently reserves is not
    // -- bytes a row's tile freed by pruning/compressing shouldn't stay
    // permanently locked to that one row forever (see block4.hpp's
    // comment on Block4Store::equalize_step).
    //
    // Only 2 block rows -- block4_row_shift (like delta_csr_shift_row)
    // never touches the LAST row directly, so with more rows the
    // store-wide average gets diluted by rows that were never involved
    // at all, making the exact numbers unpredictable. With exactly 2,
    // row 0 is the only one equalize_step ever resizes, and shrinking it
    // directly grows row 1's window (row 1's start boundary IS row 0's
    // end boundary) -- a clean, deterministic redistribution to check.
    Block4Store store;
    store.init(8, 16); // 2 block rows, 4 block cols
    store.switch_point = 10;

    // Row 0: build a tile fully dense (16 bytes used+reserved)...
    {
        auto h = store.get_or_create(0, 0);
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj)
                h.at(li, lj) = 0x11;
    }
    CHECK(!store.is_sparse(0, 0), "setup: row0 tile should be dense (16 bytes)");
    // ...then prune it back down to 2 live and compress -- alloc stays
    // 16 (rows never shrink their own allocation on a compress/prune),
    // used drops to 4, leaving 12 bytes of slack ONLY row 0 can reach.
    {
        auto h = store.find(0, 0);
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj)
                if (!(li == 0 && (lj == 0 || lj == 1))) h.at(li, lj) = 0x00;
    }
    store.maybe_compress(0, 0);
    CHECK(store.is_sparse(0, 0), "setup: row0 tile pruned+compressed to 4 bytes, 12 bytes of unreachable slack remain");

    // Row 1: a small tile at its own natural (zero-slack) size.
    {
        auto h = store.get_or_create(1, 0);
        h.at(0, 0) = 0x11;
        h.at(0, 1) = 0x22; // 2 live -> 4 bytes, zero headroom
    }
    CHECK(store.is_sparse(1, 0), "setup: row1 tile should size itself sparse (4 bytes)");

    const std::size_t total_before = store.total_tile_alloc_bytes();
    // Cap the GLOBAL budget at exactly what's allocated right now -- row
    // 1 cannot grow AT ALL without reclaiming bytes from row 0's slack.
    store.set_limits(std::numeric_limits<std::size_t>::max(), total_before);

    {
        auto h = store.find(1, 0);
        h.at(0, 2) = 0x33; // 3 live -> needs to grow from 4 to 6 bytes
    }
    CHECK(store.dropped_growth_events > 0,
          "row1's tile should be unable to grow before any redistribution");
    const std::uint64_t drops_before_equalize = store.dropped_growth_events;
    {
        auto h = store.find(1, 0);
        CHECK(h.at(0, 2) == 0, "the declined growth must not have been persisted");
    }

    // Redistribute: shrinking row0 toward its fair share physically
    // truncates tile_data (see block4_row_shift's shrink branch --
    // mirrors delta_csr_shift_row exactly), which is what actually
    // "gives the space back to the rest of the matrix" -- total
    // allocation must NEVER exceed what it was (equalize_step only
    // reshuffles/frees, never grows), and dropping it re-opens headroom
    // under the fixed max_tile_bytes ceiling for ANY row's later growth
    // (not literally moving bytes into row1's own window directly).
    std::size_t cursor = 0;
    for (int i = 0; i < 4; ++i) store.equalize_step(cursor);
    CHECK(store.total_tile_alloc_bytes() <= total_before,
          "equalize_step must never grow the pool (%zu -> %zu)",
          total_before, store.total_tile_alloc_bytes());
    CHECK(store.total_tile_alloc_bytes() < total_before,
          "row0's slack (12 bytes) should have been freed, shrinking the real total (%zu -> %zu)",
          total_before, store.total_tile_alloc_bytes());
    CHECK(store.tile_byte_start[1] - store.tile_byte_start[0] < 16,
          "row0's allocation should have shrunk from its original 16 bytes, got %zu",
          store.tile_byte_start[1] - store.tile_byte_start[0]);

    // Retry the SAME growth that was declined before -- should now
    // succeed, drawing on space equalize_step freed from row 0.
    {
        auto h = store.find(1, 0);
        h.at(0, 2) = 0x33;
    }
    CHECK(store.dropped_growth_events == drops_before_equalize,
          "after redistribution the same growth that failed before should now succeed (no new drop)");
    {
        auto h = store.find(1, 0);
        CHECK(h.at(0, 0) == 0x11 && h.at(0, 1) == 0x22 && h.at(0, 2) == 0x33,
              "row1's tile should hold all 3 written values after the post-redistribution growth succeeds");
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
    test_real_compression_shrinks_footprint();
    test_equalize_step_redistributes_block4_row_headroom();
    test_move_semantics();

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
