// apply_amortized_block4_decay_stats (sili/lib/headers/block4_decay_TODO_DELETE.hpp)
// -- the block4-storage counterpart to apply_amortized_decay_stats
// (test_amortized_decay_stats.cpp). Direct instruction: "we're going to
// need block4 and dense layers to work with this" (importance/weight
// amortized decay must cover block4 packed-tile storage, not just
// scattered CSR), and: "add tests to make sure things match in any edge
// cases the code indicates" -- this file is that requirement for block4.
//
// Verifies: (1) weight decay leaves importance untouched and vice versa;
// (2) a tile forced into SPARSE-packed format decays correctly and its
// live-cell count/format is stable (doesn't spuriously flip to dense) if
// no cell crosses zero; (3) a tile in DENSE format decays correctly; (4)
// the cursor correctly skips an empty row between populated rows and
// wraps back to row 0, reporting cycle_complete exactly once per full
// pass; (5) an empty store (no tiles) reports cycle_complete=true, n=0,
// no crash; (6) BIT-EQUIVALENCE -- decaying a set of values through the
// block4 path produces IDENTICAL float32 results (not just "close") to
// decaying the same logical values through the scattered CSR path
// (apply_amortized_decay_stats on DeltaCSRBiValues<float>), since
// decay is a per-cell `v * decay_factor` with no cross-cell reduction --
// the two storage formats should never disagree about the arithmetic
// itself, only about where the bytes live.
#include "../../sili/lib/headers/block4_decay_TODO_DELETE.hpp"
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

struct Block4DecayState {
    Block4DecayCursor cursor;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    std::size_t n = 0;
};

static void test_empty_store() {
    Block4Store32 store;
    store.init(16, 16);
    Block4DecayState st;
    auto r = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq, st.max_abs,
                                                st.n, 4, 0.5f, /*decay_importance=*/false);
    CHECK(r.cycle_complete, "empty block4 store should report cycle_complete=true immediately");
    CHECK(r.n == 0, "empty block4 store should report n=0");
}

static void test_weight_decay_isolation_dense_tile() {
    Block4Store32 store;
    store.init(16, 16);
    store.switch_point = 0; // force dense format, no sparse-packing at all
    {
        auto h = store.get_or_create(0, 0);
        h.set_weight(0, 0, 4.0f);
        h.set_importance(0, 0, 9.0f);
        h.set_weight(1, 1, -2.0f);
        h.set_importance(1, 1, 7.0f);
    }
    CHECK(store.is_sparse(0, 0) == false, "setup: switch_point=0 should keep tile dense");

    Block4DecayState st;
    auto r = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq, st.max_abs,
                                                st.n, 8, 0.5f, /*decay_importance=*/false);
    CHECK(r.cycle_complete, "single tile, chunk_size=8 should complete the cycle");
    auto h = store.find(0, 0);
    CHECK(h.get_weight(0, 0) == 2.0f, "w(0,0) should decay to 2.0, got %f", h.get_weight(0, 0));
    CHECK(h.get_weight(1, 1) == -1.0f, "w(1,1) should decay to -1.0, got %f", h.get_weight(1, 1));
    CHECK(h.get_importance(0, 0) == 9.0f, "importance(0,0) must be untouched by weight decay");
    CHECK(h.get_importance(1, 1) == 7.0f, "importance(1,1) must be untouched by weight decay");
    CHECK(r.n == 2, "n should be 2 (two live cells), got %zu", r.n);
}

static void test_importance_decay_isolation_dense_tile() {
    Block4Store32 store;
    store.init(16, 16);
    store.switch_point = 0;
    {
        auto h = store.get_or_create(0, 0);
        h.set_weight(0, 0, 4.0f);
        h.set_importance(0, 0, 9.0f);
    }
    Block4DecayState st;
    auto r = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq, st.max_abs,
                                                st.n, 8, 0.5f, /*decay_importance=*/true);
    CHECK(r.cycle_complete, "single tile should complete the cycle");
    auto h = store.find(0, 0);
    CHECK(h.get_weight(0, 0) == 4.0f, "weight must be untouched by importance decay, got %f",
          h.get_weight(0, 0));
    CHECK(h.get_importance(0, 0) == 4.5f, "importance(0,0) should decay to 4.5, got %f",
          h.get_importance(0, 0));
}

static void test_sparse_tile_decay_and_format_stability() {
    Block4Store32 store;
    store.init(16, 16);
    store.switch_point = 14; // default BLOCK4_SPARSE_MAX_COUNT32
    {
        auto h = store.get_or_create(0, 0);
        h.set_weight(0, 0, 8.0f);
        h.set_importance(0, 0, 1.0f);
        h.set_weight(0, 1, -8.0f);
        h.set_importance(0, 1, 1.0f);
        h.set_weight(1, 0, 3.0f);
        h.set_importance(1, 0, 1.0f);
    }
    store.maybe_compress(0, 0);
    CHECK(store.is_sparse(0, 0) == true,
          "setup: 3 active <= switch_point=14 should compress to sparse");

    Block4DecayState st;
    // Two touches at decay=0.5, chunk_size=1 each -- also exercises that a
    // single-tile chunk still reports cycle_complete on the very first call
    // (n_tiles==1) without needing a second call to notice the wrap.
    auto r1 = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq,
                                                 st.max_abs, st.n, 1, 0.5f, false);
    CHECK(r1.cycle_complete, "single-tile store, chunk_size=1 should still complete the cycle");
    CHECK(
        store.is_sparse(0, 0) == true,
        "tile must still be reported sparse after a decay pass that touches no zero-crossing cell");
    auto h1 = store.find(0, 0);
    CHECK(h1.get_weight(0, 0) == 4.0f, "sparse tile w(0,0) should decay to 4.0, got %f",
          h1.get_weight(0, 0));
    CHECK(h1.get_weight(0, 1) == -4.0f, "sparse tile w(0,1) should decay to -4.0, got %f",
          h1.get_weight(0, 1));
    CHECK(h1.get_weight(1, 0) == 1.5f, "sparse tile w(1,0) should decay to 1.5, got %f",
          h1.get_weight(1, 0));
    CHECK(h1.get_importance(0, 0) == 1.0f, "importance must be untouched by weight decay");

    auto r2 = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq,
                                                 st.max_abs, st.n, 1, 0.5f, false);
    CHECK(r2.cycle_complete, "second pass over the same single-tile store should also complete");
    auto h2 = store.find(0, 0);
    CHECK(h2.get_weight(0, 0) == 2.0f, "sparse tile w(0,0) after 2 decays should be 2.0, got %f",
          h2.get_weight(0, 0));
    CHECK(store.is_sparse(0, 0) == true, "tile should remain sparse-packed across repeated decay");
}

static void test_cursor_skips_empty_row_and_wraps() {
    // Rows 0 and 2 populated, row 1 left empty -- the cursor must skip row
    // 1 entirely (not stall or crash), and must wrap back to row 0 (not
    // row 1) once row 2's tile is consumed.
    Block4Store32 store;
    store.init(16, 16); // 4 block-rows for n_in=16, BLOCK4_TILE=4
    store.switch_point = 0;
    {
        auto h = store.get_or_create(0, 0);
        h.set_weight(0, 0, 2.0f);
        h.set_importance(0, 0, 1.0f);
    }
    {
        auto h = store.get_or_create(2, 1);
        h.set_weight(0, 0, 4.0f);
        h.set_importance(0, 0, 1.0f);
    }
    CHECK(store.n_tiles() == 2, "setup: expected 2 tiles total, got %zu", store.n_tiles());

    Block4DecayState st;
    // chunk_size=1: first call should touch row 0's tile only.
    auto r1 = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq,
                                                 st.max_abs, st.n, 1, 0.5f, false);
    CHECK(!r1.cycle_complete, "1/2 tiles touched should not complete the cycle");
    CHECK(store.find(0, 0).get_weight(0, 0) == 1.0f, "row0 tile should have decayed to 1.0");
    CHECK(store.find(2, 1).get_weight(0, 0) == 4.0f,
          "row2 tile should be untouched by the first call");

    // Second call: cursor must skip empty row 1 and land on row 2's tile.
    auto r2 = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq,
                                                 st.max_abs, st.n, 1, 0.5f, false);
    CHECK(r2.cycle_complete, "2/2 tiles touched should complete the cycle");
    CHECK(store.find(2, 1).get_weight(0, 0) == 2.0f,
          "row2 tile should have decayed to 2.0 after cursor skipped empty row1, got %f",
          store.find(2, 1).get_weight(0, 0));
    CHECK(r2.n == 2, "full-cycle n should be 2 (one live cell per tile), got %zu", r2.n);

    // Third call: must have wrapped to row 0, not row 1 (which has no
    // tile and would otherwise stall the cursor's row-skip loop).
    auto r3 = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq,
                                                 st.max_abs, st.n, 1, 0.5f, false);
    CHECK(!r3.cycle_complete, "new cycle, 1/2 touched, should not complete yet");
    CHECK(store.find(0, 0).get_weight(0, 0) == 0.5f,
          "row0 tile should have decayed again after wrap, to 0.5, got %f",
          store.find(0, 0).get_weight(0, 0));
}

static void test_not_live_cells_are_skipped() {
    // A cell that is exactly 0 in BOTH weight and importance is not live
    // and must not be touched/counted -- verifies the block4 path applies
    // the same "live iff weight or importance nonzero" definition the
    // rest of the codebase uses (block4_count_live32), not just "nonzero
    // weight".
    Block4Store32 store;
    store.init(16, 16);
    store.switch_point = 0;
    {
        auto h = store.get_or_create(0, 0);
        h.set_weight(0, 0, 0.0f);
        h.set_importance(0, 0, 5.0f); // live via importance only
        h.set_weight(1, 1, 3.0f);
        h.set_importance(1, 1, 0.0f); // live via weight only
        // (2,2) left fully zero/untouched -- not live at all.
    }
    Block4DecayState st;
    auto r = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq, st.max_abs,
                                                st.n, 8, 0.5f, /*decay_importance=*/true);
    CHECK(r.cycle_complete, "single tile should complete");
    CHECK(r.n == 2, "only the 2 live cells should be touched/counted, got %zu", r.n);
    auto h = store.find(0, 0);
    CHECK(h.get_importance(0, 0) == 2.5f,
          "live-via-importance cell should have decayed to 2.5, got %f", h.get_importance(0, 0));
    CHECK(h.get_importance(1, 1) == 0.0f,
          "live-via-weight-only cell has importance=0, decaying 0 stays 0");
}

static void test_bit_equivalence_vs_scattered_decay() {
    // Same logical (row, col) -> (weight, importance) values, decayed
    // through block4 vs through the scattered CSR path
    // (apply_amortized_decay_stats on DeltaCSRBiValues<float>, tested in
    // test_amortized_decay_stats.cpp) -- decay is a per-cell
    // `v * decay_factor` multiply with no cross-cell reduction, so the
    // two storage formats must agree BIT-FOR-BIT, not just approximately.
    const float decay = 0.7f;
    const float values_in[6] = {4.0f, -2.5f, 0.125f, 17.0f, -0.001f, 1e6f};

    // -- scattered path --
    using VT = DeltaCSRBiValues<float>;
    VT scattered;
    scattered.weights.assign(values_in, values_in + 6);
    scattered.importance.assign(6, 1.0f);
    std::size_t s_cursor = 0;
    double s_sum_abs = 0, s_sum_sq = 0, s_max_abs = 0;
    std::size_t s_n = 0;
    apply_amortized_decay_stats<VT, float>(scattered, s_cursor, s_sum_abs, s_sum_sq, s_max_abs, s_n,
                                           6, decay);

    // -- block4 path: same 6 values placed one per tile (br=0..5, bc=0),
    // dense format so no packing quirks are in play -- isolating exactly
    // the arithmetic, not the storage format, as the thing under test.
    Block4Store32 store;
    store.init(6 * 4, 4); // 6 block-rows
    store.switch_point = 0;
    for (int i = 0; i < 6; ++i) {
        auto h = store.get_or_create(uint32_t(i), 0);
        h.set_weight(0, 0, values_in[i]);
        h.set_importance(0, 0, 1.0f);
    }
    Block4DecayState st;
    auto r = apply_amortized_block4_decay_stats(store, st.cursor, st.sum_abs, st.sum_sq, st.max_abs,
                                                st.n, 6, decay, false);
    CHECK(r.cycle_complete, "6 tiles, chunk_size=6 should complete in one call");

    for (int i = 0; i < 6; ++i) {
        auto h = store.find(uint32_t(i), 0);
        const float block4_val = h.get_weight(0, 0);
        const float scattered_val = scattered.weights[i];
        CHECK(block4_val == scattered_val,
              "bit-equivalence failed at i=%d: block4=%.9g scattered=%.9g (bits %08x vs %08x)", i,
              double(block4_val), double(scattered_val),
              *reinterpret_cast<const uint32_t*>(&block4_val),
              *reinterpret_cast<const uint32_t*>(&scattered_val));
    }
}

int main() {
    test_empty_store();
    test_weight_decay_isolation_dense_tile();
    test_importance_decay_isolation_dense_tile();
    test_sparse_tile_decay_and_format_stability();
    test_cursor_skips_empty_row_and_wraps();
    test_not_live_cells_are_skipped();
    test_bit_equivalence_vs_scattered_decay();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
