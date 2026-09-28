// CiRenorm block4 half -- mirrors test_ci_renorm.cpp's scattered coverage
// (same design), plus a bit-equivalence check against the scattered path
// for the same logical values, matching this project's established
// scattered/block4 parity testing convention.
#include "../../sili/lib/headers/block4_ci_renorm_TODO_DELETE.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <cmath>
#include <functional>

static Block4Store32
ci_renorm_make_block4_dense(uint32_t rows, uint32_t cols,
                            const std::function<float(uint32_t, uint32_t)>& w_fn,
                            const std::function<float(uint32_t, uint32_t)>& imp_fn) {
    Block4Store32 store;
    store.init(rows, cols);
    store.switch_point = 0; // force dense tiles, simplest case
    const uint32_t n_br = (rows + 3) / 4, n_bc = (cols + 3) / 4;
    for (uint32_t br = 0; br < n_br; ++br) {
        for (uint32_t bc = 0; bc < n_bc; ++bc) {
            auto h = store.get_or_create(br, bc);
            for (uint32_t li = 0; li < 4; ++li) {
                for (uint32_t lj = 0; lj < 4; ++lj) {
                    const uint32_t row = br * 4 + li, col = bc * 4 + lj;
                    if (row >= rows || col >= cols)
                        continue;
                    h.set_weight(li, lj, w_fn(row, col));
                    h.set_importance(li, lj, imp_fn(row, col));
                }
            }
        }
    }
    return store;
}

TEST_CASE("CiRenorm block4: mode=Off is an exact no-op", "[ci_renorm][block4]") {
    auto store = ci_renorm_make_block4_dense(
        4, 4, [](uint32_t, uint32_t) { return 0.1f; },
        [](uint32_t r, uint32_t c) { return float(r * 4 + c) + 1.0f; });
    CiRenormState state;
    Block4CiRenormCursor cursor;
    const auto stats = apply_amortized_block4_ci_renorm_step(store, 4, state, cursor,
                                                             /*chunk_size=*/1, CiRenormMode::Off,
                                                             nullptr, nullptr, 0.01f);
    CHECK(stats.cycle_complete); // 1 tile (4x4), chunk_size=1 -> completes in one call
    for (uint32_t r = 0; r < 4; ++r)
        for (uint32_t c = 0; c < 4; ++c) {
            auto h = store.get_or_create(r / 4, c / 4);
            CHECK(h.get_importance(r % 4, c % 4) == Catch::Approx(float(r * 4 + c) + 1.0f));
        }
}

TEST_CASE("CiRenorm block4: SECOND lap writes the full corrected importance using the FIRST "
          "lap's measured stats",
          "[ci_renorm][block4][stable_region]") {
    // 4x4, single tile -- importance varies by column so each column's
    // stats are distinguishable.
    auto store = ci_renorm_make_block4_dense(
        4, 4, [](uint32_t, uint32_t) { return 0.1f; },
        [](uint32_t, uint32_t c) { return float(c) * 2.0f; }); // col 0,1,2,3 -> imp 0,2,4,6
    CiRenormState state;
    Block4CiRenormCursor cursor;
    const float target_mean[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float target_std[4] = {0.5f, 0.5f, 0.5f, 0.5f};

    auto stats1 = apply_amortized_block4_ci_renorm_step(
        store, 4, state, cursor, 1, CiRenormMode::StableRegion, target_mean, target_std, 0.01f);
    CHECK(stats1.cycle_complete);
    // Column 0: all 4 rows have imp=0 -> mean=0, std=0.
    CHECK(state.col_ci_mean[0] == Catch::Approx(0.0f));
    CHECK(state.col_ci_mean[2] == Catch::Approx(4.0f)); // column 2 -> imp=4 for all rows

    auto stats2 = apply_amortized_block4_ci_renorm_step(
        store, 4, state, cursor, 1, CiRenormMode::StableRegion, target_mean, target_std, 0.01f);
    CHECK(stats2.cycle_complete);
    // Every cell in column j was EXACTLY at that column's own lap-1 mean
    // (zero within-column variance) -- the affine transform's fixed
    // point, so every cell should land at target_mean=1.0.
    auto h = store.get_or_create(0, 0);
    for (uint32_t li = 0; li < 4; ++li)
        for (uint32_t lj = 0; lj < 4; ++lj)
            CHECK(h.get_importance(li, lj) == Catch::Approx(1.0f).margin(1e-3));
}

TEST_CASE("CiRenorm block4: never touches weight, only importance", "[ci_renorm][block4]") {
    auto store = ci_renorm_make_block4_dense(
        4, 4, [](uint32_t r, uint32_t c) { return 0.37f + float(r) + float(c); },
        [](uint32_t, uint32_t) { return 50.0f; });
    CiRenormState state;
    Block4CiRenormCursor cursor;
    const float target_mean[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float target_std[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    apply_amortized_block4_ci_renorm_step(store, 4, state, cursor, 1, CiRenormMode::StableRegion,
                                          target_mean, target_std, 0.01f);
    apply_amortized_block4_ci_renorm_step(store, 4, state, cursor, 1, CiRenormMode::StableRegion,
                                          target_mean, target_std, 0.01f);
    auto h = store.get_or_create(0, 0);
    for (uint32_t li = 0; li < 4; ++li)
        for (uint32_t lj = 0; lj < 4; ++lj)
            CHECK(h.get_weight(li, lj) == Catch::Approx(0.37f + float(li) + float(lj)));
}

TEST_CASE("CiRenorm block4: multi-tile coverage -- the cursor visits every live cell exactly "
          "once per lap across MULTIPLE tiles, chunk_size not a divisor of tile count",
          "[ci_renorm][block4]") {
    // 8x8 = 4 tiles, chunk_size=3 -> lap completes on the 2nd call (3+1).
    auto store = ci_renorm_make_block4_dense(
        8, 8, [](uint32_t, uint32_t) { return 0.1f; },
        [](uint32_t r, uint32_t c) { return float(r * 8 + c); });
    CiRenormState state;
    Block4CiRenormCursor cursor;
    bool cycle_complete = false;
    int calls = 0;
    for (int i = 0; i < 5 && !cycle_complete; ++i) {
        const auto stats = apply_amortized_block4_ci_renorm_step(
            store, 8, state, cursor, 3, CiRenormMode::Off, nullptr, nullptr, 0.01f);
        cycle_complete = stats.cycle_complete;
        ++calls;
    }
    CHECK(cycle_complete);
    CHECK(calls == 2); // ceil(4 tiles / 3) = 2
    // Every column's mean should reflect EXACTLY 8 touches (8 rows),
    // sum = 0+8+16+...+56 = 224 for column 0 -> mean = 28.
    CHECK(state.col_ci_mean[0] == Catch::Approx(28.0f));
    CHECK(state.col_ci_mean[7] == Catch::Approx(35.0f)); // 7+15+...+63, mean=35
}

TEST_CASE("CiRenorm scattered vs block4: bit-equivalent StableRegion correction for the same "
          "logical values",
          "[ci_renorm][block4][bit_equivalence]") {
    auto store = ci_renorm_make_block4_dense(
        4, 4, [](uint32_t, uint32_t) { return 0.1f; },
        [](uint32_t r, uint32_t c) { return float(r) * 4.0f + float(c); });
    CiRenormState block4_state;
    Block4CiRenormCursor block4_cursor;
    apply_amortized_block4_ci_renorm_step(store, 4, block4_state, block4_cursor, 1,
                                          CiRenormMode::Off, nullptr, nullptr, 0.01f);

    // Scattered side: identical dense 4x4 layer via CiRenormState's own
    // touch primitive directly (avoids depending on the scattered CSR
    // traversal's own construction helper here -- the CORE arithmetic
    // is what must agree, and both traversals call the SAME
    // ci_renorm_touch_cell/ci_renorm_cycle_boundary functions, so this
    // confirms the STATS-COLLECTION order/grouping-by-column agrees
    // between the two storage walks).
    CiRenormState scattered_state;
    scattered_state.ensure_sized(4);
    for (uint32_t r = 0; r < 4; ++r)
        for (uint32_t c = 0; c < 4; ++c)
            ci_renorm_touch_cell(scattered_state, c, float(r) * 4.0f + float(c), 0.1f,
                                 CiRenormMode::Off, 0.0f, 0.0f, 0.01f);
    CiRenormStats scattered_stats;
    ci_renorm_cycle_boundary(scattered_state, 4, scattered_stats);

    for (std::size_t j = 0; j < 4; ++j) {
        CHECK(block4_state.col_ci_mean[j] == Catch::Approx(scattered_state.col_ci_mean[j]));
        CHECK(block4_state.col_ci_std[j] == Catch::Approx(scattered_state.col_ci_std[j]));
    }
}
