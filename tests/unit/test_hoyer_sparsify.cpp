#include "../../sili/lib/headers/hoyer_sparsify.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <algorithm>
#include <cmath>

// Not wired into an automatic dense/sparse dispatch (see TODO.md) -- these
// tests are for the standalone Hoyer's-Sparsity-Measure operation itself.

TEST_CASE("hoyer_sparsify_row: exact k-sparse equal-magnitude vector gives k_estimate == k exactly",
         "[hoyer]") {
    for (int k : {1, 3, 5, 10}) {
        std::vector<float> x(20, 0.0f);
        for (int i = 0; i < k; ++i) x[i] = 2.5f;
        auto r = hoyer_sparsify_row<float>(x.data(), x.size());
        CHECK(r.k_estimate == k);
    }
}

TEST_CASE("hoyer_sparsify_row: fully dense uniform vector -> k_estimate == n, hoyer_score == 0",
         "[hoyer]") {
    std::vector<float> x(16, 1.0f);
    auto r = hoyer_sparsify_row<float>(x.data(), x.size());
    CHECK(r.k_estimate == 16);
    CHECK(r.hoyer_score == Catch::Approx(0.0f).margin(1e-4f));
}

TEST_CASE("hoyer_sparsify_row: single nonzero isolated correctly", "[hoyer]") {
    std::vector<float> x(50, 0.0f);
    x[7] = 3.0f;
    auto r = hoyer_sparsify_row<float>(x.data(), x.size());
    REQUIRE(r.k_estimate == 1);
    REQUIRE(r.indices.size() == 1);
    CHECK(r.indices[0] == 7);
    CHECK(r.values[0] == 3.0f);
    CHECK(r.hoyer_score == Catch::Approx(1.0f));
}

TEST_CASE("hoyer_sparsify_row: all-zero vector doesn't NaN, defines k=0",
         "[hoyer][edge_case]") {
    std::vector<float> x(10, 0.0f);
    auto r = hoyer_sparsify_row<float>(x.data(), x.size());
    CHECK(r.k_estimate == 0);
    CHECK(r.indices.empty());
    CHECK_FALSE(std::isnan(r.hoyer_score));
}

TEST_CASE("hoyer_sparsify_row: mixed-magnitude vector selects the genuinely large entries",
         "[hoyer]") {
    // 3 large values, 7 small ones -- realistic activation shape.
    std::vector<float> x = {5.0f, 4.5f, 4.0f, 0.1f, 0.05f, 0.02f, 0.01f, 0.3f, 0.15f, 0.01f};
    auto r = hoyer_sparsify_row<float>(x.data(), x.size());
    CHECK(r.k_estimate == 3);
    for (int expected : {0, 1, 2})
        CHECK(std::find(r.indices.begin(), r.indices.end(), expected) != r.indices.end());
}

TEST_CASE("hoyer_sparsify_per_batch: per-row k_estimate genuinely differs, not shared across batch",
         "[hoyer][batch]") {
    std::vector<float> batch(2 * 10, 0.0f);
    batch[0] = 1.0f; batch[1] = 1.0f;                    // row 0: 2 nonzero
    for (int i = 0; i < 8; ++i) batch[10 + i] = 1.0f;    // row 1: 8 nonzero
    auto rows = hoyer_sparsify_per_batch<float>(batch.data(), 2, 10);
    REQUIRE(rows.size() == 2);
    CHECK(rows[0].k_estimate == 2);
    CHECK(rows[1].k_estimate == 8);
}

// ── hoyer_score: the actual routing-decision quantity ──────────────────
//
// Per conversation: per-row k_estimate isn't actionable for deciding
// forward_dense vs forward_sparse, since that call happens once for the
// WHOLE batch. hoyer_score aggregates over the flattened batch to
// give the one number that decision actually needs.

TEST_CASE("hoyer_score: distinguishes a mostly-sparse batch from a mostly-dense one",
         "[hoyer][batch][routing]") {
    std::vector<float> mostly_sparse(5 * 100, 0.0f);
    for (int r = 0; r < 5; ++r)
        for (int i = 0; i < 3; ++i) mostly_sparse[r*100 + i] = 1.0f;   // 3/100 per row, every row
    auto agg_sparse = hoyer_score<float>(mostly_sparse.data(), 5, 100);

    std::vector<float> mostly_dense(5 * 100, 1.0f);
    auto agg_dense = hoyer_score<float>(mostly_dense.data(), 5, 100);

    CHECK(agg_sparse.hoyer_score > agg_dense.hoyer_score);
    CHECK(agg_dense.hoyer_score  < 0.01f);   // ~0 -> correctly signals "use forward_dense"
    CHECK(agg_sparse.hoyer_score > 0.5f);    // high -> correctly signals "use forward_sparse"
}

TEST_CASE("hoyer_score: aggregate is genuinely distinct from any single row's own value",
         "[hoyer][batch][routing]") {
    // Same construction as the per-row test: row0 very sparse, row1 medium,
    // row2 fully dense -- the aggregate must reflect the WHOLE batch, not
    // echo any one row.
    std::vector<float> batch(3 * 20, 0.0f);
    batch[0] = 1.0f; batch[1] = 1.0f;
    for (int i = 0; i < 10; ++i) batch[20 + i] = 1.0f;
    for (int i = 0; i < 20; ++i) batch[40 + i] = 1.0f;

    auto per_row = hoyer_sparsify_per_batch<float>(batch.data(), 3, 20);
    auto agg     = hoyer_score<float>(batch.data(), 3, 20);

    REQUIRE(agg.k_estimate > 0);
    REQUIRE(agg.k_estimate <= 60);
    CHECK(agg.k_estimate != per_row[0].k_estimate);
    CHECK(agg.k_estimate != per_row[2].k_estimate);
}
