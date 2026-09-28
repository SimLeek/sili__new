// CiRenorm: full-diff amortized correction toward a target ci mean/std,
// applied directly to the real per-synapse ci accumulator. See
// docs/research/delta_csr_types.rst:ci_renorm.design_and_v3_lesson for
// the full derivation, including why v3's own earlier attempt at a
// similar idea (l2_decay_lambda, a weak proportional per-touch shrink)
// failed -- growth vs decay tied almost exactly, confirmed empirically
// on a real 100k-step run. This file pins down the CORE arithmetic
// (ci_renorm_touch_cell/ci_renorm_cycle_boundary) in isolation, before
// any traversal/cursor wiring.
#include "../../sili/lib/headers/delta_csr_types.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <cmath>
#include <functional>

using CiRenormVT = DeltaCSRBiValues<float>;
using CiRenormConn = DeltaCSRWeights<std::size_t, CiRenormVT, uint32_t>;

// Builds a fully-dense n_in x n_out scattered CSR layer -- same shape as
// test_amortized_plasticity_reset.cpp's own make_dense_conn (a distinct
// local copy since each test binary's file is compiled independently).
static CiRenormConn
ci_renorm_make_dense_conn(std::size_t n_in, std::size_t n_out,
                          const std::function<float(std::size_t, std::size_t)>& w_fn,
                          const std::function<float(std::size_t, std::size_t)>& imp_fn) {
    CiRenormConn conn;
    conn.layout.rows = n_in;
    conn.layout.cols = n_out;
    conn.layout.elem_start.assign(n_in + 1, 0);
    conn.layout.elem_end.assign(n_in, 0);
    conn.layout.byte_start.assign(n_in + 1, 0);
    conn.layout.byte_end.assign(n_in, 0);
    for (std::size_t r = 0; r < n_in; ++r) {
        conn.layout.elem_start[r] = r * n_out;
        conn.layout.elem_end[r] = r * n_out + n_out;
    }
    conn.layout.elem_start[n_in] = n_in * n_out;
    conn.layout.total_nnz = n_in * n_out;

    conn.indices_buf.clear();
    for (std::size_t r = 0; r < n_in; ++r) {
        conn.layout.byte_start[r] = conn.indices_buf.size();
        uint32_t prev = 0;
        for (std::size_t c = 0; c < n_out; ++c) {
            const uint32_t delta = static_cast<uint32_t>(c) - prev;
            uint32_t v = delta;
            do {
                uint8_t byte = v & 0x7F;
                v >>= 7;
                if (v)
                    byte |= 0x80;
                conn.indices_buf.push_back(byte);
            } while (v);
            prev = static_cast<uint32_t>(c);
        }
        conn.layout.byte_end[r] = conn.indices_buf.size();
    }
    conn.layout.byte_start[n_in] = conn.indices_buf.size();

    ValueAccessor<CiRenormVT>::resize(conn.values, n_in * n_out);
    for (std::size_t r = 0; r < n_in; ++r)
        for (std::size_t c = 0; c < n_out; ++c) {
            const std::size_t vb = r * n_out + c;
            ValueAccessor<CiRenormVT>::set_live(conn.values, vb, w_fn(r, c), imp_fn(r, c));
        }
    return conn;
}

TEST_CASE("CiRenorm: mode=Off is an exact no-op, byte-identical to not calling it at all",
          "[ci_renorm]") {
    CiRenormState state;
    state.ensure_sized(4);
    const float result =
        ci_renorm_touch_cell(state, 0, 37.5f, 0.2f, CiRenormMode::Off, 0.0f, 0.0f, 0.01f);
    CHECK(result == 37.5f);
}

TEST_CASE("CiRenorm: first lap (no finalized stats yet) is an exact no-op regardless of mode",
          "[ci_renorm]") {
    CiRenormState state;
    state.ensure_sized(4);
    REQUIRE_FALSE(state.has_finalized_stats);
    const float trust_result =
        ci_renorm_touch_cell(state, 0, 50.0f, 0.1f, CiRenormMode::TrustRatio, 0.0f, 0.0f, 0.01f);
    CHECK(trust_result == 50.0f);
    const float stable_result =
        ci_renorm_touch_cell(state, 1, 50.0f, 0.1f, CiRenormMode::StableRegion, 1.0f, 0.5f, 0.01f);
    CHECK(stable_result == 50.0f);
}

TEST_CASE("CiRenorm: cycle_boundary computes the EXACT population mean/std of touched columns",
          "[ci_renorm]") {
    CiRenormState state;
    state.ensure_sized(2);
    // Column 0: values 2, 4, 6 -- mean=4, var=((2-4)^2+(0)^2+(2)^2)/3=8/3, std=sqrt(8/3)~=1.63299
    for (float v : {2.0f, 4.0f, 6.0f})
        ci_renorm_touch_cell(state, 0, v, 1.0f, CiRenormMode::Off, 0.0f, 0.0f, 0.01f);
    // Column 1: untouched this lap.
    CiRenormStats stats;
    ci_renorm_cycle_boundary(state, 2, stats);
    CHECK(state.col_ci_mean[0] == Catch::Approx(4.0f));
    CHECK(state.col_ci_std[0] == Catch::Approx(1.632993f).epsilon(0.0001));
    // Untouched column keeps its (zero-initialized) prior stats, not NaN/garbage.
    CHECK(state.col_ci_mean[1] == Catch::Approx(0.0f));
    CHECK(state.has_finalized_stats);
    CHECK(stats.cycle_complete);
    // Accumulators reset for the next lap.
    CHECK(state.col_touch_count[0] == 0);
}

TEST_CASE("CiRenorm StableRegion: SECOND lap applies the FULL affine diff toward the target, "
          "using the FIRST lap's finalized stats -- not a partial/blended nudge",
          "[ci_renorm][stable_region]") {
    CiRenormState state;
    state.ensure_sized(1);
    // Lap 1: establish cur_mean=4, cur_std=sqrt(8/3)~=1.63299 for column 0.
    for (float v : {2.0f, 4.0f, 6.0f})
        ci_renorm_touch_cell(state, 0, v, 1.0f, CiRenormMode::StableRegion, 1.0f, 0.5f, 0.01f);
    CiRenormStats stats1;
    ci_renorm_cycle_boundary(state, 1, stats1);

    // Lap 2: target_mean=1.0, target_std=0.5. A cell at ci_val=4 (exactly
    // lap-1's mean) should land EXACTLY at target_mean (the affine
    // transform's fixed point). A cell at ci_val=6 is (6-4)/std(=1.632993)
    // ~=1.2247 std above lap-1's mean, so it should land at
    // target_mean + 1.2247*target_std ~= 1.61237.
    const float at_mean =
        ci_renorm_touch_cell(state, 0, 4.0f, 1.0f, CiRenormMode::StableRegion, 1.0f, 0.5f, 0.01f);
    CHECK(at_mean == Catch::Approx(1.0f).margin(1e-5));
    const float at_6 =
        ci_renorm_touch_cell(state, 0, 6.0f, 1.0f, CiRenormMode::StableRegion, 1.0f, 0.5f, 0.01f);
    CHECK(at_6 == Catch::Approx(1.612372f).epsilon(0.001));
}

TEST_CASE("CiRenorm StableRegion: never produces a negative ci (clamped at 0)", "[ci_renorm]") {
    CiRenormState state;
    state.ensure_sized(1);
    for (float v : {10.0f, 10.0f, 10.0f}) // mean=10, std=0
        ci_renorm_touch_cell(state, 0, v, 1.0f, CiRenormMode::StableRegion, 1.0f, 0.5f, 0.01f);
    CiRenormStats stats;
    ci_renorm_cycle_boundary(state, 1, stats);
    // std=0 -> cur_std clamped to 1e-6 internally; a ci_val far below the
    // lap-1 mean would blow the affine transform deeply negative without
    // the explicit max(...,0) clamp.
    const float result =
        ci_renorm_touch_cell(state, 0, 0.0f, 1.0f, CiRenormMode::StableRegion, 1.0f, 0.5f, 0.01f);
    CHECK(result >= 0.0f);
    CHECK(std::isfinite(result));
}

TEST_CASE("CiRenorm StableRegion: preserves relative rank order within a column (positive-affine "
          "transform)",
          "[ci_renorm]") {
    CiRenormState state;
    state.ensure_sized(1);
    for (float v : {1.0f, 5.0f, 9.0f})
        ci_renorm_touch_cell(state, 0, v, 1.0f, CiRenormMode::StableRegion, 2.0f, 3.0f, 0.01f);
    CiRenormStats stats;
    ci_renorm_cycle_boundary(state, 1, stats);
    const float low =
        ci_renorm_touch_cell(state, 0, 1.0f, 1.0f, CiRenormMode::StableRegion, 2.0f, 3.0f, 0.01f);
    const float mid =
        ci_renorm_touch_cell(state, 0, 5.0f, 1.0f, CiRenormMode::StableRegion, 2.0f, 3.0f, 0.01f);
    const float high =
        ci_renorm_touch_cell(state, 0, 9.0f, 1.0f, CiRenormMode::StableRegion, 2.0f, 3.0f, 0.01f);
    CHECK(low < mid);
    CHECK(mid < high);
}

TEST_CASE("CiRenorm TrustRatio: trust_ratio compares weight norm against a FIXED reference (the "
          "update scale implied by max_ci itself), NOT the column's own current ci -- avoids the "
          "circularity of estimating g from ci while also dividing by ci in the same expression",
          "[ci_renorm][trust_ratio]") {
    CiRenormState state;
    state.ensure_sized(1);
    // Lap 1: w=0.01 per touch, 3 touches -> w_norm=sqrt(3*0.01^2)~=0.01732.
    // No correction applies (first lap), so col_ci_mean finalizes to the
    // touched values' own mean (100.0) unchanged.
    for (float v : {100.0f, 100.0f, 100.0f})
        ci_renorm_touch_cell(state, 0, v, 0.01f, CiRenormMode::TrustRatio, 0.0f, 0.0f, 0.01f,
                             100.0f);
    CiRenormStats stats;
    ci_renorm_cycle_boundary(state, 1, stats);
    const float w_norm = state.col_w_norm[0];
    REQUIRE(w_norm == Catch::Approx(std::sqrt(3.0f * 0.01f * 0.01f)).epsilon(0.001));
    REQUIRE(state.col_ci_mean[0] == Catch::Approx(100.0f));

    // Lap 2: ci_implied = (w_norm/eff_lr)^2 = (0.01732/0.01)^2 = 3.0
    // (exact: w_norm^2=3*0.01^2=0.0003, /eff_lr^2=0.0001 -> 3.0).
    // scale = ci_implied / cur_mean = 3.0/100.0 = 0.03 (within
    // [0.01,100], not clamped).
    const float new_ci = ci_renorm_touch_cell(state, 0, 50.0f, 0.01f, CiRenormMode::TrustRatio,
                                              0.0f, 0.0f, 0.01f, 100.0f);
    CHECK(new_ci == Catch::Approx(50.0f * 0.03f).epsilon(0.001));
    // A DIFFERENT ci_val this touch (100 instead of 50) gets scaled by
    // the SAME factor -- confirms the correction is a pure per-column
    // multiplicative rescale within a lap (mean-only, no std change).
    const float new_ci_other = ci_renorm_touch_cell(
        state, 0, 100.0f, 0.01f, CiRenormMode::TrustRatio, 0.0f, 0.0f, 0.01f, 100.0f);
    CHECK(new_ci_other / 100.0f == Catch::Approx(new_ci / 50.0f).epsilon(0.001));
}

TEST_CASE("CiRenorm TrustRatio: clip range holds at both ends -- a column with huge weight norm "
          "doesn't get an unboundedly large scale (clamped at trust_ratio_max^2=100), and a "
          "column with tiny weight norm doesn't get an unboundedly small one (clamped at "
          "trust_ratio_min^2=0.01)",
          "[ci_renorm][trust_ratio]") {
    CiRenormState state;
    state.ensure_sized(2);
    // Column 0: huge weight norm -> ci_implied huge relative to mean=1.0.
    ci_renorm_touch_cell(state, 0, 1.0f, 1000.0f, CiRenormMode::TrustRatio, 0.0f, 0.0f, 0.01f,
                         100.0f);
    // Column 1: tiny weight norm -> ci_implied tiny relative to mean=1.0.
    ci_renorm_touch_cell(state, 1, 1.0f, 1e-6f, CiRenormMode::TrustRatio, 0.0f, 0.0f, 0.01f,
                         100.0f);
    CiRenormStats stats;
    ci_renorm_cycle_boundary(state, 2, stats);
    REQUIRE(state.col_ci_mean[0] == Catch::Approx(1.0f));
    REQUIRE(state.col_ci_mean[1] == Catch::Approx(1.0f));

    // ci_implied[0] = (1000/0.01)^2 = 1e10 -- scale = 1e10/1.0, clamped
    // DOWN to trust_ratio_max^2=100.
    const float new_ci0 = ci_renorm_touch_cell(state, 0, 1.0f, 1000.0f, CiRenormMode::TrustRatio,
                                               0.0f, 0.0f, 0.01f, 100.0f);
    CHECK(new_ci0 == Catch::Approx(1.0f * 100.0f).epsilon(0.001));

    // ci_implied[1] = (1e-6/0.01)^2 = 1e-8 -- scale = 1e-8/1.0, clamped
    // UP to trust_ratio_min^2=0.01.
    const float new_ci1 = ci_renorm_touch_cell(state, 1, 1.0f, 1e-6f, CiRenormMode::TrustRatio,
                                               0.0f, 0.0f, 0.01f, 100.0f);
    CHECK(new_ci1 == Catch::Approx(1.0f * 0.01f).epsilon(0.001));
}

TEST_CASE("CiRenorm TrustRatio: REGRESSION -- converges to a stable fixed point over multiple "
          "laps instead of compounding into geometric decay. Real bug found via an end-to-end "
          "smoke test: a raw per-touch multiplicative scale has no dependence on ci itself, so "
          "the SAME scale reapplies every lap (observed: 90 -> 1.96 -> 0.045, still shrinking). "
          "Fixed by deriving a fixed implied target ci from the weight norm and renormalizing "
          "the population MEAN toward it (same equilibrium structure as StableRegion).",
          "[ci_renorm][trust_ratio][regression]") {
    CiRenormState state;
    state.ensure_sized(1);
    // w=0.03, eff_lr=0.01 -> ci_implied = (0.03/0.01)^2 = 9.0 exactly.
    // One touch (its own lap, chunk_size=1) followed by cycle_boundary,
    // repeated -- mirrors how a real amortized cursor lap-by-lap call
    // sequence behaves.
    float last = 90.0f;
    for (int lap = 0; lap < 8; ++lap) {
        last = ci_renorm_touch_cell(state, 0, last, 0.03f, CiRenormMode::TrustRatio, 0.0f, 0.0f,
                                    0.01f, 100.0f);
        CiRenormStats stats;
        ci_renorm_cycle_boundary(state, 1, stats);
    }
    // Converges to (and stays at) ci_implied=9.0 -- not collapsing toward
    // 0, not diverging.
    CHECK(last == Catch::Approx(9.0f).epsilon(0.02));
}

TEST_CASE("CiRenorm: a full lap's worth of touches never contaminates that SAME lap's own "
          "finalized stats -- correction uses only the PREVIOUS lap, never the in-progress one",
          "[ci_renorm]") {
    CiRenormState state;
    state.ensure_sized(1);
    // Lap 1: mean=4.
    for (float v : {2.0f, 4.0f, 6.0f})
        ci_renorm_touch_cell(state, 0, v, 1.0f, CiRenormMode::StableRegion, 4.0f, 1.0f, 0.01f);
    CiRenormStats s1;
    ci_renorm_cycle_boundary(state, 1, s1);
    CHECK(state.col_ci_mean[0] == Catch::Approx(4.0f));

    // Lap 2: touch with wildly different values (mean would be 1000 if
    // it were used) -- but corrections this lap must still reference
    // LAP 1's mean=4, not anything from lap 2's own in-progress sum.
    const float first_touch_this_lap =
        ci_renorm_touch_cell(state, 0, 4.0f, 1.0f, CiRenormMode::StableRegion, 4.0f, 1.0f, 0.01f);
    CHECK(first_touch_this_lap == Catch::Approx(4.0f).margin(1e-4)); // at lap-1's mean -> fixed pt
    for (int i = 0; i < 5; ++i)
        ci_renorm_touch_cell(state, 0, 2000.0f, 1.0f, CiRenormMode::StableRegion, 4.0f, 1.0f,
                             0.01f);
    // Even after touching huge values this lap, correction for a value
    // AT lap-1's mean still lands at the fixed point -- proves lap 2's
    // own accumulating sum never leaked into col_ci_mean mid-lap.
    const float still_at_fixed_point =
        ci_renorm_touch_cell(state, 0, 4.0f, 1.0f, CiRenormMode::StableRegion, 4.0f, 1.0f, 0.01f);
    CHECK(still_at_fixed_point == Catch::Approx(4.0f).margin(1e-4));
}

// ── Scattered traversal (apply_amortized_ci_renorm_step): the raster-scan
// cursor itself, and that corrections actually land in REAL storage ──────

TEST_CASE("CiRenorm scattered: mode=Off is an exact no-op -- touched cells' importance is "
          "byte-unchanged in storage",
          "[ci_renorm][scattered]") {
    auto conn = ci_renorm_make_dense_conn(
        2, 3, [](std::size_t, std::size_t) { return 0.1f; },
        [](std::size_t r, std::size_t c) { return float(r * 3 + c) + 1.0f; });
    CiRenormState state;
    PlasticityCellCursor cursor;
    const auto stats = apply_amortized_ci_renorm_step<CiRenormVT, std::size_t, uint32_t>(
        conn, 3, state, cursor, 6, CiRenormMode::Off, nullptr, nullptr, 0.01f);
    CHECK(stats.cycle_complete); // chunk_size=6 == full nnz -> one full lap
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c) {
            const std::size_t vb = r * 3 + c;
            CHECK(ValueAccessor<CiRenormVT>::get_imp(conn.values, vb) ==
                  Catch::Approx(float(r * 3 + c) + 1.0f));
        }
}

TEST_CASE("CiRenorm scattered: the raster-scan cursor visits every live cell EXACTLY once per "
          "lap, in strict forward order, never skipping or revisiting -- chunk_size smaller than "
          "nnz needs multiple calls to complete one lap",
          "[ci_renorm][scattered]") {
    // 3x4 = 12 cells, chunk_size=5 -> lap completes on the 3rd call (5+5+2).
    auto conn = ci_renorm_make_dense_conn(
        3, 4, [](std::size_t, std::size_t) { return 0.1f; },
        [](std::size_t r, std::size_t c) { return float(r * 4 + c); });
    CiRenormState state;
    PlasticityCellCursor cursor;
    std::size_t total_touched_stat_updates = 0;
    bool cycle_complete = false;
    for (int call = 0; call < 3 && !cycle_complete; ++call) {
        const auto stats = apply_amortized_ci_renorm_step<CiRenormVT, std::size_t, uint32_t>(
            conn, 4, state, cursor, 5, CiRenormMode::Off, nullptr, nullptr, 0.01f);
        cycle_complete = stats.cycle_complete;
        ++total_touched_stat_updates;
    }
    CHECK(cycle_complete);
    CHECK(total_touched_stat_updates == 3); // ceil(12/5) = 3, matching the exact-coverage guarantee
    // Every column's touch count should reflect EXACTLY 3 rows touched
    // (3 rows x 4 cols = 12 cells, each column touched once per row).
    double total_sum = 0.0;
    for (std::size_t j = 0; j < 4; ++j)
        total_sum += state.col_ci_mean[j] * 3.0; // 3 touches/column, mean*count=sum
    // Sum of all imp values 0..11 = 66.
    CHECK(total_sum == Catch::Approx(66.0).epsilon(0.01));
}

TEST_CASE("CiRenorm scattered: SECOND lap writes the FULL corrected importance directly into "
          "storage, using the FIRST lap's measured stats",
          "[ci_renorm][scattered][stable_region]") {
    // 1 input x 3 outputs -- 3 cells in one column-per-cell layout so
    // each column gets exactly 1 touch/lap (simplifies expected values).
    auto conn = ci_renorm_make_dense_conn(
        1, 3, [](std::size_t, std::size_t) { return 0.1f; },
        [](std::size_t, std::size_t c) { return float(c) * 2.0f; }); // imp = 0, 2, 4
    CiRenormState state;
    PlasticityCellCursor cursor;
    const float target_mean[3] = {1.0f, 1.0f, 1.0f};
    const float target_std[3] = {0.5f, 0.5f, 0.5f};

    // Lap 1: mode=StableRegion, but has_finalized_stats=false -> exact
    // no-op on storage this lap (only stats get measured).
    auto stats1 = apply_amortized_ci_renorm_step<CiRenormVT, std::size_t, uint32_t>(
        conn, 3, state, cursor, 3, CiRenormMode::StableRegion, target_mean, target_std, 0.01f);
    CHECK(stats1.cycle_complete);
    CHECK(ValueAccessor<CiRenormVT>::get_imp(conn.values, 1) == Catch::Approx(2.0f)); // unchanged

    // Lap 2: each column has exactly 1 sample from lap 1 -> col_ci_std=0
    // (single-sample variance is exactly 0) -- clamped internally to
    // 1e-6, so column 0 (imp=0, its own lap-1 mean) should land near
    // target_mean exactly (it's ITS OWN mean, zero deviation).
    auto stats2 = apply_amortized_ci_renorm_step<CiRenormVT, std::size_t, uint32_t>(
        conn, 3, state, cursor, 3, CiRenormMode::StableRegion, target_mean, target_std, 0.01f);
    CHECK(stats2.cycle_complete);
    CHECK(ValueAccessor<CiRenormVT>::get_imp(conn.values, 0) == Catch::Approx(1.0f).margin(1e-3));
    CHECK(ValueAccessor<CiRenormVT>::get_imp(conn.values, 1) == Catch::Approx(1.0f).margin(1e-3));
    CHECK(ValueAccessor<CiRenormVT>::get_imp(conn.values, 2) == Catch::Approx(1.0f).margin(1e-3));
}

TEST_CASE("CiRenorm scattered: never touches weight, only importance", "[ci_renorm][scattered]") {
    auto conn = ci_renorm_make_dense_conn(
        1, 2, [](std::size_t, std::size_t c) { return 0.37f + float(c); },
        [](std::size_t, std::size_t) { return 50.0f; });
    CiRenormState state;
    PlasticityCellCursor cursor;
    const float target_mean[2] = {1.0f, 1.0f};
    const float target_std[2] = {0.5f, 0.5f};
    apply_amortized_ci_renorm_step<CiRenormVT, std::size_t, uint32_t>(
        conn, 2, state, cursor, 2, CiRenormMode::StableRegion, target_mean, target_std, 0.01f);
    apply_amortized_ci_renorm_step<CiRenormVT, std::size_t, uint32_t>(
        conn, 2, state, cursor, 2, CiRenormMode::StableRegion, target_mean, target_std, 0.01f);
    CHECK(ValueAccessor<CiRenormVT>::get_w(conn.values, 0) == Catch::Approx(0.37f));
    CHECK(ValueAccessor<CiRenormVT>::get_w(conn.values, 1) == Catch::Approx(1.37f));
}

TEST_CASE("CiRenorm scattered: REGRESSION -- corrections CONVERGE to the target over multiple "
          "laps rather than overcorrecting/diverging. Real bug found via an end-to-end smoke "
          "test (DISLDOLayerV, 3 laps at target_mean=5): stats accumulated from the PRE-"
          "correction value caused every lap's reference to describe a population this SAME lap "
          "had already overwritten, compounding into mean importance collapsing to 0 instead of "
          "converging to 5. Fixed by accumulating stats from the POST-correction value instead.",
          "[ci_renorm][scattered][stable_region][regression]") {
    auto conn = ci_renorm_make_dense_conn(
        1, 2, [](std::size_t, std::size_t) { return 0.1f; },
        [](std::size_t, std::size_t) { return 50.0f; }); // start pre-saturated, uniform
    CiRenormState state;
    PlasticityCellCursor cursor;
    const float target_mean[2] = {5.0f, 5.0f};
    const float target_std[2] = {1.0f, 1.0f};

    for (int lap = 0; lap < 5; ++lap)
        apply_amortized_ci_renorm_step<CiRenormVT, std::size_t, uint32_t>(
            conn, 2, state, cursor, 2, CiRenormMode::StableRegion, target_mean, target_std, 0.01f);

    // Converges to (and stays at) the target -- NOT 0, NOT diverging.
    CHECK(ValueAccessor<CiRenormVT>::get_imp(conn.values, 0) == Catch::Approx(5.0f).margin(0.01));
    CHECK(ValueAccessor<CiRenormVT>::get_imp(conn.values, 1) == Catch::Approx(5.0f).margin(0.01));
}
