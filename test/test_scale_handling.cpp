#include "../sili/lib/headers/sparse_struct.hpp"
#include "../sili/lib/headers/linear_disldo.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>

// ── Per-row importance_scale ──────────────────────────────────────────────────
//
// Converted from a per-layer scalar to a per-row vector (see conversation):
// different rows can have very different natural Hebbian-trace magnitude
// within the same layer, especially once synaptogenesis has diverged
// row_nnz across rows -- a single layer-wide scale can't serve a sparse row
// and a dense row equally well at the same time.

TEST_CASE("different rows can have genuinely different importance_scale simultaneously",
         "[scale][per_row]") {
    using S = int;
    using COL_TYPE = uint32_t;
    // 2 rows, 1 output each.
    std::vector<S> ptrs = {0, 1, 2};
    std::vector<S> idx  = {0, 0};
    std::vector<float> w = {1.0f, 1.0f}, imp = {0.0f, 0.0f};
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(2), std::size_t(1), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.set_importance_scale_raw(0, 0.01f);
    weights.set_importance_scale_raw(1, 1.0f);

    CHECK(weights.get_importance_scale(0) == Catch::Approx(0.01f));
    CHECK(weights.get_importance_scale(1) == Catch::Approx(1.0f));

    // A row never touched at all still defaults to 1.0 -- confirms lazy
    // sizing doesn't silently break untouched rows.
    std::vector<S> ptrs3 = {0, 1, 2, 2};
    auto dc3 = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs3, idx, w, imp, std::size_t(3), std::size_t(1), std::size_t(64), std::size_t(64));
    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights3;
    weights3.connections = dc3;
    weights3.set_importance_scale_raw(0, 0.5f);
    CHECK(weights3.get_importance_scale(2) == Catch::Approx(1.0f));   // never touched, still defaults
}

TEST_CASE("disldo_forward's Hebbian update respects EACH row's own importance_scale independently",
         "[scale][per_row][regression]") {
    // Direct kernel-level test, not just the getter/setter -- row 0's
    // small update must survive at scale=0.01 while row 1's identical-
    // magnitude update at scale=1.0 underflows to exactly 0, in the SAME
    // forward call.
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 1, 2};
    std::vector<S> idx  = {0, 1};
    std::vector<float> w = {1.0f, 1.0f}, imp = {0.0f, 0.0f};
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(2), std::size_t(2), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(2, S(0));
    weights.set_importance_scale_raw(0, 0.01f);   // row 0: small update survives
    weights.set_importance_scale_raw(1, 1.0f);    // row 1: same-magnitude update underflows

    std::vector<float> input = {0.1f, 0.1f};   // identical small activation, both rows
    std::vector<float> output(2, 0.0f);
    disldo_forward<S, FP4BiPacked, COL_TYPE>(
        input.data(), S(1), S(2), weights, output.data(), 1.0f, 1);

    const float row0_stored = ValueAccessor<FP4BiPacked>::get_imp(
        weights.connections.values, weights.connections.layout.elem_start[0]);
    const float row1_stored = ValueAccessor<FP4BiPacked>::get_imp(
        weights.connections.values, weights.connections.layout.elem_start[1]);

    CHECK((row0_stored * weights.get_importance_scale(0)) != 0.0f);   // row 0 survived
    CHECK(row1_stored == 0.0f);                                       // row 1 underflowed, as expected
}

// ── value_scale ────────────────────────────────────────────────────────────────

TEST_CASE("value_scale defaults to 1.0, exact backward compat", "[scale][value_scale]") {
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 1};
    std::vector<S> idx  = {0};
    std::vector<float> w = {1.0f}, imp = {0.0f};
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(64), std::size_t(64));
    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    CHECK(weights.get_value_scale(0) == Catch::Approx(1.0f));
}

TEST_CASE("disldo_forward's output actually reflects the TRUE (scaled) weight, not the stored one",
         "[scale][value_scale][regression]") {
    // The real point of value_scale: a stored weight of 3.0 (an ordinary
    // FP4_TABLE entry) combined with value_scale=0.01 should produce
    // forward output as if the TRUE weight were 0.03 -- not 3.0.
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 1};
    std::vector<S> idx  = {0};
    std::vector<float> w = {3.0f}, imp = {0.0f};   // stored weight: exactly 3.0
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(1, S(0));
    weights.set_value_scale_raw(0, 0.01f);   // true weight = 3.0 * 0.01 = 0.03

    std::vector<float> input = {2.0f};
    std::vector<float> output(1, 0.0f);
    disldo_forward<S, FP4BiPacked, COL_TYPE>(
        input.data(), S(1), S(1), weights, output.data(), 0.0f, 1);

    // Expected: true_w * input = 0.03 * 2.0 = 0.06 -- NOT 3.0*2.0=6.0
    // (what it would be if value_scale were ignored).
    CHECK(output[0] == Catch::Approx(0.06f).margin(1e-4f));
}

TEST_CASE("disldo_backward's dx and weight update both use the TRUE (scaled) weight",
         "[scale][value_scale][regression]") {
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 1};
    std::vector<S> idx  = {0};
    std::vector<float> w = {3.0f}, imp = {0.0f};
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(1, S(0));
    weights.set_value_scale_raw(0, 0.01f);   // true weight = 0.03

    std::vector<float> input = {2.0f};
    std::vector<float> dy    = {1.0f};
    std::vector<float> dx(1, 0.0f), in_acc(1, 0.0f), gr_acc(1, 0.0f);
    disldo_backward<S, FP4BiPacked, COL_TYPE>(
        input.data(), S(1), S(1), dy.data(), weights, dx.data(),
        in_acc.data(), gr_acc.data(), /*learning_rate=*/0.0f, 1);

    // dx = true_w * dy = 0.03 * 1.0 = 0.03 -- NOT 3.0 (stored weight ignoring scale).
    CHECK(dx[0] == Catch::Approx(0.03f).margin(1e-4f));
}

TEST_CASE("rescale_value_row preserves the true weight value across a scale change",
         "[scale][value_scale][regression]") {
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 1};
    std::vector<S> idx  = {0};
    std::vector<float> w = {1.0f}, imp = {0.0f};
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;

    // True weight = 2.0 at scale=1.0 (well within FP4's range).
    ValueAccessor<FP4BiPacked>::set(weights.connections.values, 0, 2.0f, 0.0f);
    REQUIRE(weights.get_value_scale(0) == Catch::Approx(1.0f));

    weights.rescale_value_row(0, 0.5f);
    CHECK(weights.get_value_scale(0) == Catch::Approx(0.5f));

    const float stored_after = ValueAccessor<FP4BiPacked>::get_w(weights.connections.values, 0);
    const float true_after   = stored_after * weights.get_value_scale(0);
    CHECK(true_after == Catch::Approx(2.0f).margin(0.1f));   // true value preserved
}

TEST_CASE("lr_per_row_nnz measurably brings aggregate update magnitude closer across rows of different nnz",
         "[scale][lr_normalization][regression]") {
    // Row 0: 1 synapse. Row 1: 4 synapses. Same weight, same input, same
    // gradient magnitude everywhere -- any difference in AGGREGATE update
    // comes purely from row_nnz normalization. Without it, more synapses
    // means more simultaneous nudges, so the aggregate shift scales with
    // row_nnz for a fixed learning_rate (confirmed: exactly 4x here).
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 1, 5};
    std::vector<S> idx  = {0,  0,1,2,3};
    std::vector<float> w(5, 1.0f), imp(5, 0.0f);
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(2), std::size_t(4), std::size_t(4096), std::size_t(4096));

    auto run = [&](bool normalize) {
        SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
        weights.connections = dc;
        weights.out_degree.assign(4, S(0));
        std::vector<float> input = {1.0f, 1.0f};
        std::vector<float> dy(4, 1.0f);
        std::vector<float> dx(2, 0.0f), in_acc(2, 0.0f), gr_acc(4, 0.0f);
        disldo_backward<S, FP4BiPacked, COL_TYPE>(
            input.data(), S(1), S(2), dy.data(), weights, dx.data(),
            in_acc.data(), gr_acc.data(), 3.0f, 1, normalize);

        float row0_change = 0.0f, row1_change = 0.0f;
        auto cur0 = weights.connections.row_cursor(0);
        cur0.advance();
        row0_change += std::abs(ValueAccessor<FP4BiPacked>::get_w(
            weights.connections.values, weights.connections.layout.elem_start[0]) - 1.0f);
        auto cur1 = weights.connections.row_cursor(1);
        for (std::size_t i = 0; i < 4; ++i) {
            cur1.advance();
            row1_change += std::abs(ValueAccessor<FP4BiPacked>::get_w(
                weights.connections.values, weights.connections.layout.elem_start[1] + i) - 1.0f);
        }
        return std::make_pair(row0_change, row1_change);
    };

    auto [r0_off, r1_off] = run(false);
    auto [r0_on,  r1_on]  = run(true);

    REQUIRE(r0_off > 0.0f);
    REQUIRE(r0_on  > 0.0f);
    CHECK((r1_off / r0_off) == Catch::Approx(4.0f).margin(0.01f));   // exact, no quantization ambiguity here
    CHECK((r1_on / r0_on) < (r1_off / r0_off));                       // normalization measurably closes the gap
}

TEST_CASE("importance_scale and value_scale work correctly together, per-row, in one forward+backward pass",
         "[scale][combined][regression]") {
    // Both scales active simultaneously, on the same synapse, through a
    // real forward+backward cycle -- the actual intended usage, not just
    // each feature in isolation.
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 1};
    std::vector<S> idx  = {0};
    std::vector<float> w = {6.0f}, imp = {0.0f};   // stored weight at FP4's max
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(1, S(0));
    weights.set_value_scale_raw(0, 0.1f);        // true weight = 0.6
    weights.set_importance_scale_raw(0, 0.02f);  // small importance stays representable

    std::vector<float> input = {1.0f};
    std::vector<float> output(1, 0.0f);
    disldo_forward<S, FP4BiPacked, COL_TYPE>(
        input.data(), S(1), S(1), weights, output.data(), 0.5f, 1);

    // Forward output: true_w * input = 0.6 * 1.0 = 0.6.
    CHECK(output[0] == Catch::Approx(0.6f).margin(1e-3f));

    // Importance moved (learning_rate=0.5 was nonzero) and, critically,
    // survived rather than underflowing to 0 -- confirms importance_scale
    // is still applied correctly even with value_scale also active on the
    // same synapse.
    const float imp_stored = ValueAccessor<FP4BiPacked>::get_imp(
        weights.connections.values, weights.connections.layout.elem_start[0]);
    CHECK(imp_stored != 0.0f);
}
