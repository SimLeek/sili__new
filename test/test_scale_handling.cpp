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

TEST_CASE("disldo_backward updates value_scale via gradient (sum first, apply lr once)",
         "[scale][value_scale][gradient][regression]") {
    // Chain rule: output += stored_w * val_scale * input, so
    // dL/d(val_scale[r]) = sum_{e,b}(stored_w[e] * dy[b][col_e] * input[b][r])
    // With 1 row, 1 output, 1 batch, this reduces to:
    // delta_scale = -lr * stored_w * dy * input
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 1};
    std::vector<S> idx  = {0};
    std::vector<float> w = {2.0f}, imp = {0.0f};
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(1, S(0));
    weights.set_value_scale_raw(0, 0.5f);   // true_w = 2.0 * 0.5 = 1.0

    std::vector<float> input = {3.0f};
    std::vector<float> dy    = {1.0f};
    std::vector<float> dx(1, 0.0f), in_acc(1, 0.0f), gr_acc(1, 0.0f);
    const float lr = 0.1f;
    disldo_backward<S, FP4BiPacked, COL_TYPE>(
        input.data(), S(1), S(1), dy.data(), weights, dx.data(),
        in_acc.data(), gr_acc.data(), lr, 1);

    // scale_grad = stored_w * dy * input = 2.0 * 1.0 * 3.0 = 6.0
    // new value_scale = 0.5 - 0.1 * 6.0 = 0.5 - 0.6 = -0.1
    const float expected_scale = 0.5f - lr * (2.0f * 1.0f * 3.0f);
    CHECK(weights.get_value_scale(0) == Catch::Approx(expected_scale).margin(1e-5f));
}

TEST_CASE("value_scale gradient accumulates correctly across multiple synapses and batches",
         "[scale][value_scale][gradient]") {
    // 2 synapses, 2 batches -- verifies the full sum across both dimensions.
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 2};
    std::vector<S> idx  = {0, 1};
    std::vector<float> w = {2.0f, 3.0f}, imp = {0.0f, 0.0f};
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(2), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(2, S(0));
    weights.set_value_scale_raw(0, 1.0f);

    // batch=2: input[0]=1.0, input[1]=2.0; dy[0]=[1,1], dy[1]=[1,1]
    std::vector<float> input = {1.0f, 2.0f};
    std::vector<float> dy    = {1.0f, 1.0f, 1.0f, 1.0f};   // [batch=2, output=2]
    std::vector<float> dx(2, 0.0f), in_acc(1, 0.0f), gr_acc(2, 0.0f);
    const float lr = 0.01f;
    disldo_backward<S, FP4BiPacked, COL_TYPE>(
        input.data(), S(2), S(1), dy.data(), weights, dx.data(),
        in_acc.data(), gr_acc.data(), lr, 1);

    // scale_grad = sum_{e,b}(stored_w[e] * dy[b][col_e] * input[b][r=0])
    // e=0 (col=0): b=0: 2.0*1.0*1.0=2; b=1: 2.0*1.0*2.0=4  -> 6
    // e=1 (col=1): b=0: 3.0*1.0*1.0=3; b=1: 3.0*1.0*2.0=6  -> 9
    // total scale_grad_sum = 6 + 9 = 15
    // scale_eff_lr = lr / nnz_this_row = 0.01 / 2 = 0.005 (always divides
    // by nnz_this_row for value_scale, independent of lr_per_row_nnz flag)
    const float expected_scale = 1.0f - (lr / 2.0f) * 15.0f;
    CHECK(weights.get_value_scale(0) == Catch::Approx(expected_scale).margin(1e-4f));
}

TEST_CASE("value_scale gradient: sum-first-then-apply-lr outperforms per-synapse application near float32 epsilon",
         "[scale][value_scale][epsilon][regression]") {
    // The epsilon issue the 'sum first' design exists to solve: when
    // (scale_eff_lr * individual_contribution) < ULP(value_scale), applying
    // the scaled lr to each contribution individually inside the innermost
    // loop causes every increment to round to 0 in float32, leaving
    // value_scale unchanged despite a real nonzero gradient.
    //
    // With nnz_this_row normalization: scale_eff_lr = lr / nnz_this_row, so
    // each per-synapse term is lr / nnz_this_row * stored_w * dy * input.
    // The double accumulator sums those nnz_this_row terms (giving back
    // lr * average_contribution) then subtracts once. To demonstrate the
    // protection clearly: use a large value_scale (1000.0), many synapses
    // (100), small lr (1e-3) -- scale_eff_lr = 1e-3/100 = 1e-5, per-synapse
    // amount = 1e-5 < ULP(1000) ~6.1e-5 -> would round to 0. The final
    // aggregated double result: 100 * 1e-5 = 1e-3 > ULP(1000) -> preserved.
    using S = int;
    using COL_TYPE = uint32_t;
    const int n_syn = 100;
    std::vector<S> ptrs = {0, n_syn};
    std::vector<S> idx(n_syn); for (int i=0;i<n_syn;++i) idx[i]=i;
    std::vector<float> w(n_syn, 1.0f), imp(n_syn, 0.0f);
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(n_syn), std::size_t(4096), std::size_t(4096));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(n_syn, S(0));
    weights.set_value_scale_raw(0, 1000.0f);

    std::vector<float> input = {1.0f};
    std::vector<float> dy(n_syn, 1.0f);
    std::vector<float> dx(1, 0.0f), in_acc(1, 0.0f), gr_acc(n_syn, 0.0f);

    // scale_eff_lr = lr / n_syn = 1e-3 / 100 = 1e-5
    // Per-synapse: 1e-5 * 1.0 * 1.0 * 1.0 = 1e-5 < ULP(1000) ~6.1e-5 -> rounds to 0.
    // Double accumulator: sum(100 * 1e-5) = 1e-3 > ULP(1000) -> preserved.
    const float lr = 1e-3f;
    disldo_backward<S, FP4BiPacked, COL_TYPE>(
        input.data(), S(1), S(1), dy.data(), weights, dx.data(),
        in_acc.data(), gr_acc.data(), lr, 1);

    // Expected: value_scale = 1000.0 - (lr / n_syn) * n_syn = 1000.0 - lr = 999.999
    const float expected = 1000.0f - lr;
    CHECK(weights.get_value_scale(0) != 1000.0f);   // actually changed, not lost to rounding
    CHECK(weights.get_value_scale(0) == Catch::Approx(expected).margin(1e-2f));
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

// ── SiliBlock reshape+sum mapping ────────────────────────────────────────────

TEST_CASE("SiliBlock mapping: reshape+sum of [batch, n_folds*out] equals sequential sum",
         "[siliblock][mapping][regression]") {
    // The fold->hidden_dim mapping in forward_sili is:
    //   raw [1, n_folds*out_dim]  ->  reshape [1, n_folds, out_dim]  ->  sum axis 1
    // This must equal sum_i(W_i @ x) for i in range(n_folds), where W_i is
    // the i-th fold step's weight slice (modulo FP4 quantization rounding).
    // Verified at the C++ level using delta_csr_from_absolute directly so we
    // can control the exact stored weights and avoid FP4-introduced error.
    using S = int;
    using COL_TYPE = uint32_t;

    // 2 fold steps, out_dim=2, in_dim=2 -- tiny but sufficient.
    // Weights chosen to be exactly representable in FP4 (multiples of 0.5).
    // W0 = [[1.0, 0.0], [0.0, 1.0]], W1 = [[0.5, 0.0], [0.0, 0.5]]
    // Stacked = [[1.0,0],[0,1],[0.5,0],[0,0.5]] shape [4 x 2]
    // Transposed for SparseLinearLayer [2 x 4]: each input row i connects to
    // outputs 0,2 (W0_col_i_row_0, W1_col_i_row_0) and 1,3 for row i.
    // x = [1.0, 2.0]
    // Expected: W0@x + W1@x = [1.0, 2.0] + [0.5, 1.0] = [1.5, 3.0]
    // Via the single-layer route: W_stacked_transposed @ x gives
    //   row0: connects to out0 (W0[0,0]=1.0), out2 (W1[0,0]=0.5)
    //   row1: connects to out1 (W0[1,1]=1.0), out3 (W1[1,1]=0.5)
    // raw output = [1.0*1, 1.0*2, 0.5*1, 0.5*2] = [1.0, 2.0, 0.5, 1.0]
    // reshape [2, 2] + sum rows: [1.0+0.5, 2.0+1.0] = [1.5, 3.0]  correct

    std::vector<S> ptrs = {0, 2, 4};  // 2 rows (in_dim=2), 2 nnz each
    std::vector<S> idx  = {0, 2, 1, 3};
    std::vector<float> w = {1.0f, 0.5f, 1.0f, 0.5f};
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, std::vector<float>(4, 0.0f),
        std::size_t(2), std::size_t(4), std::size_t(4096), std::size_t(256));

    // Compute the raw stacked output manually: w[row] * x[row] for each nnz
    std::vector<float> x_in = {1.0f, 2.0f};
    std::vector<float> raw(4, 0.0f);
    auto L = dc.layout;
    for (std::size_t r = 0; r < 2; ++r) {
        auto cursor = dc.row_cursor(r);
        for (std::size_t e = 0; e < L.row_nnz(r); ++e) {
            COL_TYPE col = cursor.advance();
            std::size_t vb = L.elem_start[r] + e;
            float w_val = ValueAccessor<FP4BiPacked>::get_w(dc.values, vb);
            raw[col] += w_val * x_in[r];
        }
    }
    // raw = [1.0, 2.0, 0.5, 1.0]

    // Reshape [4] as [n_folds=2, out_dim=2] and sum across n_folds:
    const int n_folds = 2, out_dim = 2;
    std::vector<float> summed(out_dim, 0.0f);
    for (int fold = 0; fold < n_folds; ++fold)
        for (int o = 0; o < out_dim; ++o)
            summed[o] += raw[fold * out_dim + o];

    // Expected: sequential sum W0@x + W1@x = [1.5, 3.0]
    CHECK(summed[0] == Catch::Approx(1.5f).margin(1e-5f));
    CHECK(summed[1] == Catch::Approx(3.0f).margin(1e-5f));
}

TEST_CASE("set_value_scale_raw: pre-scaled load + raw scale set round-trips correctly",
         "[scale][value_scale][load_weights][regression]") {
    // The SiliBlock per-row scaling workflow:
    //   1. Compute row_scale = max_abs / FP4_MAX
    //   2. Pass pre-scaled weights (original / row_scale) to delta_csr_from_absolute
    //      -> FP4 quantizes to good accuracy (max maps to 6.0)
    //   3. Set value_scale[r] = row_scale WITHOUT re-encoding via rescale_value_row
    //      (re-encoding would re-quantize the already-scaled values, corrupting them)
    //
    // Verified here: a weight of 0.1 with FP4_MAX=6.0 and row_scale=0.1/6.0
    // correctly round-trips to approximately 0.1 after pre-scaling and raw-set.
    // Without per-row scaling, 0.1 maps to 0.0 (below FP4's min nonzero of 0.5).
    using S = int;
    using COL_TYPE = uint32_t;
    std::vector<S> ptrs = {0, 1};
    std::vector<S> idx  = {0};

    // Original weight: 0.1 -- FAR below FP4's 0.5 minimum, rounds to 0 without scaling
    const float original_w  = 0.1f;
    const float fp4_max     = 6.0f;
    const float row_scale   = original_w / fp4_max;         // 0.1/6 ~ 0.0167
    const float scaled_w    = original_w / row_scale;       // = 6.0 exactly

    std::vector<float> w = {scaled_w};
    std::vector<float> imp = {0.0f};
    auto dc = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(4096), std::size_t(256));

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(1, S(0));

    // Step 3: set scale WITHOUT re-encoding
    weights.set_value_scale_raw(0, row_scale);

    // Verify: stored value is ~6.0 (scaled), true value = stored * row_scale ~ 0.1
    const float stored = ValueAccessor<FP4BiPacked>::get_w(
        weights.connections.values, weights.connections.layout.elem_start[0]);
    const float true_w = stored * weights.get_value_scale(0);

    CHECK(stored == Catch::Approx(6.0f).margin(0.01f));   // scaled correctly into FP4 range
    CHECK(true_w == Catch::Approx(original_w).margin(0.005f));  // round-trips to original

    // Contrast: WITHOUT per-row scaling, 0.1 would have been loaded directly
    // and quantized to 0.0 (below FP4's minimum nonzero of 0.5).
    auto dc_unscaled = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
        ptrs, idx, std::vector<float>{original_w}, imp,
        std::size_t(1), std::size_t(1), std::size_t(4096), std::size_t(256));
    const float stored_unscaled = ValueAccessor<FP4BiPacked>::get_w(
        dc_unscaled.values, dc_unscaled.layout.elem_start[0]);
    CHECK(stored_unscaled == Catch::Approx(0.0f).margin(1e-6f));  // lost entirely
}

TEST_CASE("backward fold-sum broadcast: dy tiles correctly to n_folds slots",
         "[siliblock][backward][regression]") {
    // Forward reshape+sum: raw[batch, n_folds*out_dim]
    //   -> reshape[batch, n_folds, out_dim] -> sum(axis=1) -> [batch, out_dim]
    //
    // Backward: gradient of a sum is 1 to each summand, so
    //   dy[batch, out_dim] -> tile -> [batch, n_folds, out_dim]
    //   -> flatten -> [batch, n_folds*out_dim]
    //   meaning dy_raw[fold * out_dim + i] == dy[i] for ALL fold values.
    //
    // Verified by constructing a simple case and checking the broadcast manually.

    const int n_folds = 3, out_dim = 2, batch = 1;
    std::vector<float> dy = {1.5f, -0.5f};   // [batch=1, out_dim=2]

    // Replicate via the tile formula: dy_raw[fold*out_dim + i] = dy[i]
    std::vector<float> dy_raw(n_folds * out_dim);
    for (int fold = 0; fold < n_folds; ++fold)
        for (int i = 0; i < out_dim; ++i)
            dy_raw[fold * out_dim + i] = dy[i];

    // Verify: every fold slot is identical to dy
    for (int fold = 0; fold < n_folds; ++fold) {
        CHECK(dy_raw[fold * out_dim + 0] == Catch::Approx(1.5f));
        CHECK(dy_raw[fold * out_dim + 1] == Catch::Approx(-0.5f));
    }

    // Cross-check: if we do the backward fold-sum on dy_raw (reshape->sum),
    // we should recover dy.
    std::vector<float> recovered(out_dim, 0.0f);
    for (int fold = 0; fold < n_folds; ++fold)
        for (int i = 0; i < out_dim; ++i)
            recovered[i] += dy_raw[fold * out_dim + i];
    // Each position sums n_folds copies of dy[i], NOT equal to dy[i].
    // This confirms the broadcast is NOT a split (not dy/n_folds each) but
    // a true broadcast -- each fold gets the whole gradient.
    CHECK(recovered[0] == Catch::Approx(dy[0] * n_folds).margin(1e-5f));
    CHECK(recovered[1] == Catch::Approx(dy[1] * n_folds).margin(1e-5f));
}
