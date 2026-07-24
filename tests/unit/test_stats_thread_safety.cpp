#include "../../sili/lib/headers/sparse_struct.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <random>
#include <set>

// ── FP4BiPacked copy semantics ────────────────────────────────────────────────
//
// Documents real, DELIBERATE, pre-existing behavior found while verifying
// the thread-safety fixes below (see conversation) -- FP4BiPacked's copy
// constructor/assignment explicitly SHARES the underlying byte buffer
// (copies the shared_ptr, not its contents), not a bug, but a real
// footgun: `objA = objB` does NOT give an independent copy for anything
// using FP4BiPacked storage. Use delta_csr_from_absolute() (or another
// path that starts from a default-constructed FP4BiPacked, which
// legitimately allocates fresh storage) to get independent data, not
// copy-assignment from an existing populated instance.

TEST_CASE("FP4BiPacked copy-assignment shares storage, does NOT give an independent copy",
         "[fp4][copy_semantics][documented_behavior]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;
    std::vector<SIZE_TYPE> ptrs = {0, 1};
    std::vector<SIZE_TYPE> idx  = {0};
    std::vector<float> w = {1.0f}, imp = {0.0f};
    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights1;
    weights1.connections = dc;
    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights2;
    weights2.connections = dc;

    ValueAccessor<FP4BiPacked>::set(weights1.connections.values, 0, 6.0f, 0.0f);

    // This IS the documented behavior, not a regression -- weights2
    // changes too, because both share the same underlying buffer.
    CHECK(ValueAccessor<FP4BiPacked>::get_w(weights2.connections.values, 0) == Catch::Approx(6.0f));
}

TEST_CASE("delta_csr_from_absolute gives genuinely independent storage each call",
         "[fp4][copy_semantics]") {
    // The correct way to get independent data -- confirms the escape hatch
    // actually works, not just that the footgun above exists.
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;
    std::vector<SIZE_TYPE> ptrs = {0, 1};
    std::vector<SIZE_TYPE> idx  = {0};
    std::vector<float> w = {1.0f}, imp = {0.0f};

    auto dc1 = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(64), std::size_t(64));
    auto dc2 = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(1), std::size_t(1), std::size_t(64), std::size_t(64));

    ValueAccessor<FP4BiPacked>::set(dc1.values, 0, 6.0f, 0.0f);
    CHECK(ValueAccessor<FP4BiPacked>::get_w(dc2.values, 0) == Catch::Approx(1.0f));   // unaffected
}

// ── Thread safety ──────────────────────────────────────────────────────────────
//
// Regression tests for a real bug found and fixed this session: the running
// L1/L2/max stats were originally updated by calling
// weights.update_importance_stats()/update_value_stats() directly inside
// each kernel's #pragma omp parallel loop -- a data race on the shared
// importance_l1/l2_sq/max_abs (and value_ equivalents) fields, undetected
// because every earlier test used num_cpus=1. Fixed via per-thread local
// accumulation + one aggregate call per thread (not per synapse) after the
// parallel region -- these tests actually exercise num_cpus>1, which is
// the one thing that would have caught the original bug.
//
// NOTE: independent trial data is constructed via delta_csr_from_absolute()
// per trial, NOT via copy-assignment from a shared source -- see the
// copy-semantics tests above for why that distinction matters here
// specifically.

static SparseLinearWeightsDelta<int, FP4BiPacked, uint32_t> build_test_layer(
    const std::vector<int>& ptrs, const std::vector<int>& idx,
    const std::vector<float>& w0, const std::vector<float>& imp0,
    std::size_t rows, std::size_t cols)
{
    auto dc = delta_csr_from_absolute<int, FP4BiPacked, uint32_t>(
        ptrs, idx, w0, imp0, rows, cols, std::size_t(1) << 20, std::size_t(1) << 20);
    SparseLinearWeightsDelta<int, FP4BiPacked, uint32_t> weights;
    weights.connections = dc;
    weights.out_degree.assign(cols, int(0));
    weights.recompute_stats();
    return weights;
}

TEST_CASE("delta_csr_synap_row_step's row cursor advances even when the merge itself throws",
         "[synaptogenesis][regression]") {
    // Underpins SparseLinearLayer::synap_step()'s stateful auto-advancing
    // wrapper: a caller doing "for _ in range(n): layer.synap_step(...)"
    // needs the sweep to keep moving even when one row's merge fails
    // (insufficient headroom -- throws, per the loud-failure design), not
    // get stuck repeatedly retrying the same row. Row advancement happens
    // early in delta_csr_synap_row_step, before the throw -- verified
    // directly here, not assumed from reading the code.
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;
    std::vector<SIZE_TYPE> ptrs = {0, 2, 3, 4};
    std::vector<SIZE_TYPE> idx  = {0, 2, 1, 3};
    std::vector<float>     w    = {1.5f, -2.0f, 0.5f, 3.0f};
    std::vector<float>     imp  = {0.0f, 0.0f, 0.0f, 0.0f};
    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(3), std::size_t(4), std::size_t(4096), std::size_t(4096));

    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(4, SIZE_TYPE(0));

    std::vector<float> input_accum = {5.0f, 3.0f, 4.0f};
    std::vector<float> grad_accum  = {2.0f, 3.0f, 1.0f, 4.0f};
    delta_csr_build_probes<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, input_accum.data(), grad_accum.data(), SIZE_TYPE(3));

    std::size_t cursor = 0;
    std::vector<std::size_t> rows_visited;
    for (int i = 0; i < 6; ++i) {
        std::size_t before = cursor;
        try {
            delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights, cursor, -1e9f, SIZE_TYPE(10));
        } catch (const std::runtime_error&) {
            // Expected on this tiny test layer -- headroom exhausts fast.
        }
        rows_visited.push_back(before);
    }
    CHECK(rows_visited == std::vector<std::size_t>({0,1,2,0,1,2}));
}

TEST_CASE("disldo_forward/backward: num_cpus=1 and num_cpus=8 give matching stats",
         "[thread_safety][regression]") {
    using S = int; using COL_TYPE = uint32_t;
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> col_dist(0, 199);
    std::uniform_real_distribution<float> val_dist(-2.0f, 2.0f);
    std::vector<S> ptrs(201, 0), idx; std::vector<float> w0, imp0;
    for (int r = 0; r < 200; ++r) {
        std::set<int> cols;
        while ((int)cols.size() < 10) cols.insert(col_dist(rng));
        for (int c : cols) { idx.push_back(c); w0.push_back(val_dist(rng)); imp0.push_back(0.0f); }
        ptrs[r+1] = (S)idx.size();
    }

    auto run_trial = [&](int num_cpus) {
        auto weights = build_test_layer(ptrs, idx, w0, imp0, 200, 200);
        std::mt19937 trial_rng(999);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (int step = 0; step < 10; ++step) {
            std::vector<float> input(200), output(200, 0.0f);
            for (auto& v : input) v = d(trial_rng);
            disldo_forward<S, FP4BiPacked, COL_TYPE>(
                input.data(), S(1), S(200), weights, output.data(), 0.3f, num_cpus);
            std::vector<float> dy(200), dx(200, 0.0f), in_acc(200, 0.0f), gr_acc(200, 0.0f);
            for (auto& v : dy) v = d(trial_rng);
            disldo_backward<S, FP4BiPacked, COL_TYPE>(
                input.data(), S(1), S(200), dy.data(), weights, dx.data(),
                in_acc.data(), gr_acc.data(), 0.3f, num_cpus);
        }
        return weights;
    };

    auto w1 = run_trial(1);
    auto w8 = run_trial(8);
    CHECK(w1.value_l1        == Catch::Approx(w8.value_l1).margin(1e-2));
    CHECK(w1.importance_l1   == Catch::Approx(w8.importance_l1).margin(1e-2));
    CHECK(w1.value_l2_sq     == Catch::Approx(w8.value_l2_sq).margin(1e-2));
    CHECK(w1.importance_l2_sq == Catch::Approx(w8.importance_l2_sq).margin(1e-2));

    for (int rep = 0; rep < 5; ++rep) {
        auto w = run_trial(8);
        CHECK(w.value_l1      == Catch::Approx(w8.value_l1).margin(1e-2));
        CHECK(w.importance_l1 == Catch::Approx(w8.importance_l1).margin(1e-2));
    }
}
