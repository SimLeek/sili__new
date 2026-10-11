// apply_amortized_decay_stats (delta_csr_types.hpp) -- the amortized
// decoupled-decay + rolling-stats mechanism (direct instruction: "we can
// calculate the exact L2 regularization we need... amortized by holding an
// iterator num per layer and iterating over a fixed amount of synapses").
// Generic over VALUES_TYPE via ValueAccessor -- covers fp32
// (DeltaCSRBiValues<float>, DISLDOLayerV's storage), FP4 (FP4BiPacked,
// SparseLinearLayer's storage), and FP8 (FP8BiValues, SparseLinearLayer8's
// storage), since the mechanism is meant to generalize across all three
// (later direct instruction: "Can this work on the FP4 and FP8 layers as
// well?").
//
// Verifies: (1) each touch actually multiplies the weight by decay_factor
// (importance left unchanged); (2) the rolling cursor wraps correctly and
// reports cycle_complete exactly when a full pass finishes; (3) stats
// (mean_abs/rms/max_abs/n) are correct for a just-finished cycle and are
// reset afterward; (4) mid-cycle calls report cycle_complete=false; (5) an
// empty layer (nnz=0) reports cycle_complete=true immediately with n=0,
// no crash.
#include "../../sili/lib/headers/delta_csr_types.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

// State bundle mirroring what each real layer class (DISLDOLayerV,
// SparseLinearLayerImpl, SparseLinearLayer8Impl) holds as member fields.
struct DecayState {
    std::size_t cursor = 0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    std::size_t n = 0;
};

// ── fp32 (DeltaCSRBiValues<float>) ──────────────────────────────────────────

static void test_fp32_basic_decay_and_cycle() {
    using VT = DeltaCSRBiValues<float>;
    VT values;
    const std::size_t total = 5;
    values.weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    values.importance = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f};

    DecayState st;
    const float decay = 0.5f;

    // chunk_size=2: touches idx 0,1 -- not a full cycle yet.
    auto r1 = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                     st.max_abs, st.n, 2, decay);
    CHECK(!r1.cycle_complete, "fp32: 2/5 touched should not complete a cycle");
    CHECK(std::abs(values.weights[0] - 0.5f) < 1e-6f, "fp32: w[0] should decay to 0.5, got %.6f",
          values.weights[0]);
    CHECK(std::abs(values.weights[1] - 1.0f) < 1e-6f, "fp32: w[1] should decay to 1.0, got %.6f",
          values.weights[1]);
    CHECK(std::abs(values.importance[0] - 9.0f) < 1e-6f, "fp32: importance must be untouched");
    CHECK(st.cursor == 2, "fp32: cursor should sit at 2, got %zu", st.cursor);

    // chunk_size=3: touches idx 2,3,4 -- completes the cycle (total=5).
    auto r2 = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                     st.max_abs, st.n, 3, decay);
    CHECK(r2.cycle_complete, "fp32: full pass (2+3=5) should complete a cycle");
    CHECK(std::abs(values.weights[2] - 1.5f) < 1e-6f, "fp32: w[2] should decay to 1.5, got %.6f",
          values.weights[2]);
    CHECK(std::abs(values.weights[3] - 2.0f) < 1e-6f, "fp32: w[3] should decay to 2.0, got %.6f",
          values.weights[3]);
    CHECK(std::abs(values.weights[4] - 2.5f) < 1e-6f, "fp32: w[4] should decay to 2.5, got %.6f",
          values.weights[4]);
    CHECK(st.cursor == 0, "fp32: cursor should wrap to 0 after a full cycle, got %zu", st.cursor);

    // Expected post-decay abs values across the WHOLE cycle: 0.5,1.0,1.5,2.0,2.5
    const double expect_mean = (0.5 + 1.0 + 1.5 + 2.0 + 2.5) / 5.0;
    const double expect_rms = std::sqrt((0.25 + 1.0 + 2.25 + 4.0 + 6.25) / 5.0);
    CHECK(std::abs(r2.mean_abs - expect_mean) < 1e-5, "fp32: mean_abs=%.6f expected %.6f",
          r2.mean_abs, expect_mean);
    CHECK(std::abs(r2.rms - expect_rms) < 1e-5, "fp32: rms=%.6f expected %.6f", r2.rms, expect_rms);
    CHECK(std::abs(r2.max_abs - 2.5) < 1e-6, "fp32: max_abs=%.6f expected 2.5", r2.max_abs);
    CHECK(r2.n == 5, "fp32: n should be 5 (full cycle), got %zu", r2.n);

    // Accumulators must have been reset after being read.
    CHECK(st.sum_abs == 0.0 && st.sum_sq == 0.0 && st.max_abs == 0.0 && st.n == 0,
          "fp32: accumulators should reset after a completed cycle is read");

    // Next call (mid-cycle again) should report cycle_complete=false with a
    // FRESH (zeroed) stats accumulation, not leftover state from the prior cycle.
    auto r3 = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                     st.max_abs, st.n, 1, decay);
    CHECK(!r3.cycle_complete, "fp32: 1/5 into a new cycle should not complete");
}

static void test_fp32_empty_layer() {
    using VT = DeltaCSRBiValues<float>;
    VT values; // no weights/importance -- nnz=0
    DecayState st;
    auto r = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                    st.max_abs, st.n, 4, 0.5f);
    CHECK(r.cycle_complete, "empty layer should report cycle_complete=true immediately");
    CHECK(r.n == 0, "empty layer should report n=0");
}

static void test_fp32_multi_cycle_half_life() {
    // Verify the intended half-life semantics directly: a synapse touched
    // once per cycle, decayed by `decay` each touch, should sit at
    // initial_w * decay^num_cycles after num_cycles full passes.
    using VT = DeltaCSRBiValues<float>;
    VT values;
    const std::size_t total = 4;
    values.weights.assign(total, 8.0f);
    values.importance.assign(total, 1.0f);
    DecayState st;
    const float decay = 0.5f; // half-life == 1 cycle by construction
    const int num_cycles = 3;
    for (int c = 0; c < num_cycles; ++c) {
        apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq, st.max_abs,
                                               st.n, total, decay);
    }
    const float expect = 8.0f * std::pow(0.5f, float(num_cycles)); // 1.0
    for (std::size_t i = 0; i < total; ++i) {
        CHECK(std::abs(values.weights[i] - expect) < 1e-4f,
              "fp32: after %d cycles at decay=0.5, w[%zu]=%.6f expected %.6f", num_cycles, i,
              values.weights[i], expect);
    }
}

static void test_fp32_decay_importance_isolation() {
    // decay_importance=true (EXPERIMENTAL, 2026-09-20 loss-of-plasticity
    // mechanism): must decay importance only, leave weight byte-identical,
    // and its own stats (mean_abs/rms/max_abs) must reflect the DECAYED
    // IMPORTANCE values, not the untouched weights.
    using VT = DeltaCSRBiValues<float>;
    VT values;
    values.weights = {1.0f, 2.0f, 3.0f};
    values.importance = {10.0f, 20.0f, 30.0f};
    DecayState st;
    auto r = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                    st.max_abs, st.n, 3, 0.5f,
                                                    /*decay_importance=*/true);
    CHECK(r.cycle_complete, "fp32 decay_importance: full pass should complete a cycle");
    CHECK(values.weights[0] == 1.0f && values.weights[1] == 2.0f && values.weights[2] == 3.0f,
          "fp32 decay_importance=true: weights must be byte-identical to input, untouched");
    CHECK(values.importance[0] == 5.0f && values.importance[1] == 10.0f &&
              values.importance[2] == 15.0f,
          "fp32 decay_importance=true: importance should have decayed by 0.5 each");
    const double expect_mean = (5.0 + 10.0 + 15.0) / 3.0;
    CHECK(std::abs(r.mean_abs - expect_mean) < 1e-6,
          "fp32 decay_importance: mean_abs should reflect decayed IMPORTANCE (%.6f), got %.6f",
          expect_mean, r.mean_abs);
    CHECK(std::abs(r.max_abs - 15.0) < 1e-6,
          "fp32 decay_importance: max_abs should be 15.0, got %.6f", r.max_abs);
}

static void test_fp32_decay_importance_default_false_unchanged() {
    // Default-argument sanity: omitting decay_importance must reproduce
    // the exact original weight-only behavior (no accidental behavior
    // change from adding the new parameter).
    using VT = DeltaCSRBiValues<float>;
    VT values;
    values.weights = {2.0f};
    values.importance = {6.0f};
    DecayState st;
    auto r = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                    st.max_abs, st.n, 1, 0.5f);
    CHECK(r.cycle_complete, "single-element cycle should complete");
    CHECK(values.weights[0] == 1.0f, "default call should still decay weight to 1.0, got %f",
          values.weights[0]);
    CHECK(values.importance[0] == 6.0f, "default call must leave importance untouched");
}

// ── apply_amortized_flat_decay_stats (DenseLinearWeights' w/ci) ────────────

static void test_flat_decay_basic_and_cycle() {
    std::vector<float> w = {1.0f, 2.0f, 3.0f, 4.0f};
    DecayState st;
    auto r1 = apply_amortized_flat_decay_stats<float>(w, st.cursor, st.sum_abs, st.sum_sq,
                                                      st.max_abs, st.n, 2, 0.5f);
    CHECK(!r1.cycle_complete, "flat: 2/4 touched should not complete a cycle");
    CHECK(w[0] == 0.5f && w[1] == 1.0f, "flat: first 2 elements should have decayed");
    CHECK(w[2] == 3.0f && w[3] == 4.0f, "flat: untouched elements must be unchanged so far");

    auto r2 = apply_amortized_flat_decay_stats<float>(w, st.cursor, st.sum_abs, st.sum_sq,
                                                      st.max_abs, st.n, 2, 0.5f);
    CHECK(r2.cycle_complete, "flat: 2+2=4 should complete the cycle");
    CHECK(w[2] == 1.5f && w[3] == 2.0f, "flat: remaining elements should have decayed");
    const double expect_mean = (0.5 + 1.0 + 1.5 + 2.0) / 4.0;
    CHECK(std::abs(r2.mean_abs - expect_mean) < 1e-6, "flat: mean_abs=%.6f expected %.6f",
          r2.mean_abs, expect_mean);
    CHECK(r2.n == 4, "flat: n should be 4, got %zu", r2.n);
}

static void test_flat_decay_empty() {
    std::vector<float> w; // empty -- e.g. an unallocated ci vector
    DecayState st;
    auto r = apply_amortized_flat_decay_stats<float>(w, st.cursor, st.sum_abs, st.sum_sq,
                                                     st.max_abs, st.n, 4, 0.5f);
    CHECK(r.cycle_complete, "flat: empty vector should report cycle_complete=true immediately");
    CHECK(r.n == 0, "flat: empty vector should report n=0");
}

static void test_flat_decay_independent_from_weights_channel() {
    // Simulates DIDLDOLayerV's real usage: separate cursor/accumulator
    // state per channel (w vs ci), each its own apply_amortized_flat_decay_stats
    // call on its own vector -- verifies decaying one vector never touches
    // the other's storage (they're just two independent std::vector<float>,
    // but this is the actual invariant the C++ layer methods rely on).
    std::vector<float> w = {4.0f, 4.0f};
    std::vector<float> ci = {8.0f, 8.0f};
    DecayState w_st, ci_st;
    apply_amortized_flat_decay_stats<float>(w, w_st.cursor, w_st.sum_abs, w_st.sum_sq, w_st.max_abs,
                                            w_st.n, 2, 0.5f);
    CHECK(w[0] == 2.0f && w[1] == 2.0f, "flat: w channel should have decayed");
    CHECK(ci[0] == 8.0f && ci[1] == 8.0f, "flat: ci channel must be untouched by decaying w");
}

// ── FP4 (FP4BiPacked) ───────────────────────────────────────────────────────

static void test_fp4_decay_and_stats() {
    using VT = FP4BiPacked;
    VT values;
    const std::size_t total = 4;
    values.resize(total);
    // FP4_TABLE only has 16 discrete levels -- pick values exactly
    // representable so the decay math is checkable without quantization
    // slop swamping the assertion.
    values.set_live(0, 1.0f, 0.5f);
    values.set_live(1, 2.0f, 0.5f);
    values.set_live(2, -1.0f, 0.5f);
    values.set_live(3, 0.5f, 0.5f);

    DecayState st;
    auto r = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                    st.max_abs, st.n, total, 0.5f);
    CHECK(r.cycle_complete, "fp4: full-chunk touch should complete the cycle");
    CHECK(r.n == 4, "fp4: n should be 4, got %zu", r.n);
    // FP4_TABLE decode of 0.5 may itself not be exact -- check via the
    // accessor (round-trip through the SAME quantization the mechanism
    // uses) rather than assuming exact float equality.
    CHECK(std::abs(ValueAccessor<VT>::get_w(values, 0) - 0.5f) < 0.2f,
          "fp4: w[0] should have decayed toward 0.5, got %.4f",
          ValueAccessor<VT>::get_w(values, 0));
    CHECK(std::abs(ValueAccessor<VT>::get_imp(values, 0) - 0.5f) < 1e-4f,
          "fp4: importance must be untouched by decay, got %.4f",
          ValueAccessor<VT>::get_imp(values, 0));
    CHECK(std::abs(ValueAccessor<VT>::get_imp(values, 3) - 0.5f) < 1e-4f,
          "fp4: importance[3] must be untouched by decay, got %.4f",
          ValueAccessor<VT>::get_imp(values, 3));
}

static void test_fp4_never_zero_after_decay() {
    // The whole point of routing decay through set_live (not a raw set())
    // is to preserve FP4's never-0 live-quantize invariant even as the
    // decayed value shrinks toward (but never reaches, by design) exact 0.
    using VT = FP4BiPacked;
    VT values;
    values.resize(2);
    values.set_live(0, 0.05f, 0.3f); // near the smallest positive FP4 level
    values.set_live(1, -0.05f, 0.3f);
    DecayState st;
    // Decay hard, many touches -- if this ever produced a raw 0-code the
    // never-0 invariant (fp4_encode_bits_live) would have been violated.
    for (int i = 0; i < 20; ++i) {
        apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq, st.max_abs,
                                               st.n, 2, 0.3f);
    }
    CHECK(ValueAccessor<VT>::get_w(values, 0) != 0.0f,
          "fp4: decayed weight[0] must never hit exact 0");
    CHECK(ValueAccessor<VT>::get_w(values, 1) != 0.0f,
          "fp4: decayed weight[1] must never hit exact 0");
}

static void test_fp4_decay_importance_isolation() {
    using VT = FP4BiPacked;
    VT values;
    values.resize(2);
    values.set_live(0, 1.0f, 1.0f);
    values.set_live(1, -1.0f, 0.5f);
    DecayState st;
    auto r = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                    st.max_abs, st.n, 2, 0.5f,
                                                    /*decay_importance=*/true);
    CHECK(r.cycle_complete, "fp4 decay_importance: full pass should complete");
    // FP4 quantizes both channels through the same table -- check via the
    // accessor (round-trip through quantization) same as the weight test.
    CHECK(std::abs(ValueAccessor<VT>::get_w(values, 0) - 1.0f) < 0.2f,
          "fp4 decay_importance=true: weight[0] must be untouched, got %.4f",
          ValueAccessor<VT>::get_w(values, 0));
    CHECK(std::abs(ValueAccessor<VT>::get_imp(values, 0) - 0.5f) < 0.2f,
          "fp4 decay_importance=true: importance[0] should have decayed toward 0.5, got %.4f",
          ValueAccessor<VT>::get_imp(values, 0));
}

// ── FP8 (FP8BiValues) ───────────────────────────────────────────────────────

static void test_fp8_decay_and_stats() {
    using VT = FP8BiValues;
    VT values;
    const std::size_t total = 4;
    values.weights.resize(total);
    values.importance.resize(total);
    ValueAccessor<VT>::set_live(values, 0, 4.0f, 0.5f);
    ValueAccessor<VT>::set_live(values, 1, -4.0f, 0.5f);
    ValueAccessor<VT>::set_live(values, 2, 2.0f, 0.5f);
    ValueAccessor<VT>::set_live(values, 3, 1.0f, 0.5f);

    DecayState st;
    auto r = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                    st.max_abs, st.n, total, 0.5f);
    CHECK(r.cycle_complete, "fp8: full-chunk touch should complete the cycle");
    CHECK(r.n == 4, "fp8: n should be 4, got %zu", r.n);
    // E4M3 is exact for these particular values (powers of two / their
    // halves), so the tolerance can be tight.
    CHECK(std::abs(ValueAccessor<VT>::get_w(values, 0) - 2.0f) < 1e-3f,
          "fp8: w[0] should decay to 2.0, got %.4f", ValueAccessor<VT>::get_w(values, 0));
    CHECK(std::abs(ValueAccessor<VT>::get_w(values, 1) - (-2.0f)) < 1e-3f,
          "fp8: w[1] should decay to -2.0, got %.4f", ValueAccessor<VT>::get_w(values, 1));
    CHECK(std::abs(ValueAccessor<VT>::get_imp(values, 0) - 0.5f) < 1e-3f,
          "fp8: importance must be untouched by decay, got %.4f",
          ValueAccessor<VT>::get_imp(values, 0));
    CHECK(r.max_abs > 1.9 && r.max_abs < 2.1, "fp8: max_abs should be ~2.0, got %.4f", r.max_abs);
}

static void test_fp8_decay_importance_isolation() {
    using VT = FP8BiValues;
    VT values;
    values.weights.resize(2);
    values.importance.resize(2);
    ValueAccessor<VT>::set_live(values, 0, 4.0f, 8.0f);
    ValueAccessor<VT>::set_live(values, 1, -4.0f, 2.0f);
    DecayState st;
    auto r = apply_amortized_decay_stats<VT, float>(values, st.cursor, st.sum_abs, st.sum_sq,
                                                    st.max_abs, st.n, 2, 0.5f,
                                                    /*decay_importance=*/true);
    CHECK(r.cycle_complete, "fp8 decay_importance: full pass should complete");
    CHECK(std::abs(ValueAccessor<VT>::get_w(values, 0) - 4.0f) < 1e-3f,
          "fp8 decay_importance=true: weight[0] must be untouched, got %.4f",
          ValueAccessor<VT>::get_w(values, 0));
    CHECK(std::abs(ValueAccessor<VT>::get_imp(values, 0) - 4.0f) < 1e-3f,
          "fp8 decay_importance=true: importance[0] should decay to 4.0, got %.4f",
          ValueAccessor<VT>::get_imp(values, 0));
    CHECK(std::abs(ValueAccessor<VT>::get_imp(values, 1) - 1.0f) < 1e-3f,
          "fp8 decay_importance=true: importance[1] should decay to 1.0, got %.4f",
          ValueAccessor<VT>::get_imp(values, 1));
}

int main() {
    test_fp32_basic_decay_and_cycle();
    test_fp32_empty_layer();
    test_fp32_multi_cycle_half_life();
    test_fp32_decay_importance_isolation();
    test_fp32_decay_importance_default_false_unchanged();
    test_flat_decay_basic_and_cycle();
    test_flat_decay_empty();
    test_flat_decay_independent_from_weights_channel();
    test_fp4_decay_and_stats();
    test_fp4_never_zero_after_decay();
    test_fp4_decay_importance_isolation();
    test_fp8_decay_and_stats();
    test_fp8_decay_importance_isolation();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
