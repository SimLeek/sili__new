// apply_amortized_l2_init (delta_csr_types.hpp) -- L2 Init (Kumar,
// Marklund & Van Roy, "Maintaining Plasticity in Continual Learning via
// Regenerative Regularization", CoLLAs 2025, arXiv:2308.11958):
// regularize weights toward their OWN initial value, not toward zero.
// Direct instruction, after the offline Python sandbox
// (scripts/plasticity_sim.py, sili_peridot) showed this candidate
// avoiding importance saturation while keeping real accumulated
// importance: "let's try both of them and record while seeing if it
// helps reliably beat mqar rather than randomly stalling."
//
// A cell's FIRST touch only captures its current weight as the
// reference "initial" value (no modification that touch -- there's
// nothing to regularize toward yet). Every SUBSEQENT touch pulls the
// weight `rate` of the way back toward that captured reference.
// Importance is never touched (L2 Init targets trainable parameters,
// i.e. weight, not this project's own ci accumulator). Same
// amortized/chunked-cursor shape as apply_amortized_decay_stats, for
// interface consistency.
#include "../../sili/lib/headers/delta_csr_types.hpp"
#include <cmath>
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

using VT = DeltaCSRBiValues<float>;

struct L2InitTestState {
    std::size_t cursor = 0;
    std::vector<float> initial_weight;
    std::vector<uint8_t> captured;
};

static void test_first_touch_only_captures_no_modification() {
    VT values;
    values.weights = {1.0f, 2.0f, 3.0f};
    values.importance = {9.0f, 9.0f, 9.0f};
    L2InitTestState st;

    auto r = apply_amortized_l2_init<VT, float>(values, st.cursor, st.initial_weight, st.captured,
                                                3, /*rate=*/0.5f);
    CHECK(r.cycle_complete, "3/3 touched should complete the cycle");
    CHECK(values.weights[0] == 1.0f, "first touch must not modify weight, got %f",
          values.weights[0]);
    CHECK(values.weights[1] == 2.0f, "first touch must not modify weight, got %f",
          values.weights[1]);
    CHECK(values.weights[2] == 3.0f, "first touch must not modify weight, got %f",
          values.weights[2]);
    CHECK(values.importance[0] == 9.0f, "importance must never be touched");
    CHECK(st.initial_weight[0] == 1.0f && st.initial_weight[1] == 2.0f &&
              st.initial_weight[2] == 3.0f,
          "initial_weight should capture the first-seen values exactly");
    CHECK(st.captured[0] && st.captured[1] && st.captured[2],
          "all 3 cells should be marked captured");
}

static void test_second_touch_pulls_toward_captured_initial() {
    VT values;
    values.weights = {1.0f};
    values.importance = {9.0f};
    L2InitTestState st;

    apply_amortized_l2_init<VT, float>(values, st.cursor, st.initial_weight, st.captured, 1, 0.5f);
    CHECK(values.weights[0] == 1.0f, "first touch: no change yet");

    values.weights[0] = 5.0f; // simulate real training moving the weight away from init
    apply_amortized_l2_init<VT, float>(values, st.cursor, st.initial_weight, st.captured, 1, 0.5f);
    // pull: new_w = w + rate*(initial - w) = 5 + 0.5*(1-5) = 3.0
    CHECK(std::abs(values.weights[0] - 3.0f) < 1e-6f,
          "second touch should pull halfway back to initial (1.0) from 5.0 -> 3.0, got %f",
          values.weights[0]);
    CHECK(values.importance[0] == 9.0f, "importance must still be untouched");
}

static void test_at_initial_value_stays_unchanged() {
    VT values;
    values.weights = {2.0f};
    values.importance = {9.0f};
    L2InitTestState st;

    apply_amortized_l2_init<VT, float>(values, st.cursor, st.initial_weight, st.captured, 1, 0.5f);
    apply_amortized_l2_init<VT, float>(values, st.cursor, st.initial_weight, st.captured, 1, 0.5f);
    CHECK(std::abs(values.weights[0] - 2.0f) < 1e-6f,
          "a weight already AT its initial value should stay there, got %f", values.weights[0]);
}

static void test_cursor_wraps_and_reports_cycle_complete() {
    VT values;
    values.weights = {1.0f, 2.0f, 3.0f};
    values.importance = {0.0f, 0.0f, 0.0f};
    L2InitTestState st;

    auto r1 = apply_amortized_l2_init<VT, float>(values, st.cursor, st.initial_weight, st.captured,
                                                 2, 0.5f);
    CHECK(!r1.cycle_complete, "2/3 touched should not complete a cycle");
    CHECK(st.cursor == 2, "cursor should sit at 2, got %zu", st.cursor);

    auto r2 = apply_amortized_l2_init<VT, float>(values, st.cursor, st.initial_weight, st.captured,
                                                 2, 0.5f);
    CHECK(r2.cycle_complete, "2+2=4 touches over 3 cells should complete a cycle");
    CHECK(st.cursor == 1, "cursor should wrap and sit at 1 (4 touches mod 3), got %zu", st.cursor);
}

static void test_empty_layer_completes_immediately() {
    VT values;
    L2InitTestState st;
    auto r = apply_amortized_l2_init<VT, float>(values, st.cursor, st.initial_weight, st.captured,
                                                4, 0.5f);
    CHECK(r.cycle_complete, "empty layer should report cycle_complete=true immediately");
    CHECK(r.n_touched == 0, "empty layer should touch nothing, got %zu", r.n_touched);
}

int main() {
    test_first_touch_only_captures_no_modification();
    test_second_touch_pulls_toward_captured_initial();
    test_at_initial_value_stays_unchanged();
    test_cursor_wraps_and_reports_cycle_complete();
    test_empty_layer_completes_immediately();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
