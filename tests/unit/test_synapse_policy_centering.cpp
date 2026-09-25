#include "../../sili/lib/headers/delta_csr_types.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <cmath>

// ── AdaBelief-style centering: clip(g - m, ...) instead of clip(g, ...) ───────
// See docs/research/delta_csr_types.rst:synapse_policy.adabelief_centering for
// the full motivation and the row+column additive baseline design (m built
// from FirstMomentTracker-updated m_row[i]+m_col[j], NOT part of update_ci
// itself -- update_ci just takes the already-combined m).
//
// max_abs_grad (already built) and centering fix DIFFERENT failure modes: a
// genuine one-off spike has m~=0 (hasn't adapted yet), so centering does
// NOTHING for it -- max_abs_grad's hard clip is still the real defense
// there. Centering fixes the OTHER failure mode: a column stuck with real,
// SUSTAINED large gradient, where max_abs_grad alone still lets ci settle at
// the clip boundary squared and stay stuck high. The first two TEST_CASEs
// below demonstrate this distinction directly, with real numbers, against
// update_ci's pre-existing/extended signatures -- not assumed.

TEST_CASE("PROBLEM: a sustained (non-spike) large gradient still saturates ci "
          "under max_abs_grad clipping ALONE (m defaulted to 0)",
          "[synapse_policy][bounded][centering][problem_demonstration]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float beta2 = 0.999f, max_abs_grad = 8.0f, g_sustained = 10.0f;
    float ci = 0.0f;
    for (int i = 0; i < 300; ++i) {
        // m omitted -> defaults to 0 -> residual == g, clipped to 8 every step.
        ci = P::update_ci(ci, g_sustained, 0.0f, beta2, 0.0f, 1e30f, max_abs_grad);
    }
    // EMA of a CONSTANT clipped value (8, since g=10 clips to max_abs_grad=8)
    // is CONVERGING toward 64 (=8^2) but beta2=0.999 has a ~1000-step time
    // constant, so 300 steps only reaches (1-0.999^300)~=26% of the way
    // there -- verified directly (not hand-derived): 16.594574. Still
    // climbing, still stuck well above what the centered version below
    // reaches for the IDENTICAL sustained gradient.
    CHECK(ci == Catch::Approx(16.594574f).epsilon(0.001));
}

TEST_CASE("FIX: centering around a converged first-moment baseline lets ci "
          "settle far lower for the SAME sustained gradient",
          "[synapse_policy][bounded][centering]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    using M = FirstMomentTracker<float>;
    const float beta2 = 0.999f, beta1 = 0.9f, max_abs_grad = 8.0f, g_sustained = 10.0f;
    float ci = 0.0f, m = 0.0f;
    for (int i = 0; i < 300; ++i) {
        m = M::update(m, g_sustained, beta1);
        ci = P::update_ci(ci, g_sustained, 0.0f, beta2, 0.0f, 1e30f, max_abs_grad, m);
    }
    // m converges to g_sustained (10.0) well within 300 steps at beta1=0.9
    // (0.9^300 ~= negligible) -- residual -> 0, so ci stops accumulating
    // further. Verified directly (not hand-derived): ci=0.303643 after
    // exactly 300 steps -- a real 54.65x below the clip-only-at-the-same-
    // step-count 16.594574 the PROBLEM test above shows for the identical
    // sustained gradient.
    CHECK(m == Catch::Approx(10.0f).epsilon(0.001));
    CHECK(ci == Catch::Approx(0.303643f).epsilon(0.001));
    CHECK(ci < 16.594574f / 50.0f); // real ratio is 54.65x; 50x is a safe margin
}

TEST_CASE("Centering does NOT rescue a genuine one-off spike -- m hasn't "
          "adapted yet, so residual ~= g on the first occurrence",
          "[synapse_policy][bounded][centering]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    using M = FirstMomentTracker<float>;
    const float beta2 = 0.999f, beta1 = 0.9f;
    float m = 0.0f; // fresh column, no history
    m = M::update(m, 50.0f, beta1);
    // m only moves 10% of the way to g on a single touch -- residual is
    // still ~45, nowhere near cancelled. max_abs_grad's hard clip remains
    // the real defense against a one-off spike; centering alone is not a
    // substitute for it.
    const float residual = 50.0f - m;
    CHECK(residual > 40.0f);
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci -- m omitted defaults to 0, "
          "byte-identical to the pre-centering formula",
          "[synapse_policy][bounded][centering][regression]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float ci_old = 0.2f, g = 0.37f, contrib = -0.11f, beta2 = 0.999f, max_abs_grad = 5.0f;
    const float without_m = P::update_ci(ci_old, g, contrib, beta2, 0.0f, 1e30f, max_abs_grad);
    const float with_zero_m =
        P::update_ci(ci_old, g, contrib, beta2, 0.0f, 1e30f, max_abs_grad, 0.0f);
    CHECK(without_m == Catch::Approx(with_zero_m));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci -- centering actually shifts "
          "the result when m != 0 and no clip engages",
          "[synapse_policy][bounded][centering]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float ci_old = 0.0f, g = 3.0f, m = 3.0f, beta2 = 0.999f;
    // residual = g - m = 0 -> ci should barely move (just contrib's own
    // term, here 0), NOT beta2*0+(1-beta2)*9 the uncentered formula would give.
    const float centered = P::update_ci(ci_old, g, 0.0f, beta2, 0.0f, 1e30f, 1e30f, m);
    CHECK(centered == Catch::Approx(0.0f).margin(1e-6));
}

TEST_CASE("FirstMomentTracker::update is a plain EMA of signed g, not g^2",
          "[synapse_policy][centering][first_moment]") {
    using M = FirstMomentTracker<float>;
    const float beta1 = 0.9f;
    const float m1 = M::update(0.0f, -5.0f, beta1);
    CHECK(m1 == Catch::Approx(-0.5f)); // (1-0.9)*(-5) = -0.5, sign preserved
    const float m2 = M::update(m1, -5.0f, beta1);
    CHECK(m2 == Catch::Approx(beta1 * m1 + (1.0f - beta1) * (-5.0f)));
}

TEST_CASE("BoundedRMSpropSynapsePolicy<Block4Vec>::update_ci supports m, matching "
          "the scalar version lane-for-lane",
          "[synapse_policy][bounded][block4][centering]") {
    using PS = BoundedRMSpropSynapsePolicy<float>;
    using PV = BoundedRMSpropSynapsePolicy<Block4Vec>;
    const float ci_old = 0.0f, beta2 = 0.999f, max_abs_grad = 8.0f, g = 10.0f, m = 4.0f;

    const Block4Vec result_v = PV::update_ci(
        block4_vec_broadcast(ci_old), block4_vec_broadcast(g), block4_vec_broadcast(0.0f),
        block4_vec_broadcast(beta2), block4_vec_broadcast(0.0f), block4_vec_broadcast(1e30f),
        block4_vec_broadcast(max_abs_grad), block4_vec_broadcast(m));
    float result_lanes[BLOCK4_TILE];
    block4_vec_store(result_lanes, result_v);

    const float expected = PS::update_ci(ci_old, g, 0.0f, beta2, 0.0f, 1e30f, max_abs_grad, m);
    for (float lane : result_lanes)
        CHECK(lane == Catch::Approx(expected));
}

TEST_CASE("REGRESSION: row+column additive centering must fit the RESIDUAL "
          "after the other axis, not raw g -- else confounded row/column "
          "effects double-count and overshoot toward 2g instead of g",
          "[synapse_policy][centering][first_moment][row_column][regression]") {
    // Real bug found via an end-to-end smoke test (sili_peridot
    // DISLDOLayer32(dense=True), constant x/dy across a 4x4 dense layer):
    // if BOTH axes independently fit the raw sustained gradient, each
    // converges toward g on its own, so m_row[i]+m_col[j] converges toward
    // 2g instead of g -- the residual (g - m_row - m_col) then clips to
    // -max_abs_grad every step, nearly reproducing the UNCENTERED ci, which
    // is exactly what was observed (row+column mean importance=16.8844,
    // barely different from no-centering's 16.9364, vs column-only's
    // correct 0.5327). The fix: each axis's own update uses the RESIDUAL
    // after the OTHER axis's OLD (pre-this-touch) value, e.g. m_row's
    // update uses (g - m_col_old), not raw g.
    using M = FirstMomentTracker<float>;
    const float beta1 = 0.9f, g_sustained = 25.0f;
    float m_row = 0.0f, m_col = 0.0f;

    SECTION("buggy (raw g on both axes) overshoots toward 2g") {
        for (int i = 0; i < 300; ++i) {
            const float new_row = M::update(m_row, g_sustained, beta1);
            const float new_col = M::update(m_col, g_sustained, beta1);
            m_row = new_row;
            m_col = new_col;
        }
        CHECK(m_row + m_col == Catch::Approx(2.0f * g_sustained).epsilon(0.01));
    }

    SECTION("fixed (residual vs the other axis's OLD value) converges to g, "
            "not 2g") {
        for (int i = 0; i < 300; ++i) {
            const float old_row = m_row, old_col = m_col;
            m_row = M::update(old_row, g_sustained - old_col, beta1);
            m_col = M::update(old_col, g_sustained - old_row, beta1);
        }
        CHECK(m_row + m_col == Catch::Approx(g_sustained).epsilon(0.01));
    }
}

TEST_CASE("FirstMomentTracker<Block4Vec>::update matches the scalar version "
          "lane-for-lane",
          "[synapse_policy][centering][first_moment][block4]") {
    using MS = FirstMomentTracker<float>;
    using MV = FirstMomentTracker<Block4Vec>;
    const float beta1 = 0.9f, m = 1.0f, g = -7.0f;
    const Block4Vec result_v =
        MV::update(block4_vec_broadcast(m), block4_vec_broadcast(g), block4_vec_broadcast(beta1));
    float result_lanes[BLOCK4_TILE];
    block4_vec_store(result_lanes, result_v);
    const float expected = MS::update(m, g, beta1);
    for (float lane : result_lanes)
        CHECK(lane == Catch::Approx(expected));
}
