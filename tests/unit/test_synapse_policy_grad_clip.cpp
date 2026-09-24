#include "../../sili/lib/headers/delta_csr_types.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <cmath>

// ── max_abs_grad: clip g/contrib BEFORE they enter the ci EMA ─────────────────
// See docs/research/delta_csr_types.rst:synapse_policy.max_abs_grad_clip
// for the full motivation, literature citation, and the exact numbers a
// standalone probe produced. The FIRST test case below deliberately calls
// ONLY update_ci's pre-existing 6-argument signature -- no max_abs_grad --
// so it stands alone as proof the failure mode is real in THIS codebase's
// actual formula, on a worktree that has never seen the fix. Every other
// test case exercises the fix itself and fails to compile until it exists.

TEST_CASE("PROBLEM: an unclipped gradient spike poisons BoundedRMSpropSynapsePolicy::update_ci "
          "for hundreds of subsequent steps (today's real 6-argument formula, no fix involved)",
          "[synapse_policy][bounded][grad_clip][problem_demonstration]") {
    using P = BoundedRMSpropSynapsePolicy<float>;

    // Warmup: 200 steps of a small, steady gradient (g=0.05, contrib=0)
    // from ci=0, using ONLY today's real 6-arg update_ci -- establishes a
    // realistic starting ci before any spike.
    float ci_start = 0.0f;
    for (int i = 0; i < 200; ++i) {
        ci_start = P::update_ci(ci_start, 0.05f, 0.0f, 0.999f, /*min_decay_frac=*/0.0f,
                                /*max_ci=*/1e30f);
    }

    // Baseline: no spike at all, 249 more steps of the same small gradient.
    float ci_baseline = ci_start;
    for (int i = 0; i < 249; ++i) {
        ci_baseline = P::update_ci(ci_baseline, 0.05f, 0.0f, 0.999f, 0.0f, 1e30f);
    }

    // Spike branch: ONE outlier gradient (g=50 -- a heavy-tailed attention
    // gradient per Zhang et al. 2020's own empirical finding that attention
    // architectures produce heavy-tailed gradient noise independent of
    // input data), then the SAME 249 steps of small gradients afterward.
    // Still only the 6-arg call -- this IS what every real call site in
    // sisldo_ops.hpp/linear_disldo_backward.hpp does today.
    float ci_spiked = P::update_ci(ci_start, 50.0f, 0.0f, 0.999f, 0.0f, 1e30f);
    for (int i = 0; i < 249; ++i) {
        ci_spiked = P::update_ci(ci_spiked, 0.05f, 0.0f, 0.999f, 0.0f, 1e30f);
    }

    // The spike injects (1-beta2)*50^2 = 2.5 into ci at the moment it
    // occurs -- 1000x the ~0.0025 a normal g=0.05 step contributes. After
    // 249 more EMA steps that excess has only decayed by 0.999^249 ~= 0.78,
    // so ~1.95 of it is STILL present -- ci_spiked is roughly 3 orders of
    // magnitude above ci_baseline, not a small residual bump. This is the
    // actual bug -- verified against today's real formula, before any fix.
    REQUIRE(ci_baseline > 0.0f); // sanity: warmup actually produced a nonzero baseline
    CHECK(ci_spiked > 100.0f * ci_baseline);
    CHECK(ci_spiked == Catch::Approx(ci_baseline + 2.5f * std::pow(0.999f, 249)).epsilon(0.05));
}

namespace {
float warmup_ci_clipped(float max_abs_grad) {
    using P = BoundedRMSpropSynapsePolicy<float>;
    float ci = 0.0f;
    for (int i = 0; i < 200; ++i) {
        ci = P::update_ci(ci, 0.05f, 0.0f, 0.999f, /*min_decay_frac=*/0.0f, /*max_ci=*/1e30f,
                          max_abs_grad);
    }
    return ci;
}
} // namespace

TEST_CASE("FIX: max_abs_grad clips the spike before it enters ci, avoiding the long-lived "
          "poisoning demonstrated above",
          "[synapse_policy][bounded][grad_clip]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float max_abs_grad = 1.0f;
    const float ci_start = warmup_ci_clipped(max_abs_grad);

    float ci_baseline = ci_start;
    for (int i = 0; i < 249; ++i) {
        ci_baseline = P::update_ci(ci_baseline, 0.05f, 0.0f, 0.999f, 0.0f, 1e30f, max_abs_grad);
    }

    // Same g=50 spike, but now clipped to max_abs_grad=1.0 before squaring.
    float ci_clipped = P::update_ci(ci_start, 50.0f, 0.0f, 0.999f, 0.0f, 1e30f, max_abs_grad);
    for (int i = 0; i < 249; ++i) {
        ci_clipped = P::update_ci(ci_clipped, 0.05f, 0.0f, 0.999f, 0.0f, 1e30f, max_abs_grad);
    }

    // The clipped spike injects only (1-beta2)*1.0^2 = 0.001 -- 2500x
    // smaller than the unclipped 2.5. Clipped ci should stay within a small
    // factor of baseline, nowhere near the >100x blowup the PROBLEM test
    // above shows for the identical g=50 spike.
    CHECK(ci_clipped < 3.0f * ci_baseline);
    CHECK(ci_clipped == Catch::Approx(ci_baseline + 0.001f * std::pow(0.999f, 249)).epsilon(0.05));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci -- max_abs_grad omitted defaults to no "
          "clipping, bit-identical to current behavior",
          "[synapse_policy][bounded][grad_clip][regression]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float ci_old = 0.2f, g = 0.37f, contrib = -0.11f, beta2 = 0.999f;
    const float without_arg = P::update_ci(ci_old, g, contrib, beta2, 0.0f, 1e30f);
    const float with_huge_default = P::update_ci(ci_old, g, contrib, beta2, 0.0f, 1e30f, 1e30f);
    CHECK(without_arg == Catch::Approx(with_huge_default));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci -- max_abs_grad clips a normal (non-spike) "
          "gradient too, and clips contrib independently of g",
          "[synapse_policy][bounded][grad_clip]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    // g under the clip, contrib over it -- both must be clipped
    // independently, not just their sum/combination.
    const float ci_old = 0.0f, g = 0.1f, contrib = 10.0f, beta2 = 0.999f;
    const float max_abs_grad = 1.0f;
    const float expected =
        beta2 * ci_old + (1.0f - beta2) * (g * g + 1.0f * 1.0f); // contrib clamped to 1.0
    CHECK(P::update_ci(ci_old, g, contrib, beta2, 0.0f, 1e30f, max_abs_grad) ==
          Catch::Approx(expected));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci -- max_abs_grad clips a large NEGATIVE "
          "gradient the same as a large positive one",
          "[synapse_policy][bounded][grad_clip]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float ci_old = 0.0f, beta2 = 0.999f, max_abs_grad = 1.0f;
    const float ci_pos = P::update_ci(ci_old, 50.0f, 0.0f, beta2, 0.0f, 1e30f, max_abs_grad);
    const float ci_neg = P::update_ci(ci_old, -50.0f, 0.0f, beta2, 0.0f, 1e30f, max_abs_grad);
    CHECK(ci_pos == Catch::Approx(ci_neg));
}

TEST_CASE("PlainRMSpropSynapsePolicy::update_ci also supports max_abs_grad, same semantics",
          "[synapse_policy][plain][grad_clip]") {
    using P = PlainRMSpropSynapsePolicy<float>;
    const float ci_old = 0.0f, beta2 = 0.999f, max_abs_grad = 1.0f;
    const float clipped = P::update_ci(ci_old, 50.0f, 0.0f, beta2, 0.0f, 1e30f, max_abs_grad);
    const float expected = beta2 * ci_old + (1.0f - beta2) * (1.0f * 1.0f);
    CHECK(clipped == Catch::Approx(expected));
    // Omitted -> unclipped, matches the un-clamped formula exactly.
    const float unclipped = P::update_ci(ci_old, 50.0f, 0.0f, beta2, 0.0f, 1e30f);
    const float expected_unclipped = beta2 * ci_old + (1.0f - beta2) * (50.0f * 50.0f);
    CHECK(unclipped == Catch::Approx(expected_unclipped));
}

TEST_CASE("BoundedRMSpropSynapsePolicy<Block4Vec>::update_ci supports max_abs_grad, matching "
          "the scalar version lane-for-lane",
          "[synapse_policy][bounded][block4][grad_clip]") {
    using PS = BoundedRMSpropSynapsePolicy<float>;
    using PV = BoundedRMSpropSynapsePolicy<Block4Vec>;
    const float ci_old = 0.0f, beta2 = 0.999f, max_abs_grad = 1.0f;

    const Block4Vec ci_v = block4_vec_broadcast(ci_old);
    const Block4Vec g_v = block4_vec_broadcast(50.0f);
    const Block4Vec contrib_v = block4_vec_broadcast(0.0f);
    const Block4Vec beta2_v = block4_vec_broadcast(beta2);
    const Block4Vec min_decay_v = block4_vec_broadcast(0.0f);
    const Block4Vec max_ci_v = block4_vec_broadcast(1e30f);
    const Block4Vec max_abs_grad_v = block4_vec_broadcast(max_abs_grad);

    const Block4Vec result_v =
        PV::update_ci(ci_v, g_v, contrib_v, beta2_v, min_decay_v, max_ci_v, max_abs_grad_v);
    float result_lanes[BLOCK4_TILE];
    block4_vec_store(result_lanes, result_v);

    const float expected = PS::update_ci(ci_old, 50.0f, 0.0f, beta2, 0.0f, 1e30f, max_abs_grad);
    for (float lane : result_lanes)
        CHECK(lane == Catch::Approx(expected));
}
