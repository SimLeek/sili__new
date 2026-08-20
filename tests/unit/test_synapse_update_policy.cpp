#include "../../sili/lib/headers/delta_csr_types.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <cmath>

// ── Per-synapse ci-update policy (PlainRMSpropSynapsePolicy /
// BoundedRMSpropSynapsePolicy) -- see delta_csr_types.hpp's own docstring
// for the full root-cause story (late-training RMSprop divergence once ci
// decays down to match a shrinking residual gradient). Exactly two entry
// points per policy, each owning its full formula end-to-end -- update_ci
// (the complete per-synapse second-moment update, Bounded's version floors
// the decay rate) and update_cw (the complete weight-delta computation:
// damp_by_importance branch, division, and Bounded's hard clip, all in one
// place -- no math left inline at any call site, no third "combine"
// wrapper).

TEST_CASE("PlainRMSpropSynapsePolicy::update_ci reproduces plain unbounded EMA exactly",
         "[synapse_policy][plain]") {
    using P = PlainRMSpropSynapsePolicy<float>;
    const float ci_old = 0.5f, g = 0.1f, contrib = 0.05f, beta2 = 0.999f;
    const float expected = beta2 * ci_old + (1.0f - beta2) * (g * g + contrib * contrib);
    // min_decay_frac and max_ci are both accepted but ignored -- no floor,
    // no ceiling at all.
    CHECK(P::update_ci(ci_old, g, contrib, beta2, 0.99f, 1e30f) == Catch::Approx(expected));
    CHECK(P::update_ci(ci_old, g, contrib, beta2, 0.0f, 1e30f) == Catch::Approx(expected));
}

TEST_CASE("PlainRMSpropSynapsePolicy::update_cw reproduces plain RMSprop delta exactly",
         "[synapse_policy][plain]") {
    using P = PlainRMSpropSynapsePolicy<float>;
    const float g = 0.37f, ci = 0.2f, S = 1.0f, eff_lr = 0.02f, eps = 1e-8f;
    const float expected_damped = (-eff_lr * g * S) / (std::sqrt(ci) + eps);
    const float expected_undamped = (-eff_lr * g * S);
    CHECK(P::update_cw(g, ci, S, eff_lr, eps, /*damp_by_importance=*/true, 1e30f)
          == Catch::Approx(expected_damped));
    CHECK(P::update_cw(g, ci, S, eff_lr, eps, /*damp_by_importance=*/false, 1e30f)
          == Catch::Approx(expected_undamped));
    // max_abs_delta is accepted but ignored -- no clip at all, even when
    // it would otherwise bind.
    CHECK(P::update_cw(g, ci, S, eff_lr, eps, true, 1e-10f) == Catch::Approx(expected_damped));
}

TEST_CASE("PlainRMSpropSynapsePolicy::update_cw applies S correctly",
         "[synapse_policy][plain]") {
    using P = PlainRMSpropSynapsePolicy<float>;
    const float g = 0.5f, ci = 0.1f, eff_lr = 0.01f, eps = 1e-8f;
    const float with_S2 = P::update_cw(g, ci, 2.0f, eff_lr, eps, true, 1e30f);
    const float with_S1 = P::update_cw(g, ci, 1.0f, eff_lr, eps, true, 1e30f);
    CHECK(with_S2 == Catch::Approx(2.0f * with_S1));
}

TEST_CASE("PlainRMSpropSynapsePolicy::update_cw's scale_invariant flag makes the "
         "TRUE-WEIGHT delta (S*update_cw) independent of S",
         "[synapse_policy][plain][scale_invariant]") {
    // Root cause this flag fixes (see sili_peridot's scale_invariant_chain_rule
    // torch prototype): the historical formula computes raw from g*S, so
    // Δtrue_weight = S*update_cw(...) scales QUADRATICALLY with S once S
    // deviates from 1.0 (S²-collapse). scale_invariant=true instead computes
    // raw from the bare g and divides by S at apply time, so Δtrue_weight is
    // exactly S*(eff_lr*raw/S) = eff_lr*raw -- flat in S.
    using P = PlainRMSpropSynapsePolicy<float>;
    const float g = 0.5f, ci = 0.1f, eff_lr = 0.01f, eps = 1e-8f;
    const float dcw_S1 = P::update_cw(g, ci, 1.0f, eff_lr, eps, true, 1e30f, /*scale_invariant=*/true);
    const float dcw_S4 = P::update_cw(g, ci, 4.0f, eff_lr, eps, true, 1e30f, /*scale_invariant=*/true);
    const float true_delta_S1 = 1.0f * dcw_S1;
    const float true_delta_S4 = 4.0f * dcw_S4;
    CHECK(true_delta_S1 == Catch::Approx(true_delta_S4));

    // Confirm the OLD (default/false) formula does NOT have this property --
    // it reproduces the quadratic-in-S collapse the flag exists to fix.
    const float dcw_S1_old = P::update_cw(g, ci, 1.0f, eff_lr, eps, true, 1e30f, /*scale_invariant=*/false);
    const float dcw_S4_old = P::update_cw(g, ci, 4.0f, eff_lr, eps, true, 1e30f, /*scale_invariant=*/false);
    const float true_delta_S1_old = 1.0f * dcw_S1_old;
    const float true_delta_S4_old = 4.0f * dcw_S4_old;
    CHECK(true_delta_S4_old == Catch::Approx(16.0f * true_delta_S1_old));

    // Omitting the argument entirely must default to the old behavior --
    // exact bit-for-bit backward compatibility for every existing call site.
    CHECK(P::update_cw(g, ci, 1.0f, eff_lr, eps, true, 1e30f) == Catch::Approx(dcw_S1_old));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci matches plain EMA when ci is GROWING",
         "[synapse_policy][bounded]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    // A large gradient pushes the EMA up past the floor -- floor is
    // irrelevant here, should match plain EMA exactly.
    const float ci_old = 0.01f, g = 1.0f, contrib = 0.0f, beta2 = 0.999f;
    const float plain_ema = beta2 * ci_old + (1.0f - beta2) * (g * g + contrib * contrib);
    CHECK(P::update_ci(ci_old, g, contrib, beta2, 0.99f, 1e30f) == Catch::Approx(plain_ema));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci clamps a fast-decaying ci to the floor",
         "[synapse_policy][bounded]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    // Tiny gradient -- plain EMA's own worst-case retention is exactly
    // beta2 per step (see update_ci's own docstring), so min_decay_frac
    // must be ABOVE beta2 to bind at all -- picking beta2=0.999,
    // min_decay_frac=0.9999 (retain 99.99%, decaying only 0.01%/step,
    // slower than beta2's natural 0.1%/step) demonstrates the floor
    // actually engaging.
    const float ci_old = 1.0f, g = 0.0f, contrib = 0.0f, beta2 = 0.999f;
    const float min_decay_frac = 0.9999f;
    const float plain_ema = beta2 * ci_old;  // g=contrib=0, EMA decays toward 0
    const float floored = P::update_ci(ci_old, g, contrib, beta2, min_decay_frac, 1e30f);
    CHECK(plain_ema < min_decay_frac * ci_old);          // confirms the floor is actually binding here
    CHECK(floored == Catch::Approx(min_decay_frac * ci_old));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci with min_decay_frac=0 reproduces plain EMA",
         "[synapse_policy][bounded]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float ci_old = 1.0f, g = 0.0f, contrib = 0.0f, beta2 = 0.999f;
    const float plain_ema = beta2 * ci_old;
    CHECK(P::update_ci(ci_old, g, contrib, beta2, 0.0f, 1e30f) == Catch::Approx(plain_ema));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci with min_decay_frac=1 never decays",
         "[synapse_policy][bounded]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    // AMSGrad-style permanent freeze -- explicitly NOT the intended default
    // (see delta_csr_types.hpp's own docstring), but should still behave
    // correctly if requested.
    const float ci_old = 1.0f;
    float ci = ci_old;
    for (int i = 0; i < 50; ++i) {
        ci = P::update_ci(ci, 0.0f, 0.0f, 0.999f, 1.0f, 1e30f);
    }
    CHECK(ci == Catch::Approx(ci_old));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci's max_ci clamps a growing ci",
         "[synapse_policy][bounded][max_ci]") {
    // ci has no ceiling anywhere else in this design -- min_decay_frac's
    // floor only bounds how fast ci can DECAY, never how fast it can grow
    // (confirmed directly this session: ci was measured climbing
    // continuously and unboundedly, 0.0005 -> 163+, across a 30000-step
    // unsafe-pocket run with no max_ci applied). A huge gradient should
    // push the EMA well past a small max_ci, and the result must be
    // clamped exactly at max_ci.
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float ci_old = 0.1f, g = 100.0f, contrib = 0.0f, beta2 = 0.999f;
    const float max_ci = 0.5f;
    const float unclipped_ema = beta2 * ci_old + (1.0f - beta2) * (g * g + contrib * contrib);
    REQUIRE(unclipped_ema > max_ci);  // sanity: this really would overshoot without the clip
    CHECK(P::update_ci(ci_old, g, contrib, beta2, 0.0f, max_ci) == Catch::Approx(max_ci));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci's max_ci does nothing when ci stays under it",
         "[synapse_policy][bounded][max_ci]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float ci_old = 0.1f, g = 0.05f, contrib = 0.02f, beta2 = 0.999f;
    const float plain_ema = beta2 * ci_old + (1.0f - beta2) * (g * g + contrib * contrib);
    CHECK(P::update_ci(ci_old, g, contrib, beta2, 0.0f, 1e30f) == Catch::Approx(plain_ema));
    CHECK(P::update_ci(ci_old, g, contrib, beta2, 0.0f, 5.0f) == Catch::Approx(plain_ema));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_ci's max_ci and min_decay_frac floor compose correctly",
         "[synapse_policy][bounded][max_ci]") {
    // max_ci must win even when the floor alone would also want to push ci
    // upward relative to a shrinking EMA -- std::min(std::max(ema,floor),max_ci)
    // ordering: floor first, then ceiling, so max_ci always has final say.
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float ci_old = 10.0f, g = 0.0f, contrib = 0.0f, beta2 = 0.999f;
    const float min_decay_frac = 0.9999f;  // floor would keep ci near ci_old
    const float max_ci = 1.0f;             // but ceiling is well below ci_old
    CHECK(P::update_ci(ci_old, g, contrib, beta2, min_decay_frac, max_ci) == Catch::Approx(max_ci));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_cw matches plain RMSprop delta when under the clip",
         "[synapse_policy][bounded]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    using Plain = PlainRMSpropSynapsePolicy<float>;
    const float g = 0.1f, ci = 1.0f, S = 1.0f, eff_lr = 0.01f, eps = 1e-8f;
    const float plain = Plain::update_cw(g, ci, S, eff_lr, eps, true, 1e30f);
    const float bounded = P::update_cw(g, ci, S, eff_lr, eps, true, 1e30f);  // max_abs_delta huge, no clip
    CHECK(bounded == Catch::Approx(plain));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_cw's clip actually engages on a real large step",
         "[synapse_policy][bounded]") {
    // update_cw clips the LR-INDEPENDENT raw update, THEN multiplies by
    // eff_lr (see its own docstring, delta_csr_types.hpp, for why -- clip
    // AFTER the lr multiply made max_abs_delta an absolute cap regardless
    // of lr, silently crushing every step for callers using a much bigger
    // lr than whatever this policy was tuned against). So the final
    // clipped-and-scaled result is eff_lr*max_abs_delta, not max_abs_delta
    // itself.
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float g = 1.0f, ci = 1e-6f, S = 1.0f, eff_lr = 0.05f, eps = 1e-8f;
    const float max_abs_delta = 0.01f;

    const float unclipped_raw_would_be = (-g * S) / (std::sqrt(ci) + eps);
    REQUIRE(std::abs(unclipped_raw_would_be) > max_abs_delta);  // sanity: this really would overshoot without the clip

    const float delta = P::update_cw(g, ci, S, eff_lr, eps, true, max_abs_delta);
    CHECK(std::abs(delta) == Catch::Approx(eff_lr * max_abs_delta));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_cw's clip preserves sign",
         "[synapse_policy][bounded]") {
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float ci = 1e-6f, S = 1.0f, eff_lr = 0.05f, eps = 1e-8f, max_abs_delta = 0.01f;
    const float delta_pos_g = P::update_cw(1.0f, ci, S, eff_lr, eps, true, max_abs_delta);
    const float delta_neg_g = P::update_cw(-1.0f, ci, S, eff_lr, eps, true, max_abs_delta);
    CHECK(delta_pos_g == Catch::Approx(-eff_lr * max_abs_delta));  // positive g -> negative (descent) delta
    CHECK(delta_neg_g == Catch::Approx(eff_lr * max_abs_delta));
}

TEST_CASE("BoundedRMSpropSynapsePolicy::update_cw's scale_invariant flag makes the "
         "TRUE-WEIGHT delta (S*update_cw) independent of S, clip included",
         "[synapse_policy][bounded][scale_invariant]") {
    // Same property as the Plain-policy version of this test, but also
    // confirms the clip (max_abs_delta) is applied to the S-INDEPENDENT raw
    // value before the eff_lr*raw/S division -- clipping the raw gradient
    // step, not the final S-scaled delta, is what makes the clip threshold
    // mean the same thing regardless of a synapse's current output_scale.
    using P = BoundedRMSpropSynapsePolicy<float>;
    const float g = 0.5f, ci = 0.1f, eff_lr = 0.01f, eps = 1e-8f;
    const float max_abs_delta = 1e30f;  // huge -- clip must not engage here
    const float dcw_S1 = P::update_cw(g, ci, 1.0f, eff_lr, eps, true, max_abs_delta, /*scale_invariant=*/true);
    const float dcw_S4 = P::update_cw(g, ci, 4.0f, eff_lr, eps, true, max_abs_delta, /*scale_invariant=*/true);
    CHECK(1.0f * dcw_S1 == Catch::Approx(4.0f * dcw_S4));

    // Old (default/false) formula: quadratic-in-S collapse still present.
    const float dcw_S1_old = P::update_cw(g, ci, 1.0f, eff_lr, eps, true, max_abs_delta, /*scale_invariant=*/false);
    const float dcw_S4_old = P::update_cw(g, ci, 4.0f, eff_lr, eps, true, max_abs_delta, /*scale_invariant=*/false);
    CHECK(4.0f * dcw_S4_old == Catch::Approx(16.0f * (1.0f * dcw_S1_old)));

    // Omitting the argument defaults to the old (false) behavior exactly.
    CHECK(P::update_cw(g, ci, 1.0f, eff_lr, eps, true, max_abs_delta) == Catch::Approx(dcw_S1_old));

    // Clip engages on the raw (pre-division) value: with a tiny max_abs_delta,
    // both S=1 and S=4 must clip to the SAME true-weight delta magnitude,
    // proving the clip threshold is S-independent under scale_invariant=true.
    const float tight_clip = 0.001f;
    const float clipped_S1 = P::update_cw(g, ci, 1.0f, eff_lr, eps, true, tight_clip, /*scale_invariant=*/true);
    const float clipped_S4 = P::update_cw(g, ci, 4.0f, eff_lr, eps, true, tight_clip, /*scale_invariant=*/true);
    CHECK(1.0f * clipped_S1 == Catch::Approx(4.0f * clipped_S4));
    CHECK(std::abs(clipped_S1) == Catch::Approx(eff_lr * tight_clip));
}

TEST_CASE("update_ci/update_cw match linear_disldo.hpp's existing inline formula exactly",
         "[synapse_policy][plain][regression]") {
    // Extracted verbatim from linear_disldo.hpp's own inline math (see e.g.
    // line ~719): ci = beta2*ci + (1-beta2)*(g*g+contrib*contrib);
    // cw += damp_by_importance ? (-effective_lr*g)/(sqrt(ci)+eps) : (-effective_lr*g);
    // -- must stay bit-identical, this IS the backward-compat contract
    // PlainRMSpropSynapsePolicy exists to guarantee before any real call
    // site is switched over to it.
    using P = PlainRMSpropSynapsePolicy<float>;
    float ci_direct = 0.2f;
    const float g = 0.37f, contrib = -0.11f, eff_lr = 0.02f, beta2 = 0.999f, eps = 1e-8f;

    ci_direct = beta2 * ci_direct + (1.0f - beta2) * (g * g + contrib * contrib);
    const float delta_direct = (-eff_lr * g) / (std::sqrt(ci_direct) + eps);

    const float ci_policy = P::update_ci(0.2f, g, contrib, beta2, /*min_decay_frac=*/0.0f, /*max_ci=*/1e30f);
    const float delta_policy = P::update_cw(g, ci_policy, /*S=*/1.0f, eff_lr, eps,
                                             /*damp_by_importance=*/true, /*max_abs_delta=*/1e30f);

    CHECK(ci_policy == Catch::Approx(ci_direct));
    CHECK(delta_policy == Catch::Approx(delta_direct));
}
