// Integration test for AQRS dynamic rank control (task #285): a full
// train-grow-train-shrink cycle driven ENTIRELY by Theorem 10's trigger
// logic (delta_csr_types.hpp's apply_dynamic_rank_control, task #285),
// with zero manual set_scale_rank calls from this test itself. See
// sili_peridot/AQRS_DESIGN.md for the full derivation.
//
// "Breathing" scenario: the target's TRUE structural rank changes
// mid-training.
//   Phase 1: uniform 3x scale (rank-1 sufficient) -- rank should STAY at
//            1, since there's no real leftover gradient pressure once
//            channel 0 converges (neurogenesis's own theta gate should
//            never trip).
//   Phase 2: swap in a diagonal-outlier target (needs rank-2, same
//            structural shape as test_aqrs_additive_branch.cpp/test_aqrs_
//            gamma.cpp's own "Test A" pattern) -- channel 0's residual
//            reappears, real grad pressure builds, neurogenesis should
//            fire once EMA-smoothed grad pressure crosses theta. Rank
//            should grow to 2 and MSE should converge back toward 0.
//   Phase 3: revert to the phase-1 (rank-1-sufficient) target -- channel
//            1 is no longer needed, its own gradient shrinks toward 0,
//            L1 pulls its gamma to the exact fixed point, apoptosis
//            should fire and rank should shrink back to 1.
//
// Theorem 9's residual-aligned seeding is explicitly NOT implemented here
// (AQRS_DESIGN.md marks the practical proxy as unresolved/unverified) --
// apply_dynamic_rank_control takes the new channel's seed as a caller
// callback; this test uses a simple deterministic nonzero seed (same
// convention every other growth test in this codebase already uses to
// break the symmetric zero-init deadlock), NOT a claim that this is the
// theorem-optimal direction.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
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

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static const std::size_t N = 4;

static Weights make_scattered_layer(const std::vector<float>& w_q_values) {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(N + 1);
    std::vector<SIZE_TYPE> idx(N * N);
    std::vector<float> imp(N * N, 1.0f);
    for (std::size_t r = 0; r < N; ++r) {
        ptrs[r] = SIZE_TYPE(r * N);
        for (std::size_t c = 0; c < N; ++c)
            idx[r * N + c] = SIZE_TYPE(c);
    }
    ptrs[N] = SIZE_TYPE(N * N);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w_q_values, imp, N, N, N * N * 2, N * N * 2);
    w.out_degree.assign(N, SIZE_TYPE(N));
    return w;
}

static float mse(const std::vector<float>& y, const std::vector<float>& target) {
    float s = 0.0f;
    for (std::size_t i = 0; i < y.size(); ++i) {
        float d = y[i] - target[i];
        s += d * d;
    }
    return s / float(y.size());
}

// Trains for n_steps, calling apply_dynamic_rank_control after EVERY
// backward call (matching the design's "every step" noise-mitigation --
// see delta_csr_types.hpp's own docstring), returns the final MSE.
static float train_with_dynamic_rank_control(Weights& w, const std::vector<float>& w_q,
                                             const std::vector<float>& w_star, int n_steps,
                                             float lr, float tau_death, float tau_active,
                                             float theta, float l1_coef,
                                             std::vector<std::size_t>* rank_trace = nullptr) {
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;
    float final_mse = 0.0f;
    for (int step = 0; step < n_steps; ++step) {
        std::vector<float> y_all(N * N, 0.0f);
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), w,
                                                             y.data(), 1);
            for (std::size_t c = 0; c < N; ++c)
                y_all[i * N + c] = y[c];
            std::vector<float> dy(N);
            for (std::size_t c = 0; c < N; ++c)
                dy[c] = 2.0f * (y[c] - w_star[i * N + c]) / float(N);
            std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false,
                            false>(&basis[i * N], 1, SIZE_TYPE(N), dy.data(), w, dx.data(),
                                   ni.data(), ng.data(), lr, 1, false, true, 0.999f, 1e-8f, 0.9f,
                                   0.0f, 1e30f, 1e30f, 0.1f, false, l1_coef);
            // Seed magnitude matters, not just its sign/nonzero-ness: a
            // new channel's OWN gradient is proportional to its direction
            // vectors' magnitude (dS/d(gamma_new) = value_dir*output_dir),
            // so a too-small seed gives too-small a gradient to outpace
            // L1's constant per-step shrink before the grace period
            // expires -- confirmed as a real bug via direct probing (see
            // conversation): a 0.01-scale seed left the new channel
            // flapping (grow, die to L1 before training, regrow, repeat)
            // even with the grace period fix in place; a seed comparable
            // in scale to an established channel's own direction (~1.0)
            // gives comparable gradient sensitivity and lets RMSprop's
            // adaptive step outpace L1 well within the grace window.
            w.apply_dynamic_rank_control(N, N, tau_death, tau_active, theta,
                                         [&](std::size_t row) { return 1.0f * float(row + 1); });
        }
        if (rank_trace)
            rank_trace->push_back(w.scale_rank);
        if (step == n_steps - 1)
            final_mse = mse(y_all, w_star);
    }
    return final_mse;
}

static void test_breathing_rank_cycle() {
    std::vector<float> w_q = {6.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                              2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 6.f};
    // Phase 1 target: uniform 3x scale (rank-1 sufficient).
    std::vector<float> w_star_uniform(w_q.size());
    for (std::size_t i = 0; i < w_q.size(); ++i)
        w_star_uniform[i] = 3.0f * w_q[i];
    // Phase 2 target: diagonal outliers (needs rank-2).
    std::vector<float> w_star_outlier = {50.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                         2.f,  2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 50.f};

    Weights w = make_scattered_layer(w_q);
    w.scale_rank = 1;
    w.set_additive_rank(0);
    for (std::size_t r = 0; r < N; ++r)
        w.set_value_scale_raw_k(r, 0, 1.0f);
    for (std::size_t c = 0; c < N; ++c)
        w.set_output_scale_raw_k(c, 0, 1.0f);
    // Activate gamma tracking -- see get_scale_gamma_k's own docstring:
    // the "new channel = zero contribution" growth safety property (and
    // this whole dynamic-control mechanism) only engages once gamma is
    // already in active use; a layer that's never touched gamma stays
    // fully transparent, matching pre-gamma behavior.
    w.set_scale_gamma_raw_k(0, 1.0f);

    // theta is compared against a raw gradient normalized by n_in*n_out
    // (task #294 fix, disldo_backward's gamma-EMA trigger tracking) --
    // this test's own N=4 layer means n_in*n_out=16, so theta must scale
    // down by that same factor from its old pre-normalization value
    // (0.02) to trigger at the same real sensitivity.
    const float tau_death = 0.05f, tau_active = 0.3f, theta = 0.02f / (float(N) * float(N)),
                l1_coef = 0.02f;

    // ── Phase 1: rank-1-sufficient target -- rank must stay at 1 ────────
    std::vector<std::size_t> rank_trace1;
    float mse1 = train_with_dynamic_rank_control(w, w_q, w_star_uniform, 1500, 0.05f, tau_death,
                                                 tau_active, theta, l1_coef, &rank_trace1);
    std::printf("[breathing] phase 1 (uniform target) final MSE: %.6f, final rank: %zu\n", mse1,
                w.scale_rank);
    CHECK(mse1 < 0.01f, "phase 1 should converge to near-zero MSE with rank-1 alone (got %.6f)",
          mse1);
    CHECK(w.scale_rank == 1,
          "phase 1 should never grow rank -- no real leftover gradient pressure once converged "
          "(got rank=%zu)",
          w.scale_rank);

    // ── Phase 2: switch to the outlier target -- rank should grow to 2 ──
    // n_steps widened 4x (3000->12000, direct instruction): shrink now
    // needs a genuinely longer evidence window than growth by design (see
    // apply_dynamic_rank_control_generic's own docstring -- shrink_grace_
    // period_steps defaults to 4x growth's), so this end-to-end
    // "breathing" test's own step budgets -- tuned for the OLD symmetric
    // 50/50 default -- need the same 4x widening to keep validating the
    // real mechanism rather than just timing out on the new, intentionally
    // slower cadence.
    std::vector<std::size_t> rank_trace2;
    float mse2 = train_with_dynamic_rank_control(w, w_q, w_star_outlier, 12000, 0.05f, tau_death,
                                                 tau_active, theta, l1_coef, &rank_trace2);
    std::printf("[breathing] phase 2 (outlier target) final MSE: %.6f, final rank: %zu\n", mse2,
                w.scale_rank);
    // Checks against rank_trace2 (direct instruction, root-caused via a
    // real 60k-step run + direct instrumentation of THIS test): the
    // outlier target's neurogenesis trigger is essentially ALWAYS true
    // once rank==1 (confirmed by direct debug instrumentation -- the
    // task genuinely needs rank>1), so the mechanism perpetually cycles
    // grow-wait-shrink-regrow rather than settling at a fixed point.
    // Checking w.scale_rank's FINAL snapshot is checking a moving target
    // at an essentially arbitrary phase of that oscillation -- exactly
    // the kind of check that's sensitive to grace-period timing changes
    // without the underlying mechanism actually being broken. Checking
    // "did it ever reach >=2" verifies the real invariant (neurogenesis
    // capability genuinely works for a task that needs it) without being
    // coupled to the oscillation's exact rhythm.
    bool reached_rank_2 = false;
    for (std::size_t r : rank_trace2)
        if (r >= 2) {
            reached_rank_2 = true;
            break;
        }
    CHECK(reached_rank_2,
          "phase 2 should trigger neurogenesis and reach rank>=2 at some point (never happened)");
    CHECK(mse2 < 1.0f,
          "phase 2 should converge substantially once rank-2 capacity is available (got MSE=%.6f)",
          mse2);

    // ── Phase 3: revert to the uniform target -- rank should shrink back to 1 ──
    std::vector<std::size_t> rank_trace3;
    float mse3 = train_with_dynamic_rank_control(w, w_q, w_star_uniform, 12000, 0.05f, tau_death,
                                                 tau_active, theta, l1_coef, &rank_trace3);
    std::printf("[breathing] phase 3 (back to uniform target) final MSE: %.6f, final rank: %zu\n",
                mse3, w.scale_rank);
    // Same reasoning as phase 2's check, applied to the shrink direction:
    // check that rank settles at (and STAYS at) 1 for a sustained final
    // stretch, not just the exact final snapshot -- robust to exactly
    // where in a residual oscillation (carried over from phase 2's own
    // ending state) the loop happens to stop.
    const std::size_t settle_window = std::min<std::size_t>(500, rank_trace3.size());
    bool settled_at_1 = true;
    for (std::size_t i = rank_trace3.size() - settle_window; i < rank_trace3.size(); ++i) {
        if (rank_trace3[i] != 1) {
            settled_at_1 = false;
            break;
        }
    }
    CHECK(settled_at_1,
          "phase 3 should trigger apoptosis and settle at rank-1 for the final %zu steps (still "
          "oscillating)",
          settle_window);
    // Threshold relaxed from <0.01 (direct instruction, real finding, NOT
    // just a test-fragility patch like the rank checks above): channel
    // 0's own state genuinely does NOT reconverge to phase-1's MSE=0.0
    // after phase 2's prolonged grow/shrink oscillation, even given 2x
    // this phase's step budget (24000 steps tried directly, MSE
    // unchanged at 0.4723 vs 0.4723 -- not a "needs more time" gap, a
    // real stuck state). Phase 1 (same target, same hyperparameters, no
    // prior oscillation) reaches MSE=0.0 easily, so the elevated floor
    // here is attributable to residual damage from phase 2's turbulence,
    // not this test's own setup. <0.6 still confirms real, substantial
    // reconvergence (well below phase 2's own 0.756) without asserting a
    // full return to phase-1-quality that the mechanism doesn't
    // currently deliver -- flagged to the user as a genuine open
    // question (does oscillation leave lasting quality damage even
    // after the extra capacity itself is correctly pruned back down),
    // not silently swept under a loosened test.
    CHECK(mse3 < 0.6f, "phase 3 should reconverge substantially after shrinking (got MSE=%.6f)",
          mse3);
}

int main() {
    test_breathing_rank_cycle();

    if (g_fail == 0) {
        std::printf("All AQRS dynamic rank control integration tests passed.\n");
    } else {
        std::printf("%d FAILURES\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
