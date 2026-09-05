// Direct instruction (see conversation): ci (the RMSprop second-moment
// accumulator) was confirmed to have NO ceiling anywhere in the pre-
// existing design -- min_decay_frac's floor only bounds how fast ci can
// DECAY, never how fast it can GROW. In the known unsafe pocket
// (probe_unstable_pocket_growth.cpp's own exact scenario: raw
// max_abs_delta=12.0, min_decay_frac=0.9995, deterministic rounding), ci
// was directly measured climbing continuously and unboundedly the entire
// 30000-step run (0.0005 -> 163+, still setting a new max at literally
// every checkpoint, never plateauing), while the healthy production
// default (raw max_abs_delta=2.0, stochastic rounding) plateaus at
// ci~0.5. This test adds BoundedRMSpropSynapsePolicy::update_ci's new
// max_ci ceiling parameter and directly verifies -- in the SAME unsafe
// pocket, same duration -- that ci itself no longer exceeds the chosen
// ceiling, using the same "wrap the real policy, track ci as a side
// effect" technique used ad hoc during investigation (DebugBoundedPolicy):
// update_ci/update_cw are pure functions taking/returning values by
// value, so a thin forwarding wrapper can intercept every real call site
// disldo_backward makes, through the SynapsePolicyT template parameter,
// without needing access to any internal weights-struct state.
//
// RESULT (confirmed directly, this run, max_ci=100.0): the ceiling holds
// EXACTLY -- ci climbs normally, hits 100.0 around step 23000, and never
// exceeds it for the remaining 7000 steps (15.36M total update_ci calls
// checked). This confirms max_ci genuinely stops ci's own growth.
// HOWEVER: capping ci does NOT rescue this unsafe pocket's own SSE
// divergence -- SSE still climbs continuously and smoothly (692-ish
// early on -> 313544 by step 29950, matching probe_unstable_pocket_
// growth.cpp's own uncapped run almost exactly) and STILL collapses to
// the same all-zero-output value (sse=1.0952, max_abs_y=0.0) in the
// final ~50 steps. This means ci overflowing was NOT the (sole) root
// cause of that collapse-to-zero masking behavior after all -- the
// weight/cw accumulator itself has its own SEPARATE unbounded-growth
// mechanism (already documented: max_abs_delta's clip bounds each
// individual step, not the cumulative drift over thousands of
// consistently-signed steps), and capping ci alone does not touch that.
// PRACTICAL IMPLICATION: max_ci is still worth defaulting on (it's a
// real, free structural safety property -- ci can no longer itself be a
// source of eventual float32 overflow, and it's a mathematically
// guaranteed no-op for the actual production-safe zone, where healthy ci
// plateaus at ~0.5, far below any reasonable ceiling) -- but it is NOT a
// fix for out-of-safe-zone max_abs_delta/lr pockets, which remain
// mitigated only by staying inside the validated safe range documented
// in sweep_synapse_policy_stochastic.cpp and the lr-ceiling warning in
// cpu_backend.cpp, not by anything ci-ceiling-related.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static const std::size_t N = 8;
static const std::size_t N_STEPS = 30000;
static const float LR = 0.05f;

// Candidate ceiling: ~200x the healthy production-default plateau
// (~0.5, see conversation), giving generous margin so normal operation
// never touches it, while still being reached well before the unsafe
// pocket's own unchecked trajectory (163+ by step 29999 and still
// climbing) would otherwise carry it.
static const float MAX_CI = 100.0f;

static float g_max_ci_ever = 0.0f;
static std::size_t g_max_ci_step = 0;
static std::size_t g_call_count = 0;

// Forwards to the real BoundedRMSpropSynapsePolicy formula (the actual
// production code, not a re-derivation) while tracking ci's own peak
// value as a side effect. update_cw is passed through unchanged.
template <typename VALUE_TYPE> struct DebugBoundedPolicy {
    static VALUE_TYPE update_ci(VALUE_TYPE ci, VALUE_TYPE g, VALUE_TYPE contrib, VALUE_TYPE beta2,
                                VALUE_TYPE min_decay_frac, VALUE_TYPE max_ci) {
        const VALUE_TYPE result = BoundedRMSpropSynapsePolicy<VALUE_TYPE>::update_ci(
            ci, g, contrib, beta2, min_decay_frac, max_ci);
        ++g_call_count;
        if (result > g_max_ci_ever) {
            g_max_ci_ever = result;
            g_max_ci_step = g_call_count;
        }
        return result;
    }
    static VALUE_TYPE update_cw(VALUE_TYPE g, VALUE_TYPE ci, VALUE_TYPE S, VALUE_TYPE eff_lr,
                                VALUE_TYPE eps, bool damp_by_importance, VALUE_TYPE max_abs_delta,
                                bool scale_invariant = false) {
        return BoundedRMSpropSynapsePolicy<VALUE_TYPE>::update_cw(
            g, ci, S, eff_lr, eps, damp_by_importance, max_abs_delta, scale_invariant);
    }
};

// Block4Vec specialization required by disldo_backward's SIMD path, even
// though this test's N=8 dense layer is small enough it may not always
// hit it -- matches SentinelSynapsePolicy's own precedent
// (test_synapse_policy_dispatch.cpp) of always providing both.
template <> struct DebugBoundedPolicy<Block4Vec> {
    static Block4Vec update_ci(Block4Vec ci, Block4Vec g, Block4Vec contrib, Block4Vec beta2,
                               Block4Vec min_decay_frac, Block4Vec max_ci) {
        const Block4Vec result = BoundedRMSpropSynapsePolicy<Block4Vec>::update_ci(
            ci, g, contrib, beta2, min_decay_frac, max_ci);
        ++g_call_count;
        for (int i = 0; i < SILI_BLOCK4_TILE_SIZE; ++i) {
            if (result[i] > g_max_ci_ever) {
                g_max_ci_ever = result[i];
                g_max_ci_step = g_call_count;
            }
        }
        return result;
    }
    static Block4Vec update_cw(Block4Vec g, Block4Vec ci, Block4Vec S, Block4Vec eff_lr,
                               Block4Vec eps, bool damp_by_importance, Block4Vec max_abs_delta,
                               bool scale_invariant = false) {
        return BoundedRMSpropSynapsePolicy<Block4Vec>::update_cw(
            g, ci, S, eff_lr, eps, damp_by_importance, max_abs_delta, scale_invariant);
    }
};

static std::vector<float> permutation_target(std::size_t n) {
    std::vector<float> t(n * n, 0.0f);
    for (std::size_t i = 0; i < n; ++i)
        t[i * n + (i + 1) % n] = 0.37f;
    return t;
}

static Weights make_dense_layer(std::size_t n) {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(n + 1);
    std::vector<SIZE_TYPE> idx(n * n);
    std::vector<float> wv(n * n, 0.0f), imp(n * n, 0.0f);
    for (std::size_t r = 0; r < n; ++r) {
        ptrs[r] = SIZE_TYPE(r * n);
        for (std::size_t c = 0; c < n; ++c)
            idx[r * n + c] = SIZE_TYPE(c);
    }
    ptrs[n] = SIZE_TYPE(n * n);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, wv, imp, n, n, n * n * 2, n * n * 2);
    w.out_degree.assign(n, SIZE_TYPE(n));
    return w;
}

int main() {
    const float min_decay_frac = 0.9995f; // matches probe_unstable_pocket_growth.cpp exactly
    const float max_abs_delta = 12.0f;    // known unsafe pocket (raw-space)

    Weights weights = make_dense_layer(N);
    const std::vector<float> target = permutation_target(N);
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;

    const std::size_t checkpoint_every = N_STEPS / 30;
    std::vector<float> checkpoints;
    bool saw_non_finite = false;

    for (std::size_t step = 0; step < N_STEPS; ++step) {
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N),
                                                             weights, y.data(), 1);
            std::vector<float> dy(N);
            for (std::size_t c = 0; c < N; ++c)
                dy[c] = 2.0f * (y[c] - target[i * N + c]);
            std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false,
                            false, DebugBoundedPolicy>(
                &basis[i * N], 1, SIZE_TYPE(N), dy.data(), weights, dx.data(), ni.data(), ng.data(),
                LR, 1, false, true, 0.999f, 1e-8f, 0.9f, min_decay_frac, max_abs_delta, MAX_CI);
        }

        const bool fine_grained_tail = step >= N_STEPS - 2000 && step % 50 == 0;
        if (step % checkpoint_every == 0 || step == N_STEPS - 1 || fine_grained_tail) {
            float sse = 0.0f;
            float max_abs_y = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                std::vector<float> y(N, 0.0f);
                disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N),
                                                                 weights, y.data(), 1);
                for (std::size_t c = 0; c < N; ++c) {
                    max_abs_y = std::max(max_abs_y, std::abs(y[c]));
                    const float d = y[c] - target[i * N + c];
                    sse += d * d;
                }
            }
            checkpoints.push_back(sse);
            std::printf("step=%6zu sse=%12.4f max_abs_y=%.6f max_ci_so_far=%.6f (at call %zu)\n",
                        step, sse, max_abs_y, g_max_ci_ever, g_max_ci_step);
            if (!std::isfinite(sse) || !std::isfinite(g_max_ci_ever)) {
                saw_non_finite = true;
            }
        }
    }

    std::printf("\nFINAL: max_ci ever observed = %.6f (ceiling = %.6f), over %zu update_ci calls\n",
                g_max_ci_ever, MAX_CI, g_call_count);

    int fail = 0;
    if (g_max_ci_ever > MAX_CI + 1e-3f) {
        std::printf("FAIL: ci exceeded max_ci -- ceiling did not hold (%.6f > %.6f)\n",
                    g_max_ci_ever, MAX_CI);
        ++fail;
    } else {
        std::printf("PASS: ci never exceeded max_ci across the full %zu-step unsafe-pocket run.\n",
                    N_STEPS);
    }
    if (saw_non_finite) {
        std::printf("FAIL: saw a non-finite SSE or ci value during the run.\n");
        ++fail;
    }
    // This test's job is specifically to confirm ci itself stays bounded
    // -- it does NOT assert SSE/weight-level convergence, since capping ci
    // is an independent fix from max_abs_delta's own per-step clip (which
    // already bounds the weight-level update directly) and does not by
    // itself guarantee this particular unsafe pocket produces good
    // training outcomes. See stdout log above for the actual SSE
    // trajectory under the ceiling.

    return fail == 0 ? 0 : 1;
}
