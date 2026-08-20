// Long-horizon stability test: proves BoundedRMSpropSynapsePolicy actually
// fixes the late-training divergence PlainRMSpropSynapsePolicy is prone to,
// across a horizon several times longer than where the divergence was
// first found -- not just "the loss looks fine at whatever step count we
// happened to check." Direct motivation (see conversation): a real
// DISLDOLayer32 permutation-regression run (Python, sili_peridot's
// eval_rank_floor.py) stayed rock-stable for 300+ steps, then diverged
// past step ~400 at a CONSTANT lr -- a "looks good, then fails later"
// pattern that a single-snapshot test would silently miss entirely if it
// only ever checked, say, step 50 or step 100.
//
// Same permutation-regression setup as test_synapse_policy_dispatch.cpp
// (8x8 cyclic-shift target, one-hot probes), but run as a REAL multi-step
// training loop (forward -> MSE gradient -> backward, repeated) for many
// times longer than the original 400-step divergence point, checking SSE
// at EVERY step (not just the end) so a late spike can't hide between
// checkpoints.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); std::fflush(stdout); ++g_fail; } \
} while (0)

using SIZE_TYPE = int;
using COL_TYPE  = uint32_t;
using Weights   = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static const std::size_t N = 8;
static const std::size_t N_STEPS = 3000;  // ~7.5x the ~400 steps where the real divergence was found
static const float LR = 0.05f;

// Cyclic shift by 1, scaled to 0.37 instead of 1.0 -- deliberately NOT an
// exact FP4 code (nearest codes are 0.0 and 0.5). A plain 0/1 permutation
// target is fully FP4-representable with unrestricted connectivity, so
// training can reach an EXACT zero-residual global minimum and then sits
// at a permanently-zero gradient forever -- which never triggers the
// resonance this test exists to catch at all (confirmed directly: an
// earlier version of this test used 1.0 and both arms trivially hit
// SSE=0.0000 by step 300, never diverging, a false-negative test that
// wasn't actually exercising the failure mode). 0.37 forces training to
// converge to the NEAREST representable code and then sit at a real,
// persistent, nonzero residual gradient forever -- the exact "converged
// but not exactly" condition the real divergence needs (matches the
// original Python finding too: that used unquantized float32, so the
// resonance mechanism itself doesn't require FP4 at all, just a
// converged point with a persistent small nonzero residual).
static std::vector<float> permutation_target(std::size_t n) {
    std::vector<float> t(n * n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) t[i * n + (i + 1) % n] = 0.37f;
    return t;
}

static Weights make_dense_layer(std::size_t n) {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(n + 1);
    std::vector<SIZE_TYPE> idx(n * n);
    std::vector<float> wv(n * n, 0.0f), imp(n * n, 0.0f);
    for (std::size_t r = 0; r < n; ++r) {
        ptrs[r] = SIZE_TYPE(r * n);
        for (std::size_t c = 0; c < n; ++c) idx[r * n + c] = SIZE_TYPE(c);
    }
    ptrs[n] = SIZE_TYPE(n * n);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, wv, imp, n, n, n * n * 2, n * n * 2);
    w.out_degree.assign(n, SIZE_TYPE(n));
    return w;
}

// Runs N_STEPS of real forward+backward training, recording the WORST
// (max) SSE seen at ANY step from step 200 onward (the original run was
// already stable well past step 200, so any spike after that point is a
// genuine late-onset failure, not early-training noise) and the SSE at
// several evenly-spaced checkpoints across the whole horizon.
template <template <typename> class SynapsePolicyT>
static void run_arm(const char* label, float min_decay_frac, float max_abs_delta,
                     std::vector<float>& checkpoint_sse, float& max_sse_after_200) {
    Weights weights = make_dense_layer(N);
    const std::vector<float> target = permutation_target(N);
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) basis[i * N + i] = 1.0f;

    max_sse_after_200 = 0.0f;
    checkpoint_sse.clear();
    const std::size_t checkpoint_every = N_STEPS / 10;

    for (std::size_t step = 0; step < N_STEPS; ++step) {
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), weights, y.data(), 1);
            std::vector<float> dy(N);
            for (std::size_t c = 0; c < N; ++c) dy[c] = 2.0f * (y[c] - target[i * N + c]);
            std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
            // Explicit beta2=0.999, eps=1e-8, beta1=0.9 (matching the
            // function's own defaults) -- min_decay_frac/max_abs_delta MUST
            // be passed explicitly here: the function's own DEFAULTS
            // (0.0, 1e30) are a documented no-op for BOTH policies (confirmed
            // directly -- an earlier version of this test omitted these and
            // BoundedRMSpropSynapsePolicy diverged identically to Plain,
            // because at the default values it computes the exact same
            // formula as Plain, not because the fix itself doesn't work).
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false, SynapsePolicyT>(
                &basis[i * N], 1, SIZE_TYPE(N), dy.data(), weights, dx.data(), ni.data(), ng.data(), LR, 1,
                false, true, 0.999f, 1e-8f, 0.9f, min_decay_frac, max_abs_delta);
        }

        // SSE over all N probes, every step -- a spike hiding between
        // checkpoints is exactly the failure mode this test exists to catch.
        float sse = 0.0f;
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), weights, y.data(), 1);
            for (std::size_t c = 0; c < N; ++c) {
                const float d = y[c] - target[i * N + c];
                sse += d * d;
            }
        }
        if (step >= 200) max_sse_after_200 = std::max(max_sse_after_200, sse);
        if (step % checkpoint_every == 0) {
            checkpoint_sse.push_back(sse);
            std::printf("[%s] step=%zu sse=%.4f\n", label, step, sse);
        }
    }
}

int main() {
    std::vector<float> plain_checkpoints, bounded_checkpoints;
    float plain_max_after_200 = 0.0f, bounded_max_after_200 = 0.0f;

    // Plain: no-op values (Plain ignores both params anyway). Bounded:
    // min_decay_frac=0.9995 > beta2=0.999 (must exceed beta2 to bind at
    // all, see BoundedRMSpropSynapsePolicy's own docstring), max_abs_delta
    // =1.0 as a reasonable per-step cap relative to this test's LR=0.05.
    run_arm<PlainRMSpropSynapsePolicy>("plain", 0.0f, 1e30f, plain_checkpoints, plain_max_after_200);
    run_arm<BoundedRMSpropSynapsePolicy>("bounded", 0.9995f, 1.0f, bounded_checkpoints, bounded_max_after_200);

    std::printf("plain:   max SSE after step 200 = %.4f\n", plain_max_after_200);
    std::printf("bounded: max SSE after step 200 = %.4f\n", bounded_max_after_200);

    // UPDATED (see conversation): the root cause of Plain's own
    // catastrophic late-onset blowup was traced to disldo_backward's
    // per-synapse ci/cw update being applied SEQUENTIALLY, once per
    // batch row, mutating the weight mid-batch instead of aggregating
    // g/contrib across the whole batch and applying exactly ONE update
    // (matching how every real mini-batch optimizer, including the
    // torch reference this project compares against, actually works).
    // Once that batch-inconsistency was fixed (all 8 real call sites),
    // PlainRMSpropSynapsePolicy -- NO floor, NO clip, the exact
    // no-op-parameters case -- no longer diverges AT ALL over this same
    // 3000-step horizon (confirmed directly: max SSE after step 200 was
    // 0.0004, vs the pre-fix ~285736). So this test's original property
    // ("Bounded must be dramatically better than Plain's catastrophic
    // blowup") is now moot -- there's no blowup for EITHER arm to be
    // better than. The property worth checking now is simply that
    // NEITHER arm exceeds a real convergence-scale bound (the healthy
    // baseline SSE elsewhere in this suite is ~1.0952) by orders of
    // magnitude, i.e. that the fix didn't just relocate the instability.
    CHECK(plain_max_after_200 < 10.0f,
          "PlainRMSpropSynapsePolicy's max SSE (%.4f) is unexpectedly large -- the "
          "batch-consistency fix (see conversation) was expected to keep even the "
          "no-op-parameters case stable on this scenario now",
          plain_max_after_200);
    CHECK(bounded_max_after_200 < 10.0f,
          "BoundedRMSpropSynapsePolicy's max SSE (%.4f) is unexpectedly large",
          bounded_max_after_200);

    // Bounded's own trajectory should PLATEAU, not keep growing -- the last
    // checkpoint shouldn't be dramatically larger than the middle of the
    // run (that pattern IS what Plain's own divergence looks like).
    CHECK(bounded_checkpoints.back() < bounded_checkpoints[bounded_checkpoints.size() / 2] * 3.0f,
          "BoundedRMSpropSynapsePolicy's SSE is still growing at the end of the run (last=%.4f, "
          "mid=%.4f) -- looks like the same unbounded-growth pattern as Plain's divergence, "
          "just slower",
          bounded_checkpoints.back(), bounded_checkpoints[bounded_checkpoints.size() / 2]);

    if (g_fail == 0) {
        std::printf("test_synapse_policy_long_horizon: bounded policy stable across all %zu steps "
                    "(plain's own max-after-200 was %.4f, for reference)\n",
                    N_STEPS, plain_max_after_200);
        return 0;
    }
    std::printf("test_synapse_policy_long_horizon: %d failure(s)\n", g_fail);
    return 1;
}
