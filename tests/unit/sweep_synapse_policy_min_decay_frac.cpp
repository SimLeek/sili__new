// Tuning sweep for BoundedRMSpropSynapsePolicy's min_decay_frac/max_abs_delta
// (task #216, sili_peridot persistent task list) -- BEFORE making Bounded the
// real default everywhere, find parameters that (a) still prevent Plain's
// catastrophic late-training divergence and (b) minimize the steady-state
// SSE cost the original min_decay_frac=0.9995/max_abs_delta=1.0 choice paid
// (that pair was only ever picked to DEMONSTRATE the fix works in
// test_synapse_policy_long_horizon.cpp, never tuned for quality).
//
// Same permutation-regression long-horizon setup as
// test_synapse_policy_long_horizon.cpp (8x8 cyclic-shift target scaled to
// 0.37, N_STEPS=3000) -- reusing the identical helper functions rather than
// sharing a header, matching this codebase's own established precedent for
// standalone (own main()) test files (test_synapse_policy_dispatch.cpp and
// test_synapse_policy_long_horizon.cpp already duplicate similar structure
// rather than factoring out a shared header).
//
// ═══════════════════════════════════════════════════════════════════════
// FINDINGS (three sweep rounds, see conversation) -- CONCLUSIONS BELOW ARE
// SPECIFIC TO THIS ONE FAILURE MODE (late-training RMSprop resonance on an
// 8x8 permutation-regression target); re-verify before assuming they
// generalize to a materially different task/scale.
// ═══════════════════════════════════════════════════════════════════════
//
// Round 1 (min_decay_frac in {0.9993,0.9995,0.9997,0.9999} x max_abs_delta
// in {0.5,1.0,2.0,5.0}): plain baseline max_after_200=285736.78,
// final=285556.91. EVERY min_decay_frac gave bit-identical results at each
// max_abs_delta -- e.g. at max_abs_delta=1.0, all four min_decay_frac
// values gave max_after_200=109.4301, final=1.0952 exactly. max_abs_delta
// alone determined safe (<plain_max/100) vs not:
//     max_delta=0.5   max_after_200=27.3556    safe
//     max_delta=1.0   max_after_200=109.4301   safe
//     max_delta=2.0   max_after_200=11352.46   NOT safe
//     max_delta=5.0   max_after_200=118679.31  NOT safe
//
// Round 2: narrowed max_abs_delta (0.05-1.0) and added min_decay_frac=0.999
// (== beta2 exactly, the documented no-op threshold) alongside 0.9995/0.9999
// -- STILL bit-identical results at every point. Also found max_abs_delta
// safety is NOT monotonic:
//     max_delta=0.01-0.20   max_after_200=1.0952 (flat, best quality)  safe
//     max_delta=0.30        max_after_200=2.4660                      safe
//     max_delta=0.50        max_after_200=27.3556                     safe
//     max_delta=0.75        max_after_200=7325.85                     NOT safe
//     max_delta=1.00        max_after_200=109.4301                    safe
//
// Round 3: mapped the anomaly further -- max_delta=0.60/0.65/0.70 ALL
// unsafe (a real band, not a fluke single point), 0.80/0.85/0.90 safe again,
// then ANOTHER unsafe pocket at max_delta=1.20 (max_after_200=7453.07),
// then safe again at 1.50. So safety-vs-max_abs_delta is NOT monotonic at
// all -- scattered unsafe pockets between safe regions. Working hypothesis
// (not independently confirmed, but consistent with everything observed):
// this is a resonance/aliasing effect between the fixed clip magnitude and
// the FP4 quantization grid spacing -- certain clip values happen to align
// with a harmonic of the underlying divergence oscillation and let it
// re-establish itself despite the clip, while most values don't. This is
// exactly why the chosen default should sit deep inside a WIDE safe band
// with margin on both sides, not just barely under the nearest observed
// unsafe point -- other untested clip values could easily hit a different
// unmapped pocket.
//
// Round 4 (isolation): tested min_decay_frac alone with max_abs_delta=1e30
// (clip effectively OFF) to check whether the ci-floor has ANY independent
// protective power, not just "redundant given the clip already helps."
// Result: even min_decay_frac=0.99995 (far above anything tested in rounds
// 1-3) gave max_after_200=285777-285793 -- statistically indistinguishable
// from Plain's own 285736.78. The ci floor provides ZERO measurable
// protection on its own in this failure mode, at any tested strength.
//
// CONCLUSION / chosen defaults for this failure mode (rounds 1-4, all
// numbers in FINAL-DELTA space, i.e. update_cw's OLD design that clipped
// AFTER multiplying by eff_lr):
//   max_abs_delta = 0.1   -- deep inside the flat, best-quality, safe band
//                            (0.01-0.20 all identical, max margin from the
//                            nearest unsafe pocket at 0.60).
//   min_decay_frac = true no-op (<= beta2, i.e. the ci floor never binds)
//                            -- zero evidence of any independent benefit
//                            found across 4 rounds including an isolated
//                            no-clip test; keeping it "on" at a value with
//                            zero demonstrated upside just adds an unproven
//                            mechanism and an extra untested interaction
//                            surface with max_abs_delta for no known gain.
//
// ═══════════════════════════════════════════════════════════════════════
// REDESIGN (see conversation): the chosen max_abs_delta=0.1 above turned
// out to only be valid AT lr=0.05 -- update_cw originally clipped the
// ALREADY-lr-scaled delta, so a caller using a much bigger lr (e.g.
// FoldedColumnLayer's lr=1.0) got every real step silently crushed down to
// the same flat 0.1 cap regardless of how much bigger lr was set,
// effectively capping lr's own meaning. Fixed by moving the clip BEFORE
// the eff_lr multiply (update_cw now clips the lr-INDEPENDENT raw update,
// then multiplies by eff_lr) -- standard gradient-clipping convention
// (clip before scaling by lr, matching this project's own clip_grad_norm_
// in Python), and it makes the EFFECTIVE final-delta clip naturally
// proportional to eff_lr without a separate multiply.
//
// Since round 1-4's own lr was fixed at 0.05 throughout, and eff_lr is a
// constant multiplier during any single run, this reordering is a pure
// algebraic identity there -- the OLD final-space value divided by 0.05
// gives the NEW raw-space value that reproduces bit-identical behavior:
// max_abs_delta = 0.1 / 0.05 = 2.0 (raw-space). All rounds 1-4's
// documented numbers above translate the same way (divide by 0.05) if you
// need to cross-reference them against a raw-space rerun.
// ═══════════════════════════════════════════════════════════════════════
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static const std::size_t N = 8;
static const std::size_t N_STEPS = 3000;
static const float LR = 0.05f;

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

// Returns (max_sse_after_200, final_sse, mean_sse_last_10pct) for one
// (min_decay_frac, max_abs_delta) configuration of BoundedRMSpropSynapsePolicy.
struct RunResult {
    float max_after_200;
    float final_sse;
    float mean_last_10pct;
};

static RunResult run_bounded(float min_decay_frac, float max_abs_delta) {
    Weights weights = make_dense_layer(N);
    const std::vector<float> target = permutation_target(N);
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;

    float max_after_200 = 0.0f;
    std::vector<float> last_10pct_sse;
    const std::size_t last_10pct_start = N_STEPS - N_STEPS / 10;
    float final_sse = 0.0f;

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
                            false, BoundedRMSpropSynapsePolicy>(
                &basis[i * N], 1, SIZE_TYPE(N), dy.data(), weights, dx.data(), ni.data(), ng.data(),
                LR, 1, false, true, 0.999f, 1e-8f, 0.9f, min_decay_frac, max_abs_delta);
        }

        float sse = 0.0f;
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N),
                                                             weights, y.data(), 1);
            for (std::size_t c = 0; c < N; ++c) {
                const float d = y[c] - target[i * N + c];
                sse += d * d;
            }
        }
        if (step >= 200)
            max_after_200 = std::max(max_after_200, sse);
        if (step >= last_10pct_start)
            last_10pct_sse.push_back(sse);
        if (step == N_STEPS - 1)
            final_sse = sse;
    }

    float mean_last = 0.0f;
    for (float v : last_10pct_sse)
        mean_last += v;
    mean_last /= float(last_10pct_sse.size());

    return {max_after_200, final_sse, mean_last};
}

static RunResult run_plain() {
    Weights weights = make_dense_layer(N);
    const std::vector<float> target = permutation_target(N);
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;

    float max_after_200 = 0.0f;
    std::vector<float> last_10pct_sse;
    const std::size_t last_10pct_start = N_STEPS - N_STEPS / 10;
    float final_sse = 0.0f;

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
                            false, PlainRMSpropSynapsePolicy>(
                &basis[i * N], 1, SIZE_TYPE(N), dy.data(), weights, dx.data(), ni.data(), ng.data(),
                LR, 1, false, true, 0.999f, 1e-8f, 0.9f, 0.0f, 1e30f);
        }
        float sse = 0.0f;
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N),
                                                             weights, y.data(), 1);
            for (std::size_t c = 0; c < N; ++c) {
                const float d = y[c] - target[i * N + c];
                sse += d * d;
            }
        }
        if (step >= 200)
            max_after_200 = std::max(max_after_200, sse);
        if (step >= last_10pct_start)
            last_10pct_sse.push_back(sse);
        if (step == N_STEPS - 1)
            final_sse = sse;
    }
    float mean_last = 0.0f;
    for (float v : last_10pct_sse)
        mean_last += v;
    mean_last /= float(last_10pct_sse.size());
    return {max_after_200, final_sse, mean_last};
}

int main() {
    RunResult plain = run_plain();
    std::printf("plain: max_after_200=%.4f final=%.4f mean_last_10pct=%.4f\n", plain.max_after_200,
                plain.final_sse, plain.mean_last_10pct);

    // Final sanity grid, in RAW-SPACE units (post-redesign, see the
    // REDESIGN section above) -- values here are the old final-space grid
    // (0.1, 0.6, 1.0) divided by this file's own fixed LR=0.05, so this
    // should reproduce IDENTICAL max_after_200/final/mean_last_10pct
    // numbers to the pre-redesign run: exercises the chosen production
    // default (max_abs_delta=2.0 raw == 0.1 final), a known unsafe pocket
    // (12.0 raw == 0.6 final, should still be caught), and confirms
    // min_decay_frac's no-op value (0.999) vs a nonzero-effect value
    // (0.9995) still agree (see the four rounds documented above).
    // NOTE (confirmed directly): 2.0 (the chosen production default)
    // reproduces the pre-redesign result bit-identically, as does 12.0 (the
    // known unsafe pocket). 20.0 does NOT reproduce the pre-redesign
    // max_abs_delta=1.0 result (was safe at 109.43, now unsafe at 6626.75)
    // -- old and new formulas are equal in exact real arithmetic
    // (clip(lr*raw,M) == lr*clip(raw,M/lr) for lr>0), but float32 rounds
    // differently depending on operation order, and this system is already
    // established (rounds 1-3 above) to be chaotically sensitive near
    // resonance boundaries. max_abs_delta=1.0 (final-space) was only
    // marginally safe to begin with (109.43, nowhere near the deep flat
    // zone's ~1.0 baseline) -- exactly the kind of borderline point this
    // sensitivity would show up at first. The CHOSEN default sits deep in a
    // stable zone specifically to avoid this; it isn't affected.
    const float min_decay_fracs[] = {0.999f, 0.9995f};
    const float max_abs_deltas[] = {2.0f, 12.0f, 20.0f};

    std::printf("%12s %12s %14s %10s %16s %10s\n", "min_decay", "max_delta", "max_after_200",
                "final", "mean_last_10pct", "safe");
    for (float mdf : min_decay_fracs) {
        for (float mad : max_abs_deltas) {
            RunResult r = run_bounded(mdf, mad);
            bool safe = r.max_after_200 < plain.max_after_200 / 100.0f;
            std::printf("%12.4f %12.4f %14.4f %10.4f %16.4f %10s\n", mdf, mad, r.max_after_200,
                        r.final_sse, r.mean_last_10pct, safe ? "yes" : "NO");
        }
    }
    return 0;
}
