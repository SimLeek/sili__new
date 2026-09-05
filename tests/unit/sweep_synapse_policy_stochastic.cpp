// STOCHASTIC-rounding tuning sweep for BoundedRMSpropSynapsePolicy, mirroring
// sweep_synapse_policy_min_decay_frac.cpp's own investigation but with
// StochasticRounding=true instead of false.
//
// MOTIVATION (see conversation): the deterministic sweep's own "safe zone"
// (max_abs_delta=2.0 raw-space, min_decay_frac=no-op) was picked entirely
// under DETERMINISTIC rounding -- and deterministic rounding turned out to
// get PERMANENTLY FROZEN AT ZERO OUTPUT at that exact config (and even
// Plain/unclipped deterministic collapses to the same frozen state by
// ~step 100), a failure mode the deterministic sweep's own SSE-only metric
// couldn't distinguish from genuine convergence (both give the same
// "all-zero-output" SSE for this target). Production uses STOCHASTIC
// rounding, which was directly confirmed (see conversation) to escape this
// frozen state and converge cleanly at the SAME max_abs_delta=2.0 -- but
// that was one point, not a real sweep, and deterministic's own unsafe
// pockets / min_decay_frac irrelevance may not transfer to stochastic's
// completely different dynamics (dithering noise fundamentally changes
// this). Building a real stochastic-specific map here rather than
// assuming the deterministic one applies.
//
// Uses max_abs_y (not just SSE) in the safety classification, unlike the
// deterministic sweep's original design -- learned the hard way that SSE
// alone can't distinguish "converged" from "frozen/masked" for this target.
//
// ═══════════════════════════════════════════════════════════════════════
// FINDINGS (see conversation for full detail)
// ═══════════════════════════════════════════════════════════════════════
//
// Round 1 (lr=0.05 fixed, matching the deterministic sweep): stochastic
// Plain still diverges catastrophically when unclipped (max_after_200~
// 615061) -- the clip is still necessary under stochastic rounding, it's
// not a free fix. The chosen production default (max_abs_delta=2.0
// raw-space) is confirmed deep in a genuinely safe, well-converging zone
// regardless of min_decay_frac. Degradation past ~max_abs_delta=8-12 is
// gradual/monotonic under stochastic, NOT the sharp scattered "unsafe
// pockets" deterministic rounding showed -- dithering noise appears to
// smooth out whatever exact-value resonance caused those.
//
// Unlike deterministic (where min_decay_frac provably did NOTHING, see
// sweep_synapse_policy_min_decay_frac.cpp), under stochastic rounding
// min_decay_frac IS measurably non-inert at the riskier end of the range
// (e.g. max_abs_delta=16: min_decay_frac=0.9995 clearly beats 0.999,
// 3260 vs 6207 max_after_200) -- single-seed result, treat as a real but
// unconfirmed lead, not yet multi-seed-verified.
//
// Round 2 (does min_decay_frac extend safety to HIGHER lr, not just
// lr=0.05?): NO -- min_decay_frac's benefit (present at lr=0.02-0.05)
// essentially vanishes by lr=0.2 and is bit-identical between the two
// tested values by lr=1.0. min_decay_frac is not a lever for extending
// the safe lr range.
//
// More importantly: the production default's OWN safe lr range is
// bounded, not unlimited. Because the lr-independent redesign
// (delta_csr_types.hpp's update_cw) makes the EFFECTIVE clip scale as
// lr*max_abs_delta_raw, a large enough lr eventually pushes the effective
// clip back into the same large-final-space-step territory that was
// always risky -- the redesign fixes "clip silently shrinks the wrong way
// at large lr" (the original FoldedColumnLayer bug), it does NOT make one
// fixed raw value safe for unlimited lr, since the underlying resonance
// risk is about the ABSOLUTE step size, not the raw/lr split. Measured
// (max_abs_delta=2.0, lr swept): excellent at lr<=0.05 (max_after_200
// ~0.7-1.1), visibly degraded but still nominally "safe" (relative to
// Plain's own even-worse blowup) by lr=0.2 (~18-20), diverging in
// absolute terms by lr=0.5 (~673741), genuinely flagged unsafe by lr=1.0
// (~26045158). PRACTICAL CEILING: treat lr > ~0.2 with the default
// max_abs_delta=2.0 as outside the validated-safe range -- a caller
// needing a genuinely larger lr should get its own max_abs_delta tuned
// for that regime, not rely on this default.
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

struct RunResult {
    float max_after_200;
    float final_sse;
    float mean_last_10pct;
    float max_abs_y_final;
    bool ever_frozen;
};

template <template <typename> class SynapsePolicyT>
static RunResult run_arm(float min_decay_frac, float max_abs_delta, unsigned seed_offset,
                         float lr = LR) {
    Weights weights = make_dense_layer(N);
    const std::vector<float> target = permutation_target(N);
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;

    float max_after_200 = 0.0f;
    std::vector<float> last_10pct_sse;
    const std::size_t last_10pct_start = N_STEPS - N_STEPS / 10;
    float final_sse = 0.0f, max_abs_y_final = 0.0f;
    bool ever_frozen_after_warmup =
        true; // starts true, cleared the first time we see real movement past step 500

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
                            true, SynapsePolicyT>(
                &basis[i * N], 1, SIZE_TYPE(N), dy.data(), weights, dx.data(), ni.data(), ng.data(),
                lr, 1, false, true, 0.999f, 1e-8f, 0.9f, min_decay_frac, max_abs_delta);
        }
        float sse = 0.0f, max_abs_y = 0.0f;
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
        if (step >= 500 && max_abs_y > 1e-6f)
            ever_frozen_after_warmup = false;
        if (step >= 200)
            max_after_200 = std::max(max_after_200, sse);
        if (step >= last_10pct_start)
            last_10pct_sse.push_back(sse);
        if (step == N_STEPS - 1) {
            final_sse = sse;
            max_abs_y_final = max_abs_y;
        }
    }
    float mean_last = 0.0f;
    for (float v : last_10pct_sse)
        mean_last += v;
    mean_last /= float(last_10pct_sse.size());
    return {max_after_200, final_sse, mean_last, max_abs_y_final, ever_frozen_after_warmup};
}

int main() {
    // Stochastic Plain (no clip, no floor) reference -- same role as the
    // deterministic sweep's own "plain" baseline.
    RunResult plain = run_arm<PlainRMSpropSynapsePolicy>(0.0f, 1e30f, 0);
    std::printf("stochastic plain: max_after_200=%.4f final=%.4f mean_last_10pct=%.4f "
                "max_abs_y_final=%.4f frozen=%s\n",
                plain.max_after_200, plain.final_sse, plain.mean_last_10pct, plain.max_abs_y_final,
                plain.ever_frozen ? "YES" : "no");

    // Round 1: broad max_abs_delta grid (raw-space) x min_decay_frac
    // (no-op vs a nonzero-effect value), Bounded policy, stochastic.
    const float min_decay_fracs[] = {0.999f, 0.9995f};
    const float max_abs_deltas[] = {0.2f,  0.5f,  1.0f,  2.0f,  4.0f,  8.0f,
                                    12.0f, 16.0f, 20.0f, 24.0f, 32.0f, 1e30f};

    std::printf("%12s %12s %14s %10s %16s %14s %8s\n", "min_decay", "max_delta", "max_after_200",
                "final", "mean_last_10pct", "max_abs_y_fin", "frozen");
    for (float mdf : min_decay_fracs) {
        for (float mad : max_abs_deltas) {
            RunResult r = run_arm<BoundedRMSpropSynapsePolicy>(mdf, mad, 0);
            bool safe =
                r.max_after_200 < plain.max_after_200 / 100.0f || plain.max_after_200 < 1e-3f;
            std::printf("%12.4f %12.4f %14.4f %10.4f %16.4f %14.4f %8s %s\n", mdf, mad,
                        r.max_after_200, r.final_sse, r.mean_last_10pct, r.max_abs_y_final,
                        r.ever_frozen ? "YES" : "no", safe ? "" : "(unsafe-vs-plain)");
        }
    }

    // Round 2: does min_decay_frac give MORE stability across a range of
    // lr values, not just the one lr=0.05 everything above used? Direct
    // instruction (see conversation). max_abs_delta fixed at 2.0 (the
    // actual production default -- already deep-safe at lr=0.05, checking
    // whether that holds at other lr) and 8.0/16.0 (the riskier points
    // where min_decay_frac showed a real, non-no-op difference at
    // lr=0.05) -- if min_decay_frac's benefit is real and lr-related, it
    // should show up more clearly at some lr values than others.
    const float lrs[] = {0.02f, 0.05f, 0.2f, 0.5f, 1.0f};
    const float mads_for_lr_sweep[] = {2.0f, 8.0f, 16.0f};
    std::printf("\n=== Round 2: min_decay_frac across multiple lr values ===\n");
    std::printf("%8s %10s %12s %14s %10s %16s %8s\n", "lr", "max_delta", "min_decay",
                "max_after_200", "final", "mean_last_10pct", "frozen");
    for (float lr : lrs) {
        RunResult plain_lr = run_arm<PlainRMSpropSynapsePolicy>(0.0f, 1e30f, 0, lr);
        std::printf("%8.3f %10s %12s %14.4f %10.4f %16.4f %8s   (plain reference)\n", lr, "-", "-",
                    plain_lr.max_after_200, plain_lr.final_sse, plain_lr.mean_last_10pct,
                    plain_lr.ever_frozen ? "YES" : "no");
        for (float mad : mads_for_lr_sweep) {
            for (float mdf : min_decay_fracs) {
                RunResult r = run_arm<BoundedRMSpropSynapsePolicy>(mdf, mad, 0, lr);
                bool safe = r.max_after_200 < plain_lr.max_after_200 / 100.0f ||
                            plain_lr.max_after_200 < 1e-3f;
                std::printf("%8.3f %10.4f %12.4f %14.4f %10.4f %16.4f %8s %s\n", lr, mad, mdf,
                            r.max_after_200, r.final_sse, r.mean_last_10pct,
                            r.ever_frozen ? "YES" : "no", safe ? "" : "(unsafe-vs-plain)");
            }
        }
    }
    return 0;
}
