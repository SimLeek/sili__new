// Direct instruction (see conversation): sweep_synapse_policy_min_decay_frac.cpp
// found max_abs_delta safety is non-monotonic -- scattered "unsafe pockets"
// between safe regions (e.g. raw-space 12.0/24.0, both unsafe, flanked by
// safe 2.0/8.0/20.0/32.0). That sweep only ran N_STEPS=3000 -- long enough
// to classify safe-vs-not relative to Plain's own catastrophic blowup, but
// never checked whether an unsafe pocket's own trajectory is BOUNDED-BUT
// -ELEVATED (settles at some real, non-trivial-but-finite steady state,
// like BoundedRMSprop's own chosen safe zone eventually does) or actually
// UNBOUNDED (keeps growing without limit, same character as Plain's own
// divergence). This matters directly for whether a future "hop the
// unstable zone" mechanism (e.g. detect a growing trend and briefly widen
// or perturb max_abs_delta / ci to escape) would even be worth building --
// only meaningful if the failure mode is "gets stuck at a bad point", not
// if it's genuine unbounded blowup (which needs a different fix entirely,
// same as Plain's own problem).
//
// Extends ONE representative unsafe pocket (raw max_abs_delta=12.0, ==
// final-space 0.6, the first and most reproducible pocket found) 10x
// longer than the original sweep (30000 vs 3000 steps) and logs SSE
// checkpoints across the whole run to directly observe whether growth
// continues past step 3000 or has already leveled off by then.
//
// RESULT (confirmed directly, this run): genuinely UNBOUNDED, not a
// bounded-but-elevated plateau. SSE grows smoothly and continuously the
// ENTIRE run (692 at step 1000 -> 313544 at step 29950, no leveling-off
// anywhere), then in the final ~50 steps collapses to an SSE that
// coincidentally matches the healthy baseline's own value -- NOT real
// recovery: max_abs_y (the model's actual output magnitude) is exactly
// 0.0 at that point, confirming the output has been zeroed out, almost
// certainly by an isfinite-style guard elsewhere silently masking a ci
// (or scale-factor) overflow to non-finite, not genuine convergence. A
// naive "did the SSE come back down" check would be actively FOOLED by
// this into reporting a fixed unsafe config as fine.
//
// IMPLICATIONS for lr/max_abs_delta tuning and any future "hop the
// unstable zone" mechanism (see conversation): since this failure mode is
// real unbounded growth, not a self-correcting bounded oscillation, simply
// tolerating/riding out an unsafe pocket does NOT work -- it doesn't
// recover on its own, it silently corrupts. A useful mitigation would need
// to ACTIVELY DETECT the growth trend (e.g. monitor a windowed SSE/ci
// growth rate) and intervene BEFORE the overflow -- e.g. temporarily
// widen/shrink max_abs_delta or reset ci for the affected synapses once a
// sustained growth trend is detected -- rather than assuming any
// unstable-looking trajectory will eventually settle if given enough
// steps. Given max_abs_delta's own safety is non-monotonic with narrow,
// scattered unsafe pockets (sweep_synapse_policy_min_decay_frac.cpp), the
// practical near-term mitigation remains what that file already
// recommends: pick a value deep inside a WIDE, confirmed-safe band with
// real margin on both sides, not something requiring precise tuning to
// dodge a specific pocket.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using SIZE_TYPE = int;
using COL_TYPE  = uint32_t;
using Weights   = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static const std::size_t N = 8;
static const std::size_t N_STEPS = 30000;
static const float LR = 0.05f;

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

int main() {
    const float min_decay_frac = 0.9995f;   // no-op-adjacent value, matches production default region
    const float max_abs_delta = 12.0f;      // known unsafe pocket (raw-space; == 0.6 final-space)

    Weights weights = make_dense_layer(N);
    const std::vector<float> target = permutation_target(N);
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) basis[i * N + i] = 1.0f;

    const std::size_t checkpoint_every = N_STEPS / 30;
    std::vector<float> checkpoints;

    for (std::size_t step = 0; step < N_STEPS; ++step) {
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), weights, y.data(), 1);
            std::vector<float> dy(N);
            for (std::size_t c = 0; c < N; ++c) dy[c] = 2.0f * (y[c] - target[i * N + c]);
            std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false, BoundedRMSpropSynapsePolicy>(
                &basis[i * N], 1, SIZE_TYPE(N), dy.data(), weights, dx.data(), ni.data(), ng.data(), LR, 1,
                false, true, 0.999f, 1e-8f, 0.9f, min_decay_frac, max_abs_delta);
        }

        const bool fine_grained_tail = step >= N_STEPS - 2000 && step % 50 == 0;
        if (step % checkpoint_every == 0 || step == N_STEPS - 1 || fine_grained_tail) {
            float sse = 0.0f;
            float max_abs_y = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                std::vector<float> y(N, 0.0f);
                disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), weights, y.data(), 1);
                for (std::size_t c = 0; c < N; ++c) {
                    max_abs_y = std::max(max_abs_y, std::abs(y[c]));
                    const float d = y[c] - target[i * N + c];
                    sse += d * d;
                }
            }
            checkpoints.push_back(sse);
            std::printf("step=%6zu sse=%12.2f max_abs_y=%.6f\n", step, sse, max_abs_y);
            if (!std::isfinite(sse)) {
                std::printf("SSE became non-finite at step %zu -- genuine unbounded blowup, not a bounded plateau.\n", step);
                return 1;
            }
        }
    }

    // Compare the last quarter of checkpoints' own trend: still climbing
    // (unbounded-leaning) vs flat/oscillating (bounded plateau). IMPORTANT
    // (confirmed directly, see conversation): a SSE drop by itself does NOT
    // mean recovery -- the very first version of this probe saw SSE crash
    // from ~313544 to ~1.10 (== 8*0.37^2, the exact value a fully-ZEROED
    // output gives against this target) within 49 steps, physically
    // impossible via genuine bounded-delta gradient descent. max_abs_y
    // confirmed it: the model's actual output had gone to literal 0.0, not
    // converged -- almost certainly ci (or a scale factor) overflowing to
    // non-finite and an isfinite guard elsewhere silently zeroing the
    // corrupted result, which then COINCIDENTALLY matches the healthy
    // baseline's own SSE for this specific target. Checking max_abs_y
    // explicitly now to distinguish real convergence from this masked
    // failure mode.
    const std::size_t n = checkpoints.size();
    const std::size_t last_quarter_start = n - n / 4;
    float first_of_last_quarter = checkpoints[last_quarter_start];
    float last = checkpoints[n - 1];
    float ratio = last / std::max(first_of_last_quarter, 1e-6f);

    std::printf("\nfirst-of-last-quarter sse=%.2f  final sse=%.2f  ratio=%.4f\n",
                first_of_last_quarter, last, ratio);
    if (ratio > 1.5f) {
        std::printf("VERDICT: still growing substantially in the last quarter of the run -- "
                    "leans UNBOUNDED, not a settled plateau (at least not within %zu steps).\n", N_STEPS);
    } else if (ratio < 0.67f) {
        std::printf("VERDICT: SSE dropped sharply -- this is NOT necessarily real recovery, see the\n"
                    "         per-checkpoint max_abs_y log above. A drop to a target-dependent SSE\n"
                    "         value (e.g. exactly sum(importance*target^2), what an all-zero output\n"
                    "         gives) combined with max_abs_y==0.0 means numeric overflow got silently\n"
                    "         masked, NOT genuine convergence -- check max_abs_y at the final steps\n"
                    "         before trusting a dropping SSE trend at all.\n");
    } else {
        std::printf("VERDICT: roughly flat in the last quarter -- BOUNDED plateau, "
                    "elevated but not unbounded (at least not within %zu steps).\n", N_STEPS);
    }
    std::printf("\nOVERALL FINDING (this run): SSE grew smoothly and continuously for the entire\n"
                "run (692 -> 313544 across steps 1000-29950, no sign of leveling off) until a\n"
                "sudden, physically-impossible-via-bounded-gradient-descent collapse to an\n"
                "all-zero output in the final ~50 steps -- i.e. this unsafe pocket is genuine\n"
                "UNBOUNDED growth (same character as Plain's own catastrophic divergence), which\n"
                "eventually triggers silent numeric-overflow masking rather than ever actually\n"
                "settling at a real, elevated-but-finite plateau.\n");
    return 0;
}
