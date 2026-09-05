// Direct follow-up to probe_unstable_pocket_growth.cpp (see its own header
// for the original finding): that probe showed the known unsafe pocket
// (raw max_abs_delta=12.0) grows SSE unboundedly for 30000 steps, then
// silently collapses to an all-zero output near the end -- almost
// certainly a ci/cw value overflowing to non-finite and getting masked by
// SOME isfinite guard downstream, not real convergence.
//
// This session's own investigation (see conversation) found two real bugs
// on the way to that masking: (1) linear_disldo.hpp's `contrib` signal
// used the RAW unscaled FP4 code (cw_orig) instead of the fully-scaled
// true weight at all 8 real call sites -- fixed; (2) update_ci/update_cw
// (delta_csr_types.hpp) had NO isfinite guard at all, unlike
// RMSpropScalePolicy's own value_scale/output_scale update (ba4af42) --
// fixed, matching that same pattern (skip the update rather than write
// NaN). Separately, a torch-only ablation (not this file) found that
// max_abs_delta itself (the raw-space hard clip on cw's own update,
// production default 2.0) measurably HURTS learning on a real task, and
// removing it entirely reached perfect accuracy where the clipped version
// plateaued well short.
//
// This probe asks the real question directly, in the real C++ engine
// (not torch): with BOTH bugs above fixed, does running the SAME known-
// divergent scenario with max_abs_delta effectively OFF (1e30, the
// function's own true no-op value) now stay NaN-free over a long
// horizon, or does it still corrupt via some other unguarded path?
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

static void run_arm(const char* label, float max_abs_delta) {
    const float min_decay_frac = 0.0f; // true no-op, matches production default

    Weights weights = make_dense_layer(N);
    const std::vector<float> target = permutation_target(N);
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;

    const std::size_t checkpoint_every = N_STEPS / 30;
    std::vector<float> checkpoints;
    bool went_nonfinite = false;
    std::size_t nonfinite_step = 0;

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

        const bool fine_grained_tail = step >= N_STEPS - 2000 && step % 50 == 0;
        if (step % checkpoint_every == 0 || step == N_STEPS - 1 || fine_grained_tail) {
            float sse = 0.0f;
            float max_abs_y = 0.0f;
            bool any_nonfinite = false;
            for (std::size_t i = 0; i < N; ++i) {
                std::vector<float> y(N, 0.0f);
                disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N),
                                                                 weights, y.data(), 1);
                for (std::size_t c = 0; c < N; ++c) {
                    if (!std::isfinite(y[c]))
                        any_nonfinite = true;
                    max_abs_y = std::max(max_abs_y, std::abs(y[c]));
                    const float d = y[c] - target[i * N + c];
                    sse += d * d;
                }
            }
            checkpoints.push_back(sse);
            std::printf("[%s] step=%6zu sse=%12.4f max_abs_y=%.6f%s\n", label, step, sse, max_abs_y,
                        any_nonfinite ? "  <-- NON-FINITE OUTPUT" : "");
            if (any_nonfinite || !std::isfinite(sse)) {
                went_nonfinite = true;
                nonfinite_step = step;
                std::printf("[%s] NON-FINITE detected at step %zu -- the NaN/Inf guard fix did NOT "
                            "prevent this.\n",
                            label, step);
                break;
            }
        }
    }

    if (!went_nonfinite) {
        const std::size_t n = checkpoints.size();
        const std::size_t last_quarter_start = n - n / 4;
        float first_of_last_quarter = checkpoints[last_quarter_start];
        float last = checkpoints[n - 1];
        float ratio = last / std::max(first_of_last_quarter, 1e-6f);
        std::printf("[%s] FINISHED %zu steps with ZERO non-finite values. "
                    "first-of-last-quarter sse=%.4f final sse=%.4f ratio=%.4f\n",
                    label, N_STEPS, first_of_last_quarter, last, ratio);
    } else {
        std::printf("[%s] STOPPED EARLY at step %zu due to non-finite value.\n", label,
                    nonfinite_step);
    }
    std::printf("\n");
}

int main() {
    std::printf("=== production (max_abs_delta=2.0), sanity check the fix didn't break the "
                "known-good case ===\n");
    run_arm("production", 2.0f);

    std::printf("=== known unsafe pocket, clip STILL ON (max_abs_delta=12.0) -- reproduce the "
                "original finding ===\n");
    run_arm("unsafe_pocket_12", 12.0f);

    std::printf("=== clip EFFECTIVELY OFF (max_abs_delta=1e30) -- the actual question ===\n");
    run_arm("clip_off", 1e30f);

    return 0;
}
