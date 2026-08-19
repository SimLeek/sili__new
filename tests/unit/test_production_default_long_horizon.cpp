// Direct gap identified by the user (see conversation): all the
// unsafe-pocket stress tests (probe_unstable_pocket_growth.cpp,
// test_ci_ceiling.cpp) run 30000 steps to confirm a KNOWN-BAD config
// really is unbounded, not just elevated -- but the actual REAL
// production default (the exact values cpu_backend.cpp's 3 real
// disldo_backward call sites use) had only ever been confirmed clean at
// this duration via a throwaway /tmp probe during investigation, never
// committed as a permanent regression test. The two committed
// long-horizon-style tests that DO exist only cover 3000 steps
// (test_synapse_policy_long_horizon.cpp, sweep_synapse_policy_stochastic
// .cpp), and the former additionally uses an older max_abs_delta=1.0
// under deterministic rounding, not today's real shipped values. This
// test closes that gap: exact production config, stochastic rounding,
// 30000 steps (matching the unsafe-pocket tests' own duration), checking
// SSE at every step for the entire run.
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
static const std::size_t N_STEPS = 30000;
static const float LR = 0.05f;

// Exact values cpu_backend.cpp's 3 real disldo_backward call sites use,
// as of this test's writing: beta1=0.9, min_decay_frac=0.0 (true no-op --
// this was the CHOSEN PRODUCTION DEFAULT, see delta_csr_types.hpp's own
// update_ci docstring), max_abs_delta=2.0 (raw-space), max_ci=100.0.
static const float BETA1 = 0.9f;
static const float MIN_DECAY_FRAC = 0.0f;
static const float MAX_ABS_DELTA = 2.0f;
static const float MAX_CI = 100.0f;

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
    Weights weights = make_dense_layer(N);
    const std::vector<float> target = permutation_target(N);
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) basis[i * N + i] = 1.0f;

    const std::size_t checkpoint_every = N_STEPS / 30;
    std::vector<float> checkpoints;
    float max_sse_after_200 = 0.0f;
    bool saw_non_finite = false;

    for (std::size_t step = 0; step < N_STEPS; ++step) {
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), weights, y.data(), 1);
            std::vector<float> dy(N);
            for (std::size_t c = 0; c < N; ++c) dy[c] = 2.0f * (y[c] - target[i * N + c]);
            std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
            // StochasticRounding=true -- matches the real production tile-
            // recurrence arm's own rounding mode (see conversation: this
            // session switched it to stochastic specifically because
            // deterministic rounding was found to permanently freeze at
            // this exact config).
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, true, BoundedRMSpropSynapsePolicy>(
                &basis[i * N], 1, SIZE_TYPE(N), dy.data(), weights, dx.data(), ni.data(), ng.data(), LR, 1,
                false, true, 0.999f, 1e-8f, BETA1, MIN_DECAY_FRAC, MAX_ABS_DELTA, MAX_CI);
        }

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
        if (!std::isfinite(sse) || !std::isfinite(max_abs_y)) saw_non_finite = true;
        if (step >= 200) max_sse_after_200 = std::max(max_sse_after_200, sse);
        if (step % checkpoint_every == 0 || step == N_STEPS - 1) {
            checkpoints.push_back(sse);
            std::printf("step=%6zu sse=%10.4f max_abs_y=%.6f\n", step, sse, max_abs_y);
        }
    }

    CHECK(!saw_non_finite, "production default produced a non-finite SSE or output somewhere in %zu steps", N_STEPS);
    // Reasonable ceiling with generous margin: healthy convergence on this
    // task settles near SSE~0 (0.37^2*8=1.0952 is the all-zero-output
    // reference value used elsewhere in this suite); 50.0 gives 45x margin
    // above that reference while still catching any real late-onset
    // divergence (the unsafe pocket, by contrast, reaches SSE>100000).
    CHECK(max_sse_after_200 < 50.0f,
          "production default's SSE exceeded the long-horizon stability ceiling after step 200: max=%.4f (limit 50.0)",
          max_sse_after_200);

    std::printf("\nFINAL: max_sse_after_200=%.4f (limit 50.0), %zu steps checked, non_finite=%s\n",
                max_sse_after_200, N_STEPS, saw_non_finite ? "YES" : "no");

    if (g_fail == 0) {
        std::printf("test_production_default_long_horizon: production default stayed stable for the full %zu-step run.\n", N_STEPS);
        return 0;
    }
    std::printf("test_production_default_long_horizon: %d failure(s)\n", g_fail);
    return 1;
}
