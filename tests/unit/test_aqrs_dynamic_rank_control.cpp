// TDD spec test for AQRS dynamic rank control's EMA tracking + Theorem 10
// trigger evaluation (task #284, precedes #285's real grow/shrink wiring
// -- see sili_peridot/AQRS_DESIGN.md's "corrected" noise-mitigation
// design). Written BEFORE the EMA update is wired into disldo_backward's
// real gamma update block -- the manual-call tests (1-3) exercise the raw
// storage functions directly (already implemented in delta_csr_types.hpp
// as part of this task), while the integration test (4) DOES NOT COMPILE/
// PASS YET on purpose: it expects disldo_backward to automatically update
// the EMA arrays every step as a side effect of training, which isn't
// wired in yet -- that's the next implementation step this test drives.
//
// Design recap: EMA-smoothed |gamma_i|, C_i=|gamma_i|/||gamma||_1, and
// |dL/d(gamma_i)| (gamma is a scalar per channel, so "||grad||_F" reduces
// to plain magnitude -- no row/col structure to norm over), updated EVERY
// step (not every N -- see AQRS_DESIGN.md's rejection of periodic
// checking as a "luck filter"). Apoptosis: A(gamma_i) = (|gamma_i|_ema <
// tau_death) AND (C_i_ema < tau_death). Neurogenesis: N(gamma,grad) =
// (min_j |gamma_j|_ema > tau_active) AND (max_j grad_j_ema > theta), with
// tau_death < tau_active (the hysteresis gap).
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

// ── Test 1: EMA update matches the exact decay formula ─────────────────
static void test_ema_update_formula() {
    Weights w;
    const float decay = 0.98f;
    float expected_abs = 0.0f, expected_share = 0.0f, expected_grad = 0.0f;
    const float inputs_abs[] = {1.0f, 2.0f, 0.5f, 3.0f};
    const float inputs_share[] = {0.5f, 0.3f, 0.1f, 0.4f};
    const float inputs_grad[] = {0.2f, 0.1f, 0.05f, 0.3f};
    for (int i = 0; i < 4; ++i) {
        w.update_scale_gamma_ema_k(0, inputs_abs[i], inputs_share[i], inputs_grad[i], decay);
        expected_abs = decay * expected_abs + (1.0f - decay) * inputs_abs[i];
        expected_share = decay * expected_share + (1.0f - decay) * inputs_share[i];
        expected_grad = decay * expected_grad + (1.0f - decay) * inputs_grad[i];
    }
    CHECK(std::fabs(w.get_scale_gamma_abs_ema_k(0) - expected_abs) < 1e-6f,
          "abs EMA should match manual decay formula exactly (got %.6f expected %.6f)",
          w.get_scale_gamma_abs_ema_k(0), expected_abs);
    CHECK(std::fabs(w.get_scale_gamma_share_ema_k(0) - expected_share) < 1e-6f,
          "share EMA should match manual decay formula exactly (got %.6f expected %.6f)",
          w.get_scale_gamma_share_ema_k(0), expected_share);
    CHECK(std::fabs(w.get_scale_gamma_grad_ema_k(0) - expected_grad) < 1e-6f,
          "grad EMA should match manual decay formula exactly (got %.6f expected %.6f)",
          w.get_scale_gamma_grad_ema_k(0), expected_grad);
}

// ── Test 2: apoptosis needs BOTH abs and share EMA below tau_death ──────
static void test_apoptosis_trigger() {
    Weights w;
    const float tau_death = 0.05f;
    // Channel 0: both low -- should apoptose.
    for (int i = 0; i < 50; ++i)
        w.update_scale_gamma_ema_k(0, 0.01f, 0.02f, 0.0f, 0.9f);
    CHECK(w.scale_gamma_should_apoptose(0, tau_death),
          "channel with both |gamma|_ema and C_ema below tau_death should apoptose");

    // Channel 1: low absolute magnitude but HIGH share (whole group is
    // legitimately small, this channel isn't relatively dead) -- must NOT apoptose.
    for (int i = 0; i < 50; ++i)
        w.update_scale_gamma_ema_k(1, 0.01f, 0.9f, 0.0f, 0.9f);
    CHECK(!w.scale_gamma_should_apoptose(1, tau_death),
          "channel with low |gamma|_ema but high C_ema (whole group small) should NOT apoptose");

    // Channel 2: high absolute magnitude -- must NOT apoptose regardless of share.
    for (int i = 0; i < 50; ++i)
        w.update_scale_gamma_ema_k(2, 5.0f, 0.01f, 0.0f, 0.9f);
    CHECK(!w.scale_gamma_should_apoptose(2, tau_death),
          "channel with high |gamma|_ema should NOT apoptose regardless of share");
}

// ── Test 3: neurogenesis needs ALL channels active AND real grad pressure ─
static void test_neurogenesis_trigger() {
    Weights w;
    const float tau_active = 0.1f, theta = 0.05f;

    // All channels active, real grad pressure -- should fire.
    for (int i = 0; i < 50; ++i) {
        w.update_scale_gamma_ema_k(0, 1.0f, 0.5f, 0.2f, 0.9f);
        w.update_scale_gamma_ema_k(1, 2.0f, 0.5f, 0.1f, 0.9f);
    }
    CHECK(w.scale_gamma_should_neurogenesis(2, tau_active, theta),
          "all channels active + real grad pressure should trigger neurogenesis");

    // Reset a fresh weights struct: one channel still idle (below tau_active) --
    // must NOT fire, even with grad pressure (an idle channel could absorb it).
    Weights w2;
    for (int i = 0; i < 50; ++i) {
        w2.update_scale_gamma_ema_k(0, 1.0f, 0.5f, 0.2f, 0.9f);
        w2.update_scale_gamma_ema_k(1, 0.01f, 0.5f, 0.2f, 0.9f); // idle channel
    }
    CHECK(!w2.scale_gamma_should_neurogenesis(2, tau_active, theta),
          "an idle (below tau_active) existing channel should block neurogenesis");

    // Fresh weights: all channels active, but no real grad pressure -- must NOT fire.
    Weights w3;
    for (int i = 0; i < 50; ++i) {
        w3.update_scale_gamma_ema_k(0, 1.0f, 0.5f, 0.001f, 0.9f);
        w3.update_scale_gamma_ema_k(1, 2.0f, 0.5f, 0.001f, 0.9f);
    }
    CHECK(!w3.scale_gamma_should_neurogenesis(2, tau_active, theta),
          "no real leftover grad pressure should block neurogenesis even with all channels active");
}

// ── Test 4: INTEGRATION -- disldo_backward must update EMAs automatically ──
// Does NOT rely on manually calling update_scale_gamma_ema_k -- trains a
// real layer via disldo_backward and confirms the EMA arrays track gamma's
// real trajectory as a side effect of training, same as gamma's own value
// and its RMSprop state already do. This is the part not wired in yet.
static void test_ema_updates_automatically_during_training() {
    std::vector<float> w_q = {6.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                              2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 6.f};
    std::vector<float> w_star = {50.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                 2.f,  2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 50.f};
    Weights w = make_scattered_layer(w_q);
    w.scale_rank = 1;
    w.set_additive_rank(0);
    for (std::size_t r = 0; r < N; ++r)
        w.set_value_scale_raw_k(r, 0, 1.0f);
    for (std::size_t c = 0; c < N; ++c)
        w.set_output_scale_raw_k(c, 0, 1.0f);
    w.set_scale_gamma_raw_k(0, 0.1f);

    CHECK(w.get_scale_gamma_abs_ema_k(0) == 0.0f, "EMA should start at 0 before any training step");

    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;
    for (int step = 0; step < 500; ++step) {
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), w,
                                                             y.data(), 1);
            std::vector<float> dy(N);
            for (std::size_t c = 0; c < N; ++c)
                dy[c] = 2.0f * (y[c] - w_star[i * N + c]) / float(N);
            std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false,
                            false>(&basis[i * N], 1, SIZE_TYPE(N), dy.data(), w, dx.data(),
                                   ni.data(), ng.data(), 0.05f, 1, false, true, 0.999f, 1e-8f, 0.9f,
                                   0.0f, 1e30f);
        }
    }
    const float gamma_final = w.get_scale_gamma_k(0);
    const float ema_final = w.get_scale_gamma_abs_ema_k(0);
    std::printf("[ema-integration] gamma final: %.4f, abs EMA final: %.4f\n", gamma_final,
                ema_final);
    CHECK(ema_final > 0.5f,
          "abs EMA should have tracked gamma's real growth during training, not stayed at 0 (got "
          "%.4f, gamma=%.4f)",
          ema_final, gamma_final);
    // Since gamma grows MONOTONICALLY here (converging toward the target),
    // the EMA (a lagging average) should end up BELOW the final raw value,
    // not equal to or above it -- a real behavioral check that it's
    // actually smoothing, not just mirroring the current value 1:1.
    CHECK(ema_final < gamma_final,
          "EMA of a monotonically-growing signal should lag behind the current raw value (ema=%.4f "
          "gamma=%.4f)",
          ema_final, gamma_final);
}

int main() {
    test_ema_update_formula();
    test_apoptosis_trigger();
    test_neurogenesis_trigger();
    test_ema_updates_automatically_during_training();

    if (g_fail == 0) {
        std::printf("All AQRS dynamic rank control tests passed.\n");
    } else {
        std::printf("%d FAILURES\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
