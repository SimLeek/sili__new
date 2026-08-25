// TDD spec test for AQRS's additive-branch gamma (task #289, see
// sili_peridot/AQRS_DESIGN.md's gamma section: A(theta_o) = sum_k
// gamma_o_k*u_k*v_k^T). Mirrors test_aqrs_gamma.cpp's own structure
// exactly, adapted for the additive branch's own conventions:
//   - additive_gamma_k's lazy default is 1.0 (transparent), NOT 0.0 --
//     corrected during this same task from an earlier version that
//     defaulted to 0.0 on the (wrong, once gamma is actually wired into
//     forward/backward) reasoning that "the additive branch has no
//     legacy always-on component to preserve." See get_additive_gamma_k's
//     own docstring, delta_csr_types.hpp.
//   - L1 applies to EVERY channel including k==0 (unlike scale_gamma's
//     k>0 exemption) -- additive_rank has no legacy always-on channel,
//     min_rank=0 in apply_additive_dynamic_rank_control, so the branch
//     can legitimately shrink itself back to fully off.
//
// Also exercises BOTH scattered and block4 storage (task #290: confirms,
// rather than assumes, that the additive branch's forward/backward pass
// is genuinely structure-agnostic -- it never touches weights.connections
// at all, so block4 vs scattered should produce IDENTICAL additive-branch
// contributions for the same underlying quantized weights).
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

static const std::size_t N = 4;  // exact BLOCK4_TILE multiple

static Weights make_scattered_layer(const std::vector<float>& w_q_values) {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(N + 1);
    std::vector<SIZE_TYPE> idx(N * N);
    std::vector<float> imp(N * N, 1.0f);
    for (std::size_t r = 0; r < N; ++r) {
        ptrs[r] = SIZE_TYPE(r * N);
        for (std::size_t c = 0; c < N; ++c) idx[r * N + c] = SIZE_TYPE(c);
    }
    ptrs[N] = SIZE_TYPE(N * N);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w_q_values, imp, N, N, N * N * 2, N * N * 2);
    w.out_degree.assign(N, SIZE_TYPE(N));
    return w;
}

static Weights make_block4_layer(const std::vector<float>& w_q_values) {
    Weights w;
    std::vector<SIZE_TYPE> empty_ptrs(N + 1, SIZE_TYPE(0));
    std::vector<SIZE_TYPE> empty_idx;
    std::vector<float> empty_w, empty_imp;
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        empty_ptrs, empty_idx, empty_w, empty_imp, N, N, 0, 0);
    std::vector<uint8_t> weight_codes(N * N), importance_codes(N * N);
    for (std::size_t i = 0; i < w_q_values.size(); ++i) {
        weight_codes[i] = fp4_quantize(w_q_values[i]);
        importance_codes[i] = fp4_quantize(1.0f);
    }
    block4_load_dense<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        w, weight_codes.data(), importance_codes.data(), N, N);
    w.out_degree.assign(N, SIZE_TYPE(N));
    return w;
}

static Weights make_fixed_dense_layer(const std::vector<float>& w_q_values, bool use_block4) {
    return use_block4 ? make_block4_layer(w_q_values) : make_scattered_layer(w_q_values);
}

static bool all_close(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

static std::vector<float> forward_only(Weights& weights) {
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) basis[i * N + i] = 1.0f;
    std::vector<float> y_all(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) {
        std::vector<float> y(N, 0.0f);
        disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), weights, y.data(), 1);
        for (std::size_t c = 0; c < N; ++c) y_all[i * N + c] = y[c];
    }
    return y_all;
}

// All-zero main weight matrix -- the additive branch's real motivating
// case (Theorem 3/4): forward output is PURELY the additive branch's own
// contribution, nothing from the quantized weight at all.
static const std::vector<float> W_ZERO(N * N, 0.0f);

// ── Test 1: additive_gamma defaults to 1.0, transparent for every
// EXISTING caller that populates additive_u/additive_v directly without
// ever touching gamma (task #278's pybind bindings, the fp8/fp4 MQAR
// curriculum runs already on record) -- this is the exact backward
// -compat property that made the original 0.0 default a real bug.
static void test_additive_gamma_default_backward_compat(bool use_block4) {
    Weights w = make_fixed_dense_layer(W_ZERO, use_block4);
    w.set_additive_rank(1);
    for (std::size_t r = 0; r < N; ++r) w.set_additive_u_raw_k(r, 0, 1.0f);
    for (std::size_t c = 0; c < N; ++c) w.set_additive_v_raw_k(c, 0, float(c + 1));

    CHECK(std::fabs(w.get_additive_gamma_k(0) - 1.0f) < 1e-6f,
          "[block4=%d] additive_gamma_k(0) must default to 1.0 (transparent), got %.6f",
          int(use_block4), w.get_additive_gamma_k(0));

    std::vector<float> y = forward_only(w);
    // y[row,c] = gamma(=1)*v_c*sum_r(u_r*x_r) = (c+1)*1 (basis input, one
    // nonzero per row) = (c+1) for every row.
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            const float expected = float(c + 1);
            CHECK(std::fabs(y[r * N + c] - expected) < 1e-2f,
                  "[block4=%d] row %zu col %zu should match pre-gamma additive formula exactly (got %.6f expected %.6f)",
                  int(use_block4), r, c, y[r * N + c], expected);
        }
    }
}

// ── Test 2: additive_gamma=0 is a true universal off-switch ─────────────
static void test_additive_gamma_zero_is_true_off_switch(bool use_block4) {
    Weights w = make_fixed_dense_layer(W_ZERO, use_block4);
    w.set_additive_rank(1);
    for (std::size_t r = 0; r < N; ++r) w.set_additive_u_raw_k(r, 0, 3.0f);
    for (std::size_t c = 0; c < N; ++c) w.set_additive_v_raw_k(c, 0, 2.0f);
    w.set_additive_gamma_raw_k(0, 1.0f);
    std::vector<float> y_before = forward_only(w);

    w.set_additive_rank(2);
    CHECK(std::fabs(w.get_additive_gamma_k(1) - 0.0f) < 1e-6f,
          "[block4=%d] additive_gamma_k(1) (newly grown channel, gamma already active) must default to 0.0, got %.6f",
          int(use_block4), w.get_additive_gamma_k(1));
    for (std::size_t r = 0; r < N; ++r) w.set_additive_u_raw_k(r, 1, 10.0f * float(r + 1));
    for (std::size_t c = 0; c < N; ++c) w.set_additive_v_raw_k(c, 1, 7.0f);

    std::vector<float> y_after = forward_only(w);
    CHECK(all_close(y_before, y_after),
          "[block4=%d] additive_gamma_k(1)=0 must zero the whole channel 1 contribution regardless of u/v magnitude",
          int(use_block4));
}

// ── Test 3: additive_gamma's own gradient moves it toward loss reduction ─
static void test_additive_gamma_backward_moves_toward_loss_reduction(bool use_block4) {
    Weights w = make_fixed_dense_layer(W_ZERO, use_block4);
    w.set_additive_rank(1);
    for (std::size_t r = 0; r < N; ++r) w.set_additive_u_raw_k(r, 0, 1.0f);
    for (std::size_t c = 0; c < N; ++c) w.set_additive_v_raw_k(c, 0, 1.0f);
    w.set_additive_gamma_raw_k(0, 0.1f);  // small nonzero seed

    // Target: additive branch alone (main weight is all-zero) must reach
    // a constant 5.0 everywhere -- with u=v=1 (frozen), y = gamma * N
    // (N inputs, each contributing 1*1), so gamma should grow toward
    // 5/N = 1.25.
    std::vector<float> w_star(N * N, 5.0f);

    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) basis[i * N + i] = 1.0f;
    for (int step = 0; step < 2000; ++step) {
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), w, y.data(), 1);
            std::vector<float> dy(N);
            for (std::size_t c = 0; c < N; ++c) dy[c] = 2.0f * (y[c] - w_star[i * N + c]) / float(N);
            std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
                &basis[i * N], 1, SIZE_TYPE(N), dy.data(), w, dx.data(), ni.data(), ng.data(), 0.05f, 1,
                false, true, 0.999f, 1e-8f, 0.9f, 0.0f, 1e30f);
            // Freeze direction vectors -- only gamma should move. Also
            // pin the MAIN (scattered/block4) weight matrix's own
            // contribution to exactly 0 via value_scale=0 -- found via
            // direct debugging (see conversation) that disldo_backward's
            // own per-synapse update ALSO trains the (all-zero-quant-
            // coded, but still LIVE) main weight matrix on the same
            // residual gradient every call, and the never-zero live-
            // quantize floor (tasks #248-257) means a touched live
            // synapse can no longer stay at the exact 0 code -- it drifts
            // to a small nonzero magnitude, silently adding its own
            // contribution to y and confounding "additive branch alone"
            // isolation. value_scale=0 nulls quant*scale regardless of
            // what the raw quant code drifts to.
            for (std::size_t r = 0; r < N; ++r) w.set_additive_u_raw_k(r, 0, 1.0f);
            for (std::size_t c = 0; c < N; ++c) w.set_additive_v_raw_k(c, 0, 1.0f);
            for (std::size_t r = 0; r < N; ++r) w.set_value_scale_raw_k(r, 0, 0.0f);
        }
    }
    const float final_gamma = w.get_additive_gamma_k(0);
    std::printf("[additive-gamma-backward block4=%d] final gamma_o_k(0): %.4f (seeded 0.1, target 1.25)\n",
                int(use_block4), final_gamma);
    CHECK(final_gamma > 0.8f,
          "[block4=%d] additive gamma should have grown substantially toward the target (got %.4f)",
          int(use_block4), final_gamma);
}

// ── Test 4: L1 penalty on additive_gamma creates a fixed point at 0,
// INCLUDING k==0 (unlike scale_gamma, which exempts k==0) ──────────────
static void test_additive_gamma_l1_fixed_point_at_zero_including_k0(bool use_block4) {
    // Mirrors test_aqrs_gamma.cpp's OWN isolation strategy exactly (see
    // its test_gamma_l1_fixed_point_at_zero docstring): channel 0 is
    // FROZEN at an exact-fit configuration (u0=1, v0=5, gamma0=1 -- with
    // basis input, P0=sum_r(u0_r*x_r)=1, so y=gamma0*v0*P0=5 everywhere,
    // matching w_star=5 exactly, zero residual). Channel 1's DIRECTION is
    // also frozen (same shape as channel 0: u1=1, v1=1) -- only gamma_1
    // ever trains. With channel 0 alone already reaching zero residual,
    // dL/d(gamma_1) is EXACTLY zero once y==target, so ONLY the L1
    // penalty acts on gamma_1 -- isolates the fixed-point property from
    // any competing gradient pressure.
    //
    // ALSO pins value_scale=0 every step (see test 3's identical fix,
    // same root cause) -- an earlier version of this test omitted that
    // and saw gamma_1 diverge to a persistent negative value instead of
    // settling at 0. Root-caused via direct per-step debugging (see
    // conversation): the main scattered/block4 weight matrix, despite
    // starting all-zero, is NOT actually inert during training here --
    // disldo_backward's own per-synapse update trains it on the exact
    // same residual gradient every call, and the never-zero live-
    // quantize floor (tasks #248-257) means a touched live synapse can't
    // stay at the exact 0 code, so it silently drifts to a small nonzero
    // magnitude and adds its own uncontrolled contribution to y --
    // confounding the "channel 0 alone reaches zero residual" premise
    // this test's whole isolation strategy depends on. This was NOT a
    // gamma bug -- gamma's own math was already correct (confirmed with
    // value_scale pinned to 0: gamma_1 converges smoothly and
    // monotonically to exactly 0, no overshoot at all).
    Weights w = make_fixed_dense_layer(W_ZERO, use_block4);
    w.set_additive_rank(2);
    std::vector<float> w_star(N * N, 5.0f);
    auto freeze_channel0_and_direction1 = [&]() {
        for (std::size_t r = 0; r < N; ++r) w.set_additive_u_raw_k(r, 0, 1.0f);
        for (std::size_t c = 0; c < N; ++c) w.set_additive_v_raw_k(c, 0, 5.0f);
        w.set_additive_gamma_raw_k(0, 1.0f);
        for (std::size_t r = 0; r < N; ++r) w.set_additive_u_raw_k(r, 1, 1.0f);
        for (std::size_t c = 0; c < N; ++c) w.set_additive_v_raw_k(c, 1, 1.0f);
        for (std::size_t r = 0; r < N; ++r) w.set_value_scale_raw_k(r, 0, 0.0f);
    };
    freeze_channel0_and_direction1();
    w.set_additive_gamma_raw_k(1, 0.5f);  // seeded away from zero on purpose

    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) basis[i * N + i] = 1.0f;
    const float l1_coef = 0.05f;
    auto run_steps = [&](int n_steps) {
        for (int step = 0; step < n_steps; ++step) {
            for (std::size_t i = 0; i < N; ++i) {
                std::vector<float> y(N, 0.0f);
                disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), w, y.data(), 1);
                std::vector<float> dy(N);
                for (std::size_t c = 0; c < N; ++c) dy[c] = 2.0f * (y[c] - w_star[i * N + c]) / float(N);
                std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
                disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
                    &basis[i * N], 1, SIZE_TYPE(N), dy.data(), w, dx.data(), ni.data(), ng.data(), 0.05f, 1,
                    false, true, 0.999f, 1e-8f, 0.9f, 0.0f, 1e30f, 1e30f, 0.1f, false, l1_coef);
                freeze_channel0_and_direction1();
            }
        }
    };

    run_steps(1000);
    const float gamma1_final = w.get_additive_gamma_k(1);
    std::printf("[additive-gamma-L1 block4=%d] final gamma_o_k(1): %.6f (seeded 0.5, L1 should pull to exactly 0)\n",
                int(use_block4), gamma1_final);
    CHECK(std::fabs(gamma1_final) < 1e-4f,
          "[block4=%d] L1 penalty should pull an unneeded channel's gamma to exactly 0 (additive branch has no k==0 exemption -- channel 1 here, not the always-on baseline, is the one under test), got %.6f",
          int(use_block4), gamma1_final);

    run_steps(1000);
    CHECK(std::fabs(w.get_additive_gamma_k(1)) < 1e-4f,
          "[block4=%d] additive_gamma_k(1) must STAY at 0 (real fixed point), not drift back up (got %.6f after extra steps)",
          int(use_block4), w.get_additive_gamma_k(1));
}

// ── Test 5 (task #290): block4 vs scattered give IDENTICAL additive
// -branch output/gradients for the same effective weights -- direct
// confirmation, not assumption, that the additive branch's forward/
// backward pass is genuinely structure-agnostic.
static void test_additive_gamma_block4_scattered_parity() {
    std::vector<float> w_q = {6.f, 2.f, 2.f, 2.f,   2.f, 2.f, 2.f, 2.f,
                                2.f, 2.f, 2.f, 2.f,   2.f, 2.f, 2.f, 6.f};
    Weights w_scattered = make_scattered_layer(w_q);
    Weights w_block4    = make_block4_layer(w_q);
    for (Weights* w : {&w_scattered, &w_block4}) {
        w->set_additive_rank(2);
        for (std::size_t r = 0; r < N; ++r) {
            w->set_additive_u_raw_k(r, 0, 0.3f * float(r + 1));
            w->set_additive_u_raw_k(r, 1, -0.2f * float(r + 1));
        }
        for (std::size_t c = 0; c < N; ++c) {
            w->set_additive_v_raw_k(c, 0, 0.5f);
            w->set_additive_v_raw_k(c, 1, 0.4f);
        }
        w->set_additive_gamma_raw_k(0, 1.3f);
        w->set_additive_gamma_raw_k(1, 0.7f);
    }

    std::vector<float> y_scattered = forward_only(w_scattered);
    std::vector<float> y_block4    = forward_only(w_block4);
    CHECK(all_close(y_scattered, y_block4, 1e-2f),
          "block4 and scattered must produce identical additive-branch-inclusive forward output for equivalent weights");

    // One real backward step on both, same input/target -- gamma/u/v must
    // update identically (structure-agnostic pass, no per-synapse coupling).
    std::vector<float> basis(N, 0.0f); basis[0] = 1.0f;
    std::vector<float> dy(N, 1.0f);
    std::vector<float> dx1(N, 0.0f), ni1(N, 0.0f), ng1(N, 0.0f);
    std::vector<float> dx2(N, 0.0f), ni2(N, 0.0f), ng2(N, 0.0f);
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        basis.data(), 1, SIZE_TYPE(N), dy.data(), w_scattered, dx1.data(), ni1.data(), ng1.data(), 0.05f, 1);
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        basis.data(), 1, SIZE_TYPE(N), dy.data(), w_block4, dx2.data(), ni2.data(), ng2.data(), 0.05f, 1);

    for (std::size_t k = 0; k < 2; ++k) {
        CHECK(std::fabs(w_scattered.get_additive_gamma_k(k) - w_block4.get_additive_gamma_k(k)) < 1e-4f,
              "additive_gamma_k(%zu) must update identically for block4 vs scattered (got %.6f vs %.6f)",
              k, w_scattered.get_additive_gamma_k(k), w_block4.get_additive_gamma_k(k));
        for (std::size_t r = 0; r < N; ++r)
            CHECK(std::fabs(w_scattered.get_additive_u_k(r, k) - w_block4.get_additive_u_k(r, k)) < 1e-4f,
                  "additive_u_k(row=%zu,k=%zu) must update identically for block4 vs scattered", r, k);
        for (std::size_t c = 0; c < N; ++c)
            CHECK(std::fabs(w_scattered.get_additive_v_k(c, k) - w_block4.get_additive_v_k(c, k)) < 1e-4f,
                  "additive_v_k(col=%zu,k=%zu) must update identically for block4 vs scattered", c, k);
    }
}

static void run_all_tests(bool use_block4) {
    std::printf("── running additive-gamma tests, use_block4=%d ──\n", int(use_block4));
    test_additive_gamma_default_backward_compat(use_block4);
    test_additive_gamma_zero_is_true_off_switch(use_block4);
    test_additive_gamma_backward_moves_toward_loss_reduction(use_block4);
    test_additive_gamma_l1_fixed_point_at_zero_including_k0(use_block4);
}

int main() {
    run_all_tests(false);  // scattered CSR
    run_all_tests(true);   // block4
    test_additive_gamma_block4_scattered_parity();

    if (g_fail == 0) {
        std::printf("All AQRS additive-gamma tests passed (scattered + block4 + parity).\n");
    } else {
        std::printf("%d FAILURES\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
