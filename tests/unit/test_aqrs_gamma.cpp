// TDD spec test for AQRS's per-channel gamma (task #282, precedes #283's
// implementation -- see sili_peridot/AQRS_DESIGN.md's gamma section and
// task #273's own description). Written BEFORE gamma exists in
// delta_csr_types.hpp/linear_disldo.hpp -- DOES NOT COMPILE YET, on
// purpose, matching the same TDD convention already used for the additive
// branch (test_aqrs_additive_branch.cpp) and rank growth/shrink
// (test_aqrs_rank_growth_shrink.cpp).
//
// WHY gamma, now: value_scale_k/output_scale_k (multiplicative) and
// additive_u/additive_v (additive) currently conflate a channel's
// DIRECTION and MAGNITUDE into the same two factors. test_aqrs_rank_growth_
// shrink.cpp found this forces a two-sided "LoRA-style" seeding hack (one
// side small-nonzero, the other left at its zero default) to escape a
// genuine symmetric zero-init deadlock on every newly-grown channel.
// gamma_s_k/gamma_o_k (ONE scalar per channel k, not per row/col) decouple
// this: value_scale_k/output_scale_k (or additive_u/additive_v) hold pure
// DIRECTION and can both be seeded with real, meaningful nonzero values
// (e.g. aligned with a residual, Theorem 9) with ZERO risk of premature
// contribution, because gamma alone gates whether the channel contributes
// at all. This also gives Theorem 8's L1 penalty ONE clean parameter to
// create a genuine attracting fixed point at exactly 0 (not two separate
// parameter groups), and gives Theorem 10's apoptosis/neurogenesis
// triggers (|gamma_i|, C_i=|gamma_i|/||gamma||_1) something to threshold
// directly.
//
// Default convention, matching value_scale_k/output_scale_k's own
// precedent EXACTLY: gamma_s_k defaults to 1.0 for EVERY k until a caller
// explicitly engages gamma (see get_scale_gamma_k's own docstring,
// delta_csr_types.hpp, for why the earlier k==0-only default broke
// backward compat and was corrected) -- k=0 is the original always-on
// rank-1 behavior every existing caller depends on (gamma=1 there is a
// pure identity multiplier), any newly-grown channel starts silent (0.0)
// ONLY once gamma is already in active use (set_scale_rank's own
// reshuffle, not the lazy default, is what enforces that -- see Test 2
// below). gamma_o_k (additive) defaults to 0.0 for EVERY k including 0 --
// the additive branch has no legacy "always-on" component (additive_rank
// itself defaults to 0, fully opt-in), so there's no k==0 special case to
// preserve there.
//
// EXERCISES BOTH SCATTERED AND BLOCK4 STORAGE (per direct instruction --
// "block4 and scattered should always be able to compute the same
// things"): every test function takes a `bool use_block4` and is run
// twice, once against a plain scattered-CSR layer and once against a
// layer loaded entirely into block4 via block4_load_dense (N=4, an exact
// BLOCK4_TILE multiple, so every position lands in one full tile with no
// boundary-clipping special case). Both constructions start from the
// SAME W_Q/W_STAR values, so passing in both modes is a genuine
// cross-check that gamma's forward/backward math (particularly the block4
// SIMD/scalar gradient-accumulation sites fixed alongside this test) is
// consistent with the scattered path, not just independently plausible.
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

static const std::size_t N = 4; // exact BLOCK4_TILE multiple -- one full tile, no boundary case

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

static Weights make_block4_layer(const std::vector<float>& w_q_values) {
    Weights w;
    // block4_load_dense only populates block4 storage -- it does NOT
    // initialize the scattered-CSR layout at all (confirmed by reading
    // test_block4_load_dense.cpp, which never calls disldo_backward and
    // so never hits this). disldo_backward's dead-row bootstrap pass
    // unconditionally calls L.row_nnz(row) for every row regardless of
    // whether a row is scattered or block4-resident -- with a bare
    // `layout.rows/cols` assignment (test_block4_load_dense.cpp's own
    // convention, sufficient for THAT test since it never calls
    // disldo_backward), elem_start/elem_end stay empty vectors and
    // row_nnz's read is out-of-bounds. A REAL fully-block4 layer would
    // still have a valid, merely-empty scattered layout (e.g. after
    // synaptogenesis moves everything out of scattered CSR into block4)
    // -- delta_csr_from_absolute with all-zero ptrs is the correct way to
    // construct that: a genuinely empty but well-formed scattered layout.
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
    block4_load_dense<SIZE_TYPE, FP4BiPacked, COL_TYPE>(w, weight_codes.data(),
                                                        importance_codes.data(), N, N);
    w.out_degree.assign(N, SIZE_TYPE(N));
    return w;
}

static Weights make_fixed_dense_layer(const std::vector<float>& w_q_values, bool use_block4) {
    return use_block4 ? make_block4_layer(w_q_values) : make_scattered_layer(w_q_values);
}

static float mse(const std::vector<float>& y, const std::vector<float>& target) {
    float s = 0.0f;
    for (std::size_t i = 0; i < y.size(); ++i) {
        float d = y[i] - target[i];
        s += d * d;
    }
    return s / float(y.size());
}

static std::vector<float> forward_only(Weights& weights) {
    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;
    std::vector<float> y_all(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) {
        std::vector<float> y(N, 0.0f);
        disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), weights,
                                                         y.data(), 1);
        for (std::size_t c = 0; c < N; ++c)
            y_all[i * N + c] = y[c];
    }
    return y_all;
}

static bool all_close(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-5f) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol)
            return false;
    return true;
}

// 4x4 diagonal-outlier pattern: W_q clips 50->6 at (0,0) and (3,3), rest
// stays 2. Same structural shape as the earlier 3x3 additive-branch tests,
// extended by one row/col to reach BLOCK4_TILE's exact size.
static const std::vector<float> W_Q = {6.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                       2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 6.f};
static const std::vector<float> W_STAR = {50.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                                          2.f,  2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 50.f};

// ── Test 1: gamma default preserves today's rank-1 behavior exactly ─────
static void test_gamma_default_backward_compat(bool use_block4) {
    Weights w = make_fixed_dense_layer(W_Q, use_block4);
    w.scale_rank = 1;
    w.set_additive_rank(0);
    for (std::size_t r = 0; r < N; ++r)
        w.set_value_scale_raw_k(r, 0, 2.5f * float(r + 1));
    for (std::size_t c = 0; c < N; ++c)
        w.set_output_scale_raw_k(c, 0, 1.5f);

    CHECK(std::fabs(w.get_scale_gamma_k(0) - 1.0f) < 1e-6f,
          "[block4=%d] gamma_s_k(0) must default to 1.0 (identity multiplier for the legacy "
          "always-on channel), got %.6f",
          int(use_block4), w.get_scale_gamma_k(0));

    std::vector<float> y = forward_only(w);
    // With gamma_s_k(0)=1 (default), S[row,col] = value_scale_k(row,0)*output_scale_k(col,0)
    // exactly as before gamma existed -- true_w = stored_w * S.
    for (std::size_t r = 0; r < N; ++r) {
        const float expected = W_Q[r * N + r] * (2.5f * float(r + 1)) * 1.5f;
        CHECK(std::fabs(y[r * N + r] - expected) < 1e-2f,
              "[block4=%d] row %zu diagonal output should match pre-gamma S formula exactly (got "
              "%.6f expected %.6f)",
              int(use_block4), r, y[r * N + r], expected);
    }
}

// ── Test 2: gamma=0 is a true universal off-switch, regardless of direction magnitude ─
static void test_gamma_zero_is_true_off_switch(bool use_block4) {
    Weights w = make_fixed_dense_layer(W_Q, use_block4);
    w.scale_rank = 1;
    w.set_additive_rank(0);
    for (std::size_t r = 0; r < N; ++r)
        w.set_value_scale_raw_k(r, 0, 3.0f);
    for (std::size_t c = 0; c < N; ++c)
        w.set_output_scale_raw_k(c, 0, 1.0f);
    // Explicitly touch gamma at k=0 BEFORE growing -- this is what puts
    // gamma into "active use," which is what makes set_scale_rank's own
    // reshuffle (not the lazy default) responsible for the new channel's
    // gamma, matching task #273's real dynamic-rank-control usage pattern
    // (see get_scale_gamma_k's own docstring: growth on a layer that has
    // NEVER touched gamma leaves it transparently 1.0 everywhere, for
    // backward compat with every pre-gamma caller).
    w.set_scale_gamma_raw_k(0, 1.0f);
    std::vector<float> y_before = forward_only(w);

    w.set_scale_rank(2);
    CHECK(std::fabs(w.get_scale_gamma_k(1) - 0.0f) < 1e-6f,
          "[block4=%d] gamma_s_k(1) (newly grown channel, gamma already in active use) must "
          "default to 0.0, got %.6f",
          int(use_block4), w.get_scale_gamma_k(1));
    // Seed BOTH direction sides of the new channel with large, meaningfully
    // "aligned" nonzero values -- this is exactly what gamma is supposed to
    // make safe (no more one-side-must-stay-zero hack).
    for (std::size_t r = 0; r < N; ++r)
        w.set_value_scale_raw_k(r, 1, 10.0f * float(r + 1));
    for (std::size_t c = 0; c < N; ++c)
        w.set_output_scale_raw_k(c, 1, 7.0f);

    std::vector<float> y_after = forward_only(w);
    CHECK(all_close(y_before, y_after),
          "[block4=%d] gamma_s_k(1)=0 must zero the whole channel 1 contribution regardless of how "
          "large value_scale/output_scale k=1 are",
          int(use_block4));
}

// ── Test 3: gamma's own backward gradient moves it in the loss-reducing direction ─
static void test_gamma_backward_moves_toward_loss_reduction(bool use_block4) {
    Weights w = make_fixed_dense_layer(W_Q, use_block4);
    w.scale_rank = 1;
    w.set_additive_rank(0);
    // Fixed direction vectors for k=0 (never trained here -- frozen via a
    // manual reset each step, same trick test_aqrs_additive_branch.cpp
    // used to isolate one branch) so gamma's gradient is unambiguous.
    for (std::size_t r = 0; r < N; ++r)
        w.set_value_scale_raw_k(r, 0, 1.0f);
    for (std::size_t c = 0; c < N; ++c)
        w.set_output_scale_raw_k(c, 0, 1.0f);
    w.set_scale_gamma_raw_k(0, 0.1f); // small nonzero seed, deliberately not the 1.0 default

    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;
    // W_STAR/W_Q at the diagonal outliers is 50/6 ~= 8.33 -- gamma should
    // grow from 0.1 toward something near that, since S[row,col] =
    // gamma*1*1 = gamma directly with these fixed direction vectors.
    for (int step = 0; step < 2000; ++step) {
        for (std::size_t i = 0; i < N; ++i) {
            std::vector<float> y(N, 0.0f);
            disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[i * N], 1, SIZE_TYPE(N), w,
                                                             y.data(), 1);
            std::vector<float> dy(N);
            for (std::size_t c = 0; c < N; ++c)
                dy[c] = 2.0f * (y[c] - W_STAR[i * N + c]) / float(N);
            std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false,
                            false>(&basis[i * N], 1, SIZE_TYPE(N), dy.data(), w, dx.data(),
                                   ni.data(), ng.data(), 0.05f, 1, false, true, 0.999f, 1e-8f, 0.9f,
                                   0.0f, 1e30f);
            // Freeze direction vectors -- only gamma should move.
            for (std::size_t r = 0; r < N; ++r)
                w.set_value_scale_raw_k(r, 0, 1.0f);
            for (std::size_t c = 0; c < N; ++c)
                w.set_output_scale_raw_k(c, 0, 1.0f);
        }
    }
    const float final_gamma = w.get_scale_gamma_k(0);
    std::printf("[gamma-backward block4=%d] final gamma_s_k(0): %.4f (seeded 0.1, target ~8.33)\n",
                int(use_block4), final_gamma);
    CHECK(final_gamma > 3.0f,
          "[block4=%d] gamma should have grown substantially toward the target scale via its own "
          "gradient (got %.4f)",
          int(use_block4), final_gamma);
}

// ── Test 4: L1 penalty on gamma creates a genuine fixed point at 0 (Theorem 8) ─
static void test_gamma_l1_fixed_point_at_zero(bool use_block4) {
    // Uniform 3x target. Channel 0 is FROZEN at an exact-fit configuration
    // (value_scale_k(.,0)=3, output_scale_k(.,0)=1, gamma_s_k(0)=1 -- S=3
    // everywhere, matching w_star=3*w_q exactly with zero residual).
    // Channel 1's direction is ALSO frozen (1,1 -- same "shape" as channel
    // 0, so it's at least plausible it could help) -- only gamma_s_k(1)
    // ever trains. With channel 0 alone already reaching zero residual,
    // dL/d(gamma_1) = sum dL/dS * value_dir_1 * output_dir_1 is EXACTLY
    // zero once y==target, so ONLY the L1 penalty acts on gamma_1 --
    // isolates the fixed-point property from any competing gradient
    // pressure (an earlier version of this test let k=0 train too, which
    // let gradient descent and L1 fight over an ambiguous split between
    // two channels with IDENTICAL direction -- confounded, not a clean
    // test of Theorem 8 alone).
    std::vector<float> w_q = {2.f, 1.f, 3.f, 2.f, 1.f, 2.f, 1.f, 3.f,
                              3.f, 1.f, 2.f, 1.f, 2.f, 3.f, 1.f, 2.f};
    std::vector<float> w_star(w_q.size());
    for (std::size_t i = 0; i < w_q.size(); ++i)
        w_star[i] = 3.0f * w_q[i]; // 3x w_q

    Weights w = make_fixed_dense_layer(w_q, use_block4);
    w.scale_rank = 2;
    w.set_additive_rank(0);
    auto freeze_channel0_and_direction1 = [&]() {
        for (std::size_t r = 0; r < N; ++r)
            w.set_value_scale_raw_k(r, 0, 3.0f);
        for (std::size_t c = 0; c < N; ++c)
            w.set_output_scale_raw_k(c, 0, 1.0f);
        w.set_scale_gamma_raw_k(0, 1.0f);
        for (std::size_t r = 0; r < N; ++r)
            w.set_value_scale_raw_k(r, 1, 1.0f);
        for (std::size_t c = 0; c < N; ++c)
            w.set_output_scale_raw_k(c, 1, 1.0f);
    };
    freeze_channel0_and_direction1();
    w.set_scale_gamma_raw_k(1, 0.5f); // seeded away from zero on purpose

    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;
    const float l1_coef = 0.05f;
    auto run_steps = [&](int n_steps) {
        for (int step = 0; step < n_steps; ++step) {
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
                                       ni.data(), ng.data(), 0.05f, 1, false, true, 0.999f, 1e-8f,
                                       0.9f, 0.0f, 1e30f, 1e30f, 0.1f, false, l1_coef);
                freeze_channel0_and_direction1();
            }
        }
    };

    run_steps(1000);
    const float gamma1_final = w.get_scale_gamma_k(1);
    std::printf(
        "[gamma-L1 block4=%d] final gamma_s_k(1): %.6f (seeded 0.5, L1 should pull to exactly 0)\n",
        int(use_block4), gamma1_final);
    CHECK(std::fabs(gamma1_final) < 1e-4f,
          "[block4=%d] L1 penalty should pull an unneeded channel's gamma to exactly 0, a real "
          "fixed point (got %.6f)",
          int(use_block4), gamma1_final);

    // Run MORE steps -- a true fixed point must stay settled, not drift.
    run_steps(1000);
    CHECK(std::fabs(w.get_scale_gamma_k(1)) < 1e-4f,
          "[block4=%d] gamma_s_k(1) must STAY at 0 (real fixed point), not drift back up (got %.6f "
          "after extra steps)",
          int(use_block4), w.get_scale_gamma_k(1));
}

static void run_all_tests(bool use_block4) {
    std::printf("── running gamma tests, use_block4=%d ──\n", int(use_block4));
    test_gamma_default_backward_compat(use_block4);
    test_gamma_zero_is_true_off_switch(use_block4);
    test_gamma_backward_moves_toward_loss_reduction(use_block4);
    test_gamma_l1_fixed_point_at_zero(use_block4);
}

int main() {
    run_all_tests(false); // scattered CSR
    run_all_tests(true);  // block4

    if (g_fail == 0) {
        std::printf("All AQRS gamma tests passed (scattered + block4).\n");
    } else {
        std::printf("%d FAILURES\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
