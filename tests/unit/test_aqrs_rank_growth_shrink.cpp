// C++ tests for AQRS rank growth/shrinking (task #281, precedes #273's real
// dynamic-rank-control mechanism -- per direct user instruction this
// session: "the 4 new rank-mutation op tests should probably be in c++
// first as well ... verify add-multiplier-rank, sub-multiplier-rank,
// add-addition-rank, sub-addition-rank in tests"). See sili_peridot/
// AQRS_DESIGN.md for the full derivation.
//
// set_scale_rank/set_additive_rank (delta_csr_types.hpp, task #275) already
// implement a real reshuffle -- but every EXISTING call site (including
// test_aqrs_additive_branch.cpp) only ever calls them ONCE, from a fresh
// weights object, before any training happens. Task #273's real dynamic
// rank control needs to resize a LIVE, already-trained layer -- these
// tests are the first real exercise of that path: mutate rank mid-training
// and verify (a) existing channels' data survives exactly, (b) growth is a
// true no-op at the instant it happens (matches the additive branch's own
// zero-init-is-a-no-op convention), (c) a grown channel can actually be
// trained and measurably helps beyond what the old rank could reach, and
// (d) shrinking removes a channel's real (already-trained, nonzero)
// contribution rather than silently doing nothing.
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

static const std::size_t N = 3; // all test matrices are 3x3

// Same fixed-dense-layer convention as test_aqrs_additive_branch.cpp: W_q
// codes are frozen (never re-quantized), only the multiplicative/additive
// branches ever move.
static Weights make_fixed_dense_layer(const std::vector<float>& w_q_values) {
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

static float mse(const std::vector<float>& y, const std::vector<float>& target) {
    float s = 0.0f;
    for (std::size_t i = 0; i < y.size(); ++i) {
        float d = y[i] - target[i];
        s += d * d;
    }
    return s / float(y.size());
}

// Runs one forward+backward step per basis row (same "3x identity-basis
// probe" convention as test_aqrs_additive_branch.cpp), returns the full
// N x N output. freeze_multiplicative mirrors the additive-branch test's
// own trick: disldo_backward has no per-branch freeze flag, so isolating
// the additive branch means resetting value_scale/output_scale back to
// identity after every backward call.
static std::vector<float> step(Weights& weights, const std::vector<float>& w_star, float lr,
                               bool freeze_multiplicative = false) {
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
        std::vector<float> dy(N);
        for (std::size_t c = 0; c < N; ++c)
            dy[c] = 2.0f * (y[c] - w_star[i * N + c]) / float(N);
        std::vector<float> dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
        disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
            &basis[i * N], 1, SIZE_TYPE(N), dy.data(), weights, dx.data(), ni.data(), ng.data(), lr,
            1, false, true, 0.999f, 1e-8f, 0.9f, 0.0f, 1e30f);
        if (freeze_multiplicative) {
            for (std::size_t row = 0; row < N; ++row)
                for (std::size_t k = 0; k < weights.scale_rank; ++k)
                    weights.set_value_scale_raw_k(row, k, k == 0 ? 1.0f : 0.0f);
            for (std::size_t col = 0; col < N; ++col)
                for (std::size_t k = 0; k < weights.scale_rank; ++k)
                    weights.set_output_scale_raw_k(col, k, k == 0 ? 1.0f : 0.0f);
        }
    }
    return y_all;
}

// Pure forward pass over the same 3x identity basis, no training -- used
// to check growth is a no-op at the instant it happens.
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

// Test matrices reused from test_aqrs_additive_branch.cpp's "Test A": two
// independent diagonal outliers, W_q clips 50 -> 6 at (0,0) and (2,2).
// Established there that rank-1 (either branch) plateaus well above 5.0
// MSE and rank-2 (either branch) can drive it near zero -- exactly the
// kind of target where a genuine rank increase should measurably help,
// making it a good discriminator for these growth/shrink tests too.
static const std::vector<float> W_Q = {6.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 6.f};
static const std::vector<float> W_STAR = {50.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 50.f};

// ── add-multiplier-rank ──────────────────────────────────────────────────
static void test_add_multiplier_rank() {
    Weights w = make_fixed_dense_layer(W_Q);
    w.scale_rank = 1;
    w.set_additive_rank(0);

    for (int i = 0; i < 1000; ++i)
        step(w, W_STAR, 0.05f);
    std::vector<float> snap_v0(N), snap_o0(N);
    for (std::size_t r = 0; r < N; ++r)
        snap_v0[r] = w.get_value_scale_k(r, 0);
    for (std::size_t c = 0; c < N; ++c)
        snap_o0[c] = w.get_output_scale_k(c, 0);
    std::vector<float> y_before_grow = forward_only(w);

    w.set_scale_rank(2); // add-multiplier-rank

    CHECK(w.scale_rank == 2, "scale_rank should be 2 after growth (got %zu)", w.scale_rank);
    for (std::size_t r = 0; r < N; ++r)
        CHECK(std::fabs(w.get_value_scale_k(r, 0) - snap_v0[r]) < 1e-6f,
              "value_scale k=0 row %zu should survive growth unchanged (before=%.6f after=%.6f)", r,
              snap_v0[r], w.get_value_scale_k(r, 0));
    for (std::size_t c = 0; c < N; ++c)
        CHECK(std::fabs(w.get_output_scale_k(c, 0) - snap_o0[c]) < 1e-6f,
              "output_scale k=0 col %zu should survive growth unchanged (before=%.6f after=%.6f)",
              c, snap_o0[c], w.get_output_scale_k(c, 0));
    for (std::size_t r = 0; r < N; ++r)
        CHECK(w.get_value_scale_k(r, 1) == 0.0f,
              "new value_scale channel k=1 should default to 0.0 (row %zu)", r);

    std::vector<float> y_after_grow = forward_only(w);
    CHECK(all_close(y_before_grow, y_after_grow),
          "growth alone (before any further training) must be a true no-op on the forward output");

    // Same symmetric zero-init deadlock as the additive branch (see
    // seed_additive_u's own comment) -- confirmed by direct probe after
    // fixing the resize-default bug this test uncovered (linear_disldo.hpp,
    // delta_csr_types.hpp): dL/d(value_scale_k(r,1)) uses output_scale_k(.,1)
    // as a multiplier and vice versa, so with BOTH correctly defaulting to
    // 0.0 now, k=1 would never move without an explicit seed. Seed only
    // value_scale's new channel (output_scale's stays at its correct 0.0
    // default) -- growth must still be a no-op up to this point since
    // output_scale_k(.,1)=0 makes the product zero regardless of
    // value_scale_k(.,1).
    //
    // SECOND finding from writing this test (see conversation): output_scale
    // is gated by a separate output_scale_is_trainable flag (delta_csr_types.hpp)
    // that starts false and is flipped true ONLY by an explicit
    // set_output_scale_raw{,_k} call -- disldo_backward silently skips
    // applying any gradient to output_scale until that happens (by design,
    // not a bug: growing scale_rank alone doesn't imply the caller wants
    // output_scale trained). Without this, k=1 would stay permanently
    // frozen at its 0.0 default even with value_scale_k(.,1) seeded and
    // moving -- S[row,col] would reduce to exactly value_scale_k(row,0)
    // regardless of how many value_scale components exist, i.e. still
    // rank-1 in effect. Write the literal default (0.0) explicitly so the
    // flag flips true while the no-op property is preserved.
    for (std::size_t r = 0; r < N; ++r)
        w.set_value_scale_raw_k(r, 1, 0.01f * float(r + 1));
    for (std::size_t c = 0; c < N; ++c)
        w.set_output_scale_raw_k(c, 1, 0.0f);
    std::vector<float> y_after_seed = forward_only(w);
    CHECK(all_close(y_before_grow, y_after_seed),
          "seeding only value_scale's new channel (output_scale stays 0) must still be a no-op");

    Weights w_grown = w;
    for (int i = 0; i < 2000; ++i)
        step(w_grown, W_STAR, 0.05f);
    float mse_after_grow = mse(step(w_grown, W_STAR, 0.0f), W_STAR);

    Weights w_control = make_fixed_dense_layer(W_Q);
    w_control.scale_rank = 1;
    w_control.set_additive_rank(0);
    for (int i = 0; i < 3000; ++i)
        step(w_control, W_STAR, 0.05f);
    float mse_no_grow = mse(step(w_control, W_STAR, 0.0f), W_STAR);

    std::printf("[add-multiplier-rank] after grow+train: %.4f, rank-1-only control: %.4f\n",
                mse_after_grow, mse_no_grow);
    CHECK(mse_after_grow < mse_no_grow - 0.5f,
          "growing to rank-2 should measurably beat staying at rank-1 on a target rank-1 can't "
          "fully explain (grown=%.4f control=%.4f)",
          mse_after_grow, mse_no_grow);
}

// ── sub-multiplier-rank ──────────────────────────────────────────────────
static void test_sub_multiplier_rank() {
    Weights w = make_fixed_dense_layer(W_Q);
    w.scale_rank = 2;
    w.set_additive_rank(0);
    // Same symmetric zero-init deadlock AND output_scale_is_trainable gate
    // as add-multiplier-rank (see its own comments) -- without both fixes,
    // k=1 would never move, and this test would trivially "pass" by
    // shrinking away a channel that was never trained, which isn't the
    // real property being checked.
    for (std::size_t r = 0; r < N; ++r)
        w.set_value_scale_raw_k(r, 1, 0.01f * float(r + 1));
    for (std::size_t c = 0; c < N; ++c)
        w.set_output_scale_raw_k(c, 1, 0.0f);

    for (int i = 0; i < 3000; ++i)
        step(w, W_STAR, 0.05f);
    std::vector<float> snap_v0(N), snap_o0(N);
    for (std::size_t r = 0; r < N; ++r)
        snap_v0[r] = w.get_value_scale_k(r, 0);
    for (std::size_t c = 0; c < N; ++c)
        snap_o0[c] = w.get_output_scale_k(c, 0);
    std::vector<float> y_before_shrink = forward_only(w);

    w.set_scale_rank(1); // sub-multiplier-rank

    CHECK(w.scale_rank == 1, "scale_rank should be 1 after shrink (got %zu)", w.scale_rank);
    for (std::size_t r = 0; r < N; ++r)
        CHECK(std::fabs(w.get_value_scale_k(r, 0) - snap_v0[r]) < 1e-6f,
              "value_scale k=0 row %zu should survive shrink unchanged (before=%.6f after=%.6f)", r,
              snap_v0[r], w.get_value_scale_k(r, 0));
    for (std::size_t c = 0; c < N; ++c)
        CHECK(std::fabs(w.get_output_scale_k(c, 0) - snap_o0[c]) < 1e-6f,
              "output_scale k=0 col %zu should survive shrink unchanged (before=%.6f after=%.6f)",
              c, snap_o0[c], w.get_output_scale_k(c, 0));

    std::vector<float> y_after_shrink = forward_only(w);
    float shrink_delta = mse(y_before_shrink, y_after_shrink);
    std::printf("[sub-multiplier-rank] output change from dropping k=1: MSE-vs-self=%.6f\n",
                shrink_delta);
    CHECK(shrink_delta > 1e-4f,
          "shrinking a rank-2 layer that trained for real should actually remove a nonzero "
          "contribution, not be a no-op (delta=%.6f)",
          shrink_delta);
}

// ── add-addition-rank ────────────────────────────────────────────────────
static void test_add_addition_rank() {
    Weights w = make_fixed_dense_layer(W_Q);
    w.scale_rank = 1; // pinned to identity via freeze_multiplicative below
    w.set_additive_rank(1);
    w.set_additive_u_raw_k(0, 0, 0.01f);
    w.set_additive_u_raw_k(1, 0, 0.02f);
    w.set_additive_u_raw_k(2, 0, 0.03f);

    for (int i = 0; i < 1000; ++i)
        step(w, W_STAR, 0.05f, true);
    std::vector<float> snap_u0(N), snap_v0(N);
    for (std::size_t r = 0; r < N; ++r)
        snap_u0[r] = w.get_additive_u_k(r, 0);
    for (std::size_t c = 0; c < N; ++c)
        snap_v0[c] = w.get_additive_v_k(c, 0);
    std::vector<float> y_before_grow = forward_only(w);

    w.set_additive_rank(2); // add-addition-rank
    // Seed ONLY the new channel's u (matches test_aqrs_additive_branch.cpp's
    // own symmetric zero-init-deadlock fix) -- channel 0 must be left alone
    // to prove growth doesn't disturb an already-trained channel.
    w.set_additive_u_raw_k(0, 1, 0.015f);
    w.set_additive_u_raw_k(1, 1, 0.025f);
    w.set_additive_u_raw_k(2, 1, 0.035f);

    CHECK(w.additive_rank == 2, "additive_rank should be 2 after growth (got %zu)",
          w.additive_rank);
    for (std::size_t r = 0; r < N; ++r)
        CHECK(std::fabs(w.get_additive_u_k(r, 0) - snap_u0[r]) < 1e-6f,
              "additive_u k=0 row %zu should survive growth unchanged (before=%.6f after=%.6f)", r,
              snap_u0[r], w.get_additive_u_k(r, 0));
    for (std::size_t c = 0; c < N; ++c)
        CHECK(std::fabs(w.get_additive_v_k(c, 0) - snap_v0[c]) < 1e-6f,
              "additive_v k=0 col %zu should survive growth unchanged (before=%.6f after=%.6f)", c,
              snap_v0[c], w.get_additive_v_k(c, 0));

    std::vector<float> y_after_grow = forward_only(w);
    CHECK(all_close(y_before_grow, y_after_grow),
          "growth (new channel's v still zero-default) must be a true no-op on the forward output");

    Weights w_grown = w;
    for (int i = 0; i < 2000; ++i)
        step(w_grown, W_STAR, 0.05f, true);
    float mse_after_grow = mse(step(w_grown, W_STAR, 0.0f, true), W_STAR);

    Weights w_control = make_fixed_dense_layer(W_Q);
    w_control.scale_rank = 1;
    w_control.set_additive_rank(1);
    w_control.set_additive_u_raw_k(0, 0, 0.01f);
    w_control.set_additive_u_raw_k(1, 0, 0.02f);
    w_control.set_additive_u_raw_k(2, 0, 0.03f);
    for (int i = 0; i < 3000; ++i)
        step(w_control, W_STAR, 0.05f, true);
    float mse_no_grow = mse(step(w_control, W_STAR, 0.0f, true), W_STAR);

    std::printf("[add-addition-rank] after grow+train: %.4f, rank-1-only control: %.4f\n",
                mse_after_grow, mse_no_grow);
    CHECK(mse_after_grow < mse_no_grow - 0.5f,
          "growing additive_rank to 2 should measurably beat staying at rank-1 on two independent "
          "diagonal spikes (grown=%.4f control=%.4f)",
          mse_after_grow, mse_no_grow);
}

// ── sub-addition-rank ─────────────────────────────────────────────────────
static void test_sub_addition_rank() {
    Weights w = make_fixed_dense_layer(W_Q);
    w.scale_rank = 1;
    w.set_additive_rank(2);
    w.set_additive_u_raw_k(0, 0, 0.01f);
    w.set_additive_u_raw_k(0, 1, 0.015f);
    w.set_additive_u_raw_k(1, 0, 0.02f);
    w.set_additive_u_raw_k(1, 1, 0.025f);
    w.set_additive_u_raw_k(2, 0, 0.03f);
    w.set_additive_u_raw_k(2, 1, 0.035f);

    for (int i = 0; i < 3000; ++i)
        step(w, W_STAR, 0.05f, true);
    std::vector<float> snap_u0(N), snap_v0(N);
    for (std::size_t r = 0; r < N; ++r)
        snap_u0[r] = w.get_additive_u_k(r, 0);
    for (std::size_t c = 0; c < N; ++c)
        snap_v0[c] = w.get_additive_v_k(c, 0);
    std::vector<float> y_before_shrink = forward_only(w);

    w.set_additive_rank(1); // sub-addition-rank

    CHECK(w.additive_rank == 1, "additive_rank should be 1 after shrink (got %zu)",
          w.additive_rank);
    for (std::size_t r = 0; r < N; ++r)
        CHECK(std::fabs(w.get_additive_u_k(r, 0) - snap_u0[r]) < 1e-6f,
              "additive_u k=0 row %zu should survive shrink unchanged (before=%.6f after=%.6f)", r,
              snap_u0[r], w.get_additive_u_k(r, 0));
    for (std::size_t c = 0; c < N; ++c)
        CHECK(std::fabs(w.get_additive_v_k(c, 0) - snap_v0[c]) < 1e-6f,
              "additive_v k=0 col %zu should survive shrink unchanged (before=%.6f after=%.6f)", c,
              snap_v0[c], w.get_additive_v_k(c, 0));

    std::vector<float> y_after_shrink = forward_only(w);
    float shrink_delta = mse(y_before_shrink, y_after_shrink);
    std::printf("[sub-addition-rank] output change from dropping k=1: MSE-vs-self=%.6f\n",
                shrink_delta);
    CHECK(shrink_delta > 1e-4f,
          "shrinking a rank-2 additive branch that trained for real should actually remove a "
          "nonzero contribution, not be a no-op (delta=%.6f)",
          shrink_delta);
}

int main() {
    test_add_multiplier_rank();
    test_sub_multiplier_rank();
    test_add_addition_rank();
    test_sub_addition_rank();

    if (g_fail == 0) {
        std::printf("All AQRS rank growth/shrink tests passed.\n");
    } else {
        std::printf("%d FAILURES\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
