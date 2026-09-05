// Regression test for the grad_norm_divisor / dy-density fix (direct
// instruction, sili_peridot session 2026-08-31): AQRS's scale_gamma and
// additive_gamma neurogenesis-trigger EMA (GammaEMATracker::grad_ema, fed
// via update_scale_gamma_ema_k/update_additive_gamma_ema_k in
// disldo_backward_sparse_grad) is normalized by grad_norm_divisor =
// n_inputs*out_cols*dy_density. Before this fix, dy_density was implicitly
// 1.0 regardless of how sparse out_grad_sparse actually was -- but
// disldo_backward_sparse_grad is ONLY ever called when dy IS genuinely
// sparse (dy_sparsity_p set; see the function's own "only sparse-gradient
// backward variant" docstring), and both scale_gamma's merge-walk (against
// out_grad_sparse's own CSR indices) and additive_gamma's dP/dgamma
// computation only accumulate contributions from out_grad_sparse's
// SURVIVING columns. So a sparser dy structurally produced a smaller raw
// gradient sum with nothing to compensate -- confirmed via a real 20k-step
// sili_peridot curriculum run where input_sparsity_p/dy_sparsity_p=0.5
// pinned every layer's AQRS rank at 1 the entire run (a same-seed,
// same-everything-else isolation run with dy dense again grew ranks
// normally, up to additive_rank=43).
//
// This test proves the FIX directly: feed disldo_backward_sparse_grad two
// out_grad_sparse CSRs with the SAME per-surviving-column magnitude but
// different densities (all n_out columns present vs half), and check the
// resulting grad_ema (what should_neurogenesis actually compares against
// theta) lands in the same ballpark either way -- not roughly halved for
// the sparser call, which is what the pre-fix code would have produced.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/sisldo_ops.hpp"
#include "../../sili/lib/headers/csr.hpp"
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

static const std::size_t N_IN = 8, N_OUT = 8;

static Weights make_weights(std::size_t scale_rank, std::size_t additive_rank) {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(N_IN + 1);
    std::vector<SIZE_TYPE> idx(N_IN * N_OUT);
    std::vector<float> wv(N_IN * N_OUT), imp(N_IN * N_OUT, 0.0f);
    for (std::size_t r = 0; r < N_IN; ++r) {
        ptrs[r] = SIZE_TYPE(r * N_OUT);
        for (std::size_t c = 0; c < N_OUT; ++c) {
            idx[r * N_OUT + c] = SIZE_TYPE(c);
            wv[r * N_OUT + c] = 1.0f + 0.05f * float((r + c) % 5);
        }
    }
    ptrs[N_IN] = SIZE_TYPE(N_IN * N_OUT);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, wv, imp, N_IN, N_OUT, N_IN * N_OUT * 2, N_IN * N_OUT * 2);
    w.out_degree.assign(N_OUT, SIZE_TYPE(N_IN));

    w.set_scale_rank(scale_rank);
    w.scale_gamma_is_trainable = true;
    for (std::size_t k = 0; k < scale_rank; ++k)
        w.set_scale_gamma_raw_k(k, 1.0f);

    if (additive_rank > 0) {
        w.set_additive_rank(additive_rank);
        for (std::size_t r = 0; r < N_IN; ++r)
            for (std::size_t k = 0; k < additive_rank; ++k)
                w.set_additive_u_raw_k(r, k, 0.1f * float((r + k) % 4 + 1));
        for (std::size_t c = 0; c < N_OUT; ++c)
            for (std::size_t k = 0; k < additive_rank; ++k)
                w.set_additive_v_raw_k(c, k, 0.1f * float((c + k) % 3 + 1));
        w.additive_gamma_is_trainable = true;
        for (std::size_t k = 0; k < additive_rank; ++k)
            w.set_additive_gamma_raw_k(k, 1.0f);
    }
    w.output_scale_is_trainable = true;
    return w;
}

// Runs one backward call with a dy CSR of the given column subset (all
// entries at the SAME magnitude M, so only density differs between calls),
// returns the resulting scale_gamma/additive_gamma grad_ema[0].
static void run_backward(Weights& w, const std::vector<SIZE_TYPE>& present_cols, float magnitude,
                         float& out_scale_grad_ema, float& out_additive_grad_ema) {
    std::vector<float> input(N_IN);
    for (std::size_t r = 0; r < N_IN; ++r)
        input[r] = 0.2f + 0.1f * float(r);

    std::vector<SIZE_TYPE> dy_idx = present_cols;
    std::vector<float> dy_val(present_cols.size(), magnitude);
    auto dy_csr = make_csr_input<SIZE_TYPE, float>(SIZE_TYPE(1), SIZE_TYPE(N_OUT),
                                                   {0, SIZE_TYPE(dy_idx.size())}, dy_idx, dy_val);

    std::vector<float> dx(N_IN, 0.0f), ni(N_IN, 0.0f), ng(N_OUT, 0.0f);
    disldo_backward_sparse_grad<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false>(
        input.data(), 1, w, dy_csr, dx.data(), ni.data(), ng.data(), 0.01f, 1);

    out_scale_grad_ema = w.get_scale_gamma_grad_ema_k(0);
    out_additive_grad_ema = w.get_additive_gamma_grad_ema_k(0);
}

static void test_grad_ema_density_independent() {
    Weights w_dense_dy = make_weights(/*scale_rank=*/2, /*additive_rank=*/2);
    Weights w_sparse_dy = make_weights(/*scale_rank=*/2, /*additive_rank=*/2);

    std::vector<SIZE_TYPE> all_cols;
    for (SIZE_TYPE c = 0; c < SIZE_TYPE(N_OUT); ++c)
        all_cols.push_back(c);
    std::vector<SIZE_TYPE> half_cols;
    for (SIZE_TYPE c = 0; c < SIZE_TYPE(N_OUT); c += 2)
        half_cols.push_back(c); // 50% density

    const float magnitude = -0.3f;
    float scale_ema_dense, additive_ema_dense;
    float scale_ema_sparse, additive_ema_sparse;
    run_backward(w_dense_dy, all_cols, magnitude, scale_ema_dense, additive_ema_dense);
    run_backward(w_sparse_dy, half_cols, magnitude, scale_ema_sparse, additive_ema_sparse);

    std::printf("[grad_norm_dy_density] scale_gamma grad_ema: dense_dy=%.6f sparse_dy(50%%)=%.6f\n",
                scale_ema_dense, scale_ema_sparse);
    std::printf(
        "[grad_norm_dy_density] additive_gamma grad_ema: dense_dy=%.6f sparse_dy(50%%)=%.6f\n",
        additive_ema_dense, additive_ema_sparse);

    // Both nonzero (real signal reached the trigger at all -- the whole
    // point of the fix; pre-fix the sparse arm would still be nonzero, just
    // roughly half).
    CHECK(scale_ema_dense > 0.0f, "dense-dy scale_gamma grad_ema should be nonzero (got %.6f)",
          scale_ema_dense);
    CHECK(scale_ema_sparse > 0.0f, "sparse-dy scale_gamma grad_ema should be nonzero (got %.6f)",
          scale_ema_sparse);
    CHECK(additive_ema_dense > 0.0f,
          "dense-dy additive_gamma grad_ema should be nonzero (got %.6f)", additive_ema_dense);
    CHECK(additive_ema_sparse > 0.0f,
          "sparse-dy additive_gamma grad_ema should be nonzero (got %.6f)", additive_ema_sparse);

    // The real property under test: density-compensated, so the sparse arm
    // should land within a real tolerance of the dense arm -- NOT at ~50%
    // of it (the pre-fix, uncompensated behavior).
    const float scale_ratio = scale_ema_sparse / scale_ema_dense;
    const float additive_ratio = additive_ema_sparse / additive_ema_dense;
    CHECK(scale_ratio > 0.7f && scale_ratio < 1.4f,
          "scale_gamma grad_ema should be density-independent after the fix -- "
          "ratio sparse/dense = %.3f (pre-fix this would land near 0.5)",
          scale_ratio);
    CHECK(additive_ratio > 0.7f && additive_ratio < 1.4f,
          "additive_gamma grad_ema should be density-independent after the fix -- "
          "ratio sparse/dense = %.3f (pre-fix this would land near 0.5)",
          additive_ratio);
}

int main() {
    test_grad_ema_density_independent();

    if (g_fail == 0) {
        std::printf("All grad_norm_divisor dy-density tests passed.\n");
    } else {
        std::printf("%d FAILURES\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
