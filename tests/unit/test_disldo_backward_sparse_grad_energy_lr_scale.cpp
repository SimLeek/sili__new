// Arm H: per-output-neuron post-clip learning-rate multiplier for
// disldo_backward_sparse_grad, threaded as a new `energy_lr_scale`
// parameter (nullable, default nullptr = exact no-op). Multiplies
// effective_lr AT THE update_cw CALL SITE (after the clip has already
// decided sign/saturation), indexed by the OUTPUT column -- not folded
// into the per-row effective_lr, since row=input axis here, col=output
// axis (see JOURNAL.md's 2026-09-29 axis-tracing note).
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
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
using VT = DeltaCSRBiValues<float>;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, VT, COL_TYPE>;

static Weights make_weights(const std::vector<float>& dense_w, std::size_t n_in,
                            std::size_t n_out) {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(n_in + 1);
    std::vector<SIZE_TYPE> idx(n_in * n_out);
    std::vector<float> wv(n_in * n_out), imp(n_in * n_out, 0.0f);
    for (std::size_t r = 0; r < n_in; ++r) {
        ptrs[r] = SIZE_TYPE(r * n_out);
        for (std::size_t c = 0; c < n_out; ++c) {
            idx[r * n_out + c] = SIZE_TYPE(c);
            wv[r * n_out + c] = dense_w[r * n_out + c];
        }
    }
    ptrs[n_in] = SIZE_TYPE(n_in * n_out);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, VT, COL_TYPE>(
        ptrs, idx, wv, imp, n_in, n_out, n_in * n_out * 2, n_in * n_out * 2);
    w.out_degree.assign(n_out, SIZE_TYPE(n_in));
    w.set_scale_rank_max(1);
    w.set_scale_rank(1);
    w.output_scale_is_trainable = true;
    return w;
}

static Weights make_dense_input_csr_setup(const std::vector<float>& dense_w, std::size_t n_in,
                                          std::size_t n_out, std::vector<float>& input,
                                          CSRInput<SIZE_TYPE, float>& dy_csr) {
    Weights weights = make_weights(dense_w, n_in, n_out);
    input.assign(n_in, 0.0f);
    std::vector<SIZE_TYPE> dy_idx;
    std::vector<float> dy_val;
    for (std::size_t r = 0; r < n_in; ++r)
        input[r] = 0.3f + 0.1f * float(r);
    for (std::size_t c = 0; c < n_out; ++c) {
        dy_idx.push_back(SIZE_TYPE(c));
        dy_val.push_back(-0.2f + 0.05f * float(c));
    }
    dy_csr = make_csr_input<SIZE_TYPE, float>(SIZE_TYPE(1), SIZE_TYPE(n_out),
                                              {0, SIZE_TYPE(dy_idx.size())}, dy_idx, dy_val);
    return weights;
}

// ── nullptr is an exact no-op ────────────────────────────────────────────
static void test_nullptr_is_exact_noop() {
    std::printf("--- test_nullptr_is_exact_noop ---\n");
    const std::size_t n_in = 6, n_out = 6;
    std::vector<float> dense_w(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        dense_w[i] = 1.0f + 0.07f * float(i % 7);

    std::vector<float> input_a, input_b;
    CSRInput<SIZE_TYPE, float> dy_a, dy_b;
    Weights w_a = make_dense_input_csr_setup(dense_w, n_in, n_out, input_a, dy_a);
    Weights w_b = make_dense_input_csr_setup(dense_w, n_in, n_out, input_b, dy_b);

    std::vector<float> dx_a(n_in, 0.0f), dx_b(n_in, 0.0f);
    std::vector<float> ni_a(n_in, 0.0f), ng_a(n_out, 0.0f), ni_b(n_in, 0.0f), ng_b(n_out, 0.0f);
    const float lr = 0.05f;

    // a: no energy_lr_scale argument at all (default).
    disldo_backward_sparse_grad<SIZE_TYPE, VT, COL_TYPE, RMSpropScalePolicy<float>, false>(
        input_a.data(), 1, w_a, dy_a, dx_a.data(), ni_a.data(), ng_a.data(), lr, 1);
    // b: explicit nullptr energy_lr_scale, all other trailing params at their defaults.
    disldo_backward_sparse_grad<SIZE_TYPE, VT, COL_TYPE, RMSpropScalePolicy<float>, false>(
        input_b.data(), 1, w_b, dy_b, dx_b.data(), ni_b.data(), ng_b.data(), lr, 1, false, true,
        0.999f, 1e-8f, 0.0f, 1e30f, 1e30f, false, 1e30f, nullptr, nullptr, 0.9f, nullptr);

    for (std::size_t r = 0; r < n_in; ++r)
        CHECK(dx_a[r] == dx_b[r], "dx[%zu] diverges under nullptr energy_lr_scale: %.8f vs %.8f", r,
              dx_a[r], dx_b[r]);
    for (std::size_t r = 0; r < n_in; ++r) {
        auto cursor_a = w_a.connections.row_cursor(r);
        auto cursor_b = w_b.connections.row_cursor(r);
        const auto& L = w_a.connections.layout;
        const std::size_t row_nnz = L.row_nnz(r);
        for (std::size_t e = 0; e < row_nnz; ++e) {
            cursor_a.advance();
            cursor_b.advance();
            const std::size_t vb = L.elem_start[r] + e;
            const float wa = ValueAccessor<VT>::get_w(w_a.connections.values, vb);
            const float wb = ValueAccessor<VT>::get_w(w_b.connections.values, vb);
            CHECK(wa == wb, "weight[%zu,%zu] diverges under nullptr energy_lr_scale: %.8f vs %.8f",
                  r, e, wa, wb);
        }
    }
}

// ── a real energy_lr_scale array proportionally scales each output
//    column's update magnitude ──────────────────────────────────────────
static void test_scale_multiplies_column_update() {
    std::printf("--- test_scale_multiplies_column_update ---\n");
    const std::size_t n_in = 4, n_out = 4;
    std::vector<float> dense_w(n_in * n_out,
                               1.0f); // uniform weights: isolates the scale's own effect

    std::vector<float> input_1x, input_2x;
    CSRInput<SIZE_TYPE, float> dy_1x, dy_2x;
    Weights w_1x = make_dense_input_csr_setup(dense_w, n_in, n_out, input_1x, dy_1x);
    Weights w_2x = make_dense_input_csr_setup(dense_w, n_in, n_out, input_2x, dy_2x);

    std::vector<float> dx_1x(n_in, 0.0f), dx_2x(n_in, 0.0f);
    std::vector<float> ni_1(n_in, 0.0f), ng_1(n_out, 0.0f), ni_2(n_in, 0.0f), ng_2(n_out, 0.0f);
    const float lr = 0.05f;

    // scale=1.0 everywhere (baseline) vs scale=2.0 on column 1 only.
    std::vector<float> scale_ones(n_out, 1.0f);
    std::vector<float> scale_col1_2x(n_out, 1.0f);
    scale_col1_2x[1] = 2.0f;

    disldo_backward_sparse_grad<SIZE_TYPE, VT, COL_TYPE, RMSpropScalePolicy<float>, false>(
        input_1x.data(), 1, w_1x, dy_1x, dx_1x.data(), ni_1.data(), ng_1.data(), lr, 1, false, true,
        0.999f, 1e-8f, 0.0f, 1e30f, 1e30f, false, 1e30f, nullptr, nullptr, 0.9f, scale_ones.data());
    disldo_backward_sparse_grad<SIZE_TYPE, VT, COL_TYPE, RMSpropScalePolicy<float>, false>(
        input_2x.data(), 1, w_2x, dy_2x, dx_2x.data(), ni_2.data(), ng_2.data(), lr, 1, false, true,
        0.999f, 1e-8f, 0.0f, 1e30f, 1e30f, false, 1e30f, nullptr, nullptr, 0.9f,
        scale_col1_2x.data());

    for (std::size_t r = 0; r < n_in; ++r) {
        auto cursor_1 = w_1x.connections.row_cursor(r);
        auto cursor_2 = w_2x.connections.row_cursor(r);
        const auto& L = w_1x.connections.layout;
        const std::size_t row_nnz = L.row_nnz(r);
        for (std::size_t e = 0; e < row_nnz; ++e) {
            const COL_TYPE col = cursor_1.advance();
            cursor_2.advance();
            const std::size_t vb = L.elem_start[r] + e;
            const float w1 = ValueAccessor<VT>::get_w(w_1x.connections.values, vb);
            const float w2 = ValueAccessor<VT>::get_w(w_2x.connections.values, vb);
            const float delta1 = w1 - dense_w[r * n_out + col];
            const float delta2 = w2 - dense_w[r * n_out + col];
            if (col == 1) {
                CHECK(std::abs(delta2 - 2.0f * delta1) < 1e-5f,
                      "row=%zu col=1: expected 2x delta (base=%.8f scaled=%.8f)", r, delta1,
                      delta2);
            } else {
                CHECK(std::abs(delta2 - delta1) < 1e-6f,
                      "row=%zu col=%u: untouched column's delta changed (base=%.8f scaled=%.8f)", r,
                      col, delta1, delta2);
            }
        }
    }
}

int main() {
    test_nullptr_is_exact_noop();
    test_scale_multiplies_column_update();
    if (g_fail == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d FAILURES\n", g_fail);
    return 1;
}
