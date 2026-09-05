// Regression test: DISLDOLayerV's new forward_sparse/backward_sparse
// (added to expose the fp32 "sisldo" sparse-input/sparse-gradient path,
// direct instruction -- the underlying sisldo_forward/disldo_backward_
// sparse_grad functions already existed generic over VALUES_TYPE, they
// were just never instantiated for VT=DeltaCSRBiValues<float> or exposed
// to Python) must produce matching forward output, dx, and post-update
// true weights as disldo_forward/disldo_backward (DISLDOLayerV's existing
// dense path) when given the SAME weights and a dense-with-zeros
// input/gradient represented two ways: as a plain dense array (dense
// path) and as the equivalent CSR (sparse path).
//
// Mirrors test_sisldo_disldo_parity.cpp exactly, just VALUES_TYPE=
// DeltaCSRBiValues<float> instead of FP4BiPacked (no quantization codes
// -- weight values are loaded directly, not through fp4_quantize/
// FP4_TABLE) and scattered-CSR only (no block4 sweep): DISLDOLayerV's
// own Python wrapper always constructs via _preseed_dense_scattered
// (dense=True), never block4-resident, for this VALUES_TYPE -- see its
// own docstring. No AQRS config sweep either (fp32 has no scale concept
// in this project's actual usage -- additive_rank=0, dynamic_rank_
// control=False always) -- base (scale_rank=1, additive_rank=0) is the
// only configuration this VALUES_TYPE is ever actually run at.
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
    w.set_scale_rank(1);
    w.output_scale_is_trainable = true;
    return w;
}

static void run_config() {
    std::printf("--- config: fp32 (DeltaCSRBiValues<float>) scattered, base ---\n");
    const std::size_t n_in = 8, n_out = 8;

    std::vector<float> dense_w(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        dense_w[i] = 1.0f + 0.07f * float(i % 7);

    // Dense-with-zeros input/gradient: SOME entries exactly zero, so the
    // CSR representation genuinely omits them.
    std::vector<float> input(n_in, 0.0f), dy(n_out, 0.0f);
    std::vector<SIZE_TYPE> in_idx, dy_idx;
    std::vector<float> in_val, dy_val;
    for (std::size_t r = 0; r < n_in; ++r) {
        if (r % 3 == 2)
            continue;
        const float v = 0.3f + 0.1f * float(r);
        input[r] = v;
        in_idx.push_back(SIZE_TYPE(r));
        in_val.push_back(v);
    }
    for (std::size_t c = 0; c < n_out; ++c) {
        if (c % 4 == 3)
            continue;
        const float v = -0.2f + 0.05f * float(c);
        dy[c] = v;
        dy_idx.push_back(SIZE_TYPE(c));
        dy_val.push_back(v);
    }
    auto in_csr = make_csr_input<SIZE_TYPE, float>(SIZE_TYPE(1), SIZE_TYPE(n_in),
                                                   {0, SIZE_TYPE(in_idx.size())}, in_idx, in_val);
    auto dy_csr = make_csr_input<SIZE_TYPE, float>(SIZE_TYPE(1), SIZE_TYPE(n_out),
                                                   {0, SIZE_TYPE(dy_idx.size())}, dy_idx, dy_val);

    Weights weights_dense = make_weights(dense_w, n_in, n_out);
    Weights weights_sparse = make_weights(dense_w, n_in, n_out);

    // ── Forward: outputs must match ────────────────────────────────────────
    std::vector<float> y_dense(n_out, 0.0f), y_sparse(n_out, 0.0f);
    disldo_forward<SIZE_TYPE, VT, COL_TYPE>(input.data(), 1, SIZE_TYPE(n_in), weights_dense,
                                            y_dense.data(), 1);
    sisldo_forward<SIZE_TYPE, VT, COL_TYPE>(in_csr, weights_sparse, y_sparse.data(), 1);
    for (std::size_t c = 0; c < n_out; ++c)
        CHECK(std::abs(y_dense[c] - y_sparse[c]) < 1e-4f,
              "forward output[%zu] diverges: dense=%.6f sparse=%.6f", c, y_dense[c], y_sparse[c]);

    // ── Backward: dx must match ────────────────────────────────────────────
    // StochasticRounding=false on both arms -- meaningless for a plain
    // float VALUES_TYPE (no quantization step to round), kept for
    // template-parameter parity with the FP4 test's own convention.
    std::vector<float> dx_dense(n_in, 0.0f), dx_sparse(n_in, 0.0f);
    std::vector<float> ni_d(n_in, 0.0f), ng_d(n_out, 0.0f);
    std::vector<float> ni_s(n_in, 0.0f), ng_s(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, VT, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), 1, SIZE_TYPE(n_in), dy.data(), weights_dense, dx_dense.data(), ni_d.data(),
        ng_d.data(), lr, 1);
    disldo_backward_sparse_grad<SIZE_TYPE, VT, COL_TYPE, RMSpropScalePolicy<float>, false>(
        input.data(), 1, weights_sparse, dy_csr, dx_sparse.data(), ni_s.data(), ng_s.data(), lr, 1);

    for (std::size_t r = 0; r < n_in; ++r)
        CHECK(std::abs(dx_dense[r] - dx_sparse[r]) < 1e-4f,
              "dx[%zu] diverges: dense=%.6f sparse=%.6f", r, dx_dense[r], dx_sparse[r]);

    // ── Post-update true weights must match ────────────────────────────────
    for (std::size_t r = 0; r < n_in; ++r) {
        auto cursor = weights_dense.connections.row_cursor(r);
        const auto& L = weights_dense.connections.layout;
        const std::size_t row_nnz = L.row_nnz(r);
        for (std::size_t e = 0; e < row_nnz; ++e) {
            const COL_TYPE col = cursor.advance();
            const std::size_t vb = L.elem_start[r] + e;
            const float w_dense = ValueAccessor<VT>::get_w(weights_dense.connections.values, vb);
            const float w_sparse = ValueAccessor<VT>::get_w(weights_sparse.connections.values, vb);
            const float true_w_dense = w_dense * weights_dense.get_scale(r, col);
            const float true_w_sparse = w_sparse * weights_sparse.get_scale(r, col);
            // Tolerance looser than forward/dx (1e-4f) but still 50x
            // tighter than the FP4 parity test's own post-update bound
            // (5e-2f, which also has to absorb FP4 quantization noise on
            // top of loop-order floating-point reassociation) -- this
            // VALUES_TYPE has no quantization step, so the only source
            // of divergence here is legitimate float32 summation-order
            // difference between the dense nested loop and the CSR-based
            // sparse loop (RMSprop's division chain amplifies rounding
            // more than the forward/dx sums do, hence looser than those).
            CHECK(std::abs(true_w_dense - true_w_sparse) < 5e-4f,
                  "true weight[%zu][%u] diverges after update: dense=%.6f sparse=%.6f", r, col,
                  true_w_dense, true_w_sparse);
        }
    }
}

int main() {
    run_config();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
