// Regression test: disldo_backward_sparse_grad must match disldo_backward's
// MULTI-SAMPLE-BATCH weight-update semantics, not just its single-sample
// numerics. disldo_backward accumulates grad/contrib across the WHOLE batch
// and applies exactly ONE ci/cw (RMSprop state + weight) update per synapse
// at the end -- a real batch-gradient-descent step. disldo_backward_sparse_
// grad instead applies a SEPARATE sequential ci/cw update per batch sample
// (b outer, row/tile inner, weight write-back happens inside the b loop),
// which is a different optimizer (sequential SGD across the batch dimension)
// whenever the same synapse is touched by more than one sample in a call.
// test_sisldo_disldo_parity.cpp never caught this because it only ever
// exercises batch=1, where "aggregate then apply once" and "apply once per
// (single) sample" are the identical operation.
//
// This test uses batch=2 with input/dy rows deliberately overlapping on the
// same synapses, so the divergence (if any) is forced to manifest. See
// docs/research/sisldo_ops.rst:disldo_backward_sparse_grad.batch_outer_row_
// inner_layout for the design this is probing.
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
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static Weights make_weights(const std::vector<uint8_t>& weight_codes, std::size_t n_in,
                            std::size_t n_out) {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(n_in + 1);
    std::vector<SIZE_TYPE> idx(n_in * n_out);
    std::vector<float> wv(n_in * n_out), imp(n_in * n_out, 0.0f);
    for (std::size_t r = 0; r < n_in; ++r) {
        ptrs[r] = SIZE_TYPE(r * n_out);
        for (std::size_t c = 0; c < n_out; ++c) {
            idx[r * n_out + c] = SIZE_TYPE(c);
            wv[r * n_out + c] = FP4_TABLE[weight_codes[r * n_out + c] & 0x0Fu];
        }
    }
    ptrs[n_in] = SIZE_TYPE(n_in * n_out);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, wv, imp, n_in, n_out, n_in * n_out * 2, n_in * n_out * 2);
    w.out_degree.assign(n_out, SIZE_TYPE(n_in));
    w.set_scale_rank(1);
    w.output_scale_is_trainable = true;
    return w;
}

static Weights make_weights_block4(const std::vector<uint8_t>& weight_codes, std::size_t n_in,
                                   std::size_t n_out) {
    Weights w;
    std::vector<uint8_t> importance_codes(n_in * n_out, 0);
    block4_load_dense<SIZE_TYPE, FP4BiPacked, COL_TYPE>(w, weight_codes.data(),
                                                        importance_codes.data(), n_in, n_out);
    w.set_scale_rank(1);
    w.output_scale_is_trainable = true;
    return w;
}

// batch=2, deliberately overlapping: BOTH samples touch row 0 (nonzero input)
// and BOTH have nonzero dy on column 0 -- so synapse (0,0) is hit twice in
// one call, the exact condition needed to force sequential-vs-aggregated
// divergence to appear.
static void run_multibatch(const char* name, Weights& weights_dense, Weights& weights_sparse,
                           std::size_t n_in, std::size_t n_out) {
    std::printf("--- multibatch config: %s ---\n", name);
    const SIZE_TYPE batch = 2;
    std::vector<float> input(std::size_t(batch) * n_in, 0.0f), dy(std::size_t(batch) * n_out, 0.0f);
    std::vector<SIZE_TYPE> in_ptrs = {0, 0, 0}, dy_ptrs = {0, 0, 0};
    std::vector<SIZE_TYPE> in_idx, dy_idx;
    std::vector<float> in_val, dy_val;
    for (SIZE_TYPE b = 0; b < batch; ++b) {
        for (std::size_t r = 0; r < n_in; ++r) {
            if (r % 3 == 2)
                continue; // every row (incl. row 0) present in both samples
            const float v = 0.2f + 0.05f * float(r) + 0.03f * float(b);
            input[std::size_t(b) * n_in + r] = v;
            in_idx.push_back(SIZE_TYPE(r));
            in_val.push_back(v);
        }
        in_ptrs[b + 1] = SIZE_TYPE(in_idx.size());
        for (std::size_t c = 0; c < n_out; ++c) {
            if (c % 4 == 3)
                continue; // every col (incl. col 0) present in both samples
            const float v = -0.15f + 0.04f * float(c) - 0.02f * float(b);
            dy[std::size_t(b) * n_out + c] = v;
            dy_idx.push_back(SIZE_TYPE(c));
            dy_val.push_back(v);
        }
        dy_ptrs[b + 1] = SIZE_TYPE(dy_idx.size());
    }
    auto dy_csr =
        make_csr_input<SIZE_TYPE, float>(batch, SIZE_TYPE(n_out), dy_ptrs, dy_idx, dy_val);

    std::vector<float> dx_dense(std::size_t(batch) * n_in, 0.0f),
        dx_sparse(std::size_t(batch) * n_in, 0.0f);
    std::vector<float> ni_d(n_in, 0.0f), ng_d(n_out, 0.0f);
    std::vector<float> ni_s(n_in, 0.0f), ng_s(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_dense, dx_dense.data(),
        ni_d.data(), ng_d.data(), lr, 1);
    disldo_backward_sparse_grad<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false>(
        input.data(), batch, weights_sparse, dy_csr, dx_sparse.data(), ni_s.data(), ng_s.data(), lr,
        1);

    for (std::size_t i = 0; i < dx_dense.size(); ++i)
        CHECK(std::abs(dx_dense[i] - dx_sparse[i]) < 1e-3f,
              "%s dx[%zu] diverges: dense=%.6f sparse=%.6f", name, i, dx_dense[i], dx_sparse[i]);

    for (std::size_t r = 0; r < n_in; ++r) {
        auto cursor = weights_dense.connections.row_cursor(r);
        const auto& L = weights_dense.connections.layout;
        const std::size_t row_nnz = L.row_nnz(r);
        for (std::size_t e = 0; e < row_nnz; ++e) {
            const COL_TYPE col = cursor.advance();
            const std::size_t vb = L.elem_start[r] + e;
            const float w_dense =
                ValueAccessor<FP4BiPacked>::get_w(weights_dense.connections.values, vb);
            const float w_sparse =
                ValueAccessor<FP4BiPacked>::get_w(weights_sparse.connections.values, vb);
            const float true_w_dense = w_dense * weights_dense.get_scale(r, col);
            const float true_w_sparse = w_sparse * weights_sparse.get_scale(r, col);
            CHECK(
                std::abs(true_w_dense - true_w_sparse) < 5e-2f,
                "%s true weight[%zu][%u] diverges after multi-batch update: dense=%.6f sparse=%.6f "
                "(sequential-per-sample vs aggregated-batch update mismatch?)",
                name, r, col, true_w_dense, true_w_sparse);
        }
    }
}

static void run_multibatch_block4(const char* name, Weights& weights_dense, Weights& weights_sparse,
                                  std::size_t n_in, std::size_t n_out) {
    for (std::size_t r = 0; r < n_in; ++r)
        for (std::size_t c = 0; c < n_out; ++c) {
            const uint32_t br = uint32_t(r / BLOCK4_TILE), bc = uint32_t(c / BLOCK4_TILE);
            const uint32_t li = uint32_t(r % BLOCK4_TILE), lj = uint32_t(c % BLOCK4_TILE);
            const float w_dense = FP4_TABLE[weights_dense.block4.find(br, bc).at(li, lj) & 0x0Fu];
            const float w_sparse = FP4_TABLE[weights_sparse.block4.find(br, bc).at(li, lj) & 0x0Fu];
            const float true_w_dense = w_dense * weights_dense.get_scale(r, c);
            const float true_w_sparse = w_sparse * weights_sparse.get_scale(r, c);
            CHECK(std::abs(true_w_dense - true_w_sparse) < 5e-2f,
                  "%s [block4] true weight[%zu][%zu] diverges after multi-batch update: dense=%.6f "
                  "sparse=%.6f (sequential-per-sample vs aggregated-batch update mismatch?)",
                  name, r, c, true_w_dense, true_w_sparse);
        }
}

int main() {
    const std::size_t n_in = 8, n_out = 8;
    std::vector<float> dense_w(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        dense_w[i] = 1.0f + 0.07f * float(i % 7);
    std::vector<uint8_t> weight_codes(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        weight_codes[i] = fp4_quantize(dense_w[i]);

    {
        Weights wd = make_weights(weight_codes, n_in, n_out);
        Weights ws = make_weights(weight_codes, n_in, n_out);
        run_multibatch("scattered", wd, ws, n_in, n_out);
    }
    {
        Weights wd = make_weights_block4(weight_codes, n_in, n_out);
        Weights ws = make_weights_block4(weight_codes, n_in, n_out);
        // Forward + dx already covered by run_multibatch's own dx check
        // path when routed through block4 -- reuse it, then check block4
        // weight storage specifically (different accessor than scattered).
        const SIZE_TYPE batch = 2;
        std::vector<float> input(std::size_t(batch) * n_in, 0.0f),
            dy(std::size_t(batch) * n_out, 0.0f);
        std::vector<SIZE_TYPE> dy_ptrs = {0, 0, 0};
        std::vector<SIZE_TYPE> dy_idx;
        std::vector<float> dy_val;
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            for (std::size_t r = 0; r < n_in; ++r) {
                if (r % 3 == 2)
                    continue;
                input[std::size_t(b) * n_in + r] = 0.2f + 0.05f * float(r) + 0.03f * float(b);
            }
            for (std::size_t c = 0; c < n_out; ++c) {
                if (c % 4 == 3)
                    continue;
                const float v = -0.15f + 0.04f * float(c) - 0.02f * float(b);
                dy[std::size_t(b) * n_out + c] = v;
                dy_idx.push_back(SIZE_TYPE(c));
                dy_val.push_back(v);
            }
            dy_ptrs[b + 1] = SIZE_TYPE(dy_idx.size());
        }
        auto dy_csr =
            make_csr_input<SIZE_TYPE, float>(batch, SIZE_TYPE(n_out), dy_ptrs, dy_idx, dy_val);
        std::vector<float> dx_dense(std::size_t(batch) * n_in, 0.0f),
            dx_sparse(std::size_t(batch) * n_in, 0.0f);
        std::vector<float> ni_d(n_in, 0.0f), ng_d(n_out, 0.0f);
        std::vector<float> ni_s(n_in, 0.0f), ng_s(n_out, 0.0f);
        const float lr = 0.05f;
        disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
            input.data(), batch, SIZE_TYPE(n_in), dy.data(), wd, dx_dense.data(), ni_d.data(),
            ng_d.data(), lr, 1);
        disldo_backward_sparse_grad<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>,
                                    false>(input.data(), batch, ws, dy_csr, dx_sparse.data(),
                                           ni_s.data(), ng_s.data(), lr, 1);
        for (std::size_t i = 0; i < dx_dense.size(); ++i)
            CHECK(std::abs(dx_dense[i] - dx_sparse[i]) < 1e-3f,
                  "block4 dx[%zu] diverges: dense=%.6f sparse=%.6f", i, dx_dense[i], dx_sparse[i]);
        run_multibatch_block4("block4", wd, ws, n_in, n_out);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
