// Regression test: sisldo_forward/disldo_backward_sparse_grad (sparse-input/
// sparse-gradient path) must produce matching forward output, dx, weight
// updates, and AQRS state (rank-N value_scale/output_scale, additive_u/v/
// gamma) as disldo_forward/disldo_backward (dense path) when given the SAME
// weights and a dense-with-zeros input/gradient represented two ways: as a
// plain dense array (disldo_forward/disldo_backward) and as the equivalent
// CSR (sisldo_forward/disldo_backward_sparse_grad). This is the direct proof
// that top-k-sparsifying an activation before feeding it through the sparse
// path is mathematically a no-op relative to the dense path -- the whole
// premise the sili_peridot width-doubling work depends on.
//
// Used as a TDD driver while porting AQRS (rank-N scale + additive branch +
// gamma) into disldo_backward_sparse_grad: config (a) exercises only the
// pre-AQRS path (should pass as soon as basic template-parameter parity is
// in place); config (b) exercises full AQRS and is the actual target this
// whole port is building toward.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include "../../sili/lib/headers/sisldo_ops.hpp"
#include "../../sili/lib/headers/csr.hpp"
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

// Builds a fresh, identically-initialized Weights object each call so the
// two arms (dense vs sparse) never share state.
static Weights make_weights(const std::vector<uint8_t>& weight_codes,
                             std::size_t n_in, std::size_t n_out,
                             std::size_t scale_rank, std::size_t additive_rank,
                             bool seed_additive, bool gamma_trainable) {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(n_in + 1);
    std::vector<SIZE_TYPE> idx(n_in * n_out);
    std::vector<float> wv(n_in * n_out), imp(n_in * n_out, 0.0f);
    for (std::size_t r = 0; r < n_in; ++r) {
        ptrs[r] = SIZE_TYPE(r * n_out);
        for (std::size_t c = 0; c < n_out; ++c) {
            idx[r * n_out + c] = SIZE_TYPE(c);
            wv[r * n_out + c]  = FP4_TABLE[weight_codes[r * n_out + c] & 0x0Fu];
        }
    }
    ptrs[n_in] = SIZE_TYPE(n_in * n_out);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, wv, imp, n_in, n_out, n_in * n_out * 2, n_in * n_out * 2);
    w.out_degree.assign(n_out, SIZE_TYPE(n_in));

    w.set_scale_rank(scale_rank);
    if (scale_rank > 1) {
        for (std::size_t r = 0; r < n_in; ++r)
            for (std::size_t k = 1; k < scale_rank; ++k)
                w.set_value_scale_raw_k(r, k, 0.3f + 0.05f * float((r + k) % 5));
        for (std::size_t c = 0; c < n_out; ++c)
            for (std::size_t k = 1; k < scale_rank; ++k)
                w.set_output_scale_raw_k(c, k, 0.4f + 0.05f * float((c + k) % 5));
        if (gamma_trainable) {
            w.scale_gamma_is_trainable = true;
            for (std::size_t k = 0; k < scale_rank; ++k)
                w.set_scale_gamma_raw_k(k, k == 0 ? 1.0f : 0.5f);
        }
    }
    if (additive_rank > 0 && seed_additive) {
        w.set_additive_rank(additive_rank);
        for (std::size_t r = 0; r < n_in; ++r)
            for (std::size_t k = 0; k < additive_rank; ++k)
                w.set_additive_u_raw_k(r, k, 0.1f * float((r + k) % 4 + 1));
        for (std::size_t c = 0; c < n_out; ++c)
            for (std::size_t k = 0; k < additive_rank; ++k)
                w.set_additive_v_raw_k(c, k, 0.1f * float((c + k) % 3 + 1));
        w.additive_gamma_is_trainable = gamma_trainable;
        for (std::size_t k = 0; k < additive_rank; ++k)
            w.set_additive_gamma_raw_k(k, 1.0f);
    }
    w.output_scale_is_trainable = true;
    return w;
}

// Block4-resident variant of make_weights -- same weight values, loaded via
// block4_load_dense instead of delta_csr_from_absolute, so the "sparse" arm
// below exercises sisldo_forward/disldo_backward_sparse_grad's BLOCK4 phase
// (not just the scattered-CSR phase every run_config() call above already
// covers). n_in/n_out must be BLOCK4_TILE-aligned (both are 8 here, 2 tiles).
static Weights make_weights_block4(const std::vector<uint8_t>& weight_codes,
                                    std::size_t n_in, std::size_t n_out,
                                    std::size_t scale_rank, std::size_t additive_rank,
                                    bool seed_additive, bool gamma_trainable) {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
    std::vector<SIZE_TYPE> idx;
    std::vector<float> wv, imp;
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, wv, imp, n_in, n_out, std::size_t(64), std::size_t(64));
    std::vector<uint8_t> importance_codes(n_in * n_out, 0);
    block4_load_dense<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        w, weight_codes.data(), importance_codes.data(), n_in, n_out);
    w.out_degree.assign(n_out, SIZE_TYPE(n_in));

    w.set_scale_rank(scale_rank);
    if (scale_rank > 1) {
        for (std::size_t r = 0; r < n_in; ++r)
            for (std::size_t k = 1; k < scale_rank; ++k)
                w.set_value_scale_raw_k(r, k, 0.3f + 0.05f * float((r + k) % 5));
        for (std::size_t c = 0; c < n_out; ++c)
            for (std::size_t k = 1; k < scale_rank; ++k)
                w.set_output_scale_raw_k(c, k, 0.4f + 0.05f * float((c + k) % 5));
        if (gamma_trainable) {
            w.scale_gamma_is_trainable = true;
            for (std::size_t k = 0; k < scale_rank; ++k)
                w.set_scale_gamma_raw_k(k, k == 0 ? 1.0f : 0.5f);
        }
    }
    if (additive_rank > 0 && seed_additive) {
        w.set_additive_rank(additive_rank);
        for (std::size_t r = 0; r < n_in; ++r)
            for (std::size_t k = 0; k < additive_rank; ++k)
                w.set_additive_u_raw_k(r, k, 0.1f * float((r + k) % 4 + 1));
        for (std::size_t c = 0; c < n_out; ++c)
            for (std::size_t k = 0; k < additive_rank; ++k)
                w.set_additive_v_raw_k(c, k, 0.1f * float((c + k) % 3 + 1));
        w.additive_gamma_is_trainable = gamma_trainable;
        for (std::size_t k = 0; k < additive_rank; ++k)
            w.set_additive_gamma_raw_k(k, 1.0f);
    }
    w.output_scale_is_trainable = true;
    return w;
}

static void run_config(const char* name, std::size_t scale_rank, std::size_t additive_rank,
                        bool gamma_trainable) {
    std::printf("--- config: %s (scale_rank=%zu additive_rank=%zu gamma_trainable=%d) ---\n",
                name, scale_rank, additive_rank, int(gamma_trainable));
    const std::size_t n_in = 8, n_out = 8;

    std::vector<float> dense_w(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        dense_w[i] = 1.0f + 0.07f * float(i % 7);
    std::vector<uint8_t> weight_codes(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        weight_codes[i] = fp4_quantize(dense_w[i]);

    // Dense-with-zeros input/gradient: SOME entries exactly zero, so the
    // CSR representation genuinely omits them (not just "every entry
    // happens to be present") -- the real property under test.
    std::vector<float> input(n_in, 0.0f), dy(n_out, 0.0f);
    std::vector<SIZE_TYPE> in_idx, dy_idx;
    std::vector<float> in_val, dy_val;
    for (std::size_t r = 0; r < n_in; ++r) {
        if (r % 3 == 2) continue;   // leave every 3rd entry at zero
        const float v = 0.3f + 0.1f * float(r);
        input[r] = v;
        in_idx.push_back(SIZE_TYPE(r));
        in_val.push_back(v);
    }
    for (std::size_t c = 0; c < n_out; ++c) {
        if (c % 4 == 3) continue;   // leave every 4th entry at zero
        const float v = -0.2f + 0.05f * float(c);
        dy[c] = v;
        dy_idx.push_back(SIZE_TYPE(c));
        dy_val.push_back(v);
    }
    auto in_csr = make_csr_input<SIZE_TYPE, float>(
        SIZE_TYPE(1), SIZE_TYPE(n_in), {0, SIZE_TYPE(in_idx.size())}, in_idx, in_val);
    auto dy_csr = make_csr_input<SIZE_TYPE, float>(
        SIZE_TYPE(1), SIZE_TYPE(n_out), {0, SIZE_TYPE(dy_idx.size())}, dy_idx, dy_val);

    Weights weights_dense  = make_weights(weight_codes, n_in, n_out, scale_rank, additive_rank, true, gamma_trainable);
    Weights weights_sparse = make_weights(weight_codes, n_in, n_out, scale_rank, additive_rank, true, gamma_trainable);

    // ── Forward: outputs must match ────────────────────────────────────────
    std::vector<float> y_dense(n_out, 0.0f), y_sparse(n_out, 0.0f);
    disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        input.data(), 1, SIZE_TYPE(n_in), weights_dense, y_dense.data(), 1);
    sisldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        in_csr, weights_sparse, y_sparse.data(), 1);
    for (std::size_t c = 0; c < n_out; ++c)
        CHECK(std::abs(y_dense[c] - y_sparse[c]) < 1e-3f,
              "forward output[%zu] diverges: dense=%.6f sparse=%.6f", c, y_dense[c], y_sparse[c]);

    // ── Backward: dx must match ────────────────────────────────────────────
    // StochasticRounding=false on both arms -- required for byte-exact
    // comparison (see test_block4_scattered_divergence.cpp's identical
    // convention).
    std::vector<float> dx_dense(n_in, 0.0f), dx_sparse(n_in, 0.0f);
    std::vector<float> ni_d(n_in, 0.0f), ng_d(n_out, 0.0f);
    std::vector<float> ni_s(n_in, 0.0f), ng_s(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), 1, SIZE_TYPE(n_in), dy.data(), weights_dense, dx_dense.data(),
        ni_d.data(), ng_d.data(), lr, 1);
    disldo_backward_sparse_grad<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false>(
        input.data(), 1, weights_sparse, dy_csr, dx_sparse.data(),
        ni_s.data(), ng_s.data(), lr, 1);

    for (std::size_t r = 0; r < n_in; ++r)
        CHECK(std::abs(dx_dense[r] - dx_sparse[r]) < 1e-3f,
              "dx[%zu] diverges: dense=%.6f sparse=%.6f", r, dx_dense[r], dx_sparse[r]);

    // ── Post-update true weights must match ────────────────────────────────
    for (std::size_t r = 0; r < n_in; ++r) {
        auto cursor = weights_dense.connections.row_cursor(r);
        const auto& L = weights_dense.connections.layout;
        const std::size_t row_nnz = L.row_nnz(r);
        for (std::size_t e = 0; e < row_nnz; ++e) {
            const COL_TYPE col = cursor.advance();
            const std::size_t vb = L.elem_start[r] + e;
            const float w_dense  = ValueAccessor<FP4BiPacked>::get_w(weights_dense.connections.values, vb);
            const float w_sparse = ValueAccessor<FP4BiPacked>::get_w(weights_sparse.connections.values, vb);
            const float true_w_dense  = w_dense  * weights_dense.get_scale(r, col);
            const float true_w_sparse = w_sparse * weights_sparse.get_scale(r, col);
            CHECK(std::abs(true_w_dense - true_w_sparse) < 5e-2f,
                  "true weight[%zu][%u] diverges after update: dense=%.6f sparse=%.6f",
                  r, col, true_w_dense, true_w_sparse);
        }
    }

    // ── value_scale / output_scale must match, PER COMPONENT ───────────────
    for (std::size_t r = 0; r < n_in; ++r)
        for (std::size_t k = 0; k < scale_rank; ++k)
            CHECK(std::abs(weights_dense.get_value_scale_k(r, k) - weights_sparse.get_value_scale_k(r, k)) < 1e-3f,
                  "value_scale[%zu][k=%zu] diverges: dense=%.6f sparse=%.6f",
                  r, k, weights_dense.get_value_scale_k(r, k), weights_sparse.get_value_scale_k(r, k));
    for (std::size_t c = 0; c < n_out; ++c)
        for (std::size_t k = 0; k < scale_rank; ++k)
            CHECK(std::abs(weights_dense.get_output_scale_k(c, k) - weights_sparse.get_output_scale_k(c, k)) < 1e-3f,
                  "output_scale[%zu][k=%zu] diverges: dense=%.6f sparse=%.6f",
                  c, k, weights_dense.get_output_scale_k(c, k), weights_sparse.get_output_scale_k(c, k));

    // ── additive_u/v/gamma must match ───────────────────────────────────────
    if (additive_rank > 0) {
        for (std::size_t r = 0; r < n_in; ++r)
            for (std::size_t k = 0; k < additive_rank; ++k)
                CHECK(std::abs(weights_dense.get_additive_u_k(r, k) - weights_sparse.get_additive_u_k(r, k)) < 1e-3f,
                      "additive_u[%zu][k=%zu] diverges: dense=%.6f sparse=%.6f",
                      r, k, weights_dense.get_additive_u_k(r, k), weights_sparse.get_additive_u_k(r, k));
        for (std::size_t c = 0; c < n_out; ++c)
            for (std::size_t k = 0; k < additive_rank; ++k)
                CHECK(std::abs(weights_dense.get_additive_v_k(c, k) - weights_sparse.get_additive_v_k(c, k)) < 1e-3f,
                      "additive_v[%zu][k=%zu] diverges: dense=%.6f sparse=%.6f",
                      c, k, weights_dense.get_additive_v_k(c, k), weights_sparse.get_additive_v_k(c, k));
        for (std::size_t k = 0; k < additive_rank; ++k)
            CHECK(std::abs(weights_dense.get_additive_gamma_k(k) - weights_sparse.get_additive_gamma_k(k)) < 1e-3f,
                  "additive_gamma[k=%zu] diverges: dense=%.6f sparse=%.6f",
                  k, weights_dense.get_additive_gamma_k(k), weights_sparse.get_additive_gamma_k(k));
    }
    if (scale_rank > 1 && gamma_trainable) {
        for (std::size_t k = 0; k < scale_rank; ++k)
            CHECK(std::abs(weights_dense.get_scale_gamma_k(k) - weights_sparse.get_scale_gamma_k(k)) < 1e-3f,
                  "scale_gamma[k=%zu] diverges: dense=%.6f sparse=%.6f",
                  k, weights_dense.get_scale_gamma_k(k), weights_sparse.get_scale_gamma_k(k));
    }
}

// Block4-resident sweep (Phase 3's own follow-up, per the approved plan --
// run_config() above only ever exercises scattered-CSR weights, never the
// BLOCK4 phase of sisldo_forward/disldo_backward_sparse_grad). Both arms
// (dense-path and sparse-path) load the SAME weight codes via
// block4_load_dense instead of delta_csr_from_absolute -- disldo_forward/
// disldo_backward already handle block4-resident weights transparently
// (test_block4_scattered_divergence.cpp), so this isolates whether
// sisldo_forward/disldo_backward_sparse_grad's own block4 phase (including
// this session's rank-N generalization of its value_scale update) matches.
static void run_config_block4(const char* name, std::size_t scale_rank, std::size_t additive_rank,
                               bool gamma_trainable) {
    std::printf("--- config: %s [block4] (scale_rank=%zu additive_rank=%zu gamma_trainable=%d) ---\n",
                name, scale_rank, additive_rank, int(gamma_trainable));
    const std::size_t n_in = 8, n_out = 8;   // BLOCK4_TILE-aligned (2x2 tiles)

    std::vector<float> dense_w(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        dense_w[i] = 1.0f + 0.07f * float(i % 7);
    std::vector<uint8_t> weight_codes(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        weight_codes[i] = fp4_quantize(dense_w[i]);

    std::vector<float> input(n_in, 0.0f), dy(n_out, 0.0f);
    std::vector<SIZE_TYPE> in_idx, dy_idx;
    std::vector<float> in_val, dy_val;
    for (std::size_t r = 0; r < n_in; ++r) {
        if (r % 3 == 2) continue;
        const float v = 0.3f + 0.1f * float(r);
        input[r] = v;
        in_idx.push_back(SIZE_TYPE(r));
        in_val.push_back(v);
    }
    for (std::size_t c = 0; c < n_out; ++c) {
        if (c % 4 == 3) continue;
        const float v = -0.2f + 0.05f * float(c);
        dy[c] = v;
        dy_idx.push_back(SIZE_TYPE(c));
        dy_val.push_back(v);
    }
    auto in_csr = make_csr_input<SIZE_TYPE, float>(
        SIZE_TYPE(1), SIZE_TYPE(n_in), {0, SIZE_TYPE(in_idx.size())}, in_idx, in_val);
    auto dy_csr = make_csr_input<SIZE_TYPE, float>(
        SIZE_TYPE(1), SIZE_TYPE(n_out), {0, SIZE_TYPE(dy_idx.size())}, dy_idx, dy_val);

    Weights weights_dense  = make_weights_block4(weight_codes, n_in, n_out, scale_rank, additive_rank, true, gamma_trainable);
    Weights weights_sparse = make_weights_block4(weight_codes, n_in, n_out, scale_rank, additive_rank, true, gamma_trainable);

    std::vector<float> y_dense(n_out, 0.0f), y_sparse(n_out, 0.0f);
    disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        input.data(), 1, SIZE_TYPE(n_in), weights_dense, y_dense.data(), 1);
    sisldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        in_csr, weights_sparse, y_sparse.data(), 1);
    for (std::size_t c = 0; c < n_out; ++c)
        CHECK(std::abs(y_dense[c] - y_sparse[c]) < 1e-3f,
              "[block4] forward output[%zu] diverges: dense=%.6f sparse=%.6f", c, y_dense[c], y_sparse[c]);

    std::vector<float> dx_dense(n_in, 0.0f), dx_sparse(n_in, 0.0f);
    std::vector<float> ni_d(n_in, 0.0f), ng_d(n_out, 0.0f);
    std::vector<float> ni_s(n_in, 0.0f), ng_s(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), 1, SIZE_TYPE(n_in), dy.data(), weights_dense, dx_dense.data(),
        ni_d.data(), ng_d.data(), lr, 1);
    disldo_backward_sparse_grad<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false>(
        input.data(), 1, weights_sparse, dy_csr, dx_sparse.data(),
        ni_s.data(), ng_s.data(), lr, 1);

    for (std::size_t r = 0; r < n_in; ++r)
        CHECK(std::abs(dx_dense[r] - dx_sparse[r]) < 1e-3f,
              "[block4] dx[%zu] diverges: dense=%.6f sparse=%.6f", r, dx_dense[r], dx_sparse[r]);

    for (std::size_t r = 0; r < n_in; ++r)
        for (std::size_t c = 0; c < n_out; ++c) {
            const uint32_t br = uint32_t(r / 4), bc = uint32_t(c / 4);
            const uint32_t li = uint32_t(r % 4), lj = uint32_t(c % 4);
            const float w_dense  = FP4_TABLE[weights_dense.block4.find(br, bc).at(li, lj) & 0x0Fu];
            const float w_sparse = FP4_TABLE[weights_sparse.block4.find(br, bc).at(li, lj) & 0x0Fu];
            const float true_w_dense  = w_dense  * weights_dense.get_scale(r, c);
            const float true_w_sparse = w_sparse * weights_sparse.get_scale(r, c);
            CHECK(std::abs(true_w_dense - true_w_sparse) < 5e-2f,
                  "[block4] true weight[%zu][%zu] diverges after update: dense=%.6f sparse=%.6f",
                  r, c, true_w_dense, true_w_sparse);
        }

    for (std::size_t r = 0; r < n_in; ++r)
        for (std::size_t k = 0; k < scale_rank; ++k)
            CHECK(std::abs(weights_dense.get_value_scale_k(r, k) - weights_sparse.get_value_scale_k(r, k)) < 1e-3f,
                  "[block4] value_scale[%zu][k=%zu] diverges: dense=%.6f sparse=%.6f",
                  r, k, weights_dense.get_value_scale_k(r, k), weights_sparse.get_value_scale_k(r, k));
    for (std::size_t c = 0; c < n_out; ++c)
        for (std::size_t k = 0; k < scale_rank; ++k)
            CHECK(std::abs(weights_dense.get_output_scale_k(c, k) - weights_sparse.get_output_scale_k(c, k)) < 1e-3f,
                  "[block4] output_scale[%zu][k=%zu] diverges: dense=%.6f sparse=%.6f",
                  c, k, weights_dense.get_output_scale_k(c, k), weights_sparse.get_output_scale_k(c, k));

    if (additive_rank > 0) {
        for (std::size_t r = 0; r < n_in; ++r)
            for (std::size_t k = 0; k < additive_rank; ++k)
                CHECK(std::abs(weights_dense.get_additive_u_k(r, k) - weights_sparse.get_additive_u_k(r, k)) < 1e-3f,
                      "[block4] additive_u[%zu][k=%zu] diverges: dense=%.6f sparse=%.6f",
                      r, k, weights_dense.get_additive_u_k(r, k), weights_sparse.get_additive_u_k(r, k));
        for (std::size_t c = 0; c < n_out; ++c)
            for (std::size_t k = 0; k < additive_rank; ++k)
                CHECK(std::abs(weights_dense.get_additive_v_k(c, k) - weights_sparse.get_additive_v_k(c, k)) < 1e-3f,
                      "[block4] additive_v[%zu][k=%zu] diverges: dense=%.6f sparse=%.6f",
                      c, k, weights_dense.get_additive_v_k(c, k), weights_sparse.get_additive_v_k(c, k));
        for (std::size_t k = 0; k < additive_rank; ++k)
            CHECK(std::abs(weights_dense.get_additive_gamma_k(k) - weights_sparse.get_additive_gamma_k(k)) < 1e-3f,
                  "[block4] additive_gamma[k=%zu] diverges: dense=%.6f sparse=%.6f",
                  k, weights_dense.get_additive_gamma_k(k), weights_sparse.get_additive_gamma_k(k));
    }
    if (scale_rank > 1 && gamma_trainable) {
        for (std::size_t k = 0; k < scale_rank; ++k)
            CHECK(std::abs(weights_dense.get_scale_gamma_k(k) - weights_sparse.get_scale_gamma_k(k)) < 1e-3f,
                  "[block4] scale_gamma[k=%zu] diverges: dense=%.6f sparse=%.6f",
                  k, weights_dense.get_scale_gamma_k(k), weights_sparse.get_scale_gamma_k(k));
    }
}

int main() {
    run_config("base (rank1, no additive)", 1, 0, false);
    run_config("rank-N scale only",         2, 0, false);
    run_config("additive branch only",      1, 1, false);
    run_config("full AQRS (rank-N + additive + gamma)", 2, 1, true);

    run_config_block4("base (rank1, no additive)", 1, 0, false);
    run_config_block4("rank-N scale only",         2, 0, false);
    run_config_block4("additive branch only",      1, 1, false);
    run_config_block4("full AQRS (rank-N + additive + gamma)", 2, 1, true);

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
