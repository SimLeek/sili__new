// Regression test: block4 and scattered-CSR paths must produce matching
// forward output, dx, weight updates, and scale updates when given the
// SAME weights, inputs, and gradients. Direct motivation: test #85 (see
// JOURNAL.md/conversation) was a real bug where the scattered path's own
// first-ever value_scale update corrupted a row shared with block4 within
// the SAME disldo_backward call -- exactly the kind of scattered/block4
// divergence that's otherwise easy to miss, since most existing tests only
// ever exercise one representation at a time. This test builds two layers
// with IDENTICAL content -- one stored entirely in scattered CSR, one
// stored entirely in block4 -- and runs the same forward+backward call on
// both.
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

int main() {
    const std::size_t n_in = 8, n_out = 8;  // 2x2 block4 tiles, fully dense both ways

    // Varied but comfortably mid-range values -- avoids FP4 rounding
    // boundaries and near-zero RMSprop denominators that would otherwise
    // amplify a trivial floating-point summation-order difference
    // (scattered sums per-row; block4 sums per-tile, a genuinely different
    // order) into a different quantized code at the very last bit. 0.07
    // spacing (not 0.05) deliberately avoids landing exactly on FP4's
    // 1.0/1.5 halfway boundary (1.25) the way 1.0+0.05*(i%7) does at
    // i%7==5 -- confirmed directly: that exact coincidence made row 4
    // col 1 (dense_w=1.25) round to opposite FP4 codes between the two
    // arms after the update, a genuine floating-point non-associativity
    // effect (not a bug), not something this test should be sensitive to.
    std::vector<float> dense_w(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        dense_w[i] = 1.0f + 0.07f * float(i % 7);
    std::vector<uint8_t> weight_codes(n_in * n_out), importance_codes(n_in * n_out, 0);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        weight_codes[i] = fp4_quantize(dense_w[i]);

    std::vector<float> input(n_in), dy(n_out);
    for (std::size_t r = 0; r < n_in; ++r)  input[r] = 0.3f + 0.1f * float(r);
    for (std::size_t c = 0; c < n_out; ++c) dy[c]    = -0.2f + 0.05f * float(c);

    // ── Arm A: everything in scattered CSR, block4 left empty ─────────────
    Weights weights_a;
    {
        std::vector<SIZE_TYPE> ptrs(n_in + 1);
        std::vector<SIZE_TYPE> idx(n_in * n_out);
        std::vector<float> w(n_in * n_out), imp(n_in * n_out, 0.0f);
        for (std::size_t r = 0; r < n_in; ++r) {
            ptrs[r] = SIZE_TYPE(r * n_out);
            for (std::size_t c = 0; c < n_out; ++c) {
                idx[r * n_out + c] = SIZE_TYPE(c);
                w[r * n_out + c]   = FP4_TABLE[weight_codes[r * n_out + c] & 0x0Fu];
            }
        }
        ptrs[n_in] = SIZE_TYPE(n_in * n_out);
        weights_a.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            ptrs, idx, w, imp, n_in, n_out, n_in * n_out * 2, n_in * n_out * 2);
    }
    weights_a.out_degree.assign(n_out, SIZE_TYPE(n_in));

    // ── Arm B: everything in block4, scattered CSR left empty ─────────────
    // Built via delta_csr_from_absolute with all-empty rows, NOT by
    // hand-setting layout.rows/cols alone (test_block4_load_dense.cpp gets
    // away with that since it only ever calls disldo_forward -- but
    // disldo_backward's dead-row value_scale bootstrap unconditionally
    // calls L.row_nnz(row) = elem_end[row]-elem_start[row] for every row,
    // regardless of dc.empty(), and those arrays are never sized unless
    // built through the real constructor path -- confirmed directly: hand-
    // setting just rows/cols crashes with an out-of-bounds vector access
    // the moment disldo_backward runs).
    Weights weights_b;
    {
        std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        weights_b.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            ptrs, idx, w, imp, n_in, n_out, std::size_t(64), std::size_t(64));
    }
    block4_load_dense<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights_b, weight_codes.data(), importance_codes.data(), n_in, n_out);
    weights_b.out_degree.assign(n_out, SIZE_TYPE(n_in));

    // ── Forward: outputs must match ────────────────────────────────────────
    std::vector<float> y_a(n_out, 0.0f), y_b(n_out, 0.0f);
    disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(input.data(), 1, SIZE_TYPE(n_in), weights_a, y_a.data(), 1);
    disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(input.data(), 1, SIZE_TYPE(n_in), weights_b, y_b.data(), 1);
    for (std::size_t c = 0; c < n_out; ++c)
        CHECK(std::abs(y_a[c] - y_b[c]) < 1e-4f,
              "forward output[%zu] diverges: scattered=%.6f block4=%.6f", c, y_a[c], y_b[c]);

    // ── Backward: dx must match ────────────────────────────────────────────
    // Deterministic rounding (StochasticRounding=false, the last template
    // arg) -- required for a byte-exact comparison; the default (true)
    // would make the two arms' quantized codes diverge by design, defeating
    // the whole point of this test.
    std::vector<float> dx_a(n_in, 0.0f), dx_b(n_in, 0.0f);
    std::vector<float> ni_a(n_in, 0.0f), ng_a(n_out, 0.0f);
    std::vector<float> ni_b(n_in, 0.0f), ng_b(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), 1, SIZE_TYPE(n_in), dy.data(), weights_a, dx_a.data(),
        ni_a.data(), ng_a.data(), lr, 1);
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), 1, SIZE_TYPE(n_in), dy.data(), weights_b, dx_b.data(),
        ni_b.data(), ng_b.data(), lr, 1);

    for (std::size_t r = 0; r < n_in; ++r)
        CHECK(std::abs(dx_a[r] - dx_b[r]) < 1e-3f,
              "dx[%zu] diverges: scattered=%.6f block4=%.6f", r, dx_a[r], dx_b[r]);

    // ── Post-update true weights must match ────────────────────────────────
    for (std::size_t r = 0; r < n_in; ++r) {
        auto cursor = weights_a.connections.row_cursor(r);
        const auto& L = weights_a.connections.layout;
        const std::size_t row_nnz = L.row_nnz(r);
        for (std::size_t e = 0; e < row_nnz; ++e) {
            const COL_TYPE col = cursor.advance();
            const std::size_t c = std::size_t(col);
            const std::size_t vb = L.elem_start[r] + e;
            const float w_a = ValueAccessor<FP4BiPacked>::get_w(weights_a.connections.values, vb);

            const uint32_t br = uint32_t(r / 4), bc = uint32_t(c / 4);
            const uint32_t li = uint32_t(r % 4), lj = uint32_t(c % 4);
            auto tile = weights_b.block4.find(br, bc);
            const float w_b = FP4_TABLE[tile.at(li, lj) & 0x0Fu];

            const float true_w_a = w_a * weights_a.get_scale(r, col);
            const float true_w_b = w_b * weights_b.get_scale(r, col);
            CHECK(std::abs(true_w_a - true_w_b) < 5e-2f,
                  "true weight[%zu][%zu] diverges after update: scattered=%.6f block4=%.6f",
                  r, c, true_w_a, true_w_b);
        }
    }

    // ── value_scale / output_scale must match ──────────────────────────────
    for (std::size_t r = 0; r < n_in; ++r)
        CHECK(std::abs(weights_a.get_value_scale(r) - weights_b.get_value_scale(r)) < 1e-3f,
              "value_scale[%zu] diverges: scattered=%.6f block4=%.6f",
              r, weights_a.get_value_scale(r), weights_b.get_value_scale(r));
    for (std::size_t c = 0; c < n_out; ++c)
        CHECK(std::abs(weights_a.get_output_scale(c) - weights_b.get_output_scale(c)) < 1e-3f,
              "output_scale[%zu] diverges: scattered=%.6f block4=%.6f",
              c, weights_a.get_output_scale(c), weights_b.get_output_scale(c));

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
