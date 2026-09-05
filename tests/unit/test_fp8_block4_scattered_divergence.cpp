// FP8 counterpart of test_block4_scattered_divergence.cpp -- same
// motivation (block4 and scattered-CSR must produce matching forward
// output, dx, weight updates, and scale updates for IDENTICAL content),
// but for E4M3 instead of FP4. Direct motivation: the FP4 version of
// this test existed and passed, but FP8's block4 backward had a real,
// separate bug (S hardcoded to 1 instead of the real per-synapse
// combined_scale -- see conversation/linear_disldo.hpp's own comment
// at the FP8 block4 decode site) that this exact style of test would
// have caught immediately, since scattered's own S=1 convention is
// correct there (its cw is genuinely true-weight-space throughout) --
// an FP8-block4-vs-FP8-scattered comparison is a direct, mechanical
// way to verify block4's own convention is self-consistent with a
// known-good reference, not just "looks stable" in an end-to-end run.
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
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP8BiValues, COL_TYPE>;

int main() {
    const std::size_t n_in = 8, n_out = 8; // 2x2 block4 tiles, fully dense both ways

    // Deliberately NOT a small/near-1 magnitude range -- the whole bug
    // being guarded against only shows up once combined_scale meaningfully
    // deviates from 1.0 (see conversation: it's ~1/S^2 amplification, a
    // no-op at S=1). scale=6.0 combined with the fan-in-correcting
    // output_scale this test sets up below keeps codes well inside E4M3's
    // representable range while giving combined_scale a real, non-trivial
    // value to catch a wrong S argument.
    std::vector<float> dense_w(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        dense_w[i] = 6.0f + 0.4f * float(i % 7);
    std::vector<uint8_t> weight_codes(n_in * n_out), importance_codes(n_in * n_out, 0);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        weight_codes[i] = fp8_quantize(dense_w[i]);

    std::vector<float> input(n_in), dy(n_out);
    for (std::size_t r = 0; r < n_in; ++r)
        input[r] = 0.3f + 0.1f * float(r);
    for (std::size_t c = 0; c < n_out; ++c)
        dy[c] = -0.2f + 0.05f * float(c);

    // Non-trivial per-row/per-column scale, same value applied identically
    // to both arms below -- this is exactly the axis the bug lived on
    // (combined_scale = value_scale[row] * output_scale[col]).
    const float value_scale = 0.5f;
    const float output_scale = 0.125f;

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
                w[r * n_out + c] = fp8_decode_bits(weight_codes[r * n_out + c]);
            }
        }
        ptrs[n_in] = SIZE_TYPE(n_in * n_out);
        weights_a.connections = delta_csr_from_absolute<SIZE_TYPE, FP8BiValues, COL_TYPE>(
            ptrs, idx, w, imp, n_in, n_out, n_in * n_out * 2, n_in * n_out * 2);
    }
    weights_a.out_degree.assign(n_out, SIZE_TYPE(n_in));
    for (std::size_t r = 0; r < n_in; ++r)
        weights_a.set_value_scale_raw(SIZE_TYPE(r), value_scale);
    for (std::size_t c = 0; c < n_out; ++c)
        weights_a.set_output_scale_raw(COL_TYPE(c), output_scale);

    // ── Arm B: everything in block4, scattered CSR left empty ─────────────
    Weights weights_b;
    {
        std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        weights_b.connections = delta_csr_from_absolute<SIZE_TYPE, FP8BiValues, COL_TYPE>(
            ptrs, idx, w, imp, n_in, n_out, std::size_t(64), std::size_t(64));
    }
    block4_load_dense<SIZE_TYPE, FP8BiValues, COL_TYPE>(weights_b, weight_codes.data(),
                                                        importance_codes.data(), n_in, n_out);
    weights_b.out_degree.assign(n_out, SIZE_TYPE(n_in));
    for (std::size_t r = 0; r < n_in; ++r)
        weights_b.set_value_scale_raw(SIZE_TYPE(r), value_scale);
    for (std::size_t c = 0; c < n_out; ++c)
        weights_b.set_output_scale_raw(COL_TYPE(c), output_scale);

    // ── Forward: outputs must match ────────────────────────────────────────
    std::vector<float> y_a(n_out, 0.0f), y_b(n_out, 0.0f);
    disldo_forward<SIZE_TYPE, FP8BiValues, COL_TYPE>(input.data(), 1, SIZE_TYPE(n_in), weights_a,
                                                     y_a.data(), 1);
    disldo_forward<SIZE_TYPE, FP8BiValues, COL_TYPE>(input.data(), 1, SIZE_TYPE(n_in), weights_b,
                                                     y_b.data(), 1);
    for (std::size_t c = 0; c < n_out; ++c)
        CHECK(std::abs(y_a[c] - y_b[c]) < 1e-3f,
              "forward output[%zu] diverges: scattered=%.6f block4=%.6f", c, y_a[c], y_b[c]);

    // ── Backward: dx must match ────────────────────────────────────────────
    // Deterministic rounding (StochasticRounding=false, the last template
    // arg) -- required for a byte-exact comparison.
    std::vector<float> dx_a(n_in, 0.0f), dx_b(n_in, 0.0f);
    std::vector<float> ni_a(n_in, 0.0f), ng_a(n_out, 0.0f);
    std::vector<float> ni_b(n_in, 0.0f), ng_b(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, FP8BiValues, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), 1, SIZE_TYPE(n_in), dy.data(), weights_a, dx_a.data(), ni_a.data(),
        ng_a.data(), lr, 1);
    disldo_backward<SIZE_TYPE, FP8BiValues, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), 1, SIZE_TYPE(n_in), dy.data(), weights_b, dx_b.data(), ni_b.data(),
        ng_b.data(), lr, 1);

    for (std::size_t r = 0; r < n_in; ++r)
        CHECK(std::abs(dx_a[r] - dx_b[r]) < 1e-2f, "dx[%zu] diverges: scattered=%.6f block4=%.6f",
              r, dx_a[r], dx_b[r]);

    // ── Post-update true weights must match ────────────────────────────────
    for (std::size_t r = 0; r < n_in; ++r) {
        auto cursor = weights_a.connections.row_cursor(r);
        const auto& L = weights_a.connections.layout;
        const std::size_t row_nnz = L.row_nnz(r);
        for (std::size_t e = 0; e < row_nnz; ++e) {
            const COL_TYPE col = cursor.advance();
            const std::size_t c = std::size_t(col);
            const std::size_t vb = L.elem_start[r] + e;
            const float w_a = ValueAccessor<FP8BiValues>::get_w(weights_a.connections.values, vb);

            const uint32_t br = uint32_t(r / 4), bc = uint32_t(c / 4);
            const uint32_t li = uint32_t(r % 4), lj = uint32_t(c % 4);
            auto tile = weights_b.block4.find(br, bc);
            const float w_b = fp8_decode_bits(tile.at_weight(li, lj));

            const float true_w_a = w_a * weights_a.get_scale(r, col);
            const float true_w_b = w_b * weights_b.get_scale(r, col);
            CHECK(std::abs(true_w_a - true_w_b) < std::max(0.05f, 0.02f * std::abs(true_w_a)),
                  "true weight[%zu][%zu] diverges after update: scattered=%.6f block4=%.6f", r, c,
                  true_w_a, true_w_b);
        }
    }

    // ── value_scale / output_scale must match ──────────────────────────────
    for (std::size_t r = 0; r < n_in; ++r)
        CHECK(std::abs(weights_a.get_value_scale(r) - weights_b.get_value_scale(r)) < 1e-3f,
              "value_scale[%zu] diverges: scattered=%.6f block4=%.6f", r,
              weights_a.get_value_scale(r), weights_b.get_value_scale(r));
    for (std::size_t c = 0; c < n_out; ++c)
        CHECK(std::abs(weights_a.get_output_scale(c) - weights_b.get_output_scale(c)) < 1e-3f,
              "output_scale[%zu] diverges: scattered=%.6f block4=%.6f", c,
              weights_a.get_output_scale(c), weights_b.get_output_scale(c));

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
