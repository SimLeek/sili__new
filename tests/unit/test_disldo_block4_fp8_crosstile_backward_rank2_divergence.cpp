// Rank-2 correctness gate for the fp8 block4 backward cross-tile pairing
// generalization (process_tile_pair_fp8 is now rank-N generic, not just
// rank==1 -- see disldo_backward.fp8_block4_cross_tile_pairing in
// docs/research/linear_disldo.rst). Same striped scattered-vs-block4
// pattern as test_disldo_block4_fp8_crosstile_backward_divergence.cpp, but
// with scale_rank=2 and a nonzero second rank channel on BOTH arms
// (identically), so the k=1 branch of the cross-tile pairing math is
// actually exercised, not just defaulted away.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cmath>
#include <cstdio>
#include <random>
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
using VALUES_TYPE = FP8BiValues;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>;

int main() {
    const std::size_t n_in = 32, n_out = 24;
    const SIZE_TYPE batch = 4;
    std::mt19937 rng(112233);
    std::uniform_real_distribution<float> wdist(1.0f, 4.0f), idist(0.0f, 1.0f), xdist(-1.0f, 1.0f),
        k1dist(0.3f, 1.5f);
    const uint32_t block_cols = uint32_t((n_out + BLOCK4_TILE - 1) / BLOCK4_TILE);

    std::vector<float> dense_w(n_in * n_out, 0.0f), dense_imp(n_in * n_out, 0.0f);
    for (std::size_t row = 0; row < n_in; ++row) {
        for (uint32_t bc = 0; bc < block_cols; ++bc) {
            if ((bc % 2) != 0)
                continue;
            const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
            const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, n_out);
            for (std::size_t col = col_lo; col < col_hi; ++col) {
                dense_w[row * n_out + col] = wdist(rng) * (rng() % 2 ? 1.0f : -1.0f);
                dense_imp[row * n_out + col] = idist(rng);
            }
        }
    }

    std::vector<float> input(std::size_t(batch) * n_in), dy(std::size_t(batch) * n_out);
    for (auto& v : input)
        v = xdist(rng);
    for (auto& v : dy)
        v = xdist(rng);

    std::vector<float> row_scale1(n_in), col_scale1(n_out);
    for (auto& v : row_scale1)
        v = k1dist(rng);
    for (auto& v : col_scale1)
        v = k1dist(rng);

    // ── Arm A: scattered CSR ────────────────────────────────────────────
    Weights weights_a;
    {
        std::vector<SIZE_TYPE> ptrs(n_in + 1);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        SIZE_TYPE cursor = 0;
        for (std::size_t row = 0; row < n_in; ++row) {
            ptrs[row] = cursor;
            for (uint32_t bc = 0; bc < block_cols; ++bc) {
                if ((bc % 2) != 0)
                    continue;
                const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
                const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, n_out);
                for (std::size_t col = col_lo; col < col_hi; ++col) {
                    idx.push_back(SIZE_TYPE(col));
                    w.push_back(fp8_decode_bits(fp8_encode_bits(dense_w[row * n_out + col])));
                    imp.push_back(fp8_decode_bits(fp8_encode_bits(dense_imp[row * n_out + col])));
                    ++cursor;
                }
            }
        }
        ptrs[n_in] = cursor;
        weights_a.connections = delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
            ptrs, idx, w, imp, n_in, n_out, std::size_t(idx.size()) * 2,
            std::size_t(idx.size()) * 2);
    }
    weights_a.out_degree.assign(n_out, SIZE_TYPE(n_in));
    weights_a.set_scale_rank(2);
    for (std::size_t row = 0; row < n_in; ++row)
        weights_a.set_value_scale_raw_k(row, 1, row_scale1[row]);
    for (std::size_t col = 0; col < n_out; ++col)
        weights_a.set_output_scale_raw_k(col, 1, col_scale1[col]);

    // ── Arm B: block4, striped ──────────────────────────────────────────
    Weights weights_b;
    {
        std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        weights_b.connections = delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
            ptrs, idx, w, imp, n_in, n_out, std::size_t(64), std::size_t(64));
    }
    weights_b.block4.init(n_in, n_out);
    weights_b.block4.set_limits(std::size_t(n_in / 4 + 1) * block_cols * 16,
                                std::size_t(n_in / 4 + 1) * block_cols * BLOCK4_TILE_SLOTS8_BYTES);
    {
        const uint32_t block_rows = uint32_t((n_in + BLOCK4_TILE - 1) / BLOCK4_TILE);
        for (uint32_t br = 0; br < block_rows; ++br) {
            const std::size_t row_lo = std::size_t(br) * BLOCK4_TILE;
            const std::size_t row_hi = std::min(row_lo + BLOCK4_TILE, n_in);
            for (uint32_t bc = 0; bc < block_cols; ++bc) {
                if ((bc % 2) != 0)
                    continue;
                const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
                const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, n_out);
                auto tile = weights_b.block4.get_or_create(br, bc);
                for (std::size_t row = row_lo; row < row_hi; ++row) {
                    const uint32_t li = uint32_t(row - row_lo);
                    for (std::size_t col = col_lo; col < col_hi; ++col) {
                        const uint32_t lj = uint32_t(col - col_lo);
                        tile.at_weight(li, lj) = fp8_encode_bits(dense_w[row * n_out + col]);
                        tile.at_importance(li, lj) = fp8_encode_bits(dense_imp[row * n_out + col]);
                    }
                }
            }
        }
    }
    weights_b.out_degree.assign(n_out, SIZE_TYPE(n_in));
    weights_b.set_scale_rank(2);
    for (std::size_t row = 0; row < n_in; ++row)
        weights_b.set_value_scale_raw_k(row, 1, row_scale1[row]);
    for (std::size_t col = 0; col < n_out; ++col)
        weights_b.set_output_scale_raw_k(col, 1, col_scale1[col]);

    // ── Forward: outputs must match ─────────────────────────────────────
    std::vector<float> y_a(std::size_t(batch) * n_out, 0.0f), y_b(std::size_t(batch) * n_out, 0.0f);
    disldo_forward<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(input.data(), batch, SIZE_TYPE(n_in),
                                                     weights_a, y_a.data(), 4);
    disldo_forward<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(input.data(), batch, SIZE_TYPE(n_in),
                                                     weights_b, y_b.data(), 4);
    double err_fwd = 0.0;
    for (std::size_t i = 0; i < y_a.size(); ++i)
        err_fwd = std::max(err_fwd, double(std::fabs(y_a[i] - y_b[i])));
    CHECK(err_fwd < 1e-2, "rank2 forward output diverges: scattered vs block4 max abs err %.6f",
          err_fwd);

    // ── Backward: dx AND post-update weights must match ──────────────────
    std::vector<float> dx_a(std::size_t(batch) * n_in, 0.0f), dx_b(std::size_t(batch) * n_in, 0.0f);
    std::vector<float> ni_a(n_in, 0.0f), ng_a(n_out, 0.0f), ni_b(n_in, 0.0f), ng_b(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, VALUES_TYPE, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_a, dx_a.data(), ni_a.data(),
        ng_a.data(), lr, 4, false, true);
    disldo_backward<SIZE_TYPE, VALUES_TYPE, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_b, dx_b.data(), ni_b.data(),
        ng_b.data(), lr, 4, false, true);
    double err_dx = 0.0;
    for (std::size_t i = 0; i < dx_a.size(); ++i)
        err_dx = std::max(err_dx, double(std::fabs(dx_a[i] - dx_b[i])));
    CHECK(err_dx < 1e-2, "rank2 dx diverges: scattered vs block4 max abs err %.6f", err_dx);

    // Post-backward forward probe.
    std::vector<float> input2(std::size_t(batch) * n_in);
    for (auto& v : input2)
        v = xdist(rng);
    std::vector<float> y2_a(std::size_t(batch) * n_out, 0.0f),
        y2_b(std::size_t(batch) * n_out, 0.0f);
    disldo_forward<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(input2.data(), batch, SIZE_TYPE(n_in),
                                                     weights_a, y2_a.data(), 4);
    disldo_forward<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(input2.data(), batch, SIZE_TYPE(n_in),
                                                     weights_b, y2_b.data(), 4);
    double err_probe = 0.0;
    for (std::size_t i = 0; i < y2_a.size(); ++i)
        err_probe = std::max(err_probe, double(std::fabs(y2_a[i] - y2_b[i])));
    CHECK(err_probe < 1e-2,
          "rank2 post-backward forward probe diverges: scattered vs block4 max abs err %.6f",
          err_probe);

    // Direct per-component scale check -- proves the k=1 mrow/mgamma/mcol
    // gradient terms (not just the summed S=get_scale() output) matched.
    double err_vs1 = 0.0, err_os1 = 0.0;
    for (std::size_t row = 0; row < n_in; ++row)
        err_vs1 = std::max(err_vs1, double(std::fabs(weights_a.get_value_scale_k(row, 1) -
                                                     weights_b.get_value_scale_k(row, 1))));
    for (std::size_t col = 0; col < n_out; ++col)
        err_os1 = std::max(err_os1, double(std::fabs(weights_a.get_output_scale_k(col, 1) -
                                                     weights_b.get_output_scale_k(col, 1))));
    CHECK(err_vs1 < 1e-4, "rank2 k=1 value_scale diverges: scattered vs block4 max abs err %.6f",
          err_vs1);
    CHECK(err_os1 < 1e-4, "rank2 k=1 output_scale diverges: scattered vs block4 max abs err %.6f",
          err_os1);

    std::printf("fp8 rank2 striped scattered-vs-block4: forward err=%.6f dx err=%.6f probe "
                "err=%.6f vs1 err=%.6f os1 err=%.6f\n",
                err_fwd, err_dx, err_probe, err_vs1, err_os1);
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
