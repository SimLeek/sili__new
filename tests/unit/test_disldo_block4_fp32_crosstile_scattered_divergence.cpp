// Production correctness gate for the fp32 block4 cross-tile pairing port
// (disldo_forward AND disldo_backward, see
// disldo_forward.fp32_block4_cross_tile_pairing /
// disldo_backward.fp32_block4_cross_tile_pairing in
// docs/research/linear_disldo.rst). Mirrors test_block4_scattered_
// divergence.cpp's established pattern (identical weights, one arm stored
// entirely in scattered CSR -- untouched by this change -- one arm stored
// entirely in block4 with a CHECKERED (every-other-block-column) occupancy
// pattern, so backward's write loop actually exercises real tile PAIRS,
// not just solos), but for fp32 and specifically checkered (not fully
// dense) since that's the scenario this change targets.
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
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

int main() {
    // n_out=24 (6 block-cols) so checkered occupancy (bc=0,2,4) gives both
    // a genuine pair (bc=0,2) AND a solo (bc=4) per row -- exercises both
    // code paths in the new write loop, not just the all-paired case.
    const std::size_t n_in = 32, n_out = 24;
    const SIZE_TYPE batch = 4;
    std::mt19937 rng(556677);
    std::uniform_real_distribution<float> wdist(1.0f, 4.0f), idist(0.0f, 1.0f), xdist(-1.0f, 1.0f);

    std::vector<float> dense_w(n_in * n_out, 0.0f), dense_imp(n_in * n_out, 0.0f);
    const uint32_t block_cols = uint32_t((n_out + BLOCK4_TILE - 1) / BLOCK4_TILE);
    for (std::size_t row = 0; row < n_in; ++row) {
        for (uint32_t bc = 0; bc < block_cols; ++bc) {
            if ((bc % 2) != 0)
                continue; // checkered: only even block-columns occupied
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

    // ── Arm A: everything in scattered CSR (untouched by this change) ──────
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
                    w.push_back(dense_w[row * n_out + col]);
                    imp.push_back(dense_imp[row * n_out + col]);
                    ++cursor;
                }
            }
        }
        ptrs[n_in] = cursor;
        weights_a.connections =
            delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                ptrs, idx, w, imp, n_in, n_out, std::size_t(idx.size()) * 2,
                std::size_t(idx.size()) * 2);
    }
    weights_a.out_degree.assign(n_out, SIZE_TYPE(n_in));

    // ── Arm B: everything in block4, checkered ─────────────────────────────
    Weights weights_b;
    {
        std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        weights_b.connections =
            delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                ptrs, idx, w, imp, n_in, n_out, std::size_t(64), std::size_t(64));
    }
    weights_b.block4.init(n_in, n_out);
    weights_b.block4.set_limits(std::size_t(n_in / 4 + 1) * block_cols * 16,
                                std::size_t(n_in / 4 + 1) * block_cols * BLOCK4_TILE_SLOTS32_BYTES);
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
                        tile.set_weight(li, lj, dense_w[row * n_out + col]);
                        tile.set_importance(li, lj, dense_imp[row * n_out + col]);
                    }
                }
            }
        }
    }
    weights_b.out_degree.assign(n_out, SIZE_TYPE(n_in));

    // ── Forward: outputs must match ─────────────────────────────────────────
    std::vector<float> y_a(std::size_t(batch) * n_out, 0.0f), y_b(std::size_t(batch) * n_out, 0.0f);
    disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        input.data(), batch, SIZE_TYPE(n_in), weights_a, y_a.data(), 4);
    disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        input.data(), batch, SIZE_TYPE(n_in), weights_b, y_b.data(), 4);
    double err_fwd = 0.0;
    for (std::size_t i = 0; i < y_a.size(); ++i)
        err_fwd = std::max(err_fwd, double(std::fabs(y_a[i] - y_b[i])));
    CHECK(err_fwd < 1e-3, "forward output diverges: scattered vs block4 max abs err %.6f", err_fwd);

    // ── Backward: dx AND post-update weights must match ─────────────────────
    // Deterministic rounding (StochasticRounding=false) for a tight
    // comparison -- fp32 has no quantization noise anyway, but this matches
    // the established test_block4_scattered_divergence.cpp convention.
    std::vector<float> dx_a(std::size_t(batch) * n_in, 0.0f), dx_b(std::size_t(batch) * n_in, 0.0f);
    std::vector<float> ni_a(n_in, 0.0f), ng_a(n_out, 0.0f), ni_b(n_in, 0.0f), ng_b(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE, RMSpropScalePolicy<float>, false,
                    false>(input.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_a, dx_a.data(),
                           ni_a.data(), ng_a.data(), lr, 4, false, true);
    disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE, RMSpropScalePolicy<float>, false,
                    false>(input.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_b, dx_b.data(),
                           ni_b.data(), ng_b.data(), lr, 4, false, true);
    double err_dx = 0.0;
    for (std::size_t i = 0; i < dx_a.size(); ++i)
        err_dx = std::max(err_dx, double(std::fabs(dx_a[i] - dx_b[i])));
    CHECK(err_dx < 1e-3, "dx diverges: scattered vs block4 max abs err %.6f", err_dx);

    // Post-backward forward probe: both arms' post-update weights must
    // still produce matching forward output.
    std::vector<float> input2(std::size_t(batch) * n_in);
    for (auto& v : input2)
        v = xdist(rng);
    std::vector<float> y2_a(std::size_t(batch) * n_out, 0.0f),
        y2_b(std::size_t(batch) * n_out, 0.0f);
    disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        input2.data(), batch, SIZE_TYPE(n_in), weights_a, y2_a.data(), 4);
    disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        input2.data(), batch, SIZE_TYPE(n_in), weights_b, y2_b.data(), 4);
    double err_probe = 0.0;
    for (std::size_t i = 0; i < y2_a.size(); ++i)
        err_probe = std::max(err_probe, double(std::fabs(y2_a[i] - y2_b[i])));
    CHECK(err_probe < 1e-3,
          "post-backward forward probe diverges: scattered vs block4 max abs err %.6f", err_probe);

    std::printf("checkered scattered-vs-block4: forward err=%.6f dx err=%.6f probe err=%.6f\n",
                err_fwd, err_dx, err_probe);
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
