// Production correctness gate for the fp8 and fp4 block4 forward
// cross-tile pairing port (disldo_forward.fp8_block4_cross_tile_pairing /
// disldo_forward.fp4_block4_cross_tile_pairing in
// docs/research/linear_disldo.rst). Mirrors test_disldo_block4_fp32_
// crosstile_scattered_divergence.cpp's pattern: identical weights, one arm
// entirely scattered-CSR (untouched by this change), one arm entirely
// block4 with STRIPED (every-other-block-column) occupancy so both a real
// tile pair AND a real solo occur in the same row.
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
// n_out=24 (6 block-cols) -> striped occupancy bc=0,2,4 gives a real
// pair (bc=0,2) AND a real solo (bc=4) per row -- exercises both new
// code paths at once.
static const std::size_t N_IN = 32, N_OUT = 24;
static const SIZE_TYPE BATCH = 4;

static double run_fp8() {
    using VALUES_TYPE = FP8BiValues;
    using Weights = SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>;
    std::mt19937 rng(224466);
    std::uniform_real_distribution<float> wdist(1.0f, 4.0f), xdist(-1.0f, 1.0f);
    const uint32_t block_cols = uint32_t((N_OUT + BLOCK4_TILE - 1) / BLOCK4_TILE);

    std::vector<float> dense_w(N_IN * N_OUT, 0.0f);
    for (std::size_t row = 0; row < N_IN; ++row)
        for (uint32_t bc = 0; bc < block_cols; ++bc) {
            if ((bc % 2) != 0)
                continue;
            const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
            const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, N_OUT);
            for (std::size_t col = col_lo; col < col_hi; ++col)
                dense_w[row * N_OUT + col] = wdist(rng) * (rng() % 2 ? 1.0f : -1.0f);
        }

    std::vector<float> input(std::size_t(BATCH) * N_IN);
    for (auto& v : input)
        v = xdist(rng);

    Weights weights_a;
    {
        std::vector<SIZE_TYPE> ptrs(N_IN + 1);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        SIZE_TYPE cursor = 0;
        for (std::size_t row = 0; row < N_IN; ++row) {
            ptrs[row] = cursor;
            for (uint32_t bc = 0; bc < block_cols; ++bc) {
                if ((bc % 2) != 0)
                    continue;
                const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
                const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, N_OUT);
                for (std::size_t col = col_lo; col < col_hi; ++col) {
                    idx.push_back(SIZE_TYPE(col));
                    w.push_back(dense_w[row * N_OUT + col]);
                    imp.push_back(0.0f);
                    ++cursor;
                }
            }
        }
        ptrs[N_IN] = cursor;
        weights_a.connections = delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
            ptrs, idx, w, imp, N_IN, N_OUT, std::size_t(idx.size()) * 2,
            std::size_t(idx.size()) * 2);
    }
    weights_a.out_degree.assign(N_OUT, SIZE_TYPE(N_IN));

    Weights weights_b;
    {
        std::vector<SIZE_TYPE> ptrs(N_IN + 1, 0);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        weights_b.connections = delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
            ptrs, idx, w, imp, N_IN, N_OUT, std::size_t(64), std::size_t(64));
    }
    weights_b.block4.init(N_IN, N_OUT);
    weights_b.block4.set_limits(std::size_t(N_IN / 4 + 1) * block_cols * 16,
                                std::size_t(N_IN / 4 + 1) * block_cols * BLOCK4_TILE_SLOTS8_BYTES);
    {
        const uint32_t block_rows = uint32_t((N_IN + BLOCK4_TILE - 1) / BLOCK4_TILE);
        for (uint32_t br = 0; br < block_rows; ++br) {
            const std::size_t row_lo = std::size_t(br) * BLOCK4_TILE;
            const std::size_t row_hi = std::min(row_lo + BLOCK4_TILE, N_IN);
            for (uint32_t bc = 0; bc < block_cols; ++bc) {
                if ((bc % 2) != 0)
                    continue;
                const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
                const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, N_OUT);
                auto tile = weights_b.block4.get_or_create(br, bc);
                for (std::size_t row = row_lo; row < row_hi; ++row) {
                    const uint32_t li = uint32_t(row - row_lo);
                    for (std::size_t col = col_lo; col < col_hi; ++col) {
                        const uint32_t lj = uint32_t(col - col_lo);
                        tile.at_weight(li, lj) = fp8_encode_bits(dense_w[row * N_OUT + col]);
                        tile.at_importance(li, lj) = 0;
                    }
                }
            }
        }
    }
    weights_b.out_degree.assign(N_OUT, SIZE_TYPE(N_IN));

    std::vector<float> y_a(std::size_t(BATCH) * N_OUT, 0.0f), y_b(std::size_t(BATCH) * N_OUT, 0.0f);
    disldo_forward<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(input.data(), BATCH, SIZE_TYPE(N_IN),
                                                     weights_a, y_a.data(), 4);
    disldo_forward<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(input.data(), BATCH, SIZE_TYPE(N_IN),
                                                     weights_b, y_b.data(), 4);
    double err = 0.0;
    for (std::size_t i = 0; i < y_a.size(); ++i)
        err = std::max(err, double(std::fabs(y_a[i] - y_b[i])));
    CHECK(err < 1e-2, "fp8 forward output diverges: scattered vs block4 max abs err %.6f", err);
    std::printf("fp8 striped forward scattered-vs-block4 err=%.6f\n", err);
    return err;
}

static double run_fp4() {
    using VALUES_TYPE = FP4BiPacked;
    using Weights = SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>;
    std::mt19937 rng(335577);
    std::uniform_real_distribution<float> wdist(1.0f, 4.0f), xdist(-1.0f, 1.0f);
    const uint32_t block_cols = uint32_t((N_OUT + BLOCK4_TILE - 1) / BLOCK4_TILE);

    std::vector<float> dense_w(N_IN * N_OUT, 0.0f);
    for (std::size_t row = 0; row < N_IN; ++row)
        for (uint32_t bc = 0; bc < block_cols; ++bc) {
            if ((bc % 2) != 0)
                continue;
            const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
            const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, N_OUT);
            for (std::size_t col = col_lo; col < col_hi; ++col)
                dense_w[row * N_OUT + col] = wdist(rng) * (rng() % 2 ? 1.0f : -1.0f);
        }

    std::vector<float> input(std::size_t(BATCH) * N_IN);
    for (auto& v : input)
        v = xdist(rng);

    Weights weights_a;
    {
        std::vector<SIZE_TYPE> ptrs(N_IN + 1);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        SIZE_TYPE cursor = 0;
        for (std::size_t row = 0; row < N_IN; ++row) {
            ptrs[row] = cursor;
            for (uint32_t bc = 0; bc < block_cols; ++bc) {
                if ((bc % 2) != 0)
                    continue;
                const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
                const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, N_OUT);
                for (std::size_t col = col_lo; col < col_hi; ++col) {
                    idx.push_back(SIZE_TYPE(col));
                    const uint8_t code = fp4_quantize(dense_w[row * N_OUT + col]);
                    w.push_back(FP4_TABLE[code & 0x0Fu]);
                    imp.push_back(0.0f);
                    ++cursor;
                }
            }
        }
        ptrs[N_IN] = cursor;
        weights_a.connections = delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
            ptrs, idx, w, imp, N_IN, N_OUT, std::size_t(idx.size()) * 2,
            std::size_t(idx.size()) * 2);
    }
    weights_a.out_degree.assign(N_OUT, SIZE_TYPE(N_IN));

    Weights weights_b;
    {
        std::vector<SIZE_TYPE> ptrs(N_IN + 1, 0);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        weights_b.connections = delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
            ptrs, idx, w, imp, N_IN, N_OUT, std::size_t(64), std::size_t(64));
    }
    weights_b.block4.init(N_IN, N_OUT);
    weights_b.block4.set_limits(std::size_t(N_IN / 4 + 1) * block_cols * 16,
                                std::size_t(N_IN / 4 + 1) * block_cols * BLOCK4_TILE_SLOTS);
    {
        const uint32_t block_rows = uint32_t((N_IN + BLOCK4_TILE - 1) / BLOCK4_TILE);
        for (uint32_t br = 0; br < block_rows; ++br) {
            const std::size_t row_lo = std::size_t(br) * BLOCK4_TILE;
            const std::size_t row_hi = std::min(row_lo + BLOCK4_TILE, N_IN);
            for (uint32_t bc = 0; bc < block_cols; ++bc) {
                if ((bc % 2) != 0)
                    continue;
                const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
                const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, N_OUT);
                auto tile = weights_b.block4.get_or_create(br, bc);
                for (std::size_t row = row_lo; row < row_hi; ++row) {
                    const uint32_t li = uint32_t(row - row_lo);
                    for (std::size_t col = col_lo; col < col_hi; ++col) {
                        const uint32_t lj = uint32_t(col - col_lo);
                        const uint8_t wcode = fp4_quantize(dense_w[row * N_OUT + col]) & 0x0Fu;
                        // packed nibble: importance<<4 | weight, matching
                        // process_tile's own write convention.
                        tile.at(li, lj) = uint8_t((0u << 4) | wcode);
                    }
                }
            }
        }
    }
    weights_b.out_degree.assign(N_OUT, SIZE_TYPE(N_IN));

    std::vector<float> y_a(std::size_t(BATCH) * N_OUT, 0.0f), y_b(std::size_t(BATCH) * N_OUT, 0.0f);
    disldo_forward<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(input.data(), BATCH, SIZE_TYPE(N_IN),
                                                     weights_a, y_a.data(), 4);
    disldo_forward<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(input.data(), BATCH, SIZE_TYPE(N_IN),
                                                     weights_b, y_b.data(), 4);
    double err = 0.0;
    for (std::size_t i = 0; i < y_a.size(); ++i)
        err = std::max(err, double(std::fabs(y_a[i] - y_b[i])));
    CHECK(err < 1e-2, "fp4 forward output diverges: scattered vs block4 max abs err %.6f", err);
    std::printf("fp4 striped forward scattered-vs-block4 err=%.6f\n", err);
    return err;
}

int main() {
    run_fp8();
    run_fp4();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
