// Correctness check for FP8's block4 dense-tile path inside
// disldo_forward/disldo_backward (linear_disldo.hpp) -- the if constexpr
// dispatch added alongside the existing FP4 code (untouched). Mirrors
// test_disldo_block4_forward.cpp's own structure (hand-placed tile,
// manual dense reference) but for a hand-promoted Block4Tile8 and real
// E4M3 values instead of FP4's table.
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

int main() {
    const int n_in = 8, n_out = 8;

    // Scattered part: row 1 -> col 2 (weight -1.5), kept OUT of block4
    // tile (0,0)'s coverage (rows/cols 4-7 for row1... wait tile (0,0)
    // covers rows/cols 0-3, so row 1 IS inside it -- use a scattered
    // entry outside the tile's row range instead: row 5 -> col 2).
    std::vector<int> ptrs(n_in + 1, 0), idx;
    std::vector<float> w, imp;
    for (int r = 0; r <= 5; ++r)
        ptrs[r] = 0;
    idx.push_back(2);
    w.push_back(-1.5f);
    imp.push_back(0.5f);
    for (int r = 6; r <= n_in; ++r)
        ptrs[r] = 1;

    SparseLinearWeightsDelta<int, FP8BiValues, uint32_t> weights;
    weights.connections = delta_csr_from_absolute<int, FP8BiValues, uint32_t>(
        ptrs, idx, w, imp, n_in, n_out, 256, 64, 0.2f);
    weights.block4.init(n_in, n_out);
    weights.recompute_stats();

    // Block4 tile (block_row=0, block_col=0): covers rows/cols 0-3.
    // Two live entries: (row=0,col=1) weight=1.0 imp=0.5, (row=2,col=3)
    // weight=0.5 imp=0.25 -- exact E4M3-representable values (powers of
    // two / simple fractions), so no quantization-error tolerance needed.
    {
        auto tile = weights.block4.get_or_create(0, 0);
        tile.at_weight(0, 1) = fp8_quantize(1.0f);
        tile.at_importance(0, 1) = fp8_quantize(0.5f);
        tile.at_weight(2, 3) = fp8_quantize(0.5f);
        tile.at_importance(2, 3) = fp8_quantize(0.25f);
    }
    CHECK(weights.block4.n_tiles() == 1, "expected exactly 1 block4 tile, got %zu",
          weights.block4.n_tiles());

    std::vector<float> x(n_in, 0.f);
    x[0] = 3.0f; // feeds block4's (row=0,col=1)
    x[2] = 2.0f; // feeds block4's (row=2,col=3)
    x[5] = 1.0f; // feeds scattered row=5

    std::vector<float> y_ref(n_out, 0.f);
    y_ref[2] = -1.5f; // scattered: x[5]*-1.5
    y_ref[1] = 3.0f;  // block4: x[0]*1.0
    y_ref[3] = 1.0f;  // block4: x[2]*0.5

    std::vector<float> y(n_out, 0.f);
    disldo_forward<int, FP8BiValues, uint32_t>(x.data(), 1, n_in, weights, y.data(), 1);

    for (int c = 0; c < n_out; ++c)
        CHECK(std::abs(y[c] - y_ref[c]) < 1e-4f, "forward output[%d]: got %.4f expected %.4f", c,
              y[c], y_ref[c]);

    // Regression: with block4 empty, output must match scattered-only.
    {
        SparseLinearWeightsDelta<int, FP8BiValues, uint32_t> weights2;
        weights2.connections = delta_csr_from_absolute<int, FP8BiValues, uint32_t>(
            ptrs, idx, w, imp, n_in, n_out, 256, 64, 0.2f);
        weights2.block4.init(n_in, n_out);
        weights2.recompute_stats();
        std::vector<float> y2(n_out, 0.f);
        disldo_forward<int, FP8BiValues, uint32_t>(x.data(), 1, n_in, weights2, y2.data(), 1);
        CHECK(std::abs(y2[2] - (-1.5f)) < 1e-4f,
              "scattered-only regression: output[2] got %.4f expected -1.5", y2[2]);
        CHECK(std::abs(y2[1]) < 1e-6f,
              "scattered-only regression: output[1] should be 0 (no block4), got %.4f", y2[1]);
    }

    // ── backward: trains the block4 tile, checks finite + real movement ──
    std::vector<float> dy(n_out, 0.f);
    dy[1] = 1.0f; // gradient only at the block4-covered output
    dy[3] = -1.0f;
    std::vector<float> dx(n_in, 0.f);
    std::vector<float> nia(n_in, 0.f), nga(n_out, 0.f);

    // Re-run forward first (disldo_backward reads no cached input of its
    // own -- caller passes x/dy explicitly, matching disldo_forward's
    // own no-state convention).
    float w_before, imp_before;
    {
        auto tile = weights.block4.find(0, 0);
        w_before = fp8_decode_bits(tile.at_weight(0, 1));
        imp_before = fp8_decode_bits(tile.at_importance(0, 1));
    }

    disldo_backward<int, FP8BiValues, uint32_t>(x.data(), 1, n_in, dy.data(), weights, dx.data(),
                                                nia.data(), nga.data(), 0.05f, 1, false, true);

    CHECK(weights.block4.n_tiles() == 1, "tile should still exist after backward, got %zu tiles",
          weights.block4.n_tiles());
    for (int c = 0; c < n_in; ++c)
        CHECK(std::isfinite(dx[c]), "dx[%d] not finite: %f", c, dx[c]);

    {
        auto tile = weights.block4.find(0, 0);
        CHECK(bool(tile), "tile (0,0) should still be findable after backward");
        const float w_after = fp8_decode_bits(tile.at_weight(0, 1));
        const float imp_after = fp8_decode_bits(tile.at_importance(0, 1));
        CHECK(std::isfinite(w_after) && std::isfinite(imp_after),
              "weight/importance not finite after backward");
        CHECK(w_after != w_before || imp_after != imp_before,
              "backward should have changed (row=0,col=1)'s weight or importance -- w %.4f->%.4f "
              "imp %.4f->%.4f",
              w_before, w_after, imp_before, imp_after);
    }

    // dx sanity: only rows feeding a nonzero-gradient column should move
    // (row 0 feeds col1 which has dy=1.0; row 2 feeds col3 which has
    // dy=-1.0; row 5 only feeds the scattered col2, dy=0 there).
    CHECK(dx[0] != 0.0f, "dx[0] should be nonzero (row 0 feeds col 1, which has nonzero dy)");
    CHECK(dx[2] != 0.0f, "dx[2] should be nonzero (row 2 feeds col 3, which has nonzero dy)");
    CHECK(dx[5] == 0.0f, "dx[5] should be zero (row 5 only feeds col 2, dy[2]=0)");

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
