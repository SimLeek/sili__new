// Correctness check for FP32's block4 dense-tile path inside
// disldo_forward/disldo_backward (linear_disldo.hpp) -- the if constexpr
// dispatch added alongside the existing FP4/FP8 code (both untouched).
// Mirrors test_disldo_block4_fp8.cpp's structure (hand-placed tile, manual
// dense reference), but since float32 has NO quantization step, this test
// also asserts BIT-EXACT parity against an equivalent pure-scattered layer
// -- a strictly stronger guarantee than FP4/FP8 can offer, since there's no
// rounding error to tolerate.
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

    // Scattered part: row 5 -> col 2 (weight -1.5), outside block4 tile
    // (0,0)'s coverage (rows/cols 0-3).
    std::vector<int> ptrs(n_in + 1, 0), idx;
    std::vector<float> w, imp;
    for (int r = 0; r <= 5; ++r)
        ptrs[r] = 0;
    idx.push_back(2);
    w.push_back(-1.5f);
    imp.push_back(0.5f);
    for (int r = 6; r <= n_in; ++r)
        ptrs[r] = 1;

    SparseLinearWeightsDelta<int, DeltaCSRBiValues<float>, uint32_t> weights;
    weights.connections = delta_csr_from_absolute<int, DeltaCSRBiValues<float>, uint32_t>(
        ptrs, idx, w, imp, n_in, n_out, 256, 64, 0.2f);
    weights.block4.init(n_in, n_out);
    weights.recompute_stats();

    // Block4 tile (block_row=0, block_col=0): covers rows/cols 0-3.
    // Two live entries: (row=0,col=1) weight=1.0 imp=0.5, (row=2,col=3)
    // weight=0.5 imp=0.25 -- exact, no quantization to worry about at all.
    {
        auto tile = weights.block4.get_or_create(0, 0);
        tile.set_weight(0, 1, 1.0f);
        tile.set_importance(0, 1, 0.5f);
        tile.set_weight(2, 3, 0.5f);
        tile.set_importance(2, 3, 0.25f);
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
    disldo_forward<int, DeltaCSRBiValues<float>, uint32_t>(x.data(), 1, n_in, weights, y.data(), 1);

    // Bit-exact (not just within tolerance) -- no quantization in this path.
    for (int c = 0; c < n_out; ++c)
        CHECK(y[c] == y_ref[c], "forward output[%d]: got %.6f expected %.6f", c, y[c], y_ref[c]);

    // Regression: with block4 empty, output must match scattered-only.
    {
        SparseLinearWeightsDelta<int, DeltaCSRBiValues<float>, uint32_t> weights2;
        weights2.connections = delta_csr_from_absolute<int, DeltaCSRBiValues<float>, uint32_t>(
            ptrs, idx, w, imp, n_in, n_out, 256, 64, 0.2f);
        weights2.block4.init(n_in, n_out);
        weights2.recompute_stats();
        std::vector<float> y2(n_out, 0.f);
        disldo_forward<int, DeltaCSRBiValues<float>, uint32_t>(x.data(), 1, n_in, weights2,
                                                               y2.data(), 1);
        CHECK(y2[2] == -1.5f, "scattered-only regression: output[2] got %.4f expected -1.5", y2[2]);
        CHECK(y2[1] == 0.0f,
              "scattered-only regression: output[1] should be 0 (no block4), got %.4f", y2[1]);
    }

    // Bit-exact cross-check: an all-scattered layer holding the SAME
    // logical weights (block4 tile's two entries moved into `connections`
    // instead) must produce IDENTICAL forward output -- float32 has no
    // encode/decode step, so block4 residency must be a pure storage
    // choice with zero numerical effect, unlike FP4/FP8 (whose scattered
    // vs block4 encodings aren't bit-identical by construction).
    {
        std::vector<int> ptrs_all(n_in + 1, 0);
        std::vector<int> idx_all;
        std::vector<float> w_all, imp_all;
        // Row 0 -> col 1 (from block4), row 2 -> col 3 (from block4), row 5 -> col 2 (scattered).
        int nnz = 0;
        for (int r = 0; r <= n_in; ++r) {
            if (r == 1) {
                idx_all.push_back(1);
                w_all.push_back(1.0f);
                imp_all.push_back(0.5f);
                ++nnz;
            }
            if (r == 3) {
                idx_all.push_back(3);
                w_all.push_back(0.5f);
                imp_all.push_back(0.25f);
                ++nnz;
            }
            if (r == 6) {
                idx_all.push_back(2);
                w_all.push_back(-1.5f);
                imp_all.push_back(0.5f);
                ++nnz;
            }
            ptrs_all[r] = nnz;
        }
        SparseLinearWeightsDelta<int, DeltaCSRBiValues<float>, uint32_t> weights_all;
        weights_all.connections = delta_csr_from_absolute<int, DeltaCSRBiValues<float>, uint32_t>(
            ptrs_all, idx_all, w_all, imp_all, n_in, n_out, 256, 64, 0.2f);
        weights_all.block4.init(n_in, n_out);
        weights_all.recompute_stats();
        std::vector<float> y_all(n_out, 0.f);
        disldo_forward<int, DeltaCSRBiValues<float>, uint32_t>(x.data(), 1, n_in, weights_all,
                                                               y_all.data(), 1);
        for (int c = 0; c < n_out; ++c)
            CHECK(y_all[c] == y[c],
                  "all-scattered vs block4-resident forward MUST be bit-exact for float32: "
                  "output[%d] scattered=%.8f block4=%.8f",
                  c, y_all[c], y[c]);
    }

    // ── backward: trains the block4 tile, checks finite + real movement ──
    std::vector<float> dy(n_out, 0.f);
    dy[1] = 1.0f; // gradient only at the block4-covered output
    dy[3] = -1.0f;
    std::vector<float> dx(n_in, 0.f);
    std::vector<float> nia(n_in, 0.f), nga(n_out, 0.f);

    float w_before, imp_before;
    {
        auto tile = weights.block4.find(0, 0);
        w_before = tile.get_weight(0, 1);
        imp_before = tile.get_importance(0, 1);
    }

    disldo_backward<int, DeltaCSRBiValues<float>, uint32_t>(x.data(), 1, n_in, dy.data(), weights,
                                                            dx.data(), nia.data(), nga.data(),
                                                            0.05f, 1, false, true);

    CHECK(weights.block4.n_tiles() == 1, "tile should still exist after backward, got %zu tiles",
          weights.block4.n_tiles());
    for (int c = 0; c < n_in; ++c)
        CHECK(std::isfinite(dx[c]), "dx[%d] not finite: %f", c, dx[c]);

    {
        auto tile = weights.block4.find(0, 0);
        CHECK(bool(tile), "tile (0,0) should still be findable after backward");
        const float w_after = tile.get_weight(0, 1);
        const float imp_after = tile.get_importance(0, 1);
        CHECK(std::isfinite(w_after) && std::isfinite(imp_after),
              "weight/importance not finite after backward");
        CHECK(w_after != w_before || imp_after != imp_before,
              "backward should have changed (row=0,col=1)'s weight or importance -- w %.4f->%.4f "
              "imp %.4f->%.4f",
              w_before, w_after, imp_before, imp_after);
    }

    // dx sanity: only rows feeding a nonzero-gradient column should move.
    CHECK(dx[0] != 0.0f, "dx[0] should be nonzero (row 0 feeds col 1, which has nonzero dy)");
    CHECK(dx[2] != 0.0f, "dx[2] should be nonzero (row 2 feeds col 3, which has nonzero dy)");
    CHECK(dx[5] == 0.0f, "dx[5] should be zero (row 5 only feeds col 2, dy[2]=0)");

    // Bit-exact backward cross-check: the all-scattered layer trained on
    // the IDENTICAL x/dy/lr must land on the IDENTICAL post-update weight
    // for (row=0,col=1) as the block4-resident tile did above -- same
    // rationale as the forward cross-check.
    {
        std::vector<int> ptrs_all(n_in + 1, 0);
        std::vector<int> idx_all;
        std::vector<float> w_all, imp_all;
        int nnz = 0;
        for (int r = 0; r <= n_in; ++r) {
            if (r == 1) {
                idx_all.push_back(1);
                w_all.push_back(1.0f);
                imp_all.push_back(0.5f);
                ++nnz;
            }
            if (r == 3) {
                idx_all.push_back(3);
                w_all.push_back(0.5f);
                imp_all.push_back(0.25f);
                ++nnz;
            }
            if (r == 6) {
                idx_all.push_back(2);
                w_all.push_back(-1.5f);
                imp_all.push_back(0.5f);
                ++nnz;
            }
            ptrs_all[r] = nnz;
        }
        SparseLinearWeightsDelta<int, DeltaCSRBiValues<float>, uint32_t> weights_all;
        weights_all.connections = delta_csr_from_absolute<int, DeltaCSRBiValues<float>, uint32_t>(
            ptrs_all, idx_all, w_all, imp_all, n_in, n_out, 256, 64, 0.2f);
        weights_all.block4.init(n_in, n_out);
        weights_all.recompute_stats();
        std::vector<float> dx_all(n_in, 0.f), nia_all(n_in, 0.f), nga_all(n_out, 0.f);
        disldo_backward<int, DeltaCSRBiValues<float>, uint32_t>(
            x.data(), 1, n_in, dy.data(), weights_all, dx_all.data(), nia_all.data(),
            nga_all.data(), 0.05f, 1, false, true);
        for (int c = 0; c < n_in; ++c)
            CHECK(dx_all[c] == dx[c],
                  "all-scattered vs block4-resident backward dx MUST be bit-exact: dx[%d] "
                  "scattered=%.8f block4=%.8f",
                  c, dx_all[c], dx[c]);
        // Find (row=0,col=1)'s post-update weight in the all-scattered layer.
        auto cur = weights_all.connections.row_cursor(0);
        const std::size_t n_row0 = weights_all.connections.layout.row_nnz(0);
        float w_all_after = std::numeric_limits<float>::quiet_NaN();
        for (std::size_t e = 0; e < n_row0; ++e) {
            const uint32_t col = cur.advance();
            if (col == 1) {
                w_all_after = ValueAccessor<DeltaCSRBiValues<float>>::get_w(
                    weights_all.connections.values,
                    weights_all.connections.layout.elem_start[0] + e);
                break;
            }
        }
        const float w_after_real = weights.block4.find(0, 0).get_weight(0, 1);
        CHECK(w_all_after == w_after_real,
              "all-scattered vs block4-resident post-backward weight MUST be bit-exact for "
              "(row=0,col=1): scattered=%.8f block4=%.8f",
              w_all_after, w_after_real);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
