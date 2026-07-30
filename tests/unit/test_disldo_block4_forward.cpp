// Correctness check for block4's forward contribution inside disldo_forward
// (linear_disldo.hpp): builds a small SparseLinearWeightsDelta with SOME
// entries in the scattered CSR and a hand-placed Block4Tile, checks the
// combined output against a manual dense reference.
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); ++g_fail; } \
} while (0)

int main() {
    const int n_in = 8, n_out = 8;

    // Scattered part: row 0 -> col 5 (weight 2.0), row 1 -> col 2 (weight -1.5).
    // Kept OUT of the block4 tile's coverage (tile (0,0) covers rows/cols 0-3)
    // so the two paths are cleanly separable in this test.
    std::vector<int> ptrs(n_in + 1, 0), idx;
    std::vector<float> w, imp;
    idx.push_back(5); w.push_back(2.0f); imp.push_back(0.5f);
    ptrs[1] = 1;
    idx.push_back(2); w.push_back(-1.5f); imp.push_back(0.5f);
    ptrs[2] = 2;
    for (int r = 2; r <= n_in; ++r) ptrs[r] = 2;

    SparseLinearWeightsDelta<int, FP4BiPacked, uint32_t> weights;
    weights.connections = delta_csr_from_absolute<int, FP4BiPacked, uint32_t>(
        ptrs, idx, w, imp, n_in, n_out, 256, 64, 0.2f);
    weights.block4.init(n_in, n_out);
    weights.recompute_stats();

    // Block4 tile (block_row=0, block_col=0): covers rows/cols 0-3.
    // Place two live entries: (row=0,col=1)=1.0, (row=2,col=3)=0.5,
    // matching FP4_TABLE exact values (no quantization ambiguity).
    auto tile = weights.block4.get_or_create(0, 0);
    tile.at(0, 1) = fp4_quantize(1.0f) | (fp4_quantize(0.5f) << 4);   // row=0(local_i),col=1(local_j)
    tile.at(2, 3) = fp4_quantize(0.5f) | (fp4_quantize(0.5f) << 4);   // row=2,col=3

    std::vector<float> x(n_in, 0.f);
    x[0] = 3.0f;   // feeds block4's (row=0,col=1)
    x[1] = 1.0f;   // feeds scattered row=1
    x[2] = 2.0f;   // feeds scattered row=0? no -- feeds block4's (row=2,col=3) AND scattered row=2(none)
    // Recompute expected by hand:
    // scattered: row0->col5 w=2.0, contributes x[0]*2.0=3.0*2.0=6.0 to output[5]
    //            row1->col2 w=-1.5, contributes x[1]*-1.5=1.0*-1.5=-1.5 to output[2]
    // block4: (row0,col1) w=1.0 -> x[0]*1.0=3.0*1.0=3.0 to output[1]
    //         (row2,col3) w=0.5 -> x[2]*0.5=2.0*0.5=1.0 to output[3]
    std::vector<float> y_ref(n_out, 0.f);
    y_ref[5] = 6.0f;
    y_ref[2] = -1.5f;
    y_ref[1] = 3.0f;
    y_ref[3] = 1.0f;

    std::vector<float> y(n_out, 0.f);
    disldo_forward<int, FP4BiPacked, uint32_t>(x.data(), 1, n_in, weights, y.data(), 0.0f, 1);

    for (int c = 0; c < n_out; ++c)
        CHECK(std::abs(y[c] - y_ref[c]) < 1e-4f, "output[%d]: got %.4f expected %.4f", c, y[c], y_ref[c]);

    // Regression check: with block4 empty, output must match the
    // scattered-only contributions exactly (no change to the pre-block4
    // behavior when nothing has been promoted).
    {
        SparseLinearWeightsDelta<int, FP4BiPacked, uint32_t> w2;
        w2.connections = delta_csr_from_absolute<int, FP4BiPacked, uint32_t>(
            ptrs, idx, w, imp, n_in, n_out, 256, 64, 0.2f);
        w2.recompute_stats();

        std::vector<float> y2(n_out, 0.f);
        disldo_forward<int, FP4BiPacked, uint32_t>(x.data(), 1, n_in, w2, y2.data(), 0.0f, 1);

        std::vector<float> y2_ref(n_out, 0.f);
        y2_ref[5] = 6.0f;
        y2_ref[2] = -1.5f;
        for (int c = 0; c < n_out; ++c)
            CHECK(std::abs(y2[c] - y2_ref[c]) < 1e-4f,
                  "scattered-only output[%d]: got %.4f expected %.4f", c, y2[c], y2_ref[c]);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
