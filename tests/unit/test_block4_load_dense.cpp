// Correctness check for block4_load_dense (delta_csr_memory.hpp) -- bulk
// LOADING of already-quantized dense codes directly into block4, bypassing
// scattered CSR and the importance-gated growth-insertion path entirely.
// Deliberately tests loading in isolation from quantization: codes here are
// hand-picked/round-tripped through the real fp4_quantize scalar codec, not
// produced by any "smart" scheme -- this test is about whether LOADING is
// lossless relative to whatever codes were handed in, not about whether a
// particular quantization choice was good.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); std::fflush(stdout); ++g_fail; } \
} while (0)

using SIZE_TYPE = int;
using COL_TYPE  = uint32_t;
using Weights   = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

int main() {
    const std::size_t n_in = 8, n_out = 8; // exact multiple of BLOCK4_TILE (4) -- no boundary tiles

    // Hand-picked weight values, all safely above FP4's zero-rounding floor
    // (~0.25) so every one of the 64 codes is genuinely live -- this test
    // is about loading, not about which values happen to survive FP4's
    // coarse resolution (that's a real, separate consideration for
    // whatever code chooses init scale, not this loader's concern).
    std::vector<float> dense(n_in * n_out);
    for (std::size_t i = 0; i < dense.size(); ++i)
        dense[i] = 1.0f + 0.05f * float(i % 7); // varied, all in FP4's well-represented range

    std::vector<uint8_t> weight_codes(n_in * n_out), importance_codes(n_in * n_out, 0);
    for (std::size_t i = 0; i < dense.size(); ++i)
        weight_codes[i] = fp4_quantize(dense[i]);

    Weights weights;
    weights.connections.layout.rows = n_in; // block4_load_dense reads n_in/n_out as explicit args,
    weights.connections.layout.cols = n_out; // not from .connections -- these two lines just mirror
                                              // what the real SparseLinearLayer wrapper does for
                                              // n_inputs()/n_outputs() bookkeeping elsewhere; harmless here.

    block4_load_dense<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, weight_codes.data(), importance_codes.data(), n_in, n_out);

    // ── nnz / tile population ───────────────────────────────────────────
    CHECK(weights.block4.live_synapses() == n_in * n_out,
          "live_synapses should be %zu, got %zu", n_in * n_out, weights.block4.live_synapses());
    CHECK(weights.connections.nnz() == 0,
          "scattered side must stay untouched (0 nnz), got %zu", weights.connections.nnz());

    const uint32_t block_rows = uint32_t(n_in / 4), block_cols = uint32_t(n_out / 4);
    for (uint32_t br = 0; br < block_rows; ++br) {
        for (uint32_t bc = 0; bc < block_cols; ++bc) {
            auto tile = weights.block4.find(br, bc);
            CHECK(bool(tile), "tile (%u,%u) must exist", br, bc);
            CHECK(tile && tile.count_live() == 16,
                  "tile (%u,%u) should be fully dense (16 live), got %u",
                  br, bc, bool(tile) ? tile.count_live() : 999u);
        }
    }

    // ── Losslessness: reading each slot's stored code back must match
    // exactly what was handed in (round-tripped through the SAME fp4_quantize
    // codec, so this proves the LOAD path preserves codes verbatim, not that
    // quantization itself is exact -- that's fp4_quantize's own concern). ──
    bool codes_match = true;
    for (std::size_t row = 0; row < n_in && codes_match; ++row) {
        for (std::size_t col = 0; col < n_out && codes_match; ++col) {
            const uint32_t br = uint32_t(row / 4), bc = uint32_t(col / 4);
            const uint32_t li = uint32_t(row % 4), lj = uint32_t(col % 4);
            auto tile = weights.block4.find(br, bc);
            const uint8_t stored = tile.at(li, lj) & 0x0F; // low nibble = weight code
            const uint8_t expected = weight_codes[row * n_out + col];
            if (stored != expected) {
                codes_match = false;
                CHECK(false, "row=%zu col=%zu: stored code %u != expected %u", row, col, stored, expected);
            }
        }
    }
    CHECK(codes_match, "all stored codes should round-trip losslessly");

    // ── forward_dense-equivalent check: disldo_forward's output must match
    // a numpy-style dense reference computed directly from the SAME codes
    // (fp4_decode via FP4_TABLE), not re-quantized. ────────────────────────
    std::vector<float> x(n_in, 0.0f);
    x[0] = 1.0f; // probe row 0 alone -- output[c] should equal FP4_TABLE[weight_codes[0*n_out+c]]
    std::vector<float> y(n_out, 0.0f);
    disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(x.data(), 1, SIZE_TYPE(n_in), weights, y.data(), 1);
    for (std::size_t c = 0; c < n_out; ++c) {
        const float expected = FP4_TABLE[weight_codes[c]];
        CHECK(std::abs(y[c] - expected) < 1e-4f,
              "forward output[%zu] should be %.4f (decoded code %u), got %.4f",
              c, expected, weight_codes[c], y[c]);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
