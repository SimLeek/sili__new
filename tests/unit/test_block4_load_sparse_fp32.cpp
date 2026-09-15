// Correctness check for block4_load_sparse_fp32 (delta_csr_memory.hpp) --
// the FP32 loader that skips get_or_create() for any (br,bc) block with no
// live content, unlike block4_load_dense_fp32 which allocates every block
// unconditionally. See its own docstring for the measured motivation
// (a 10%-density banded weight matrix loaded via block4_load_dense_fp32
// ran forward() only ~11% faster than fully dense -- noise-level, not the
// real speedup a genuinely sparse structure should give).
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>

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
    const std::size_t n_in = 16, n_out = 16; // 4x4 grid of block4 tiles

    // ── Sparse pattern: only tile (0,0) and tile (3,3) are live -- the
    // other 14 (br,bc) positions are entirely zero. ────────────────────
    std::vector<float> dense(n_in * n_out, 0.0f);
    std::vector<float> imp(n_in * n_out, 0.0f);
    dense[0 * n_out + 1] = 1.0f;   // tile (0,0), local (0,1)
    dense[2 * n_out + 3] = 0.5f;   // tile (0,0), local (2,3)
    dense[12 * n_out + 13] = 2.0f; // tile (3,3), local (0,1)

    Weights weights;
    // disldo_forward reads n_in/n_out from weights.connections.layout (the
    // SCATTERED side), not from block4 -- block4_load_sparse_fp32 (like
    // block4_load_dense_fp32) deliberately leaves connections untouched,
    // so this mirrors what the real DISLDOLayerV wrapper's constructor
    // already does for n_inputs()/n_outputs() bookkeeping.
    weights.connections.layout.rows = n_in;
    weights.connections.layout.cols = n_out;
    block4_load_sparse_fp32<SIZE_TYPE, COL_TYPE>(weights, dense.data(), imp.data(), n_in, n_out);

    // ── Only the 2 live blocks should exist -- 14/16 positions skipped. ──
    CHECK(weights.block4.n_tiles() == 2, "expected exactly 2 live tiles, got %zu",
          weights.block4.n_tiles());
    CHECK(bool(weights.block4.find(0, 0)), "tile (0,0) should exist");
    CHECK(bool(weights.block4.find(3, 3)), "tile (3,3) should exist");
    CHECK(!bool(weights.block4.find(1, 1)), "tile (1,1) should NOT exist (all-zero block)");
    CHECK(!bool(weights.block4.find(0, 3)), "tile (0,3) should NOT exist (all-zero block)");
    CHECK(weights.connections.nnz() == 0, "scattered side must stay untouched (0 nnz), got %zu",
          weights.connections.nnz());

    // ── forward output must match a plain dense reference. ─────────────
    std::vector<float> x(n_in, 0.0f);
    x[0] = 3.0f;  // feeds tile (0,0)'s (0,1)
    x[2] = 2.0f;  // feeds tile (0,0)'s (2,3)
    x[12] = 1.0f; // feeds tile (3,3)'s (0,1)
    std::vector<float> y(n_out, 0.0f);
    disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x.data(), 1, SIZE_TYPE(n_in),
                                                                 weights, y.data(), 1);
    for (std::size_t c = 0; c < n_out; ++c) {
        float expected = 0.0f;
        for (std::size_t r = 0; r < n_in; ++r)
            expected += x[r] * dense[r * n_out + c];
        CHECK(y[c] == expected, "forward output[%zu]: got %.6f expected %.6f", c, y[c], expected);
    }

    // ── Bit-exact cross-check: the SAME sparse pattern loaded via the
    // dense loader (which allocates every tile, including the 14 all-zero
    // ones) must give IDENTICAL forward output -- the sparse loader is a
    // pure storage-footprint optimization, zero numerical effect. ──────
    {
        Weights weights_dense;
        weights_dense.connections.layout.rows = n_in;
        weights_dense.connections.layout.cols = n_out;
        block4_load_dense_fp32<SIZE_TYPE, COL_TYPE>(weights_dense, dense.data(), imp.data(), n_in,
                                                    n_out);
        CHECK(weights_dense.block4.n_tiles() == 16,
              "dense loader should allocate all 16 tiles regardless of content, got %zu",
              weights_dense.block4.n_tiles());
        std::vector<float> y_dense(n_out, 0.0f);
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
            x.data(), 1, SIZE_TYPE(n_in), weights_dense, y_dense.data(), 1);
        for (std::size_t c = 0; c < n_out; ++c)
            CHECK(y_dense[c] == y[c],
                  "sparse-loader vs dense-loader forward MUST be bit-exact: output[%zu] "
                  "sparse=%.8f dense=%.8f",
                  c, y[c], y_dense[c]);
    }

    // ── Fully dense content: sparse loader should allocate ALL tiles too
    // (nothing to skip), matching the dense loader's tile count exactly. ─
    {
        std::vector<float> full(n_in * n_out);
        for (std::size_t i = 0; i < full.size(); ++i)
            full[i] = 1.0f + 0.01f * float(i);
        Weights weights_full_sparse_loader;
        block4_load_sparse_fp32<SIZE_TYPE, COL_TYPE>(weights_full_sparse_loader, full.data(),
                                                     imp.data(), n_in, n_out);
        CHECK(weights_full_sparse_loader.block4.n_tiles() == 16,
              "fully dense content: sparse loader should still allocate all 16 tiles, got %zu",
              weights_full_sparse_loader.block4.n_tiles());
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
