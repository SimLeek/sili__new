// Correctness check for delta_csr_combined_to_absolute (delta_csr_memory.hpp):
// must expose EVERY live synapse regardless of representation (scattered CSR
// or block4), sorted back into column order per row -- this is what
// cpu_backend.cpp's get_weights_vals()/get_indices()/get_ptrs()/
// get_importance() rely on so saving a layer doesn't silently drop any
// synapse currently promoted into block4.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); ++g_fail; } \
} while (0)

using SIZE_TYPE = int;
using COL_TYPE  = uint32_t;
using Weights   = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

int main() {
    const int n_in = 8, n_out = 8;

    // Scattered: row0->col5 (w=2.0), row1->col2 (w=-1.5).
    std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
    std::vector<SIZE_TYPE> idx;
    std::vector<float> w, imp;
    idx.push_back(5); w.push_back(2.0f); imp.push_back(0.5f);
    ptrs[1] = 1;
    idx.push_back(2); w.push_back(-1.5f); imp.push_back(0.5f);
    ptrs[2] = 2;
    for (int r = 2; r <= n_in; ++r) ptrs[r] = 2;

    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, n_in, n_out, 256, 64, 0.2f);
    weights.block4.init(n_in, n_out);
    weights.recompute_stats();

    // block4: tile (0,0) with (row0,col1)=1.0 and (row2,col3)=0.5 -- row0
    // now has BOTH a scattered entry (col5) AND a block4 entry (col1), so
    // this also checks per-row merging/sorting, not just concatenation.
    auto tile = weights.block4.get_or_create(0, 0);
    tile.at(0, 1) = fp4_quantize(1.0f) | (fp4_quantize(0.5f) << 4);
    tile.at(2, 3) = fp4_quantize(0.5f) | (fp4_quantize(0.75f) << 4);

    std::vector<SIZE_TYPE> op, oi;
    std::vector<float> ow, oimp;
    delta_csr_combined_to_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights, op, oi, ow, oimp);

    CHECK(op.size() == std::size_t(n_in + 1), "ptrs size should be n_in+1, got %zu", op.size());
    CHECK(oi.size() == 4, "should expose all 4 synapses (2 scattered + 2 block4), got %zu", oi.size());

    // row0: cols {1, 5} in ascending order (block4's col1 merged before scattered's col5).
    CHECK(op[0] == 0 && op[1] == 2, "row0 should have 2 entries: ptrs[0]=%d ptrs[1]=%d", (int)op[0], (int)op[1]);
    if (op[1] - op[0] == 2) {
        CHECK(oi[0] == 1, "row0 entry0 should be col1 (block4, sorted first), got %d", (int)oi[0]);
        CHECK(oi[1] == 5, "row0 entry1 should be col5 (scattered), got %d", (int)oi[1]);
        CHECK(std::abs(ow[0] - 1.0f) < 1e-4f, "row0 col1 weight should be 1.0, got %.4f", ow[0]);
        CHECK(std::abs(ow[1] - 2.0f) < 1e-4f, "row0 col5 weight should be 2.0, got %.4f", ow[1]);
        CHECK(std::abs(oimp[0] - 0.5f) < 1e-4f, "row0 col1 importance should be 0.5, got %.4f", oimp[0]);
    }

    // row1: scattered only, col2.
    CHECK(op[2] - op[1] == 1, "row1 should have 1 entry");
    // row2: block4 only, col3.
    CHECK(op[3] - op[2] == 1, "row2 should have 1 entry");
    if (op[3] - op[2] == 1) {
        CHECK(oi[op[2]] == 3, "row2's entry should be col3, got %d", (int)oi[op[2]]);
        CHECK(std::abs(ow[op[2]] - 0.5f) < 1e-4f, "row2 col3 weight should be 0.5, got %.4f", ow[op[2]]);
    }
    // rows 3-7: nothing.
    for (int r = 3; r < n_in; ++r)
        CHECK(op[r + 1] == op[r], "row%d should have 0 entries", r);

    // Regression: with block4 EMPTY, must match delta_csr_to_absolute exactly
    // (the fast-path branch that skips the merge entirely).
    {
        Weights w2;
        w2.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            ptrs, idx, w, imp, n_in, n_out, 256, 64, 0.2f);
        std::vector<SIZE_TYPE> op1, oi1, op2, oi2;
        std::vector<float> ow1, oimp1, ow2, oimp2;
        delta_csr_to_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(w2.connections, op1, oi1, ow1, oimp1);
        delta_csr_combined_to_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(w2, op2, oi2, ow2, oimp2);
        CHECK(op1 == op2, "empty-block4 fast path: ptrs should match delta_csr_to_absolute exactly");
        CHECK(oi1 == oi2, "empty-block4 fast path: indices should match delta_csr_to_absolute exactly");
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
