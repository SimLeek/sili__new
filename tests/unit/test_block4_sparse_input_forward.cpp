// Correctness check for sisldo_forward's block4 phase (forward_sparse's
// underlying op) vs disldo_forward (forward_dense) on the SAME mixed
// scattered+block4 layer -- must match exactly (same weights, same math,
// only input representation differs). Closes a real, previously-silent
// bug: sisldo_forward never referenced weights.block4 at all before
// this, so any layer with block4-promoted synapses silently dropped their
// contribution through forward_sparse(). See TODO_DUAL_BLOCK4.md.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/sisldo_ops.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using SIZE_TYPE = int;
using COL_TYPE  = uint32_t;
using Weights   = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); ++g_fail; } \
} while (0)

int main() {
    const int n_in = 32, n_out = 32;

    // Scattered part: a handful of entries outside block4's coverage.
    std::vector<SIZE_TYPE> ptrs(n_in + 1, 0), idx;
    std::vector<float> w, imp;
    idx.push_back(20); w.push_back(1.5f); imp.push_back(0.3f);
    ptrs[9] = 1; // row 8 -> col 20 (row 8 is outside tiles (0,0)/(1,1))
    for (int r = 9; r <= n_in; ++r) ptrs[r] = 1;

    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, n_in, n_out, 4096, 128, 0.2f);
    weights.block4.init(n_in, n_out);
    weights.out_degree.assign(n_out, SIZE_TYPE(0));
    weights.recompute_stats();

    // Block4 tile (0,0): covers rows/cols 0-3. Tile (2,2): covers rows/cols 8-11.
    {
        auto tile = weights.block4.get_or_create(0, 0);
        tile.at(0, 1) = fp4_quantize(1.0f) | (fp4_quantize(0.5f) << 4);
        tile.at(2, 3) = fp4_quantize(0.5f) | (fp4_quantize(0.5f) << 4);
        tile.at(1, 0) = fp4_quantize(-1.5f) | (fp4_quantize(0.5f) << 4);
    }
    {
        auto tile = weights.block4.get_or_create(2, 2);
        tile.at(0, 0) = fp4_quantize(2.0f) | (fp4_quantize(0.5f) << 4);
        tile.at(3, 1) = fp4_quantize(-0.5f) | (fp4_quantize(0.5f) << 4);
    }

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> data_dist(-1.0f, 1.0f);
    std::bernoulli_distribution keep(0.5); // ~50% density input

    int fails_before = g_fail;
    for (int trial = 0; trial < 30; ++trial) {
        std::vector<float> x_dense(n_in, 0.0f);
        std::vector<SIZE_TYPE> x_ptrs = {0};
        std::vector<SIZE_TYPE> x_idx;
        std::vector<float> x_vals;
        for (int i = 0; i < n_in; ++i) {
            if (!keep(rng)) continue;
            const float v = data_dist(rng);
            x_dense[i] = v;
            x_idx.push_back(i);
            x_vals.push_back(v);
        }
        x_ptrs.push_back(static_cast<SIZE_TYPE>(x_idx.size()));

        std::vector<float> y_dense(n_out, 0.0f);
        disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            x_dense.data(), 1, n_in, weights, y_dense.data(), 0.0f, 4);

        CSRInput<SIZE_TYPE, float> x_sparse;
        x_sparse.rows = 1; x_sparse.cols = n_in;
        x_sparse.ptrs[0]    = std::make_shared<std::vector<SIZE_TYPE>>(x_ptrs);
        x_sparse.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(x_idx);
        x_sparse.values[0]  = std::make_shared<std::vector<float>>(x_vals);

        std::vector<float> y_sparse(n_out, 0.0f);
        sisldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            x_sparse, weights, y_sparse.data(), 0.0f, 4);

        for (int c = 0; c < n_out; ++c) {
            CHECK(std::abs(y_dense[c] - y_sparse[c]) < 1e-4f,
                  "trial %d col %d: dense=%.5f sparse=%.5f", trial, c, y_dense[c], y_sparse[c]);
        }
    }
    std::printf("30 trials at ~50%% input density: %d failures\n", g_fail - fails_before);

    // Also check num_cpus=1 and a very sparse / very dense edge.
    for (double density : {0.05, 0.95}) {
        std::bernoulli_distribution keep2(density);
        std::vector<float> x_dense(n_in, 0.0f);
        std::vector<SIZE_TYPE> x_ptrs = {0};
        std::vector<SIZE_TYPE> x_idx;
        std::vector<float> x_vals;
        for (int i = 0; i < n_in; ++i) {
            if (!keep2(rng)) continue;
            const float v = data_dist(rng);
            x_dense[i] = v; x_idx.push_back(i); x_vals.push_back(v);
        }
        x_ptrs.push_back(static_cast<SIZE_TYPE>(x_idx.size()));
        std::vector<float> y_dense(n_out, 0.0f);
        disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(x_dense.data(), 1, n_in, weights, y_dense.data(), 0.0f, 1);
        CSRInput<SIZE_TYPE, float> x_sparse;
        x_sparse.rows = 1; x_sparse.cols = n_in;
        x_sparse.ptrs[0] = std::make_shared<std::vector<SIZE_TYPE>>(x_ptrs);
        x_sparse.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(x_idx);
        x_sparse.values[0] = std::make_shared<std::vector<float>>(x_vals);
        std::vector<float> y_sparse(n_out, 0.0f);
        sisldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(x_sparse, weights, y_sparse.data(), 0.0f, 1);
        int local_fail = 0;
        for (int c = 0; c < n_out; ++c)
            if (std::abs(y_dense[c] - y_sparse[c]) >= 1e-4f) ++local_fail;
        std::printf("density=%.2f num_cpus=1: %d/%d mismatches\n", density, local_fail, n_out);
        g_fail += local_fail;
    }

    std::printf("%s (%d total failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
