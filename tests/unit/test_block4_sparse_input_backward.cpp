// Correctness check for disldo_backward_sparse_grad's block4 phase
// (backward_sparse's underlying op) vs disldo_backward (backward_dense) on
// the SAME mixed scattered+block4 layer. Closes the same class of bug as
// test_block4_sparse_input_forward.cpp -- disldo_backward_sparse_grad
// never referenced weights.block4 before this. See TODO_DUAL_BLOCK4.md.
//
// Primary check is at learning_rate=0 (dx only, deterministic -- no RNG
// involved), since learning_rate>0 quantizes stochastically and the two
// implementations don't call the RNG in the same order (disldo_backward's
// SIMD batch-loop vs this function's scalar per-(li,lj) loop), so bit-exact
// matching after a real learning step isn't meaningful to assert. A
// separate lr>0 smoke test just checks for crashes/non-finite values.
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

static Weights make_mixed_layer(int n_in, int n_out) {
    std::vector<SIZE_TYPE> ptrs(n_in + 1, 0), idx;
    std::vector<float> w, imp;
    idx.push_back(20); w.push_back(1.5f); imp.push_back(0.3f);
    ptrs[9] = 1; // row 8 -> col 20 (outside tiles (0,0)/(2,2))
    for (int r = 9; r <= n_in; ++r) ptrs[r] = 1;

    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, n_in, n_out, 4096, 128, 0.2f);
    weights.block4.init(n_in, n_out);
    // Real forward/backward calls in this test run at num_cpus=4 --
    // concurrent block4 growth requires a finite tile budget (set_limits)
    // so tile_data's capacity gets pre-reserved and never reallocates
    // mid-call; see Block4Store::set_limits()'s comment. Generous cap,
    // not meant to be exercised -- this test is about the sparse-input
    // gather logic, not budget enforcement (see
    // test_block4_memory_cap_and_compression.cpp for that).
    weights.block4.set_limits(1u << 20, 1u << 20);
    weights.out_degree.assign(n_out, SIZE_TYPE(0));
    weights.recompute_stats();

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
    return weights;
}

int main() {
    const int n_in = 32, n_out = 32;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> data_dist(-1.0f, 1.0f);

    // ---- Primary: lr=0, dx must match exactly (deterministic) ----
    int fails_before = g_fail;
    for (double density : {0.5, 0.05, 0.95}) {
        std::bernoulli_distribution keep(density);
        for (int trial = 0; trial < 20; ++trial) {
            Weights weights = make_mixed_layer(n_in, n_out);
            std::vector<float> input(n_in);
            for (auto& v : input) v = data_dist(rng);

            std::vector<float> dy_dense(n_out, 0.0f);
            std::vector<SIZE_TYPE> g_ptrs = {0};
            std::vector<SIZE_TYPE> g_idx;
            std::vector<float> g_vals;
            for (int c = 0; c < n_out; ++c) {
                if (!keep(rng)) continue;
                const float v = data_dist(rng);
                dy_dense[c] = v; g_idx.push_back(c); g_vals.push_back(v);
            }
            g_ptrs.push_back(static_cast<SIZE_TYPE>(g_idx.size()));

            std::vector<float> dx_dense(n_in, 0.0f), ni_a(n_in, 0.0f), ng_a(n_out, 0.0f);
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
                input.data(), 1, n_in, dy_dense.data(), weights,
                dx_dense.data(), ni_a.data(), ng_a.data(), 0.0f, 4, false, true);

            CSRInput<SIZE_TYPE, float> dy_sparse;
            dy_sparse.rows = 1; dy_sparse.cols = n_out;
            dy_sparse.ptrs[0]    = std::make_shared<std::vector<SIZE_TYPE>>(g_ptrs);
            dy_sparse.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(g_idx);
            dy_sparse.values[0]  = std::make_shared<std::vector<float>>(g_vals);

            std::vector<float> dx_sparse(n_in, 0.0f), ni_a2(n_in, 0.0f), ng_a2(n_out, 0.0f);
            disldo_backward_sparse_grad<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
                input.data(), 1, weights, dy_sparse,
                dx_sparse.data(), ni_a2.data(), ng_a2.data(), 0.0f, 4, false);

            for (int r = 0; r < n_in; ++r)
                CHECK(std::abs(dx_dense[r] - dx_sparse[r]) < 1e-4f,
                      "density=%.2f trial %d row %d: dense=%.5f sparse=%.5f",
                      density, trial, r, dx_dense[r], dx_sparse[r]);
        }
    }
    std::printf("lr=0 dx match: %d failures across 60 trials\n", g_fail - fails_before);

    // ---- Secondary: lr>0 smoke test -- no crash/NaN. Independent stochastic
    // requantization on the two paths means dx can legitimately drift; that's
    // informational only, not a failure. ----
    fails_before = g_fail;
    {
        Weights weights_d = make_mixed_layer(n_in, n_out);
        Weights weights_s = make_mixed_layer(n_in, n_out);
        std::bernoulli_distribution keep(0.4);
        int mismatches = 0;
        for (int step = 0; step < 20; ++step) {
            std::vector<float> input(n_in);
            for (auto& v : input) v = data_dist(rng);

            std::vector<float> dy_dense(n_out, 0.0f);
            std::vector<SIZE_TYPE> g_ptrs = {0};
            std::vector<SIZE_TYPE> g_idx;
            std::vector<float> g_vals;
            for (int c = 0; c < n_out; ++c) {
                if (!keep(rng)) continue;
                const float v = data_dist(rng);
                dy_dense[c] = v; g_idx.push_back(c); g_vals.push_back(v);
            }
            g_ptrs.push_back(static_cast<SIZE_TYPE>(g_idx.size()));
            CSRInput<SIZE_TYPE, float> dy_sparse;
            dy_sparse.rows = 1; dy_sparse.cols = n_out;
            dy_sparse.ptrs[0]    = std::make_shared<std::vector<SIZE_TYPE>>(g_ptrs);
            dy_sparse.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(g_idx);
            dy_sparse.values[0]  = std::make_shared<std::vector<float>>(g_vals);

            std::vector<float> dx_d(n_in, 0.0f), ni_a(n_in, 0.0f), ng_a(n_out, 0.0f);
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
                input.data(), 1, n_in, dy_dense.data(), weights_d,
                dx_d.data(), ni_a.data(), ng_a.data(), 0.05f, 4, false, true);

            std::vector<float> dx_s(n_in, 0.0f), ni_a2(n_in, 0.0f), ng_a2(n_out, 0.0f);
            disldo_backward_sparse_grad<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
                input.data(), 1, weights_s, dy_sparse,
                dx_s.data(), ni_a2.data(), ng_a2.data(), 0.05f, 4, false);

            for (int r = 0; r < n_in; ++r) {
                CHECK(std::isfinite(dx_d[r]), "dense dx not finite at step %d row %d", step, r);
                CHECK(std::isfinite(dx_s[r]), "sparse dx not finite at step %d row %d", step, r);
                if (std::abs(dx_d[r] - dx_s[r]) > 0.5f) ++mismatches;
            }
        }
        std::printf("lr>0 smoke test: %d finite-failures, %d large dx drifts (informational, RNG-order dependent)\n",
                     g_fail - fails_before, mismatches);
    }

    std::printf("%s (%d total failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
