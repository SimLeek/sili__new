// Regression test for a real, ASan-confirmed heap-corruption bug:
// disldo_backward_sparse_grad's block4 write-back loop (sisldo_ops.hpp,
// Pass 3) tracked each tile's byte position in a row's workspace using a
// STALE length whenever an earlier tile in the same row changed its
// stored encoding (sparse<->dense, i.e. its live-slot count crossing
// Block4Store32::switch_point) during commit_dirty_tile_in_workspace --
// every later tile's position was then wrong, and block4_resize_tile_in_
// row computed a negative (wrapped-around size_t) shift length, corrupting
// the heap. Reproduced directly via AddressSanitizer (negative-size-param
// inside a memmove) with a genuinely sparse (not fully dense) block4
// layer trained repeatedly with real weight updates (lr != 0) -- a shape
// that load_dense_values could never produce (it allocates every tile
// unconditionally, so tiles almost never sit near switch_point), only
// reachable once block4_load_sparse_fp32 (load_sparse_values) existed.
//
// Fixed in two places:
//  1. sisldo_ops.hpp Pass 3: local_pos is now advanced ONCE per tile,
//     AFTER any commit, using the tile's ACTUAL post-commit length.
//  2. block4.hpp block4_resize_tile_in_row: a defense-in-depth guard --
//     if the position/length invariant is ever violated again (a
//     different bug), warn once to stderr and skip the unsafe shift
//     instead of corrupting memory or crashing.
//
// This test exercises MANY repeated real-update backward calls on a
// genuinely sparse (banded, ~25% density) layer with multiple live tiles
// per row, driving natural weight/importance movement that should cross
// switch_point in both directions over the run. Passing means: the
// process does not crash/corrupt memory (the ASan build this was found
// under is the strongest check; a plain build still catches a hard
// crash), and every dx/weight value stays finite throughout.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include "../../sili/lib/headers/sisldo_ops.hpp"
#include "../../sili/lib/headers/csr.hpp"
#include <cstdio>
#include <cmath>
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
using VT = DeltaCSRBiValues<float>;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, VT, COL_TYPE>;

// [n_in, n_out] weights, banded around the diagonal (~density fraction of
// columns live per row, contiguous run) -- same shape as bench_sili_vs_
// torch_matrix.py's banded_weights(), which is what originally surfaced
// this bug via load_sparse_values.
static std::vector<float> banded_weights(std::size_t n_in, std::size_t n_out, double density,
                                         std::mt19937& rng) {
    std::vector<float> out(n_in * n_out, 0.0f);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    const std::size_t k = std::max<std::size_t>(1, std::size_t(density * double(n_out)));
    for (std::size_t i = 0; i < n_in; ++i) {
        const std::size_t center = n_in > 1 ? (i * (n_out - 1)) / (n_in - 1) : 0;
        const std::size_t lo =
            std::min(n_out > k ? n_out - k : 0, center >= k / 2 ? center - k / 2 : 0);
        for (std::size_t j = lo; j < lo + k && j < n_out; ++j)
            out[i * n_out + j] = nd(rng);
    }
    return out;
}

int main() {
    const std::size_t n_in = 64, n_out = 64; // 16x16 tile grid -- multiple tiles/row when banded
    std::mt19937 rng(0);

    std::vector<float> dense_w = banded_weights(n_in, n_out, 0.25, rng);
    std::vector<float> importance_values(n_in * n_out, 0.0f);

    Weights weights;
    // Empty scattered side, properly initialized (not just .rows/.cols
    // set by hand) -- recompute_stats() walks every row's row_nnz(),
    // which needs elem_start/elem_end/byte_start/byte_end sized, matching
    // make_weights_block4()'s pattern in test_sisldo_disldo_parity_fp32.cpp.
    {
        std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> wv, imp;
        weights.connections = delta_csr_from_absolute<SIZE_TYPE, VT, COL_TYPE>(
            ptrs, idx, wv, imp, n_in, n_out, std::size_t(64), std::size_t(64));
    }
    block4_load_sparse_fp32<SIZE_TYPE, COL_TYPE>(weights, dense_w.data(), importance_values.data(),
                                                 n_in, n_out);
    weights.recompute_stats();
    weights.out_degree.assign(n_out, SIZE_TYPE(n_in));
    const std::size_t full_tiles = (n_in / 4) * (n_out / 4);
    std::printf("sparse-loaded tiles: %zu / %zu\n", weights.block4.n_tiles(), full_tiles);
    CHECK(weights.block4.n_tiles() < full_tiles,
          "banded load should be genuinely sparser than full (%zu tiles), got %zu", full_tiles,
          weights.block4.n_tiles());

    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> input(n_in, 0.0f);
    std::vector<SIZE_TYPE> dy_idx;
    std::vector<float> dy_val;
    for (std::size_t i = 0; i < n_in; ++i)
        input[i] = nd(rng);
    // Dense gradient (every output column touched) -- maximizes how many
    // tiles per row see real, varied gradient signal across iterations,
    // matching the real crash's trigger conditions.
    for (std::size_t c = 0; c < n_out; ++c) {
        dy_idx.push_back(SIZE_TYPE(c));
        dy_val.push_back(nd(rng));
    }
    auto dy_csr = make_csr_input<SIZE_TYPE, float>(SIZE_TYPE(1), SIZE_TYPE(n_out),
                                                   {0, SIZE_TYPE(dy_idx.size())}, dy_idx, dy_val);

    const int n_calls = 500;
    for (int call = 0; call < n_calls; ++call) {
        std::vector<float> dx(n_in, 0.0f), nia(n_in, 0.0f), nga(n_out, 0.0f);
        disldo_backward_sparse_grad<SIZE_TYPE, VT, COL_TYPE, RMSpropScalePolicy<float>, true>(
            input.data(), 1, weights, dy_csr, dx.data(), nia.data(), nga.data(), 1e-3f, 1, true);
        for (std::size_t i = 0; i < n_in; ++i)
            CHECK(std::isfinite(dx[i]), "call %d: dx[%zu] not finite: %f", call, i, dx[i]);
        // Refresh dy each call so different synapses see nonzero
        // grad/contrib over the run (a fixed dy converges and stops
        // moving importance, which stops exercising the sparse<->dense
        // transition this test targets).
        for (auto& v : dy_val)
            v = nd(rng);
        dy_csr = make_csr_input<SIZE_TYPE, float>(SIZE_TYPE(1), SIZE_TYPE(n_out),
                                                  {0, SIZE_TYPE(dy_idx.size())}, dy_idx, dy_val);
    }

    // Surviving 500 real-update iterations on a genuinely sparse layer
    // without crashing/corrupting memory (ASan build) or producing
    // non-finite output is the actual regression check. Spot-check final
    // weights are finite too.
    for (std::size_t br = 0; br < n_in / 4; ++br) {
        for (std::size_t bc = 0; bc < n_out / 4; ++bc) {
            auto tile = weights.block4.find(uint32_t(br), uint32_t(bc));
            if (!tile)
                continue;
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj)
                    CHECK(std::isfinite(tile.get_weight(li, lj)),
                          "final weight at tile(%zu,%zu)[%u][%u] not finite", br, bc, li, lj);
        }
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
