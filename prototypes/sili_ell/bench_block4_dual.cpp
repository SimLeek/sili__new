// bench_block4_dual.cpp
// ===========================================================================
// User's design: two parallel weight matrices whose outputs sum (y = A(x) +
// B(x)) -- A is sili's existing disldo (per-synapse ULEB128 delta-CSR,
// unchanged), B is a NEW coarse block-CSR: same ULEB128-delta indexing
// scheme, but indices address 4x4 BLOCKS (row/col divided by 4), and each
// active block stores a fully DENSE 4x4 tile (16 floats, zero-filled where
// empty) -- reusing the block kernel already validated in
// bench_block_kernel.cpp (real ~1.44-2x speedup on this CPU, gather-free).
// No permute/shuffle instructions anywhere (direct critique of a fancier
// local-permutation idea considered first) -- a block's output write is a
// FIXED, contiguous 4-wide range (y[4*bi : 4*bi+4]), known from the outer
// loop index, not a scatter to a data-dependent address.
//
// This file benchmarks ONLY matrix B's own forward throughput across a
// range of block-fill-rates (20%-100%), to find the real breakeven density
// against disldo's own measured per-synapse rate (~700-820 Msyn/s, from
// sweep_disldo_density.cpp) -- checking the user's "should work well from
// 20% local sparsity" claim rather than assuming it.
// ===========================================================================
#include "../../sili/lib/headers/delta_csr_types.hpp"  // uleb128_encode/decode
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Coarse block-CSR: block_ptrs[block_row+1] = cumulative block count;
// block_col_bytes = ULEB128-delta-encoded block-column indices (exactly
// sili's existing scheme, just at block granularity); block_data = 16
// floats per active block, row-major within the tile (contiguous).
struct Block4CSR {
    uint32_t n_block_rows, n_block_cols;
    std::vector<uint32_t> block_ptrs;      // element count, not bytes
    std::vector<uint8_t>  block_col_bytes; // ULEB128 deltas
    std::vector<uint32_t> block_byte_ptrs; // byte offset per block-row (for decode start)
    std::vector<float>    block_data;      // 16 floats/block, contiguous
};

static Block4CSR build_block4csr(uint32_t M, uint32_t N,
                                  const std::vector<std::vector<uint32_t>>& active_block_cols_per_row,
                                  std::mt19937& wrng) {
    Block4CSR bc;
    bc.n_block_rows = M / 4;
    bc.n_block_cols = N / 4;
    bc.block_ptrs.assign(bc.n_block_rows + 1, 0);
    bc.block_byte_ptrs.assign(bc.n_block_rows + 1, 0);
    std::uniform_real_distribution<float> wv(-1, 1);

    for (uint32_t r = 0; r < bc.n_block_rows; ++r) {
        uint32_t prev = 0;
        for (uint32_t bc_col : active_block_cols_per_row[r]) {
            uint8_t tmp[6];
            std::size_t n = uleb128_encode<uint32_t>(bc_col - prev, tmp);
            bc.block_col_bytes.insert(bc.block_col_bytes.end(), tmp, tmp + n);
            prev = bc_col;
            for (int k = 0; k < 16; ++k) bc.block_data.push_back(wv(wrng));
        }
        bc.block_ptrs[r + 1] = bc.block_ptrs[r] + uint32_t(active_block_cols_per_row[r].size());
        bc.block_byte_ptrs[r + 1] = uint32_t(bc.block_col_bytes.size());
    }
    return bc;
}

// Forward: for each block-row, walk its active blocks (ULEB128 decode,
// cheap -- few blocks per row), accumulate a LOCAL 4-wide y register, write
// once per block-row (fixed, contiguous, no scatter).
static void block4_forward(const Block4CSR& bc, const float* x, float* y, int num_cpus) {
    #pragma omp parallel for schedule(static) num_threads(num_cpus)
    for (int64_t rr = 0; rr < int64_t(bc.n_block_rows); ++rr) {
        const uint32_t r = uint32_t(rr);
        const uint32_t nblocks = bc.block_ptrs[r + 1] - bc.block_ptrs[r];
        float yl[4] = {0, 0, 0, 0};
        std::size_t bytepos = bc.block_byte_ptrs[r];
        uint32_t prev = 0;
        for (uint32_t b = 0; b < nblocks; ++b) {
            std::size_t dlen = 0;
            uint32_t delta = uleb128_decode<uint32_t>(bc.block_col_bytes.data() + bytepos, dlen);
            bytepos += dlen;
            uint32_t bcol = prev + delta;
            prev = bcol;
            const float* W = bc.block_data.data() + (std::size_t(bc.block_ptrs[r]) + b) * 16;
            const float* xb = x + bcol * 4;
            #pragma omp simd
            for (int j = 0; j < 4; ++j) {
                const float xj = xb[j];
                for (int i = 0; i < 4; ++i) yl[i] += W[j * 4 + i] * xj;
            }
        }
        y[r * 4 + 0] = yl[0]; y[r * 4 + 1] = yl[1]; y[r * 4 + 2] = yl[2]; y[r * 4 + 3] = yl[3];
    }
}

int main() {
    const uint32_t M = 4096, N = 4096;
    const int NUM_CPUS = 8;
    const double DISLDO_MSYN_S = 750.0;  // representative measured rate from sweep_disldo_density.cpp (600-820 range)

    std::printf("%14s %14s %14s %10s\n", "block_fill%", "total_syn", "block4 Msyn/s", "vs disldo");
    for (double fill : {0.20, 0.30, 0.40, 0.50, 0.70, 1.00}) {
        std::mt19937 brng(3), wrng(7);
        const uint32_t n_block_rows = M / 4, n_block_cols = N / 4;
        // Every block-row has the SAME number of active blocks (uniform
        // fill across rows, deliberately -- not testing global sparsity
        // here, just the per-block dense-tile kernel's real throughput at
        // a controlled local fill rate).
        const uint32_t blocks_per_row = std::max<uint32_t>(1, uint32_t(n_block_cols * 0.05)); // 5% of block-cols active per row (global block density)
        std::vector<std::vector<uint32_t>> active_cols(n_block_rows);
        std::uniform_int_distribution<uint32_t> colpick(0, n_block_cols - 1);
        for (uint32_t r = 0; r < n_block_rows; ++r) {
            std::vector<bool> seen(n_block_cols, false);
            while (active_cols[r].size() < blocks_per_row) {
                uint32_t c = colpick(brng);
                if (!seen[c]) { seen[c] = true; active_cols[r].push_back(c); }
            }
            std::sort(active_cols[r].begin(), active_cols[r].end());
        }
        Block4CSR bc = build_block4csr(M, N, active_cols, wrng);
        // Zero out (1-fill) fraction of each block's 16 slots to hit the target LOCAL fill rate.
        std::uniform_real_distribution<float> u01(0, 1);
        for (auto& v : bc.block_data) if (u01(brng) > fill) v = 0.0f;

        double total_live_syn = double(bc.block_ptrs[n_block_rows]) * 16.0 * fill;

        std::vector<float> x(N), y(M, 0.f);
        std::mt19937 xrng(1);
        std::uniform_real_distribution<float> xv(-1, 1);
        for (auto& v : x) v = xv(xrng);

        block4_forward(bc, x.data(), y.data(), NUM_CPUS);  // warm
        double best = 1e18;
        for (int r = 0; r < 15; ++r) {
            std::fill(y.begin(), y.end(), 0.f);
            double t0 = now_s();
            block4_forward(bc, x.data(), y.data(), NUM_CPUS);
            best = std::min(best, now_s() - t0);
        }
        double msyn_s = total_live_syn / best * 1e-6;
        std::printf("%13.0f%% %14.0f %14.1f %9.3fx\n",
                    fill * 100, total_live_syn, msyn_s, msyn_s / DISLDO_MSYN_S);
    }
    return 0;
}
