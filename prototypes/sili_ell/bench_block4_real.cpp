// bench_block4_real.cpp
// ===========================================================================
// Validates dense_block4.hpp -- the REAL library header, using FP4-packed
// storage (1 byte/slot, table-lookup decode) -- not the raw-float prototype
// in bench_block4_dual.cpp. Open risk being checked: does FP4_TABLE[...]
// (a gather from a tiny 16-entry table, inside the #pragma omp simd loop)
// still vectorize, or does per-slot table lookup silently kill the
// speedup the raw-float version showed?
// ===========================================================================
#include "../../sili/lib/headers/linear_disldo.hpp"
#include "../../sili/lib/headers/dense_block4.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    const uint32_t M = 16384, N = 16384;   // large enough to be past OS-noise floor (see conversation)
    const int NUM_CPUS = 8;
    const double DISLDO_MSYN_S = 750.0;

    std::printf("%14s %14s %16s %16s %10s\n", "block_fill%", "total_syn", "block4(raw) Msyn/s", "block4(FP4) Msyn/s", "FP4/raw");
    for (double fill : {0.10, 0.15, 0.20, 0.30, 0.50, 1.00}) {
        const uint32_t Mb = M / 4, Nb = N / 4;
        std::mt19937 brng(3), wrng(7);
        const uint32_t blocks_per_row = std::max<uint32_t>(1, uint32_t(Nb * 0.05));

        // Build a CSR directly at the target per-tile fill (fill*16 entries/active tile),
        // via per-row column lists.
        std::uniform_int_distribution<uint32_t> bcolpick(0, Nb - 1);
        std::uniform_real_distribution<float> wv(-6, 6), impv(0, 1);
        std::vector<std::vector<uint32_t>> row_cols(M);
        std::vector<std::vector<float>> row_w(M), row_imp(M);
        std::mt19937 brng2(3), wrng2(7);
        std::uniform_real_distribution<float> u01b(0, 1);
        for (uint32_t br = 0; br < Mb; ++br) {
            std::vector<bool> seen(Nb, false);
            std::vector<uint32_t> bcols;
            while (bcols.size() < blocks_per_row) {
                uint32_t c = bcolpick(brng2);
                if (!seen[c]) { seen[c] = true; bcols.push_back(c); }
            }
            for (uint32_t bc : bcols) {
                for (int li = 0; li < 4; ++li)
                    for (int lj = 0; lj < 4; ++lj)
                        if (u01b(brng2) < fill) {
                            uint32_t row = br * 4 + li, col = bc * 4 + lj;
                            const uint8_t code = fp4_quantize(wv(wrng2));
                            row_cols[row].push_back(col);
                            row_w[row].push_back(FP4_TABLE[code]);
                            row_imp[row].push_back(impv(wrng2));
                        }
            }
        }
        std::vector<uint32_t> ptrs(M + 1, 0);
        std::vector<uint32_t> idx;
        std::vector<float> wgt, imp;
        for (uint32_t row = 0; row < M; ++row) {
            std::vector<uint32_t> order(row_cols[row].size());
            for (std::size_t k = 0; k < order.size(); ++k) order[k] = uint32_t(k);
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return row_cols[row][a] < row_cols[row][b]; });
            for (uint32_t k : order) {
                idx.push_back(row_cols[row][k]);
                wgt.push_back(row_w[row][k]);
                imp.push_back(row_imp[row][k]);
            }
            ptrs[row + 1] = uint32_t(idx.size());
        }
        const std::size_t total_syn = idx.size();

        // Real dense_block4.hpp split
        auto split = split_for_block4(M, N, ptrs, idx, wgt, imp, 0.0f);  // min_fill_frac=0 -> put EVERYTHING active into block4, testing the kernel itself at controlled fill

        std::vector<float> x(N), y(M, 0.f);
        std::mt19937 xrng(1);
        std::uniform_real_distribution<float> xv(-1, 1);
        for (auto& v : x) v = xv(xrng);

        block4_forward(split.block4, x.data(), y.data(), NUM_CPUS);  // warm
        double best = 1e18;
        for (int r = 0; r < 30; ++r) {
            std::fill(y.begin(), y.end(), 0.f);
            double t0 = now_s();
            block4_forward(split.block4, x.data(), y.data(), NUM_CPUS);
            best = std::min(best, now_s() - t0);
        }
        double msyn_s = double(total_syn) / best * 1e-6;
        std::printf("%13.0f%% %14zu %16s %16.1f %9s   (real block4 header, FP4 storage; total_syn=%zu leftover=%zu)\n",
                    fill * 100, total_syn, "n/a", msyn_s, "-", total_syn, split.leftover_rc.size());
    }
    return 0;
}
