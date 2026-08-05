// sweep_disldo_density.cpp
// ===========================================================================
// disldo_forward's own throughput (Msyn/s) across a density sweep, at
// fixed M=N=4096, batch=1 (this project's real, no-batching online
// convention) -- to find the density crossover against a dense matvec's
// real, measured throughput (from numpy+OpenBLAS, sili_v_torch.md's fix):
// ~3997.7 Msyn/s-equivalent for a batch=1 [4096,4096]@[4096] matvec.
// ===========================================================================
#include "../../sili/lib/headers/linear_disldo.hpp"
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
    const uint32_t M = 4096, N = 4096;
    const int NUM_CPUS = 8;
    const double DENSE_MSYN_S = 3997.7;  // measured: numpy+OpenBLAS, batch=1 matvec, same M,N

    std::printf("%12s %10s %14s %10s\n", "nnz/row", "density%", "disldo Msyn/s", "vs dense");
    for (uint32_t target : {10u, 25u, 50u, 100u, 200u, 400u, 800u, 1600u, 3200u}) {
        std::mt19937 rng(42);
        std::vector<std::vector<uint32_t>> row_cols(M);
        {
            std::set<uint32_t> seen;
            std::uniform_int_distribution<uint32_t> colpick(0, N - 1);
            for (uint32_t i = 0; i < M; ++i) {
                seen.clear();
                while (seen.size() < target) seen.insert(colpick(rng));
                for (uint32_t j : seen) row_cols[i].push_back(j);
                std::sort(row_cols[i].begin(), row_cols[i].end());
            }
        }
        const size_t total_syn = size_t(M) * target;

        // disldo CSR: rows = N (inputs) -- see bench_vs_disldo.cpp's orientation note.
        std::vector<std::vector<uint32_t>> by_input(N);
        for (uint32_t i = 0; i < M; ++i) for (uint32_t j : row_cols[i]) by_input[j].push_back(i);
        for (auto& l : by_input) std::sort(l.begin(), l.end());
        std::vector<int> csr_ptrs(N + 1, 0);
        std::vector<int> csr_idx;
        std::vector<float> csr_w, csr_imp;
        csr_idx.reserve(total_syn); csr_w.reserve(total_syn); csr_imp.reserve(total_syn);
        std::mt19937 wrng(7);
        std::uniform_real_distribution<float> wval(-6.0f, 6.0f);
        for (uint32_t j = 0; j < N; ++j) {
            for (uint32_t i : by_input[j]) {
                csr_idx.push_back(int(i));
                csr_w.push_back(FP4_TABLE[fp4_quantize(wval(wrng))]);
                csr_imp.push_back(0.5f);
            }
            csr_ptrs[j + 1] = int(csr_idx.size());
        }
        sfc_unused:
        SparseLinearWeightsDelta<int, FP4BiPacked, uint32_t> weights;
        weights.connections = delta_csr_from_absolute<int, FP4BiPacked, uint32_t>(
            csr_ptrs, csr_idx, csr_w, csr_imp, N, M,
            total_syn * 8 + 4096, total_syn + 64, 0.2f);
        weights.recompute_stats();
        double sili_nnz = double(weights.connections.nnz());

        std::vector<float> x(N), y(M, 0.f);
        std::mt19937 xrng(1);
        std::uniform_real_distribution<float> xval(-1, 1);
        for (auto& v : x) v = xval(xrng);

        // warm + best-of-N timing
        disldo_forward<int, FP4BiPacked, uint32_t>(x.data(), 1, N, weights, y.data(), 0.0f, NUM_CPUS);
        double best = 1e18;
        for (int r = 0; r < 15; ++r) {
            std::fill(y.begin(), y.end(), 0.f);
            double t0 = now_s();
            disldo_forward<int, FP4BiPacked, uint32_t>(x.data(), 1, N, weights, y.data(), 0.0f, NUM_CPUS);
            best = std::min(best, now_s() - t0);
        }
        double msyn_s = sili_nnz / best * 1e-6;
        std::printf("%12u %9.3f%% %14.1f %9.3fx\n",
                    target, 100.0 * target / N, msyn_s, msyn_s / DENSE_MSYN_S);
    }
    return 0;
}
