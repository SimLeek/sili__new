// sweep_banked_demotion.cpp
// ===========================================================================
// Does banked's bank-collision demotion rate improve at lower density
// (sparser nets), separately from headroom ratio? Sweeps target_nnz_per_row
// at a roughly-constant headroom ratio (capacity = next pow2 >= 1.28x
// target, matching bench_vs_disldo.cpp's original ratio), then separately
// sweeps headroom ratio at one fixed density, to disentangle the two axes.
// ===========================================================================
#include "sparse_format_controller.hpp"
#include <cstdio>
#include <random>
#include <set>
#include <vector>

using namespace sfc;

static ToBankedResult run_one(uint32_t M, uint32_t N, uint32_t target_nnz_per_row,
                              uint32_t R_LOG, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> colpick(0, N - 1);
    std::vector<Syn> syns;
    syns.reserve(size_t(M) * target_nnz_per_row);
    std::set<uint32_t> seen;
    for (uint32_t i = 0; i < M; ++i) {
        seen.clear();
        while (seen.size() < target_nnz_per_row) seen.insert(colpick(rng));
        for (uint32_t j : seen) syns.push_back({i, j, uint8_t(0x11)});
    }
    // S = M/C = N/R must be a power of 2; keep C_LOG = R_LOG (square M=N here)
    // so S = N/R is fixed regardless of R -- LOG_S must satisfy N = R * S,
    // i.e. LOG_S = log2(N) - R_LOG.
    uint32_t log2N = 0; while ((1u << log2N) < N) ++log2N;
    uint32_t LOG_S = log2N - R_LOG;
    return to_banked(syns, LOG_S, R_LOG, R_LOG, M, N, 32, 9);
}

int main() {
    const uint32_t M = 4096, N = 4096;
    const uint32_t log2N = 12;  // N = 4096

    std::printf("=== Sweep 1: density, ~constant headroom ratio (~1.25-1.5x target, rounded to pow2) ===\n");
    std::printf("%12s %10s %8s %8s %10s\n", "target/row", "density%", "R(cap)", "headroom", "demoted%");
    for (uint32_t target : {8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u}) {
        uint32_t R = 1u; while (R < target) R <<= 1;
        // R is already the next pow2 >= target -- headroom ratio varies
        // 1.0-2.0x depending on how close target is to a power of 2; also
        // try one deliberate extra doubling for a consistently-generous case.
        uint32_t R_LOG = 0; while ((1u << R_LOG) < R) ++R_LOG;
        if (R_LOG >= log2N) continue;  // S would be 0 or negative
        auto res = run_one(M, N, target, R_LOG, 42);
        std::printf("%12u %9.2f%% %8u %9.2fx %9.3f%%\n",
                    target, 100.0 * target / N, R, double(R) / target, 100.0 * res.demoted_frac);
    }

    std::printf("\n=== Sweep 2: headroom ratio at FIXED density (target=100/row, 2.44%%) ===\n");
    std::printf("%10s %8s %10s %10s\n", "target", "R(cap)", "headroom", "demoted%");
    const uint32_t target2 = 100;
    for (uint32_t R_LOG = 7; R_LOG <= 10; ++R_LOG) {
        uint32_t R = 1u << R_LOG;
        if (R < target2) continue;
        auto res = run_one(M, N, target2, R_LOG, 42);
        std::printf("%10u %8u %9.2fx %9.3f%%\n",
                    target2, R, double(R) / target2, 100.0 * res.demoted_frac);
    }

    return 0;
}
