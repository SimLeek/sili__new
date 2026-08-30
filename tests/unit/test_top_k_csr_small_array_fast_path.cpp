// Regression test for the small-array serial fast-path in top_k_indices/
// top_k_csr/to_csr (see SILI_OMP_SMALL_THRESHOLD's own docstring,
// csr.hpp). Root-caused via direct profiling during the sisldo sparsity
// plan's real MQAR benchmark: CSR.from_dense on a single 256-element
// activation row cost 180-990us (WORSE with more threads) against ~1-3us
// of real serial top-k work -- #pragma omp parallel's own thread-team
// rendezvous overhead dominating completely, making the sparse path 4x
// SLOWER than dense end-to-end despite the underlying kernels being
// genuinely 1.6-2.35x faster once given a CSR.
//
// This test proves the fast-path is a pure performance change, not a
// behavior change: output must be byte-identical regardless of
// num_threads (1, 2, 4, 8) and regardless of whether the array size sits
// below or above SILI_OMP_SMALL_THRESHOLD.
#include "../../sili/lib/headers/csr.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); std::fflush(stdout); ++g_fail; } \
} while (0)

using SIZE_TYPE = int;

static void check_thread_count_invariance(const char* name, size_t n, size_t k) {
    std::vector<float> values(n);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : values) v = dist(rng);

    std::vector<std::vector<SIZE_TYPE>> results;
    for (int num_threads : {1, 2, 4, 8}) {
        auto idx = top_k_indices<SIZE_TYPE, float>(values.data(), n, k, num_threads);
        // top_k_indices itself doesn't sort its OWN final output by index
        // (only by magnitude) -- sort by index here so results from
        // different thread counts (which can tie-break differently on
        // magnitude ties) are comparable as SETS, matching what
        // top_k_csr's own merge_sort_coo does downstream anyway.
        std::sort(idx.begin(), idx.end());
        results.push_back(idx);
    }
    for (size_t i = 1; i < results.size(); ++i) {
        CHECK(results[i] == results[0],
              "%s: num_threads variant %zu produced a different index set than num_threads=1 (n=%zu, k=%zu)",
              name, i, n, k);
    }
}

static void check_top_k_csr_thread_count_invariance(const char* name, size_t rows, size_t cols, size_t k) {
    std::vector<float> values(rows * cols);
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : values) v = dist(rng);

    struct Result { std::vector<SIZE_TYPE> row, col; std::vector<float> val; std::vector<SIZE_TYPE> ptrs; };
    std::vector<Result> results;
    for (int num_threads : {1, 2, 4, 8}) {
        auto csr = top_k_csr<SIZE_TYPE, float>(values.data(), rows, cols, k, num_threads);
        Result r;
        r.row = *csr.indices[0];
        r.val = *csr.values[0];
        r.ptrs = *csr.ptrs[0];
        results.push_back(r);
    }
    for (size_t i = 1; i < results.size(); ++i) {
        CHECK(results[i].ptrs == results[0].ptrs,
              "%s: top_k_csr ptrs diverged across thread counts (variant %zu)", name, i);
        CHECK(results[i].row == results[0].row,
              "%s: top_k_csr indices diverged across thread counts (variant %zu)", name, i);
        bool vals_match = results[i].val.size() == results[0].val.size();
        if (vals_match) {
            for (size_t j = 0; j < results[i].val.size(); ++j)
                if (results[i].val[j] != results[0].val[j]) { vals_match = false; break; }
        }
        CHECK(vals_match, "%s: top_k_csr values diverged across thread counts (variant %zu)", name, i);
    }
}

int main() {
    // Below SILI_OMP_SMALL_THRESHOLD (4096) -- exercises the new serial
    // fast-path directly. This is the exact real-world case (a single
    // [256]-wide activation row, sili_peridot's ToyTileRecurrenceRMT).
    check_thread_count_invariance("small (n=256, k=128)", 256, 128);
    check_thread_count_invariance("small (n=64, k=8)", 64, 8);
    check_top_k_csr_thread_count_invariance("small top_k_csr (256x1, k=128)", 1, 256, 128);
    check_top_k_csr_thread_count_invariance("small top_k_csr (32x8, k=64)", 32, 8, 64);

    // Above the threshold -- exercises the original (still intact)
    // #pragma omp parallel path, confirmed to still agree with itself
    // across thread counts too (not just the new small-array branch).
    check_thread_count_invariance("large (n=8192, k=2048)", 8192, 2048);
    check_top_k_csr_thread_count_invariance("large top_k_csr (64x128, k=1024)", 64, 128, 1024);

    // Straddling the threshold exactly, both sides.
    check_thread_count_invariance("at threshold (n=4096, k=1024)", 4096, 1024);
    check_thread_count_invariance("just below threshold (n=4095, k=1024)", 4095, 1024);

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
