// TDD for top_k_csr_nucleus (see sili_peridot/JOURNAL.md's "nucleus/
// energy-threshold top-k math" design note, and memory attention_sparsity_
// full_reachability -- this ONLY thins connection weight, never severs
// reachability structurally, matching that constraint since it's a
// per-call activation/gradient selection, not a permanent topology edit).
//
// Contract: row r keeps the SMALLEST set of its own top-|v| entries such
// that the captured squared-magnitude ratio
//     R(v, k) = sum(v_topk^2) / sum(v^2)
// is >= r_target[r]. k is a CONSEQUENCE of r_target and the row's actual
// data, not a fixed constant -- this is the exact invariant the whole
// mechanism depends on, so this test verifies it directly and adversarially
// (not just "compiles and returns something plausible").
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

// Compute R(v,k) directly from a dense row + the CSR's kept values for
// that row, independent of however top_k_csr_nucleus internally works --
// an external, from-first-principles check, not a copy of the kernel's
// own logic.
static double captured_ratio(const std::vector<float>& row, const std::vector<float>& kept) {
    double total = 0.0;
    for (float v : row) total += static_cast<double>(v) * v;
    if (total == 0.0) return 0.0;
    double captured = 0.0;
    for (float v : kept) captured += static_cast<double>(v) * v;
    return captured / total;
}

// Verify k is MINIMAL: dropping the smallest-magnitude kept entry must
// violate R>=r_target (unless k==0 or k==cols already). Otherwise a
// kernel could satisfy the invariant by trivially keeping everything.
static bool is_minimal(const std::vector<float>& row, std::vector<float> kept, float r_target) {
    if (kept.empty() || kept.size() >= row.size()) return true;
    // kept is already row-major/index order, not magnitude order -- find
    // and drop the single smallest-magnitude entry.
    size_t drop = 0;
    for (size_t i = 1; i < kept.size(); ++i)
        if (std::abs(kept[i]) < std::abs(kept[drop])) drop = i;
    std::vector<float> smaller = kept;
    smaller.erase(smaller.begin() + drop);
    return captured_ratio(row, smaller) < static_cast<double>(r_target);
}

static void check_r_target_invariant(const char* name, size_t rows, size_t cols,
                                      const std::vector<float>& r_targets, unsigned seed) {
    std::vector<float> values(rows * cols);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : values) v = dist(rng);

    for (int num_threads : {1, 4}) {
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), rows, cols, r_targets.data(), num_threads);
        auto& ptrs = *csr.ptrs[0];
        auto& idx  = *csr.indices[0];
        auto& val  = *csr.values[0];

        for (size_t r = 0; r < rows; ++r) {
            std::vector<float> row(values.begin() + r * cols, values.begin() + (r + 1) * cols);
            std::vector<float> kept(val.begin() + ptrs[r], val.begin() + ptrs[r + 1]);
            std::vector<SIZE_TYPE> kept_idx(idx.begin() + ptrs[r], idx.begin() + ptrs[r + 1]);

            double R = captured_ratio(row, kept);
            CHECK(R >= static_cast<double>(r_targets[r]) - 1e-6,
                  "%s (threads=%d) row %zu: R=%.6f < r_target=%.6f (k=%zu/%zu)",
                  name, num_threads, r, R, r_targets[r], kept.size(), cols);

            CHECK(is_minimal(row, kept, r_targets[r]),
                  "%s (threads=%d) row %zu: k=%zu is not minimal for r_target=%.6f",
                  name, num_threads, r, kept.size(), r_targets[r]);

            // indices strictly ascending (row-major CSR convention)
            for (size_t i = 1; i < kept_idx.size(); ++i)
                CHECK(kept_idx[i] > kept_idx[i - 1],
                      "%s row %zu: kept indices not strictly ascending", name, r);

            // every kept value must equal the true dense value at that index
            for (size_t i = 0; i < kept_idx.size(); ++i)
                CHECK(kept[i] == row[kept_idx[i]],
                      "%s row %zu: kept value/index mismatch", name, r);
        }
    }
}

static void check_edge_cases() {
    // All-zero row: nothing to capture, must keep k=0 regardless of r_target.
    {
        std::vector<float> values = {0.0f, 0.0f, 0.0f, 0.0f};
        std::vector<float> r_target = {0.5f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 1, 4, r_target.data(), 1);
        auto& ptrs = *csr.ptrs[0];
        CHECK(ptrs[1] - ptrs[0] == 0, "all-zero row must keep k=0, got %d", ptrs[1] - ptrs[0]);
    }
    // r_target == 0: nothing required, must keep k=0 even with nonzero data.
    {
        std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> r_target = {0.0f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 1, 4, r_target.data(), 1);
        auto& ptrs = *csr.ptrs[0];
        CHECK(ptrs[1] - ptrs[0] == 0, "r_target=0 must keep k=0, got %d", ptrs[1] - ptrs[0]);
    }
    // r_target == 1.0: must keep everything (exact full-energy capture),
    // exercising the floating-point-rounding fallback path directly.
    {
        std::vector<float> values = {1.0f, -2.0f, 0.5f, 3.0f, -0.1f};
        std::vector<float> r_target = {1.0f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 1, 5, r_target.data(), 1);
        auto& ptrs = *csr.ptrs[0];
        CHECK(ptrs[1] - ptrs[0] == 5, "r_target=1.0 must keep all 5 entries, got %d", ptrs[1] - ptrs[0]);
    }
    // One dominant entry: r_target=0.9 should need only that one entry.
    {
        std::vector<float> values = {100.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> r_target = {0.9f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 1, 4, r_target.data(), 1);
        auto& ptrs = *csr.ptrs[0];
        auto& idx = *csr.indices[0];
        CHECK(ptrs[1] - ptrs[0] == 1, "dominant-entry row: expected k=1, got %d", ptrs[1] - ptrs[0]);
        if (ptrs[1] - ptrs[0] == 1) CHECK(idx[0] == 0, "dominant-entry row: expected index 0 kept, got %d", idx[0]);
    }
    // Flat/uniform row: every entry contributes equally, so R_target=0.5
    // over 4 equal-magnitude entries needs k=2 (ties broken consistently,
    // any 2 of the 4 satisfy R exactly = 0.5).
    {
        std::vector<float> values = {2.0f, 2.0f, 2.0f, 2.0f};
        std::vector<float> r_target = {0.5f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 1, 4, r_target.data(), 1);
        auto& ptrs = *csr.ptrs[0];
        CHECK(ptrs[1] - ptrs[0] == 2, "uniform row r_target=0.5: expected k=2, got %d", ptrs[1] - ptrs[0]);
    }
    // Per-row independence: two very different rows in the same call must
    // each get their own k, not a shared/global one.
    {
        std::vector<float> values = {
            100.0f, 1.0f, 1.0f, 1.0f,   // needs k=1 at r_target=0.9
            1.0f, 1.0f, 1.0f, 1.0f,     // needs k=4 at r_target=0.9 (flat)
        };
        std::vector<float> r_target = {0.9f, 0.9f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 2, 4, r_target.data(), 1);
        auto& ptrs = *csr.ptrs[0];
        CHECK(ptrs[1] - ptrs[0] == 1, "row 0 (dominant): expected k=1, got %d", ptrs[1] - ptrs[0]);
        CHECK(ptrs[2] - ptrs[1] == 4, "row 1 (flat): expected k=4, got %d", ptrs[2] - ptrs[1]);
    }
}

static void check_k_min_max_clamp() {
    // r_target=0 would normally give k=0 -- k_min must force a floor,
    // pulled from the same magnitude-sorted order (still "top-k", not
    // arbitrary padding). Direct instruction: hardware-driven density
    // floor, independent of how little energy R_target says is needed.
    {
        std::vector<float> values = {5.0f, -4.0f, 1.0f, 0.5f};
        std::vector<float> r_target = {0.0f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 1, 4, r_target.data(), 1,
                                                         /*k_min=*/2, /*k_max=*/SIZE_MAX);
        auto& ptrs = *csr.ptrs[0];
        auto& idx = *csr.indices[0];
        CHECK(ptrs[1] - ptrs[0] == 2, "k_min=2 floor: expected k=2, got %d", ptrs[1] - ptrs[0]);
        // must be the two LARGEST-magnitude entries (indices 0 and 1), not arbitrary.
        if (ptrs[1] - ptrs[0] == 2) {
            CHECK(idx[0] == 0 && idx[1] == 1,
                  "k_min padding should keep the top-magnitude entries, got idx[0]=%d idx[1]=%d", idx[0], idx[1]);
        }
    }
    // r_target=1.0 would normally give k=cols -- k_max must force a ceiling.
    {
        std::vector<float> values = {5.0f, -4.0f, 1.0f, 0.5f};
        std::vector<float> r_target = {1.0f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 1, 4, r_target.data(), 1,
                                                         /*k_min=*/0, /*k_max=*/2);
        auto& ptrs = *csr.ptrs[0];
        CHECK(ptrs[1] - ptrs[0] == 2, "k_max=2 ceiling: expected k=2, got %d", ptrs[1] - ptrs[0]);
    }
    // An all-zero row can't manufacture k_min entries out of nothing.
    {
        std::vector<float> values = {0.0f, 0.0f, 0.0f, 0.0f};
        std::vector<float> r_target = {0.5f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 1, 4, r_target.data(), 1,
                                                         /*k_min=*/3, /*k_max=*/SIZE_MAX);
        auto& ptrs = *csr.ptrs[0];
        CHECK(ptrs[1] - ptrs[0] == 0, "all-zero row: k_min cannot manufacture entries, expected k=0, got %d",
              ptrs[1] - ptrs[0]);
    }
    // Default (no clamp) behavior must be unaffected -- same as the
    // pre-clamp edge-case checks above, called through the new signature.
    {
        std::vector<float> values = {100.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> r_target = {0.9f};
        auto csr = top_k_csr_nucleus<SIZE_TYPE, float>(values.data(), 1, 4, r_target.data(), 1);
        auto& ptrs = *csr.ptrs[0];
        CHECK(ptrs[1] - ptrs[0] == 1, "default (no clamp): expected k=1, got %d", ptrs[1] - ptrs[0]);
    }
}

int main() {
    check_edge_cases();
    check_k_min_max_clamp();

    check_r_target_invariant("random rows, r_target=0.5", 8, 32, std::vector<float>(8, 0.5f), 1);
    check_r_target_invariant("random rows, r_target=0.95", 8, 64, std::vector<float>(8, 0.95f), 2);
    check_r_target_invariant("random rows, r_target=0.05", 16, 48, std::vector<float>(16, 0.05f), 3);

    // Heterogeneous per-row targets in a single call.
    std::vector<float> mixed_targets = {0.1f, 0.99f, 0.5f, 0.75f, 0.3f, 0.6f, 0.2f, 0.8f};
    check_r_target_invariant("mixed per-row r_target", 8, 40, mixed_targets, 4);

    // Larger, above-threshold shape (exercises the #pragma omp path).
    check_r_target_invariant("large (128x256, r_target=0.8)", 128, 256, std::vector<float>(128, 0.8f), 5);

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
