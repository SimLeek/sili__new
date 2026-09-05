#include "../../sili/lib/headers/csr.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <random>
#include <set>
#include <vector>

// csr_union: construction/loading-time CSR merge (see sili_peridot/JOURNAL.md
// and sili.sparse_rnn.csr_union, the Python wrapper this backs).

TEST_CASE("csr_union: disjoint positions all kept", "[csr_union]") {
    using SIZE_TYPE = int;
    std::vector<SIZE_TYPE> ptrs_a = {0, 1, 1};
    std::vector<SIZE_TYPE> idx_a = {0};
    std::vector<float> vals_a = {1.0f};
    std::vector<SIZE_TYPE> ptrs_b = {0, 0, 1};
    std::vector<SIZE_TYPE> idx_b = {1};
    std::vector<float> vals_b = {2.0f};

    std::vector<SIZE_TYPE> out_ptrs, out_idx;
    std::vector<float> out_vals;
    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b, 2, 0, 1, out_ptrs,
                                out_idx, out_vals);

    CHECK_VECTOR_EQUAL(out_ptrs, std::vector<SIZE_TYPE>({0, 1, 2}));
    CHECK_VECTOR_EQUAL(out_idx, std::vector<SIZE_TYPE>({0, 1}));
    CHECK_VECTOR_EQUAL(out_vals, std::vector<float>({1.0f, 2.0f}));
}

TEST_CASE("csr_union: overlap prefer a/b/sum", "[csr_union]") {
    using SIZE_TYPE = int;
    std::vector<SIZE_TYPE> ptrs_a = {0, 2};
    std::vector<SIZE_TYPE> idx_a = {0, 3};
    std::vector<float> vals_a = {1.0f, 2.0f};
    std::vector<SIZE_TYPE> ptrs_b = {0, 2};
    std::vector<SIZE_TYPE> idx_b = {3, 5};
    std::vector<float> vals_b = {9.0f, 4.0f};

    std::vector<SIZE_TYPE> out_ptrs, out_idx;
    std::vector<float> out_vals;

    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b, 1, /*prefer=a*/ 0, 1,
                                out_ptrs, out_idx, out_vals);
    CHECK_VECTOR_EQUAL(out_idx, std::vector<SIZE_TYPE>({0, 3, 5}));
    CHECK_VECTOR_EQUAL(out_vals, std::vector<float>({1.0f, 2.0f, 4.0f}));

    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b, 1, /*prefer=b*/ 1, 1,
                                out_ptrs, out_idx, out_vals);
    CHECK_VECTOR_EQUAL(out_idx, std::vector<SIZE_TYPE>({0, 3, 5}));
    CHECK_VECTOR_EQUAL(out_vals, std::vector<float>({1.0f, 9.0f, 4.0f}));

    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b, 1, /*prefer=sum*/ 2,
                                1, out_ptrs, out_idx, out_vals);
    CHECK_VECTOR_EQUAL(out_idx, std::vector<SIZE_TYPE>({0, 3, 5}));
    CHECK_VECTOR_EQUAL(out_vals, std::vector<float>({1.0f, 11.0f, 4.0f}));
}

TEST_CASE("csr_union: parallel vs sequential agree on a larger random case", "[csr_union]") {
    using SIZE_TYPE = int;
    std::mt19937_64 gen(12345);
    std::uniform_int_distribution<int> col_dist(0, 63);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    const SIZE_TYPE rows = 40;

    auto make_random_csr = [&](std::vector<SIZE_TYPE>& ptrs, std::vector<SIZE_TYPE>& idx,
                               std::vector<float>& vals) {
        ptrs.assign(rows + 1, 0);
        for (SIZE_TYPE r = 0; r < rows; ++r) {
            std::set<int> cols;
            int n = col_dist(gen) % 8;
            for (int k = 0; k < n; ++k)
                cols.insert(col_dist(gen));
            for (int c : cols) {
                idx.push_back(c);
                vals.push_back(val_dist(gen));
            }
            ptrs[r + 1] = (SIZE_TYPE)idx.size();
        }
    };

    std::vector<SIZE_TYPE> ptrs_a, idx_a, ptrs_b, idx_b;
    std::vector<float> vals_a, vals_b;
    make_random_csr(ptrs_a, idx_a, vals_a);
    make_random_csr(ptrs_b, idx_b, vals_b);

    std::vector<SIZE_TYPE> out_ptrs_p, out_idx_p, out_ptrs_s, out_idx_s;
    std::vector<float> out_vals_p, out_vals_s;
    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b, rows, 2, 4,
                                out_ptrs_p, out_idx_p, out_vals_p);
    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b, rows, 2, 1,
                                out_ptrs_s, out_idx_s, out_vals_s);

    CHECK_VECTOR_EQUAL(out_ptrs_p, out_ptrs_s);
    CHECK_VECTOR_EQUAL(out_idx_p, out_idx_s);
    CHECK_VECTOR_ALMOST_EQUAL(out_vals_p, out_vals_s, 0.0000001);
}

// top_k_indices / top_k_csr: also declared in csr.hpp -- see
// sili_peridot/JOURNAL.md for the real-checkpoint activation-sparsity
// eval that surfaced this (accuracy collapsed to 0.0 at EVERY density
// tested, including 0.9, which should be nearly identical to dense).

TEST_CASE("top_k_indices selects by magnitude, not raw signed value", "[top_k]") {
    // Regression: the comparator used to be `a.second > b.second` (raw
    // value), which for zero-mean data keeps only the largest POSITIVE
    // entries and discards every negative one regardless of magnitude --
    // silently wrong for any caller selecting "most important" entries
    // from signed data (e.g. post-RMSNorm activations) without first
    // taking abs() themselves. Must be by |value| instead.
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    std::vector<VALUE_TYPE> values = {1.0f, -5.0f, 2.0f, -3.0f, 0.5f, 4.0f, -0.1f, 0.2f};
    // |values| sorted descending: -5(1), 4(5), -3(3), 2(2), 1(0), 0.5(4), 0.2(7), -0.1(6)
    // top 3 by magnitude: indices {1, 5, 3} (values -5, 4, -3)
    // top 3 by raw value (the old, wrong behavior) would be {5, 2, 0} (values 4, 2, 1)
    auto top3 = top_k_indices<SIZE_TYPE, VALUE_TYPE>(values.data(), values.size(), 3, 2);

    std::set<SIZE_TYPE> got(top3.begin(), top3.end());
    std::set<SIZE_TYPE> expected_by_magnitude = {1, 5, 3};
    std::set<SIZE_TYPE> old_wrong_by_raw_value = {5, 2, 0};

    CHECK(got == expected_by_magnitude);
    CHECK(got != old_wrong_by_raw_value);
}

TEST_CASE("dense_to_top_k_csr keeps the largest-magnitude entries, not raw value", "[top_k]") {
    // NOTE: top_k_csr's k is a GLOBAL budget over the whole flattened
    // [rows, cols] array (top_k_indices(values, rows*cols, k, ...) --
    // see csr.hpp), NOT a per-row budget -- a separate semantics quirk
    // (see sili_peridot's model/sili_block.py _forward, which loops per
    // row itself to get a genuine per-row top-k). This test only checks
    // the magnitude-vs-raw-value regression, not per-row distribution.
    using SIZE_TYPE = int;
    using VALUE_TYPE = float;

    std::vector<VALUE_TYPE> dense = {
        -9.0f, 1.0f,  2.0f, -3.0f, 0.5f,  // row 0
        1.0f,  -8.0f, 7.0f, 0.2f,  -0.1f, // row 1
    };
    // Global top-4 by |.|: -9(r0,c0), -8(r1,c1), 7(r1,c2), -3(r0,c3).
    // Global top-4 by raw value (the old, wrong behavior) would instead
    // keep the positives/near-zeros: 7(r1,c2), 2(r0,c2), 1(r0,c1), 1(r1,c0).
    size_t rows = 2, cols = 5, k = 4;

    auto result = top_k_csr<SIZE_TYPE, VALUE_TYPE>(dense.data(), rows, cols, k, 2);

    auto& ptrs = *result.ptrs[0];
    auto& indices = *result.indices[0];
    std::set<std::pair<SIZE_TYPE, SIZE_TYPE>> got;
    for (SIZE_TYPE row = 0; row < (SIZE_TYPE)rows; ++row)
        for (SIZE_TYPE j = ptrs[row]; j < ptrs[row + 1]; ++j)
            got.insert({row, indices[j]});

    std::set<std::pair<SIZE_TYPE, SIZE_TYPE>> expected_by_magnitude = {
        {0, 0}, {1, 1}, {1, 2}, {0, 3}};

    CHECK(got == expected_by_magnitude);
}
