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
    std::vector<SIZE_TYPE> idx_a  = {0};
    std::vector<float>     vals_a = {1.0f};
    std::vector<SIZE_TYPE> ptrs_b = {0, 0, 1};
    std::vector<SIZE_TYPE> idx_b  = {1};
    std::vector<float>     vals_b = {2.0f};

    std::vector<SIZE_TYPE> out_ptrs, out_idx;
    std::vector<float>     out_vals;
    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b,
                                2, 0, 1, out_ptrs, out_idx, out_vals);

    CHECK_VECTOR_EQUAL(out_ptrs, std::vector<SIZE_TYPE>({0, 1, 2}));
    CHECK_VECTOR_EQUAL(out_idx,  std::vector<SIZE_TYPE>({0, 1}));
    CHECK_VECTOR_EQUAL(out_vals, std::vector<float>({1.0f, 2.0f}));
}

TEST_CASE("csr_union: overlap prefer a/b/sum", "[csr_union]") {
    using SIZE_TYPE = int;
    std::vector<SIZE_TYPE> ptrs_a = {0, 2};
    std::vector<SIZE_TYPE> idx_a  = {0, 3};
    std::vector<float>     vals_a = {1.0f, 2.0f};
    std::vector<SIZE_TYPE> ptrs_b = {0, 2};
    std::vector<SIZE_TYPE> idx_b  = {3, 5};
    std::vector<float>     vals_b = {9.0f, 4.0f};

    std::vector<SIZE_TYPE> out_ptrs, out_idx;
    std::vector<float>     out_vals;

    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b,
                                1, /*prefer=a*/0, 1, out_ptrs, out_idx, out_vals);
    CHECK_VECTOR_EQUAL(out_idx,  std::vector<SIZE_TYPE>({0, 3, 5}));
    CHECK_VECTOR_EQUAL(out_vals, std::vector<float>({1.0f, 2.0f, 4.0f}));

    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b,
                                1, /*prefer=b*/1, 1, out_ptrs, out_idx, out_vals);
    CHECK_VECTOR_EQUAL(out_idx,  std::vector<SIZE_TYPE>({0, 3, 5}));
    CHECK_VECTOR_EQUAL(out_vals, std::vector<float>({1.0f, 9.0f, 4.0f}));

    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b,
                                1, /*prefer=sum*/2, 1, out_ptrs, out_idx, out_vals);
    CHECK_VECTOR_EQUAL(out_idx,  std::vector<SIZE_TYPE>({0, 3, 5}));
    CHECK_VECTOR_EQUAL(out_vals, std::vector<float>({1.0f, 11.0f, 4.0f}));
}

TEST_CASE("csr_union: parallel vs sequential agree on a larger random case", "[csr_union]") {
    using SIZE_TYPE = int;
    std::mt19937_64 gen(12345);
    std::uniform_int_distribution<int> col_dist(0, 63);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    const SIZE_TYPE rows = 40;

    auto make_random_csr = [&](std::vector<SIZE_TYPE>& ptrs,
                               std::vector<SIZE_TYPE>& idx,
                               std::vector<float>& vals) {
        ptrs.assign(rows + 1, 0);
        for (SIZE_TYPE r = 0; r < rows; ++r) {
            std::set<int> cols;
            int n = col_dist(gen) % 8;
            for (int k = 0; k < n; ++k) cols.insert(col_dist(gen));
            for (int c : cols) { idx.push_back(c); vals.push_back(val_dist(gen)); }
            ptrs[r + 1] = (SIZE_TYPE)idx.size();
        }
    };

    std::vector<SIZE_TYPE> ptrs_a, idx_a, ptrs_b, idx_b;
    std::vector<float> vals_a, vals_b;
    make_random_csr(ptrs_a, idx_a, vals_a);
    make_random_csr(ptrs_b, idx_b, vals_b);

    std::vector<SIZE_TYPE> out_ptrs_p, out_idx_p, out_ptrs_s, out_idx_s;
    std::vector<float> out_vals_p, out_vals_s;
    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b,
                                rows, 2, 4, out_ptrs_p, out_idx_p, out_vals_p);
    csr_union<SIZE_TYPE, float>(ptrs_a, idx_a, vals_a, ptrs_b, idx_b, vals_b,
                                rows, 2, 1, out_ptrs_s, out_idx_s, out_vals_s);

    CHECK_VECTOR_EQUAL(out_ptrs_p, out_ptrs_s);
    CHECK_VECTOR_EQUAL(out_idx_p,  out_idx_s);
    CHECK_VECTOR_ALMOST_EQUAL(out_vals_p, out_vals_s, 0.0000001);
}
