#include "parallel.hpp"
#include <catch2/catch_all.hpp>
#include <vector>
#include <algorithm>
#include <cstdlib> // for rand()
#include <numeric> // for std::iota
#include "tests_main.hpp"

// Test ComparatorLT
TEST_CASE("ComparatorLT works correctly", "[comparator]") {
    ComparatorLT<int, int> cmp_int;
    CHECK(cmp_int(1, 2) == true);    // 1 < 2
    CHECK(cmp_int(2, 1) == false);   // 2 < 1 is false
    CHECK(cmp_int(1, 1) == false);   // 1 < 1 is false

    ComparatorLT<double, double> cmp_double;
    CHECK(cmp_double(1.5, 2.5) == true);  // 1.5 < 2.5
    CHECK(cmp_double(2.5, 1.5) == false); // 2.5 < 1.5 is false
    CHECK(cmp_double(1.5, 1.5) == false); // 1.5 < 1.5 is false
}

// Test ComparatorGT
TEST_CASE("ComparatorGT works correctly", "[comparator]") {
    ComparatorGT<int, int> cmp_int;
    CHECK(cmp_int(2, 1) == true);    // 2 > 1
    CHECK(cmp_int(1, 2) == false);   // 1 > 2 is false
    CHECK(cmp_int(1, 1) == false);   // 1 > 1 is false

    ComparatorGT<double, double> cmp_double;
    CHECK(cmp_double(2.5, 1.5) == true);  // 2.5 > 1.5
    CHECK(cmp_double(1.5, 2.5) == false); // 1.5 > 2.5 is false
    CHECK(cmp_double(1.5, 1.5) == false); // 1.5 > 1.5 is false
}

// Test PermutationComparator with ComparatorLT
TEST_CASE("PermutationComparator with LT sorts indices ascending", "[comparator]") {
    std::vector<int> v = {3, 1, 4, 2};
    std::vector<size_t> indices = {0, 1, 2, 3};
    auto cmp = PermutationComparator<int, ComparatorLT<int, int>>(v, ComparatorLT<int, int>());
    std::sort(indices.begin(), indices.end(), cmp);
    std::vector<size_t> expected = {1, 3, 0, 2}; // v[1]=1, v[3]=2, v[0]=3, v[2]=4
    CHECK_VECTOR_EQUAL(indices, expected);
}

// Test PermutationComparator with ComparatorGT
TEST_CASE("PermutationComparator with GT sorts indices descending", "[comparator]") {
    std::vector<int> v = {3, 1, 4, 2};
    std::vector<size_t> indices = {0, 1, 2, 3};
    auto cmp = PermutationComparator<int, ComparatorGT<int, int>>(v, ComparatorGT<int, int>());
    std::sort(indices.begin(), indices.end(), cmp);
    std::vector<size_t> expected = {2, 0, 3, 1}; // v[2]=4, v[0]=3, v[3]=2, v[1]=1
    CHECK_VECTOR_EQUAL(indices, expected);
}

// Test PermutationComparator with floats
TEST_CASE("PermutationComparator works with floats", "[comparator]") {
    std::vector<double> v = {3.5, 1.2, 4.8, 2.3};
    std::vector<size_t> indices = {0, 1, 2, 3};
    auto cmp = PermutationComparator<double, ComparatorLT<double, double>>(v, ComparatorLT<double, double>());
    std::sort(indices.begin(), indices.end(), cmp);
    std::vector<size_t> expected = {1, 3, 0, 2}; // 1.2, 2.3, 3.5, 4.8
    CHECK_VECTOR_EQUAL(indices, expected);
}

// Test merge sort with default ComparatorLT
TEST_CASE("Merge sort with LT sorts ascending", "[merge_sort]") {
    std::vector<int> v = {3, 1, 4, 2};
    std::vector<int> v_sorted = v;
    omp_merge_sort(v_sorted);
    std::vector<int> expected = {1, 2, 3, 4};
    CHECK_VECTOR_EQUAL(v_sorted, expected);
}

// Test merge sort with ComparatorGT
TEST_CASE("Merge sort with GT sorts descending", "[merge_sort]") {
    std::vector<int> v = {3, 1, 4, 2};
    std::vector<int> v_sorted = v;
    omp_merge_sort<int, ComparatorGT<int, int>>(v_sorted);
    std::vector<int> expected = {4, 3, 2, 1};
    CHECK_VECTOR_EQUAL(v_sorted, expected);
}

// Test merge sort with empty vector
TEST_CASE("Merge sort handles empty vector", "[merge_sort]") {
    std::vector<int> v;
    // For empty vector, left >= right, so no sorting occurs
    omp_merge_sort(v);
    CHECK(v.empty());
}

// Test merge sort with single element
TEST_CASE("Merge sort handles single element", "[merge_sort]") {
    std::vector<int> v = {5};
    std::vector<int> v_sorted = v;
    _omp_merge_sort_recursive(v_sorted, 0, 0); // left == right, no sorting
    CHECK_VECTOR_EQUAL(v_sorted, std::vector<int>{5});
}

// Test merge sort with two elements
TEST_CASE("Merge sort sorts two elements", "[merge_sort]") {
    std::vector<int> v = {2, 1};
    std::vector<int> v_sorted = v;
    omp_merge_sort(v_sorted);
    std::vector<int> expected = {1, 2};
    CHECK_VECTOR_EQUAL(v_sorted, expected);
}

// Test merge sort on partial segment
TEST_CASE("Merge sort sorts partial segment", "[merge_sort]") {
    std::vector<int> v = {3, 1, 4, 2, 5};
    std::vector<int> v_partial = v;
    _omp_merge_sort_recursive(v_partial, 1, 3); // Sorts {1, 4, 2} -> {1, 2, 4}
    std::vector<int> expected = {3, 1, 2, 4, 5};
    CHECK_VECTOR_EQUAL(v_partial, expected);
}

// Test merge sort with large vector to engage parallelism
TEST_CASE("Merge sort sorts large vector", "[merge_sort]") {
    const size_t N = 100000;
    std::vector<int> v(N);
    std::generate(v.begin(), v.end(), [](){ return rand() % 1000; });
    std::vector<int> v_sorted = v;
    omp_merge_sort(v_sorted);
    CHECK(std::is_sorted(v_sorted.begin(), v_sorted.end()));
}

// Test merge sort with floats
TEST_CASE("Merge sort works with floats", "[merge_sort]") {
    std::vector<double> v = {3.5, 1.2, 4.8, 2.3};
    std::vector<double> v_sorted = v;
    omp_merge_sort(v_sorted);
    std::vector<double> expected = {1.2, 2.3, 3.5, 4.8};
    CHECK_VECTOR_ALMOST_EQUAL(v_sorted, expected);
}

// Test omp_merge_sort_permutation with ascending order
TEST_CASE("omp_merge_sort_permutation sorts ascending with ComparatorLT", "[permutation]") {
    std::vector<int> a = {5, 2, 9, 1, 5};
    auto p = omp_merge_sort_permutation(a, ComparatorLT<int, int>());
    std::vector<size_t> expected = {3, 1, 0, 4, 2}; // Indices for sorted order: 1, 2, 5, 5, 9
    CHECK_VECTOR_EQUAL(p, expected);
}

// Test omp_merge_sort_permutation with descending order
TEST_CASE("omp_merge_sort_permutation sorts descending with ComparatorGT", "[permutation]") {
    std::vector<int> a = {5, 2, 9, 1, 5};
    auto p = omp_merge_sort_permutation(a, ComparatorGT<int, int>());
    std::vector<size_t> expected = {2, 0, 4, 1, 3}; // Indices for sorted order: 9, 5, 5, 2, 1
    CHECK_VECTOR_EQUAL(p, expected);
}

// Test omp_merge_sort_permutation with empty vector
TEST_CASE("omp_merge_sort_permutation handles empty vector", "[permutation]") {
    std::vector<int> a;
    auto p = omp_merge_sort_permutation(a, ComparatorLT<int, int>());
    CHECK(p.empty());
}

// Test omp_apply_permutation_parallel
TEST_CASE("omp_apply_permutation_parallel permutes vector correctly", "[permutation]") {
    std::vector<size_t> p = {2, 0, 3, 1};
    std::vector<int> vec = {10, 20, 30, 40};
    omp_apply_permutation_parallel(p, vec);
    std::vector<int> expected = {30, 10, 40, 20}; // vec[2], vec[0], vec[3], vec[1]
    CHECK_VECTOR_EQUAL(vec, expected);
}

// Test apply_permutation_to_all_parallel
TEST_CASE("apply_permutation_to_all_parallel permutes multiple vectors", "[permutation]") {
    std::vector<size_t> p = {2, 0, 3, 1};
    std::vector<int> vec1 = {10, 20, 30, 40};
    std::vector<double> vec2 = {1.1, 2.2, 3.3, 4.4};
    apply_permutation_to_all_parallel(p, vec1, vec2);
    std::vector<int> expected1 = {30, 10, 40, 20};
    std::vector<double> expected2 = {3.3, 1.1, 4.4, 2.2};
    CHECK_VECTOR_EQUAL(vec1, expected1);
    CHECK_VECTOR_ALMOST_EQUAL(vec2, expected2);
}

// Test omp_sort_multiple_vectors
TEST_CASE("omp_sort_multiple_vectors sorts based on primary vector", "[sort]") {
    std::vector<int> primary = {5, 2, 9, 1};
    std::vector<int> vec1 = {10, 20, 30, 40};
    omp_sort_multiple_vectors(primary, ComparatorLT<int, int>(), vec1);
    std::vector<int> expected = {40, 20, 10, 30}; // Order corresponds to 1, 2, 5, 9
    CHECK_VECTOR_EQUAL(vec1, expected);
}

// Test omp_sort_ascending
TEST_CASE("omp_sort_ascending sorts vectors ascending", "[sort]") {
    std::vector<int> primary = {5, 2, 9, 1};
    std::vector<int> vec1 = {10, 20, 30, 40};
    omp_sort_ascending(primary, vec1);
    std::vector<int> expected = {40, 20, 10, 30};
    CHECK_VECTOR_EQUAL(vec1, expected);
}

// Test omp_sort_descending
TEST_CASE("omp_sort_descending sorts vectors descending", "[sort]") {
    std::vector<int> primary = {5, 2, 9, 1};
    std::vector<int> vec1 = {10, 20, 30, 40};
    omp_sort_descending(primary, vec1);
    std::vector<int> expected = {30, 10, 20, 40}; // Order corresponds to 9, 5, 2, 1
    CHECK_VECTOR_EQUAL(vec1, expected);
}

// Test omp_lower_bound with value present
TEST_CASE("omp_lower_bound finds lower bound for existing value", "[lower_bound]") {
    std::vector<int> arr = {1, 3, 3, 5, 7};
    int val = 3;
    size_t idx = omp_lower_bound(arr, arr.size(), val, 2); // 2 threads
    CHECK(idx == 1); // First 3 is at index 1
}

// Test omp_lower_bound with value not present
TEST_CASE("omp_lower_bound finds lower bound for missing value", "[lower_bound]") {
    std::vector<int> arr = {1, 3, 5, 7};
    int val = 4;
    size_t idx = omp_lower_bound(arr, arr.size(), val, 2);
    CHECK(idx == 2); // 4 would go before 5 at index 2
}

// Test omp_lower_bound with empty vector
TEST_CASE("omp_lower_bound handles empty vector", "[lower_bound]") {
    std::vector<int> arr;
    int val = 1;
    size_t idx = omp_lower_bound(arr, arr.size(), val, 2);
    CHECK(idx == 0);
}

// Test omp_scan (exclusive prefix sum)
TEST_CASE("omp_scan computes exclusive prefix sum correctly", "[scan]") {
    std::vector<int> input = {1, 2, 3, 4};
    std::unique_ptr<int[]> output(new int[input.size()]);
    omp_scan_exclusive(input, output, input.size());
    std::vector<int> expected = {0, 1, 3, 6}; // Exclusive: 0, 1, 1+2, 1+2+3
    
    std::vector<int> out_vec(output.get(), output.get() + input.size());
    CHECK_VECTOR_EQUAL(out_vec, expected);
}

// Test omp_full_scan (inclusive prefix sum)
TEST_CASE("omp_full_scan computes inclusive prefix sum correctly", "[scan]") {
    std::vector<int> input = {1, 2, 3, 4};
    std::unique_ptr<int[]> output(new int[input.size() + 1]);
    omp_full_scan(input, output, input.size());
    std::vector<int> expected = {0, 1, 3, 6, 10}; // Inclusive: 0, 1, 1+2, 1+2+3, 1+2+3+4

    std::vector<int> out_vec(output.get(), output.get() + input.size()+1);
    CHECK_VECTOR_EQUAL(out_vec, expected);
}

// Test omp_scan with empty vector
TEST_CASE("omp_scan handles empty vector", "[scan]") {
    std::vector<int> input;
    std::unique_ptr<int[]> output(new int[0]);
    omp_scan_exclusive(input, output, 0);
    // Should not crash; no output to check
}

// Test omp_full_scan with single element
TEST_CASE("omp_full_scan handles single-element vector", "[scan]") {
    std::vector<int> input = {42};
    std::unique_ptr<int[]> output(new int[2]);
    omp_full_scan(input, output, 1);

    std::vector<int> out_vec(output.get(), output.get() + input.size()+1);
    std::vector<int> expected({0, 42});
    CHECK_VECTOR_EQUAL(out_vec, expected);
}

// Basic functionality
TEST_CASE("omp_top_k_per_row basic functionality", "[top_k]") {
    std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    size_t rows = 3, cols = 4, k = 2;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, false, false, 4, ComparatorGT<int, int>());
    std::vector<size_t> expected_ptrs = {0, 2, 4, 6};
    std::vector<size_t> expected_indices = {2, 3, 2, 3, 2, 3};
    std::vector<int> expected_values = {3, 4, 7, 8, 11, 12};
    CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}

// k = 0
TEST_CASE("omp_top_k_per_row with k=0", "[top_k]") {
    std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    size_t rows = 3, cols = 4, k = 0;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, false, false, 4, ComparatorGT<int, int>());
    std::vector<size_t> expected_ptrs = {0, 0, 0, 0};
    std::vector<size_t> expected_indices = {};
    std::vector<int> expected_values = {};
    CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}

// k > cols
TEST_CASE("omp_top_k_per_row with k > cols", "[top_k]") {
    std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    size_t rows = 3, cols = 4, k = 5;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, false, false, 4, ComparatorGT<int, int>());
    std::vector<size_t> expected_ptrs = {0, 4, 8, 12};
    std::vector<size_t> expected_indices = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<int> expected_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}

// rows = 0
TEST_CASE("omp_top_k_per_row with rows=0", "[top_k]") {
    std::vector<int> matrix = {};
    size_t rows = 0, cols = 4, k = 2;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, false, false, 4, ComparatorGT<int, int>());
    std::vector<size_t> expected_ptrs = {0};
    std::vector<size_t> expected_indices = {};
    std::vector<int> expected_values = {};
    CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}

// cols = 0
TEST_CASE("omp_top_k_per_row with cols=0", "[top_k]") {
    std::vector<int> matrix = {};
    size_t rows = 3, cols = 0, k = 2;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, false, false, 4, ComparatorGT<int, int>());
    std::vector<size_t> expected_ptrs = {0, 0, 0, 0};
    std::vector<size_t> expected_indices = {};
    std::vector<int> expected_values = {};
    CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}

// Less-than comparator
TEST_CASE("omp_top_k_per_row with ComparatorLT", "[top_k]") {
    std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    size_t rows = 3, cols = 4, k = 2;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, false, false, 4, ComparatorLT<int, int>());
    std::vector<size_t> expected_ptrs = {0, 2, 4, 6};
    std::vector<size_t> expected_indices = {0, 1, 0, 1, 0, 1};
    std::vector<int> expected_values = {1, 2, 5, 6, 9, 10};
    CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}

// Floating-point numbers
TEST_CASE("omp_top_k_per_row with floating-point numbers", "[top_k]") {
    std::vector<double> matrix = {1.5, 2.5, 3.5, 4.5, 0.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5};
    size_t rows = 3, cols = 4, k = 2;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, false, false, 4, ComparatorGT<double, double>());
    std::vector<size_t> expected_ptrs = {0, 2, 4, 6};
    std::vector<size_t> expected_indices = {2, 3, 2, 3, 2, 3};
    std::vector<double> expected_values = {3.5, 4.5, 7.5, 8.5, 11.5, 12.5};
    CHECK_CSR_ALMOST_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}

// Duplicates
TEST_CASE("omp_top_k_per_row with duplicates", "[top_k]") {
    std::vector<int> matrix = {1, 3, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8};
    size_t rows = 3, cols = 4, k = 2;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, false, false, 4, ComparatorGT<int, int>());
    std::vector<size_t> expected_ptrs = {0, 2, 4, 6};
    std::vector<size_t> expected_indices = {1, 3, 0, 1, 2, 3};
    std::vector<int> expected_values = {3, 3, 4, 4, 7, 8};
    CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}

// exclude_zeroes = true
TEST_CASE("omp_top_k_per_row with exclude_zeroes=true", "[top_k]") {
    std::vector<int> matrix = {0, 1, 0, 2, 0, 0, 3, 4, 5, 0, 6, 7};
    size_t rows = 3, cols = 4, k = 3;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, true, false, 4, ComparatorGT<int, int>());
    std::vector<size_t> expected_ptrs = {0, 2, 4, 7};
    std::vector<size_t> expected_indices = {1, 3, 2, 3, 0, 2, 3};
    std::vector<int> expected_values = {1, 2, 3, 4, 5, 6, 7};
    CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}

// use_abs = true
TEST_CASE("omp_top_k_per_row with use_abs=true", "[top_k]") {
    std::vector<int> matrix = {-1, 2, -3, 4, -5, 6, -7, 8, 9, -10, 11, -12};
    size_t rows = 3, cols = 4, k = 2;
    auto csr = omp_top_k_per_row(matrix.data(), rows, cols, k, false, true, 4, ComparatorGT<int, int>());
    std::vector<size_t> expected_ptrs = {0, 2, 4, 6};
    std::vector<size_t> expected_indices = {2, 3, 2, 3, 2, 3};
    std::vector<int> expected_values = {-3, 4, -7, 8, 11, -12};
    CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, rows, cols);
}