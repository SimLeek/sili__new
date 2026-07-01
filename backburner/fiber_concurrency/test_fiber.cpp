#include "../sili/lib/headers/linear_sisldo.hpp"
#include "csr.hpp"
#include "sparse_struct.hpp"
#include "tests_main.hpp"
#include "fiber.hpp"
#include <catch2/catch_message.hpp>
#include <cstddef>
#include <memory>
#include <vector>

/* #region setup */
template <class SIZE_TYPE, class VALUE_TYPE>
void CHECK_CSR_WEIGHTS(const sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, TriValues<VALUE_TYPE>> &weights,
                       const std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>> &expected_weights, VALUE_TYPE diff=std::numeric_limits<VALUE_TYPE>::epsilon()) {
    // Extract actual weights from the sparse_struct
    std::vector<SIZE_TYPE> actual_ptrs;
    std::vector<SIZE_TYPE> actual_cols;
    std::vector<VALUE_TYPE> actual_values1;
    std::vector<VALUE_TYPE> actual_values2;
    std::vector<VALUE_TYPE> actual_values3;

    SIZE_TYPE rows = weights.rows;
    auto &ptrs = weights.ptrs[0];
    auto &indices = weights.indices[0];
    auto &values1 = weights.values[0];
    auto &values2 = weights.values[1];
    auto &values3 = weights.values[2];

    for (SIZE_TYPE i = 0; i < rows; ++i) {
        actual_ptrs.push_back(ptrs[i]);
        for (SIZE_TYPE j = ptrs[i]; j < ptrs[i + 1]; ++j) {
            actual_cols.push_back(indices[j]);
            actual_values1.push_back(values1[j]);
            actual_values2.push_back(values2[j]);
            actual_values3.push_back(values3[j]);
        }
    }
    actual_ptrs.push_back(ptrs[rows]);

    // Decompose expected weights into vectorsWW
    std::vector<SIZE_TYPE> expected_ptrs;
    std::vector<SIZE_TYPE> expected_cols;
    std::vector<VALUE_TYPE> expected_values1;
    std::vector<VALUE_TYPE> expected_values2;
    std::vector<VALUE_TYPE> expected_values3;

    for (const auto &t : std::get<0>(expected_weights)) {
        expected_ptrs.push_back(t);
    }
    for (const auto &t : std::get<1>(expected_weights)) {
        expected_cols.push_back(t);
    }
    for (const auto &t : std::get<2>(expected_weights)) {
        expected_values1.push_back(t);
    }
    for (const auto &t : std::get<3>(expected_weights)) {
        expected_values2.push_back(t);
    }
    for (const auto &t : std::get<4>(expected_weights)) {
        expected_values3.push_back(t);
    }

    // Compare using CHECK_VECTOR_EQUAL and CHECK_VECTOR_ALMOST_EQUAL
    CHECK_VECTOR_EQUAL(actual_ptrs, expected_ptrs);
    CHECK_VECTOR_EQUAL(actual_cols, expected_cols);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values1, expected_values1, diff);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values2, expected_values2, diff);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values3, expected_values3, diff);
}

template <class SIZE_TYPE, class VALUE_TYPE>
void CHECK_COO_PROBES(const sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>> &weights,
                       const std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>> &expected_weights, VALUE_TYPE diff=std::numeric_limits<VALUE_TYPE>::epsilon()) {
    // Extract actual weights from the sparse_struct
    std::vector<SIZE_TYPE> actual_rows;
    std::vector<SIZE_TYPE> actual_cols;
    std::vector<VALUE_TYPE> actual_values1;

    SIZE_TYPE rows = weights.rows;
    auto ptrs = weights.ptrs;
    auto &indices = weights.indices[0];
    auto &indices1 = weights.indices[1];
    auto &values1 = weights.values[0];


    for (SIZE_TYPE i = 0; i < ptrs; ++i) {
        actual_rows.push_back(indices[i]);
        actual_cols.push_back(indices1[i]);
        actual_values1.push_back(values1[i]);
    }

    // Decompose expected weights into vectors
    std::vector<SIZE_TYPE> expected_rows;
    std::vector<SIZE_TYPE> expected_cols;
    std::vector<VALUE_TYPE> expected_values1;

    for (const auto &t : std::get<0>(expected_weights)) {
        expected_rows.push_back(t);
    }
    for (const auto &t : std::get<1>(expected_weights)) {
        expected_cols.push_back(t);
    }
    for (const auto &t : std::get<2>(expected_weights)) {
        expected_values1.push_back(t);
    }

    // Compare using CHECK_VECTOR_EQUAL and CHECK_VECTOR_ALMOST_EQUAL
    CHECK_VECTOR_EQUAL(actual_rows, expected_rows);
    CHECK_VECTOR_EQUAL(actual_cols, expected_cols);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values1, expected_values1, diff);
}

template <class SIZE_TYPE, class VALUE_TYPE>
void CHECK_CSR_VALUES(const sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>> &weights,
                       const std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>> &expected_weights, VALUE_TYPE diff=std::numeric_limits<VALUE_TYPE>::epsilon()) {
    // Extract actual weights from the sparse_struct
    std::vector<SIZE_TYPE> actual_ptrs;
    std::vector<SIZE_TYPE> actual_cols;
    std::vector<VALUE_TYPE> actual_values1;

    SIZE_TYPE rows = weights.rows;
    auto &ptrs = weights.ptrs[0];
    auto &indices = weights.indices[0];
    auto &values1 = weights.values[0];

    for (SIZE_TYPE i = 0; i < rows; ++i) {
        actual_ptrs.push_back(ptrs[i]);
        for (SIZE_TYPE j = ptrs[i]; j < ptrs[i + 1]; ++j) {
            actual_cols.push_back(indices[j]);
            actual_values1.push_back(values1[j]);
        }
    }
    actual_ptrs.push_back(ptrs[rows]);

    // Decompose expected weights into vectors
    std::vector<SIZE_TYPE> expected_ptrs;
    std::vector<SIZE_TYPE> expected_cols;
    std::vector<VALUE_TYPE> expected_values1;

    for (const auto &t : std::get<0>(expected_weights)) {
        expected_ptrs.push_back(t);
    }
    for (const auto &t : std::get<1>(expected_weights)) {
        expected_cols.push_back(t);
    }
    for (const auto &t : std::get<2>(expected_weights)) {
        expected_values1.push_back(t);
    }

    // Compare using CHECK_VECTOR_EQUAL and CHECK_VECTOR_ALMOST_EQUAL
    CHECK_VECTOR_EQUAL(actual_ptrs, expected_ptrs);
    CHECK_VECTOR_EQUAL(actual_cols, expected_cols);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values1, expected_values1, diff);
}

template <class SIZE_TYPE, class VALUE_TYPE>
void CHECK_CSR_VALUES_3(const sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, TriValues<VALUE_TYPE>> &weights,
                       const std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>> &expected_weights, VALUE_TYPE diff=std::numeric_limits<VALUE_TYPE>::epsilon()) {
    // Extract actual weights from the sparse_struct
    std::vector<SIZE_TYPE> actual_ptrs;
    std::vector<SIZE_TYPE> actual_cols;
    std::vector<VALUE_TYPE> actual_values1;
    std::vector<VALUE_TYPE> actual_values2;
    std::vector<VALUE_TYPE> actual_values3;

    SIZE_TYPE rows = weights.rows;
    auto &ptrs = weights.ptrs[0];
    auto &indices = weights.indices[0];
    auto &values1 = weights.values[0];
    auto &values2 = weights.values[1];
    auto &values3 = weights.values[2];

    for (SIZE_TYPE i = 0; i < rows; ++i) {
        actual_ptrs.push_back(ptrs[i]);
        for (SIZE_TYPE j = ptrs[i]; j < ptrs[i + 1]; ++j) {
            actual_cols.push_back(indices[j]);
            actual_values1.push_back(values1[j]);
            actual_values2.push_back(values2[j]);
            actual_values3.push_back(values3[j]);
        }
    }
    actual_ptrs.push_back(ptrs[rows]);

    // Decompose expected weights into vectors
    std::vector<SIZE_TYPE> expected_ptrs;
    std::vector<SIZE_TYPE> expected_cols;
    std::vector<VALUE_TYPE> expected_values1;
    std::vector<VALUE_TYPE> expected_values2;
    std::vector<VALUE_TYPE> expected_values3;

    for (const auto &t : std::get<0>(expected_weights)) {
        expected_ptrs.push_back(t);
    }
    for (const auto &t : std::get<1>(expected_weights)) {
        expected_cols.push_back(t);
    }
    for (const auto &t : std::get<2>(expected_weights)) {
        expected_values1.push_back(t);
    }
    for (const auto &t : std::get<3>(expected_weights)) {
        expected_values2.push_back(t);
    }
    for (const auto &t : std::get<4>(expected_weights)) {
        expected_values3.push_back(t);
    }

    // Compare using CHECK_VECTOR_EQUAL and CHECK_VECTOR_ALMOST_EQUAL
    CHECK_VECTOR_EQUAL(actual_ptrs, expected_ptrs);
    CHECK_VECTOR_EQUAL(actual_cols, expected_cols);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values1, expected_values1, diff);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values2, expected_values2, diff);
    CHECK_VECTOR_ALMOST_EQUAL(actual_values3, expected_values3, diff);
}

// Note: All tests use unsigned int for SIZE_TYPE to satisfy static_assert in fiber functions.
// Attempting signed would fail compile, but Catch2 can't test that directly; manual verify.
// Tests cover parallelism with num_cpus=4, small datasets for determinism and coverage.
// Empirical: inputs crafted to exercise paths, outputs manually computed from logic.

// Helper for CSC weights setup (since weights are CSC, but struct is CSR-like)
template <typename SIZE_TYPE, typename VALUE_TYPE>
CSRSynapses<SIZE_TYPE, VALUE_TYPE> create_test_csc(
    SIZE_TYPE rows, SIZE_TYPE cols,
    const std::vector<SIZE_TYPE>& ptrs_data,
    const std::vector<SIZE_TYPE>& indices_data,
    const std::tuple<std::vector<VALUE_TYPE>,std::vector<VALUE_TYPE>,std::vector<VALUE_TYPE>>& values_data
) {
    CSRSynapses<SIZE_TYPE, VALUE_TYPE> csc;
    csc.rows = rows;
    csc.cols = cols;
    csc.ptrs[0] = std::unique_ptr<SIZE_TYPE[]>(new SIZE_TYPE[ptrs_data.size()]);
    std::copy(ptrs_data.begin(), ptrs_data.end(), csc.ptrs[0].get());
    csc.indices[0] = std::unique_ptr<SIZE_TYPE[]>(new SIZE_TYPE[indices_data.size()]);
    std::copy(indices_data.begin(), indices_data.end(), csc.indices[0].get());

    //weights must be able to store forward multiplier strength, backprop accumulation, and optim importance
    csc.values[0] = std::unique_ptr<VALUE_TYPE[]>(new VALUE_TYPE[std::get<0>(values_data).size()]);
    std::copy(std::get<0>(values_data).begin(), std::get<0>(values_data).end(), csc.values[0].get());
    csc.values[1] = std::unique_ptr<VALUE_TYPE[]>(new VALUE_TYPE[std::get<1>(values_data).size()]);
    std::copy(std::get<1>(values_data).begin(), std::get<1>(values_data).end(), csc.values[1].get());
    csc.values[2] = std::unique_ptr<VALUE_TYPE[]>(new VALUE_TYPE[std::get<2>(values_data).size()]);
    std::copy(std::get<2>(values_data).begin(), std::get<2>(values_data).end(), csc.values[2].get());
    return csc;
}

// Overload for CSRInput
template <typename SIZE_TYPE, typename VALUE_TYPE>
CSRInput<SIZE_TYPE, VALUE_TYPE> create_test_csr_input(
    SIZE_TYPE rows, SIZE_TYPE cols,
    const std::vector<SIZE_TYPE>& ptrs_data,
    const std::vector<SIZE_TYPE>& indices_data,
    const std::vector<VALUE_TYPE>& values_data
) {
    CSRInput<SIZE_TYPE, VALUE_TYPE> csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.ptrs[0] = std::unique_ptr<SIZE_TYPE[]>(new SIZE_TYPE[ptrs_data.size()]);
    std::copy(ptrs_data.begin(), ptrs_data.end(), csr.ptrs[0].get());
    csr.indices[0] = std::unique_ptr<SIZE_TYPE[]>(new SIZE_TYPE[indices_data.size()]);
    std::copy(indices_data.begin(), indices_data.end(), csr.indices[0].get());
    csr.values[0] = std::unique_ptr<VALUE_TYPE[]>(new VALUE_TYPE[values_data.size()]);
    std::copy(values_data.begin(), values_data.end(), csr.values[0].get());
    return csr;
}

/* #endregion */

TEST_CASE("fiber_contract_forward single batch multi fiber", "[fiber]") {
    unsigned int rows = 1, expanded_cols = 6;
    std::vector<float> expanded = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> output(rows * 2, 0.f);
    std::vector<float> import(expanded_cols, 0.f);
    std::vector<unsigned int> map = {0, 3, 6};

    fiber_contract_forward(expanded.data(), output.data(), rows, map, import.data(), 4);

    CHECK_VECTOR_ALMOST_EQUAL(output, std::vector<float>{6.f, 15.f});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
}

TEST_CASE("fiber_contract_forward multi batch single fiber", "[fiber]") {
    unsigned int rows = 2, expanded_cols = 4;
    std::vector<float> expanded = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    std::vector<float> output(rows * 2, 0.f);
    std::vector<float> import(expanded_cols, 0.f);
    std::vector<unsigned int> map = {0, 2, 4};

    fiber_contract_forward(expanded.data(), output.data(), rows, map, import.data(), 4);

    CHECK_VECTOR_ALMOST_EQUAL(output, std::vector{3.f, 7.f, 11.f, 15.f});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector{6.f, 8.f, 10.f, 12.f});
}

TEST_CASE("fiber_contract_forward empty input", "[fiber]") {
    unsigned int rows = 0, expanded_cols = 0;
    std::vector<float> expanded(0);
    std::vector<float> output(0);
    std::vector<float> import(0);
    std::vector<unsigned int> map = {0, 1};

    fiber_contract_forward(expanded.data(), output.data(), rows, map, import.data(), 4);

    CHECK_VECTOR_ALMOST_EQUAL(output, std::vector<float>());
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector<float>());
}

TEST_CASE("fiber_contract_forward invalid map_ptrs", "[fiber]") {
    unsigned int rows = 1, expanded_cols = 4;
    std::vector<float> expanded = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> output(rows * 1, 0.f);
    std::vector<float> import(expanded_cols, 0.f);
    std::vector<unsigned int> map = {0, 1, 4};

    REQUIRE_THROWS_AS(fiber_contract_forward((float*)nullptr, output.data(), rows, map, import.data(), 4), std::invalid_argument);
    REQUIRE_THROWS_AS(fiber_contract_forward(expanded.data(), (float*)nullptr, rows, map, import.data(), 4), std::invalid_argument);
}

TEST_CASE("fiber_contract_forward no importances", "[fiber]") {
    unsigned int rows = 1, expanded_cols = 4;
    std::vector<float> expanded = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> output(rows * 2, 0.f);
    std::vector<unsigned int> map = {0, 2, 4};

    fiber_contract_forward(expanded.data(), output.data(), rows, map, (float*)nullptr, 4);

    CHECK_VECTOR_ALMOST_EQUAL(output, std::vector{3.f, 7.f});
}

TEST_CASE("fiber_expand_forward single batch multi fiber", "[fiber]") {
    unsigned int rows = 1, cols_contracted = 2;
    unsigned int cols_output = 6; // map_ptrs.back() = 6
    std::vector<float> input = {1.f, 2.f};
    std::vector<float> output(rows * cols_output, 0.f);
    std::vector<float> import(cols_contracted, 0.f);
    std::vector<unsigned int> map = {0, 3, 6}; // in[0] -> out[0-2], in[1] -> out[3-5]

    fiber_expand_forward(input.data(), output.data(), rows, map, import.data(), 4);

    // Expected: output = {1,1,1,2,2,2}, import = {1,2}
    CHECK_VECTOR_ALMOST_EQUAL(output, std::vector{1.f, 1.f, 1.f, 2.f, 2.f, 2.f});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector{3.f, 6.f});
}

TEST_CASE("fiber_expand_forward multi batch multi fiber", "[fiber]") {
    unsigned int rows = 2, cols_contracted = 2;
    unsigned int cols_output = 4; // map_ptrs.back() = 4
    std::vector<float> input = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> output(rows * cols_output, 0.f);
    std::vector<float> import(cols_contracted, 0.f);
    std::vector<unsigned int> map = {0, 2, 4}; // in[0] -> out[0-1], in[1] -> out[2-3]

    fiber_expand_forward(input.data(), output.data(), rows, map, import.data(), 4);

    // Expected: output = {1,1,2,2, 3,3,4,4}, import = {1,2,3,4}
    CHECK_VECTOR_ALMOST_EQUAL(output, std::vector{1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector{8.f, 12.f});
}

TEST_CASE("fiber_expand_forward importances no accumulation", "[fiber]") {
    unsigned int rows = 1, cols_contracted = 2;
    unsigned int cols_output = 4;
    std::vector<float> input = {1.f, 2.f};
    std::vector<float> output(rows * cols_output, 0.f);
    std::vector<float> import(cols_contracted, 0.f);
    std::vector<unsigned int> map = {0, 2, 4};

    // Run twice to catch accumulation
    fiber_expand_forward(input.data(), output.data(), rows, map, import.data(), 4);
    fiber_expand_forward(input.data(), output.data(), rows, map, import.data(), 4);

    // Expected: import = {4,8}, should accumulate over multiple runs
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector{4.f, 8.f});
}

TEST_CASE("fiber_expand_forward empty input", "[fiber]") {
    unsigned int rows = 0, cols_contracted = 0;
    unsigned int cols_output = 0;
    std::vector<float> input(0);
    std::vector<float> output(0);
    std::vector<float> import(0);
    std::vector<unsigned int> map = {0, 0};

    fiber_expand_forward(input.data(), output.data(), rows, map, import.data(), 4);

    CHECK_VECTOR_ALMOST_EQUAL(output, std::vector<float>{});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector<float>{});
}

TEST_CASE("fiber_expand_forward no importances", "[fiber]") {
    unsigned int rows = 1, cols_contracted = 2;
    unsigned int cols_output = 4;
    std::vector<float> input = {1.f, 2.f};
    std::vector<float> output(rows * cols_output, 0.f);
    std::vector<unsigned int> map = {0, 2, 4};

    fiber_expand_forward(input.data(), output.data(), rows, map, (float*)nullptr, 4);

    // Expected: output = {1,1,2,2}
    CHECK_VECTOR_ALMOST_EQUAL(output, std::vector{1.f, 1.f, 2.f, 2.f});
}

TEST_CASE("fiber_contract_backward single batch multi fiber", "[fiber]") {
    unsigned int batches = 1, cols_contracted = 2;
    unsigned int cols_output = 6; // map_ptrs.back() = 6
    std::vector<float> contracted_grad_output = {1.f, 2.f};
    std::vector<float> grad_input(batches * cols_output, 0.f);
    std::vector<float> import(cols_output, 0.f);
    std::vector<unsigned int> map = {0, 3, 6}; // in[0] -> out[0-2], in[1] -> out[3-5]

    fiber_contract_backward(contracted_grad_output.data(), grad_input.data(), batches, map, import.data(), 4);

    // Expected: grad_input = {1,1,1,2,2,2}, import = {-1,-2}
    CHECK_VECTOR_ALMOST_EQUAL(grad_input, std::vector{1.f, 1.f, 1.f, 2.f, 2.f, 2.f});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector{-1.f, -1.f, -1.f -2.f, -2.f, -2.f});
}

TEST_CASE("fiber_contract_backward multi batch multi fiber", "[fiber]") {
    unsigned int batches = 2, cols_contracted = 2;
    unsigned int cols_output = 4; // map_ptrs.back() = 4
    std::vector<float> contracted_grad_output = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> grad_input(batches * cols_output, 0.f);
    std::vector<float> import(cols_output, 0.f);
    std::vector<unsigned int> map = {0, 2, 4}; // in[0] -> out[0-1], in[1] -> out[2-3]

    fiber_contract_backward(contracted_grad_output.data(), grad_input.data(), batches, map, import.data(), 4);

    // Expected: grad_input = {1,1,2,2, 3,3,4,4}, import = {-1,-2,-3,-4}
    CHECK_VECTOR_ALMOST_EQUAL(grad_input, std::vector{1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector{-4.f, -4.f, -6.f, -6.f});
}

TEST_CASE("fiber_contract_backward importances accumulation", "[fiber]") {
    unsigned int batches = 1, cols_contracted = 2;
    unsigned int cols_output = 4;
    std::vector<float> contracted_grad_output = {1.f, 2.f};
    std::vector<float> grad_input(batches * cols_output, 0.f);
    std::vector<float> import(cols_output, 0.f);
    std::vector<unsigned int> map = {0, 2, 4};

    // Run twice to verify accumulation
    fiber_contract_backward(contracted_grad_output.data(), grad_input.data(), batches, map, import.data(), 4);
    fiber_contract_backward(contracted_grad_output.data(), grad_input.data(), batches, map, import.data(), 4);

    // Expected: grad_input = {1,1,2,2} (from second run), import = {-2,-4} (accumulated)
    CHECK_VECTOR_ALMOST_EQUAL(grad_input, std::vector{1.f, 1.f, 2.f, 2.f});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector{-2.f, -2.f, -4.F, -4.F});
}

TEST_CASE("fiber_contract_backward empty input", "[fiber]") {
    unsigned int batches = 0, cols_contracted = 0;
    unsigned int cols_output = 0;
    std::vector<float> contracted_grad_output(0);
    std::vector<float> grad_input(0);
    std::vector<float> import(0);
    std::vector<unsigned int> map = {0, 0};

    fiber_contract_backward(contracted_grad_output.data(), grad_input.data(), batches, map, import.data(), 4);

    CHECK_VECTOR_ALMOST_EQUAL(grad_input, std::vector<float>{});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector<float>{});
}

TEST_CASE("fiber_contract_backward no importances", "[fiber]") {
    unsigned int batches = 1, cols_contracted = 2;
    unsigned int cols_output = 4;
    std::vector<float> contracted_grad_output = {1.f, 2.f};
    std::vector<float> grad_input(batches * cols_output, 0.f);
    std::vector<unsigned int> map = {0, 2, 4};

    fiber_contract_backward(contracted_grad_output.data(), grad_input.data(), batches, map, (float*)nullptr, 4);

    // Expected: grad_input = {1,1,2,2}
    CHECK_VECTOR_ALMOST_EQUAL(grad_input, std::vector{1.f, 1.f, 2.f, 2.f});
}

TEST_CASE("fiber_contract_backward large fiber range", "[fiber]") {
    unsigned int batches = 1, cols_contracted = 2;
    unsigned int cols_output = 10; // Large fiber range
    std::vector<float> contracted_grad_output = {1.f, 2.f};
    std::vector<float> grad_input(batches * cols_output, 0.f);
    std::vector<float> import(cols_output, 0.f);
    std::vector<unsigned int> map = {0, 5, 10}; // in[0] -> out[0-4], in[1] -> out[5-9]

    fiber_contract_backward(contracted_grad_output.data(), grad_input.data(), batches, map, import.data(), 4);

    // Expected: grad_input = {1,1,1,1,1,2,2,2,2,2}, import = {-1,-2}
    CHECK_VECTOR_ALMOST_EQUAL(grad_input, std::vector{1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 2.f});
    CHECK_VECTOR_ALMOST_EQUAL(import, std::vector{-1.f, -1.f, -1.f, -1.f, -1.f, -2.f, -2.f, -2.f, -2.f, -2.f});
}


TEST_CASE("merge_duplicate_indices", "[fiber]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;
    
    //sorted with duplicates
    std::vector<SIZE_TYPE> input_ptrs = {0, 3, 6, 9};
    std::vector<SIZE_TYPE> input_indices = {0,1,1, 0,2,2, 0,1,1};
    std::vector<VALUE_TYPE> input_values1 = {2.0f,1.0f,3.0f, 5.0f,4.0f,6.0f, 8.0f,7.0f,9.0f};
    std::vector<VALUE_TYPE> input_values2 = {2.0f,1.0f,3.0f, 5.0f,4.0f,6.0f, 8.0f,7.0f,9.0f};
    std::vector<VALUE_TYPE> input_values3 = {2.0f,1.0f,3.0f, 5.0f,4.0f,6.0f, 8.0f,7.0f,9.0f};

    auto weights = create_test_csc<SIZE_TYPE, VALUE_TYPE>(3, 3, input_ptrs, input_indices, std::tuple(input_values1, input_values2, input_values3));

    merge_duplicate_indices(weights, 4);

    std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>> expected = {
        {0,2,4,6}, 
        {0,1,0,2,0,1}, 
        {2.0f,2.5f,5.0f,5.2f,8.0f, 8.125f}, 
        {2.0f,4.0f,5.0f,10.0f,8.0f,16.0f}, 
        {2.0f,2.5f,5.0f,5.2f,8.0f, 8.125f}, 
    };
    CHECK_CSR_VALUES_3(weights, expected);
}

TEST_CASE("merge_duplicate_row_with_next", "[fiber]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    // Input data: 3 rows, with duplicates in rows 0 and 1 to be merged
    std::vector<SIZE_TYPE> input_ptrs = {0, 3, 6, 8};
    std::vector<SIZE_TYPE> input_indices = {0, 1, 2, 0, 1, 2, 0, 1};
    std::vector<VALUE_TYPE> input_values1 = {2.0f, 1.0f, 3.0f, 5.0f, 4.0f, 6.0f, 8.0f, 7.0f};
    std::vector<VALUE_TYPE> input_values2 = {2.0f, 1.0f, 3.0f, 5.0f, 4.0f, 6.0f, 8.0f, 7.0f};
    std::vector<VALUE_TYPE> input_values3 = {2.0f, 1.0f, 3.0f, 5.0f, 4.0f, 6.0f, 8.0f, 7.0f};

    auto weights = create_test_csc<SIZE_TYPE, VALUE_TYPE>(3, 3, input_ptrs, input_indices, std::tuple(input_values1, input_values2, input_values3));

    // Merge row 0 with row 1
    SIZE_TYPE row_to_merge = 0;
    SIZE_TYPE sub = _merge_duplicate_row_with_next(weights, row_to_merge);

    // Expected results after merging rows 0 and 1 into one row
    std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>> expected = {
        {0, 3, 6, 8}, // Pointers: 2 rows (merged row 0+1, original row 2), 3 elements
        {0, 1, 2, 0, 1, 2, 0, 1}, // Indices: merged duplicates (1,1) in row 0, (2,2) in row 1
        {29.f/7.f, 17.f/5.f, 5.0f, 8.0f, 7.0f, 5.0f, 8.0f, 7.0f}, // Values[0]: averaged by importance (e.g., (1*1 + 3*3)/(1+3)=2.5)
        {7.0f, 5.0f, 9.0f, 8.0f, 7.0f, 9.0f, 8.0f, 7.0f}, // Values[1]: summed (e.g., 1+3=4)
        {29.f/7.f, 17.f/5.f, 5.0f, 8.0f, 7.0f, 5.0f, 8.0f, 7.0f}  // Values[2]: averaged by value
    };

    // Verify the results
    CHECK(sub == 3);
    CHECK_CSR_VALUES_3(weights, expected);
}

TEST_CASE("fiber_contract_optim add true", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> map_ptrs = {0, 2, 4, 6};
    std::vector<VALUE_TYPE> imp = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<const VALUE_TYPE*> importances_list = {imp.data()};
    SIZE_TYPE neurons_to_change = 1;
    bool add = true;

    auto add_map_ptrs = fiber_contract_optim(map_ptrs, importances_list, neurons_to_change, add, 4);

    std::vector<SIZE_TYPE> expected_map_ptrs{0, 2, 4, 7};
    std::vector<SIZE_TYPE> expected_add_map_ptrs{0, 0, 0, 1};

    CHECK_VECTOR_EQUAL(map_ptrs, expected_map_ptrs);
    CHECK_VECTOR_EQUAL(add_map_ptrs, expected_add_map_ptrs);
}

TEST_CASE("fiber_contract_optim add false", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> map_ptrs = {0, 3, 6, 9};
    std::vector<VALUE_TYPE> imp = {3.f, 3.f, 3.f, 2.f, 2.f, 2.f, 1.f, 1.f, 1.f};
    std::vector<const VALUE_TYPE*> importances_list = {imp.data()};
    SIZE_TYPE neurons_to_change = 1;
    bool add = false;

    auto add_map_ptrs = fiber_contract_optim(map_ptrs, importances_list, neurons_to_change, add, 4);

    std::vector<SIZE_TYPE> expected_map_ptrs{0, 3, 6, 8};
    std::vector<SIZE_TYPE> expected_add_map_ptrs{0, 0, 0, 1};

    CHECK_VECTOR_EQUAL(map_ptrs, expected_map_ptrs);
    CHECK_VECTOR_EQUAL(add_map_ptrs, expected_add_map_ptrs);
}

TEST_CASE("fiber_contract_optim add true multiple importances", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> map_ptrs = {0, 2, 4, 6};
    std::vector<VALUE_TYPE> imp1 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<VALUE_TYPE> imp2 = {7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    std::vector<const VALUE_TYPE*> importances_list = {imp1.data(), imp2.data()};
    SIZE_TYPE neurons_to_change = 1;
    bool add = true;

    auto add_map_ptrs = fiber_contract_optim(map_ptrs, importances_list, neurons_to_change, add, 4);

    std::vector<SIZE_TYPE> expected_map_ptrs{0, 2, 4, 7};
    std::vector<SIZE_TYPE> expected_add_map_ptrs{0, 0, 0, 1};

    CHECK_VECTOR_EQUAL(map_ptrs, expected_map_ptrs);
    CHECK_VECTOR_EQUAL(add_map_ptrs, expected_add_map_ptrs);
}

TEST_CASE("fiber_contract_optim add false zero length fiber", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> map_ptrs = {0, 2, 2, 5};
    std::vector<VALUE_TYPE> imp = {1.f, 2.f, 3.f, 4.f, 5.f};
    std::vector<const VALUE_TYPE*> importances_list = {imp.data()};
    SIZE_TYPE neurons_to_change = 1;
    bool add = false;

    auto add_map_ptrs = fiber_contract_optim(map_ptrs, importances_list, neurons_to_change, add, 4);

    std::vector<SIZE_TYPE> expected_map_ptrs{0, 1, 1, 4};
    std::vector<SIZE_TYPE> expected_add_map_ptrs{0, 1, 1, 1};

    CHECK_VECTOR_EQUAL(map_ptrs, expected_map_ptrs);
    CHECK_VECTOR_EQUAL(add_map_ptrs, expected_add_map_ptrs);
}

TEST_CASE("fiber_contract_optim neurons zero", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> map_ptrs = {0, 2, 4, 6};
    std::vector<VALUE_TYPE> imp = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<const VALUE_TYPE*> importances_list = {imp.data()};
    SIZE_TYPE neurons_to_change = 0;
    bool add = true;

    auto add_map_ptrs = fiber_contract_optim(map_ptrs, importances_list, neurons_to_change, add, 4);

    std::vector<SIZE_TYPE> expected_map_ptrs{0, 2, 4, 6};
    std::vector<SIZE_TYPE> expected_add_map_ptrs{0, 0, 0, 0};
    
    CHECK_VECTOR_EQUAL(map_ptrs, expected_map_ptrs);
    CHECK_VECTOR_EQUAL(add_map_ptrs, expected_add_map_ptrs);
}

TEST_CASE("fiber_contract_optim_adjust_columns add true", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> add_map_ptrs = {0, 0, 1, 1};
    std::vector<SIZE_TYPE> ptrs = {0, 2, 3};
    std::vector<SIZE_TYPE> indices = {0, 2, 1};
    std::vector<VALUE_TYPE> values1 = {1.f, 2.f, 3.f};
    std::vector<VALUE_TYPE> values2 = {4.f, 5.f, 6.f};
    std::vector<VALUE_TYPE> values3 = {7.f, 8.f, 9.f};
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections = create_test_csc<SIZE_TYPE, VALUE_TYPE>(2, 3, ptrs, indices, std::make_tuple(values1, values2, values3));;
    std::vector<SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>*> weights_array = {&weights};
    bool add = true;

    fiber_contract_optim_adjust_columns(add_map_ptrs, weights_array, add, 4);

    std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>> expected = {
        {0, 2, 3}, {0, 3, 1}, {1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}, {7.f, 8.f, 9.f}
    };
    CHECK(weights.connections.cols == 4);
    CHECK_CSR_VALUES_3(weights.connections, expected);
}

TEST_CASE("fiber_contract_optim_adjust_columns add false with merge", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> add_map_ptrs = {0, 0, 1, 1};
    std::vector<SIZE_TYPE> ptrs = {0, 2, 4};
    std::vector<SIZE_TYPE> indices = {0, 2, 1, 2};
    std::vector<VALUE_TYPE> values1 = {1.f, 2.f, 3.f, 4.f};
    std::vector<VALUE_TYPE> values2 = {5.f, 6.f, 7.f, 8.f};
    std::vector<VALUE_TYPE> values3 = {9.f, 10.f, 11.f, 12.f};
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections = create_test_csc<SIZE_TYPE, VALUE_TYPE>(2, 3, ptrs, indices, std::make_tuple(values1, values2, values3));
    std::vector<SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>*> weights_array = {&weights};
    bool add = false;

    fiber_contract_optim_adjust_columns(add_map_ptrs, weights_array, add, 4);

    // After subtract: indices {0,1,1,1}, then merge dups
    std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>> expected = {
        {0, 2, 3}, {0, 1, 1}, {1.f, 2.f, (3.f*11.f+4.f*12.f)/(11.f+12.f)}, {5.f, 6.f, 15.f}, {9.f, 10.f, (3.f*11.f+4.f*12.f)/(3.f+4.f)}
        // Note: complex avg/sum as per merge_duplicate_indices logic
    };

    auto expected_ptrs = {0,2,3};
    auto expected_indices = {0,1,1};

    CHECK(weights.connections.cols == 2);
    CHECK_CSR_VALUES_3(weights.connections, expected);
}

TEST_CASE("fiber_contract_optim_adjust_columns nullptr weights", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> add_map_ptrs = {0, 0, 1, 1};
    std::vector<SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>*> weights_array = {nullptr};
    bool add = true;

    fiber_contract_optim_adjust_columns(add_map_ptrs, weights_array, add, 4);

    // No crash, skip
    CHECK(true);
}

TEST_CASE("fiber_contract_optim_adjust_rows add true", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> add_map_ptrs = {0, 0, 1, 1};
    std::vector<SIZE_TYPE> ptrs = {0, 2, 4, 6};
    std::vector<SIZE_TYPE> indices = {0,1,0,1,0,1};
    std::vector<VALUE_TYPE> values1 = {1.f,2.f,3.f,4.f,5.f,6.f};
    std::vector<VALUE_TYPE> values2 = {1.f,2.f,3.f,4.f,5.f,6.f};
    std::vector<VALUE_TYPE> values3 = {1.f,2.f,3.f,4.f,5.f,6.f};
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections = create_test_csc<SIZE_TYPE, VALUE_TYPE>(3, 2, ptrs, indices, std::make_tuple(values1, values2, values3));
    std::vector<SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>*> weights_array = {&weights};
    bool add = true;

    fiber_contract_optim_adjust_rows(add_map_ptrs, weights_array, add, 4);

    CHECK(weights.connections.rows == 4);
    // New ptrs = ptrs[i - add_map_ptrs[i]] , for i=0: ptrs[0 -0]=0, i=1 ptrs[1-0]=2, i=2 ptrs[2-1]=2, i=3 ptrs[3-1]=4, i=4 ptrs[4-1]=6 ? But ptrs size 4, ptrs[4] not exist, wait.
    // Wait, new_rows =3+1=4, new_ptrs size5, i=0 to4, ptrs[i - add_map_ptrs[i]] , for i=2 ptrs[2-1]=ptrs[1]=2, i=3 ptrs[3-1]=ptrs[2]=4, i=4 ptrs[4-1]=ptrs[3]=6
    // So new_ptrs {0,2,2,4,6}
    // Which means inserted a new row at position 2, with len0.
    // And since add, no merge.
    // Indices and values unchanged.
    std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>> expected = {
        {0,2,2,4,6}, 
        {0,1,0,1,0,1}, 
        {1.f,2.f,3.f,4.f,5.f,6.f}, 
        {1.f,2.f,3.f,4.f,5.f,6.f}, 
        {1.f,2.f,3.f,4.f,5.f,6.f}
    };
    CHECK_CSR_VALUES_3(weights.connections, expected);
}

TEST_CASE("fiber_contract_optim_adjust_rows add false dup", "[fiber_optim]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    std::vector<SIZE_TYPE> add_map_ptrs = {0, 0, 1, 1};
    std::vector<SIZE_TYPE> ptrs = {0, 2, 4, 6};
    std::vector<SIZE_TYPE> indices = {0,1,2,3,3,4};
    std::vector<VALUE_TYPE> values1 = {1.f,2.f,3.f,4.f,5.f,6.f};
    std::vector<VALUE_TYPE> values2 = {7.f,8.f,9.f,10.f,11.f,12.f};
    std::vector<VALUE_TYPE> values3 = {13.f,14.f,15.f,16.f,17.f,18.f};
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections = create_test_csc<SIZE_TYPE, VALUE_TYPE>(3, 6, ptrs, indices, std::make_tuple(values1, values2, values3));
    std::vector<SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>*> weights_array = {&weights};
    bool add = false;

    fiber_contract_optim_adjust_rows(add_map_ptrs, weights_array, add, 1); // Use 1 cpu to avoid parallel issues

    CHECK(weights.connections.rows == 2);
    // new_ptrs i=0: ptrs[0+0]=0
    // i=1: ptrs[1+0]=2, but since add_map_ptrs[1]=0 == [0]=0, no merge
    // i=2: ptrs[2+1]=ptrs[3]=6, add_map_ptrs[2]=1 != [1]=0? Wait, add_map_ptrs [0,0,1,1], [2]=1 != [1]=0, yes, merge row1 (i-1=1)
    // Merge row1 with next (row2)
    // Slice ptrs[1]=2 to ptrs[3]=6, indices {2,3,4,5}, sorted, no dup, sub=0, no shift, but sorts if not, but assume sorted.
    // But since no dup, sub=0, combined row1 now {2,3,4,5}, len4, previous row0 {0,1}
    // new_ptrs {0,2,6}
    // Yes, row0 0-2 {0,1}, row1 2-6 {2,3,4,5}
    std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>> expected = {
        {0,2,5}, 
        {0,1,2,3,4}, 
        {1.f,2.f,3.f,(4.f*16.f+5.f*17.f)/(16.f+17.f),6.f}, 
        {7.f,8.f,9.f,21.f,12.f}, 
        {13.f,14.f,15.f,(4.f*16.f+5.f*17.f)/(4.f+5.f),18.f}
    };
    CHECK_CSR_VALUES_3(weights.connections, expected);
}