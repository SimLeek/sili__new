#include "../../sili/lib/headers/linear_sisldo.hpp"
#include "csr.hpp"
#include "fiber.hpp"
#include "sparse_struct.hpp"
#include "tests_main.hpp"
#include <catch2/catch_message.hpp>
#include <vector>


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

TEST_CASE("train loop from zero with RNN fiber and dynamic contraction", "[integration_train_loop]") {
    using SIZE_TYPE = unsigned int;
    using VALUE_TYPE = float;

    constexpr SIZE_TYPE num_iterations = 3;

    // Expected CSRs after input and hidden starmap iterations
    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>>> expected_input_starmap_csrs = {
        {{0, 2, 2}, {1, 2}, {7.22026e-05, 4.68585e-05}}, // After first iteration
        {{0, 2, 2}, {1, 2}, {0.000106619, 9.59013e-05}}, // After second iteration
        {{0, 2, 2}, {1, 2}, {0.000108175, 0.000161712}}  // After third iteration
    };

    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>>> expected_hidden_starmap_csrs = {
        {{0, 1, 1}, {1}, {1.1821e-05}}, // After first iteration
        {{0, 2, 2}, {1, 2}, {6.08639e-05, 1.55606e-06}}, // After second iteration
        {{0, 2, 2}, {0, 1}, {0.000126675, 8.76698e-05}}  // After third iteration
    };

    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>>> expected_input_portion = {
        {{0, 2, 2}, {1, 2}, {0, 0}}, // After first iteration
        {{0, 2, 2}, {1, 2}, {0.1, 0.1}}, // After second iteration
        {{0, 2, 2}, {1, 2}, {0.2, 0.2}}  // After third iteration
    };

    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>>> expected_output_grad_portion = {
        {{0, 1, 1}, {0}, {-0.333333}}, // After first iteration
        {{0, 1, 1}, {0}, {-0.3}}, // After second iteration
        {{0, 1, 1}, {0}, {-0.266627}}  // After third iteration
    };

    // Input data for three iterations
    std::vector<std::array<VALUE_TYPE, 8>> input_values_data({
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
        {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}
    });

    // Expected outputs per iteration
    std::vector<std::vector<VALUE_TYPE>> expected_outputs = {
        {0, 0, 0, 0, 0, 0},
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
        {0.20012, 0.2, 0.2, 0.2, 0.2, 0.2}
    };

    std::vector<std::vector<VALUE_TYPE>> expected_in_grad = {
        {-0.333333, -0.666667, -1, 0, -1.33333, -1.66667, -2, 0},
        {-0.3, -0.633333, -0.966667, 0, -1.3, -1.63333, -1.96667, 0},
        {-0.266627, -0.60008, -0.933413, 0, -1.26667, -1.6, -1.93333, 0}
    };

    // Expected probes for w_ih
    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>>> expected_w_ih_probes = {
        {{1, 2}, {0, 0}, {0, 0}},
        {{1, 2}, {0, 0}, {-0.03, -0.03}},
        {{1, 2}, {0, 0}, {-0.0533253, -0.0533253}}
    };

    // Expected weights for w_ih after each iteration after optim
    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>>> expected_w_ih_weights = {
        {{0, 0, 0, 0}, {}, {}, {}, {}},
        {{0, 0, 1, 2, 2}, {0, 0}, {0.0003, 0.0003}, {0, 0}, {0, 0}},
        {{0, 0, 1, 2}, {0}, {0.000710195}, {0}, {0.833254}}
    };

    // Expected weights for w_ih after each iteration after synaptogenesis
    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>>> expected_w_ih_weights_after_synaptogenesis = {
        {{0, 0, 1, 2, 2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{0, 0, 1, 2, 2}, {0, 0}, {0.0003, 0.0003}, {0, 0}, {0.3, 0.3}},
        {{0, 0, 1, 2}, {0}, {0.000710195}, {0}, {0.833254}}
    };

    // Expected weights for w_hh after each iteration after optim
    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>>> expected_w_hh_weights = {
        {{0, 0, 0}, {}, {}, {}, {}},
        {{0, 0, 1, 2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{0, 0, 1}, {0}, {0}, {0}, {0}}
    };

    // Expected weights for w_hh after each iteration after synaptogenesis
    std::vector<std::tuple<std::vector<SIZE_TYPE>, std::vector<SIZE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>, std::vector<VALUE_TYPE>>> expected_w_hh_weights_after_synaptogenesis = {
        {{0, 0, 0}, {}, {}, {}, {}},
        {{0, 0, 1, 2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
        {{0, 0, 1}, {0}, {0}, {0}, {0}}
    };

    // Set up starmaps for input and hidden before the loop
    CSRInput<SIZE_TYPE, VALUE_TYPE> layer_input_train_bias_tensor;
    layer_input_train_bias_tensor.rows = 2;
    layer_input_train_bias_tensor.cols = 4;
    layer_input_train_bias_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    SIZE_TYPE layer_input_train_bias_ptrs_data[] = {0, 0, 0};
    std::copy(layer_input_train_bias_ptrs_data, layer_input_train_bias_ptrs_data + 3, layer_input_train_bias_tensor.ptrs[0].get());

    auto input_starmap = CSRStarmap(layer_input_train_bias_tensor, 42);

    CSRInput<SIZE_TYPE, VALUE_TYPE> layer_hidden_train_bias_tensor;
    layer_hidden_train_bias_tensor.rows = 2;
    layer_hidden_train_bias_tensor.cols = 3;
    layer_hidden_train_bias_tensor.ptrs = {std::make_unique<SIZE_TYPE[]>(3)};
    SIZE_TYPE layer_hidden_train_bias_ptrs_data[] = {0, 0, 0};
    std::copy(layer_hidden_train_bias_ptrs_data, layer_hidden_train_bias_ptrs_data + 3, layer_hidden_train_bias_tensor.ptrs[0].get());

    auto hidden_starmap = CSRStarmap(layer_hidden_train_bias_tensor, 42);

    // Weights setup before the loop
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> w_ih;
    w_ih.connections.rows = 4;
    w_ih.connections.cols = 3;
    w_ih.connections.ptrs = {std::make_unique<SIZE_TYPE[]>(5)};
    SIZE_TYPE w_ih_ptrs_data[] = {0, 0, 0, 0, 0};
    std::copy(w_ih_ptrs_data, w_ih_ptrs_data + 5, w_ih.connections.ptrs[0].get());

    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> w_hh;
    w_hh.connections.rows = 3;
    w_hh.connections.cols = 3;
    w_hh.connections.ptrs = {std::make_unique<SIZE_TYPE[]>(4)};
    SIZE_TYPE w_hh_ptrs_data[] = {0, 0, 0, 0};
    std::copy(w_hh_ptrs_data, w_hh_ptrs_data + 4, w_hh.connections.ptrs[0].get());

    // Train flag and learning rate setup
    bool train = true;
    float learning_rate = 0.01;
    float solidify = 0.01;

    // Fiber setup
    std::vector<SIZE_TYPE> map_ptrs = {0, 1, 2, 3};
    std::vector<std::vector<SIZE_TYPE>> expected_map_ptrs = {
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {0, 0, 1, 2}
    };
    SIZE_TYPE batch_size = 2;
    SIZE_TYPE current_expanded = map_ptrs.back();
    std::vector<VALUE_TYPE> importances(current_expanded, 0.0f);
    std::vector<VALUE_TYPE> prev_expanded(batch_size * current_expanded, 0.0f);

    for (SIZE_TYPE iter = 0; iter < num_iterations; ++iter) {
        INFO("iter: " << iter);
        input_starmap.iterate(2);
        hidden_starmap.iterate(2);

        // Assert correctness of input and hidden starmap CSRs
        INFO("checking expected_input_starmap_csrs");
        CHECK_CSR_VALUES(input_starmap.csrMatrix, expected_input_starmap_csrs[iter]);

        INFO("checking expected_hidden_starmap_csrs");
        CHECK_CSR_VALUES(hidden_starmap.csrMatrix, expected_hidden_starmap_csrs[iter]);

        auto input_portion = top_k_csr_biased(input_values_data[iter].data(), input_starmap.csrMatrix, 2, 4, 2, 4);

        // Assert correctness of input_portion CSR
        INFO("checking expected_input_portion");
        CHECK_CSR_VALUES(input_portion, expected_input_portion[iter]);

        auto portion_prev = top_k_csr_biased(prev_expanded.data(), hidden_starmap.csrMatrix, 2, current_expanded, 2, 4);

        std::vector<VALUE_TYPE> expanded_state(batch_size * current_expanded, 0.0f);

        sparse_linear_csr_csc_forward(input_portion, w_ih, expanded_state.data(), train, solidify);

        sparse_linear_csr_csc_forward(portion_prev, w_hh, expanded_state.data(), train, solidify);

        SIZE_TYPE contracted_hidden = map_ptrs.size() - 1;
        std::vector<VALUE_TYPE> contracted_state(batch_size * contracted_hidden, 0.0f);

        fiber_contract_forward(expanded_state.data(), contracted_state.data(), batch_size, map_ptrs, importances.data(), 4);

        std::vector<VALUE_TYPE> output(batch_size * contracted_hidden, 0.0f);

        for (SIZE_TYPE i = 0; i < contracted_hidden; ++i) {
            output[i] += contracted_state[i] + input_values_data[iter][i];
            output[i + contracted_hidden] += contracted_state[i + contracted_hidden] + input_values_data[iter][i + 4];
        }

        INFO("checking expected_outputs");
        CHECK_VECTOR_ALMOST_EQUAL(output, expected_outputs[iter]);

        // Compute MSE jacobian
        VALUE_TYPE desired_output[] = {1, 2, 3, 4, 5, 6};
        std::vector<VALUE_TYPE> mse_output_grad(2 * contracted_hidden);
        for (SIZE_TYPE i = 0; i < 2 * contracted_hidden; ++i) {
            mse_output_grad[i] = -2.0 * (desired_output[i] - output[i]) / (2 * contracted_hidden);
        }

        // Compute skip connection grad
        std::vector<VALUE_TYPE> in_grad(8, 0.0f);
        for (SIZE_TYPE i = 0; i < contracted_hidden; ++i) {
            in_grad[i] += mse_output_grad[i];
            in_grad[i + 4] += mse_output_grad[i + contracted_hidden];
        }

        std::vector<VALUE_TYPE> state_grad = mse_output_grad;

        std::vector<VALUE_TYPE> expanded_grad(batch_size * current_expanded, 0.0f);

        fiber_contract_backward(state_grad.data(), expanded_grad.data(), batch_size, map_ptrs, importances.data(), 4);

        auto output_grad_portion = top_k_csr_biased(expanded_grad.data(), hidden_starmap.csrMatrix, 2, current_expanded, 1, 4);

        INFO("checking expected_output_grad_portion");
        CHECK_CSR_VALUES(output_grad_portion, expected_output_grad_portion[iter], (VALUE_TYPE)1e-6);

        sparse_linear_vectorized_backward_is(input_portion, w_ih, output_grad_portion, output_grad_portion,
                                             in_grad.data(), expanded_grad.data(), 4);

        std::vector<VALUE_TYPE> prev_expanded_grad(batch_size * current_expanded, 0.0f);

        sparse_linear_vectorized_backward_is(portion_prev, w_hh, output_grad_portion, output_grad_portion,
                                             prev_expanded_grad.data(), expanded_grad.data(), 4);

        INFO("checking expected_probes");
        CHECK_COO_PROBES(w_ih.probes, expected_w_ih_probes[iter], (VALUE_TYPE)1e-6);

        // Assert in_grad correctness
        INFO("checking expected_in_grad");
        CHECK_VECTOR_ALMOST_EQUAL(in_grad, expected_in_grad[iter], 1e-5);

        optim_weights(w_ih, learning_rate, 4);
        optim_weights(w_hh, learning_rate, 4);

        // Verify weights after optimization
        INFO("checking expected_w_ih_weights optim");
        CHECK_CSR_WEIGHTS(w_ih.connections, expected_w_ih_weights[iter], (VALUE_TYPE)1e-6);

        INFO("checking expected_w_hh_weights optim");
        CHECK_CSR_WEIGHTS(w_hh.connections, expected_w_hh_weights[iter], (VALUE_TYPE)1e-6);

        optim_synaptogenesis(w_ih, learning_rate, (SIZE_TYPE)6, 4);
        optim_synaptogenesis(w_hh, learning_rate, (SIZE_TYPE)6, 4);

        // Verify weights after synaptogenesis
        INFO("checking expected_w_ih_weights_after_synaptogenesis");
        CHECK_CSR_WEIGHTS(w_ih.connections, expected_w_ih_weights_after_synaptogenesis[iter]);

        INFO("checking expected_w_hh_weights_after_synaptogenesis");
        CHECK_CSR_WEIGHTS(w_hh.connections, expected_w_hh_weights_after_synaptogenesis[iter]);

        // Perform fiber contraction within the loop
        SIZE_TYPE neurons_to_change = (iter == 1) ? 1 : 0; // Contract one neuron in iteration 2
        bool add = false;
        std::vector<const VALUE_TYPE*> importances_list = {importances.data()};
        auto add_map_ptrs = fiber_contract_optim(map_ptrs, importances_list, neurons_to_change, add, 4);

        fiber_contract_optim_adjust_columns(add_map_ptrs, std::vector{&w_ih, &w_hh}, add, 4);

        fiber_contract_optim_adjust_rows(add_map_ptrs, std::vector{&w_hh}, add, 4);

        // Adjust hidden_starmap (adjust columns, which for CSR is like adjusting "rows" in terms of indices)
        fiber_contract_optim_adjust_rows(add_map_ptrs, std::vector{&hidden_starmap.csrMatrix}, add, 4);

        // Update importances and related sizes
        if (neurons_to_change > 0) {
            importances.erase(importances.begin());
            current_expanded = map_ptrs.back();
            prev_expanded.resize(batch_size * current_expanded);
            w_ih.connections.cols = map_ptrs.size() - 1;
            w_hh.connections.rows = map_ptrs.size() - 1;
            w_hh.connections.cols = map_ptrs.size() - 1;
            hidden_starmap.csrMatrix.cols = map_ptrs.size() - 1;
        }

        INFO("checking expected_map_ptrs");
        CHECK_VECTOR_EQUAL(map_ptrs, expected_map_ptrs[iter]);

        // Update prev_expanded for next iteration
        prev_expanded = expanded_state;
    }
}
