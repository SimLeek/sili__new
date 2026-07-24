#include "../../sili/lib/headers/linear_sisldo.hpp"
#include "csr.hpp"
#include "sparse_struct.hpp"
#include "tests_main.hpp"
#include <catch2/catch_message.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

// Compute MSE gradient: -2*(desired-output)/n, returns dense vector
template <typename VALUE_TYPE>
std::vector<VALUE_TYPE> mse_grad(
    const std::vector<VALUE_TYPE>& output,
    const std::vector<VALUE_TYPE>& desired)
{
    const size_t n = output.size();
    std::vector<VALUE_TYPE> grad(n);
    for (size_t i = 0; i < n; ++i)
        grad[i] = VALUE_TYPE(-2) * (desired[i] - output[i]) / VALUE_TYPE(n);
    return grad;
}

// Compute MSE scalar loss
template <typename VALUE_TYPE>
VALUE_TYPE mse_loss(
    const std::vector<VALUE_TYPE>& output,
    const std::vector<VALUE_TYPE>& desired)
{
    VALUE_TYPE loss = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        VALUE_TYPE d = desired[i] - output[i];
        loss += d * d;
    }
    return loss / VALUE_TYPE(output.size());
}

// ── Integration test: multi-iteration train loop ──────────────────────────────
// No starmaps. Fixed sparse input, fixed targets.
// Verifies: forward output correctness, weight grad direction, loss decreases.

TEST_CASE("integration_train_loop", "[integration_train_loop]") {
    using SIZE_TYPE  = int;
    using VALUE_TYPE = float;

    // 4 inputs, 3 outputs, batch_size=2
    // batch0: inputs [0,2] values [1.0, 0.5]
    // batch1: inputs [1,3] values [2.0, 1.5]
    const auto input = make_csr_input<SIZE_TYPE, VALUE_TYPE>(
        2, 4,
        {0, 2, 4},
        {0, 2, 1, 3},
        {1.0f, 0.5f, 2.0f, 1.5f});

    // out_grad_sparse = full dense output grad (all outputs active both batches)
    // ptrs built per-iteration after computing MSE grad
    const SIZE_TYPE n_inputs  = 4;
    const SIZE_TYPE n_outputs = 3;
    const SIZE_TYPE batch_size= 2;
    const int       num_cpus  = 4;
    const VALUE_TYPE lr       = 0.01f;
    const VALUE_TYPE solidify = 0.01f;
    const SIZE_TYPE  max_weights = 8;

    // Initial weights: 8 connections, small positive values
    // in0→out0(0.3), in0→out1(0.4)
    // in1→out1(0.5), in1→out2(0.6)
    // in2→out0(0.7), in2→out2(0.8)
    // in3→out1(0.4), in3→out2(0.5)
    auto weights = make_weights<SIZE_TYPE, VALUE_TYPE>(
        n_inputs, n_outputs,
        {0, 2, 4, 6, 8},
        {0, 1, 1, 2, 0, 2, 1, 2},
        {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.4f, 0.5f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    std::vector<VALUE_TYPE> desired   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<VALUE_TYPE> neuron_input_accum(n_inputs,  0.0f);
    std::vector<VALUE_TYPE> neuron_grad_accum (n_outputs, 0.0f);

    VALUE_TYPE prev_loss = std::numeric_limits<VALUE_TYPE>::max();

    for (int iter = 0; iter < 5; ++iter) {
        INFO("iter: " << iter);

        // ── Forward ───────────────────────────────────────────────────────────
        std::vector<VALUE_TYPE> output(batch_size * n_outputs, 0.0f);
        sisldo_forward(input, weights, output.data(), true, solidify, num_cpus);

        const VALUE_TYPE loss = mse_loss(output, desired);
        INFO("loss: " << loss);

        // Verify forward result matches manual computation for iter 0
        if (iter == 0) {
            // batch0: in0*[0.3,0.4] + in2*[0.7,0,0.8] = [0.3+0.35, 0.4, 0.4]
            //       = [0.65, 0.4, 0.4]
            // batch1: in1*[0,0.5,0.6] + in3*[0,0.4,0.5]
            //       = [0, 1.0+0.6, 1.2+0.75] = [0, 1.6, 1.95]
            CHECK_VECTOR_ALMOST_EQUAL(output,
                std::vector<VALUE_TYPE>({0.65f, 0.4f, 0.4f, 0.0f, 1.6f, 1.95f}),
                1e-4f);
        }

        // ── MSE gradient and sparse out_grad ──────────────────────────────────
        auto grad = mse_grad(output, desired);

        // Build sparse out_grad_sparse (all outputs present, both batches)
        auto out_grad_sparse = make_csr_input<SIZE_TYPE, VALUE_TYPE>(
            batch_size, n_outputs,
            {0, 3, 6},
            {0, 1, 2, 0, 1, 2},
            {grad[0], grad[1], grad[2], grad[3], grad[4], grad[5]});

        // output_gradients layout: [out0_b0, out0_b1, out1_b0, out1_b1, out2_b0, out2_b1]
        std::vector<VALUE_TYPE> output_gradients = {
            grad[0], grad[3], grad[1], grad[4], grad[2], grad[5]};

        // ── Backward ──────────────────────────────────────────────────────────
        std::vector<VALUE_TYPE> input_gradients(n_inputs, 0.0f);
        sisldo_backward(input, weights, out_grad_sparse,
                        input_gradients.data(), output_gradients.data(),
                        neuron_input_accum.data(), neuron_grad_accum.data(),
                        num_cpus);

        // All active inputs should receive nonzero input gradient
        // (since all outputs have nonzero grad and all inputs have connections)
        for (SIZE_TYPE i = 0; i < n_inputs; ++i) {
            INFO("input_grad[" << i << "] = " << input_gradients[i]);
            CHECK(input_gradients[i] != 0.0f);
        }

        // Weight gradients should all be nonzero (all active inputs connect to outputs)
        for (SIZE_TYPE j = 0; j < (SIZE_TYPE)weights.connections.nnz(); ++j) {
            INFO("weight_grad[" << j << "] = " << (*weights.connections.values[1])[j]);
            CHECK((*weights.connections.values[1])[j] != 0.0f);
        }

        // ── Optimize weights ──────────────────────────────────────────────────
        sisldo_optim_weights(weights, lr, num_cpus);

        // Gradients should be zeroed after optim
        CHECK_VECTOR_ALMOST_EQUAL(*weights.connections.values[1],
            std::vector<VALUE_TYPE>(8, 0.0f));

        // Loss should decrease (or at worst stay the same) over iterations
        // Skip iter 0 since we have no prior loss
        if (iter > 0) {
            INFO("prev_loss=" << prev_loss << " curr_loss=" << loss);
            CHECK(loss <= prev_loss * 1.01f); // allow 1% tolerance for fp noise
        }
        prev_loss = loss;
    }

    // Accumulators should be nonzero after 5 iterations
    for (SIZE_TYPE i = 0; i < n_inputs; ++i)
        CHECK(neuron_input_accum[i] > 0.0f);
    for (SIZE_TYPE j = 0; j < n_outputs; ++j)
        CHECK(neuron_grad_accum[j] > 0.0f);
}

// ── Integration test: neurogenesis ───────────────────────────────────────────
// Starts with NO connections. Runs several iterations to accumulate
// neuron_input_accum and neuron_grad_accum. Then calls genesis_build_probes
// followed by sisldo_optim_synaptogenesis. Verifies new connections are grown
// and connect the correct high-activity neurons.

TEST_CASE("integration_neurogenesis", "[integration_neurogenesis]") {
    using SIZE_TYPE  = int;
    using VALUE_TYPE = float;

    // Setup: 4 inputs, 3 outputs, batch_size=1
    // Use inputs with very different magnitudes so top-k is unambiguous:
    //   input 1 (val=3.0) and input 3 (val=1.0) are active
    // Use output grads with clear top-k:
    //   output 0 (val=2.0) and output 2 (val=0.5) are dominant
    const SIZE_TYPE n_inputs  = 4;
    const SIZE_TYPE n_outputs = 3;
    const SIZE_TYPE k         = 2;  // top-2 inputs, top-2 outputs
    const int       num_cpus  = 4;

    auto input = make_csr_input<SIZE_TYPE, VALUE_TYPE>(
        1, n_inputs,
        {0, 2},
        {1, 3},
        {3.0f, 1.0f}); // input 1 dominant, input 3 secondary

    // No initial connections
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> weights;
    weights.connections.rows = n_inputs;
    weights.connections.cols = n_outputs;
    weights.connections.ptrs[0] = std::make_shared<std::vector<SIZE_TYPE>>(
        std::vector<SIZE_TYPE>(n_inputs + 1, 0));
    weights.connections.indices[0]= std::make_shared<std::vector<SIZE_TYPE>>();
    weights.connections.values[0] = std::make_shared<std::vector<VALUE_TYPE>>();
    weights.connections.values[1] = std::make_shared<std::vector<VALUE_TYPE>>();
    weights.connections.values[2] = std::make_shared<std::vector<VALUE_TYPE>>();
    weights.probes.rows = n_inputs;
    weights.probes.cols = n_outputs;

    // With no connections, forward does nothing — all output = 0
    std::vector<VALUE_TYPE> output(n_outputs, 0.0f);
    sisldo_forward(input, weights, output.data(), false, 0.01f, num_cpus);
    CHECK_VECTOR_ALMOST_EQUAL(output, std::vector<VALUE_TYPE>(n_outputs, 0.0f));

    // Accumulate over multiple iterations
    // out_grad_sparse: output 0 (strong) and output 2 (weak)
    auto out_grad_sparse = make_csr_input<SIZE_TYPE, VALUE_TYPE>(
        1, n_outputs,
        {0, 2},
        {0, 2},
        {2.0f, 0.5f}); // output 0 dominant, output 2 secondary

    // output_gradients: [out0_b0, out1_b0, out2_b0]
    std::vector<VALUE_TYPE> output_gradients = {2.0f, 0.0f, 0.5f};

    std::vector<VALUE_TYPE> neuron_input_accum(n_inputs,  0.0f);
    std::vector<VALUE_TYPE> neuron_grad_accum (n_outputs, 0.0f);
    std::vector<VALUE_TYPE> input_gradients   (n_inputs,  0.0f);


    const int accum_iters = 5;
    for (int i = 0; i < accum_iters; ++i) {
        std::fill(input_gradients.begin(), input_gradients.end(), 0.0f);
        sisldo_backward(input, weights, out_grad_sparse,
                        input_gradients.data(), output_gradients.data(),
                        neuron_input_accum.data(), neuron_grad_accum.data(),
                        num_cpus);
    }

    // Verify accumulators reflect dominant neurons
    CHECK(neuron_input_accum[1] > neuron_input_accum[3]); // input 1 > input 3
    CHECK(neuron_input_accum[0] == 0.0f);                  // input 0 never active
    CHECK(neuron_input_accum[2] == 0.0f);                  // input 2 never active
    CHECK(neuron_grad_accum[0] > neuron_grad_accum[2]);   // output 0 > output 2
    CHECK(neuron_grad_accum[1] == 0.0f);                   // output 1 never seen

    // ── Genesis: build probes from accumulated activity ───────────────────────
    genesis_build_probes(weights,
                         neuron_input_accum.data(),
                         neuron_grad_accum.data(),
                         n_inputs, n_outputs, k, num_cpus);

    // k=2: top-2 inputs are [1,3], top-2 outputs are [0,2]
    // Outer product → 4 probes: (1,0),(1,2),(3,0),(3,2)
    // After merge_sort_coo: sorted by (row,col) with no duplicates
    REQUIRE(weights.probes.nnz() == (SIZE_TYPE)(k * k));

    // All probe rows should be from {1,3} and cols from {0,2}
    const auto& probe_rows = *weights.probes.indices[0];
    const auto& probe_cols = *weights.probes.indices[1];
    const auto& probe_vals = *weights.probes.values[0];

    for (SIZE_TYPE i = 0; i < weights.probes.nnz(); ++i) {
        INFO("probe[" << i << "]: (" << probe_rows[i] << "," << probe_cols[i]
             << ") = " << probe_vals[i]);
        CHECK((probe_rows[i] == 1 || probe_rows[i] == 3));
        CHECK((probe_cols[i] == 0 || probe_cols[i] == 2));
        CHECK(probe_vals[i] > 0.0f); // outer product of positive values
    }

    // Sorted: (1,0),(1,2),(3,0),(3,2)
    CHECK_VECTOR_EQUAL(probe_rows, std::vector<SIZE_TYPE>({1, 1, 3, 3}));
    CHECK_VECTOR_EQUAL(probe_cols, std::vector<SIZE_TYPE>({0, 2, 0, 2}));

    // ── Synaptogenesis: merge probes into connections ─────────────────────────
    const VALUE_TYPE lr            = 0.01f;
    const VALUE_TYPE importance_beta = 0.1f;
    const SIZE_TYPE  max_weights   = 4;

    sisldo_optim_synaptogenesis(weights, lr, importance_beta, max_weights, num_cpus);

    // Should now have exactly k*k = 4 connections
    REQUIRE(weights.connections.nnz() == k * k);

    // Connections should cover the same (row,col) pairs as the probes
    const auto& conn_ptrs    = *weights.connections.ptrs[0];
    const auto& conn_indices = *weights.connections.indices[0];
    const auto& conn_val0    = *weights.connections.values[0];
    const auto& conn_val2    = *weights.connections.values[2];

    // Rows 1 and 3 should each have 2 connections, rows 0 and 2 should have 0
    CHECK(conn_ptrs[0] == 0); // row 0: 0 connections
    CHECK(conn_ptrs[1] == 0); // row 0 end
    CHECK(conn_ptrs[2] == 2); // row 1: 2 connections (cols 0,2)
    CHECK(conn_ptrs[3] == 2); // row 2: 0 connections
    CHECK(conn_ptrs[4] == 4); // row 3: 2 connections (cols 0,2)

    // All new connections start with value=0
    CHECK_VECTOR_ALMOST_EQUAL(conn_val0, std::vector<VALUE_TYPE>(4, 0.0f));

    // Importance values should be negative (probe outer-product scaled by -beta/lr)
    // probe_val = input_accum * output_accum, then -(val/lr)*beta
    for (SIZE_TYPE i = 0; i < 4; ++i) {
        INFO("importance[" << i << "] = " << conn_val2[i]);
        CHECK(conn_val2[i] < 0.0f); // new connections start with negative importance
    }

    // Probes should be cleared after optim
    REQUIRE(weights.probes.nnz() == 0);

    // ── Verify network can now produce nonzero output ─────────────────────────
    // Set connection values to something nonzero so forward produces output
    (*weights.connections.values[0])[0] = 1.0f; // (1,0)
    (*weights.connections.values[0])[1] = 0.5f; // (1,2)
    (*weights.connections.values[0])[2] = 0.3f; // (3,0)
    (*weights.connections.values[0])[3] = 0.7f; // (3,2)

    std::vector<VALUE_TYPE> output2(n_outputs, 0.0f);
    sisldo_forward(input, weights, output2.data(), false, 0.01f, num_cpus);

    // batch0 (single batch): in1=3.0, in3=1.0
    // out0: in1*1.0*3.0 + in3*0.3*1.0 = 3.0 + 0.3 = 3.3
    // out1: 0
    // out2: in1*0.5*3.0 + in3*0.7*1.0 = 1.5 + 0.7 = 2.2
    CHECK_VECTOR_ALMOST_EQUAL(output2,
        std::vector<VALUE_TYPE>({3.3f, 0.0f, 2.2f}), 1e-5f);

    // ── Full loop: several more iterations now that connections exist ──────────
    std::fill(neuron_input_accum.begin(), neuron_input_accum.end(), 0.0f);
    std::fill(neuron_grad_accum.begin(),  neuron_grad_accum.end(),  0.0f);

    std::vector<VALUE_TYPE> desired = {1.0f, 0.0f, 1.0f};
    VALUE_TYPE prev_loss = std::numeric_limits<VALUE_TYPE>::max();

    for (int iter = 0; iter < 10; ++iter) {
        INFO("post-genesis iter: " << iter);
        std::vector<VALUE_TYPE> out3(n_outputs, 0.0f);
        sisldo_forward(input, weights, out3.data(), true, 0.01f, num_cpus);

        auto grad3 = mse_grad(out3, desired);
        auto og_sparse = make_csr_input<SIZE_TYPE, VALUE_TYPE>(
            1, n_outputs,
            {0, 2},
            {0, 2},
            {grad3[0], grad3[2]});

        std::vector<VALUE_TYPE> og_dense = {grad3[0], 0.0f, grad3[2]};
        std::vector<VALUE_TYPE> in_grad3(n_inputs, 0.0f);
        sisldo_backward(input, weights, og_sparse,
                        in_grad3.data(), og_dense.data(),
                        neuron_input_accum.data(), neuron_grad_accum.data(),
                        num_cpus);
        sisldo_optim_weights(weights, 0.01f, num_cpus);

        VALUE_TYPE loss = mse_loss(out3, desired);
        if (iter > 0) {
            INFO("prev=" << prev_loss << " curr=" << loss);
            CHECK(loss <= prev_loss * 1.05f); // loss should not diverge
        }
        prev_loss = loss;
    }

    // Loss should be lower than at start (3.3 vs desired 1.0 → initial loss = ~(2.3^2+2.2^2)/3)
    CHECK(prev_loss < mse_loss(std::vector<VALUE_TYPE>({3.3f, 0.0f, 2.2f}), desired));
}