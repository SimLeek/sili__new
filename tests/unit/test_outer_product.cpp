#include "tests_main.h"
/*
#include "outer_product.cpp"

TEST_CASE("Outer Product Function Tests") {
    SECTION("Square Matrices Test") {
        int batches = 1;
        int a_size = 5;
        int b_size = 5;
        csr_struct a;
        csr_struct b;

        // Initialize A and B with some values
        a.nnz = 5;
        a.values = new float[a.nnz]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        a.indices = new int[a.nnz]{0, 1, 2, 3, 4};

        b.nnz = 5;
        b.values = new float[b.nnz]{6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        b.indices = new int[b.nnz]{0, 1, 2, 3, 4};

        // Perform outer product
        std::vector<csr_struct> result_batches = outer_product(batches, a_size, b_size, a, b);

        // Check if results match expectations
        REQUIRE_MESSAGE(result_batches.size() == batches, "Number of result batches mismatched");
        REQUIRE_MESSAGE(result_batches.front().nnz == a.nnz*b.nnz, "Incorrect number of non-zero elements");
    }

    SECTION("Rectangular Matrices Test") {
        int batches = 1;
        int a_size = 3;
        int b_size = 4;
        csr_struct a;
        csr_struct b;

        // Initialize A and B with some values
        a.nnz = 3;
        a.values = new float[a.nnz]{1.0f, 2.0f, 3.0f};
        a.indices = new int[a.nnz]{0, 1, 2};

        b.nnz = 4;
        b.values = new float[b.nnz]{6.0f, 7.0f, 8.0f, 9.0f};
        b.indices = new int[b.nnz]{0, 1, 2, 3};

        // Perform outer product
        std::vector<csr_struct> result_batches = outer_product(batches, a_size, b_size, a, b);

        // Check if results match expectations
        REQUIRE_MESSAGE(result_batches.size() == batches, "Number of result batches mismatched");
        REQUIRE_MESSAGE(result_batches.front().nnz == a.nnz*b.nnz, "Incorrect number of non-zero elements");
    }

    SECTION("Multi-Batched Operations Test") {
        int batches = 2;
        int a_size = 3;
        int b_size = 4;
        csr_struct a;
        csr_struct b;

        // Initialize A and B with some values
        a.nnz = 6;
        a.values = new float[a.nnz]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        a.indices = new int[a.nnz]{0, 1, 2, 0, 1, 2};
        a.ptrs = new int[batches + 1]{0, 3, 6}; // Define ptrs for A

        b.nnz = 8;
        b.values = new float[b.nnz]{6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f};
        b.indices = new int[b.nnz]{0, 1, 2, 3, 0, 1, 2, 3};
        b.ptrs = new int[batches + 1]{0, 4, 8}; // Define ptrs for B

        // Perform outer product
        std::vector<csr_struct> result_batches = outer_product(batches, a_size, b_size, a, b);

        // Check if results match expectations
        REQUIRE_MESSAGE(result_batches.size() == batches, "Number of result batches mismatched");
        REQUIRE_MESSAGE(result_batches.front().nnz == a.nnz*b.nnz/2, "Incorrect number of non-zero elements");
    }
}

TEST_CASE("Outer Product Mask Function Tests") {
    SECTION("Square Matrices Test") {
        int batches = 1;
        int a_size = 5;
        int b_size = 5;
        csr_struct a;
        csr_struct b;

        // Initialize A and B with some values
        a.nnz = 5;
        a.indices = new int[a.nnz]{0, 1, 2, 3, 4};

        b.nnz = 5;
        b.indices = new int[b.nnz]{0, 1, 2, 3, 4};

        // Perform outer product
        std::vector<csr_struct> result_batches = build_outer_product_mask(batches, a_size, b_size, a, b);

        // Check if results match expectations
        REQUIRE_MESSAGE(result_batches.size() == batches, "Number of result batches mismatched");
        REQUIRE_MESSAGE(result_batches.front().nnz == a.nnz*b.nnz, "Incorrect number of non-zero elements");
    }

    SECTION("Rectangular Matrices Test") {
        int batches = 1;
        int a_size = 3;
        int b_size = 4;
        csr_struct a;
        csr_struct b;

        // Initialize A and B with some values
        a.nnz = 3;
        a.indices = new int[a.nnz]{0, 1, 2};

        b.nnz = 4;
        b.indices = new int[b.nnz]{0, 1, 2, 3};

        // Perform outer product
        std::vector<csr_struct> result_batches = build_outer_product_mask(batches, a_size, b_size, a, b);

        // Check if results match expectations
        REQUIRE_MESSAGE(result_batches.size() == batches, "Number of result batches mismatched");
        REQUIRE_MESSAGE(result_batches.front().nnz == a.nnz*b.nnz, "Incorrect number of non-zero elements");
    }

    SECTION("Multi-Batched Operations Test") {
        int batches = 2;
        int a_size = 3;
        int b_size = 4;
        csr_struct a;
        csr_struct b;

        // Initialize A and B with some values
        a.nnz = 6;
        a.indices = new int[a.nnz]{0, 1, 2, 0, 1, 2};
        a.ptrs = new int[batches + 1]{0, 3, 6}; // Define ptrs for A

        b.nnz = 8;
        b.indices = new int[b.nnz]{0, 1, 2, 3, 0, 1, 2, 3};
        b.ptrs = new int[batches + 1]{0, 4, 8}; // Define ptrs for B

        // Perform outer product
        std::vector<csr_struct> result_batches = build_outer_product_mask(batches, a_size, b_size, a, b);

        // Check if results match expectations
        REQUIRE_MESSAGE(result_batches.size() == batches, "Number of result batches mismatched");
        REQUIRE_MESSAGE(result_batches.front().nnz == a.nnz*b.nnz/2, "Incorrect number of non-zero elements");
    }
}

TEST_CASE("Backpropagate To B Function Tests") {
    SECTION("Single Batch Test") {
        int batches = 1;
        int a_size = 3;
        int b_size = 4;
        csr_struct a;
        csr_struct b_grad;
        std::vector<csr_struct> o_grad_b;

        // Initialize A and B_Grad with some values
        a.nnz = 3;
        a.values = new float[a.nnz]{1.0f, 2.0f, 3.0f};
        a.indices = new int[a.nnz]{0, 1, 2};
        a.ptrs = new int[batches + 1]{0, 3};

        b_grad.nnz = 4;
        b_grad.values = new float[b_grad.nnz]{6.0f, 7.0f, 8.0f, 9.0f};
        b_grad.indices = new int[b_grad.nnz]{0, 1, 2, 3};
        b_grad.ptrs = new int[batches + 1]{0, 4};

        o_grad_b.resize(batches);
        o_grad_b.back().nnz = 4;
        o_grad_b.back().values = new float[o_grad_b.back().nnz]{10.0f, 11.0f, 12.0f, 13.0f};
        o_grad_b.back().indices = new int[o_grad_b.back().nnz]{0, 1, 2, 3};
        o_grad_b.back().ptrs = new int[batches + 1]{0, 4};

        // Perform backpropagation
        outer_product_backwards_b(batches, a_size, b_size, a, b_grad, o_grad_b);

        // Check if results match expectations
        REQUIRE_MESSAGE(b_grad.values[0] != 6.0f, "Values did not change after backpropagation");

        // For adapting this test to outer_product_backwards_a:
        // Replace a with a_grad, b_grad with b, and o_grad_b with o_grad_a.
        // Also swap the roles of a and b in the initialization section.
    }

    SECTION("Multiple Batches Test") {
        int batches = 2;
        int a_size = 3;
        int b_size = 4;
        csr_struct a;
        csr_struct b_grad;
        std::vector<csr_struct> o_grad_b;

        // Initialize A and B_Grad with some values
        a.nnz = 6;
        a.values = new float[a.nnz]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        a.indices = new int[a.nnz]{0, 1, 2, 0, 1, 2};
        a.ptrs = new int[batches + 1]{0, 3, 6};

        b_grad.nnz = 8;
        b_grad.values = new float[b_grad.nnz]{6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f};
        b_grad.indices = new int[b_grad.nnz]{0, 1, 2, 3, 0, 1, 2, 3};
        b_grad.ptrs = new int[batches + 1]{0, 4, 8};

        o_grad_b.resize(batches);
        o_grad_b.back().nnz = 4;
        o_grad_b.back().values = new float[o_grad_b.back().nnz]{14.0f, 15.0f, 16.0f, 17.0f};
        o_grad_b.back().indices = new int[o_grad_b.back().nnz]{0, 1, 2, 3};
        o_grad_b.back().ptrs = new int[batches + 1]{0, 4};

        // Perform backpropagation
        outer_product_backwards_b(batches, a_size, b_size, a, b_grad, o_grad_b);

        // Check if results match expectations
        REQUIRE_MESSAGE(b_grad.values[0] != 6.0f, "Values did not change after backpropagation");

        // For adapting this test to outer_product_backwards_a:
        // Replace a with a_grad, b_grad with b, and o_grad_b with o_grad_a.
        // Also swap the roles of a and b in the initialization section.
    }

    SECTION("Edge Case Test") {
        int batches = 1;
        int a_size = 1;
        int b_size = 1;
        csr_struct a;
        csr_struct b_grad;
        std::vector<csr_struct> o_grad_b;

        // Initialize A and B_Grad with some values
        a.nnz = 1;
        a.values = new float[a.nnz]{1.0f};
        a.indices = new int[a.nnz]{0};
        a.ptrs = new int[batches + 1]{0, 1};

        b_grad.nnz = 1;
        b_grad.values = new float[b_grad.nnz]{6.0f};
        b_grad.indices = new int[b_grad.nnz]{0};
        b_grad.ptrs = new int[batches + 1]{0, 1};

        o_grad_b.resize(batches);
        o_grad_b.back().nnz = 1;
        o_grad_b.back().values = new float[o_grad_b.back().nnz]{18.0f};
        o_grad_b.back().indices = new int[o_grad_b.back().nnz]{0};
        o_grad_b.back().ptrs = new int[batches + 1]{0, 1};

        // Perform backpropagation
        outer_product_backwards_b(batches, a_size, b_size, a, b_grad, o_grad_b);

        // Check if results match expectations
        REQUIRE_MESSAGE(b_grad.values[0] != 6.0f, "Values did not change after backpropagation");

        // For adapting this test to outer_product_backwards_a:
        // Replace a with a_grad, b_grad with b, and o_grad_b with o_grad_a.
        // Also swap the roles of a and b in the initialization section.
    }
}



TEST_CASE("Backpropagate To A Function Tests") {
    SECTION("Single Batch Test") {
        REQUIRE_MESSAGE(false, "Get backprop to B tests working before adapting them to backprop to A");
    }

    SECTION("Multiple Batches Test") {
        REQUIRE_MESSAGE(false, "Get backprop to B tests working before adapting them to backprop to A");
    }

    SECTION("Edge Case Test") {
        REQUIRE_MESSAGE(false, "Get backprop to B tests working before adapting them to backprop to A");
    }
}
*/