#include "tests_main.h"

#include "conv2d.cpp"
/*
TEST_CASE("_Setup Conv2D Row Function Tests") {
    SECTION("Basic Setup Test") {
        int input_height = 28;
        int batch = 0;
        int oi = 56;
        int oiy;
        int tid = 0;
        int kh_diff_n = 0;
        std::vector<int> start_row{0};
        std::vector<std::vector<int>> output_col_indices;
        std::vector<std::vector<std::vector<int>>> output_channel_indices;
        std::vector<std::vector<std::vector<float>>> output_values;
        std::vector<int> vip_col;
        std::vector<csf_struct> input_sparse_images;

        // Create a sample csf_struct instance using convert_vovov_to_csf
        std::vector<std::vector<int>> col_indices{{0}, {1}};
        std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}}, {{1}}};
        std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}}, {{2.0f}}};
        csf_struct image = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 2, 2, 2, 2, 2);

        input_sparse_images.push_back(image);

        // Call the _setup_sparse_io_conv2d_row function
        _setup_sparse_io_conv2d_row(input_height, batch, oi, oiy, tid, kh_diff_n, start_row, output_col_indices,
                                    output_channel_indices, output_values, vip_col, input_sparse_images);

        // Verify that oiy has been calculated correctly
        REQUIRE_MESSAGE(oiy == 2, "oyi calculation incorrect");

        // Verify that other parameters have been initialized correctly
        REQUIRE_MESSAGE(!start_row.empty(), "Start row not initialized");
        REQUIRE_MESSAGE(output_col_indices.size() > 0, "Output column indices not initialized");
        REQUIRE_MESSAGE(output_channel_indices.size() > 0, "Output channel indices not initialized");
        REQUIRE_MESSAGE(output_values.size() > 0, "Output values not initialized");
        REQUIRE_MESSAGE(vip_col.size() > 0, "VIP column not initialized");
    }
}

TEST_CASE("_Do Convolution Function Tests") {
    SECTION("Basic Convolution Test") {
        int batch = 0;
        int output_channels = 1;
        int input_width = 28;
        int input_channels = 1;
        int oix = 14;
        int kw_diff_n = 0;
        int kw_diff_p = 0;
        float W[] = {1.0f};
        bool made_this_fiber = false;
        std::vector<int> vip_col{0};
        std::vector<csf_struct> input_sparse_images;
        std::vector<std::vector<int>> output_col_indices;
        std::vector<std::vector<std::vector<int>>> output_channel_indices;
        std::vector<std::vector<std::vector<float>>> output_values;
        float eps = 0.01f;

        // Create a sample csf_struct instance using convert_vovov_to_csf
        std::vector<std::vector<int>> col_indices{{14}};
        std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}}};
        std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}}};
        csf_struct image = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 1, 1);

        input_sparse_images.push_back(image);

        // Call the _do_sparse_io_conv2d_convolution function
    _do_sparse_io_conv2d_convolution(batch, output_channels, input_width, input_channels, oix, kw_diff_n, kw_diff_p, W,
                                        made_this_fiber, vip_col, input_sparse_images, output_col_indices,
                                        output_channel_indices, output_values, eps);

        // Verify that the output values were computed correctly
        REQUIRE_MESSAGE(made_this_fiber, "Made this fiber flag not set");
        REQUIRE_MESSAGE(output_values.size() > 0, "Output values not generated");
        REQUIRE_MESSAGE(output_values.back().size() > 0, "Inner output values not generated");
        REQUIRE_MESSAGE(output_values.back()[0][0] > 0, "Convolution value not correct");
    }

    SECTION("No Input Data Within Kernel Area Test") {
        int batch = 0;
        int output_channels = 1;
        int input_width = 28;
        int input_channels = 1;
        int oix = 26;
        int kw_diff_n = 0;
        int kw_diff_p = 0;
        float W[] = {1.0f};
        bool made_this_fiber = false;
        std::vector<int> vip_col{0};
        std::vector<csf_struct> input_sparse_images;
        std::vector<std::vector<int>> output_col_indices;
        std::vector<std::vector<std::vector<int>>> output_channel_indices;
        std::vector<std::vector<std::vector<float>>> output_values;
        float eps = 0.01f;

        // Create a sample csf_struct instance using convert_vovov_to_csf
        std::vector<std::vector<int>> col_indices{{25}};
        std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}}};
        std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}}};
        csf_struct image = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 1, 1);

        input_sparse_images.push_back(image);

        // Call the_do_sparse_io_conv2d_convolution function
    _do_sparse_io_conv2d_convolution(batch, output_channels, input_width, input_channels, oix, kw_diff_n, kw_diff_p, W,
                                        made_this_fiber, vip_col, input_sparse_images, output_col_indices,
                                        output_channel_indices, output_values, eps);

        // Verify that no output values were generated
        REQUIRE_MESSAGE(!made_this_fiber, "Made this fiber flag incorrectly set");
        REQUIRE_MESSAGE(output_values.empty(), "Output values unexpectedly generated");
    }

    SECTION("Multiple Inputs Within Kernel Area Test") {
        int batch = 0;
        int output_channels = 1;
        int input_width = 28;
        int input_channels = 1;
        int oix = 14; // changed from 20 to 14
        int kw_diff_n = 0;
        int kw_diff_p = 0;
        float W[] = {1.0f};
        bool made_this_fiber = false;
        std::vector<int> vip_col{0}; // changed from {0} to {0}
        std::vector<csf_struct> input_sparse_images;
        std::vector<std::vector<int>> output_col_indices;
        std::vector<std::vector<std::vector<int>>> output_channel_indices;
        std::vector<std::vector<std::vector<float>>> output_values;
        float eps = 0.01f;

        // Create a sample csf_struct instance using convert_vovov_to_csf
        std::vector<std::vector<int>> col_indices{{14},{19}};
        std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}},{{0}}};
        std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}},{{1.0f}}};
        csf_struct image = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 2, 2);

        input_sparse_images.push_back(image);

        // Call the_do_sparse_io_conv2d_convolution function
    _do_sparse_io_conv2d_convolution(batch, output_channels, input_width, input_channels, oix, kw_diff_n, kw_diff_p, W,
                                        made_this_fiber, vip_col, input_sparse_images, output_col_indices,
                                        output_channel_indices, output_values, eps);

        // Verify that multiple output values were generated
        REQUIRE_MESSAGE(made_this_fiber, "Made this fiber flag not set");
        REQUIRE_MESSAGE(output_values.size() > 0, "Output values not generated");
        REQUIRE_MESSAGE(output_values.back().size() > 0, "Inner output values not generated");
        REQUIRE_MESSAGE((output_values.back()[0][0] > 0 && output_values.back()[0][1] > 0),
                        "Convolution values not correct");
    }

}

TEST_CASE("_Move Column Pointers Back Function Tests") {
    SECTION("Basic Pointer Movement Test") {
        int batch = 0;
        int kw_diff_n = 0;
        int oix = 14;
        std::vector<csf_struct> input_sparse_images;
        std::vector<int> vip_col{0};

        // Create a sample csf_struct instance using convert_vovov_to_csf
        std::vector<std::vector<int>> col_indices{{14}};
        std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}}};
        std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}}};
        csf_struct image = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 1, 1);

        input_sparse_images.push_back(image);

        // Call the _move_sparse_io_conv2d_column_ptrs_back function
        _move_sparse_io_conv2d_column_ptrs_back(batch, kw_diff_n, oix, input_sparse_images, vip_col);

        // Verify that the vip_col was moved back correctly
        REQUIRE_MESSAGE(vip_col[0] == -1, "Vip_col not moved back");
    }

    SECTION("No Pointer Movement Required Test") {
        int batch = 0;
        int kw_diff_n = 0;
        int oix = 14;
        std::vector<csf_struct> input_sparse_images;
        std::vector<int> vip_col{-1};

        // Create a sample csf_struct instance using convert_vovov_to_csf
        std::vector<std::vector<int>> col_indices{{14}};
        std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}}};
        std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}}};
        csf_struct image = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 1, 1);

        input_sparse_images.push_back(image);

        // Call the _move_sparse_io_conv2d_column_ptrs_back function
        _move_sparse_io_conv2d_column_ptrs_back(batch, kw_diff_n, oix, input_sparse_images, vip_col);

        // Verify that the vip_col remained unchanged
        REQUIRE_MESSAGE(vip_col[0] == -1, "Vip_col unnecessarily moved");
    }

    SECTION("Multi-Level Pointer Movement Test") {
        int batch = 0;
        int kw_diff_n = 0;
        int oix = 14;
        std::vector<csf_struct> input_sparse_images;
        std::vector<int> vip_col{2};

        // Create a sample csf_struct instance using convert_vovov_to_csf
        std::vector<std::vector<int>> col_indices{{14}, {15}};
        std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}}, {{0}}};
        std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}}, {{1.0f}}};
        csf_struct image = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 2, 2);

        input_sparse_images.push_back(image);

        // Call the _move_sparse_io_conv2d_column_ptrs_back function
        _move_sparse_io_conv2d_column_ptrs_back(batch, kw_diff_n, oix, input_sparse_images, vip_col);

        // Verify that the vip_col was moved back correctly
        REQUIRE_MESSAGE(vip_col[0] == 0, "Vip_col not moved back far enough");
    }
}

TEST_CASE("_Remove Zero Outputs Function Tests") {
    SECTION("Zero Output Removal Test") {
        std::vector<std::vector<int>> output_col_indices{{0}};
        std::vector<std::vector<std::vector<int>>> output_channel_indices{{{0}}};
        std::vector<std::vector<std::vector<float>>> output_values{{{0.5f}}};
        float eps = 0.6f;

        // Call the_conv2d_remove_zero_outputs function
    _conv2d_remove_zero_outputs(output_col_indices, output_channel_indices, output_values, eps);

        // Verify that the zero-valued output channel was removed
        REQUIRE_MESSAGE(output_channel_indices.back().empty(), "Zero-valued output channel not removed");
        REQUIRE_MESSAGE(output_values.back().empty(), "Zero-valued output channel not removed");
    }

    SECTION("Positive And Negative Contributions Test") {
        std::vector<std::vector<int>> output_col_indices{{0}};
        std::vector<std::vector<std::vector<int>>> output_channel_indices{{{0}}};
        std::vector<std::vector<std::vector<float>>> output_values{{{1.0f}, {-1.0f}}};
        float eps = 0.001f;

        // Call the _conv2d_remove_zero_outputs function
    _conv2d_remove_zero_outputs(output_col_indices, output_channel_indices, output_values, eps);

        // Verify that the net result is zero and the output channel was removed
        REQUIRE_MESSAGE(output_channel_indices.back().empty(), "Net-zero output channel not removed");
        REQUIRE_MESSAGE(output_values.back().empty(), "Net-zero output channel not removed");
    }

    SECTION("Non-Zero Valued Channels Remain Test") {
        std::vector<std::vector<int>> output_col_indices{{0}};
        std::vector<std::vector<std::vector<int>>> output_channel_indices{{{0}}};
        std::vector<std::vector<std::vector<float>>> output_values{{{1.0f}, {2.0f}}};
        float eps = 0.9f;

        // Call the_conv2d_remove_zero_outputs function
    _conv2d_remove_zero_outputs(output_col_indices, output_channel_indices, output_values, eps);

        // Verify that only the non-zero valued output channels remain
        REQUIRE_MESSAGE(output_channel_indices.back().size() == 1, "Incorrectly removed non-zero valued output channel");
        REQUIRE_MESSAGE(output_values.back().size() == 1, "Incorrectly removed non-zero valued output channel");
    }
}

TEST_CASE("_Sparse IO Conv2D Reserve Function Test") {
    int nnf = 0;
    int nnc = 0;
    std::vector<std::vector<size_t>> vec_channel_assign_locs{{3}};
    std::vector<std::vector<size_t>> vec_col_assign_locs{{3}};
    std::vector<int> start_row{0};
    std::vector<std::vector<int>> out_col_idx{std::vector<int>(1)};
    std::vector<std::vector<std::vector<int>>> out_chan_idx{std::vector<std::vector<int>>(1)};
    std::vector<std::vector<std::vector<float>>> out_val{std::vector<std::vector<float>>(1)};

    // Call the _sparse_io_conv2d_reserve function
    _sparse_io_conv2d_reserve(nnf, nnc, vec_channel_assign_locs, vec_col_assign_locs, start_row, out_col_idx,
                             out_chan_idx, out_val);

    // Verify that the memory reservation was successful
    REQUIRE_MESSAGE(nnf == 3, "NNF count incorrect");
    REQUIRE_MESSAGE(nnc == 3, "NCC count incorrect");
    REQUIRE_MESSAGE(out_col_idx[0].size() == 3, "Out col idx size incorrect");
    REQUIRE_MESSAGE(out_chan_idx[0].size() == 3, "Out chan idx size incorrect");
    REQUIRE_MESSAGE(out_val[0].size() == 3, "Out val size incorrect");
}

TEST_CASE("_Copy To Output VoVoV Function Test") {
    int num_cpus = 2;
    std::vector<std::vector<size_t>> vec_channel_assign_locs{{3}, {3}};
    std::vector<std::vector<size_t>> vec_col_assign_locs{{3}, {3}};
    std::vector<int> start_row{0, 3};
    std::vector<std::vector<std::vector<int>>> output_col_indices_chunks{
        {{1}, {2}},
        {{3}, {4}}
    };
    std::vector<std::vector<std::vector<std::vector<int>>>> output_channel_indices_chunks{
        {{{0}}, {{1}}},
        {{{2}}, {{3}}}
    };
    std::vector<std::vector<std::vector<std::vector<float>>>> output_values_chunks{
        {{{1.0f}}, {{2.0f}}},
        {{{3.0f}}, {{4.0f}}}
    };
    std::vector<std::vector<int>> out_col_idx{std::vector<int>(6), std::vector<int>(6)};
    std::vector<std::vector<std::vector<int>>> out_chan_idx{std::vector<std::vector<int>>(6),
                                                           std::vector<std::vector<int>>(6)};
    std::vector<std::vector<std::vector<float>>> out_val{std::vector<std::vector<float>>(6),
                                                        std::vector<std::vector<float>>(6)};

    // Call the _sparse_io_conv2d_copy_to_output_vovov function
    _sparse_io_conv2d_copy_to_output_vovov(num_cpus, vec_channel_assign_locs, vec_col_assign_locs, start_row,
                                         output_col_indices_chunks, output_channel_indices_chunks,
                                         output_values_chunks, out_col_idx, out_chan_idx, out_val);

    // Verify that the output voiov structures contain the copied values
    REQUIRE_MESSAGE(out_col_idx[0].size() == 6, "Copied values missing");
    REQUIRE_MESSAGE(out_chan_idx[0].size() == 6, "Copied values missing");
    REQUIRE_MESSAGE(out_val[0].size() == 6, "Copied values missing");
}

TEST_CASE("Conv2D Function Tests") {
SECTION("Basic Conv2D Test") {
    int batch_size = 1;
    int input_channels = 1;
    int output_channels = 1;
    int input_width = 10;
    int input_height = 10;
    int kernel_width = 3;
    int kernel_height = 3;
    std::vector<csf_struct> input_sparse_images;
    float W[] = {1.0f};
    float eps = 0.001f;

    // Create a sample csf_struct instance using convert_vovov_to_csf
    std::vector<std::vector<int>> col_indices{{5}};
    std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}}};
    std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}}};
    csf_struct image = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 1, 1);

    input_sparse_images.push_back(image);

    // Call the conv2d function
    std::vector<csf_struct> output_sparse_images =
        conv2d(batch_size, input_channels, output_channels, input_width, input_height, kernel_width, kernel_height,
               input_sparse_images, W, eps);

    // Verify that the output sparse images were processed successfully
    REQUIRE_MESSAGE(output_sparse_images.size() == 1, "Number of output sparse images incorrect");
    REQUIRE_MESSAGE(output_sparse_images[0].rows == input_height, "Height of output sparse image incorrect");
    REQUIRE_MESSAGE(output_sparse_images[0].cols == input_width, "Width of output sparse image incorrect");
}

SECTION("Large Input Conv2D Test") {
    int batch_size = 100;
    int input_channels = 32;
    int output_channels = 64;
    int input_width = 256;
    int input_height = 256;
    int kernel_width = 7;
    int kernel_height = 7;
    std::vector<csf_struct> input_sparse_images;
    float W[] = {1.0f};
    float eps = 0.001f;

    // Create a sample csf_struct instance using convert_vovov_to_csf
    std::vector<std::vector<int>> col_indices{{128}};
    std::vector<std::vector<std::vector<int>>> fiber_indices{{{0}}};
    std::vector<std::vector<std::vector<float>>> fiber_values{{{1.0f}}};
    csf_struct image = convert_vovov_to_csf(&col_indices, &fiber_indices, &fiber_values, 1, 1, 1, 1, 1);

    input_sparse_images.push_back(image);

    // Call the conv2d function
    std::vector<csf_struct> output_sparse_images =
        conv2d(batch_size, input_channels, output_channels, input_width, input_height, kernel_width, kernel_height,
               input_sparse_images, W, eps);

    // Verify that the output sparse images were processed successfully
    REQUIRE_MESSAGE(output_sparse_images.size() == 1, "Number of output sparse images incorrect");
    REQUIRE_MESSAGE(output_sparse_images[0].rows == input_height, "Height of output sparse image incorrect");
    REQUIRE_MESSAGE(output_sparse_images[0].cols == input_width, "Width of output sparse image incorrect");
}

SECTION("Invalid Input Conv2D Test") {
    int batch_size = -1;
    int input_channels = 0;
    int output_channels = 0;
    int input_width = 0;
    int input_height = 0;
    int kernel_width = 0;
    int kernel_height = 0;
    std::vector<csf_struct> input_sparse_images;
    float W[] = {};
    float eps = 0.001f;

    // Call the conv2d function
    std::vector<csf_struct> output_sparse_images =
        conv2d(batch_size, input_channels, output_channels, input_width, input_height, kernel_width, kernel_height,
               input_sparse_images, W, eps);

    // Verify that the output sparse images were processed successfully despite invalid inputs
    REQUIRE_MESSAGE(output_sparse_images.size() == 0, "Number of output sparse images incorrect");
}
}

TEST_CASE("Conv2D backwards tests") {
    REQUIRE_MESSAGE(false, "get the conv2d forward tests working, then translate them for backwards input, backwards W (also from linear func), and mask");
}*/