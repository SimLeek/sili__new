#include "unique_vector.hpp"
#include "tests_main.h"
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "linear.cpp"
#include "quantized_arrays.cpp"
/*
TEST_CASE("_do_linear_sidlso_fwd basic functionality", "[do_linear_sidlso_fwd]") {
    int num_cpus = 4;  //this test assumes you at least have 4 cores
    int input_size = 10;
    int output_size = 20;
    int batch = 1;
    csr_struct<int, float> input_csr;
    input_csr.ptrs.reset(new int[5]{0, 3, 6, 9, 12}); // Example pointers
    input_csr.indices.reset(new int[12]{0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2}); // Example indices
    input_csr.values.reset(new float[12]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}); // Example values
    input_csr.rows = 4;
    input_csr.cols = 10;

    float* W = new float[input_size * output_size]{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
        130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
        140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
        150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
        170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
        180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
        190, 191, 192, 193, 194, 195, 196, 197, 198, 199
    };

    sili::unique_vector<sili::unique_vector<int>> row_indices_chunks(num_cpus);
    sili::unique_vector<sili::unique_vector<float>> row_values_chunks(num_cpus);

    float eps = 0.01f;

    _do_linear_sidlso_fwd(num_cpus, input_size, output_size, batch, input_csr, W, row_indices_chunks, row_values_chunks, eps);

    sili::unique_vector<sili::unique_vector<int>> expected_row_indices_chunks{
        {0,1,2,3, 4},
        {5,6,7, 8, 9},
        {10,11, 12, 13, 14}, 
        {15, 16, 17, 18, 19}
    };
    sili::unique_vector<sili::unique_vector<float>> expected_row_values_chunks{
        {62,212,362,512, 662},
        {812,962,1112, 1262, 1412},
        {1562,1712, 1862,2012, 2162}, 
        { 2312, 2462, 2612, 2762, 2912}
    };

    // Verify results
    for (const auto& vec : row_indices_chunks) {
        REQUIRE(vec.size() == 5);
    }

    for (size_t i = 0; i < row_values_chunks.size(); i++) {
        for (size_t j = 0; j < row_values_chunks[i].size(); j++) {
            CHECK_MESSAGE(row_indices_chunks[i][j]==expected_row_indices_chunks[i][j], 
            "i,j:["<<i<<","<<j<<"] int: actual("<<row_indices_chunks[i][j]<<")!=expected("<< expected_row_indices_chunks[i][j] <<")\n");
            //std::cout<<"i,j:["<<i<<","<<j<<"] val: "<<row_values_chunks[i][j]<<"\n";
            CHECK_MESSAGE(row_values_chunks[i][j]==expected_row_values_chunks[i][j], 
            "i,j:["<<i<<","<<j<<"] int: actual("<<row_values_chunks[i][j]<<")!=expected("<< expected_row_values_chunks[i][j] <<")\n");
        }
    }

    delete[] W;
}

TEST_CASE("_do_linear_sidlso_fwd zero eps value", "[do_linear_sidlso_fwd]")
{
    int num_cpus = 4;  //this test assumes you at least have 4 cores
    int input_size = 10;
    int output_size = 20;
    int batch = 1;
    csr_struct<int, float> input_csr;
    input_csr.ptrs.reset(new int[5]{0, 3, 6, 9, 12}); // Example pointers
    input_csr.indices.reset(new int[12]{0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2}); // Example indices
    input_csr.values.reset(new float[12]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}); // Example values
    input_csr.rows = 4;
    input_csr.cols = 10;

    float* W = new float[input_size * output_size]{
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
       1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 1.};

    sili::unique_vector<sili::unique_vector<int>> row_indices_chunks(num_cpus);
    sili::unique_vector<sili::unique_vector<float>> row_values_chunks(num_cpus);

    float eps = 0.01f;

    _do_linear_sidlso_fwd(num_cpus, input_size, output_size, batch, input_csr, W, row_indices_chunks, row_values_chunks, eps);

    sili::unique_vector<sili::unique_vector<int>> expected_row_indices_chunks{
        {3, 4},
        {5},
        {13,14}, 
        {15}
    };
    sili::unique_vector<sili::unique_vector<float>> expected_row_values_chunks{
        {4,5},
        {6},
        {4,5}, 
        { 6}
    };

    // Verify results
    for (int i=0;i<num_cpus;i++) {
        REQUIRE_MESSAGE(row_indices_chunks[i].size() == expected_row_indices_chunks[i].size(), "indices size mismatch at "<< i);
        REQUIRE_MESSAGE(row_values_chunks[i].size() == expected_row_values_chunks[i].size(), "values size mismatch at "<< i);
    }

    for (size_t i = 0; i < row_values_chunks.size(); i++) {
        for (size_t j = 0; j < row_values_chunks[i].size(); j++) {
            CHECK_MESSAGE(row_indices_chunks[i][j]==expected_row_indices_chunks[i][j], 
            "i,j:["<<i<<","<<j<<"] int: actual("<<row_indices_chunks[i][j]<<")!=expected("<< expected_row_indices_chunks[i][j] <<")\n");
            CHECK_MESSAGE(row_values_chunks[i][j]==expected_row_values_chunks[i][j], 
            "i,j:["<<i<<","<<j<<"] int: actual("<<row_values_chunks[i][j]<<")!=expected("<< expected_row_values_chunks[i][j] <<")\n");
        }
    }

    delete[] W;
}

TEST_CASE("_do_linear_sidlso_fwd negative input values", "[do_linear_sidlso_fwd]")
{
    int num_cpus = 4;
    int input_size = 10;
    int output_size = 20;
    int batch = 1;
    csr_struct<int, float> input_csr;
    input_csr.ptrs.reset(new int[5]{0, 3, 6, 9, 12}); // Example pointers
    input_csr.indices.reset(new int[12]{0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2}); // Example indices
    input_csr.values.reset(new float[12]{-1.0f, -2.0f, -3.0f, 4.0f, -5.0f, -6.0f, -7.0f, 8.0f, -9.0f, -10.0f, -11.0f, 12.0f}); // Example values
    input_csr.rows = 4;
    input_csr.cols = 10;
    float* W = new float[input_size * output_size]{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
        130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
        140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
        150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
        170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
        180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
        190, 191, 192, 193, 194, 195, 196, 197, 198, 199
    };

    sili::unique_vector<sili::unique_vector<int>> row_indices_chunks(num_cpus);
    sili::unique_vector<sili::unique_vector<float>> row_values_chunks(num_cpus);

    float eps = 0.01f;

    _do_linear_sidlso_fwd(num_cpus,input_size,output_size,batch,input_csr,W,row_indices_chunks,row_values_chunks,eps);

    sili::unique_vector<sili::unique_vector<int>> expected_row_indices_chunks{
        {0, 1, 2, 3, 4},
        {5, 6,7, 8, 9},
        {10, 11, 12, 13,14}, 
        {15, 16, 17, 18, 19}
    };
    sili::unique_vector<sili::unique_vector<float>> expected_row_values_chunks{
        {-38,-108, -178, -248, -318},
        {-388, -458, -528, -598, -668},
        {-738,-808, -878, -948, -1018}, 
        { -1088, -1158, -1228, -1298, -1368}
    };

    // Verify sizes to avoid segfault
    for (int i=0;i<num_cpus;i++) {
        REQUIRE_MESSAGE(row_indices_chunks[i].size() == expected_row_indices_chunks[i].size(), "indices size mismatch at "<< i);
        REQUIRE_MESSAGE(row_values_chunks[i].size() == expected_row_values_chunks[i].size(), "values size mismatch at "<< i);
    }

    for (size_t i = 0; i < row_values_chunks.size(); i++) {
        for (size_t j = 0; j < row_values_chunks[i].size(); j++) {
            CHECK_MESSAGE(row_indices_chunks[i][j]==expected_row_indices_chunks[i][j], 
            "i,j:["<<i<<","<<j<<"] int: actual("<<row_indices_chunks[i][j]<<")!=expected("<< expected_row_indices_chunks[i][j] <<")\n");
            CHECK_MESSAGE(row_values_chunks[i][j]==expected_row_values_chunks[i][j], 
            "i,j:["<<i<<","<<j<<"] int: actual("<<row_values_chunks[i][j]<<")!=expected("<< expected_row_values_chunks[i][j] <<")\n");
        }
    }

    delete[] W;
}*/

TEST_CASE("_do_linear_sidlso_fwd quantized weights", "[do_linear_sidlso_fwd]")
{
    int num_cpus = 4;
    int input_size = 10;
    int output_size = 20;
    int batch = 1;
    csr_struct<int, float> input_csr;
    input_csr.ptrs.reset(new int[5]{0, 3, 6, 9, 12}); // Example pointers
    input_csr.indices.reset(new int[12]{0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2}); // Example indices
    input_csr.values.reset(new float[12]{-1.0f, -2.0f, -1.0f, 2.0f, -1.0f, -2.0f, -1.0f, 2.0f, -1.0f, -2.0f, -1.0f, 2.0f}); // Example values
    input_csr.rows = 4;
    input_csr.cols = 10;
    QuantizedFloatArray<float, 2> W(input_size*output_size, 2.5, 2.5);
    float* W2 = new float[input_size * output_size]{
        0.0f, 2.5f, 5.0f, 2.5f, 0.0f, 2.5f, 2.5f, 2.5f, 0.0f, 2.5f, 
        0.0f, 5.0f, 0.0f, 0.0f, 5.0f, 5.0f, -2.5f, 5.0f, 5.0f, 5.0f, 
        -2.5f, 5.0f, 2.5f, -2.5f, 0.0f, 0.0f, 2.5f, 5.0f, -2.5f, -2.5f, 
        0.0f, 2.5f, 5.0f, -2.5f, 0.0f, -2.5f, 2.5f, 2.5f, 5.0f, 0.0f, 
        5.0f, 0.0f, 5.0f, 5.0f, 2.5f, 0.0f, 0.0f, 2.5f, 2.5f, 2.5f, 
        0.0f, -2.5f, 0.0f, 5.0f, 2.5f, -2.5f, 0.0f, -2.5f, 0.0f, 0.0f, 
        2.5f, 0.0f, 5.0f, 5.0f, 5.0f, -2.5f, 0.0f, -2.5f, 5.0f, -2.5f, 
        0.0f, -2.5f, 0.0f, -2.5f, 0.0f, 2.5f, 0.0f, 5.0f, 0.0f, -2.5f, 
        0.0f, 5.0f, 0.0f, 5.0f, 5.0f, 2.5f, -2.5f, 5.0f, 0.0f, 5.0f, 
        5.0f, 2.5f, 2.5f, 5.0f, 0.0f, -2.5f, 5.0f, 5.0f, 2.5f, 2.5f, 
        0.0f, 5.0f, -2.5f, -2.5f, -2.5f, 0.0f, 0.0f, 5.0f, 0.0f, -2.5f, 
        2.5f, 0.0f, 5.0f, -2.5f, 2.5f, 0.0f, 0.0f, 5.0f, 0.0f, 2.5f, 
        0.0f, -2.5f, -2.5f, -2.5f, 0.0f, -2.5f, 5.0f, 0.0f, 2.5f, -2.5f, 
        0.0f, 5.0f, -2.5f, 2.5f, 0.0f, 5.0f, 0.0f, -2.5f, 5.0f, 0.0f, 
        0.0f, 5.0f, -2.5f, 0.0f, 5.0f, 5.0f, -2.5f, 0.0f, -2.5f, -2.5f, 
        -2.5f, 5.0f, -2.5f, 0.0f, -2.5f, 5.0f, 0.0f, 5.0f, 5.0f, 2.5f, 
        0.0f, 0.0f, -2.5f, 2.5f, 0.0f, 2.5f, 5.0f, 0.0f, 2.5f, 0.0f, 
        -2.5f, 0.0f, -2.5f, 0.0f, 5.0f, 5.0f, 2.5f, 5.0f, 2.5f, 0.0f, 
        2.5f, -2.5f, -2.5f, -2.5f, -2.5f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 
        2.5f, -2.5f, 5.0f, -2.5f, 5.0f, -2.5f, 0.0f, -2.5f, -2.5f, -2.5f
    };
    for(int i=0;i<input_size*output_size; i++){
        W[i]=W2[i];
    }
    for(int i=0;i<input_size*output_size; i++){
        REQUIRE_MESSAGE(abs(W[i]-W2[i])<std::numeric_limits<float>::epsilon(), "failed to set weight array at "<< i);
    }

    sili::unique_vector<sili::unique_vector<int>> row_indices_chunks(num_cpus);
    sili::unique_vector<sili::unique_vector<float>> row_values_chunks(num_cpus);

    float eps = std::numeric_limits<float>::epsilon();

    #pragma omp parallel num_threads(num_cpus)
    {
        _do_linear_sidlso_fwd(num_cpus,input_size,output_size,batch,input_csr,W,row_indices_chunks,row_values_chunks,eps);
    }

    // 15 out of 20 possible outputs. This just happens more often with quantized weights.
    sili::unique_vector<sili::unique_vector<int>> expected_row_indices_chunks{
        {1, 2, 4},
        {5, 6,7, 9},
        {10, 11, 13,14}, 
        {15, 17, 18, 19}
    };
    sili::unique_vector<sili::unique_vector<float>> expected_row_values_chunks{
        {-15.0, -5.0, 7.5},
        {12.5, 10.0, -10.0, 15.0},
        {-2.5,-7.5, -5.0, -15.0}, 
        { -7.5, -15.0, -2.5, -5.0}
    };

    CHECK_NESTED_VECTOR_EQUAL(row_indices_chunks, expected_row_indices_chunks);
    CHECK_NESTED_VECTOR_EQUAL(row_values_chunks, expected_row_values_chunks);

    delete[] W2;
}

TEST_CASE("linear_sidlso edge case: nullptr not allowed", "[do_linear_sidlso_fwd]") {
    int num_cpus = 4;
    int input_size = 0;
    int output_size = 20;
    int batch = 5;
    csr_struct<int, float> input_csr;
    input_csr.ptrs = nullptr;
    input_csr.indices = nullptr;
    input_csr.values = nullptr;
    input_csr.rows = 0;
    input_csr.cols = 0;

    float*W = new float[input_size* output_size];

    sili::unique_vector<sili::unique_vector<int>> row_indices_chunks(num_cpus);
    sili::unique_vector<sili::unique_vector<float>> row_values_chunks(num_cpus);

    float eps = 0.01f;

    REQUIRE_THROWS_AS(linear_sidlso(batch, input_size, output_size, input_csr, W, eps), std::runtime_error);

    delete[] W;
}

TEST_CASE("_do_linear_sidlso_fwd edge case: single-threaded execution", "[do_linear_sidlso_fwd]") {
    int num_cpus = 1;
    int input_size = 10;
    int output_size = 20;
    int batch = 1;
    csr_struct<int, float> input_csr;
    input_csr.ptrs.reset(new int[5]{0, 3, 6, 9, 12}); // Example pointers
    input_csr.indices.reset(new int[12]{0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2}); // Example indices
    input_csr.values.reset(new float[12]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}); // Example values
    input_csr.rows = 4;
    input_csr.cols = 10;

    float* W = new float[input_size * output_size]{
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
       1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 1.};


    sili::unique_vector<sili::unique_vector<int>> row_indices_chunks(num_cpus);
    sili::unique_vector<sili::unique_vector<float>> row_values_chunks(num_cpus);

    float eps = 0.01f;

    _do_linear_sidlso_fwd(num_cpus, input_size, output_size, batch, input_csr, W, row_indices_chunks, row_values_chunks, eps);

    sili::unique_vector<sili::unique_vector<int>> expected_row_indices_chunks{{3, 4, 5, 13, 14, 15}};
    sili::unique_vector<sili::unique_vector<float>> expected_row_values_chunks{
        {4,5, 6, 4, 5, 6}
    };

    // Verify results
    for (int i=0;i<num_cpus;i++) {
        REQUIRE_MESSAGE(row_indices_chunks[i].size() == expected_row_indices_chunks[i].size(), "indices size mismatch at "<< i);
        REQUIRE_MESSAGE(row_values_chunks[i].size() == expected_row_values_chunks[i].size(), "values size mismatch at "<< i);
    }

    for (size_t i = 0; i < row_values_chunks.size(); i++) {
        for (size_t j = 0; j < row_values_chunks[i].size(); j++) {
            CHECK_MESSAGE(row_indices_chunks[i][j]==expected_row_indices_chunks[i][j], 
            "i,j:["<<i<<","<<j<<"] int: actual("<<row_indices_chunks[i][j]<<")!=expected("<< expected_row_indices_chunks[i][j] <<")\n");
            CHECK_MESSAGE(row_values_chunks[i][j]==expected_row_values_chunks[i][j], 
            "i,j:["<<i<<","<<j<<"] int: actual("<<row_values_chunks[i][j]<<")!=expected("<< expected_row_values_chunks[i][j] <<")\n");
        }
    }

    delete[] W;
}

TEST_CASE("Linear SIDLSO Test Suite") {
    SECTION("Test Linear SIDLSO with valid inputs") {
        int batch_size = 4;
        int input_size = 5;
        int output_size = 10;
        csr_struct<int, float> input_csr;
        input_csr.ptrs.reset(new int[5]{0, 2, 3, 4, 5});
        input_csr.indices.reset(new int[5]{0, 1, 1,3,2});
        input_csr.values.reset(new float[5]{7.0f, -4.0f, 4.0f, 10.0f, 2.0f});
        input_csr.rows = 4;
        input_csr.cols = 5;
        float*W = new float[input_size *output_size]{
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 
            0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 
            1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 
            0.0f, 1.0f, 1.0f, 1.0f, 1.0f
        };

        csr_struct output_csr = linear_sidlso(batch_size, input_size, output_size, input_csr, W);

        sili::unique_vector<int> expected_ptrs{0,7,9,15,23};
        sili::unique_vector<int> expected_indices{1, 2, 3, 6, 7, 8, 9, 2, 9, 0, 2, 4, 5, 7, 9, 2, 3, 4, 5, 6, 7, 8, 9};
        sili::unique_vector<float> expected_values {7, 3, 7, 7, 7, 7, -4, 4, 4, 10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 2, 2, 2, 2} ;

        // Check if output_csr has correct dimensions
        REQUIRE_MESSAGE((output_csr.rows == batch_size && output_csr.cols == output_size),
                       "Output CSR dimensions incorrect.");
        CHECK(output_csr.nnz() == expected_values.size());

        for(int i=0; i<expected_ptrs.size();i++){
            CHECK_MESSAGE(expected_ptrs[i]==output_csr.ptrs[i], "Mismatch at "<<i);
        }

        // Check if there are no zeros or values less than epsilon in output_csr.values
        for (int i = 0; i < output_csr.nnz() && i<expected_values.size(); ++i) {
            CHECK_MESSAGE(expected_indices[i]==output_csr.indices[i], "Index mismatch at "<<i);
            CHECK_MESSAGE(expected_values[i]==output_csr.values[i], "Value mismatch at "<<i);
        }
    }
}







/*
TEST_CASE("_assign_spv_chunks_to_batch basic functionality", "[assign_spv_chunks_to_batch]")
{
    int batch = 0;
    int num_cpus = 4;
    sili::unique_vector<size_t> vec_assign_locs{0,3,5,8,10};
    sili::unique_vector<sili::unique_vector<int>> out_idx(1);
    sili::unique_vector<sili::unique_vector<float>> out_val(1);
    sili::unique_vector<sili::unique_vector<int>> row_indices_chunks{{0,1,2},{3,4},{5,6,7}, {8, 9}};
    sili::unique_vector<sili::unique_vector<float>> row_values_chunks{{0.1f,0.2f,0.3f},{0.4f,0.5f},{0.6f,0.7f,0.8f}, {0.9f, 1.0f}};
    int nnz = 0;

    _assign_spv_chunks_to_batch(batch,num_cpus,vec_assign_locs,out_idx,out_val,row_indices_chunks,row_values_chunks,nnz);

    REQUIRE_MESSAGE(nnz==10,"NNZ count incorrect");
    REQUIRE_MESSAGE(out_idx[0].size()==10,"Output idx size incorrect");
    REQUIRE_MESSAGE(out_val[0].size()==10,"Output val size incorrect");

    for(size_t i=0;i<out_idx[0].size();++i){
        REQUIRE_MESSAGE(i==out_idx[0][i],"Idx mismatch at position "+std::to_string(i));
    }

    for(size_t i=0;i<out_val[0].size();++i){
        REQUIRE_MESSAGE(std::abs(out_val[0][i]-i*0.1f)<0.001f,"Val mismatch at position "+std::to_string(i));
    }
}*/
/*
TEST_CASE("_assign_spv_chunks_to_batch empty row indices chunks", "[assign_spv_chunks_to_batch]")
{
    int batch = 0;
    int num_cpus = 4;
    sili::unique_vector<size_t> vec_assign_locs{0,5,10,15,20};
    sili::unique_vector<sili::unique_vector<int>> out_idx(1);
    sili::unique_vector<sili::unique_vector<float>> out_val(1);
    sili::unique_vector<sili::unique_vector<int>> row_indices_chunks{{{},{}},{},{}};
    sili::unique_vector<sili::unique_vector<float>> row_values_chunks{{{},{}}};
    int nnz = 0;

_assign_spv_chunks_to_batch(batch,num_cpus,vec_assign_locs,out_idx,out_val,row_indices_chunks,row_values_chunks,nnz);

    REQUIRE_MESSAGE(nnz==0,"NNZ count incorrect");
    REQUIRE_MESSAGE(out_idx[0].empty(),"Output idx not empty");
    REQUIRE_MESSAGE(out_val[0].empty(),"Output val not empty");
}

TEST_CASE("_assign_spv_chunks_to_batch single thread execution", "[assign_spv_chunks_to_batch]")
{
    int batch = 0;
    int num_cpus = 1;
    sili::unique_vector<size_t> vec_assign_locs{0,5};
    sili::unique_vector<sili::unique_vector<int>> out_idx(1);
    sili::unique_vector<sili::unique_vector<float>> out_val(1);
    sili::unique_vector<sili::unique_vector<int>> row_indices_chunks{{0,1,2}};
    sili::unique_vector<sili::unique_vector<float>> row_values_chunks{{0.1f,0.2f,0.3f}};
    int nnz = 0;

_assign_spv_chunks_to_batch(batch,num_cpus,vec_assign_locs,out_idx,out_val,row_indices_chunks,row_values_chunks,nnz);

    REQUIRE_MESSAGE(nnz==3,"NNZ count incorrect");
    REQUIRE_MESSAGE(out_idx[0].size()==3,"Output idx size incorrect");
    REQUIRE_MESSAGE(out_val[0].size()==3,"Output val size incorrect");

    for(size_t i=0;i<out_idx[0].size();++i){
        REQUIRE_MESSAGE(i==out_idx[0][i],"Idx mismatch at position "+std::to_string(i));
    }

    for(size_t i=0;i<out_val[0].size();++i){
        REQUIRE_MESSAGE(std::abs(out_val[0][i]-i*0.1f)<0.001f,"Val mismatch at position "+std::to_string(i));
    }
}









*/
TEST_CASE("Linear Backward SIDLSO Test Suite") {
    SECTION("Test Linear Backward SIDLSO with valid inputs") {
        int batch_size = 4;
        int input_size = 20;
        int output_size = 30;
        csr_struct<int, float> input_csr;
        input_csr.ptrs.reset(new int[5]{0, 2, 3, 4, 5});
        input_csr.indices.reset(new int[5]{0, 1, 1,3,2});
        input_csr.values.reset(new float[5]{7.0f, -4.0f, 4.0f, 10.0f, 2.0f});
        input_csr.rows = batch_size;
        input_csr.cols = input_size;
        float*W = new float[input_size *output_size]{
            0.19f, 0.03f, 0.23f, 0.80f, 0.41f, 0.22f, 0.49f, 0.08f, 0.95f, 0.03f, 
            0.24f, 0.22f, 0.45f, 0.92f, 0.98f, 0.11f, 0.06f, 0.78f, 0.37f, 0.30f, 
            0.07f, 0.48f, 0.16f, 0.25f, 0.45f, 0.66f, 0.74f, 0.68f, 0.50f, 0.81f, 
            0.89f, 0.81f, 0.90f, 0.98f, 0.44f, 0.39f, 0.28f, 0.28f, 0.63f, 0.16f, 
            0.70f, 0.95f, 0.01f, 0.51f, 0.97f, 0.72f, 0.33f, 0.71f, 0.88f, 0.32f, 
            0.57f, 0.69f, 0.37f, 0.04f, 0.75f, 0.68f, 0.28f, 0.21f, 0.02f, 0.80f, 
            0.43f, 0.83f, 0.07f, 0.14f, 0.33f, 0.85f, 0.90f, 0.19f, 0.25f, 0.72f, 
            0.03f, 0.01f, 0.28f, 0.11f, 0.03f, 0.85f, 0.81f, 0.01f, 0.67f, 0.81f, 
            0.92f, 0.36f, 0.93f, 0.03f, 0.40f, 0.18f, 0.58f, 0.68f, 0.22f, 0.98f, 
            0.11f, 0.89f, 0.21f, 0.24f, 0.99f, 0.64f, 0.19f, 0.61f, 0.99f, 0.95f, 
            0.61f, 0.85f, 0.48f, 0.22f, 0.48f, 0.32f, 0.60f, 0.43f, 0.42f, 0.95f, 
            0.58f, 0.63f, 0.87f, 0.93f, 0.39f, 0.82f, 0.32f, 0.57f, 0.78f, 0.67f, 
            0.38f, 0.02f, 0.82f, 0.25f, 0.42f, 0.71f, 0.11f, 0.57f, 0.12f, 0.35f, 
            0.03f, 0.69f, 0.34f, 0.14f, 0.48f, 0.00f, 0.59f, 0.01f, 0.99f, 0.94f, 
            0.46f, 0.28f, 0.07f, 0.10f, 0.78f, 0.72f, 0.20f, 0.96f, 0.79f, 0.10f, 
            0.89f, 0.20f, 0.35f, 0.83f, 0.68f, 0.56f, 0.94f, 0.18f, 0.96f, 0.33f, 
            0.06f, 0.10f, 0.86f, 0.05f, 0.71f, 0.68f, 0.85f, 0.78f, 0.36f, 0.95f, 
            0.21f, 0.21f, 0.80f, 0.77f, 0.79f, 0.21f, 0.03f, 0.88f, 0.13f, 0.87f, 
            0.72f, 0.51f, 0.42f, 0.81f, 0.75f, 0.63f, 0.84f, 0.61f, 0.23f, 0.38f, 
            0.77f, 0.09f, 0.56f, 0.46f, 0.47f, 0.91f, 0.60f, 0.39f, 0.49f, 0.69f
        };

        csr_struct<int, float> output_grad_csr;
        output_grad_csr.ptrs.reset(new int[5]{0, 2, 3, 4, 5});
        output_grad_csr.indices.reset(new int[5]{1, 5, 0,7,10});
        output_grad_csr.values.reset(new float[5]{7.0f, -4.0f, 4.0f, 10.0f, 2.0f});
        output_grad_csr.rows = batch_size;
        output_grad_csr.cols = output_size;
        csr_struct<int, float> I_grad;
        I_grad.ptrs.reset(new int[5]{0, 1, 3, 4, 5});
        I_grad.indices.reset(new int[5]{5, 6, 9,3,1});
        I_grad.rows = batch_size;
        I_grad.cols = input_size;

        std::shared_ptr<BasicLinearGradUpdater> updater(new BasicLinearGradUpdater(input_size, output_size));  //inits to zero
        auto W_grad_callback = get_dense_W_grad_callback(updater);

        linear_backward_sidlso(batch_size, input_size, output_size, input_csr, W, output_grad_csr, I_grad, W_grad_callback);

        std::vector<float> I_grad_expected{4.62, 1.96, 0.12, 1.00, 0};
        std::vector<float> I_grad_actual((I_grad.values.get()), (I_grad.values.get())+I_grad.nnz());

        CHECK_VECTOR_ALMOST_EQUAL(I_grad_expected, I_grad_actual, 0.001);

        std::vector<float> W_grad_expected(input_size*output_size, 0);
        W_grad_expected[1]=16;
        W_grad_expected[20]=49;
        W_grad_expected[21]=-28;
        W_grad_expected[143]=100;
        W_grad_expected[202]=4;
        std::vector<float> W_grad_actual((updater->w_grad.get()), (updater->w_grad.get())+input_size*output_size);

        CHECK_VECTOR_ALMOST_EQUAL(W_grad_expected, W_grad_actual, 0.001);

        delete[] W;
    }
/*
    SECTION("Test Linear Backward SIDLSO with invalid inputs") {
        int batch_size = 0;
        int input_size = 20;
        int output_size = 30;
        csr_struct input_csr;
        float*W = new float[input_size*output_size];

        // Initialize W with random values
        for (int i = 0; i < input_size*output_size; ++i) {
            W[i] = static_cast<float>(rand() % 100) / 100.0f;
        }

        csr_struct output_grad_csr;
        csr_struct I_grad;

        std::shared_ptr<WeightGradUpdater> updater(new WeightGradUpdater(input_size, output_size));
        auto W_grad_callback = get_dense_W_grad_callback(updater);

        linear_backward_sidlso(batch_size, input_size, output_size, input_csr, W, output_grad_csr, I_grad, W_grad_callback);

        // Verify that the weights have been updated correctly
        for (size_t i = 0; i < input_size*output_size; ++i) {
            REQUIRE_MESSAGE(updater->w_grad[i] != 0.f, "Gradient at position [" << i << "] is zero");
        }
    }

    SECTION("Test Linear Backward SIDLSO edge case") {
        int batch_size = 1;
        int input_size = 1;
        int output_size = 1;
        csr_struct input_csr;
        float*W = new float[input_size *output_size];

        // Initialize W with random value
        W[0] = static_cast<float>(rand() % 100) / 100.0f;

        csr_struct output_grad_csr;
        csr_struct I_grad;

        std::shared_ptr<WeightGradUpdater> updater(new WeightGradUpdater(input_size, output_size));
        auto W_grad_callback = get_dense_W_grad_callback(updater);

        linear_backward_sidlso(batch_size, input_size, output_size, input_csr, W, output_grad_csr, I_grad, W_grad_callback);

        // Verify that the weights have been updated correctly
        for (size_t i = 0; i < input_size* output_size; ++i) {
            REQUIRE_MESSAGE(updater->w_grad[i] != 0.f, "Gradient at position [" << i << "] is zero");
        }
    }*/
}


TEST_CASE("CSRMASK Class Tests") {
    SECTION("Add Random Value Test") {
        csr_struct<int, float> csr_matrix;
        // these need to be defined before a valid csr_mask. Nothing else does.
        csr_matrix.rows = 4;
        csr_matrix.cols = 5;
        
        CSRMask csr_mask(csr_matrix);

        csr_matrix.ptrs.reset(new int[5]{0, 2, 3, 4, 5});
        csr_matrix.indices.reset(new int[5]{0, 1, 1,3,2});
        csr_matrix.values.reset(new float[5]{0.7f, 0.4f, 0.3f, 0.1f, 0.2f});
        
        // Add some initial values to the CSR matrix
        

        // Call addRandomValue with default parameters
        csr_mask.addRandomValue(2 * std::numbers::pi / 100000, 2 * std::numbers::pi / 50000);

        std::vector<float> original_values{0.7f, 0.4f, 0.3f, 0.1f, 0.2f};

        // Check if values have changed
        for (int i = 0; i < csr_matrix.nnz(); ++i) {
            CHECK_MESSAGE(csr_matrix.values[i]>original_values[i], "Values did not change after addRandomValue call");
        }
    }

    SECTION("Remove Element Test") {
        csr_struct<int, float> csr_matrix;
        // these need to be defined before a valid csr_mask. Nothing else does.
        csr_matrix.rows = 4;
        csr_matrix.cols = 5;
        
        CSRMask csr_mask(csr_matrix);

        csr_matrix.ptrs.reset(new int[5]{0, 2, 3, 4, 5});
        csr_matrix.indices.reset(new int[5]{0, 1, 1,3,2});
        csr_matrix.values.reset(new float[5]{0.7f, 0.4f, 0.3f, 0.1f, 0.2f});


        // Remove an element at index 2
        csr_mask.removeElement(2);

        std::vector<int> expected_ptrs{0, 2, 2, 3, 4};
        std::vector<int> expected_indices{0, 1, 3, 2 };
        std::vector<float> expected_values{0.7f, 0.4f, 0.1f, 0.2f };

        std::vector<int> actual_ptrs((csr_matrix.ptrs.get()), (csr_matrix.ptrs.get())+csr_matrix.rows+1);
        std::vector<int> actual_indices((csr_matrix.indices.get()), (csr_matrix.indices.get())+csr_matrix.nnz());
        std::vector<float> actual_values((csr_matrix.values.get()), (csr_matrix.values.get())+csr_matrix.nnz());

        // Check if element was removed successfully
        REQUIRE_MESSAGE(csr_matrix.nnz() == 4, "NNZ count did not decrease after removing element");

        CHECK_VECTOR_EQUAL(expected_ptrs, actual_ptrs);
        CHECK_VECTOR_EQUAL(expected_indices, actual_indices);
        CHECK_VECTOR_ALMOST_EQUAL(expected_values, actual_values);
    }

    SECTION("Add Random Elements Test") {
        csr_struct<int, float> csr_matrix;
        // these need to be defined before a valid csr_mask. Nothing else does.
        csr_matrix.rows = 4;
        csr_matrix.cols = 5;
        
        CSRMask csr_mask(csr_matrix);

        csr_matrix.ptrs.reset(new int[5]{0, 2, 3, 4, 5});
        csr_matrix.indices.reset(new int[5]{0, 1, 1,3,2});
        csr_matrix.values.reset(new float[5]{0.7f, 0.4f, 0.3f, 0.1f, 0.2f});

        // Insert two new elements randomly
        csr_mask.addRandomElements(20);

        // Check if NNZ count increased
        REQUIRE_MESSAGE(csr_matrix.nnz()>5, "NNZ count did not increase after inserting elements");
        REQUIRE_MESSAGE(csr_matrix._reserved_indices_and_values >5, "csr_struct didn't reserve space for new elements");

        /*std::vector<int> expected_ptrs{0, 2, 3, 5, 6};
        std::vector<int> expected_indices{0, 1, 1, 3, 4, 2 };
        std::vector<float> expected_values{0.7f, 0.4f, 0.1f, 0.0f, 0.2f };

        std::vector<int> actual_ptrs((csr_matrix.ptrs.get()), (csr_matrix.ptrs.get())+csr_matrix.rows+1);
        std::vector<int> actual_indices((csr_matrix.indices.get()), (csr_matrix.indices.get())+csr_matrix.nnz());
        std::vector<float> actual_values((csr_matrix.values.get()), (csr_matrix.values.get())+csr_matrix.nnz());

        CHECK_VECTOR_EQUAL(expected_ptrs, actual_ptrs);
        CHECK_VECTOR_EQUAL(expected_indices, actual_indices);
        CHECK_VECTOR_ALMOST_EQUAL(expected_values, actual_values);*/
    }

    SECTION("Iterate while maintaining 20 random elements") {
        csr_struct<int, float> csr_matrix;
        // these need to be defined before a valid csr_mask. Nothing else does.
        csr_matrix.rows = 40;
        csr_matrix.cols = 50;
        
        CSRMask csr_mask(csr_matrix);

        /*csr_matrix.ptrs.reset(new int[5]{0, 2, 3, 4, 5});
        csr_matrix.indices.reset(new int[5]{0, 1, 1,3,2});
        csr_matrix.values.reset(new float[5]{0.7f, 0.4f, 0.3f, 0.1f, 0.2f});*/

        // Insert two new elements randomly
        csr_mask.iterate(10, 0.1, 0.6);  // first will insert zero values
        csr_mask.iterate(10, 0.1, 0.6);  // next will update values and get closer to 10

        // Check if NNZ count increased
        REQUIRE_MESSAGE(csr_matrix.nnz()>0, "NNZ didn't update");
        REQUIRE_MESSAGE(csr_matrix._reserved_indices_and_values >0, "csr didn't reserve new space");

        /*std::vector<int> expected_ptrs{0, 2, 3, 5, 6};
        std::vector<int> expected_indices{0, 1, 1, 3, 4, 2 };
        std::vector<float> expected_values{0.7f, 0.4f, 0.1f, 0.0f, 0.2f };

        std::vector<int> actual_ptrs((csr_matrix.ptrs.get()), (csr_matrix.ptrs.get())+csr_matrix.rows+1);
        std::vector<int> actual_indices((csr_matrix.indices.get()), (csr_matrix.indices.get())+csr_matrix.nnz());
        std::vector<float> actual_values((csr_matrix.values.get()), (csr_matrix.values.get())+csr_matrix.nnz());

        CHECK_VECTOR_EQUAL(expected_ptrs, actual_ptrs);
        CHECK_VECTOR_EQUAL(expected_indices, actual_indices);
        CHECK_VECTOR_ALMOST_EQUAL(expected_values, actual_values);*/
    }
}
