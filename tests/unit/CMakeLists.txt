cmake_minimum_required(VERSION 3.10.0)
project(sili_tests)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_INCLUDE_HEADERS_IN_COMPILE_COMMANDS ON)

include_directories("../../sili/")
include_directories("../../sili/lib/headers")
include_directories("../../sili/lib/src")

set(SOURCE_FILES
    #../../sili/lib/src/cache_info.cpp
    # these are not included since we're testing private c++ functions
    #../../sili/lib/src/linear.cpp
    #../../sili/lib/src/outer_product.cpp
    #../../sili/lib/src/csf.cpp
    #../../sili/lib/src/conv2d.cpp
    )

set(HEADER_FILES
    tests_main.hpp

    ../../sili/lib/headers/parallel.hpp

    ../../sili/lib/headers/linear_sisldo.hpp
    ../../sili/lib/headers/linear_disldo.hpp

    #PASSED TESTS
    ../../sili/lib/headers/csr.hpp
    ../../sili/lib/headers/loss.hpp

    #../../sili/lib/headers/unique_vector.hpp
    #../../sili/lib/headers/coo.hpp

    #OLD TESTS
    #../../sili/lib/headers/cache_info.h
    #../../sili/lib/headers/csr.h
    #../../sili/lib/headers/csf.h
    #../../sili/lib/headers/scan.h
    )

find_package(Catch2 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(
    sili_tests

    # unittest_sisldo.cpp -- PARKED (see conversation): tests the older
    # TriValues (3-separate-float32-array) high-precision path specifically,
    # not FP4 at all. Calls make_weights() with the OLD 7-arg signature
    # (shape, ptrs, indices, values, importance, backprop-array) -- current
    # signature is 5 args (shape, ptrs, indices, FP4BiPacked-values), since
    # backward now always updates value/importance internally (lr=0 to
    # disable) rather than taking a separate backprop flag/array. Per
    # project decision, high-precision support is backburner priority;
    # revisit this file's 7 TEST_CASEs individually when that's picked
    # back up rather than blocking the main (4-bit) build on it now.
    #unittest_sisldo.cpp
    #integration_test_sisldo.cpp

    test_disldo_synaptogenesis.cpp
    test_hoyer_sparsify.cpp
    test_stats_thread_safety.cpp
    test_scale_handling.cpp
    test_synapse_update_policy.cpp
    test_attention.cpp
    test_csr_union.cpp

    #test_parallel.cpp
    #test_fiber.cpp

    #test_sisldo.cpp
    #COMPLETED TESTS
    #test_coo.cpp
    #test_csr.cpp


    #OLD TESTS
    #test_unique_vector.cpp
    #test_coo.cpp
    #test_cache_info.cpp
    #test_unique_vector.cpp
    #test_outer_product.cpp
    #test_conv2d.cpp
    #test_csr_csf.cpp
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

# PORTABLE FIX (see conversation): was hardcoded to one specific machine's
# Python install path (originally /usr/lib/python3.14/... -- broke on any
# other machine/Python version, including the sandbox this was verified in,
# which has 3.12). find_package(Python3) locates whatever's actually
# installed, on any machine, without editing this file per-environment.
find_package(Python3 COMPONENTS Development REQUIRED)

# Optional MKL detection for DIDLDO/SIDLDO (TODO_BATCH_BLOCKING.md, "Also
# queued"), mirroring setup.py's `_find_mkl()`: `pip install mkl
# mkl-include` into whatever Python environment this build is configured
# against installs the payload under that interpreter's `sys.prefix`
# (`lib/libmkl_*.so.N`, `include/mkl*.h`) -- confirmed by direct install
# and inspection, not assumed. The Development-only find_package(Python3)
# above does NOT populate Python3_EXECUTABLE (only Interpreter does) --
# found directly (it silently produced an empty sys.prefix the first time
# this was written) -- so resolve the interpreter separately, NOT
# REQUIRED since MKL support itself is optional; no interpreter found
# just means no MKL detection attempted.
find_package(Python3 COMPONENTS Interpreter)
set(SILI_MKL_FOUND FALSE)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.prefix)"
        OUTPUT_VARIABLE SILI_PY_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(SILI_MKL_LIB_DIR "${SILI_PY_PREFIX}/lib")
    set(SILI_MKL_INCLUDE_DIR "${SILI_PY_PREFIX}/include")
    if(EXISTS "${SILI_MKL_LIB_DIR}/libmkl_gnu_thread.so.3" AND EXISTS "${SILI_MKL_INCLUDE_DIR}/mkl_cblas.h")
        set(SILI_MKL_FOUND TRUE)
        message("Found MKL (mkl/mkl-include pip packages) under ${SILI_PY_PREFIX} -- building test_didldo_kernel")
    else()
        message("MKL not found under ${SILI_PY_PREFIX} (pip install mkl mkl-include to enable) -- skipping test_didldo_kernel")
    endif()
else()
    message("No Python3 interpreter found for MKL detection -- skipping test_didldo_kernel")
endif()


# currently, torch is broken in arch linux. Package is marked out of date. Should be fixed in a week or so.
# pacman -S python-pytorch-opt-cuda
# add: torch torch_cpu
target_include_directories(sili_tests PRIVATE "../../sili/" )
target_include_directories(sili_tests PRIVATE "../../sili/lib/headers" )
target_include_directories(sili_tests PRIVATE "../../sili/lib/src")
target_link_libraries(sili_tests Catch2::Catch2WithMain OpenMP::OpenMP_CXX ${Python3_LIBRARIES} )

option(ENABLE_AVX "Enable AVX support" ON)
option(ENABLE_FMA "Enable FMA support" ON)
option(ENABLE_AVX2 "Enable AVX2 support" ON)

if (ENABLE_AVX2)
    message("Enabling AVX2, FMA, and AVX")
    target_compile_options(sili_tests PRIVATE -march=haswell)
elseif(ENABLE_FMA)
    message("Enabling FMA and AVX") # and FMA
    target_compile_options(sili_tests PRIVATE -march=bdver3)
elseif(ENABLE_AVX)
    message("Enabling AVX only") # and FMA
    target_compile_options(sili_tests PRIVATE -march=sandybridge)
endif()


# Set the C++ standard (adjust as needed)
set_target_properties(sili_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
#    CXX_EXTENSIONS OFF
)

message("Include dirs:")
message(${Python3_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

# Enable testing
include(CTest)
include(Catch)
catch_discover_tests(sili_tests)

# Standalone (own int main(), not Catch2 TEST_CASE) block4/fp4 tests --
# never wired into sili_tests itself (would collide on multiple main()
# definitions), and until now never wired into CTest either -- only ever
# run via ad-hoc manual g++ invocations during the block4 integration work
# (see conversation). Real coverage gap, not intentional; fixed here rather
# than left as-is.
set(SILI_STANDALONE_TESTS
    test_disldo_block4_forward
    test_disldo_block4_backward
    test_disldo_block4_promotion
    test_block4_load_dense
    test_block4_load_sparse_fp32
    test_sisldo_sparse_load_backward_stability
    test_block4_scattered_divergence
    test_sisldo_disldo_parity
    test_sisldo_disldo_parity_fp32
    test_sisldo_disldo_parity_fp8
    test_sisldo_disldo_multibatch_parity
    test_fp8_block4_scattered_divergence
    test_synapse_policy_dispatch
    test_synapse_policy_long_horizon
    sweep_synapse_policy_min_decay_frac
    sweep_synapse_policy_stochastic
    probe_unstable_pocket_growth
    probe_unclipped_after_nan_fix
    test_ci_ceiling
    test_production_default_long_horizon
    test_disldo_block4_combined_absolute
    test_block4_sparse_tile
    test_block4_memory_cap_and_compression
    test_block4_sparse_input_forward
    test_block4_sparse_input_backward
    test_fp4_bitshift
    test_fp4_stochastic
    test_disldo_block4_fp4_wide_simd
    test_disldo_block4_fp4_backward_wide_simd
    test_fp8_bitshift
    test_fp8_block4_simd
    test_fp8_value_accessor
    test_fp8_block4_sparse_tile
    test_fp8_block4_store
    test_disldo_block4_fp8
    test_disldo_block4_fp8_wide_simd
    test_disldo_block4_fp8_backward_wide_simd
    test_disldo_block4_promotion_fp8
    test_fp32_block4_sparse_tile
    test_fp32_block4_store
    test_disldo_block4_fp32
    test_disldo_block4_fp32_wide_simd
    test_disldo_block4_fp32_batch_blocked
    test_disldo_scattered_batch_blocked
    test_disldo_block4_fp32_backward_wide_simd
    test_disldo_block4_promotion_fp32
    test_never_zero_live_quantize
    test_aqrs_additive_branch
    test_aqrs_rank_growth_shrink
    test_aqrs_gamma
    test_aqrs_dynamic_rank_control
    test_aqrs_dynamic_rank_control_integration
    test_aqrs_additive_gamma
    test_aqrs_rank_scratch
    test_aqrs_scale_overflow_guard
    test_top_k_csr_small_array_fast_path
    test_aqrs_grad_norm_dy_density
    test_amortized_decay_stats
    test_top_k_csr_nucleus
    test_cpu_topology
    test_dequant_equality_fp4_vs_fp8
    test_dequant_equality_fp8_vs_fp32
    test_block4_codec
    test_disldo_backward_group_reduction
)
foreach(t ${SILI_STANDALONE_TESTS})
    add_executable(${t} ${t}.cpp)
    target_include_directories(${t} PRIVATE "../../sili/" "../../sili/lib/headers" "../../sili/lib/src")
    target_link_libraries(${t} OpenMP::OpenMP_CXX)
    set_target_properties(${t} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# MKL-dependent tests need its headers/direct-linked GNU-threading-layer
# libs (not just OpenMP::OpenMP_CXX like every other test above), so
# they're kept out of the uniform SILI_STANDALONE_TESTS loop and only
# built when SILI_MKL_FOUND -- same graceful compile-out setup.py uses.
set(SILI_MKL_TESTS test_didldo_kernel test_sidldo_kernel)
if(SILI_MKL_FOUND)
    foreach(t ${SILI_MKL_TESTS})
        add_executable(${t} ${t}.cpp)
        target_include_directories(${t} PRIVATE "../../sili/" "../../sili/lib/headers" "../../sili/lib/src" "${SILI_MKL_INCLUDE_DIR}")
        target_link_directories(${t} PRIVATE "${SILI_MKL_LIB_DIR}")
        target_compile_definitions(${t} PRIVATE SILI_HAVE_MKL=1)
        target_link_libraries(${t} OpenMP::OpenMP_CXX
            "-l:libmkl_intel_lp64.so.3" "-l:libmkl_gnu_thread.so.3" "-l:libmkl_core.so.3"
            "-Wl,-rpath,${SILI_MKL_LIB_DIR}")
        set_target_properties(${t} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
        add_test(NAME ${t} COMMAND ${t})
    endforeach()
endif()