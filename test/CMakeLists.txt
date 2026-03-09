cmake_minimum_required(VERSION 3.10.0)
project(sili_tests)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_INCLUDE_HEADERS_IN_COMPILE_COMMANDS ON)

include_directories("../sili/")
include_directories("../sili/lib/headers")
include_directories("../sili/lib/src")

set(SOURCE_FILES
    #../sili/lib/src/cache_info.cpp
    # these are not included since we're testing private c++ functions
    #../sili/lib/src/linear.cpp
    #../sili/lib/src/outer_product.cpp
    #../sili/lib/src/csf.cpp
    #../sili/lib/src/conv2d.cpp
    )

set(HEADER_FILES
    tests_main.hpp

    ../sili/lib/headers/parallel.hpp

    ../sili/lib/headers/linear_sisldo.hpp
    ../sili/lib/headers/linear_disldo.hpp

    #PASSED TESTS
    ../sili/lib/headers/csr.hpp
    ../sili/lib/headers/loss.hpp

    #../sili/lib/headers/unique_vector.hpp
    #../sili/lib/headers/coo.hpp

    #OLD TESTS
    #../sili/lib/headers/cache_info.h
    #../sili/lib/headers/csr.h
    #../sili/lib/headers/csf.h
    #../sili/lib/headers/scan.h
    )

find_package(Catch2 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(
    sili_tests

    #unittest_sisldo.cpp
    integration_test_sisldo.cpp

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

# set(Python3_LIBRARY "/usr/lib/python3.11/config-3.11-x86_64-linux-gnu/libpython3.11.a")
set(Python3_LIBRARY "/usr/lib/python3.14/config-3.14-x86_64-linux-gnu/libpython3.14.a")


# currently, torch is broken in arch linux. Package is marked out of date. Should be fixed in a week or so.
# pacman -S python-pytorch-opt-cuda
# add: torch torch_cpu
target_include_directories(sili_tests PRIVATE "../sili/" )
target_include_directories(sili_tests PRIVATE "../sili/lib/headers" )
target_include_directories(sili_tests PRIVATE "../sili/lib/src")
target_link_libraries(sili_tests Catch2::Catch2WithMain OpenMP::OpenMP_CXX ${Python3_LIBRARY} )

option(ENABLE_AVX "Enable AVX support" ON)
option(ENABLE_FMA "Enable FMA support" ON)
option(ENABLE_AVX2 "Enable AVX2 support" ON)

if (ENABLE_AVX2)
    message("Enableing AVX2, FMA, and AVX")
    target_compile_options(sili_tests PRIVATE -march=haswell)
elseif(ENABLE_FMA)
    message("Enableing FMA and AVX") # and FMA
    target_compile_options(sili_tests PRIVATE -march=bdver3)
elseif(ENABLE_AVX)
    message("Enableing AVX only") # and FMA
    target_compile_options(sili_tests PRIVATE -march=sandybridge)
endif()


# Set the C++ standard (adjust as needed)
set_target_properties(sili_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
#    CXX_EXTENSIONS OFF
)

set(Python3_INCLUDE_DIR "/usr/include/python3.14")
message("Include dirs:")
message(${Python3_INCLUDE_DIR})
include_directories(${Python3_INCLUDE_DIR})

# Enable testing
include(CTest)
include(Catch)
catch_discover_tests(sili_tests)