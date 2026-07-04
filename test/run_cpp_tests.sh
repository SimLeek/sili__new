#!/bin/bash

# Specify the build directory (create it if it doesn't exist)
build_dir="build_tests"
if [ ! -d "$build_dir" ]; then
    mkdir "$build_dir"
else
    rm -rf "$build_dir"
    mkdir "$build_dir"
fi

# Change to the build directory
cd "$build_dir" || exit

# Configure the project using CMake with default options
# Enable AVX2 by default
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build the tests (specify the number of CPU cores for parallel build with -j)
# Replace 4 with the desired number of CPU cores
# valgrind --leak-check=full ./sparse_tests
if cmake --build . -j4 && ./sili_tests; then
    echo "Tests passed."
else
    echo "Tests failed."
    exit 1
fi

exit


# Return to the original directory
cd ..

# Clean the build directory
rm -rf "$build_dir"

# Change to the build directory again
mkdir "$build_dir"
cd "$build_dir" || exit

# Configure the project using CMake with AVX enabled and AVX2 disabled
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_AVX2=OFF ..

# Build the tests (specify the number of CPU cores for parallel build with -j)
# Replace 4 with the desired number of CPU cores
if cmake --build . -j4 && ./sparse_tests; then
    echo "Tests passed."
else
    echo "Tests failed."
    exit 1
fi

# Return to the original directory
cd ..

# Clean the build directory again
rm -rf "$build_dir"

# Change to the build directory once more
mkdir "$build_dir"
cd "$build_dir" || exit

# Configure the project using CMake with both AVX and AVX2 disabled
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_AVX2=OFF -DENABLE_FMA=OFF ..

# Build the tests (specify the number of CPU cores for parallel build with -j)
# Replace 4 with the desired number of CPU cores
if cmake --build . -j4 && ./sparse_tests; then
    echo "Tests passed."
else
    echo "Tests failed."
    exit 1
fi

# Return to the original directory
cd ..

# Clean the build directory again
rm -rf "$build_dir"

# Change to the build directory once more
mkdir "$build_dir"
cd "$build_dir" || exit

# Configure the project using CMake with both AVX and AVX2 disabled
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_AVX2=OFF -DENABLE_FMA=OFF -DENABLE_AVX=OFF ..

# Build the tests (specify the number of CPU cores for parallel build with -j)
# Replace 4 with the desired number of CPU cores
if cmake --build . -j4 && ./sparse_tests; then
    echo "Tests passed."
else
    echo "Tests failed."
    exit 1
fi

# Return to the original directory
cd ..