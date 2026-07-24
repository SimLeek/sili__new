#include "tests_main.h"
#include "unique_vector.hpp"
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

// Example struct to test non-POD types
struct ExampleObject {
    int x;
    ExampleObject() : x(0) {}
    ExampleObject(int val) : x(val) {}
    ExampleObject(const ExampleObject &) = delete;                     // Make it non-copyable
    ExampleObject(ExampleObject &&other) : x(other.x) { other.x = 0; } // Support move semantics
};

struct CopyableObject {
    int x;
    CopyableObject() : x(0) {}
    CopyableObject(int val) : x(val) {}
    CopyableObject(const CopyableObject &) = default; // Make it copyable
};

// Test case 1: Initialization with POD types
TEST_CASE("unique_vector POD initialization", "[unique_vector]") {
    sili::unique_vector<int> vec = {1, 2, 3, 4, 5};
    std::vector<int> expected = {1, 2, 3, 4, 5};

    CHECK_VECTOR_EQUAL(vec, expected);
    REQUIRE(vec.size() == 5);
    CHECK(vec[0] == 1);
    CHECK(vec[4] == 5);
}

// Test case 2: Initialization with non-POD objects
TEST_CASE("unique_vector Non-Copyable Object initialization", "[unique_vector]") {
    REQUIRE_THROWS_AS((sili::unique_vector<ExampleObject>{ExampleObject(1), ExampleObject(2), ExampleObject(3)}),
                      std::invalid_argument);
}

// Test case 2: Initialization with copyable non-POD objects
TEST_CASE("unique_vector Copyable Object initialization", "[unique_vector]") {
    sili::unique_vector<CopyableObject> vec = {CopyableObject(1), CopyableObject(2), CopyableObject(3)};
    std::vector<int> expected = {1, 2, 3};

    REQUIRE(vec.size() == 3);

    // Verify the values of the objects inside the unique_vector
    for (size_t i = 0; i < vec.size(); ++i) {
        CHECK(vec[i].x == expected[i]);
    }
}

// Test case 3: Nested initializer lists
TEST_CASE("unique_vector Nested initializer lists", "[unique_vector]") {
    sili::unique_vector<sili::unique_vector<int>> vec = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<std::vector<int>> expected = {{1, 2}, {3, 4}, {5, 6}};

    REQUIRE_NESTED_VECTOR_EQUAL(vec, expected);
}

// Test case 4: Move semantics and non-copyable behavior
TEST_CASE("unique_vector Move semantics", "[unique_vector]") {
    sili::unique_vector<int> vec1 = {10, 20, 30};
    sili::unique_vector<int> vec2 = std::move(vec1);

    REQUIRE(vec1.size() == 0); // vec1 should be empty after the move
    REQUIRE(vec2.size() == 3);

    std::vector<int> expected = {10, 20, 30};
    CHECK_VECTOR_EQUAL(vec2, expected);

    // Ensure copy constructor is deleted
    static_assert(!std::is_copy_constructible<sili::unique_vector<int>>::value, "unique_vector should not be copyable");
}

TEST_CASE("unique_vector size", "[unique_vector]") {
    sili::unique_vector<int> vec;
    REQUIRE(vec.size() == 0);

    vec.push_back(1);
    vec.push_back(2);
    REQUIRE(vec.size() == 2);

    vec.resize(5);
    REQUIRE(vec.size() == 5);
}

TEST_CASE("unique_vector capacity", "[unique_vector]") {
    sili::unique_vector<int> vec;
    REQUIRE(vec.capacity() >= 0); // Ensure non-negative capacity

    vec.push_back(1);
    vec.push_back(2);
    REQUIRE(vec.capacity() >= 2); // Capacity should be at least equal to size

    vec.reserve(10);
    REQUIRE(vec.capacity() >= 10); // Capacity should be at least the reserved amount
}

TEST_CASE("unique_vector data", "[unique_vector]") {
    sili::unique_vector<int> vec = {1, 2, 3};
    int* data = vec.data();

    REQUIRE(data[0] == 1);
    REQUIRE(data[1] == 2);
    REQUIRE(data[2] == 3);

    // Modify elements through data pointer
    data[0] = 10;
    REQUIRE(vec[0] == 10);
}

TEST_CASE("unique_vector iterators", "[unique_vector]") {
    sili::unique_vector<int> vec = {1, 2, 3};

    // Check iterator values
    auto it = vec.begin();
    REQUIRE(*it == 1);
    ++it;
    REQUIRE(*it == 2);

    // Check end iterator
    auto end = vec.end();
    --end;
    REQUIRE(*end == 3);

    // Modify elements through iterator
    *it = 10;
    REQUIRE(vec[1] == 10);
}

TEST_CASE("unique_vector emplace_back and push_back", "[unique_vector]") {
    sili::unique_vector<int> vec;

    // Test push_back
    vec.push_back(1);
    vec.push_back(2);
    REQUIRE(vec.size() == 2);
    REQUIRE(vec[0] == 1);
    REQUIRE(vec[1] == 2);

    // Test emplace_back with POD types
    vec.emplace_back(3);
    REQUIRE(vec.size() == 3);
    REQUIRE(vec[2] == 3);

    // Test emplace_back with custom types
    struct MyStruct { int value; };
    sili::unique_vector<MyStruct> vec2;
    vec2.emplace_back(4);
    REQUIRE(vec2.size() == 1);
    REQUIRE(vec2[0].value == 4);
}