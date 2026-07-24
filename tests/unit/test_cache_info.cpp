#include <iostream>

#include "tests_main.h"

#include "cache_info.h"

TEST_CASE("L1 Cache size retrieval", "[cache]") {
    size_t cache_size = get_l1_cache_size();

    REQUIRE_MESSAGE(cache_size > 0, "Cache size should be greater than 0");

    std::cout << "Detected L1 Cache Size: " << cache_size << " bytes" << std::endl;

    // Check the platform-specific cache size
#ifdef _WIN32
    std::cout << "Running on Windows. Cache size detected: " << cache_size << " bytes." << std::endl;
    CHECK_MESSAGE(cache_size > 0, "Windows cache size detection failed.");
#elif defined(__linux__)
    std::cout << "Running on Linux. Cache size detected: " << cache_size << " bytes." << std::endl;
    CHECK_MESSAGE(cache_size > 0, "Linux cache size detection failed.");
#elif defined(__APPLE__)
    std::cout << "Running on macOS. Cache size detected: " << cache_size << " bytes." << std::endl;
    CHECK_MESSAGE(cache_size > 0, "macOS cache size detection failed.");
#else
    std::cout << "Running on an unsupported platform. Cache size detection not tested." << std::endl;
#endif
}

TEST_CASE("Optimal L1 unrolled size computation", "[unrolled_size]") {
    size_t l1_cache_size = get_l1_cache_size();
    size_t element_size = sizeof(float);

    REQUIRE_MESSAGE(l1_cache_size > 0, "L1 cache size should be greater than 0 for this test to be valid.");

    SECTION("Half L1 Cache Size") {
        double fraction = 0.5;
        size_t optimal_unrolled_size = compute_optimal_unrolled_size(l1_cache_size, element_size, fraction);

        REQUIRE_MESSAGE(optimal_unrolled_size > 0, "Optimal unrolled size should be greater than 0");
        REQUIRE_MESSAGE(optimal_unrolled_size <= (l1_cache_size / element_size), "Optimal unrolled size should be within half L1 cache size range");

        // Calculate expected size
        size_t expected_size = static_cast<size_t>((l1_cache_size * fraction) / element_size);
        CHECK_MESSAGE(optimal_unrolled_size == expected_size, "Optimal unrolled size does not match expected size");

#ifdef _WIN32
        std::cout << "Windows: Optimal unrolled size calculated correctly: " << optimal_unrolled_size << " elements." << std::endl;
#elif defined(__linux__)
        std::cout << "Linux: Optimal unrolled size calculated correctly: " << optimal_unrolled_size << " elements." << std::endl;
#elif defined(__APPLE__)
        std::cout << "macOS: Optimal unrolled size calculated correctly: " << optimal_unrolled_size << " elements." << std::endl;
#else
        std::cout << "Unsupported platform: Optimal unrolled size calculation not validated." << std::endl;
#endif
    }

    SECTION("Quarter L1 Cache Size") {
        double fraction = 0.25;
        size_t optimal_unrolled_size = compute_optimal_unrolled_size(l1_cache_size, element_size, fraction);

        REQUIRE_MESSAGE(optimal_unrolled_size > 0, "Optimal unrolled size should be greater than 0");
        REQUIRE_MESSAGE(optimal_unrolled_size <= (l1_cache_size / element_size), "Optimal unrolled size should be within quarter L1 cache size range");

        // Calculate expected size
        size_t expected_size = static_cast<size_t>((l1_cache_size * fraction) / element_size);
        CHECK_MESSAGE(optimal_unrolled_size == expected_size, "Optimal unrolled size does not match expected size");

#ifdef _WIN32
        std::cout << "Windows: Optimal unrolled size calculated correctly: " << optimal_unrolled_size << " elements." << std::endl;
#elif defined(__linux__)
        std::cout << "Linux: Optimal unrolled size calculated correctly: " << optimal_unrolled_size << " elements." << std::endl;
#elif defined(__APPLE__)
        std::cout << "macOS: Optimal unrolled size calculated correctly: " << optimal_unrolled_size << " elements." << std::endl;
#else
        std::cout << "Unsupported platform: Optimal unrolled size calculation not validated." << std::endl;
#endif
    }
}

TEST_CASE("Cache line size retrieval", "[cache_line_size]") {
    size_t cache_line_size = get_cache_line_size();

    REQUIRE_MESSAGE(cache_line_size > 0, "Cache line size should be greater than 0");

    std::cout << "Detected Cache Line Size: " << cache_line_size << " bytes" << std::endl;

    // Check the platform-specific cache line size
#ifdef _WIN32
    std::cout << "Running on Windows. Cache line size detected: " << cache_line_size << " bytes." << std::endl;
    CHECK_MESSAGE(cache_line_size > 0, "Windows cache line size detection failed.");
#elif defined(__linux__)
    std::cout << "Running on Linux. Cache line size detected: " << cache_line_size << " bytes." << std::endl;
    CHECK_MESSAGE(cache_line_size > 0, "Linux cache line size detection failed.");
#elif defined(__APPLE__)
    std::cout << "Running on macOS. Cache line size detected: " << cache_line_size << " bytes." << std::endl;
    CHECK_MESSAGE(cache_line_size > 0, "macOS cache line size detection failed.");
#else
    std::cout << "Running on an unsupported platform. Cache line size detection not tested." << std::endl;
#endif
}

TEST_CASE("Optimal unrolled size computation based on cache line size", "[optimal_unrolled_size]") {
    size_t cache_line_size = get_cache_line_size();
    size_t element_size = sizeof(float);

    REQUIRE_MESSAGE(cache_line_size > 0, "Cache line size should be greater than 0 for this test to be valid.");

    SECTION("Using full cache line size") {
        double fraction = 1.0;
        size_t optimal_unrolled_size = compute_optimal_unrolled_size(cache_line_size, element_size, fraction);

        REQUIRE_MESSAGE(optimal_unrolled_size > 0, "Optimal unrolled size should be greater than 0");
        REQUIRE_MESSAGE(optimal_unrolled_size <= (cache_line_size / element_size), "Optimal unrolled size should be within cache line size range");

        // Calculate expected size
        size_t expected_size = static_cast<size_t>((cache_line_size * fraction) / element_size);
        CHECK_MESSAGE(optimal_unrolled_size == expected_size, "Optimal unrolled size does not match expected size");

#ifdef _WIN32
        std::cout << "Windows: Optimal unrolled size calculated correctly: " << optimal_unrolled_size << " elements." << std::endl;
#elif defined(__linux__)
        std::cout << "Linux: Optimal unrolled size calculated correctly: " << optimal_unrolled_size << " elements." << std::endl;
#elif defined(__APPLE__)
        std::cout << "macOS: Optimal unrolled size calculated correctly: " << optimal_unrolled_size << " elements." << std::endl;
#else
        std::cout << "Unsupported platform: Optimal unrolled size calculation not validated." << std::endl;
#endif
    }
}