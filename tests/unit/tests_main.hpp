#ifndef _TEST_MAIN_H__
#define _TEST_MAIN_H__

/**
 * @file
 * @brief Testing utilities for vector and CSR comparisons.
 *
 * This header provides utilities for testing, including macros and functions for
 * comparing vectors and CSR (Compressed Sparse Row) structures, as well as helpers
 * for printing and diffing these data structures. Designed for use with the Catch2
 * testing framework.
 */

#include "csr.hpp"
#include <cstddef>
#include <limits>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <vector>

// thanks: https://github.com/catchorg/Catch2/issues/929#issuecomment-308663820
/**
 * @def REQUIRE_MESSAGE(cond, msg)
 * @brief Asserts a condition with a custom message, halting the test on failure.
 * @param cond The condition to check.
 * @param msg The message to display if the condition fails.
 */
#define REQUIRE_MESSAGE(cond, msg)                                                                                     \
    do {                                                                                                               \
        INFO(msg);                                                                                                     \
        REQUIRE(cond);                                                                                                 \
    } while ((void)0, 0)
/**
 * @def CHECK_MESSAGE(cond, msg)
 * @brief Checks a condition with a custom message, continuing the test on failure.
 * @param cond The condition to check.
 * @param msg The message to display if the condition fails.
 */
#define CHECK_MESSAGE(cond, msg)                                                                                       \
    do {                                                                                                               \
        INFO(msg);                                                                                                     \
        CHECK(cond);                                                                                                   \
    } while ((void)0, 0)

/**
 * @brief Checks if a type is a supported vector type.
 *
 * Determines if a type is `std::vector<T>`, inheriting
 * from `std::true_type` if supported, `std::false_type` otherwise.
 *
 * @tparam T The type to check.
 */
template <typename T> struct is_supported_vector : std::false_type {};

/**
 * @brief Specialization for std::vector.
 */
template <typename T> struct is_supported_vector<std::vector<T>> : std::true_type {};

/**
 * @brief Overloads for streaming various types to ostream.
 *
 * The following allows us to print pairs and vectors
 * thanks: https://stackoverflow.com/a/6245777
 */

/** @cond INTERNAL */
namespace aux {
template <std::size_t...> struct seq {};

template <std::size_t N, std::size_t... Is> struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

template <std::size_t... Is> struct gen_seq<0, Is...> : seq<Is...> {};

template <class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple(std::basic_ostream<Ch, Tr> &os, Tuple const &t, seq<Is...>) {
    using swallow = int[];
    (void)swallow{0, (void(os << (Is == 0 ? "" : ", ") << std::get<Is>(t)), 0)...};
}
} // namespace aux
/** @endcond */

// forward declarations
template <class Ch, class Tr, class T> std::ostream &operator<<(std::basic_ostream<Ch, Tr> &o, const std::vector<T> &p);
template <class Ch, class Tr, class S, class T>
std::ostream &operator<<(std::basic_ostream<Ch, Tr> &o, const std::pair<S, T> &p);

/**
 * @brief Streams a tuple to an ostream.
 *
 * @tparam Ch Character type of the stream.
 * @tparam Tr Traits type of the stream.
 * @tparam Args Types of the tuple elements.
 * @param os The output stream.
 * @param t The tuple to stream.
 * @return The output stream.
 */
template <class Ch, class Tr, class... Args>
auto operator<<(std::basic_ostream<Ch, Tr> &os, std::tuple<Args...> const &t) -> std::basic_ostream<Ch, Tr> & {
    os << "(";
    aux::print_tuple(os, t, aux::gen_seq<sizeof...(Args)>());
    return os << ")";
}

/**
 * @brief Streams a pair to an ostream.
 *
 * @tparam Ch Character type of the stream.
 * @tparam Tr Traits type of the stream.
 * @tparam S First type of the pair.
 * @tparam T Second type of the pair.
 * @param o The output stream.
 * @param p The pair to stream.
 * @return The output stream.
 */
template <class Ch, class Tr, class S, class T>
std::ostream &operator<<(std::basic_ostream<Ch, Tr> &o, const std::pair<S, T> &p) {
    return o << "(" << p.first << ", " << p.second << ")";
}

/**
 * @brief Streams a std::vector to an ostream.
 *
 * @tparam Ch Character type of the stream.
 * @tparam Tr Traits type of the stream.
 * @tparam T Type of the vector elements.
 * @param o The output stream.
 * @param p The vector to stream.
 * @return The output stream.
 */
template <class Ch, class Tr, class T>
std::ostream &operator<<(std::basic_ostream<Ch, Tr> &o, const std::vector<T> &p) {
    o << "{ ";
    for (auto v : p) {
        o << v << " ";
    }
    o << "}";
    return o;
}

/**
 * @brief Computes the symmetric difference between two vectors.
 *
 * Calculates the symmetric difference between sorted vectors `a` and `b`, storing
 * the result in `diff`.
 *
 * @tparam VecA Type of the first vector.
 * @tparam VecB Type of the second vector.
 * @param a The first vector.
 * @param b The second vector.
 * @param diff The vector to store the difference (default is empty VecA).
 * @return Reference to the diff vector.
 */
template <typename VecA, typename VecB>
inline typename std::enable_if<is_supported_vector<VecA>::value && is_supported_vector<VecB>::value, VecA>::type
vector_diff(const VecA &a, const VecB &b, VecA diff=VecA()) {
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
    return diff;
}

/**
 * @brief Extracts non-zero elements and their indices from a vector.
 *
 * Iterates through the input vector and collects indices and values of non-zero
 * elements based on the provided epsilon.
 *
 * @tparam T Type of the vector elements.
 * @param d The input vector.
 * @param nonzero_indices Vector to store indices of non-zero elements.
 * @param nonzero_values Vector to store values of non-zero elements.
 * @param epsilon Tolerance for non-zero check (default: numeric_limits<T>::epsilon()).
 */
template <typename T>
void getNonzeroIndicesAndValues(const std::vector<T> &d,
                                std::vector<size_t> &nonzero_indices,
                                std::vector<T> &nonzero_values,
                                double epsilon = std::numeric_limits<T>::epsilon()) {
    for (size_t i = 0; i < d.size(); ++i) {
        if constexpr (std::is_signed<T>::value) {
            if (std::abs(d[i]) > epsilon) {
                nonzero_indices.push_back(i);
                nonzero_values.push_back(d[i]);
            }
        }else{
            if (d[i] > epsilon) {
                nonzero_indices.push_back(i);
                nonzero_values.push_back(d[i]);
            }
        }
    }
}

/**
 * @brief Generates a string describing the difference between two vectors.
 *
 * Compares vectors `a` and `b`, returning a string with their differences. If sizes
 * match, it reports element-wise differences exceeding `epsilon`; otherwise, it
 * computes the symmetric difference.
 *
 * @tparam VecA Type of the first vector.
 * @tparam VecB Type of the second vector.
 * @tparam T Type of vector elements (default: VecA::value_type).
 * @param a The first vector.
 * @param b The second vector.
 * @param epsilon Tolerance for comparisons (default: numeric_limits<T>::epsilon()).
 * @return String describing the differences.
 */
template <typename VecA, typename VecB, typename T = typename VecA::value_type>
typename std::enable_if<is_supported_vector<VecA>::value && is_supported_vector<VecB>::value, std::string>::type vector_diff_string(
    const VecA &a,
    const VecB &b,
    double epsilon = std::numeric_limits<T>::epsilon()) {
    std::ostringstream oss;
    oss << "Vector A: " << a << "\n";
    oss << "Vector B: " << b << "\n";
    if (a.size() == b.size()) {
        std::vector<T> d(a.size());
        std::transform(a.begin(), a.end(), b.begin(), d.begin(), std::minus<T>());
        std::vector<size_t> nonzero_indices;
        std::vector<T> nonzero_values;
        getNonzeroIndicesAndValues(d, nonzero_indices, nonzero_values, epsilon);
        oss << "Diff indices: " << nonzero_indices;
        oss << "Diff values: " << nonzero_values;
    } else {
        std::vector<T> diff;
        std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
        oss << "Diff: " << vector_diff(a, b);
    }
    return oss.str();
}

/**
 * @brief Checks if two vectors are almost equal within a tolerance.
 *
 * Compares vectors element-wise, returning true if all elements are within `epsilon`.
 *
 * @tparam T Type of the first vector's elements.
 * @tparam U Type of the second vector's elements.
 * @param a The first vector.
 * @param b The second vector.
 * @param epsilon Tolerance (default: numeric_limits<T>::epsilon()).
 * @return True if vectors are almost equal, false otherwise.
 */
template <class T, class U>
bool almost_equal(const std::vector<T> &a,
                  const std::vector<U> &b,
                  double epsilon = std::numeric_limits<T>::epsilon()) {
    // Check if both vectors are of the same size
    if (a.size() != b.size()) {
        return false; // Vectors are not the same size, so they can't be equal
    }

    // Compare each element within the specified epsilon tolerance
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > epsilon) {
            return false; // The difference between elements exceeds the epsilon tolerance
        }
    }

    return true; // All elements are approximately equal within the epsilon tolerance
}

/**
 * @brief Converts an array to a std::vector.
 *
 * Creates a std::vector from a raw array.
 *
 * @tparam T Type of array elements.
 * @param arr Pointer to the array.
 * @param size Size of the array.
 * @return A std::vector with the array elements.
 */
template <typename T> std::vector<T> vec(T *arr, size_t size) { return std::vector<T>(arr, arr + size); }

/**
 * @defgroup VectorMacros Vector Comparison Macros
 * @brief Macros for checking vector equality in tests.
 *
 * @{
 */

/**
 * @def CHECK_VECTOR_EQUAL(a, b)
 * @brief Checks exact equality of two vectors.
 * @param a First vector.
 * @param b Second vector.
 */

/**
 * @def CHECK_VECTOR_ALMOST_EQUAL(a, b, [epsilon])
 * @brief Checks if two vectors are almost equal within a tolerance.
 * @param a First vector.
 * @param b Second vector.
 * @param epsilon Optional tolerance (default: type epsilon).
 */

/**
 * @def REQUIRE_VECTOR_EQUAL(a, b)
 * @brief Requires exact equality of two vectors.
 * @param a First vector.
 * @param b Second vector.
 */

/**
 * @def CHECK_NESTED_VECTOR_EQUAL(a, b)
 * @brief Checks equality of nested vectors.
 * @param a First nested vector.
 * @param b Second nested vector.
 */

/**
 * @def REQUIRE_NESTED_VECTOR_EQUAL(a, b)
 * @brief Requires equality of nested vectors.
 * @param a First nested vector.
 * @param b Second nested vector.
 */

/** @} */

#define CHECK_VECTOR_EQUAL(a, b)                                                                                       \
    do {                                                                                                               \
        INFO(vector_diff_string(a, b, 0));                                                                             \
        bool success = a == b;                                                                                         \
        CHECK(success);                                                                                                \
    } while ((void)0, 0)

#define CHECK_VECTOR_ALMOST_EQUAL_3(a, b, e)                                                                           \
    do {                                                                                                               \
        INFO(vector_diff_string(a, b, e));                                                                             \
        bool success = almost_equal(a, b, e);                                                                          \
        CHECK(success);                                                                                                \
    } while ((void)0, 0)

#define CHECK_VECTOR_ALMOST_EQUAL_2(a, b)                                                                              \
    do {                                                                                                               \
        INFO(vector_diff_string(a, b));                                                                                \
        bool success = almost_equal(a, b);                                                                             \
        CHECK(success);                                                                                                \
    } while ((void)0, 0)

#define CHECK_VECTOR_ALMOST_EQUAL_x(x, a, b, e, FUNC, ...) FUNC

#define CHECK_VECTOR_ALMOST_EQUAL(...)                                                                                 \
    CHECK_VECTOR_ALMOST_EQUAL_x(                                                                                       \
        , ##__VA_ARGS__, CHECK_VECTOR_ALMOST_EQUAL_3(__VA_ARGS__), CHECK_VECTOR_ALMOST_EQUAL_2(__VA_ARGS__), )

#define REQUIRE_VECTOR_EQUAL(a, b)                                                                                     \
    do {                                                                                                               \
        INFO(vector_diff_string(a, b));                                                                                \
        bool success = a == b;                                                                                         \
        REQUIRE(success);                                                                                              \
    } while ((void)0, 0)

#define CHECK_NESTED_VECTOR_EQUAL(a, b)                                                                                \
    do {                                                                                                               \
        REQUIRE(a.size() == b.size());                                                                                 \
        for (size_t i = 0; i < a.size(); ++i) {                                                                        \
            INFO("Mismatch at index " << i);                                                                           \
            CHECK_VECTOR_EQUAL(a[i], b[i]);                                                                            \
        }                                                                                                              \
    } while ((void)0, 0)

#define REQUIRE_NESTED_VECTOR_EQUAL(a, b)                                                                              \
    do {                                                                                                               \
        REQUIRE(a.size() == b.size());                                                                                 \
        for (size_t i = 0; i < a.size(); ++i) {                                                                        \
            INFO("Mismatch at index " << i);                                                                           \
            REQUIRE_VECTOR_EQUAL(a[i], b[i]);                                                                          \
        }                                                                                                              \
    } while ((void)0, 0)

/**
 * @brief Generates a string describing differences in a CSR structure.
 *
 * Compares a CSR structure with expected values, detailing discrepancies.
 *
 * @tparam T Type of CSR values.
 * @param csr The CSR structure.
 * @param expected_ptrs Expected pointers.
 * @param expected_indices Expected indices.
 * @param expected_values Expected values.
 * @param expected_rows Expected row count.
 * @param expected_cols Expected column count.
 * @param epsilon Tolerance (default: numeric_limits<T>::epsilon()).
 * @return String with difference details.
 */
template <typename T>
std::string csr_diff_string(const sparse_struct<size_t, CSRPointers<size_t>, CSRIndices<size_t>, UnaryValues<T>>& csr,
                            const std::vector<size_t>& expected_ptrs,
                            const std::vector<size_t>& expected_indices,
                            const std::vector<T>& expected_values,
                            size_t expected_rows,
                            size_t expected_cols,
                            double epsilon = std::numeric_limits<T>::epsilon()) {
    std::ostringstream oss;
    oss << "CSR rows: " << csr.rows << ", Expected rows: " << expected_rows << "\n";
    oss << "CSR cols: " << csr.cols << ", Expected cols: " << expected_cols << "\n";
    std::vector<size_t> ptrs_vec(csr.ptrs[0].get(), csr.ptrs[0].get() + csr.rows + 1);
    oss << "CSR ptrs: " << ptrs_vec << ", Expected ptrs: " << expected_ptrs << "\n";
    size_t nnz = csr.nnz();
    std::vector<size_t> indices_vec(csr.indices[0].get(), csr.indices[0].get() + nnz);
    oss << "CSR indices: " << indices_vec << ", Expected indices: " << expected_indices << "\n";
    std::vector<T> values_vec(csr.values[0].get(), csr.values[0].get() + nnz);
    oss << "CSR values: " << values_vec << ", Expected values: " << expected_values << "\n";
    return oss.str();
}

/**
 * @brief Checks if a CSR structure matches expected values within tolerance.
 *
 * Compares CSR dimensions, pointers, indices, and values with expected ones.
 *
 * @tparam T Type of CSR values.
 * @param csr The CSR structure.
 * @param expected_ptrs Expected pointers.
 * @param expected_indices Expected indices.
 * @param expected_values Expected values.
 * @param expected_rows Expected row count.
 * @param expected_cols Expected column count.
 * @param epsilon Tolerance (default: numeric_limits<T>::epsilon()).
 * @return True if matches within tolerance, false otherwise.
 */
template <typename T>
bool csr_almost_equal(const sparse_struct<size_t, CSRPointers<size_t>, CSRIndices<size_t>, UnaryValues<T>>& csr,
                        const std::vector<size_t>& expected_ptrs,
                        const std::vector<size_t>& expected_indices,
                        const std::vector<T>& expected_values,
                        size_t expected_rows,
                        size_t expected_cols,
                        double epsilon = std::numeric_limits<T>::epsilon()) {
    if (csr.rows != expected_rows || csr.cols != expected_cols) {
        return false;
    }
    std::vector<size_t> ptrs_vec(csr.ptrs[0].get(), csr.ptrs[0].get() + csr.rows + 1);
    if (ptrs_vec != expected_ptrs) {
        return false;
    }
    size_t nnz = csr.nnz();
    if (nnz != expected_indices.size() || nnz != expected_values.size()) {
        return false;
    }
    std::vector<size_t> indices_vec(csr.indices[0].get(), csr.indices[0].get() + nnz);
    if (indices_vec != expected_indices) {
        return false;
    }
    std::vector<T> values_vec(csr.values[0].get(), csr.values[0].get() + nnz);
    if constexpr (std::is_floating_point<T>::value) {
        return almost_equal(values_vec, expected_values, epsilon);
    } else {
        return values_vec == expected_values;
    }
}

/**
 * @defgroup CSRMacros CSR Comparison Macros
 * @brief Macros for checking CSR structure equality in tests.
 *
 * @{
 */

/**
 * @def CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols)
 * @brief Checks exact equality of a CSR structure.
 * @param csr The CSR structure.
 * @param expected_ptrs Expected pointers.
 * @param expected_indices Expected indices.
 * @param expected_values Expected values.
 * @param expected_rows Expected row count.
 * @param expected_cols Expected column count.
 */

/**
 * @def CHECK_CSR_ALMOST_EQUAL(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, [epsilon])
 * @brief Checks if a CSR structure is almost equal within tolerance.
 * @param csr The CSR structure.
 * @param expected_ptrs Expected pointers.
 * @param expected_indices Expected indices.
 * @param expected_values Expected values.
 * @param expected_rows Expected row count.
 * @param expected_cols Expected column count.
 * @param epsilon Optional tolerance (default: type epsilon).
 */

/** @} */

#define CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols) \
    do { \
        INFO(csr_diff_string(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, 0)); \
        bool success = csr_almost_equal(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, 0); \
        CHECK(success); \
    } while ((void)0, 0)

#define CHECK_CSR_ALMOST_EQUAL_3(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, epsilon) \
    do { \
        INFO(csr_diff_string(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, epsilon)); \
        bool success = csr_almost_equal(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, epsilon); \
        CHECK(success); \
    } while ((void)0, 0)

#define CHECK_CSR_ALMOST_EQUAL_2(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols) \
    do { \
        using ValueType = std::remove_reference_t<decltype(csr.values[0][0])>; \
        INFO(csr_diff_string(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, std::numeric_limits<ValueType>::epsilon())); \
        bool success = csr_almost_equal(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, std::numeric_limits<ValueType>::epsilon()); \
        CHECK(success); \
    } while ((void)0, 0)

 #define CHECK_CSR_ALMOST_EQUAL_x(x, csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, epsilon, FUNC, ...) FUNC

#define CHECK_CSR_ALMOST_EQUAL(...) \
        CHECK_CSR_ALMOST_EQUAL_x(, ##__VA_ARGS__, CHECK_CSR_ALMOST_EQUAL_3(__VA_ARGS__), CHECK_CSR_ALMOST_EQUAL_2(__VA_ARGS__))

#endif