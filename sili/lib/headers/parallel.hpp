#ifndef _PARALLEL_HPP
#define _PARALLEL_HPP

#include "csr.hpp"
#include "sparse_struct.hpp"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>
#include <omp.h>
#include <vector>
#include <cstring> // for memset

/**
 * @brief Comparator class for less-than comparison between two types.
 *
 * @tparam A First type to compare.
 * @tparam B Second type to compare.
 */
template <class A, class B> class ComparatorLT {
  public:
    /**
     * @brief Compares two values using less-than.
     * @param a First value.
     * @param b Second value.
     * @return true if a < b, false otherwise.
     */
    bool operator()(A a, B b) const { return (a < b); }
};

/**
 * @brief Comparator class for greater-than comparison between two types.
 *
 * @tparam A First type to compare.
 * @tparam B Second type to compare.
 */
template <class A, class B> class ComparatorGT {
  public:
    /**
     * @brief Compares two values using greater-than.
     * @param a First value.
     * @param b Second value.
     * @return true if a > b, false otherwise.
     */
    bool operator()(A a, B b) const { return (a > b); }
};

/**
 * @brief Comparator for sorting indices based on values in a vector.
 *
 * @tparam T Type of elements in the vector.
 * @tparam Compare Comparator type for ordering elements.
 */
template <typename VEC, typename Compare> class PermutationComparator {
    const VEC &a; ///< Reference to the vector being sorted.
    Compare cmp;             ///< Comparator instance.
  public:
    /**
     * @brief Constructs the comparator with a vector and comparison function.
     * @param a Vector whose values determine the order.
     * @param cmp Comparator to use for ordering.
     */
    PermutationComparator(const VEC &a, Compare cmp) : a(a), cmp(cmp) {}

    /**
     * @brief Compares two indices based on their corresponding values.
     * @param i First index.
     * @param j Second index.
     * @return true if a[i] precedes a[j] according to cmp, false otherwise.
     */
    bool operator()(std::size_t i, std::size_t j) const { return cmp(a[i], a[j]); }
};

/**
 * @brief Comparator for sorting indices based on values in a vector.
 *
 * @tparam T Type of elements in the vector.
 * @tparam Compare Comparator type for ordering elements.
 */
 template <typename A, typename B, typename Compare> class AbsComparator {
    Compare cmp;             ///< Comparator instance.
  public:
    /**
     * @brief Constructs the comparator with a vector and comparison function.
     * @param cmp Comparator to use for ordering.
     */
    AbsComparator(Compare cmp) : cmp(cmp) {}

    /**
     * @brief Compares two indices based on their corresponding values.
     * @param a First index.
     * @param b Second index.
     * @return true if a[i] precedes a[j] according to cmp, false otherwise.
     */
    bool operator()(A a, B b) const { return cmp(std::abs(a), std::abs(b)); }
};

/**
 * @brief Applies a permutation to a vector in parallel.
 *
 * @tparam T Type of elements in the vector.
 * @param p Permutation vector.
 * @param vec Vector to permute in-place.
 */
template <typename S, typename T> void omp_apply_permutation_vector_parallel(const std::vector<S> &p, std::vector<T> &vec) {
    std::vector<T> temp(vec.size());
#pragma omp parallel for
    for (size_t i = 0; i < p.size(); ++i) {
        temp[i] = vec[p[i]];
    }
    vec = std::move(temp);
}

template <class S, typename T> void omp_apply_permutation_array_parallel(const S num, const std::vector<std::size_t> &p, T* arr) {
    std::vector<T> temp(num);
#pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
        temp[i] = arr[p[i]];
    }
#pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
        arr[i] = temp[i];  // parallel copy instead of memcpy... probably slower
    }
}

/**
 * @brief Applies a permutation to multiple vectors in parallel.
 *
 * @tparam Vectors Variadic template for vector types.
 * @param p Permutation vector.
 * @param vectors References to vectors to permute.
 */
template <typename PermVector, typename... Vectors>
void apply_permutation_to_all_vector_parallel(const PermVector &p, Vectors &...vectors) {
    (omp_apply_permutation_vector_parallel(p, vectors), ...);
}

template <class S, typename... ARRAY_TYPES>
void apply_permutation_to_all_array_parallel(const S num, const std::vector<std::size_t> &p, ARRAY_TYPES* ...arrays) {
    (omp_apply_permutation_array_parallel(num, p, arrays), ...);
}

/**
 * @brief Recursive helper function for parallel merge sort using OpenMP.
 *
 * @tparam TYPE Type of elements in the vector.
 * @tparam COMPARE Comparator type, defaults to less-than.
 * @param v Vector to sort.
 * @param left Left boundary of the current segment.
 * @param right Right boundary of the current segment.
 */
template <class TYPE, class COMPARE = ComparatorLT<TYPE, TYPE>>
void _omp_merge_sort_vector_recursive(std::vector<TYPE> &v, unsigned long left, unsigned long right, const COMPARE& cmp=COMPARE()) {
    if (left < right) {
        if (right - left >= 32) {
            unsigned long mid = (left + right) / 2;
#pragma omp taskgroup
            {
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                _omp_merge_sort_vector_recursive(v, left, mid);
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                _omp_merge_sort_vector_recursive(v, mid + 1, right);
//#pragma omp taskyield
            }
            std::inplace_merge(v.begin() + left, v.begin() + mid + 1, v.begin() + right + 1, cmp);
        } else {
            std::sort(v.begin() + left, v.begin() + right + 1, cmp);
        }
    }
}

template <class S, class CONTAINER, class COMPARE = ComparatorLT<typename CONTAINER::value_type, typename CONTAINER::value_type>>
void _omp_merge_sort_array_recursive(CONTAINER &v, S left, S right, const COMPARE& cmp=COMPARE()) {
    if (left < right) {
        if (right - left >= 32) {
            S mid = (left + right) / 2;
#pragma omp taskgroup
            {
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                _omp_merge_sort_array_recursive(v, left, mid);
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                _omp_merge_sort_array_recursive(v, mid + 1, right);
//#pragma omp taskyield
            }
            std::inplace_merge(v + left, v + mid + 1, v + right + 1, cmp);
        } else {
            std::sort(v + left, v + right + 1, cmp);
        }
    }
}

template <class TYPE, class COMPARE = ComparatorLT<TYPE, TYPE>> 
void omp_merge_sort_vector(std::vector<TYPE> &v, const COMPARE& cmp=COMPARE()) {
    // this handles v.size==0, so v.size-1 == unsigned long max
    size_t max_val;
    if(v.size()==0){
        max_val = 0;
    }else{
        max_val = v.size()-1;
    }
#pragma omp parallel
#pragma omp single
    _omp_merge_sort_vector_recursive<TYPE, COMPARE>(v, 0, max_val, cmp);
}

template <class S, class CONTAINER, class COMPARE = ComparatorLT<typename CONTAINER::value_type, typename CONTAINER::value_type>> 
void omp_merge_sort_array(S num, CONTAINER &v, const COMPARE& cmp=COMPARE()) {
    // this handles v.size==0, so v.size-1 == unsigned long max
    S max_val;
    if(num==0){
        max_val = 0;
    }else{
        max_val = num-1;
    }
#pragma omp parallel
#pragma omp single
    _omp_merge_sort_array_recursive<S, CONTAINER, COMPARE>(v, 0, max_val, cmp);
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void sort_indices(
    CSRInput<SIZE_TYPE, VALUE_TYPE>& csr,
    const int num_cpus = 4)
{
    const SIZE_TYPE rows      = csr.rows;
    const SIZE_TYPE total_nnz = (*csr.ptrs[0])[rows];

    if (total_nnz == 0) return;

    // Build flat row_id array from ptrs
    std::vector<SIZE_TYPE> row_id(total_nnz);
    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE r = 0; r < rows; ++r)
        for (SIZE_TYPE i = (*csr.ptrs[0])[r]; i < (*csr.ptrs[0])[r + 1]; ++i)
            row_id[i] = r;

    // Sort permutation by (row_id asc, col_idx asc)
    // PermutationComparator only handles a single array, so we wrap a compound key
    struct RowColCmp {
        const SIZE_TYPE* row_id;
        const SIZE_TYPE* col_idx;
        bool operator()(SIZE_TYPE a, SIZE_TYPE b) const {
            if (row_id[a] != row_id[b]) return row_id[a] < row_id[b];
            return col_idx[a] < col_idx[b];
        }
    };

    std::vector<SIZE_TYPE> perm(total_nnz);
    std::iota(perm.begin(), perm.end(), SIZE_TYPE(0));
    RowColCmp row_col_cmp{row_id.data(), csr.indices[0].get()->data()};
    omp_merge_sort_vector(perm, PermutationComparator<SIZE_TYPE*, RowColCmp>(perm.data(), row_col_cmp));

    // Apply permutation to both indices and values
    apply_permutation_to_all_vector_parallel(
        perm,
        *(csr.indices[0].get()),
        *(csr.values[0].get()));
}


template <typename SIZE_TYPE, typename VALUE_TYPE, typename Compare = ComparatorGT<VALUE_TYPE, VALUE_TYPE>>
CSRInput<SIZE_TYPE, VALUE_TYPE> top_k(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& tensor,
    const SIZE_TYPE k,
    const bool use_abs = true,
    const int num_cpus = 4,
    Compare cmp = Compare())
{
    const SIZE_TYPE rows      = tensor.rows;
    const SIZE_TYPE total_nnz = tensor.nnz();

    // ── Pass 1: output row lengths ────────────────────────────────────────────
    CSRInput<SIZE_TYPE, VALUE_TYPE> out;
    out.rows = rows;
    out.cols = tensor.cols;
    out.ptrs[0].reset(new std::vector<SIZE_TYPE>(rows + 1));
    (*out.ptrs[0])[0] = 0;
    for (SIZE_TYPE r = 0; r < rows; ++r) {
        SIZE_TYPE nnz = (*tensor.ptrs[0])[r + 1] - (*tensor.ptrs[0])[r];
        (*out.ptrs[0])[r + 1] = (*out.ptrs[0])[r] + std::min(nnz, k);
    }

    const SIZE_TYPE total_out = out.nnz();
    out.indices[0].reset(new std::vector<SIZE_TYPE>(total_out));
    out.values[0].reset(new std::vector<VALUE_TYPE>(total_out));

    if (total_out == 0) return out;

    // ── Pass 2a: flat row_id array ────────────────────────────────────────────
    std::vector<SIZE_TYPE> row_id(total_nnz);
    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE r = 0; r < rows; ++r)
        for (SIZE_TYPE i = (*tensor.ptrs[0])[r]; i < (*tensor.ptrs[0])[r + 1]; ++i)
            row_id[i] = r;

    // ── Pass 2b: sort perm by (row_id asc, |value| desc) ─────────────────────
    struct RowMagCmp {
        const SIZE_TYPE*   row_id;
        const VALUE_TYPE*  values;
        bool               use_abs;
        bool operator()(SIZE_TYPE a, SIZE_TYPE b) const {
            if (row_id[a] != row_id[b]) return row_id[a] < row_id[b];
            const VALUE_TYPE va = use_abs ? std::abs(values[a]) : values[a];
            const VALUE_TYPE vb = use_abs ? std::abs(values[b]) : values[b];
            return va > vb;
        }
    };

    std::vector<SIZE_TYPE> perm(total_nnz);
    std::iota(perm.begin(), perm.end(), SIZE_TYPE(0));
    RowMagCmp row_mag_cmp{row_id.data(), tensor.values[0].get()->data(), use_abs};
    omp_merge_sort_vector(perm, PermutationComparator<SIZE_TYPE*, RowMagCmp>(perm.data(), row_mag_cmp));

    // ── Pass 2c: flat scatter — keep rank < k per row ─────────────────────────
    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE i = 0; i < total_nnz; ++i) {
        const SIZE_TYPE orig = perm[i];
        const SIZE_TYPE r    = row_id[orig];
        const SIZE_TYPE rank = i - (*tensor.ptrs[0])[r];
        if (rank >= k) continue;
        const SIZE_TYPE out_pos     = (*out.ptrs[0])[r] + rank;
        (*out.indices[0])[out_pos]  = (*tensor.indices[0])[orig];
        (*out.values[0]) [out_pos]  = (*tensor.values[0]) [orig];
    }

    // ── Pass 3: restore column order within each row ──────────────────────────
    sort_indices(out, num_cpus);

    return out;
}

#endif