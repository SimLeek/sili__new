#ifndef __CSR__HPP_
#define __CSR__HPP_

#include "sparse_struct.hpp"

//#include "coo.hpp"
//#include "unique_vector.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <iterator>
#include <limits>
#include <omp.h>
#include <random>

#include <memory>
#include <array>
#include <vector>
#include <type_traits>

template <class SIZE_TYPE>
using CSRPointers = std::array<std::shared_ptr<std::vector<SIZE_TYPE>>, 1>;

template <typename INDEX_ARRAYS>
struct ReduceArraySize {
    using type = std::array<
        typename INDEX_ARRAYS::value_type, 
        std::tuple_size<INDEX_ARRAYS>::value - 1
    >;
};

template <typename INDEX_ARRAYS>
using ReducedArray = typename ReduceArraySize<INDEX_ARRAYS>::type;

template <typename INDEX_ARRAYS>
using stdarr_of_uniqarr_type = typename std::tuple_element<0, INDEX_ARRAYS>::type
    ::element_type   // vector<T>
    ::value_type;    // T
template <int... selections>
struct expand {

template <typename SIZE_TYPE, typename INDEX_ARRAYS, typename VALUE_ARRAYS>
static sparse_struct
<
    SIZE_TYPE,
    COOPointers<SIZE_TYPE>,
    INDEX_ARRAYS,
    std::array<
        std::shared_ptr<std::vector<stdarr_of_uniqarr_type<VALUE_ARRAYS>>>,
        num_indices<VALUE_ARRAYS> + sizeof...(selections)
    >
>
view_coo_values(const sparse_struct<SIZE_TYPE, SIZE_TYPE, INDEX_ARRAYS, VALUE_ARRAYS>& a_coo)
{
    using T = stdarr_of_uniqarr_type<VALUE_ARRAYS>;
    constexpr std::array<int, sizeof...(selections)> selection = {selections...};
    constexpr size_t num_value_indices = num_indices<VALUE_ARRAYS>;
    constexpr size_t num_selections    = sizeof...(selections);

    sparse_struct<
        SIZE_TYPE,
        COOPointers<SIZE_TYPE>,
        INDEX_ARRAYS,
        std::array<
            std::shared_ptr<std::vector<T>>,
            num_value_indices + sizeof...(selections)
        >
    > expanded_coo;

    expanded_coo.rows = a_coo.rows;
    expanded_coo.cols = a_coo.cols;
    expanded_coo.ptrs = a_coo.ptrs;

    // Shared ownership of indices — just copy the shared_ptr
    for (std::size_t idx = 0; idx < num_indices<INDEX_ARRAYS>; ++idx)
        expanded_coo.indices[idx] = a_coo.indices[idx];

    int selections_used = 0;
    for (std::size_t idx = 0; idx < num_value_indices + num_selections; ++idx) {
        const auto true_idx = idx - selections_used;
        const bool is_next_selection = (selections_used < (int)num_selections) &&
                                       (idx == (size_t)selection[selections_used]);
        const bool exhausted_originals = true_idx >= num_value_indices;

        if (is_next_selection || exhausted_originals) {
            // New zero-filled vector — exclusively owned by expanded_coo
            expanded_coo.values[idx] = std::make_shared<std::vector<T>>(
                a_coo.nnz(), T(0));
            ++selections_used;
        } else {
            // Shared ownership — refcount handles lifetime
            expanded_coo.values[idx] = a_coo.values[true_idx];
        }
    }

    return expanded_coo;
}

// free() is a no-op — shared_ptr refcounts handle everything
template <typename SIZE_TYPE, typename INDEX_ARRAYS, typename VALUE_ARRAYS>
static void free(sparse_struct<SIZE_TYPE, SIZE_TYPE, INDEX_ARRAYS, VALUE_ARRAYS>&) {}

};

//warning: the original csr also has the same pointers after this operation.
//warning2: the input coo MUST be coalesced: it must be sorted and have no duplicates.
template <typename SIZE_TYPE, typename INDEX_ARRAYS, typename VALUE_ARRAYS>
sparse_struct<
    SIZE_TYPE,
    CSRPointers<SIZE_TYPE>, // First SIZE_TYPE transformed to CSRPointers
    ReducedArray<INDEX_ARRAYS>, // INDEX_ARRAYS reduced by one
    VALUE_ARRAYS
>to_csr(
    sparse_struct<
        SIZE_TYPE,
        COOPointers<SIZE_TYPE>, // First SIZE_TYPE is unchanged here
        INDEX_ARRAYS,           // INDEX_ARRAYS as provided
        VALUE_ARRAYS
    > &a_coo, 
    const int num_cpus)
{
    SIZE_TYPE nnz = a_coo.ptrs;
    SIZE_TYPE num_rows = a_coo.rows;
    //auto rows = a_coo.indices[0].release();

    if(a_coo.indices[0].get()==nullptr){
        sparse_struct<
        SIZE_TYPE,
        CSRPointers<SIZE_TYPE>, // First SIZE_TYPE transformed to CSRPointers
        ReducedArray<INDEX_ARRAYS>, // INDEX_ARRAYS reduced by one
        VALUE_ARRAYS
        > csr;

        csr.rows = num_rows;
        csr.cols = a_coo.cols;
        csr.ptrs[0].reset(new std::vector<SIZE_TYPE>(num_rows + 1,0));
        for (std::size_t idx = 0; idx < num_indices<INDEX_ARRAYS>-1; ++idx) {
            csr.indices[idx].reset();
        }
        for (std::size_t valIdx = 0; valIdx < num_indices<VALUE_ARRAYS>; ++valIdx) {
            csr.values[valIdx].reset();
        }

        return csr;
    }

    // Allocate accumulators for parallel histogram accumulation
    SIZE_TYPE *accum = new SIZE_TYPE[a_coo.rows]();
    if (num_cpus > 1) {
        SIZE_TYPE *thr_accum = new SIZE_TYPE[num_cpus * a_coo.rows];
        std::fill(thr_accum, thr_accum + num_cpus * a_coo.rows, 0);

        #pragma omp parallel shared(accum, thr_accum, a_coo) num_threads(num_cpus)
        {
            SIZE_TYPE tid = omp_get_thread_num();
            int my_first = tid * a_coo.rows;
            SIZE_TYPE chunk_size = (nnz + num_cpus - 1) / num_cpus;
            SIZE_TYPE start = tid * chunk_size;
            SIZE_TYPE end = std::min(start + chunk_size, nnz);

            for (SIZE_TYPE i = start; i < end; i++) {
                thr_accum[my_first + (*a_coo.indices[0])[i]]++;
            }
            #pragma omp barrier

            #pragma omp for
            for (SIZE_TYPE r = 0; r < a_coo.rows; r++) {
                for (int t = 0; t < num_cpus; t++) {
                    accum[r] += thr_accum[t * a_coo.rows + r];
                }
            }
        }

        delete[] thr_accum;
    } else {
        for (SIZE_TYPE i = 0; i < nnz; i++) {
            accum[(*a_coo.indices[0])[i]]++;
        }
    }

    std::vector<SIZE_TYPE> *ptrs = new std::vector<SIZE_TYPE>(num_rows + 1);
    SIZE_TYPE scan_a = 0;

    // Parallel scan to compute row pointers
    #pragma omp parallel for simd reduction(inscan, + : scan_a)
    for (SIZE_TYPE i = 0; i <= num_rows; i++) {
        (*ptrs)[i] = scan_a;
        #pragma omp scan exclusive(scan_a)
        {
            scan_a += accum[i];
        }
    }

    delete[] accum;
    a_coo.indices[0].reset();
    // Create and return the CSR sparse structure
    sparse_struct<
    SIZE_TYPE,
    CSRPointers<SIZE_TYPE>, // First SIZE_TYPE transformed to CSRPointers
    ReducedArray<INDEX_ARRAYS>, // INDEX_ARRAYS reduced by one
    VALUE_ARRAYS
    > csr;

    a_coo.ptrs = 0;

    csr.rows = num_rows;
    csr.cols = a_coo.cols;
    csr.ptrs[0].reset(ptrs);
    a_coo.indices[0].reset();
    for (std::size_t idx = 0; idx < num_indices<INDEX_ARRAYS>-1; ++idx) {
        csr.indices[idx] = std::move(a_coo.indices[idx+1]);
        a_coo.indices[idx+1].reset();
    }
    for (std::size_t valIdx = 0; valIdx < num_indices<VALUE_ARRAYS>; ++valIdx) {
        csr.values[valIdx] = std::move(a_coo.values[valIdx]);
        a_coo.values[valIdx].reset();
    }

    return csr;
}

template <class SIZE_TYPE, class PTRS, class INDICES, class VALUES>
void clear_csr(sparse_struct<SIZE_TYPE, PTRS, INDICES, VALUES>& csr) {
    // Clear pointers array
    for (auto& ptr : csr.ptrs) {
        ptr.reset();
    }

    // Clear indices array
    for (auto& index : csr.indices) {
        index.reset();
    }

    // Clear values array
    for (auto& value : csr.values) {
        value.reset();
    }

    // Set rows and columns to zero
    csr.rows = 0;
    csr.cols = 0;
}

#endif