#ifndef __CSR__HPP_
#define __CSR__HPP_

#include "sparse_struct.hpp"

#include "coo.hpp"
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


template <typename SIZE_TYPE, typename VALUE_TYPE>
CSRInput<SIZE_TYPE, VALUE_TYPE> make_csr_input(
    SIZE_TYPE rows, SIZE_TYPE cols,
    std::vector<SIZE_TYPE>  ptrs,
    std::vector<SIZE_TYPE>  indices,
    std::vector<VALUE_TYPE> values)
{
    CSRInput<SIZE_TYPE, VALUE_TYPE> t;
    t.rows       = rows;
    t.cols       = cols;
    t.ptrs[0]    = std::make_shared<std::vector<SIZE_TYPE>> (std::move(ptrs));
    t.indices[0] = std::make_shared<std::vector<SIZE_TYPE>> (std::move(indices));
    t.values[0]  = std::make_shared<std::vector<VALUE_TYPE>>(std::move(values));
    return t;
}

// ptrs:       rows+1 entries
// indices / values / grads / importance: nnz entries each
template <typename SIZE_TYPE, typename VALUE_TYPE>
SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> make_weights(
    SIZE_TYPE rows, SIZE_TYPE cols,
    std::vector<SIZE_TYPE>  ptrs,
    std::vector<SIZE_TYPE>  indices,
    std::vector<VALUE_TYPE> values,
    std::vector<VALUE_TYPE> grads,
    std::vector<VALUE_TYPE> importance)
{
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> w;
    w.connections.rows       = rows;
    w.connections.cols       = cols;
    w.connections.ptrs[0]    = std::make_shared<std::vector<SIZE_TYPE>> (std::move(ptrs));
    w.connections.indices[0] = std::make_shared<std::vector<SIZE_TYPE>> (std::move(indices));
    w.connections.values[0]  = std::make_shared<std::vector<VALUE_TYPE>>(std::move(values));
    w.connections.values[1]  = std::make_shared<std::vector<VALUE_TYPE>>(std::move(grads));
    w.connections.values[2]  = std::make_shared<std::vector<VALUE_TYPE>>(std::move(importance));
    w.probes.rows = rows;
    w.probes.cols = cols;
    w.out_degree.assign(cols, SIZE_TYPE(0));
    for (const auto& idx : *w.connections.indices[0])
        w.out_degree[idx]++;
    return w;
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
std::vector<SIZE_TYPE> top_k_indices_biased(VALUE_TYPE *values, CSRInput<SIZE_TYPE,  VALUE_TYPE>& bias, size_t size, size_t k, int num_threads) {    
    // Each thread processes a chunk of the array
    size_t chunk_size = (size + num_threads - 1) / num_threads;
    std::vector<std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>>> thread_pairs(num_threads);

    if(k>size){
        k=size;
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        SIZE_TYPE start = thread_id * chunk_size;
        SIZE_TYPE end = std::min(start + chunk_size, size);

        SIZE_TYPE bias_ptr = bias.ptrs[0][start/bias.cols];  // start at the correct row

        // Collect indices for this thread
        std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> local_pairs;
        for (size_t i = start; i < end; ++i) {
            SIZE_TYPE bias_row = i/bias.cols;
            while (bias.indices[0][bias_ptr] < i%bias.cols && bias_ptr<bias.ptrs[0][bias_row+1]) {
                ++bias_ptr;
            }
            if (bias.indices[0][bias_ptr] == i%bias.cols && bias_ptr<=bias.ptrs[0][bias_row+1]) {
                local_pairs.emplace_back(i, bias.values[0][bias_ptr] + values[i]);
                ++bias_ptr;
            }else{
                local_pairs.emplace_back(i, values[i]);
            }
        }

        // Sort local indices by values
        std::partial_sort(local_pairs.begin(), local_pairs.begin() + std::min(k, local_pairs.size()), local_pairs.end(),
                          [](std::pair<SIZE_TYPE, VALUE_TYPE>& a, std::pair<SIZE_TYPE, VALUE_TYPE>& b) { return a.second > b.second; });

        // Keep only the smallest k elements
        if (local_pairs.size() > k) {
            local_pairs.resize(k);
        }

        thread_pairs[thread_id] = std::move(local_pairs);
    }

    // Merge results from all threads
    std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> merged_pairs;
    for (const auto &pairs : thread_pairs) {
        merged_pairs.insert(merged_pairs.end(), pairs.begin(), pairs.end());
    }

    // Find the global bottom-k indices
    std::partial_sort(merged_pairs.begin(), merged_pairs.begin() + k, merged_pairs.end(),
                      [](std::pair<SIZE_TYPE, VALUE_TYPE>& a, std::pair<SIZE_TYPE, VALUE_TYPE>& b) { return a.second > b.second; });

    merged_pairs.resize(k);
    std::vector<SIZE_TYPE> indices;
    for(const auto & pair : merged_pairs){
        indices.push_back(pair.first);
    }

    return indices;
}

template <class SIZE_TYPE, class VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>
top_k_csr_biased(VALUE_TYPE *values, CSRInput<SIZE_TYPE,  VALUE_TYPE>& bias, size_t rows, size_t cols, size_t k, int num_threads) {
    // Step 1: Get the top-k indices
    std::vector<SIZE_TYPE> top_k = top_k_indices_biased(values, bias, rows * cols, k, num_threads);

    // Step 2: Prepare space for row/column indices
    std::unique_ptr<SIZE_TYPE[]> row_indices(new SIZE_TYPE[k]);
    std::unique_ptr<SIZE_TYPE[]> col_indices(new SIZE_TYPE[k]);
    std::unique_ptr<VALUE_TYPE[]> top_values(new VALUE_TYPE[k]);

    // Step 3: Convert flat indices to row/column indices in parallel
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < k; ++i) {
        size_t flat_idx = top_k[i];
        row_indices[i] = static_cast<SIZE_TYPE>(flat_idx / cols);
        col_indices[i] = static_cast<SIZE_TYPE>(flat_idx % cols);
        top_values[i] = values[flat_idx];
    }

    // Step 4: Create the COO sparse struct
    COOPointers<SIZE_TYPE> ptrs = k; // Store nnz directly
    COOIndices<SIZE_TYPE> indices{
        std::move(row_indices),
        std::move(col_indices)
    };
    UnaryValues<VALUE_TYPE> coo_values{
        std::move(top_values)
    };

    merge_sort_coo(indices, coo_values, k);  //there better not be any duplicates. However, Todo: check there are no duplicates

    sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>> coo_result(
        ptrs, indices, coo_values, rows, cols, k);

    return to_csr(coo_result, num_threads);
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
std::vector<SIZE_TYPE> top_k_indices(VALUE_TYPE *values, size_t size, size_t k, int num_threads) {    
    if (k > size) k = size;

    size_t chunk_size = (size + num_threads - 1) / num_threads;
    std::vector<std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>>> thread_pairs(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        SIZE_TYPE start = static_cast<SIZE_TYPE>(thread_id * chunk_size);
        SIZE_TYPE end = static_cast<SIZE_TYPE>(std::min(static_cast<size_t>(start + chunk_size), size));

        std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> local_pairs;
        local_pairs.reserve(end - start);

        for (SIZE_TYPE i = start; i < end; ++i) {
            local_pairs.emplace_back(i, values[i]);
        }

        size_t local_k = std::min(k, local_pairs.size());
        std::partial_sort(local_pairs.begin(), local_pairs.begin() + local_k, local_pairs.end(),
                          [](const std::pair<SIZE_TYPE, VALUE_TYPE>& a, const std::pair<SIZE_TYPE, VALUE_TYPE>& b) {
                              return a.second > b.second;
                          });

        if (local_pairs.size() > k) local_pairs.resize(k);
        thread_pairs[thread_id] = std::move(local_pairs);
    }

    std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> merged_pairs;
    for (auto &pairs : thread_pairs) {
        merged_pairs.insert(merged_pairs.end(), pairs.begin(), pairs.end());
    }

    size_t final_k = std::min(k, merged_pairs.size());
    std::partial_sort(merged_pairs.begin(), merged_pairs.begin() + final_k, merged_pairs.end(),
                      [](const std::pair<SIZE_TYPE, VALUE_TYPE>& a, const std::pair<SIZE_TYPE, VALUE_TYPE>& b) {
                          return a.second > b.second;
                      });

    std::vector<SIZE_TYPE> indices;
    indices.reserve(final_k);
    for (size_t i = 0; i < final_k; ++i) {
        indices.push_back(merged_pairs[i].first);
    }

    return indices;
}

template <class SIZE_TYPE, class VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>
top_k_csr(VALUE_TYPE *values, size_t rows, size_t cols, size_t k, int num_threads) {
    // Step 1: Get the unbiased top-k indices
    std::vector<SIZE_TYPE> top_k = top_k_indices<SIZE_TYPE, VALUE_TYPE>(values, rows * cols, k, num_threads);
    size_t actual_k = top_k.size();

    // Step 2: Allocate shared vectors for COO
    auto row_vec = std::make_shared<std::vector<SIZE_TYPE>>(actual_k);
    auto col_vec = std::make_shared<std::vector<SIZE_TYPE>>(actual_k);
    auto val_vec = std::make_shared<std::vector<VALUE_TYPE>>(actual_k);

    // Step 3: Map flat indices to 2D coordinates in parallel
    // We use .data() for thread-safe concurrent writing to pre-allocated indices
    SIZE_TYPE* r_ptr = row_vec->data();
    SIZE_TYPE* c_ptr = col_vec->data();
    VALUE_TYPE* v_ptr = val_vec->data();

    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < actual_k; ++i) {
        SIZE_TYPE flat_idx = top_k[i];
        r_ptr[i] = static_cast<SIZE_TYPE>(flat_idx / cols);
        c_ptr[i] = static_cast<SIZE_TYPE>(flat_idx % cols);
        v_ptr[i] = values[flat_idx];
    }

    // Step 4: Wrap into your struct types
    COOPointers<SIZE_TYPE> ptrs = static_cast<SIZE_TYPE>(actual_k);
    COOIndices<SIZE_TYPE> indices = {row_vec, col_vec};
    UnaryValues<VALUE_TYPE> coo_values = {val_vec};

    // Keep the sort to ensure indices are row-major for the CSR conversion
    merge_sort_coo(indices, coo_values, actual_k);

    sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>> coo_result(
        ptrs, indices, coo_values, rows, cols, actual_k);

    return to_csr(coo_result, num_threads);
}

#endif