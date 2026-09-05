#ifndef __CSR__HPP_
#define __CSR__HPP_

#include "fp4quant.hpp"
#include "sparse_struct.hpp"

#include "coo.hpp"
// #include "unique_vector.hpp"

#include <algorithm>
#include <cmath>
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

// Below this element count, #pragma omp parallel's own thread-team
// fork/join synchronization cost (barrier + wake-from-idle -- NOT fresh
// OS thread creation, libgomp keeps a persistent pool, but the
// rendezvous is still paid on every region entry) exceeds the entire
// serial cost of the work many times over. Confirmed directly, even with
// a fully warmed pool (50 discarded warmup calls first): CSR.from_dense
// on a single 256-element activation row cost 403us at num_cpus=4 vs
// 53us at num_cpus=1 -- a ~7.5x tax for rendezvous on essentially no
// work. This swamps block4's own genuine 1.6-2.35x kernel-level speedup
// for any caller that sparsifies small per-row arrays repeatedly (e.g.
// an RNN's per-step activation sparsification, sili_peridot's
// ToyTileRecurrenceRMT). Used by top_k_indices/top_k_csr/to_csr below to
// skip straight to a serial loop under this size, in favor of the
// #pragma omp path only once there's enough real work to amortize the
// rendezvous cost -- same algorithm/output either way, not a behavior
// change.
constexpr size_t SILI_OMP_SMALL_THRESHOLD = 4096;

template <class SIZE_TYPE>
using CSRPointers = std::array<std::shared_ptr<std::vector<SIZE_TYPE>>, 1>;

template <typename INDEX_ARRAYS> struct ReduceArraySize {
    using type =
        std::array<typename INDEX_ARRAYS::value_type, std::tuple_size<INDEX_ARRAYS>::value - 1>;
};

template <typename INDEX_ARRAYS> using ReducedArray = typename ReduceArraySize<INDEX_ARRAYS>::type;

template <typename INDEX_ARRAYS>
using stdarr_of_uniqarr_type =
    typename std::tuple_element<0, INDEX_ARRAYS>::type ::element_type // vector<T>
    ::value_type;                                                     // T
template <int... selections> struct expand {

    template <typename SIZE_TYPE, typename INDEX_ARRAYS, typename VALUE_ARRAYS>
    static sparse_struct<
        SIZE_TYPE, COOPointers<SIZE_TYPE>, INDEX_ARRAYS,
        std::array<std::shared_ptr<std::vector<stdarr_of_uniqarr_type<VALUE_ARRAYS>>>,
                   num_indices<VALUE_ARRAYS> + sizeof...(selections)>>
    view_coo_values(const sparse_struct<SIZE_TYPE, SIZE_TYPE, INDEX_ARRAYS, VALUE_ARRAYS>& a_coo) {
        using T = stdarr_of_uniqarr_type<VALUE_ARRAYS>;
        constexpr std::array<int, sizeof...(selections)> selection = {selections...};
        constexpr size_t num_value_indices = num_indices<VALUE_ARRAYS>;
        constexpr size_t num_selections = sizeof...(selections);

        sparse_struct<
            SIZE_TYPE, COOPointers<SIZE_TYPE>, INDEX_ARRAYS,
            std::array<std::shared_ptr<std::vector<T>>, num_value_indices + sizeof...(selections)>>
            expanded_coo;

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
                expanded_coo.values[idx] = std::make_shared<std::vector<T>>(a_coo.nnz(), T(0));
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

// warning: the original csr also has the same pointers after this operation.
// warning2: the input coo MUST be coalesced: it must be sorted and have no duplicates.
template <typename SIZE_TYPE, typename INDEX_ARRAYS, typename VALUE_ARRAYS>
sparse_struct<SIZE_TYPE,
              CSRPointers<SIZE_TYPE>,     // First SIZE_TYPE transformed to CSRPointers
              ReducedArray<INDEX_ARRAYS>, // INDEX_ARRAYS reduced by one
              VALUE_ARRAYS>
to_csr(sparse_struct<SIZE_TYPE,
                     COOPointers<SIZE_TYPE>, // First SIZE_TYPE is unchanged here
                     INDEX_ARRAYS,           // INDEX_ARRAYS as provided
                     VALUE_ARRAYS>& a_coo,
       const int num_cpus) {
    SIZE_TYPE nnz = a_coo.ptrs;
    SIZE_TYPE num_rows = a_coo.rows;
    // auto rows = a_coo.indices[0].release();

    if (a_coo.indices[0].get() == nullptr) {
        sparse_struct<SIZE_TYPE,
                      CSRPointers<SIZE_TYPE>,     // First SIZE_TYPE transformed to CSRPointers
                      ReducedArray<INDEX_ARRAYS>, // INDEX_ARRAYS reduced by one
                      VALUE_ARRAYS>
            csr;

        csr.rows = num_rows;
        csr.cols = a_coo.cols;
        csr.ptrs[0].reset(new std::vector<SIZE_TYPE>(num_rows + 1, 0));
        for (std::size_t idx = 0; idx < num_indices<INDEX_ARRAYS> - 1; ++idx) {
            csr.indices[idx].reset();
        }
        for (std::size_t valIdx = 0; valIdx < num_indices<VALUE_ARRAYS>; ++valIdx) {
            csr.values[valIdx].reset();
        }

        return csr;
    }

    // Allocate accumulators for parallel histogram accumulation
    SIZE_TYPE* accum = new SIZE_TYPE[a_coo.rows]();
    // nnz-size guard (see SILI_OMP_SMALL_THRESHOLD's own docstring above)
    // -- num_cpus>1 alone isn't a sufficient reason to pay a real
    // #pragma omp parallel rendezvous when nnz itself is tiny (e.g. a
    // single sparsified activation row's own top-k output).
    if (num_cpus > 1 && nnz >= SILI_OMP_SMALL_THRESHOLD) {
        SIZE_TYPE* thr_accum = new SIZE_TYPE[num_cpus * a_coo.rows];
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

    std::vector<SIZE_TYPE>* ptrs = new std::vector<SIZE_TYPE>(num_rows + 1);
    SIZE_TYPE scan_a = 0;

    // Parallel scan to compute row pointers. Unlike the histogram pass
    // above, this had no num_cpus/size guard at all -- `#pragma omp
    // parallel for simd` with no num_threads() clause ignores the
    // caller's num_cpus entirely and always uses the OpenMP runtime's
    // own default team size, so it always paid the small-region
    // rendezvous tax regardless of caller intent. num_rows+1 is almost
    // always tiny relative to SILI_OMP_SMALL_THRESHOLD for this
    // function's real callers (one row per online activation), so gate
    // it the same way.
    if (static_cast<size_t>(num_rows) + 1 >= SILI_OMP_SMALL_THRESHOLD) {
#pragma omp parallel for simd reduction(inscan, + : scan_a)
        for (SIZE_TYPE i = 0; i <= num_rows; i++) {
            (*ptrs)[i] = scan_a;
#pragma omp scan exclusive(scan_a)
            {
                scan_a += accum[i];
            }
        }
    } else {
        for (SIZE_TYPE i = 0; i <= num_rows; i++) {
            (*ptrs)[i] = scan_a;
            scan_a += accum[i];
        }
    }

    delete[] accum;
    a_coo.indices[0].reset();
    // Create and return the CSR sparse structure
    sparse_struct<SIZE_TYPE,
                  CSRPointers<SIZE_TYPE>,     // First SIZE_TYPE transformed to CSRPointers
                  ReducedArray<INDEX_ARRAYS>, // INDEX_ARRAYS reduced by one
                  VALUE_ARRAYS>
        csr;

    a_coo.ptrs = 0;

    csr.rows = num_rows;
    csr.cols = a_coo.cols;
    csr.ptrs[0].reset(ptrs);
    a_coo.indices[0].reset();
    for (std::size_t idx = 0; idx < num_indices<INDEX_ARRAYS> - 1; ++idx) {
        csr.indices[idx] = std::move(a_coo.indices[idx + 1]);
        a_coo.indices[idx + 1].reset();
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
CSRInput<SIZE_TYPE, VALUE_TYPE>
make_csr_input(SIZE_TYPE rows, SIZE_TYPE cols, std::vector<SIZE_TYPE> ptrs,
               std::vector<SIZE_TYPE> indices, std::vector<VALUE_TYPE> values) {
    CSRInput<SIZE_TYPE, VALUE_TYPE> t;
    t.rows = rows;
    t.cols = cols;
    t.ptrs[0] = std::make_shared<std::vector<SIZE_TYPE>>(std::move(ptrs));
    t.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(std::move(indices));
    t.values[0] = std::make_shared<std::vector<VALUE_TYPE>>(std::move(values));
    return t;
}

// ptrs:       rows+1 entries
// indices / values / grads / importance: nnz entries each
template <typename SIZE_TYPE, typename VALUE_TYPE>
SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>
make_weights(SIZE_TYPE rows, SIZE_TYPE cols, std::vector<SIZE_TYPE>&& ptrs,
             std::vector<SIZE_TYPE>&& indices, FP4BiPacked&& values_and_importances) {
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> w;
    w.connections.rows = rows;
    w.connections.cols = cols;
    w.connections.ptrs[0] = std::make_shared<std::vector<SIZE_TYPE>>(std::move(ptrs));
    w.connections.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(std::move(indices));
    w.connections.values = values_and_importances;
    w.probes.rows = rows;
    w.probes.cols = cols;
    w.out_degree.assign(cols, SIZE_TYPE(0));
    for (const auto& idx : *w.connections.indices[0])
        w.out_degree[idx]++;
    return w;
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
SparseLinearWeightsV<SIZE_TYPE, VALUE_TYPE>
make_weights_v(SIZE_TYPE rows, SIZE_TYPE cols, std::vector<SIZE_TYPE>&& ptrs,
               std::vector<SIZE_TYPE>&& indices, std::vector<VALUE_TYPE>&& values,
               std::vector<VALUE_TYPE>&& importance) {
    SparseLinearWeightsV<SIZE_TYPE, VALUE_TYPE> w;
    w.probes.rows = rows;
    w.probes.cols = cols;
    w.out_degree.assign(cols, SIZE_TYPE(0));
    for (const auto& idx : *w.connections.indices[0])
        w.out_degree[idx]++;
    return w;
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
std::vector<SIZE_TYPE> top_k_indices_biased(VALUE_TYPE* values,
                                            CSRInput<SIZE_TYPE, VALUE_TYPE>& bias, size_t size,
                                            size_t k, int num_threads) {
    // Each thread processes a chunk of the array
    size_t chunk_size = (size + num_threads - 1) / num_threads;
    std::vector<std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>>> thread_pairs(num_threads);

    if (k > size) {
        k = size;
    }

#pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        SIZE_TYPE start = thread_id * chunk_size;
        SIZE_TYPE end = std::min(start + chunk_size, size);

        SIZE_TYPE bias_ptr = bias.ptrs[0][start / bias.cols]; // start at the correct row

        // Collect indices for this thread
        std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> local_pairs;
        for (size_t i = start; i < end; ++i) {
            SIZE_TYPE bias_row = i / bias.cols;
            while (bias.indices[0][bias_ptr] < i % bias.cols &&
                   bias_ptr < bias.ptrs[0][bias_row + 1]) {
                ++bias_ptr;
            }
            if (bias.indices[0][bias_ptr] == i % bias.cols &&
                bias_ptr <= bias.ptrs[0][bias_row + 1]) {
                local_pairs.emplace_back(i, bias.values[0][bias_ptr] + values[i]);
                ++bias_ptr;
            } else {
                local_pairs.emplace_back(i, values[i]);
            }
        }

        // Sort local indices by values
        std::partial_sort(
            local_pairs.begin(), local_pairs.begin() + std::min(k, local_pairs.size()),
            local_pairs.end(),
            [](const std::pair<SIZE_TYPE, VALUE_TYPE>& a,
               const std::pair<SIZE_TYPE, VALUE_TYPE>& b) { return a.second > b.second; });

        // Keep only the smallest k elements
        if (local_pairs.size() > k) {
            local_pairs.resize(k);
        }

        thread_pairs[thread_id] = std::move(local_pairs);
    }

    // Merge results from all threads
    std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> merged_pairs;
    for (const auto& pairs : thread_pairs) {
        merged_pairs.insert(merged_pairs.end(), pairs.begin(), pairs.end());
    }

    // Find the global bottom-k indices
    std::partial_sort(
        merged_pairs.begin(), merged_pairs.begin() + k, merged_pairs.end(),
        [](const std::pair<SIZE_TYPE, VALUE_TYPE>& a, const std::pair<SIZE_TYPE, VALUE_TYPE>& b) {
            return a.second > b.second;
        });

    merged_pairs.resize(k);
    std::vector<SIZE_TYPE> indices;
    for (const auto& pair : merged_pairs) {
        indices.push_back(pair.first);
    }

    return indices;
}

template <class SIZE_TYPE, class VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>
top_k_csr_biased_v(VALUE_TYPE* values, CSRInput<SIZE_TYPE, VALUE_TYPE>& bias, size_t rows,
                   size_t cols, size_t k, int num_threads) {
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
    COOIndices<SIZE_TYPE> indices{std::move(row_indices), std::move(col_indices)};
    UnaryValues<VALUE_TYPE> coo_values{std::move(top_values)};

    merge_sort_coo(
        indices, coo_values,
        k); // there better not be any duplicates. However, Todo: check there are no duplicates

    sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>
        coo_result(ptrs, indices, coo_values, rows, cols, k);

    return to_csr(coo_result, num_threads);
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
std::vector<SIZE_TYPE> top_k_indices(VALUE_TYPE* values, size_t size, size_t k, int num_threads) {
    if (k > size)
        k = size;

    // BUG FIX: this compared raw signed value (a.second > b.second), not
    // magnitude -- for zero-mean data (e.g. any post-RMSNorm activation)
    // that keeps only the largest POSITIVE entries and discards every
    // negative one regardless of magnitude, no matter how large; even
    // k close to `size` still drops exactly the most-negative (often
    // highest-magnitude) entries first. dense_to_top_k_csr's own
    // docstring ("top-k sparsity conversion") and CSR.from_dense's
    // ("top-k entries by magnitude") both assume magnitude-based
    // selection -- callers that don't already pre-abs their values
    // (e.g. genesis_build_probes does) got silently wrong results.
    auto by_magnitude = [](const std::pair<SIZE_TYPE, VALUE_TYPE>& a,
                           const std::pair<SIZE_TYPE, VALUE_TYPE>& b) {
        return std::abs(a.second) > std::abs(b.second);
    };

    if (size < SILI_OMP_SMALL_THRESHOLD || num_threads <= 1) {
        std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> pairs;
        pairs.reserve(size);
        for (size_t i = 0; i < size; ++i)
            pairs.emplace_back(static_cast<SIZE_TYPE>(i), values[i]);
        size_t final_k = std::min(k, pairs.size());
        std::partial_sort(pairs.begin(), pairs.begin() + final_k, pairs.end(), by_magnitude);
        std::vector<SIZE_TYPE> indices;
        indices.reserve(final_k);
        for (size_t i = 0; i < final_k; ++i)
            indices.push_back(pairs[i].first);
        return indices;
    }

    size_t chunk_size = (size + num_threads - 1) / num_threads;
    std::vector<std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>>> thread_pairs(num_threads);

#pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        SIZE_TYPE start = static_cast<SIZE_TYPE>(thread_id * chunk_size);
        SIZE_TYPE end =
            static_cast<SIZE_TYPE>(std::min(static_cast<size_t>(start + chunk_size), size));

        std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> local_pairs;
        local_pairs.reserve(end - start);

        for (SIZE_TYPE i = start; i < end; ++i) {
            local_pairs.emplace_back(i, values[i]);
        }

        size_t local_k = std::min(k, local_pairs.size());
        std::partial_sort(local_pairs.begin(), local_pairs.begin() + local_k, local_pairs.end(),
                          by_magnitude);

        if (local_pairs.size() > k)
            local_pairs.resize(k);
        thread_pairs[thread_id] = std::move(local_pairs);
    }

    std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> merged_pairs;
    for (auto& pairs : thread_pairs) {
        merged_pairs.insert(merged_pairs.end(), pairs.begin(), pairs.end());
    }

    size_t final_k = std::min(k, merged_pairs.size());
    std::partial_sort(merged_pairs.begin(), merged_pairs.begin() + final_k, merged_pairs.end(),
                      by_magnitude);

    std::vector<SIZE_TYPE> indices;
    indices.reserve(final_k);
    for (size_t i = 0; i < final_k; ++i) {
        indices.push_back(merged_pairs[i].first);
    }

    return indices;
}

template <class SIZE_TYPE, class VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>
top_k_csr(VALUE_TYPE* values, size_t rows, size_t cols, size_t k, int num_threads) {
    // Step 1: Get the unbiased top-k indices
    std::vector<SIZE_TYPE> top_k_idx =
        top_k_indices<SIZE_TYPE, VALUE_TYPE>(values, rows * cols, k, num_threads);
    size_t actual_k = top_k_idx.size();

    // Step 2: Allocate shared vectors for COO
    auto row_vec = std::make_shared<std::vector<SIZE_TYPE>>(actual_k);
    auto col_vec = std::make_shared<std::vector<SIZE_TYPE>>(actual_k);
    auto val_vec = std::make_shared<std::vector<VALUE_TYPE>>(actual_k);

    // Step 3: Map flat indices to 2D coordinates in parallel
    // We use .data() for thread-safe concurrent writing to pre-allocated indices
    SIZE_TYPE* r_ptr = row_vec->data();
    SIZE_TYPE* c_ptr = col_vec->data();
    VALUE_TYPE* v_ptr = val_vec->data();

    // Same small-array fast path as top_k_indices above -- see its own
    // comment for the full rationale (this remap is a second, separate
    // #pragma omp region, so it pays the same fork/join tax again if
    // left unconditional).
    if (actual_k < SILI_OMP_SMALL_THRESHOLD || num_threads <= 1) {
        for (size_t i = 0; i < actual_k; ++i) {
            SIZE_TYPE flat_idx = top_k_idx[i];
            r_ptr[i] = static_cast<SIZE_TYPE>(flat_idx / cols);
            c_ptr[i] = static_cast<SIZE_TYPE>(flat_idx % cols);
            v_ptr[i] = values[flat_idx];
        }
    } else {
#pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < actual_k; ++i) {
            SIZE_TYPE flat_idx = top_k_idx[i];
            r_ptr[i] = static_cast<SIZE_TYPE>(flat_idx / cols);
            c_ptr[i] = static_cast<SIZE_TYPE>(flat_idx % cols);
            v_ptr[i] = values[flat_idx];
        }
    }

    // Step 4: Wrap into your struct types
    COOPointers<SIZE_TYPE> ptrs = static_cast<SIZE_TYPE>(actual_k);
    COOIndices<SIZE_TYPE> indices = {row_vec, col_vec};
    UnaryValues<VALUE_TYPE> coo_values = {val_vec};

    // Keep the sort to ensure indices are row-major for the CSR conversion
    merge_sort_coo(indices, coo_values, actual_k);

    sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>
        coo_result(ptrs, indices, coo_values, rows, cols, actual_k);

    return to_csr(coo_result, num_threads);
}

// Genuine per-row top-k: row r independently keeps its own top-k_per_row[r]
// largest-magnitude entries. NOT equivalent to top_k_csr above (that k is
// spent GLOBALLY across the whole flattened rows*cols array) -- this is a
// different selection, needed for graded per-row density schedules (e.g.
// sili_peridot's step_cached query-step credit assignment, where each row
// corresponds to a content position and gets its own density). Was
// previously a pure-Python/numpy per-row argpartition loop
// (sili/sparse_rnn.py's _graded_top_k_csr) -- moved here because that loop
// ran on every query/backward step and dominated the step's own cost once
// measured against a real curriculum run (use_tile_cache=1 measured
// SLOWER than the plain step() baseline, not the expected speedup).
template <typename SIZE_TYPE, typename VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>
top_k_csr_graded(VALUE_TYPE* values, size_t rows, size_t cols, const SIZE_TYPE* k_per_row,
                 int num_threads) {
    std::vector<SIZE_TYPE> row_offset(rows + 1, 0);
    for (size_t r = 0; r < rows; ++r) {
        SIZE_TYPE k = k_per_row[r];
        if (k < 0)
            k = 0;
        if (static_cast<size_t>(k) > cols)
            k = static_cast<SIZE_TYPE>(cols);
        row_offset[r + 1] = row_offset[r] + k;
    }
    size_t total = static_cast<size_t>(row_offset[rows]);

    std::vector<SIZE_TYPE> indices(total);
    std::vector<VALUE_TYPE> out_values(total);

    auto by_magnitude = [](const std::pair<SIZE_TYPE, VALUE_TYPE>& a,
                           const std::pair<SIZE_TYPE, VALUE_TYPE>& b) {
        return std::abs(a.second) > std::abs(b.second);
    };

    // Same small-region guard as top_k_indices/top_k_csr above -- per-row
    // work here is O(cols), usually a single layer's own width, so
    // #pragma omp parallel's fork/join rendezvous can easily exceed the
    // whole serial cost. Only parallelize once total work justifies it.
    // (false positive: read in the `if (use_omp)` clause of the #pragma
    // omp line right below -- cppcheck doesn't parse pragma contents)
    // cppcheck-suppress unreadVariable
    bool use_omp = (num_threads > 1 && rows * cols >= SILI_OMP_SMALL_THRESHOLD);

#pragma omp parallel for num_threads(num_threads) if (use_omp) schedule(dynamic)
    for (size_t r = 0; r < rows; ++r) {
        SIZE_TYPE k = static_cast<SIZE_TYPE>(row_offset[r + 1] - row_offset[r]);
        if (k == 0)
            continue;
        const VALUE_TYPE* row = values + r * cols;
        std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> pairs;
        pairs.reserve(cols);
        for (size_t c = 0; c < cols; ++c)
            pairs.emplace_back(static_cast<SIZE_TYPE>(c), row[c]);
        std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), by_magnitude);
        // Row-major/ascending-column convention, matching top_k_csr's own
        // merge_sort_coo step -- the selected k entries need re-sorting by
        // index since partial_sort above ordered them by magnitude.
        std::sort(pairs.begin(), pairs.begin() + k,
                  [](const std::pair<SIZE_TYPE, VALUE_TYPE>& a,
                     const std::pair<SIZE_TYPE, VALUE_TYPE>& b) { return a.first < b.first; });
        SIZE_TYPE base = row_offset[r];
        for (SIZE_TYPE i = 0; i < k; ++i) {
            indices[base + i] = pairs[i].first;
            out_values[base + i] = pairs[i].second;
        }
    }

    std::vector<SIZE_TYPE> ptrs(row_offset.begin(), row_offset.end());
    return make_csr_input<SIZE_TYPE, VALUE_TYPE>(static_cast<SIZE_TYPE>(rows),
                                                 static_cast<SIZE_TYPE>(cols), std::move(ptrs),
                                                 std::move(indices), std::move(out_values));
}

// Nucleus/energy-threshold top-k: row r independently keeps the SMALLEST
// number of its own largest-magnitude entries whose captured squared-
// magnitude ratio
//     R(v, k) = sum(v_topk^2) / sum(v^2)
// is >= r_target_per_row[r]. k is a CONSEQUENCE of r_target and the row's
// actual data, not a fixed constant -- same math as Eckart-Young truncated-
// SVD captured-variance / LLM nucleus (top-p) sampling, applied to squared
// magnitude instead of softmax probability (see sili_peridot/JOURNAL.md's
// "nucleus/energy-threshold top-k math" design note). A fully-zero row or
// r_target<=0 has nothing to capture and keeps k=0; r_target>=1 (or a
// floating-point rounding shortfall right at the boundary) keeps every
// entry -- both handled as explicit fallbacks below, not accidents of the
// loop shape.
//
// k_min/k_max (default 0/SIZE_MAX, i.e. no-op) clamp the R_target-derived
// k AFTER the fact -- direct instruction: R_target alone can degenerate to
// k=0 (a fully-dead row/synapse-update-starved layer, bad regardless of
// how little energy that row happened to carry) or to near-100% density
// (bad on hardware that specifically wants a bounded, e.g. ~10%, density
// ceiling). Applied per row against that row's own actual entry count
// (min(k_max,cols)), not globally. k_min padding pulls from the SAME
// magnitude-sorted order already computed for the R_target selection --
// still "next largest by magnitude," not arbitrary -- so a clamped row
// degrades gracefully to plain top-k instead of picking randomly.
template <typename SIZE_TYPE, typename VALUE_TYPE>
sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>
top_k_csr_nucleus(VALUE_TYPE* values, size_t rows, size_t cols, const VALUE_TYPE* r_target_per_row,
                  int num_threads, size_t k_min = 0, size_t k_max = SIZE_MAX) {
    std::vector<SIZE_TYPE> k_per_row(rows, 0);
    std::vector<std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>>> kept_rows(rows);

    // (false positive: read in the `if (use_omp)` clause of the #pragma
    // omp line right below -- cppcheck doesn't parse pragma contents)
    // cppcheck-suppress unreadVariable
    bool use_omp = (num_threads > 1 && rows * cols >= SILI_OMP_SMALL_THRESHOLD);

#pragma omp parallel for num_threads(num_threads) if (use_omp) schedule(dynamic)
    for (size_t r = 0; r < rows; ++r) {
        const VALUE_TYPE* row = values + r * cols;
        VALUE_TYPE r_target = r_target_per_row[r];

        double total_sq = 0.0;
        for (size_t c = 0; c < cols; ++c)
            total_sq += static_cast<double>(row[c]) * static_cast<double>(row[c]);

        std::vector<std::pair<SIZE_TYPE, VALUE_TYPE>> pairs;
        // Sort whenever there's real data to pick from AND either R_target
        // or a k_min floor could want some of it -- k_min padding pulls
        // from this same magnitude-sorted list (see docstring above), so
        // it must exist even when r_target alone would pick k=0.
        if (total_sq > 0.0 && (r_target > VALUE_TYPE(0) || k_min > 0)) {
            pairs.reserve(cols);
            for (size_t c = 0; c < cols; ++c)
                pairs.emplace_back(static_cast<SIZE_TYPE>(c), row[c]);
            std::sort(pairs.begin(), pairs.end(),
                      [](const std::pair<SIZE_TYPE, VALUE_TYPE>& a,
                         const std::pair<SIZE_TYPE, VALUE_TYPE>& b) {
                          return std::abs(a.second) > std::abs(b.second);
                      });

            size_t chosen = 0;
            if (r_target > VALUE_TYPE(0)) {
                double target_sq = static_cast<double>(r_target) * total_sq;
                chosen = cols; // fallback: r_target>=1 or a rounding
                               // shortfall right at the boundary -- keep
                               // everything rather than under-shoot.
                double cum = 0.0;
                for (size_t i = 0; i < cols; ++i) {
                    cum +=
                        static_cast<double>(pairs[i].second) * static_cast<double>(pairs[i].second);
                    if (cum >= target_sq) {
                        chosen = i + 1;
                        break;
                    }
                }
            }

            // Hardware-driven density floor/ceiling, applied AFTER the
            // R_target-derived choice, against this row's own actual
            // entry count -- a totally-informationless row still can't
            // manufacture k_min nonzero entries beyond what it has.
            size_t eff_k_min = std::min(k_min, pairs.size());
            size_t eff_k_max = std::min(k_max, pairs.size());
            if (chosen < eff_k_min)
                chosen = eff_k_min;
            if (chosen > eff_k_max)
                chosen = eff_k_max;

            pairs.resize(chosen);
            // Row-major/ascending-column convention, matching top_k_csr_
            // graded's own re-sort step -- the selected entries need
            // re-ordering by index since they were chosen by magnitude.
            std::sort(pairs.begin(), pairs.end(),
                      [](const std::pair<SIZE_TYPE, VALUE_TYPE>& a,
                         const std::pair<SIZE_TYPE, VALUE_TYPE>& b) { return a.first < b.first; });
        }
        k_per_row[r] = static_cast<SIZE_TYPE>(pairs.size());
        kept_rows[r] = std::move(pairs);
    }

    std::vector<SIZE_TYPE> row_offset(rows + 1, 0);
    for (size_t r = 0; r < rows; ++r)
        row_offset[r + 1] = row_offset[r] + k_per_row[r];
    size_t total = static_cast<size_t>(row_offset[rows]);

    std::vector<SIZE_TYPE> indices(total);
    std::vector<VALUE_TYPE> out_values(total);
    for (size_t r = 0; r < rows; ++r) {
        SIZE_TYPE base = row_offset[r];
        for (size_t i = 0; i < kept_rows[r].size(); ++i) {
            indices[base + i] = kept_rows[r][i].first;
            out_values[base + i] = kept_rows[r][i].second;
        }
    }

    std::vector<SIZE_TYPE> ptrs(row_offset.begin(), row_offset.end());
    return make_csr_input<SIZE_TYPE, VALUE_TYPE>(static_cast<SIZE_TYPE>(rows),
                                                 static_cast<SIZE_TYPE>(cols), std::move(ptrs),
                                                 std::move(indices), std::move(out_values));
}

// ── csr_union ─────────────────────────────────────────────────────────────
//
// Merge two same-shape absolute CSRs into the union of their nonzero
// positions. Construction/loading-time operation (e.g. combining a dense
// LLM's folded weights with a freshly pre-seeded skip-connection band --
// see sili_peridot/JOURNAL.md); not used in the forward/backward path.
//
// Each row is an independent two-pointer merge of two sorted column lists,
// so this parallelizes directly over rows -- no cross-row coordination
// needed (unlike linear_sisldo.hpp's synap_parallel_fill, which also
// enforces a global importance budget).
//
// prefer: 0 = keep A's value on overlap, 1 = keep B's, 2 = sum them.
template <typename SIZE_TYPE, typename VALUE_TYPE>
void csr_union(const std::vector<SIZE_TYPE>& ptrs_a, const std::vector<SIZE_TYPE>& idx_a,
               const std::vector<VALUE_TYPE>& vals_a, const std::vector<SIZE_TYPE>& ptrs_b,
               const std::vector<SIZE_TYPE>& idx_b, const std::vector<VALUE_TYPE>& vals_b,
               const SIZE_TYPE n_rows, const int prefer, const int num_cpus,
               std::vector<SIZE_TYPE>& out_ptrs, std::vector<SIZE_TYPE>& out_idx,
               std::vector<VALUE_TYPE>& out_vals) {
    std::vector<SIZE_TYPE> row_len(n_rows);

#pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE r = 0; r < n_rows; ++r) {
        SIZE_TYPE a = ptrs_a[r], a_end = ptrs_a[r + 1];
        SIZE_TYPE b = ptrs_b[r], b_end = ptrs_b[r + 1];
        SIZE_TYPE count = 0;
        while (a < a_end && b < b_end) {
            if (idx_a[a] < idx_b[b]) {
                ++a;
                ++count;
            } else if (idx_b[b] < idx_a[a]) {
                ++b;
                ++count;
            } else {
                ++a;
                ++b;
                ++count;
            }
        }
        count += (a_end - a) + (b_end - b);
        row_len[r] = count;
    }

    out_ptrs.assign(n_rows + 1, SIZE_TYPE(0));
    for (SIZE_TYPE r = 0; r < n_rows; ++r)
        out_ptrs[r + 1] = out_ptrs[r] + row_len[r];

    const SIZE_TYPE total = out_ptrs[n_rows];
    out_idx.resize(total);
    out_vals.resize(total);

#pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE r = 0; r < n_rows; ++r) {
        SIZE_TYPE a = ptrs_a[r], a_end = ptrs_a[r + 1];
        SIZE_TYPE b = ptrs_b[r], b_end = ptrs_b[r + 1];
        SIZE_TYPE w = out_ptrs[r];
        while (a < a_end && b < b_end) {
            if (idx_a[a] < idx_b[b]) {
                out_idx[w] = idx_a[a];
                out_vals[w] = vals_a[a];
                ++a;
                ++w;
            } else if (idx_b[b] < idx_a[a]) {
                out_idx[w] = idx_b[b];
                out_vals[w] = vals_b[b];
                ++b;
                ++w;
            } else {
                out_idx[w] = idx_a[a];
                out_vals[w] = (prefer == 0)   ? vals_a[a]
                              : (prefer == 1) ? vals_b[b]
                                              : vals_a[a] + vals_b[b];
                ++a;
                ++b;
                ++w;
            }
        }
        while (a < a_end) {
            out_idx[w] = idx_a[a];
            out_vals[w] = vals_a[a];
            ++a;
            ++w;
        }
        while (b < b_end) {
            out_idx[w] = idx_b[b];
            out_vals[w] = vals_b[b];
            ++b;
            ++w;
        }
    }
}

#endif