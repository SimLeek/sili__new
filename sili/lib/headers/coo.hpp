/**
*@file coo.hpp
*@brief Header file containing functions for sorting and manipulating COO format data.

  ▗▄▄▖ ▗▄▖  ▗▄▖
 ▐▌   ▐▌ ▐▌▐▌ ▐▌
 ▐▌   ▐▌ ▐▌▐▌ ▐▌
 ▝▚▄▄▖▝▚▄▞▘▝▚▄▞▘

Stars, because they're sparse like COOs:
                                         .     o                     .  +
           '      .                          .      '   o                     o
       +           o                                    .   o .
                                   '       .-. '
.           *       '                       ) )         o              .
   '                                |      '-´           +  '
            +                      -o-             .            '         |
             .            .         |    o                         '    --o--
         .   +                               +                            | +
        o *                .                      ~~+                     *   '
                                      +                '
                                                                    . .     |
  +                   '    '        +   .                                 *-+-
           '                                          '                     |  +
                       .                                          *o.     +
     +                   .                       '+         .
                                                '      '   o
                       .                 +      .                      +   o
          *    .  +                   .          ~~+   .
                    +~~                                  '           +        *


*This header file provides various functions for sorting and manipulating COO format data,
*including top-level sorting functions, merging sorted COO matrices, and removing elements
*based on external values.
*
*The main functions provided by this file include:
* merge_sort_coo - Sorting COO format data using merge sort
*  Use this for sorting synapses before adding them to sorted sparse layers
* parallel_merge_sorted_coos - Merging two sorted COO matrices into a single sorted COO matrix using multi-threading
*  Use this for merging sparse layers with new synapses and learning
* coo_subtract_bottom_k - Removing bottom k elements based on external values and copying the result into another COO array in parallel
*  Use this for deleting the least important synapses to keep memory/processing within limits
* merge_sort_coo_external - Sorting a COO matrix based on an external array using merge sort
*
*These functions can be used for efficient sorting and manipulation of large COO format datasets.
*/
#ifndef __COO__HPP_
#define __COO__HPP_

/*
* merge_sort_coo - this is effectively the coalesce operation in pytorch
* parallel_merge_coo - for combining two COOs quickly, like stored synapses plus some additional random or sparse backprop synapses
* coo_subtract_bottom_k - for removing the bottom k importance COO elements in order to keep the COO array under a specific size.
*/

/**
* @brief Merges two sorted subarrays in COO format, removing duplicates.
*
* @tparam SIZE_TYPE Type for indexing (e.g., unsigned int).
* @tparam VALUE_TYPE Type for values stored in the matrix (e.g., float, double).
*
* @param cols Pointer to the column index array.
* @param rows Pointer to the row index array.
* @param vals Pointer to the value array.
* @param left Left boundary of the first subarray.
* @param mid Midpoint separating the two subarrays.
* @param right Right boundary of the second subarray.
*
* @return Number of duplicates removed during merging.
*/
#include "scan.hpp"
#include "sparse_struct.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <omp.h>
#include <vector>

//todo: use a c++ ide to rename this coalesce everywhere
template <typename INDEX_ARRAYS, typename VALUE_ARRAYS>
std::size_t inplace_merge_coo(INDEX_ARRAYS &indices,
                            VALUE_ARRAYS &values,
                            std::size_t left,
                            std::size_t mid,
                            std::size_t right) {
    constexpr std::size_t numValues = std::tuple_size<VALUE_ARRAYS>::value;
    //using VALUE_TYPE = VALUE_ARRAYS::value_type;
    using VALUE_TYPE = std::remove_pointer_t<decltype(values[0].get())>::value_type;
    constexpr std::size_t numIndices = std::tuple_size<INDEX_ARRAYS>::value;
    //using INDEX_TYPE = VALUE_ARRAYS::value_type;
    using INDEX_TYPE = std::remove_pointer_t<decltype(values[0].get())>::value_type;

    INDEX_TYPE i = left;
    INDEX_TYPE j = mid + 1;
    INDEX_TYPE duplicates = 0;

    // Traverse both subarrays and merge in-place
    while (i <= mid && j <= right) {
        // If the element in the left half is smaller, move on
        // Compare indices in lexicographical order
        bool less_than = false;
        bool equal = true;
        for (std::size_t idx = 0; idx < numIndices; ++idx) {
            if ((*indices[idx])[i] < (*indices[idx])[j]) {
                less_than = true;
                equal = false;
                break;
            } else if ((*indices[idx])[i] > (*indices[idx])[j]) {
                less_than = false;
                equal = false;
                break;
            }
        }

        if (less_than) {
            i++;
        } else if (equal) {
            // Sum duplicate values
            for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                (*values[valIdx])[i] += (*values[valIdx])[j];
            }
            // Remove duplicate entry from the right subarray
            for (INDEX_TYPE k = j; k < right; ++k) {
                for (std::size_t idx = 0; idx < numIndices; ++idx) {
                    (*indices[idx])[k] = (*indices[idx])[k + 1];
                }
                for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                    (*values[valIdx])[k] = (*values[valIdx])[k + 1];
                }
            }
            duplicates++;
            #pragma omp atomic
            right--;
        } else {
            // Rotate element from right half to left
            for (std::size_t idx = 0; idx < numIndices; ++idx) {
                INDEX_TYPE temp_idx = (*indices[idx])[j];
                for (INDEX_TYPE k = j; k > i; --k) {
                    (*indices[idx])[k] = (*indices[idx])[k - 1];
                }
                (*indices[idx])[i] = temp_idx;
            }

            for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                VALUE_TYPE temp_val = (*values[valIdx])[j];
                for (INDEX_TYPE k = j; k > i; --k) {
                    (*values[valIdx])[k] = (*values[valIdx])[k - 1];
                }
                (*values[valIdx])[i] = temp_val;
            }

            // Update indices
            i++;
            mid++;
            j++;
        }
    }
    return duplicates;
}

/**
 *@brief Performs insertion sort on a subarray in COO format, removing duplicates.
*
 *@tparam SIZE_TYPE Type for indexing (e.g., unsigned int).
* @tparam VALUE_TYPE Type for values stored in the matrix (e.g., float, double).
*
* @param cols Pointer to the column index array.
* @param rows Pointer to the row index array.
* @param vals Pointer to the value array.
* @param left Left boundary of the subarray.
* @param right Right boundary of the subarray.
*
* @return Number of duplicates removed during sorting.
*/
template <typename INDEX_ARRAYS, typename VALUE_ARRAYS>
std::size_t insertion_sort_coo(INDEX_ARRAYS &indices,
                            VALUE_ARRAYS &values, 
                            std::size_t left, 
                            std::size_t right) {
    constexpr std::size_t numValues = std::tuple_size<VALUE_ARRAYS>::value;
    //using VALUE_TYPE = VALUE_ARRAYS::value_type;
    using VALUE_TYPE = std::remove_pointer_t<decltype(values[0].get())>::value_type;
    constexpr std::size_t numIndices = std::tuple_size<INDEX_ARRAYS>::value;
    //using INDEX_TYPE = VALUE_ARRAYS::value_type;
    using INDEX_TYPE = std::remove_pointer_t<decltype(indices[0].get())>::value_type;

    INDEX_TYPE duplicates = 0;
    for (INDEX_TYPE i = left + 1; i <= right; i++) {
        VALUE_TYPE temp_indices[numIndices];
        VALUE_TYPE temp_values[numValues];
        for (std::size_t idx = 0; idx < numIndices; ++idx) {
            temp_indices[idx] = (*indices[idx])[i];
        }
        for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
            temp_values[valIdx] = (*values[valIdx])[i];
        }
        INDEX_TYPE j = i;
        while (j > left ) {
            bool all_equal=true;
            for (std::size_t idx = 0; idx < numIndices; ++idx) {
                if (!((*indices[idx])[j - 1] >= temp_indices[idx])) {
                    break;
                }
                else if((*indices[idx])[j-1]!= temp_indices[idx]){
                    all_equal = false;
                }
            }
            if(all_equal){
                break;
            }
            for (std::size_t idx = 0; idx < numIndices; ++idx) {
                (*indices[idx])[j] = (*indices[idx])[j - 1];
            }
            for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                (*values[valIdx])[j] = (*values[valIdx])[j - 1];
            }
            j--;
        }
        // Insert the current element into its correct position
        for (std::size_t idx = 0; idx < numIndices; ++idx) {
            (*indices[idx])[j] = temp_indices[idx];
        }
        for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
            (*values[valIdx])[j] = temp_values[valIdx];
        }

        // Check for duplicates after insertion
        bool is_duplicate = false;
        if (j > left) {
            is_duplicate = true;
            for (std::size_t idx = 0; idx < numIndices; ++idx) {
                if ((*indices[idx])[j] != (*indices[idx])[j - 1]) {
                    is_duplicate = false;
                    break;
                }
            }
        }

        if (is_duplicate) {
            for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                (*values[valIdx])[j - 1] += (*values[valIdx])[j];
            }

            // Remove the duplicate element
            for (INDEX_TYPE k = j; k < right; k++) {
                for (std::size_t idx = 0; idx < numIndices; ++idx) {
                    (*indices[idx])[k] = (*indices[idx])[k + 1];
                }
                for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                    (*values[valIdx])[k] = (*values[valIdx])[k + 1];
                }
            }
            duplicates++;
            #pragma omp atomic
            right--;
            i--;  // Adjust 'i' because the array size has decreased
        }
    }
    return duplicates;
}

/**
* @brief Recursively sorts COO format data using merge sort, removing duplicates.
*
* @tparam SIZE_TYPE Type for indexing (e.g., unsigned int).
 *@tparam VALUE_TYPE Type for values stored in the matrix (e.g., float, double).
*
 *@param cols Pointer to the column index array.
* @param rows Pointer to the row index array.
 *@param vals Pointer to the value array.
* @param left Left boundary of the subarray.
 *@param right Right boundary of the subarray.
* @param duplicates Accumulator for the total number of duplicates removed.
 *
* @return Total number of duplicates removed during sorting.
 */
template <typename INDEX_ARRAYS, typename VALUE_ARRAYS>
std::size_t recursive_merge_sort_coo(INDEX_ARRAYS &indices,
                                   VALUE_ARRAYS &values,
                                   std::size_t left,
                                   std::size_t right,
                                   std::size_t duplicates) {
    constexpr std::size_t numValues = std::tuple_size<VALUE_ARRAYS>::value;
    //using VALUE_TYPE = VALUE_ARRAYS::value_type;
    using VALUE_TYPE = std::remove_pointer_t<decltype(values[0].get())>::value_type;
    constexpr std::size_t numIndices = std::tuple_size<INDEX_ARRAYS>::value;
    //using INDEX_TYPE = VALUE_ARRAYS::value_type;
    using INDEX_TYPE = std::remove_pointer_t<decltype(indices[0].get())>::value_type;

    if (left < right) {
        if (right - left >= 32) {
            std::size_t mid = (left + right) / 2;
            std::size_t left_duplicates = 0;
            std::size_t right_duplicates = 0;
#pragma omp taskgroup
            {
#pragma omp task shared(indices, values, left_duplicates) untied if (right - left >= (1 << 14))
                left_duplicates= recursive_merge_sort_coo(indices, values, left, mid, (std::size_t)0);
#pragma omp task shared(indices, values, right_duplicates) untied if (right - left >= (1 << 14))
                right_duplicates= recursive_merge_sort_coo(indices, values, mid + 1, right, (std::size_t)0);
#pragma omp taskyield
            }

            // Adjust indices after handling duplicates
            if (left_duplicates > 0) {
                // Shift the right subarray to the left by left_duplicates
                for (INDEX_TYPE i = mid + 1; i <= right; i++) {
                    for (std::size_t idx = 0; idx < numIndices; ++idx) {
                        (*indices[idx])[i - left_duplicates] = (*indices[idx])[i];
                    }
                    for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                        (*values[valIdx])[i - left_duplicates] = (*values[valIdx])[i + 1];
                    }
                }
                mid -= left_duplicates;  // Adjust mid after the shift
                right -= left_duplicates;  // Adjust right boundary after the shift
            }

            if (right_duplicates > 0) {
                right -= right_duplicates;  // Adjust right for right-side duplicates
            }

            duplicates += inplace_merge_coo(indices, values, left, mid, right) + left_duplicates + right_duplicates;
        } else {
            duplicates += insertion_sort_coo(indices, values, left, right);
        }
    }
    return duplicates;
}

/**
* @brief Top-level function for sorting COO format data using merge sort.
 *
* @tparam SIZE_TYPE Type for indexing (e.g., unsigned int).
* @tparam VALUE_TYPE Type for values stored in the matrix (e.g., float, double).
 *@tparam SIZE_TYPE_2 Type for the overall size of the matrix (e.g., unsigned int).
*
 *@param cols Pointer to the column index array.
* @param rows Pointer to the row index array.
 *@param vals Pointer to the value array.
* @param size Overall size of the matrix.
 *
* @return Total number of duplicates removed during sorting.
 */
template <typename INDEX_ARRAYS, typename VALUE_ARRAYS>
std::size_t merge_sort_coo(INDEX_ARRAYS &indices, VALUE_ARRAYS &values, std::size_t size) {
    int duplicates = 0;
#pragma omp parallel
#pragma omp single
    duplicates+=recursive_merge_sort_coo(indices, values, 0, (std::size_t)size - 1, 0);

    return duplicates;
}

//-------EXTERNAL SORT---------

//use these functions to sort the synapse COO by importance and then delete the least important for synaptic pruning
// then use the above sort to coalesce for conversion to CSR

/**
*@brief Merges two sorted subarrays of a COO matrix based on an external array.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@tparam EXTERNAL_TYPE Type for values in the external sorting array (e.g., double)
*@param cols Pointer to column indices
*@param rows Pointer to row indices
*@param vals Pointer to values in the COO matrix
*@param ext_vals Pointer to the external sorting array
*@param left Left index of the first subarray
*@param mid Middle index separating the two subarrays
*@param right Right index of the second subarray
*
*This function merges two sorted subarrays within the COO matrix based on the corresponding values in the external sorting array.
*/
/*template <typename SIZE_TYPE, typename VALUE_TYPE, typename EXTERNAL_TYPE>
SIZE_TYPE inplace_merge_coo_external(SIZE_TYPE *cols,
                            SIZE_TYPE *rows,
                            VALUE_TYPE *vals,
                            EXTERNAL_TYPE *ext_vals,
                            SIZE_TYPE left,
                            SIZE_TYPE mid,
                            SIZE_TYPE right) {
    SIZE_TYPE i = left;
    SIZE_TYPE j = mid + 1;

    // Traverse both subarrays and merge in-place
    while (i <= mid && j <= right) {
        // If the element in the left half is smaller, move on
        if (ext_vals[i] <= ext_vals[j]) {
            i++;
        } else if (ext_vals[i] > ext_vals[j]) {
            // Element in the right half is smaller, rotate to the left
            SIZE_TYPE temp_row = rows[j];
            SIZE_TYPE temp_col = cols[j];
            VALUE_TYPE temp_val = vals[j];
            VALUE_TYPE temp_ext_val = ext_vals[j];

            // Shift elements from i to j-1 one position to the right
            for (SIZE_TYPE k = j; k > i; k--) {
                rows[k] = rows[k - 1];
                cols[k] = cols[k - 1];
                vals[k] = vals[k - 1];
                ext_vals[k] = ext_vals[k - 1];
            }

            // Place the j-th element in its correct position
            rows[i] = temp_row;
            cols[i] = temp_col;
            vals[i] = temp_val;
            ext_vals[i] = temp_ext_val;

            // Update indices
            i++;
            mid++;
            j++;
        }
    }
    return;
}*/

/**
*@brief Sorts a COO matrix based on an external array using insertion sort.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@tparam EXTERNAL_TYPE Type for values in the external sorting array (e.g., double)
*@param cols Pointer to column indices
*@param rows Pointer to row indices
*@param vals Pointer to values in the COO matrix
*@param ext_vals Pointer to the external sorting array
*@param left Left index of the subarray
*@param right Right index of the subarray
*
*This function sorts a subarray within the COO matrix based on the corresponding values in the external sorting array using insertion sort.
*/
/*template <typename SIZE_TYPE, typename VALUE_TYPE, typename EXTERNAL_TYPE>
void insertion_sort_coo_external(SIZE_TYPE *cols,
                                 SIZE_TYPE *rows,
                                 VALUE_TYPE *vals,
                                 EXTERNAL_TYPE *ext_vals,
                                 SIZE_TYPE left,
                                 SIZE_TYPE right) {
    for (unsigned long i = left + 1; i <= right; i++) {
        SIZE_TYPE temp_row = rows[i];
        SIZE_TYPE temp_col = cols[i];
        VALUE_TYPE temp_val = vals[i];
        EXTERNAL_TYPE temp_ext_val = ext_vals[i];

        SIZE_TYPE j = i;
        while (j > left && (ext_vals[j - 1] > temp_ext_val)) {
            rows[j] = rows[j - 1];
            cols[j] = cols[j - 1];
            vals[j] = vals[j - 1];
            ext_vals[j] = ext_vals[j - 1];
            j--;
        }
        // Insert (temp_row, temp_col, temp_val) at its correct position
        rows[j] = temp_row;
        cols[j] = temp_col;
        vals[j] = temp_val;
        ext_vals[j] = temp_ext_val;
    }
    return;
}*/

/**
*@brief Recursively sorts a COO matrix based on an external array using merge sort.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@tparam EXTERNAL_TYPE Type for values in the external sorting array (e.g., double)
*@param cols Pointer to column indices
*@param rows Pointer to row indices
*@param vals Pointer to values in the COO matrix
*@param ext_vals Pointer to the external sorting array
*@param left Left index of the subarray
*@param right Right index of the subarray
*
*This function recursively sorts a subarray within the COO matrix based on the corresponding values in the external sorting array using merge sort.
*/
/*template <typename SIZE_TYPE, typename VALUE_TYPE, typename EXTERNAL_TYPE>
void recursive_merge_sort_coo_external(SIZE_TYPE *cols,
                                       SIZE_TYPE *rows,
                                       VALUE_TYPE *vals,
                                       EXTERNAL_TYPE *ext_vals,
                                       SIZE_TYPE left,
                                       SIZE_TYPE right) {
    if (left < right) {
        if (right - left >= 32) {
            SIZE_TYPE mid = (left + right) / 2;
#pragma omp taskgroup
            {
#pragma omp task shared(indices, values) untied if (right - left >= (1 << 14))
                recursive_merge_sort_coo(cols, rows, vals, ext_vals, left, mid);
#pragma omp task shared(indices, values) untied if (right - left >= (1 << 14))
                recursive_merge_sort_coo(cols, rows, vals, ext_vals, mid + 1, right);
#pragma omp taskyield
            }

            inplace_merge_coo_external(cols, rows, vals, ext_vals, left, mid, right);
        } else {
            insertion_sort_coo_external(cols, rows, vals, ext_vals, left, right);
        }
    }
    return;
}*/

/**
*@brief Sorts a COO matrix based on an external array using merge sort.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@tparam EXTERNAL_TYPE Type for values in the external sorting array (e.g., double)
*@param cols Pointer to column indices
*@param rows Pointer to row indices
*@param vals Pointer to values in the COO matrix
*@param ext_vals Pointer to the external sorting array
*
*This function sorts the entire COO matrix based on the corresponding values in the external sorting array using merge sort.
*/
/*template <typename SIZE_TYPE, typename VALUE_TYPE, typename EXTERNAL_TYPE>
void merge_sort_coo_external(SIZE_TYPE *cols, SIZE_TYPE *rows, VALUE_TYPE *vals, EXTERNAL_TYPE *ext_vals, SIZE_TYPE size) {
#pragma omp parallel
#pragma omp single
    recursive_merge_sort_coo(cols, rows, vals, 0, (SIZE_TYPE)size - 1, 0);

    return;
}*/

/*-----------------------------------------MERGE SORTED COOs-------------------------------------------*/

/**
 * @brief Performs a binary search on a COO matrix to find the appropriate position for a given index pair.
 *
 * @tparam INDEX_TYPE Type for indexing (e.g., unsigned int)
 * @tparam VALUE_TYPE Placeholder type for values (not used here)
 * @param indices Tuple of index arrays
 * @param target_indices Tuple of target index values
 * @param low Lower bound of the search range
 * @param high Upper bound of the search range
 * @return The position where the target indices should be inserted.
 */
template <typename INDEX_ARRAYS>
std::size_t binary_search_coo(const INDEX_ARRAYS &indices,
                              const std::array<std::remove_pointer_t<decltype(indices[0].get())>, std::tuple_size<INDEX_ARRAYS>::value> &target_indices,  // should this be size_t, size instead?
                              std::size_t low, std::size_t high) {
    constexpr std::size_t numIndices = std::tuple_size<INDEX_ARRAYS>::value;

    while (low < high) {
        std::size_t mid = (low + high) / 2;
        bool is_less = false;

        for (std::size_t idx = 0; idx < numIndices; ++idx) {
            auto current = indices[idx][mid];
            auto target = target_indices[idx];
            if (current < target) {
                is_less = true;
                break;
            } else if (current > target) {
                break;
            }
        }

        if (is_less)
            low = mid + 1;
        else
            high = mid;
    }

    return low;
}

template <typename INDEX_ARRAYS>
std::size_t binary_search_upper_bound_coo(
    const INDEX_ARRAYS &indices,
    const std::array<std::remove_pointer_t<decltype(indices[0].get())>, std::tuple_size<INDEX_ARRAYS>::value> &target_indices,
    std::size_t low, std::size_t high) {
    constexpr std::size_t numIndices = std::tuple_size<INDEX_ARRAYS>::value;

    while (low < high) {
        std::size_t mid = (low + high) / 2;
        bool is_less_or_equal = true;

        for (std::size_t idx = 0; idx < numIndices; ++idx) {
            auto current = indices[idx][mid];
            auto target = target_indices[idx];
            if (current < target) {
                is_less_or_equal = true;
                break;
            } else if (current > target) {
                is_less_or_equal = false;
                break;
            }
        }

        if (is_less_or_equal)
            low = mid + 1; // Proceed to the right
        else
            high = mid;     // Move to the left
    }

    return low; // Returns the first index where element > target
}

/**
*@brief Merges two sorted COO matrices into a single sorted COO matrix using multi-threading.
*
*@tparam SIZE_TYPE Type for indexing (e.g., unsigned int)
*@tparam VALUE_TYPE Type for values stored in the COO matrix (e.g., float)
*@param m_rows Pointer to row indices of the first COO matrix
*@param m_cols Pointer to column indices of the first COO matrix
*@param m_vals Pointer to values of the first COO matrix
*@param n_rows Pointer to row indices of the second COO matrix
*@param n_cols Pointer to column indices of the second COO matrix
*@param n_vals Pointer to values of the second COO matrix
*@param c_rows Pointer to row indices of the resulting COO matrix
*@param c_cols Pointer to column indices of the resulting COO matrix
*@param c_vals Pointer to values of the resulting COO matrix
*@param m_size Size of the first COO matrix
*@param n_size Size of the second COO matrix
*@param num_threads Number of threads to use for parallel processing
*
*This function merges two sorted COO matrices into a single sorted COO matrix using multi-threading.
*/
template <typename INDEX_ARRAYS, typename VALUE_ARRAYS>
std::size_t parallel_merge_sorted_coos(INDEX_ARRAYS &m_indices, VALUE_ARRAYS &m_values,
                                  INDEX_ARRAYS &n_indices, VALUE_ARRAYS &n_values,
                                  INDEX_ARRAYS &c_indices, VALUE_ARRAYS &c_values,
                                  std::size_t m_size, std::size_t n_size, int num_threads) {
    //todo: make a csr variant of this.
    constexpr std::size_t numValues = std::tuple_size<VALUE_ARRAYS>::value;
    //using VALUE_TYPE = VALUE_ARRAYS::value_type;
    using VALUE_TYPE = std::remove_pointer_t<decltype(m_values[0].get())>;
    constexpr std::size_t numIndices = std::tuple_size<INDEX_ARRAYS>::value;
    //using INDEX_TYPE = VALUE_ARRAYS::value_type;
    using INDEX_TYPE = std::remove_pointer_t<decltype(m_indices[0].get())>;

    std::vector<std::size_t> duplicates_per_thread(num_threads, 0);

    std::vector<std::size_t> n_begins(num_threads, 0);
    std::vector<std::size_t> n_ends(num_threads, 0);

    // Step 1: Determine chunk ranges for m and corresponding n ranges
    #pragma omp parallel num_threads(num_threads) shared(n_begins, n_ends, duplicates_per_thread)
    {
        int thread_id = omp_get_thread_num();
        size_t chunk_size = (m_size + num_threads - 1) / num_threads; // Ceiling division

        size_t m_begin = thread_id * chunk_size;
        size_t m_end = std::min(m_begin + chunk_size, m_size);

        if(m_begin<m_end){
        std::array<INDEX_TYPE, numIndices> m_begin_indices;
        std::array<INDEX_TYPE, numIndices> m_end_indices;

        for (std::size_t idx = 0; idx < numIndices; ++idx) {
            m_begin_indices[idx] = m_indices[idx][m_begin];
            m_end_indices[idx] = m_indices[idx][m_end - 1];
        }

        n_begins[thread_id] = binary_search_coo(n_indices, m_begin_indices, 0, n_size);
        n_ends[thread_id] = binary_search_upper_bound_coo(n_indices, m_end_indices, 0, n_size);

        size_t c_start = m_begin + n_begins[thread_id];

        // Merge subarrays into c
        size_t i = m_begin, j = n_begins[thread_id], k = c_start;
        while (i < m_end && j < n_ends[thread_id]) {
            bool m_lt_n = false, m_eq_n = true;

            for (std::size_t idx = 0; idx < numIndices; ++idx) {
                auto m_val = m_indices[idx][i];
                auto n_val = n_indices[idx][j];
                if (m_val < n_val) {
                    m_lt_n = true;
                    m_eq_n = false;
                    break;
                } else if (m_val > n_val) {
                    m_eq_n = false;
                    break;
                }
            }

            if (m_lt_n || m_eq_n) { // todo: split out m_eq_n and have it do i++ and j++ and sum std::get<valIdx>(m_values)[i] + std::get<valIdx>(m_values)[j] for a slight speed boost
                bool is_duplicate = false;
                if (k > 0) {
                    is_duplicate = true;
                    for (std::size_t idx = 0; idx < numIndices; ++idx) {
                        if (c_indices[idx][k - 1] != m_indices[idx][i]) {
                            is_duplicate = false;
                            break;
                        }
                    }
                    
                    if (is_duplicate) {
                        for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                            c_values[valIdx][k - 1] += m_values[valIdx][i];
                        }
                        duplicates_per_thread[thread_id]++;
                    }
                }
                if(!is_duplicate){
                    for (std::size_t idx = 0; idx < numIndices; ++idx) {
                        c_indices[idx][k] = m_indices[idx][i];
                    }
                    for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                        c_values[valIdx][k] = m_values[valIdx][i];
                    }
                    k++;
                }
                i++;
            } else {
                bool is_duplicate = false;
                if (k > 0) {
                    is_duplicate = true;
                    for (std::size_t idx = 0; idx < numIndices; ++idx) {
                        if (c_indices[idx][k - 1] != n_indices[idx][j]) {
                            is_duplicate = false;
                            break;
                        }
                    }
                    if (is_duplicate) {
                        for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                            c_values[valIdx][k - 1] += n_values[valIdx][j];
                        }
                        duplicates_per_thread[thread_id]++;
                    }
                } if(!is_duplicate) {
                        for (std::size_t idx = 0; idx < numIndices; ++idx) {
                            c_indices[idx][k] = n_indices[idx][j];
                        }
                        for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                            c_values[valIdx][k] = n_values[valIdx][j];
                        }
                        k++;
                    }
                    j++;
                }
            }
        while (i < m_end) {
            bool is_duplicate = false;
            if (k > 0) {
                is_duplicate = true;
                for (std::size_t idx = 0; idx < numIndices; ++idx) {
                    if (c_indices[idx][k - 1] != m_indices[idx][i]) {
                        is_duplicate = false;
                        break;
                    }
                }
                if (is_duplicate) {
                    for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                        c_values[valIdx][k - 1] += m_values[valIdx][i];
                    }
                    duplicates_per_thread[thread_id]++;
                }
            }
            if(!is_duplicate){
                for (std::size_t idx = 0; idx < numIndices; ++idx) {
                    c_indices[idx][k] = m_indices[idx][i];
                }
                for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                    c_values[valIdx][k] = m_values[valIdx][i];
                }
                k++;
            }
            i++;
        }
        while (j < n_ends[thread_id]) {
            bool is_duplicate = false;
            if (k > 0) {
                is_duplicate = true;
                for (std::size_t idx = 0; idx < numIndices; ++idx) {
                    if (c_indices[idx][k - 1] != n_indices[idx][j]) {
                        is_duplicate = false;
                        break;
                    }
                }
                if (is_duplicate) {
                    for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                        c_values[valIdx][k - 1] += n_values[valIdx][j];
                    }
                    duplicates_per_thread[thread_id]++;
                }
            } 
            if(!is_duplicate) {
                for (std::size_t idx = 0; idx < numIndices; ++idx) {
                    c_indices[idx][k] = n_indices[idx][j];
                }
                for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                    c_values[valIdx][k] = n_values[valIdx][j];
                }
                k++;
            }
            j++;
        }
        }
    }

    // step 1.5: scan the duplicates
    std::vector<std::size_t> scanned_duplicates(num_threads+1, 0);
    fullScanValues(duplicates_per_thread, scanned_duplicates);

    // step 1.6: shift the chunks
    #pragma omp parallel num_threads(num_threads) shared(scanned_duplicates)
    {
        int thread_id = omp_get_thread_num();
        size_t chunk_size = (m_size + num_threads - 1) / num_threads;
        size_t m_begin = thread_id * chunk_size;
        size_t m_end = std::min(m_begin + chunk_size, m_size);
        if(m_begin<m_end){
            size_t c_end = n_ends[thread_id] + m_end;
            size_t c_begin = n_begins[thread_id] + m_begin;

            size_t shift = scanned_duplicates[thread_id];

            for (size_t i = c_begin; i < c_end; ++i) {
                size_t new_position = i - shift;
                for (std::size_t idx = 0; idx < numIndices; ++idx) {
                    c_indices[idx][new_position] = c_indices[idx][i];
                }
                for (std::size_t valIdx = 0; valIdx < numValues; ++valIdx) {
                    c_values[valIdx][new_position] = c_values[valIdx][i];
                }
            }
        }
    }

    // Step 2: Copy remaining elements from n to c
    #pragma omp parallel num_threads(num_threads) shared(scanned_duplicates)
    {
        int thread_id = omp_get_thread_num();
        size_t chunk_size = (n_begins[0] + num_threads - 1) / num_threads;
        size_t start = thread_id * chunk_size;
        size_t end = std::min(start + chunk_size, n_begins[0]);

        for (size_t idx = start; idx < end; ++idx) {
            for (std::size_t j = 0; j < numIndices; ++j) {
                c_indices[j][idx] = n_indices[j][idx];
            }
            for (std::size_t j = 0; j < numValues; ++j) {
                c_values[j][idx] = n_values[j][idx];
            }
        }

        size_t final_end = n_ends[std::min((size_t)num_threads-1, std::min(n_size-1, m_size-1))];

        chunk_size = (n_size - final_end + num_threads - 1) / num_threads;
        size_t c_start = m_size - scanned_duplicates[num_threads] + final_end + thread_id * chunk_size;
        size_t c_end = std::min(c_start + chunk_size, m_size + n_size - scanned_duplicates[num_threads]);
        size_t n_start = final_end + thread_id * chunk_size;
        size_t n_end = std::min(n_start + chunk_size,  n_size);

        if(n_end > n_start){  // this prevents segfaults from unsigned types
            for (size_t idx = 0; idx < n_end-n_start; ++idx) {
                size_t n_idx = idx + n_start;
                size_t c_idx = idx + c_start;
                for (std::size_t j = 0; j < numIndices; ++j) {
                    c_indices[j][c_idx] = n_indices[j][n_idx];
                }
                for (std::size_t j = 0; j < numValues; ++j) {
                    c_values[j][c_idx] = n_values[j][n_idx];
                }
            }
        }
    }

    return scanned_duplicates[num_threads];
}

/*
int main() {
    // Example COO arrays
    std::vector<int> m_rows = {1, 3, 5};
    std::vector<int> m_cols = {1, 2, 3};
    std::vector<double> m_vals = {10.0, 20.0, 30.0};

    std::vector<int> n_rows = {2, 3, 6};
    std::vector<int> n_cols = {1, 2, 3};
    std::vector<double> n_vals = {15.0, 25.0, 35.0};

    size_t m_size = m_rows.size();
    size_t n_size = n_rows.size();
    size_t c_size = m_size + n_size;

    std::vector<int> c_rows(c_size);
    std::vector<int> c_cols(c_size);
    std::vector<double> c_vals(c_size);

    int num_threads = 4; // Example: 4 threads

    size_t duplicates = parallel_merge_coo(m_rows.data(), m_cols.data(), m_vals.data(),
                                           n_rows.data(), n_cols.data(), n_vals.data(),
                                           c_rows.data(), c_cols.data(), c_vals.data(),
                                           m_size, n_size, num_threads);

    // Output the merged COO array
    for (size_t i = 0; i < c_size - duplicates; i++) {
        std::cout << "(" << c_rows[i] << ", " << c_cols[i] << ") -> " << c_vals[i] << "\n";
    }

    return 0;
}
*/

/*------------------------SUBTRACT BOTTOM K FROM COOs (external importance array)----------------------------------*/

/**
*@brief Finds the bottom-k indices of an array in parallel.
*
*@tparam VALUE_TYPE Type for values stored in the array (e.g., float)
*@param values Pointer to the array of external values
*@param size The size of the input array
*@param k The number of smallest elements to find
*@param num_threads Number of threads to use for parallel processing
*
*This function finds the bottom-k indices of an array in parallel.
*/
template <typename VALUE_TYPE>
std::vector<size_t> bottom_k_indices(VALUE_TYPE *values, size_t size, size_t k, int num_threads) {    
    // Each thread processes a chunk of the array
    size_t chunk_size = (size + num_threads - 1) / num_threads;
    std::vector<std::vector<size_t>> thread_indices(num_threads);

    if(k>size){
        k=size;
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        size_t start = thread_id * chunk_size;
        size_t end = std::min(start + chunk_size, size);

        // Collect indices for this thread
        std::vector<size_t> local_indices;
        for (size_t i = start; i < end; ++i) {
            local_indices.push_back(i);
        }

        // Sort local indices by values
        std::partial_sort(local_indices.begin(), local_indices.begin() + std::min(k, local_indices.size()), local_indices.end(),
                          [&values](size_t a, size_t b) { return values[a] < values[b]; });

        // Keep only the smallest k elements
        if (local_indices.size() > k) {
            local_indices.resize(k);
        }

        thread_indices[thread_id] = std::move(local_indices);
    }

    // Merge results from all threads
    std::vector<size_t> merged_indices;
    for (const auto &indices : thread_indices) {
        merged_indices.insert(merged_indices.end(), indices.begin(), indices.end());
    }

    // Find the global bottom-k indices
    std::partial_sort(merged_indices.begin(), merged_indices.begin() + k, merged_indices.end(),
                      [&values](size_t a, size_t b) { return values[a] < values[b]; });

    merged_indices.resize(k);
    return merged_indices;
}

/**
 * @brief Removes bottom k elements based on specified values and copies the result into another COO array in parallel.
 *
 * @tparam INDEX_ARRAYS Tuple type containing index arrays
 * @tparam VALUE_ARRAYS Tuple type containing value arrays
 * @tparam VALUE_INDEX Index of the value array to use for finding bottom-k (default: -1, uses the last value array)
 * @param indices Tuple containing input column and row indices arrays
 * @param values Tuple containing input value arrays
 * @param c_indices Tuple containing output column and row indices arrays
 * @param c_values Tuple containing output value arrays
 * @param size The size of the input COO array
 * @param k Number of elements to remove based on the smallest values
 * @param num_threads Number of threads to use for parallel processing
 */
template <typename INDEX_ARRAYS, typename VALUE_ARRAYS, int VALUE_INDEX = -1>
void coo_subtract_bottom_k(INDEX_ARRAYS &indices, VALUE_ARRAYS &values,
                           INDEX_ARRAYS &c_indices, VALUE_ARRAYS &c_values,
                           size_t size, size_t k, int num_threads) {
    //todo: make a csr variant of this.

    constexpr size_t numIndices = std::tuple_size<INDEX_ARRAYS>::value;
    constexpr size_t numValues = std::tuple_size<VALUE_ARRAYS>::value;
    using INDEX_TYPE = std::remove_pointer_t<decltype(indices[0].get())>;
    using VALUE_TYPE = std::remove_pointer_t<decltype(values[0].get())>;

    // Determine the array to base the bottom-k selection on
    constexpr int selectedIndex = (VALUE_INDEX == -1) ? (numValues - 1) : VALUE_INDEX;
    static_assert(selectedIndex >= 0 && selectedIndex < numValues, "Invalid VALUE_INDEX specified.");

    // Select the array for bottom-k computation
    auto &selected_values = values[selectedIndex];

    // Step 1: Find bottom-k indices in parallel
    std::vector<size_t> bottom_k_vec = bottom_k_indices(selected_values.get(), size, k, num_threads);

    // Step 2: Copy COO elements to the output array, skipping bottom-k
    #pragma omp parallel for num_threads(num_threads)
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        size_t start = thread_id * (size / num_threads);
        size_t end = (thread_id == num_threads - 1) ? size : (thread_id + 1) * (size / num_threads);

        size_t bottom_k_index = std::lower_bound(bottom_k_vec.begin(), bottom_k_vec.end(), start) - bottom_k_vec.begin();

        size_t c_index = start - bottom_k_index;
        for (size_t i = start; i < end; ++i) {
            if (bottom_k_index < bottom_k_vec.size() && bottom_k_vec[bottom_k_index] == i) {
                ++bottom_k_index;  // Skip this index
            } else {
                // Copy indices to output COO
                for (size_t idx = 0; idx < numIndices; ++idx) {
                    c_indices[idx][c_index] = indices[idx][i];
                }

                // Copy values to output COO
                for (size_t val = 0; val < numValues; ++val) {
                    c_values[val][c_index] = values[val][i];
                }

                ++c_index;
            }
        }
    }
}

// Main function demonstrating the use of coo_subtract_bottom_k
/*int main() {
    // Example input COO array
    const size_t size = 10;
    const size_t k = 3;
    const int num_threads = 4;

    size_t cols[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t rows[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float vals[size] = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    float ext_vals[size] = {0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0};

    size_t c_cols[size - k];
    size_t c_rows[size - k];
    float c_vals[size - k];
    float c_ext_vals[size - k];

    // Call the function
    coo_subtract_bottom_k(cols, rows, vals, ext_vals, c_cols, c_rows, c_vals, c_ext_vals, size, k, num_threads);

    // Output results
    for (size_t i = 0; i < size - k; ++i) {
        printf("Row: %zu, Col: %zu, Val: %.2f, Ext Val: %.2f\n", c_rows[i], c_cols[i], c_vals[i], c_ext_vals[i]);
    }

    return 0;
}*/

template <typename VALUE_TYPE>
std::vector<size_t> top_k_indices(VALUE_TYPE *values, size_t size, size_t k, int num_threads) {    
    // Each thread processes a chunk of the array
    size_t chunk_size = (size + num_threads - 1) / num_threads;
    std::vector<std::vector<size_t>> thread_indices(num_threads);

    if(k>size){
        k=size;
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        size_t start = thread_id * chunk_size;
        size_t end = std::min(start + chunk_size, size);

        // Collect indices for this thread
        std::vector<size_t> local_indices;
        for (size_t i = start; i < end; ++i) {
            local_indices.push_back(i);
        }

        // Sort local indices by values
        std::partial_sort(local_indices.begin(), local_indices.begin() + std::min(k, local_indices.size()), local_indices.end(),
                          [&values](size_t a, size_t b) { return values[a] > values[b]; });

        // Keep only the smallest k elements
        if (local_indices.size() > k) {
            local_indices.resize(k);
        }

        thread_indices[thread_id] = std::move(local_indices);
    }

    // Merge results from all threads
    std::vector<size_t> merged_indices;
    for (const auto &indices : thread_indices) {
        merged_indices.insert(merged_indices.end(), indices.begin(), indices.end());
    }

    // Find the global bottom-k indices
    std::partial_sort(merged_indices.begin(), merged_indices.begin() + k, merged_indices.end(),
                      [&values](size_t a, size_t b) { return values[a] > values[b]; });

    merged_indices.resize(k);
    return merged_indices;
}

template <class SIZE_TYPE, class VALUE_TYPE>
sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>>
top_k_coo(VALUE_TYPE *values, size_t rows, size_t cols, size_t k, int num_threads) {
    // Step 1: Get the top-k indices
    std::vector<size_t> top_k = top_k_indices(values, rows * cols, k, num_threads);

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

    sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE>> coo_result(
        ptrs, indices, coo_values, rows, cols, k);

    return coo_result;
}



#endif