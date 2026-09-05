#ifndef __SCAN_HPP__
#define __SCAN_HPP__

#include <vector>
#ifdef __clang__
#include <numeric>
#endif

/**
 * Computes a cumulative sum of the sizes of inner vectors using OpenMP.
 *
 * @tparam T The type of elements in the inner vectors.
 * @param vec_of_vec A vector of vectors of type T.
 * @return A vector of size_t with cumulative sizes, one element larger than the input.
 */
template <class CONTAINER>
void fullScanValues(const CONTAINER& vec, CONTAINER& fullScan,
                    typename CONTAINER::value_type&& scan_a = typename CONTAINER::value_type{}) {

#ifdef __clang__ // OMP scan is broken in clang and may crash it:
                 // https://github.com/llvm/llvm-project/issues/87466
    std::inclusive_scan(vec.begin(), vec.end(), fullScan.begin() + 1,
                        std::plus<typename CONTAINER::value_type>());
#else

#pragma omp for simd reduction(inscan, + : scan_a)
    for (int i = 0; i < vec.size() + 1; i++) {
        fullScan[i] = scan_a;
#pragma omp scan exclusive(scan_a)
        {
            if (i < vec.size()) {
                scan_a += vec[i];
            } else if (i > 0) {
                // i == vec.size() here; guards the i==0 (empty vec) case,
                // where vec[i - 1] would be a negative/out-of-bounds
                // index (cppcheck negativeContainerIndex, task #386
                // followup) -- currently unreachable at this call site
                // (num_threads is always >= 1) but a real hole otherwise.
                scan_a += vec[i - 1];
            }
        }
    }
#pragma omp barrier
#endif
}

#endif