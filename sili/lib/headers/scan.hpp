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
template <class T> void fullScanSizes(const std::vector<std::vector<T>> &vec_of_vec, std::vector<size_t>&fullScan, int&& scan_a=0) {
    
#ifdef __clang__ // OMP scan is broken in clang and may crash it: https://github.com/llvm/llvm-project/issues/87466
    std::inclusive_scan(
        vec_of_vec.begin(),
        vec_of_vec.end(),
        fullScan.begin() + 1,
        [](const size_t &cum_sum, const std::vector<T> &vec) { return cum_sum + vec.size(); },
        0);
#else

#pragma omp for simd reduction(inscan, + : scan_a)
    for (int i = 0; i < vec_of_vec.size()+1; i++) {
        fullScan[i] = scan_a;
        #pragma omp scan exclusive(scan_a)
        {
            if(i<vec_of_vec.size()){
                scan_a += vec_of_vec[i].size();
            }else{
                scan_a += vec_of_vec[i-1].size();
            }
        }
        
    }
# pragma omp barrier
#endif
}

/**
 * Computes cumulative sums of the sizes of inner inner vectors using OpenMP.
 *
 * @tparam T The type of elements in the inner inner vectors.
 * @param vec_of_vec_of_vec A vector of vectors of type T.
 * @return A vector of size_t with cumulative sizes, one element larger than the input.
 */
template <class T>
void fullScanSizes2(const std::vector<std::vector<std::vector<T>>> &vec_of_vec_of_vec, std::vector<std::vector<size_t>>& fullScans) {
    for (int i = 0; i < vec_of_vec_of_vec.size(); i++) {
        fullScanSizes(vec_of_vec_of_vec[i], fullScans[i]);
    }
}

/**
 * Computes a cumulative sum of the sizes of inner vectors using OpenMP.
 *
 * @tparam T The type of elements in the inner vectors.
 * @param vec_of_vec A vector of vectors of type T.
 * @return A vector of size_t with cumulative sizes, one element larger than the input.
 */
template <class CONTAINER> void fullScanValues(const CONTAINER &vec, CONTAINER& fullScan, typename CONTAINER::value_type&& scan_a = typename CONTAINER::value_type{}) {

#ifdef __clang__ // OMP scan is broken in clang and may crash it: https://github.com/llvm/llvm-project/issues/87466
    std::inclusive_scan(
        vec.begin(),
        vec.end(),
        fullScan.begin() + 1,
        std::plus<typename CONTAINER::value_type>()
    );
#else

#pragma omp for simd reduction(inscan, + : scan_a)
    for (int i = 0; i < vec.size()+1; i++) {
        fullScan[i] = scan_a;
        #pragma omp scan exclusive(scan_a)
        {
            if(i<vec.size()){
                scan_a += vec[i];
            }else{
                scan_a += vec[i-1];
            }
        }

    }
# pragma omp barrier
#endif
}

#endif