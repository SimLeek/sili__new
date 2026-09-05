#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

// Hoyer's Sparsity Measure -> top-k sparsification. NOT wired into an
// automatic dense/sparse dispatch (see TODO.md) -- standalone, exposed
// for exploration before deciding dispatch thresholds. See
// docs/research/hoyer_sparsify.rst:hoyer_sparsify.math_derivation for
// the L1/L2-ratio derivation of k_estimate.

// No ctor, but every scalar field below is unconditionally assigned on
// every branch of hoyer_sparsify_row before any return -- cppcheck can't
// see across the early-return branch, false positive.
struct HoyerSparsifyRow {
    // cppcheck-suppress uninitMemberVarNoCtor
    float l1_norm;
    // cppcheck-suppress uninitMemberVarNoCtor
    float l2_norm;
    // cppcheck-suppress uninitMemberVarNoCtor
    float hoyer_score; // normalized [0,1], 0=dense, 1=maximally sparse
    // cppcheck-suppress uninitMemberVarNoCtor
    int k_estimate;            // (l1/l2)^2, rounded, clamped to [0, n]
    std::vector<int> indices;  // top-k_estimate indices, ascending
    std::vector<float> values; // corresponding values (not zeroed elsewhere)
};

template <typename VALUE_TYPE>
HoyerSparsifyRow hoyer_sparsify_row(const VALUE_TYPE* x, std::size_t n) {
    HoyerSparsifyRow result;

    double l1 = 0.0, l2sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = std::abs(static_cast<double>(x[i]));
        l1 += v;
        l2sq += v * v;
    }
    const double l2 = std::sqrt(l2sq);
    result.l1_norm = static_cast<float>(l1);
    result.l2_norm = static_cast<float>(l2);

    if (n == 0 || l2 <= 0.0) {
        // All-zero (or empty) input: nothing to select. Defined as
        // maximally sparse (score=1, k=0) rather than leaving NaN --
        // there's no meaningful "ratio" to compute, but "keep nothing" is
        // the only sensible top-k result either way.
        result.hoyer_score = 1.0f;
        result.k_estimate = 0;
        return result;
    }

    const double ratio = l1 / l2;
    const double sqrt_n = std::sqrt(static_cast<double>(n));

    result.hoyer_score =
        (n > 1) ? static_cast<float>((sqrt_n - ratio) / (sqrt_n - 1.0))
                : 0.0f; // n=1: a single element is trivially "all of it", not meaningfully sparse

    const double k_raw = ratio * ratio;
    result.k_estimate = static_cast<int>(std::round(std::min(k_raw, static_cast<double>(n))));

    // Top-k_estimate selection by |value|, then re-sort ascending by index
    // (matching CSR column-order convention elsewhere in this codebase).
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    const std::size_t kk = static_cast<std::size_t>(result.k_estimate);
    if (kk < n) {
        std::partial_sort(
            idx.begin(), idx.begin() + kk, idx.end(), [&](std::size_t a, std::size_t b) {
                return std::abs(static_cast<double>(x[a])) > std::abs(static_cast<double>(x[b]));
            });
        idx.resize(kk);
        std::sort(idx.begin(), idx.end());
    }

    result.indices.reserve(idx.size());
    result.values.reserve(idx.size());
    for (auto i : idx) {
        result.indices.push_back(static_cast<int>(i));
        result.values.push_back(static_cast<float>(x[i]));
    }

    return result;
}

/**
 * @brief Batched version: one HoyerSparsifyRow per BATCH SAMPLE, not
 * weight-matrix row -- see docs/research/hoyer_sparsify.rst:
 * hoyer_sparsify_per_batch.row_terminology_warning for the terminology
 * collision with DeltaCSRWeights' "row". For constructing sparse
 * representations, not for the dense-vs-sparse routing decision -- see
 * hoyer_sparsify_per_batch.not_for_routing_decision; use hoyer_score()
 * below for that.
 */
template <typename VALUE_TYPE>
std::vector<HoyerSparsifyRow> hoyer_sparsify_per_batch(const VALUE_TYPE* x, std::size_t rows,
                                                       std::size_t cols) {
    std::vector<HoyerSparsifyRow> result;
    result.reserve(rows);
    for (std::size_t r = 0; r < rows; ++r)
        result.push_back(hoyer_sparsify_row<VALUE_TYPE>(x + r * cols, cols));
    return result;
}

/**
 * @brief Batch-level aggregate Hoyer's measure -- the quantity a
 * dense-vs-sparse ROUTING decision should use, over the WHOLE flattened
 * batch. See docs/research/hoyer_sparsify.rst:
 * hoyer_score.batch_aggregate_rationale for why (forward_dense/
 * forward_sparse are invoked once per batch, not once per sample).
 */
template <typename VALUE_TYPE>
HoyerSparsifyRow hoyer_score(const VALUE_TYPE* x, std::size_t rows, std::size_t cols) {
    return hoyer_sparsify_row<VALUE_TYPE>(x, rows * cols);
}
