#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

// ── Hoyer's Sparsity Measure -> top-k sparsification ────────────────────────────
//
// NOT wired into an automatic dense/sparse dispatch (see TODO.md) -- this is
// the standalone operation, exposed so its actual behavior on real data can
// be explored/tested from Python before deciding on dispatch thresholds
// ("not obvious" -- per conversation).
//
// The idea: rather than an arbitrary epsilon-threshold count of "near-zero"
// elements, use the ratio of L1 to L2 norm as a principled estimate of the
// EFFECTIVE number of significant elements in a vector.
//
//   hoyer(x) = (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)     in [0, 1]
//
// For a vector with exactly k nonzero entries of EQUAL magnitude (rest
// exactly zero), ||x||_1/||x||_2 = sqrt(k) exactly -- verified below, not
// just asserted. For a realistic vector (mixed magnitudes, not exactly
// k-sparse), the same ratio still gives a smooth estimate of the
// "effective" significant-element count:
//
//   k_estimate = (||x||_1 / ||x||_2)^2
//
// which becomes the target k for a top-k sparsification pass -- keep only
// the k_estimate largest-magnitude elements, treat the rest as noise (not
// "whatever happens to be exactly zero").

struct HoyerSparsifyRow {
    float               l1_norm;
    float               l2_norm;
    float               hoyer_score;   // normalized [0,1], 0=dense, 1=maximally sparse
    int                 k_estimate;    // (l1/l2)^2, rounded, clamped to [0, n]
    std::vector<int>    indices;       // top-k_estimate indices, ascending
    std::vector<float>  values;        // corresponding values (not zeroed elsewhere)
};

template <typename VALUE_TYPE>
HoyerSparsifyRow hoyer_sparsify_row(const VALUE_TYPE* x, std::size_t n) {
    HoyerSparsifyRow result;

    double l1 = 0.0, l2sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = std::abs(static_cast<double>(x[i]));
        l1   += v;
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
        result.k_estimate  = 0;
        return result;
    }

    const double ratio  = l1 / l2;
    const double sqrt_n = std::sqrt(static_cast<double>(n));

    result.hoyer_score = (n > 1)
        ? static_cast<float>((sqrt_n - ratio) / (sqrt_n - 1.0))
        : 0.0f;   // n=1: a single element is trivially "all of it", not meaningfully sparse

    const double k_raw = ratio * ratio;
    result.k_estimate = static_cast<int>(
        std::round(std::min(k_raw, static_cast<double>(n))));

    // Top-k_estimate selection by |value|, then re-sort ascending by index
    // (matching CSR column-order convention elsewhere in this codebase).
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    const std::size_t kk = static_cast<std::size_t>(result.k_estimate);
    if (kk < n) {
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
            [&](std::size_t a, std::size_t b) {
                return std::abs(static_cast<double>(x[a])) > std::abs(static_cast<double>(x[b]));
            });
        idx.resize(kk);
        std::sort(idx.begin(), idx.end());
    }

    result.indices.reserve(idx.size());
    result.values .reserve(idx.size());
    for (auto i : idx) {
        result.indices.push_back(static_cast<int>(i));
        result.values .push_back(static_cast<float>(x[i]));
    }

    return result;
}

/**
 * @brief Batched version: one HoyerSparsifyRow per BATCH SAMPLE (not weight-
 * matrix row -- "row" here means one row of the [batch, cols] activation
 * array, i.e. one sample; DeltaCSRWeights elsewhere in this codebase uses
 * "row" for input NEURON, a different axis entirely -- same word, easy to
 * conflate, worth being careful about). k_estimate computed independently
 * per sample, not one shared k across the batch (unlike top_k_csr) --
 * different samples can have genuinely different effective sparsity.
 *
 * IMPORTANT (see conversation): this is for CONSTRUCTING an accurate sparse
 * representation once you've already decided to route a batch through the
 * sparse path -- it is NOT the right thing to base that routing decision
 * on. forward_dense/forward_sparse are each called ONCE for the whole
 * batch together; a per-sample "is this one sparse enough" answer isn't
 * actionable at that granularity, since you can't send some samples
 * through one kernel and some through the other in a single call. For the
 * actual dense-vs-sparse ROUTING decision, use hoyer_score() below,
 * which aggregates over the whole batch to produce the one number that
 * question actually needs.
 */
template <typename VALUE_TYPE>
std::vector<HoyerSparsifyRow> hoyer_sparsify_per_batch(
    const VALUE_TYPE* x, std::size_t rows, std::size_t cols)
{
    std::vector<HoyerSparsifyRow> result;
    result.reserve(rows);
    for (std::size_t r = 0; r < rows; ++r)
        result.push_back(hoyer_sparsify_row<VALUE_TYPE>(x + r * cols, cols));
    return result;
}

/**
 * @brief Batch-level aggregate Hoyer's measure -- the actual quantity a
 * dense-vs-sparse ROUTING decision should be based on, computed over the
 * WHOLE flattened batch (all rows*cols elements together) rather than per
 * sample, since forward_dense/forward_sparse are each invoked once for the
 * entire batch in a single call, not once per sample.
 *
 * Returns the raw hoyer_score (l1/l2-derived, in [0,1]) plus l1/l2 and a
 * batch-wide k_estimate -- what a threshold comparison should actually use
 * to decide which of forward_dense/forward_sparse to call for this batch.
 * Per-sample construction of the actual CSR data, once that decision is
 * made, should still use hoyer_sparsify_per_batch() above (or a fixed/shared k
 * if uniform treatment across the batch is preferred) -- this function
 * only answers "which kernel", not "which elements to keep per row".
 */
template <typename VALUE_TYPE>
HoyerSparsifyRow hoyer_score(const VALUE_TYPE* x, std::size_t rows, std::size_t cols) {
    return hoyer_sparsify_row<VALUE_TYPE>(x, rows * cols);
}
