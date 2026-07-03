#ifndef __ATTENTION_HPP_
#define __ATTENTION_HPP_

// Attention ops ported from the pre-merge sparse_linear_ops.hpp.
// All three functions are self-contained (float pointers only, no SILi
// types) -- ported verbatim except for ASCII-only comments and the
// include guards. The pre-merge source is in test/python/ outputs dir
// for reference.
//
// Three variants, in increasing order of sparsity:
//   banded_attention_forward       -- dense within a geometric band
//   sparse_banded_attention_forward -- top-k within the band (data-driven)
//   sparse_attention_forward        -- global top-k query/key selection

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>


// ── Banded attention ──────────────────────────────────────────────────────────

/**
 * @brief Dense banded attention.
 *
 * The outer loop walks the geometric diagonal from (q=0,k=0) to
 * (q=T-1, k=K-1) — the same line used by delta_csr_init_zero_diagonal.
 * The inner loop covers +/- half_bandwidth keys around each diagonal point.
 *
 * Cost: O(T × half_bandwidth × d) — linear in T for fixed bandwidth.
 * This is the baseline to verify sparse_banded_attention_forward against.
 *
 * Layout: q, k_mat, v are row-major [T × d] float32.
 * output is [T × d], zeroed by caller.
 *
 * NOTE (test): With half_bandwidth ≥ T/2 and a square sequence, output should
 * match naive full attention (all pairs covered by the band).
 */
inline void banded_attention_forward(
    const float* q,
    const float* k_mat,
    const float* v,
    float*       output,
    std::size_t  T,
    std::size_t  K,       ///< Number of key positions (may differ from T for cross-attn)
    std::size_t  d,
    std::size_t  half_bandwidth,
    int          num_cpus = 4)
{
    if (T == 0 || K == 0 || d == 0) return;
    const float scale = 1.0f / std::sqrt(float(d));

    // Centre key index for query position t along the geometric diagonal.
    auto centre_k = [&](std::size_t t) -> std::size_t {
        if (T <= 1 || K <= 1) return 0;
        return (t * (K - 1) + (T - 1) / 2) / (T - 1);
    };

    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (std::size_t t = 0; t < T; ++t) {
        const float* qr   = q + t * d;
        float*       outr = output + t * d;

        const std::size_t ck = centre_k(t);
        const std::size_t k_lo = (ck > half_bandwidth) ? ck - half_bandwidth : 0;
        const std::size_t k_hi = std::min(ck + half_bandwidth, K - 1);

        // Compute scores for the band.
        const std::size_t band_w = k_hi - k_lo + 1;
        std::vector<float> scores(band_w);
        for (std::size_t bi = 0; bi < band_w; ++bi) {
            const float* kr = k_mat + (k_lo + bi) * d;
            float dot = 0.0f;
            for (std::size_t i = 0; i < d; ++i) dot += qr[i] * kr[i];
            scores[bi] = dot * scale;
        }

        // Softmax over the band.
        float max_s = *std::max_element(scores.begin(), scores.end());
        float sum_e = 0.0f;
        for (float& s : scores) { s = std::exp(s - max_s); sum_e += s; }
        const float inv = (sum_e > 0.0f) ? 1.0f / sum_e : 0.0f;
        for (float& s : scores) s *= inv;

        // Weighted sum of V.
        for (std::size_t bi = 0; bi < band_w; ++bi) {
            const float  w  = scores[bi];
            const float* vr = v + (k_lo + bi) * d;
            for (std::size_t i = 0; i < d; ++i) outr[i] += w * vr[i];
        }
    }
}


/**
 * @brief Sparse banded attention.
 *
 * Combines the geometric diagonal outer loop with the L2-norm top-k inner
 * selection of sparse_attention_forward.
 *
 * For each query position t:
 *   1. Walk the geometric diagonal to find the band centre key c_k(t).
 *   2. Gather all keys in [c_k - half_bandwidth, c_k + half_bandwidth].
 *   3. Within that band, keep only the top-inner_k keys by L2 norm.
 *   4. Compute dot-products for those inner_k keys only.
 *   5. Sparse softmax (non-selected keys in the band → -inf → weight 0).
 *
 * Cost: O(T × inner_k × d).
 * The band provides locality; inner_k provides data-driven sparsity within it.
 *
 * When inner_k ≥ 2×half_bandwidth+1 this degenerates to dense banded attention.
 * When half_bandwidth ≥ T/2 and inner_k ≥ T this matches full attention.
 *
 * @param inner_k  Top-k keys within each band (0 = use all band keys = dense banded).
 *
 * NOTE (test): inner_k=0 (all band keys) should match banded_attention_forward exactly.
 * NOTE (test): inner_k=1 — output[t] == v[argmax_norm(k in band(t))].
 */
inline void sparse_banded_attention_forward(
    const float* q,
    const float* k_mat,
    const float* v,
    float*       output,
    std::size_t  T,
    std::size_t  K,
    std::size_t  d,
    std::size_t  half_bandwidth,
    std::size_t  inner_k = 0,
    int          num_cpus = 4)
{
    if (T == 0 || K == 0 || d == 0) return;
    const float scale = 1.0f / std::sqrt(float(d));

    auto centre_k = [&](std::size_t t) -> std::size_t {
        if (T <= 1 || K <= 1) return 0;
        return (t * (K - 1) + (T - 1) / 2) / (T - 1);
    };

    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (std::size_t t = 0; t < T; ++t) {
        const float* qr   = q + t * d;
        float*       outr = output + t * d;

        const std::size_t ck   = centre_k(t);
        const std::size_t k_lo = (ck > half_bandwidth) ? ck - half_bandwidth : 0;
        const std::size_t k_hi = std::min(ck + half_bandwidth, K - 1);
        const std::size_t band_w = k_hi - k_lo + 1;

        // inner_k=0 → use full band (dense banded).
        const std::size_t kk = (inner_k == 0 || inner_k >= band_w)
            ? band_w
            : inner_k;

        // Select top-kk keys in the band by L2 norm.
        std::vector<std::size_t> band_idx(band_w);
        std::iota(band_idx.begin(), band_idx.end(), k_lo);  // absolute key positions

        if (kk < band_w) {
            std::partial_sort(band_idx.begin(), band_idx.begin() + kk, band_idx.end(),
                [&](std::size_t a, std::size_t b) {
                    // Compare L2 norms (avoid sqrt — monotone).
                    float na = 0.0f, nb = 0.0f;
                    const float* ka = k_mat + a * d;
                    const float* kb = k_mat + b * d;
                    for (std::size_t i = 0; i < d; ++i) { na += ka[i]*ka[i]; nb += kb[i]*kb[i]; }
                    return na > nb;
                });
            band_idx.resize(kk);
            std::sort(band_idx.begin(), band_idx.end());  // restore key order
        }

        // Dot-products for selected keys.
        std::vector<float> scores(kk);
        for (std::size_t bi = 0; bi < kk; ++bi) {
            const float* kr = k_mat + band_idx[bi] * d;
            float dot = 0.0f;
            for (std::size_t i = 0; i < d; ++i) dot += qr[i] * kr[i];
            scores[bi] = dot * scale;
        }

        // Sparse softmax over selected keys only.
        float max_s = *std::max_element(scores.begin(), scores.end());
        float sum_e = 0.0f;
        for (float& s : scores) { s = std::exp(s - max_s); sum_e += s; }
        const float inv = (sum_e > 0.0f) ? 1.0f / sum_e : 0.0f;
        for (float& s : scores) s *= inv;

        // Weighted V sum.
        for (std::size_t bi = 0; bi < kk; ++bi) {
            const float  w  = scores[bi];
            const float* vr = v + band_idx[bi] * d;
            for (std::size_t i = 0; i < d; ++i) outr[i] += w * vr[i];
        }
    }
}


// ── Sparse attention ──────────────────────────────────────────────────────────

/**
 * @brief Top-k sparse attention over a sequence of state vectors.
 *
 * Standard dense attention is O(T² · d). Here we reduce it to O(k² · d)
 * by selecting the top-k query positions and top-k key positions by L2 norm,
 * then computing only those k² dot-products.
 *
 * Zero entries in the score matrix are treated as -∞ for softmax, which is
 * correct: a connection that was never selected contributes 0 to the weighted
 * sum of values (equivalent to masking it out).
 *
 * Selection: top-√T is the natural sweet spot — it matches the information
 * capacity of dense attention at O(T) total cost.  Pass k=0 to use √T.
 *
 * Layout:
 *   q, k, v   : row-major [T × d] float32 matrices
 *   output    : row-major [T × d] float32 — only selected query rows updated
 *
 * Thread safety: output rows for different query positions can be written
 * concurrently; same-position accumulation needs a serial reduction.
 *
 * NOTE (test): With k=T and dense Q/K/V, output should match naive attention
 * to within FP32 rounding.  Good regression test.
 *
 * NOTE (test): With k=1, exactly one (q,k) pair is scored.  The single
 * attention weight after softmax is 1.0, so output[q_row] == v[k_row].
 */
inline void sparse_attention_forward(
    const float* q,       ///< [T × d]
    const float* k_mat,   ///< [T × d]
    const float* v,       ///< [T × d]
    float*       output,  ///< [T × d]  — caller zeroes; only selected rows written
    std::size_t  T,       ///< Sequence length
    std::size_t  d,       ///< Head dimension
    std::size_t  k,       ///< Top-k budget (0 = use √T, minimum 1)
    int          num_cpus = 4)
{
    if (T == 0 || d == 0) return;

    const std::size_t kk = (k == 0)
        ? std::max(std::size_t(1), static_cast<std::size_t>(std::sqrt(float(T))))
        : std::min(k, T);

    // ── L2 norms for query and key rows ──────────────────────────────────────
    std::vector<float> q_norms(T, 0.0f), k_norms(T, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        const float* qr = q       + t * d;
        const float* kr = k_mat   + t * d;
        for (std::size_t i = 0; i < d; ++i) {
            q_norms[t] += qr[i] * qr[i];
            k_norms[t] += kr[i] * kr[i];
        }
    }

    // ── Top-k query / key indices ─────────────────────────────────────────────
    auto topk_idx = [&](const std::vector<float>& norms) {
        std::vector<std::size_t> idx(T);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
            [&](std::size_t a, std::size_t b){ return norms[a] > norms[b]; });
        idx.resize(kk);
        return idx;
    };

    const auto q_idx = topk_idx(q_norms);   // selected query positions
    const auto k_idx = topk_idx(k_norms);   // selected key positions

    const float scale = 1.0f / std::sqrt(float(d));

    // ── Sparse score matrix (COO: kk × kk) ───────────────────────────────────
    // scores[qi][ki] = Q[q_idx[qi]] · K[k_idx[ki]] * scale
    // Stored as a flat [kk × kk] matrix (small, fits in cache).
    std::vector<float> scores(kk * kk, 0.0f);

    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (std::size_t qi = 0; qi < kk; ++qi) {
        const float* qr = q     + q_idx[qi] * d;
        for (std::size_t ki = 0; ki < kk; ++ki) {
            const float* kr = k_mat + k_idx[ki] * d;
            float dot = 0.0f;
            for (std::size_t i = 0; i < d; ++i) dot += qr[i] * kr[i];
            scores[qi * kk + ki] = dot * scale;
        }
    }

    // ── Sparse softmax row by row ─────────────────────────────────────────────
    // Non-selected (i,j) pairs are -∞ → weight 0. Only the kk selected
    // keys exist in each row of the sparse score matrix.
    // We normalise only over the kk selected keys (denominator = their sum).
    std::vector<float> weights(kk * kk);
    for (std::size_t qi = 0; qi < kk; ++qi) {
        float row_max = *std::max_element(scores.data() + qi * kk,
                                          scores.data() + qi * kk + kk);
        float row_sum = 0.0f;
        for (std::size_t ki = 0; ki < kk; ++ki) {
            weights[qi * kk + ki] = std::exp(scores[qi * kk + ki] - row_max);
            row_sum += weights[qi * kk + ki];
        }
        const float inv_sum = (row_sum > 0.0f) ? 1.0f / row_sum : 0.0f;
        for (std::size_t ki = 0; ki < kk; ++ki)
            weights[qi * kk + ki] *= inv_sum;
    }

    // ── Weighted sum of V rows for each selected query ────────────────────────
    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (std::size_t qi = 0; qi < kk; ++qi) {
        float* out_row = output + q_idx[qi] * d;
        for (std::size_t ki = 0; ki < kk; ++ki) {
            const float  w  = weights[qi * kk + ki];
            const float* vr = v + k_idx[ki] * d;
            for (std::size_t i = 0; i < d; ++i)
                out_row[i] += w * vr[i];
        }
    }
}

#endif // __ATTENTION_HPP_
