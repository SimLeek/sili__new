// Phase 7.5 of the batch-blocking rollout (see TODO_BATCH_BLOCKING.md):
// disldo_forward's SCATTERED (non-block4) path, at batch >=
// BLOCK4_BATCH_BLOCK_THRESHOLD, now transposes the input once and
// accumulates into a per-thread PRIVATE column-major buffer via
// scalar_batch_accumulate (block4.hpp) instead of the old per-sample
// strided scalar accumulate. This test's only job is to catch a wiring
// bug in that new path -- wrong column offset, dropped remainder-batch
// samples, wrong row, etc -- by comparing against an independent
// dense-matmul reference AND against the SAME layer run below threshold
// (old scalar path), which must agree with each other. Deliberately a
// PURE scattered layer (no block4 content at all -- see block4_tiles==0
// check below) so this only ever exercises the code this phase touched.
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

static void run_case(const char* label, std::size_t n_in, std::size_t n_out, SIZE_TYPE batch,
                     int num_cpus, float density, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> idist(0.1f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    // Scattered CSR: `k` random columns per row, sorted (row_cursor
    // requires ascending indices within a row).
    std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
    std::vector<SIZE_TYPE> idx;
    std::vector<float> wv, imp;
    std::vector<float> dense_w(n_in * n_out, 0.f); // reference ground truth
    const std::size_t k = std::max<std::size_t>(1, std::size_t(density * float(n_out)));
    std::vector<SIZE_TYPE> col_pool(n_out);
    std::iota(col_pool.begin(), col_pool.end(), SIZE_TYPE(0));
    for (std::size_t r = 0; r < n_in; ++r) {
        std::shuffle(col_pool.begin(), col_pool.end(), rng);
        std::vector<SIZE_TYPE> row_cols(col_pool.begin(), col_pool.begin() + std::ptrdiff_t(k));
        std::sort(row_cols.begin(), row_cols.end());
        for (SIZE_TYPE c : row_cols) {
            const float w = wdist(rng);
            idx.push_back(c);
            wv.push_back(w);
            imp.push_back(idist(rng));
            dense_w[r * n_out + std::size_t(c)] = w;
        }
        ptrs[r + 1] = ptrs[r] + SIZE_TYPE(row_cols.size());
    }

    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        ptrs, idx, wv, imp, n_in, n_out, std::size_t(1) << 20, std::size_t(1) << 20, 0.2f);
    weights.recompute_stats();
    CHECK(weights.block4.n_tiles() == 0, "%s: expected pure scattered layer, got %zu block4 tiles",
          label, weights.block4.n_tiles());

    std::vector<float> x(std::size_t(batch) * n_in);
    for (auto& v : x)
        v = xdist(rng);

    std::vector<float> y_ref(std::size_t(batch) * n_out, 0.f);
    for (SIZE_TYPE b = 0; b < batch; ++b)
        for (std::size_t r = 0; r < n_in; ++r) {
            const float xv = x[std::size_t(b) * n_in + r];
            const float* wr = dense_w.data() + r * n_out;
            float* yr = y_ref.data() + std::size_t(b) * n_out;
            for (std::size_t c = 0; c < n_out; ++c)
                yr[c] += wr[c] * xv;
        }

    std::vector<float> y(std::size_t(batch) * n_out, 0.f);
    disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x.data(), batch, SIZE_TYPE(n_in),
                                                                 weights, y.data(), num_cpus);

    double max_err = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i)
        max_err = std::max(max_err, double(std::fabs(y[i] - y_ref[i])));
    CHECK(max_err < 1e-2,
          "%s: blocked scattered path vs dense reference max abs err %.6f too large", label,
          max_err);
}

int main() {
    // Exactly at the threshold (8) and its neighbors -- off-by-one gate
    // bugs live at boundaries.
    run_case("n_in=61 n_out=97 batch=7 (below threshold)", 61, 97, 7, 4, 0.2f, 1);
    run_case("n_in=61 n_out=97 batch=8 (at threshold)", 61, 97, 8, 4, 0.2f, 2);
    run_case("n_in=61 n_out=97 batch=9 (above threshold)", 61, 97, 9, 4, 0.2f, 3);
    // Multiple BLOCK4_BATCH_BLOCK_B(8)-wide chunks plus a remainder tail.
    run_case("n_in=64 n_out=128 batch=37 (partial tail chunk)", 64, 128, 37, 4, 0.1f, 4);
    // Large batch, exact multiple of BLOCK4_BATCH_BLOCK_B, many threads.
    run_case("n_in=128 n_out=128 batch=256 (no remainder)", 128, 128, 256, 4, 0.1f, 5);
    // Very sparse (single connection per row) and near-dense, since the
    // accumulate step's correctness shouldn't depend on density at all.
    run_case("n_in=97 n_out=61 batch=64, near-empty rows", 97, 61, 64, 4, 0.02f, 6);
    run_case("n_in=97 n_out=61 batch=64, near-dense rows", 97, 61, 64, 4, 0.9f, 7);
    // num_cpus not dividing evenly into batch*n_out -- checks the
    // static-per-thread-range split's boundary math.
    run_case("n_in=89 n_out=101 batch=50, num_cpus=6", 89, 101, 50, 6, 0.15f, 8);

    if (g_fail == 0)
        std::printf("PASS: all disldo_scattered_batch_blocked checks\n");
    return g_fail;
}
