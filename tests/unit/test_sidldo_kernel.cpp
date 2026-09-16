// SIDLDO forward correctness (TODO_BATCH_BLOCKING.md, "Also queued ->
// DIDLDO/SIDLDO"): checks sidldo_forward (linear_sidldo.hpp) against an
// independent dense-matmul reference built from the SAME sparse input,
// densified by hand. Only built when MKL was found (mirrors
// test_didldo_kernel.cpp's CMake gating).
#include "../../sili/lib/headers/linear_sidldo.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}

static void run_case(const char* label, int n_in, int n_out, int batch, float density,
                     unsigned seed, int num_cpus = 4) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w0(std::size_t(n_in) * n_out);
    for (auto& v : w0)
        v = dist(rng);

    // Per-sample sparse input CSR: k random feature indices per sample.
    const std::size_t k = std::max<std::size_t>(0, std::size_t(density * float(n_in)));
    std::vector<int> col_pool(n_in);
    std::iota(col_pool.begin(), col_pool.end(), 0);
    std::vector<int> x_ptrs(std::size_t(batch) + 1, 0);
    std::vector<int> x_idx;
    std::vector<float> x_val;
    std::vector<float> x_dense(std::size_t(batch) * n_in, 0.0f);
    for (int b = 0; b < batch; ++b) {
        std::shuffle(col_pool.begin(), col_pool.end(), rng);
        std::vector<int> cols(col_pool.begin(), col_pool.begin() + std::ptrdiff_t(k));
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            const float v = dist(rng);
            x_idx.push_back(c);
            x_val.push_back(v);
            x_dense[std::size_t(b) * n_in + std::size_t(c)] = v;
        }
        x_ptrs[std::size_t(b) + 1] = x_ptrs[std::size_t(b)] + int(cols.size());
    }

    DenseLinearWeights weights;
    weights.resize(std::size_t(n_in), std::size_t(n_out));
    weights.w = w0;

    SidldoForwardScratch scratch;
    std::vector<float> y(std::size_t(batch) * n_out);
    sidldo_forward(x_ptrs.data(), x_idx.data(), x_val.data(), batch, weights, scratch, y.data(),
                   num_cpus);

    std::vector<float> y_ref(std::size_t(batch) * n_out, 0.0f);
    for (int b = 0; b < batch; ++b)
        for (int r = 0; r < n_in; ++r) {
            const float xv = x_dense[std::size_t(b) * n_in + std::size_t(r)];
            if (xv == 0.0f)
                continue;
            for (int c = 0; c < n_out; ++c)
                y_ref[std::size_t(b) * n_out + std::size_t(c)] +=
                    xv * w0[std::size_t(r) * n_out + std::size_t(c)];
        }

    double err = max_abs_diff(y, y_ref);
    CHECK(err < 1e-2, "%s: sidldo_forward vs dense reference max abs err %.6f too large", label,
          err);

    // Weights must be untouched by a forward call (pure computation).
    double w_drift = max_abs_diff(weights.w, w0);
    CHECK(w_drift < 1e-8, "%s: forward must not mutate weights, drift %.8f", label, w_drift);
}

// Dense x, SPARSE dy (matches disldo_backward_sparse_grad's convention,
// not forward's). Checks dx against an independent dense reference,
// bwd0 (lr=0) leaves weights untouched, and bwdX (lr!=0) matches an
// independently computed reference RMSprop step applied to an
// independently computed reference dw.
static void run_backward_case(const char* label, int n_in, int n_out, int batch, float density,
                              unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w0(std::size_t(n_in) * n_out);
    for (auto& v : w0)
        v = dist(rng);
    std::vector<float> x(std::size_t(batch) * n_in);
    for (auto& v : x)
        v = dist(rng);

    const std::size_t k = std::max<std::size_t>(0, std::size_t(density * float(n_out)));
    std::vector<int> col_pool(n_out);
    std::iota(col_pool.begin(), col_pool.end(), 0);
    std::vector<int> dy_ptrs(std::size_t(batch) + 1, 0);
    std::vector<int> dy_idx;
    std::vector<float> dy_val;
    std::vector<float> dy_dense(std::size_t(batch) * n_out, 0.0f);
    for (int b = 0; b < batch; ++b) {
        std::shuffle(col_pool.begin(), col_pool.end(), rng);
        std::vector<int> cols(col_pool.begin(), col_pool.begin() + std::ptrdiff_t(k));
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            const float v = dist(rng);
            dy_idx.push_back(c);
            dy_val.push_back(v);
            dy_dense[std::size_t(b) * n_out + std::size_t(c)] = v;
        }
        dy_ptrs[std::size_t(b) + 1] = dy_ptrs[std::size_t(b)] + int(cols.size());
    }

    auto ref_dx = [&](const std::vector<float>& w) {
        std::vector<float> out(std::size_t(batch) * n_in, 0.0f);
        for (int b = 0; b < batch; ++b)
            for (int r = 0; r < n_in; ++r) {
                float acc = 0.0f;
                for (int c = 0; c < n_out; ++c)
                    acc += dy_dense[std::size_t(b) * n_out + std::size_t(c)] *
                           w[std::size_t(r) * n_out + std::size_t(c)];
                out[std::size_t(b) * n_in + std::size_t(r)] = acc;
            }
        return out;
    };
    auto ref_dw = [&]() {
        std::vector<float> dw(std::size_t(n_in) * n_out, 0.0f);
        for (int r = 0; r < n_in; ++r)
            for (int c = 0; c < n_out; ++c) {
                float acc = 0.0f;
                for (int b = 0; b < batch; ++b)
                    acc += x[std::size_t(b) * n_in + std::size_t(r)] *
                           dy_dense[std::size_t(b) * n_out + std::size_t(c)];
                dw[std::size_t(r) * n_out + std::size_t(c)] = acc;
            }
        return dw;
    };

    DenseLinearWeights weights;
    weights.resize(std::size_t(n_in), std::size_t(n_out));
    SidldoBackwardScratch scratch;
    std::vector<float> dx(std::size_t(batch) * n_in);

    // bwd0: dx correct, weights unchanged.
    weights.w = w0;
    sidldo_backward(x.data(), dy_ptrs.data(), dy_idx.data(), dy_val.data(), batch, weights, scratch,
                    dx.data(), 0.0f);
    auto dx_ref = ref_dx(w0);
    double dx_err = max_abs_diff(dx, dx_ref);
    CHECK(dx_err < 1e-2, "%s: bwd0 dx max abs err %.6f too large", label, dx_err);
    double w_drift = max_abs_diff(weights.w, w0);
    CHECK(w_drift < 1e-8, "%s: bwd0 (lr=0) must not mutate weights, drift %.8f", label, w_drift);

    // bwdX: dx still correct, weights match a reference RMSprop step.
    weights.w = w0;
    weights.square_avg.assign(weights.square_avg.size(), 0.0f);
    const float lr = 1e-2f;
    sidldo_backward(x.data(), dy_ptrs.data(), dy_idx.data(), dy_val.data(), batch, weights, scratch,
                    dx.data(), lr);
    dx_err = max_abs_diff(dx, dx_ref);
    CHECK(dx_err < 1e-2, "%s: bwdX dx max abs err %.6f too large", label, dx_err);

    auto dw_ref = ref_dw();
    std::vector<float> w_expect = w0;
    const float alpha = 0.99f, eps = 1e-8f;
    std::vector<float> sq(w0.size(), 0.0f);
    for (std::size_t i = 0; i < w0.size(); ++i) {
        sq[i] = alpha * sq[i] + (1.0f - alpha) * dw_ref[i] * dw_ref[i];
        w_expect[i] -= lr * dw_ref[i] / (std::sqrt(sq[i]) + eps);
    }
    double w_err = max_abs_diff(weights.w, w_expect);
    CHECK(w_err < 1e-2, "%s: bwdX RMSprop-updated weights max abs err %.6f too large", label,
          w_err);
}

int main() {
    run_case("n_in=61 n_out=97 batch=1 density=0.2", 61, 97, 1, 0.2f, 1);
    run_case("n_in=61 n_out=97 batch=9 density=0.2", 61, 97, 9, 0.2f, 2);
    run_case("n_in=128 n_out=128 batch=37 density=0.05 (sparse, low batch)", 128, 128, 37, 0.05f,
             3);
    run_case("n_in=97 n_out=61 batch=64 density=0.9 (near-dense)", 97, 61, 64, 0.9f, 4);
    run_case("n_in=200 n_out=50 batch=5 density=0.0 (fully empty)", 200, 50, 5, 0.0f, 5);
    // Threshold boundary (SIDLDO_HIGH_BATCH_THRESHOLD=64): one below and
    // one at, to catch an off-by-one in the dispatch condition -- this
    // rollout's established pattern for every other batch threshold.
    run_case("n_in=89 n_out=101 batch=63 density=0.1 (below threshold)", 89, 101, 63, 0.1f, 6);
    run_case("n_in=89 n_out=101 batch=64 density=0.1 (at threshold)", 89, 101, 64, 0.1f, 7);
    // Well above threshold, num_cpus not dividing evenly into nnz --
    // checks the nnz-balanced partition's boundary math (thread ranges,
    // the binary-search row lookup, the private-buffer reduction) and
    // multiple num_cpus so it isn't accidentally single-threaded-correct
    // only.
    run_case("n_in=150 n_out=90 batch=300 density=0.05, high batch, num_cpus=1", 150, 90, 300,
             0.05f, 8, 1);
    run_case("n_in=150 n_out=90 batch=300 density=0.6, high batch+density, num_cpus=3", 150, 90,
             300, 0.6f, 9, 3);
    run_case("n_in=150 n_out=90 batch=300 density=0.05, high batch, num_cpus=7", 150, 90, 300,
             0.05f, 10, 7);
    // Reusing the same scratch/weights across calls of DIFFERENT shapes'
    // active-row counts, checking the grow-only scratch doesn't leave
    // stale state from a prior call's larger/smaller active set.
    {
        DenseLinearWeights weights;
        weights.resize(80, 40);
        std::mt19937 rng(6);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : weights.w)
            v = dist(rng);
        SidldoForwardScratch scratch;
        std::vector<int> ptrs1 = {0, 3, 3, 6};
        std::vector<int> idx1 = {0, 1, 2, 5, 6, 7};
        std::vector<float> val1 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        std::vector<float> y1(std::size_t(3) * 40);
        sidldo_forward(ptrs1.data(), idx1.data(), val1.data(), 3, weights, scratch, y1.data(), 4);

        std::vector<int> ptrs2 = {0, 1};
        std::vector<int> idx2 = {79};
        std::vector<float> val2 = {2.0f};
        std::vector<float> y2(std::size_t(1) * 40);
        sidldo_forward(ptrs2.data(), idx2.data(), val2.data(), 1, weights, scratch, y2.data(), 4);
        std::vector<float> y2_ref(40);
        for (int c = 0; c < 40; ++c)
            y2_ref[std::size_t(c)] = 2.0f * weights.w[std::size_t(79) * 40 + std::size_t(c)];
        double err = max_abs_diff(y2, y2_ref);
        CHECK(err < 1e-2, "scratch-reuse-across-shapes: max abs err %.6f too large", err);
    }

    run_backward_case("bwd n_in=61 n_out=97 batch=1 density=0.2", 61, 97, 1, 0.2f, 11);
    run_backward_case("bwd n_in=61 n_out=97 batch=9 density=0.2", 61, 97, 9, 0.2f, 12);
    run_backward_case("bwd n_in=97 n_out=61 batch=64 density=0.9 (near-dense)", 97, 61, 64, 0.9f,
                      13);
    run_backward_case("bwd n_in=200 n_out=50 batch=5 density=0.0 (fully empty)", 200, 50, 5, 0.0f,
                      14);
    run_backward_case("bwd n_in=89 n_out=101 batch=63 density=0.1", 89, 101, 63, 0.1f, 15);
    // Reusing SidldoBackwardScratch across calls of different active-
    // column-count shapes, same grow-only-scratch-staleness check as
    // forward's own scratch-reuse case.
    {
        DenseLinearWeights weights;
        weights.resize(80, 40);
        std::mt19937 rng(16);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : weights.w)
            v = dist(rng);
        SidldoBackwardScratch scratch;
        std::vector<float> x1(std::size_t(3) * 80);
        for (auto& v : x1)
            v = dist(rng);
        std::vector<int> dy_ptrs1 = {0, 3, 3, 6};
        std::vector<int> dy_idx1 = {0, 1, 2, 5, 6, 7};
        std::vector<float> dy_val1 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
        std::vector<float> dx1(std::size_t(3) * 80);
        sidldo_backward(x1.data(), dy_ptrs1.data(), dy_idx1.data(), dy_val1.data(), 3, weights,
                        scratch, dx1.data(), 0.0f);

        std::vector<float> x2(80, 1.0f);
        std::vector<int> dy_ptrs2 = {0, 1};
        std::vector<int> dy_idx2 = {39};
        std::vector<float> dy_val2 = {2.0f};
        std::vector<float> dx2(80);
        sidldo_backward(x2.data(), dy_ptrs2.data(), dy_idx2.data(), dy_val2.data(), 1, weights,
                        scratch, dx2.data(), 0.0f);
        std::vector<float> dx2_ref(80);
        for (int r = 0; r < 80; ++r)
            dx2_ref[std::size_t(r)] = 2.0f * weights.w[std::size_t(r) * 40 + 39];
        double err2 = max_abs_diff(dx2, dx2_ref);
        CHECK(err2 < 1e-2, "bwd scratch-reuse-across-shapes: max abs err %.6f too large", err2);
    }

    if (g_fail == 0)
        std::printf("PASS: all sidldo_kernel checks\n");
    return g_fail;
}
