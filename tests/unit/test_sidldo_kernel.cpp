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
                     unsigned seed) {
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
    sidldo_forward(x_ptrs.data(), x_idx.data(), x_val.data(), batch, weights, scratch, y.data());

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

int main() {
    run_case("n_in=61 n_out=97 batch=1 density=0.2", 61, 97, 1, 0.2f, 1);
    run_case("n_in=61 n_out=97 batch=9 density=0.2", 61, 97, 9, 0.2f, 2);
    run_case("n_in=128 n_out=128 batch=37 density=0.05 (sparse, low batch)", 128, 128, 37, 0.05f,
             3);
    run_case("n_in=97 n_out=61 batch=64 density=0.9 (near-dense)", 97, 61, 64, 0.9f, 4);
    run_case("n_in=200 n_out=50 batch=5 density=0.0 (fully empty)", 200, 50, 5, 0.0f, 5);
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
        sidldo_forward(ptrs1.data(), idx1.data(), val1.data(), 3, weights, scratch, y1.data());

        std::vector<int> ptrs2 = {0, 1};
        std::vector<int> idx2 = {79};
        std::vector<float> val2 = {2.0f};
        std::vector<float> y2(std::size_t(1) * 40);
        sidldo_forward(ptrs2.data(), idx2.data(), val2.data(), 1, weights, scratch, y2.data());
        std::vector<float> y2_ref(40);
        for (int c = 0; c < 40; ++c)
            y2_ref[std::size_t(c)] = 2.0f * weights.w[std::size_t(79) * 40 + std::size_t(c)];
        double err = max_abs_diff(y2, y2_ref);
        CHECK(err < 1e-2, "scratch-reuse-across-shapes: max abs err %.6f too large", err);
    }

    if (g_fail == 0)
        std::printf("PASS: all sidldo_kernel checks\n");
    return g_fail;
}
