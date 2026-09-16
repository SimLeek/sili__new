// DIDLDO correctness (TODO_BATCH_BLOCKING.md, "Also queued -> DIDLDO/
// SIDLDO"): checks didldo_forward/didldo_backward (linear_didldo.hpp)
// against independent, hand-written dense-matmul and RMSprop references.
// Only built when MKL was found (see tests/unit/CMakeLists.txt's
// SILI_MKL_FOUND) -- absent MKL, DIDLDO doesn't exist to test.
#include "../../sili/lib/headers/linear_didldo.hpp"
#include <cmath>
#include <cstdio>
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

// Independent reference: y[b,c] = sum_r x[b,r] * w[r,c]
static std::vector<float> ref_forward(const std::vector<float>& x, const std::vector<float>& w,
                                      int batch, int n_in, int n_out) {
    std::vector<float> y(std::size_t(batch) * n_out, 0.0f);
    for (int b = 0; b < batch; ++b)
        for (int r = 0; r < n_in; ++r) {
            const float xv = x[std::size_t(b) * n_in + r];
            for (int c = 0; c < n_out; ++c)
                y[std::size_t(b) * n_out + c] += xv * w[std::size_t(r) * n_out + c];
        }
    return y;
}

static std::vector<float> ref_dx(const std::vector<float>& dy, const std::vector<float>& w,
                                 int batch, int n_in, int n_out) {
    std::vector<float> dx(std::size_t(batch) * n_in, 0.0f);
    for (int b = 0; b < batch; ++b)
        for (int r = 0; r < n_in; ++r) {
            float acc = 0.0f;
            for (int c = 0; c < n_out; ++c)
                acc += dy[std::size_t(b) * n_out + c] * w[std::size_t(r) * n_out + c];
            dx[std::size_t(b) * n_in + r] = acc;
        }
    return dx;
}

static std::vector<float> ref_dw(const std::vector<float>& x, const std::vector<float>& dy,
                                 int batch, int n_in, int n_out) {
    std::vector<float> dw(std::size_t(n_in) * n_out, 0.0f);
    for (int r = 0; r < n_in; ++r)
        for (int c = 0; c < n_out; ++c) {
            float acc = 0.0f;
            for (int b = 0; b < batch; ++b)
                acc += x[std::size_t(b) * n_in + r] * dy[std::size_t(b) * n_out + c];
            dw[std::size_t(r) * n_out + c] = acc;
        }
    return dw;
}

static void run_case(const char* label, int n_in, int n_out, int batch, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w0(std::size_t(n_in) * n_out);
    for (auto& v : w0)
        v = dist(rng);
    std::vector<float> x(std::size_t(batch) * n_in), dy(std::size_t(batch) * n_out);
    for (auto& v : x)
        v = dist(rng);
    for (auto& v : dy)
        v = dist(rng);

    DenseLinearWeights weights;
    weights.resize(std::size_t(n_in), std::size_t(n_out));
    weights.w = w0;

    // forward
    std::vector<float> y(std::size_t(batch) * n_out);
    didldo_forward(x.data(), weights, y.data(), batch, 4);
    auto y_ref = ref_forward(x, w0, batch, n_in, n_out);
    double fwd_err = max_abs_diff(y, y_ref);
    CHECK(fwd_err < 1e-2, "%s: forward max abs err %.6f too large", label, fwd_err);

    // backward, lr=0 (bwd0): dx correct, w unchanged
    weights.w = w0; // reset (forward doesn't mutate, but be explicit)
    std::vector<float> dx(std::size_t(batch) * n_in);
    didldo_backward(x.data(), dy.data(), weights, dx.data(), batch, 4, 0.0f);
    auto dx_ref = ref_dx(dy, w0, batch, n_in, n_out);
    double dx_err = max_abs_diff(dx, dx_ref);
    CHECK(dx_err < 1e-2, "%s: bwd0 dx max abs err %.6f too large", label, dx_err);
    double w_drift_bwd0 = max_abs_diff(weights.w, w0);
    CHECK(w_drift_bwd0 < 1e-8, "%s: bwd0 (lr=0) must not mutate weights, drift %.8f", label,
          w_drift_bwd0);

    // backward, lr!=0 (bwdX): dx still correct, w updates match a
    // reference RMSprop step applied to the reference dw.
    weights.w = w0;
    weights.square_avg.assign(weights.square_avg.size(), 0.0f);
    const float lr = 1e-2f;
    didldo_backward(x.data(), dy.data(), weights, dx.data(), batch, 4, lr);
    dx_err = max_abs_diff(dx, dx_ref);
    CHECK(dx_err < 1e-2, "%s: bwdX dx max abs err %.6f too large", label, dx_err);

    auto dw_ref = ref_dw(x, dy, batch, n_in, n_out);
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
    // batch=1 (sgemv dispatch) and batch>1 (sgemm dispatch) both, plus a
    // non-square shape since the sgemv row/col orientation is exactly
    // where a transpose-direction bug would hide on a square matrix.
    run_case("n_in=61 n_out=97 batch=1", 61, 97, 1, 1);
    run_case("n_in=61 n_out=97 batch=9", 61, 97, 9, 2);
    run_case("n_in=128 n_out=128 batch=1", 128, 128, 1, 3);
    run_case("n_in=128 n_out=128 batch=37", 128, 128, 37, 4);
    run_case("n_in=97 n_out=61 batch=64", 97, 61, 64, 5);

    if (g_fail == 0)
        std::printf("PASS: all didldo_kernel checks\n");
    return g_fail;
}
