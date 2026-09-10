// Cross-precision dequantization equality: fp8 E4M3's representable grid
// is a strict SUBSET of float32's (every fp8 code decodes to an exactly
// representable float32). So a weight/importance value chosen from fp8's
// own grid (round-tripped through fp8_encode_bits/fp8_decode_bits)
// decodes to the EXACT SAME float whether it's stored as fp8 or fp32 --
// forward output and backward's dx must therefore match (scattered path,
// not block4). Post-update WEIGHTS are deliberately NOT compared: fp8's
// coarser grid vs float32's effectively-continuous one round the same
// delta differently -- expected divergence once real learning happens,
// not tested here. Mirrors test_dequant_equality_fp4_vs_fp8.cpp.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
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

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using WeightsFP8 = SparseLinearWeightsDelta<SIZE_TYPE, FP8BiValues, COL_TYPE>;
using WeightsFP32 = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

int main() {
    const std::size_t n_in = 16, n_out = 16;
    const SIZE_TYPE batch = 4;
    std::mt19937 rng(778899);
    std::uniform_real_distribution<float> wdist(-6.0f, 6.0f), idist(0.0f, 6.0f), xdist(-1.0f, 1.0f);

    // Arbitrary floats, round-tripped through fp8's OWN codec so both
    // arms start from a value that is EXACTLY representable in fp8 (and
    // therefore, trivially, also exactly representable in float32).
    std::vector<float> dense_w(n_in * n_out), dense_imp(n_in * n_out);
    for (auto& v : dense_w)
        v = fp8_decode_bits(fp8_encode_bits(wdist(rng)));
    for (auto& v : dense_imp)
        v = fp8_decode_bits(fp8_encode_bits(idist(rng)));

    // ── Sanity: fp8 round-trip is a true fixed point, and (trivially)
    // float32 stores the exact same value with no further rounding.
    for (std::size_t i = 0; i < dense_w.size(); ++i) {
        const float w8 = fp8_decode_bits(fp8_encode_bits(dense_w[i]));
        CHECK(w8 == dense_w[i], "fp8 round-trip of %g changed to %g (not a fixed point)",
              dense_w[i], w8);
    }

    std::vector<float> input(std::size_t(batch) * n_in), dy(std::size_t(batch) * n_out);
    for (auto& v : input)
        v = xdist(rng);
    for (auto& v : dy)
        v = xdist(rng);

    auto build = [&](auto tag) {
        using VALUES_TYPE = decltype(tag);
        SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE> weights;
        std::vector<SIZE_TYPE> ptrs(n_in + 1);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> w, imp;
        SIZE_TYPE cursor = 0;
        for (std::size_t row = 0; row < n_in; ++row) {
            ptrs[row] = cursor;
            for (std::size_t col = 0; col < n_out; ++col) {
                idx.push_back(SIZE_TYPE(col));
                w.push_back(dense_w[row * n_out + col]);
                imp.push_back(dense_imp[row * n_out + col]);
                ++cursor;
            }
        }
        ptrs[n_in] = cursor;
        weights.connections = delta_csr_from_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
            ptrs, idx, w, imp, n_in, n_out, std::size_t(idx.size()) * 2,
            std::size_t(idx.size()) * 2);
        weights.out_degree.assign(n_out, SIZE_TYPE(n_in));
        return weights;
    };
    WeightsFP8 weights_fp8 = build(FP8BiValues{});
    WeightsFP32 weights_fp32 = build(DeltaCSRBiValues<float>{});

    // ── Forward: outputs must match ─────────────────────────────────────
    std::vector<float> y8(std::size_t(batch) * n_out, 0.0f), y32(std::size_t(batch) * n_out, 0.0f);
    disldo_forward<SIZE_TYPE, FP8BiValues, COL_TYPE>(input.data(), batch, SIZE_TYPE(n_in),
                                                     weights_fp8, y8.data(), 4);
    disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        input.data(), batch, SIZE_TYPE(n_in), weights_fp32, y32.data(), 4);
    double err_fwd = 0.0;
    for (std::size_t i = 0; i < y8.size(); ++i)
        err_fwd = std::max(err_fwd, double(std::fabs(y8[i] - y32[i])));
    CHECK(err_fwd < 1e-5,
          "fp8 vs fp32 forward output diverges on identical dequantized weights: "
          "max abs err %.8f",
          err_fwd);

    // ── Backward (real learning_rate, deterministic rounding): dx and the
    // pure input/dy accumulators must still match.
    std::vector<float> dx8(std::size_t(batch) * n_in, 0.0f), dx32(std::size_t(batch) * n_in, 0.0f);
    std::vector<float> ni8(n_in, 0.0f), ng8(n_out, 0.0f), ni32(n_in, 0.0f), ng32(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, FP8BiValues, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_fp8, dx8.data(), ni8.data(),
        ng8.data(), lr, 4, false, true);
    disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE, RMSpropScalePolicy<float>, false,
                    false>(input.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_fp32,
                           dx32.data(), ni32.data(), ng32.data(), lr, 4, false, true);
    double err_dx = 0.0;
    for (std::size_t i = 0; i < dx8.size(); ++i)
        err_dx = std::max(err_dx, double(std::fabs(dx8[i] - dx32[i])));
    CHECK(err_dx < 1e-5,
          "fp8 vs fp32 dx diverges on identical dequantized weights: max abs err %.8f", err_dx);
    double err_ni = 0.0, err_ng = 0.0;
    for (std::size_t i = 0; i < ni8.size(); ++i)
        err_ni = std::max(err_ni, double(std::fabs(ni8[i] - ni32[i])));
    for (std::size_t i = 0; i < ng8.size(); ++i)
        err_ng = std::max(err_ng, double(std::fabs(ng8[i] - ng32[i])));
    CHECK(err_ni < 1e-6, "neuron_input_accum diverges (input/dy-only, precision-independent): %.8f",
          err_ni);
    CHECK(err_ng < 1e-6, "neuron_grad_accum diverges (input/dy-only, precision-independent): %.8f",
          err_ng);

    std::printf("fp8 vs fp32 dequant equality: forward err=%.8f dx err=%.8f ni err=%.8f ng "
                "err=%.8f\n",
                err_fwd, err_dx, err_ni, err_ng);
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
