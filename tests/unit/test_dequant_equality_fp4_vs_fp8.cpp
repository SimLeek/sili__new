// Cross-precision dequantization equality: fp4's representable grid
// ({0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}) is a strict SUBSET of fp8
// E4M3's representable grid. So a weight/importance value chosen from
// fp4's grid decodes to the EXACT SAME float whether it's stored as fp4
// or fp8 -- forward output and backward's dx must therefore match
// (scattered path, not block4 -- this is testing the precisions
// themselves, not any storage format). Post-update WEIGHTS are
// deliberately NOT compared: once real learning happens, fp4's coarser
// grid and fp8's finer grid round the same delta to different codes --
// that divergence is expected/"predictable", not a bug. Only the
// no-learning-dependent quantities (forward y, dx, neuron accum) are
// required to match.
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
using WeightsFP4 = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;
using WeightsFP8 = SparseLinearWeightsDelta<SIZE_TYPE, FP8BiValues, COL_TYPE>;

int main() {
    const std::size_t n_in = 16, n_out = 16;
    const SIZE_TYPE batch = 4;
    std::mt19937 rng(556677);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    // Every value in FP4's own table (skipping 0=index0 and NaN=index8),
    // cycled across all n_in*n_out cells -- weight draws from the full
    // signed table, importance draws from the nonnegative half only
    // (indices 1..7) since importance is stored/used as a magnitude.
    std::vector<float> dense_w(n_in * n_out), dense_imp(n_in * n_out);
    for (std::size_t row = 0; row < n_in; ++row) {
        for (std::size_t col = 0; col < n_out; ++col) {
            const std::size_t idx = row * n_out + col;
            uint32_t code = uint32_t(1 + (idx % 14));
            if (code >= 8)
                ++code; // skip the NaN slot (index 8)
            dense_w[idx] = FP4_TABLE[code];
            dense_imp[idx] = FP4_TABLE[1 + (idx % 7)];
        }
    }

    // ── Sanity: every dense_w/dense_imp value round-trips EXACTLY through
    // BOTH codecs before we even build the layers -- if this fails, the
    // rest of the test is meaningless.
    for (std::size_t i = 0; i < dense_w.size(); ++i) {
        const float w4 = fp4_decode_bits(fp4_quantize(dense_w[i]));
        const float w8 = fp8_decode_bits(fp8_encode_bits(dense_w[i]));
        CHECK(w4 == dense_w[i], "fp4 round-trip of table value %g changed to %g", dense_w[i], w4);
        CHECK(w8 == dense_w[i], "fp8 round-trip of table value %g changed to %g", dense_w[i], w8);
        const float i4 = fp4_decode_bits(fp4_quantize(dense_imp[i]));
        const float i8 = fp8_decode_bits(fp8_encode_bits(dense_imp[i]));
        CHECK(i4 == dense_imp[i], "fp4 round-trip of importance %g changed to %g", dense_imp[i],
              i4);
        CHECK(i8 == dense_imp[i], "fp8 round-trip of importance %g changed to %g", dense_imp[i],
              i8);
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
    WeightsFP4 weights_fp4 = build(FP4BiPacked{});
    WeightsFP8 weights_fp8 = build(FP8BiValues{});

    // ── Forward: outputs must match ─────────────────────────────────────
    std::vector<float> y4(std::size_t(batch) * n_out, 0.0f), y8(std::size_t(batch) * n_out, 0.0f);
    disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(input.data(), batch, SIZE_TYPE(n_in),
                                                     weights_fp4, y4.data(), 4);
    disldo_forward<SIZE_TYPE, FP8BiValues, COL_TYPE>(input.data(), batch, SIZE_TYPE(n_in),
                                                     weights_fp8, y8.data(), 4);
    double err_fwd = 0.0;
    for (std::size_t i = 0; i < y4.size(); ++i)
        err_fwd = std::max(err_fwd, double(std::fabs(y4[i] - y8[i])));
    CHECK(err_fwd < 1e-5,
          "fp4 vs fp8 forward output diverges on identical dequantized weights: "
          "max abs err %.8f",
          err_fwd);

    // ── Backward (real learning_rate, deterministic rounding): dx and the
    // pure input/dy accumulators must still match -- neither depends on
    // how the post-update weight gets rounded in each precision's own
    // grid, only on the CURRENT (pre-update) dequantized weight.
    std::vector<float> dx4(std::size_t(batch) * n_in, 0.0f), dx8(std::size_t(batch) * n_in, 0.0f);
    std::vector<float> ni4(n_in, 0.0f), ng4(n_out, 0.0f), ni8(n_in, 0.0f), ng8(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_fp4, dx4.data(), ni4.data(),
        ng4.data(), lr, 4, false, true);
    disldo_backward<SIZE_TYPE, FP8BiValues, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        input.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_fp8, dx8.data(), ni8.data(),
        ng8.data(), lr, 4, false, true);
    double err_dx = 0.0;
    for (std::size_t i = 0; i < dx4.size(); ++i)
        err_dx = std::max(err_dx, double(std::fabs(dx4[i] - dx8[i])));
    CHECK(err_dx < 1e-5,
          "fp4 vs fp8 dx diverges on identical dequantized weights: max abs err %.8f", err_dx);
    double err_ni = 0.0, err_ng = 0.0;
    for (std::size_t i = 0; i < ni4.size(); ++i)
        err_ni = std::max(err_ni, double(std::fabs(ni4[i] - ni8[i])));
    for (std::size_t i = 0; i < ng4.size(); ++i)
        err_ng = std::max(err_ng, double(std::fabs(ng4[i] - ng8[i])));
    CHECK(err_ni < 1e-6, "neuron_input_accum diverges (input/dy-only, precision-independent): %.8f",
          err_ni);
    CHECK(err_ng < 1e-6, "neuron_grad_accum diverges (input/dy-only, precision-independent): %.8f",
          err_ng);

    std::printf("fp4 vs fp8 dequant equality: forward err=%.8f dx err=%.8f ni err=%.8f ng "
                "err=%.8f\n",
                err_fwd, err_dx, err_ni, err_ng);
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
