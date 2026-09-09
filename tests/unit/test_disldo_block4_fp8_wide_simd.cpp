// TDD baseline for an AVX2-width change to disldo_forward's FP8 block4 hot
// loop -- the FP8 counterpart to test_disldo_block4_fp32_wide_simd.cpp's
// forward widening. Unlike FP32 (no decode step at all), FP8 DOES have a
// real 4-wide SIMD decode (block4_vec_decode_fp8), so this widening pairs
// two adjacent columns' DECODED results into one 8-wide vector rather than
// rewriting the decode itself 8-wide -- block4_vec_decode_fp8 has a subtle
// rare-code (subnormal/NaN-slot) scalar-correction path that's safer left
// untouched, and the batch loop that follows (which runs `batch` times per
// tile, vs decode's once) is the dominant per-call cost anyway, same
// reasoning FP32's widening used for its own arithmetic-only case.
//
// Correctness (small scale, n_in=n_out=32): block4-resident forward vs an
// all-SCATTERED layer holding the IDENTICAL already-quantized weights
// (scattered's values are fp8_decode_bits(code), not the pre-quantization
// float -- matches test_fp8_block4_scattered_divergence.cpp's convention,
// so both arms represent the exact same quantized value with no
// double-quantization mismatch). Tolerance, not bit-exact -- same
// accumulation-order rationale as the FP32 forward/backward tests: at
// n_in=32 every output sums 32 rows, and block4's per-thread reduction
// order differs from scattered's sequential walk.
//
// Timing (large scale, n_in=n_out=256, block4-only): many repeated
// disldo_forward calls over randomized input instances, per-call average
// recorded -- run once against the CURRENT kernel for a baseline, then
// again after the kernel change (same test, unmodified) for the after
// number.
//
// RESULT (measured on arch-sandbox, full-rate Zen2 AVX2, num_cpus=4, this
// test's n_in=256/n_out=256/batch=8/4096-tile config; 15 before + 15 after
// binary invocations, randomly interleaved to cancel thermal/scheduling
// drift, each invocation's own median-of-200-calls used as one sample):
// before median-of-medians 176200 ns/call (mean-of-medians 168429), after
// 166960 ns/call (mean-of-medians 156001) -- a real but modest ~5.5-8%
// speedup, similar magnitude to forward's FP32 result (not backward's
// larger one) since the decode step was deliberately left unchanged here
// -- the win is entirely from widening the same batch-loop arithmetic
// FP32's forward widening targeted. AVX2 codegen confirmed via objdump
// (vmulps ymm in the compiled block4 OpenMP-outlined function). See
// docs/research/linear_disldo.rst:
// disldo_forward.fp8_block4_avx2_column_pairing.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <algorithm>
#include <chrono>
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
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP8BiValues, COL_TYPE>;

int main() {
    std::mt19937 rng(2468);
    // E4M3 range is coarse near/under 1.0 -- keep magnitudes well inside
    // the well-represented band (same rationale as
    // test_fp8_block4_scattered_divergence.cpp's own comment), not
    // because tiny values are wrong, just to avoid this TEST's own
    // tolerance being swamped by ordinary quantization coarseness rather
    // than measuring the thing it's meant to measure.
    std::uniform_real_distribution<float> wdist(1.0f, 4.0f);
    std::uniform_real_distribution<float> idist(0.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    // ── correctness: small scale, block4-resident vs all-scattered ──
    {
        const std::size_t n_in = 32, n_out = 32;
        const SIZE_TYPE batch = 4;

        std::vector<float> dense_w(n_in * n_out);
        for (auto& v : dense_w)
            v = wdist(rng) * (rng() % 2 ? 1.0f : -1.0f);
        std::vector<uint8_t> weight_codes(n_in * n_out), importance_codes(n_in * n_out);
        std::vector<float> decoded_w(n_in * n_out), decoded_imp(n_in * n_out);
        for (std::size_t i = 0; i < dense_w.size(); ++i) {
            weight_codes[i] = fp8_quantize(dense_w[i]);
            decoded_w[i] = fp8_decode_bits(weight_codes[i]);
            const float iv = idist(rng);
            importance_codes[i] = fp8_quantize(iv);
            decoded_imp[i] = fp8_decode_bits(importance_codes[i]);
        }

        Weights weights_b4;
        weights_b4.connections.layout.rows = n_in;
        weights_b4.connections.layout.cols = n_out;
        block4_load_dense<SIZE_TYPE, FP8BiValues, COL_TYPE>(weights_b4, weight_codes.data(),
                                                            importance_codes.data(), n_in, n_out);

        std::vector<int> ptrs(n_in + 1), idx(n_in * n_out);
        for (std::size_t r = 0; r <= n_in; ++r)
            ptrs[r] = int(r * n_out);
        for (std::size_t r = 0; r < n_in; ++r)
            for (std::size_t c = 0; c < n_out; ++c)
                idx[r * n_out + c] = int(c);
        Weights weights_sc;
        weights_sc.connections = delta_csr_from_absolute<SIZE_TYPE, FP8BiValues, COL_TYPE>(
            ptrs, idx, decoded_w, decoded_imp, n_in, n_out, n_in * n_out * 4, n_in * n_out * 4,
            0.2f);
        weights_sc.block4.init(n_in, n_out);

        std::vector<float> x(batch * n_in);
        for (auto& v : x)
            v = xdist(rng);

        std::vector<float> y_b4(batch * n_out, 0.f), y_sc(batch * n_out, 0.f);
        disldo_forward<SIZE_TYPE, FP8BiValues, COL_TYPE>(x.data(), batch, SIZE_TYPE(n_in),
                                                         weights_b4, y_b4.data(), 4);
        disldo_forward<SIZE_TYPE, FP8BiValues, COL_TYPE>(x.data(), batch, SIZE_TYPE(n_in),
                                                         weights_sc, y_sc.data(), 4);
        double max_abs_err = 0.0;
        for (std::size_t i = 0; i < y_b4.size(); ++i)
            max_abs_err = std::max(max_abs_err, double(std::fabs(y_b4[i] - y_sc[i])));
        CHECK(max_abs_err < 1e-1,
              "block4 forward vs scattered (same quantized weights) max abs err %.6f too large",
              max_abs_err);
    }

    // ── timing: large scale, block4-only ──
    {
        const std::size_t n_in = 256, n_out = 256;
        const SIZE_TYPE batch = 8;

        std::vector<float> dense_w(n_in * n_out), dense_imp(n_in * n_out);
        for (auto& v : dense_w)
            v = wdist(rng) * (rng() % 2 ? 1.0f : -1.0f);
        for (auto& v : dense_imp)
            v = idist(rng);
        std::vector<uint8_t> weight_codes(n_in * n_out), importance_codes(n_in * n_out);
        for (std::size_t i = 0; i < dense_w.size(); ++i) {
            weight_codes[i] = fp8_quantize(dense_w[i]);
            importance_codes[i] = fp8_quantize(dense_imp[i]);
        }

        Weights weights;
        weights.connections.layout.rows = n_in;
        weights.connections.layout.cols = n_out;
        block4_load_dense<SIZE_TYPE, FP8BiValues, COL_TYPE>(weights, weight_codes.data(),
                                                            importance_codes.data(), n_in, n_out);
        const std::size_t expect_tiles = (n_in / 4) * (n_out / 4);
        CHECK(weights.block4.n_tiles() == expect_tiles, "expected %zu tiles, got %zu", expect_tiles,
              weights.block4.n_tiles());

        const int n_instances = 16;
        std::vector<float> x_all(std::size_t(n_instances) * batch * n_in);
        for (auto& v : x_all)
            v = xdist(rng);

        const int n_reps = 200;
        std::vector<int> order(n_reps);
        for (int i = 0; i < n_reps; ++i)
            order[i] = i % n_instances;
        std::shuffle(order.begin(), order.end(), rng);

        std::vector<double> call_ns(n_reps);
        std::vector<float> y_bench(batch * n_out);
        for (int i = 0; i < 20; ++i) { // warmup, not measured
            std::fill(y_bench.begin(), y_bench.end(), 0.f);
            const float* x = x_all.data() + std::size_t(order[i % n_reps]) * batch * n_in;
            disldo_forward<SIZE_TYPE, FP8BiValues, COL_TYPE>(x, batch, SIZE_TYPE(n_in), weights,
                                                             y_bench.data(), 4);
        }
        for (int i = 0; i < n_reps; ++i) {
            std::fill(y_bench.begin(), y_bench.end(), 0.f);
            const float* x = x_all.data() + std::size_t(order[i]) * batch * n_in;
            auto t0 = std::chrono::steady_clock::now();
            disldo_forward<SIZE_TYPE, FP8BiValues, COL_TYPE>(x, batch, SIZE_TYPE(n_in), weights,
                                                             y_bench.data(), 4);
            auto t1 = std::chrono::steady_clock::now();
            call_ns[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        std::sort(call_ns.begin(), call_ns.end());
        const double median_ns = call_ns[std::size_t(n_reps) / 2];
        const double mean_ns = std::accumulate(call_ns.begin(), call_ns.end(), 0.0) / n_reps;
        std::printf("TIMING disldo_forward fp8 block4 (n_in=%zu n_out=%zu batch=%d tiles=%zu, "
                    "%d calls): mean=%.1f ns/call median=%.1f ns/call\n",
                    n_in, n_out, int(batch), weights.block4.n_tiles(), n_reps, mean_ns, median_ns);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
