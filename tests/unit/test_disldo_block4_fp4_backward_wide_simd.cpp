// TDD baseline for an AVX2-width change to disldo_backward's FP4 block4
// hot loop -- the FP4 counterpart to
// test_disldo_block4_fp8_backward_wide_simd.cpp's backward widening.
// FP4's existing block4 backward structure decodes via scalar FP4_TABLE
// lookups (one BiPacked byte -> two nibbles) rather than SIMD bit-shift --
// deliberately, per disldo_backward.fp4_table_decode -- so this pairs two
// adjacent ROWS of the same tile into one 8-wide op the same way FP8's row
// pairing does, with two real differences from FP8: (1) FP4BiPacked packs
// weight+importance into ONE byte per slot (not two separate byte arrays),
// so decode is one byte read + two FP4_TABLE lookups per (li,lj), and
// write-back is a single bit-OR pack (uint8_t((new_imp<<4)|new_w)); (2)
// FP4's gradient-accumulation terms (mrow/mcol/mgamma) use a "quant_floor"
// -- the stored quantized weight with a zero_escape_eps substitution at
// exactly quant==0, giving a currently-zero-weight cell a real
// synaptogenesis growth signal -- where FP8 has no such floor and uses
// cw_orig directly. See disldo_backward.fp4_block4_avx2_row_pairing in
// docs/research/linear_disldo.rst.
//
// Correctness (small scale, n_in=n_out=32): block4-resident backward vs
// an all-SCATTERED layer holding the IDENTICAL already-quantized weights
// (scattered's values are fp4_decode_bits(code), matching
// test_fp4_bitshift.cpp/test_block4_scattered_divergence.cpp's convention
// -- no double-quantization mismatch), run with IDENTICAL x/dy/
// learning_rate. Tolerance, not bit-exact -- same accumulation-order
// rationale as the FP32/FP8 backward tests. A fresh forward probe after
// backward on both weight sets additionally confirms ALL synapse state
// (not just dx) landed at essentially the same place.
//
// Timing (large scale, n_in=n_out=256, block4-only): many repeated
// disldo_backward calls over randomized dy/x instances, per-call average
// recorded -- run once against the CURRENT kernel to get a baseline, then
// again after the kernel change (same test, unmodified) for the after
// number. See docs/research/linear_disldo.rst:
// disldo_backward.fp4_block4_avx2_row_pairing.
//
// RESULT (measured on arch-sandbox, full-rate Zen2 AVX2, num_cpus=4, this
// test's n_in=256/n_out=256/batch=8/4096-tile config; 15 before + 15 after
// binary invocations, randomly interleaved to cancel thermal/scheduling
// drift, each invocation's own median-of-200-calls used as one sample):
// before median-of-medians 9311840 ns/call (mean-of-medians 11626223),
// after 3545899 ns/call (mean-of-medians 5461988) -- a real **~2.63x
// speedup**, matching FP8 backward's ~2.65-2.9x almost exactly (same
// reason: real per-cell RMSprop + rank-N AQRS bookkeeping to widen, unlike
// forward's much smaller/noise-dominated gain -- see
// disldo_forward.fp4_block4_avx2_column_pairing for that contrasting
// result and the local-vs-remote lesson it taught). Both before and after
// showed real bimodal run-to-run variance (a ~2x high/low split within
// each arm, likely thermal/frequency-scaling, affecting both arms
// similarly) -- the median-of-15 is what actually separates signal from
// that noise; a single non-interleaved before/after pair would have
// understated the win (one such pair measured only ~14% before this
// interleaved run was done, purely from landing in each arm's opposite
// mode). Local-only sanity check (laptop, AMD Ryzen 7 3750H) also showed a
// consistent, same-direction ~3.36x (8185684 -> 2436012 ns/call,
// single-pair, not interleaved) -- confirms the direction, not cited as
// the real number, per the lesson from FP4 forward's local/remote
// mismatch.
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
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static void backward_once(Weights& weights, const std::vector<float>& x, SIZE_TYPE batch,
                          std::size_t n_in, const std::vector<float>& dy, std::size_t n_out,
                          std::vector<float>& dx, float lr) {
    std::vector<float> nia(n_in, 0.f), nga(n_out, 0.f);
    std::fill(dx.begin(), dx.end(), 0.f);
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        x.data(), batch, SIZE_TYPE(n_in), dy.data(), weights, dx.data(), nia.data(), nga.data(), lr,
        4, false, true);
}

int main() {
    std::mt19937 rng(24680);
    // FP4's E2M1-ish range is very coarse -- keep magnitudes in a
    // well-represented band, same rationale as the FP8 test's comment.
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
            weight_codes[i] = fp4_quantize(dense_w[i]);
            decoded_w[i] = fp4_decode_bits(weight_codes[i]);
            const float iv = idist(rng);
            importance_codes[i] = fp4_quantize(iv);
            decoded_imp[i] = fp4_decode_bits(importance_codes[i]);
        }

        Weights weights_b4;
        std::vector<int> empty_ptrs(n_in + 1, 0);
        std::vector<int> empty_idx;
        std::vector<float> empty_w, empty_imp;
        weights_b4.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            empty_ptrs, empty_idx, empty_w, empty_imp, n_in, n_out, 64, 64, 0.2f);
        block4_load_dense<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights_b4, weight_codes.data(),
                                                            importance_codes.data(), n_in, n_out);

        std::vector<int> ptrs(n_in + 1), idx(n_in * n_out);
        for (std::size_t r = 0; r <= n_in; ++r)
            ptrs[r] = int(r * n_out);
        for (std::size_t r = 0; r < n_in; ++r)
            for (std::size_t c = 0; c < n_out; ++c)
                idx[r * n_out + c] = int(c);
        Weights weights_sc;
        weights_sc.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            ptrs, idx, decoded_w, decoded_imp, n_in, n_out, n_in * n_out * 4, n_in * n_out * 4,
            0.2f);
        weights_sc.block4.init(n_in, n_out);

        std::vector<float> x(batch * n_in), dy(batch * n_out);
        for (auto& v : x)
            v = xdist(rng);
        for (auto& v : dy)
            v = xdist(rng);

        std::vector<float> dx_b4(batch * n_in), dx_sc(batch * n_in);
        backward_once(weights_b4, x, batch, n_in, dy, n_out, dx_b4, 0.05f);
        backward_once(weights_sc, x, batch, n_in, dy, n_out, dx_sc, 0.05f);

        double max_abs_err_dx = 0.0;
        for (std::size_t i = 0; i < dx_b4.size(); ++i)
            max_abs_err_dx = std::max(max_abs_err_dx, double(std::fabs(dx_b4[i] - dx_sc[i])));
        CHECK(max_abs_err_dx < 1e-1, "backward dx: block4 vs scattered max abs err %.6f too large",
              max_abs_err_dx);

        std::vector<float> x2(batch * n_in);
        for (auto& v : x2)
            v = xdist(rng);
        std::vector<float> y_b4(batch * n_out, 0.f), y_sc(batch * n_out, 0.f);
        disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(x2.data(), batch, SIZE_TYPE(n_in),
                                                         weights_b4, y_b4.data(), 4);
        disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(x2.data(), batch, SIZE_TYPE(n_in),
                                                         weights_sc, y_sc.data(), 4);
        double max_abs_err_probe = 0.0;
        for (std::size_t i = 0; i < y_b4.size(); ++i)
            max_abs_err_probe = std::max(max_abs_err_probe, double(std::fabs(y_b4[i] - y_sc[i])));
        CHECK(max_abs_err_probe < 1e-1,
              "post-backward forward probe: block4 vs scattered max abs err %.6f too large",
              max_abs_err_probe);
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
            weight_codes[i] = fp4_quantize(dense_w[i]);
            importance_codes[i] = fp4_quantize(dense_imp[i]);
        }

        Weights weights;
        std::vector<int> empty_ptrs(n_in + 1, 0);
        std::vector<int> empty_idx;
        std::vector<float> empty_w, empty_imp;
        weights.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            empty_ptrs, empty_idx, empty_w, empty_imp, n_in, n_out, 64, 64, 0.2f);
        block4_load_dense<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights, weight_codes.data(),
                                                            importance_codes.data(), n_in, n_out);
        const std::size_t expect_tiles = (n_in / 4) * (n_out / 4);
        CHECK(weights.block4.n_tiles() == expect_tiles, "expected %zu tiles, got %zu", expect_tiles,
              weights.block4.n_tiles());

        const int n_instances = 16;
        std::vector<float> x_all(std::size_t(n_instances) * batch * n_in);
        std::vector<float> dy_all(std::size_t(n_instances) * batch * n_out);
        for (auto& v : x_all)
            v = xdist(rng);
        for (auto& v : dy_all)
            v = xdist(rng);

        const int n_reps = 200;
        std::vector<int> order(n_reps);
        for (int i = 0; i < n_reps; ++i)
            order[i] = i % n_instances;
        std::shuffle(order.begin(), order.end(), rng);

        std::vector<float> dx(batch * n_in), nia(n_in, 0.f), nga(n_out, 0.f);
        for (int i = 0; i < 20; ++i) { // warmup, not measured
            const float* x = x_all.data() + std::size_t(order[i % n_reps]) * batch * n_in;
            const float* dy = dy_all.data() + std::size_t(order[i % n_reps]) * batch * n_out;
            std::fill(dx.begin(), dx.end(), 0.f);
            std::fill(nia.begin(), nia.end(), 0.f);
            std::fill(nga.begin(), nga.end(), 0.f);
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false,
                            false>(x, batch, SIZE_TYPE(n_in), dy, weights, dx.data(), nia.data(),
                                   nga.data(), 0.01f, 4, false, true);
        }

        std::vector<double> call_ns(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            const float* x = x_all.data() + std::size_t(order[i]) * batch * n_in;
            const float* dy = dy_all.data() + std::size_t(order[i]) * batch * n_out;
            std::fill(dx.begin(), dx.end(), 0.f);
            std::fill(nia.begin(), nia.end(), 0.f);
            std::fill(nga.begin(), nga.end(), 0.f);
            auto t0 = std::chrono::steady_clock::now();
            disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false,
                            false>(x, batch, SIZE_TYPE(n_in), dy, weights, dx.data(), nia.data(),
                                   nga.data(), 0.01f, 4, false, true);
            auto t1 = std::chrono::steady_clock::now();
            call_ns[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
            for (float v : dx)
                CHECK(std::isfinite(v), "dx not finite at rep %d", i);
        }
        std::sort(call_ns.begin(), call_ns.end());
        const double median_ns = call_ns[std::size_t(n_reps) / 2];
        const double mean_ns = std::accumulate(call_ns.begin(), call_ns.end(), 0.0) / n_reps;
        std::printf("TIMING disldo_backward fp4 block4 (n_in=%zu n_out=%zu batch=%d tiles=%zu, "
                    "%d calls): mean=%.1f ns/call median=%.1f ns/call\n",
                    n_in, n_out, int(batch), weights.block4.n_tiles(), n_reps, mean_ns, median_ns);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
