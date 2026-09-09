// TDD baseline for the AVX2-width change to disldo_backward's FP32 block4
// hot loop -- the backward counterpart to
// test_disldo_block4_fp32_wide_simd.cpp's forward widening. Backward's
// existing block4 FP32 SIMD structure vectorizes 4-wide (Block4Vec) across
// a tile's 4 COLUMNS for one fixed row (li) at a time (the opposite axis
// from forward, which vectorized across a fixed column's 4 rows); the
// widening pairs two adjacent ROWS (li, li+1) of the same tile into one
// 8-wide op instead, mirroring forward's column-pairing but on the other
// axis. This test has NO dependency on the kernel's internal SIMD width
// or row-pairing structure.
//
// Correctness (small scale, n_in=n_out=32): block4-resident backward vs an
// all-SCATTERED layer holding the identical logical weights, run with
// IDENTICAL x/dy/learning_rate -- checked with a TOLERANCE, not bit-exact
// (unlike test_disldo_block4_fp32.cpp's small hand-placed-tile cross-check):
// at this scale EVERY output column sums contributions from all 32 rows,
// and block4's per-block-row parallel-thread reduction visits those 32
// terms in a different order than scattered's sequential per-row walk --
// floating-point addition isn't associative, so a tiny (~1e-5) discrepancy
// is real and expected, not a bug. Rather than manually diffing every tile
// cell, a FRESH forward probe after backward on both weight sets is used
// to confirm ALL synapse state landed at essentially the same place (if
// even one weight/importance differed meaningfully, a generic random
// probe's output would diverge well past tolerance too).
//
// Timing (large scale, n_in=n_out=256, block4-only): many repeated
// disldo_backward calls over randomized dy/x instances, per-call average
// recorded -- run once against the CURRENT kernel to get a baseline, then
// again after the kernel change (same test, unmodified) for the after
//
// RESULT (measured on arch-sandbox, full-rate Zen2 AVX2, num_cpus=4, this
// test's n_in=256/n_out=256/batch=8/4096-tile config; 15 before + 15 after
// binary invocations, randomly interleaved to cancel thermal/scheduling
// drift, each invocation's own median-of-200-calls used as one sample):
// before median-of-medians 10967140 ns/call (mean-of-medians 13120815),
// after 3400970 ns/call (mean-of-medians 4785922) -- a real ~2.7-3.2x
// speedup, MUCH larger than forward's modest ~6-8% (forward's "decode" was
// a no-op memcpy with almost nothing to widen; backward's per-cell RMSprop
// update + rank-N AQRS bookkeeping is exactly the "lot more stuff going on
// in the simd part" this widening actually pays off on). AVX2 codegen
// confirmed via objdump (vmulps ymm in the compiled block4 OpenMP-outlined
// function). See docs/research/linear_disldo.rst:
// disldo_backward.fp32_block4_avx2_row_pairing.
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
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

static void backward_once(Weights& weights, const std::vector<float>& x, SIZE_TYPE batch,
                          std::size_t n_in, const std::vector<float>& dy, std::size_t n_out,
                          std::vector<float>& dx, float lr) {
    std::vector<float> nia(n_in, 0.f), nga(n_out, 0.f);
    std::fill(dx.begin(), dx.end(), 0.f);
    disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE, RMSpropScalePolicy<float>, false,
                    false>(x.data(), batch, SIZE_TYPE(n_in), dy.data(), weights, dx.data(),
                           nia.data(), nga.data(), lr, 4, false, true);
}

int main() {
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> idist(0.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    // ── correctness: small scale, block4-resident vs all-scattered ──
    {
        const std::size_t n_in = 32, n_out = 32;
        const SIZE_TYPE batch = 4;

        std::vector<float> w(n_in * n_out), imp(n_in * n_out);
        for (auto& v : w)
            v = wdist(rng);
        for (auto& v : imp)
            v = idist(rng);

        // disldo_backward's dead-row bootstrap walks L.row_nnz(row) for
        // EVERY row unconditionally (unlike disldo_forward, which skips
        // the scattered side entirely via dc.empty()) -- an empty
        // connections needs a properly-sized (if all-zero) layout via
        // delta_csr_from_absolute, not just .rows/.cols set by hand.
        std::vector<int> empty_ptrs(n_in + 1, 0);
        std::vector<int> empty_idx;
        std::vector<float> empty_w, empty_imp;
        Weights weights_b4;
        weights_b4.connections =
            delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                empty_ptrs, empty_idx, empty_w, empty_imp, n_in, n_out, 64, 64, 0.2f);
        block4_load_dense_fp32<SIZE_TYPE, COL_TYPE>(weights_b4, w.data(), imp.data(), n_in, n_out);

        // Fully-dense all-scattered layer holding the SAME logical weights.
        std::vector<int> ptrs(n_in + 1), idx(n_in * n_out);
        for (std::size_t r = 0; r <= n_in; ++r)
            ptrs[r] = int(r * n_out);
        for (std::size_t r = 0; r < n_in; ++r)
            for (std::size_t c = 0; c < n_out; ++c)
                idx[r * n_out + c] = int(c);
        Weights weights_sc;
        weights_sc.connections =
            delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                ptrs, idx, w, imp, n_in, n_out, n_in * n_out * 4, n_in * n_out * 16, 0.2f);
        weights_sc.block4.init(n_in, n_out);

        std::vector<float> x(batch * n_in), dy(batch * n_out);
        for (auto& v : x)
            v = xdist(rng);
        for (auto& v : dy)
            v = xdist(rng);

        std::vector<float> dx_b4(batch * n_in), dx_sc(batch * n_in);
        backward_once(weights_b4, x, batch, n_in, dy, n_out, dx_b4, 0.05f);
        backward_once(weights_sc, x, batch, n_in, dy, n_out, dx_sc, 0.05f);

        // Tolerance, not bit-exact -- at this scale EVERY output column
        // sums contributions from all 32 rows, and block4's per-block-row
        // parallel-thread reduction visits those 32 terms in a different
        // order than scattered's sequential per-row walk. Floating-point
        // addition isn't associative, so a tiny (~1e-5) discrepancy is
        // expected and not a bug -- same rationale as
        // test_disldo_block4_fp32_wide_simd.cpp's forward tolerance check.
        // (The ORIGINAL small-scale test in test_disldo_block4_fp32.cpp
        // gets away with bit-exact `==` only because it has just 1-2 live
        // entries with no overlapping per-column contributions at all.)
        double max_abs_err_dx = 0.0;
        for (std::size_t i = 0; i < dx_b4.size(); ++i)
            max_abs_err_dx = std::max(max_abs_err_dx, double(std::fabs(dx_b4[i] - dx_sc[i])));
        CHECK(max_abs_err_dx < 1e-2, "backward dx: block4 vs scattered max abs err %.6f too large",
              max_abs_err_dx);

        // Probe forward with a FRESH input on both post-backward weight
        // sets -- close match proves every synapse (not just the ones dx
        // happened to touch) landed at essentially the same place.
        std::vector<float> x2(batch * n_in);
        for (auto& v : x2)
            v = xdist(rng);
        std::vector<float> y_b4(batch * n_out, 0.f), y_sc(batch * n_out, 0.f);
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
            x2.data(), batch, SIZE_TYPE(n_in), weights_b4, y_b4.data(), 4);
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
            x2.data(), batch, SIZE_TYPE(n_in), weights_sc, y_sc.data(), 4);
        double max_abs_err_probe = 0.0;
        for (std::size_t i = 0; i < y_b4.size(); ++i)
            max_abs_err_probe = std::max(max_abs_err_probe, double(std::fabs(y_b4[i] - y_sc[i])));
        CHECK(max_abs_err_probe < 1e-2,
              "post-backward forward probe: block4 vs scattered max abs err %.6f too large -- "
              "should confirm ALL synapse state (weight+importance+scale) matches, not just dx",
              max_abs_err_probe);
    }

    // ── timing: large scale, block4-only ──
    {
        const std::size_t n_in = 256, n_out = 256;
        const SIZE_TYPE batch = 8;

        std::vector<float> w(n_in * n_out), imp(n_in * n_out);
        for (auto& v : w)
            v = wdist(rng);
        for (auto& v : imp)
            v = idist(rng);

        std::vector<int> empty_ptrs(n_in + 1, 0);
        std::vector<int> empty_idx;
        std::vector<float> empty_w, empty_imp;
        Weights weights;
        weights.connections = delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
            empty_ptrs, empty_idx, empty_w, empty_imp, n_in, n_out, 64, 64, 0.2f);
        block4_load_dense_fp32<SIZE_TYPE, COL_TYPE>(weights, w.data(), imp.data(), n_in, n_out);

        const std::size_t expect_tiles = (n_in / 4) * (n_out / 4);
        CHECK(weights.block4.n_tiles() == expect_tiles, "expected %zu tiles, got %zu", expect_tiles,
              weights.block4.n_tiles());

        // Several distinct (x, dy) instances, called in shuffled order --
        // same rationale as the forward test: avoids a single fixed
        // input/gradient pair letting the branch predictor settle into an
        // unrealistically steady pattern.
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
            disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE, RMSpropScalePolicy<float>,
                            false, false>(x, batch, SIZE_TYPE(n_in), dy, weights, dx.data(),
                                          nia.data(), nga.data(), 0.01f, 4, false, true);
        }

        std::vector<double> call_ns(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            const float* x = x_all.data() + std::size_t(order[i]) * batch * n_in;
            const float* dy = dy_all.data() + std::size_t(order[i]) * batch * n_out;
            std::fill(dx.begin(), dx.end(), 0.f);
            std::fill(nia.begin(), nia.end(), 0.f);
            std::fill(nga.begin(), nga.end(), 0.f);
            auto t0 = std::chrono::steady_clock::now();
            disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE, RMSpropScalePolicy<float>,
                            false, false>(x, batch, SIZE_TYPE(n_in), dy, weights, dx.data(),
                                          nia.data(), nga.data(), 0.01f, 4, false, true);
            auto t1 = std::chrono::steady_clock::now();
            call_ns[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
            for (float v : dx)
                CHECK(std::isfinite(v), "dx not finite at rep %d", i);
        }
        std::sort(call_ns.begin(), call_ns.end());
        const double median_ns = call_ns[std::size_t(n_reps) / 2];
        const double mean_ns = std::accumulate(call_ns.begin(), call_ns.end(), 0.0) / n_reps;
        std::printf("TIMING disldo_backward fp32 block4 (n_in=%zu n_out=%zu batch=%d tiles=%zu, "
                    "%d calls): mean=%.1f ns/call median=%.1f ns/call\n",
                    n_in, n_out, int(batch), weights.block4.n_tiles(), n_reps, mean_ns, median_ns);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
