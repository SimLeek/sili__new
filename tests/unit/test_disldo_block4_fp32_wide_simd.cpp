// TDD baseline for the AVX2-width change to disldo_forward's FP32 block4
// hot loop: it used to process one 4x4 tile's 4 columns one at a time
// (128-bit width); the change pairs each tile's two adjacent columns
// (LJ=0,1 and LJ=2,3 -- they share the same 4 input rows, and are laid
// out back-to-back in tdata) into one 256-bit op. This test has NO
// dependency on the kernel's internal SIMD width: correctness is checked
// against an independent dense-matmul reference computed directly in this
// file (kept here permanently, not as a toggle inside the kernel -- see
// conversation), so the SAME test/reference was used to measure both the
// pre-change and post-change per-call timing.
//
// RESULT (measured on arch-sandbox, full-rate Zen2 AVX2, num_cpus=4, this
// test's n_in=256/n_out=256/batch=8/4096-tile config; 15 before + 15 after
// binary invocations, randomly interleaved to cancel thermal/scheduling
// drift, each invocation's own median-of-200-calls used as one sample):
// before median-of-medians 173990 ns/call (mean-of-medians 161149), after
// 160920 ns/call (mean-of-medians 151265) -- a real but modest ~6-8%
// speedup, well inside the ~25-30% single-run noise band at this size.
// AVX2 codegen confirmed separately via objdump (vmulps ymm in the
// compiled block4 OpenMP-outlined function). See docs/research/
// linear_disldo.rst: disldo_forward.fp32_block4_avx2_column_pairing.
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

int main() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> idist(0.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    // Exact multiple of BLOCK4_TILE on both dims (no boundary tiles --
    // those are already covered by test_disldo_block4_fp32.cpp) and large
    // enough (64x64 = 4096 tiles, all adjacent-bc pairs even-counted) to
    // give a two-tiles-at-a-time widening plenty of real opportunities.
    const std::size_t n_in = 256, n_out = 256;
    const SIZE_TYPE batch = 8;

    std::vector<float> weight_values(n_in * n_out), importance_values(n_in * n_out);
    for (auto& w : weight_values)
        w = wdist(rng);
    for (auto& im : importance_values)
        im = idist(rng);

    Weights weights;
    weights.connections.layout.rows = n_in; // disldo_forward reads n_in/n_out from here, not
    weights.connections.layout.cols =
        n_out; // from block4 -- block4_load_dense_fp32 leaves it untouched.
    block4_load_dense_fp32<SIZE_TYPE, COL_TYPE>(weights, weight_values.data(),
                                                importance_values.data(), n_in, n_out);
    // No weights.recompute_stats() -- that walks the SCATTERED side's CSR
    // layout (elem_start etc), which this pure-block4 test never
    // initializes (same convention as test_block4_load_dense.cpp). Not
    // needed here: disldo_forward's scattered branch is skipped via
    // dc.empty() (total_nnz defaults to 0), and get_scale()'s defaults
    // (scale_rank=1, component 0 = 1.0) don't depend on it either.

    const std::size_t expect_tiles = (n_in / 4) * (n_out / 4);
    CHECK(weights.block4.n_tiles() == expect_tiles, "expected %zu tiles, got %zu", expect_tiles,
          weights.block4.n_tiles());

    // Several distinct input instances, called in a shuffled order below --
    // avoids a single fixed input letting the branch predictor/cache
    // settle into an unrealistically steady pattern across all 200 calls.
    const int n_instances = 16;
    std::vector<float> x_all(std::size_t(n_instances) * batch * n_in);
    for (auto& v : x_all)
        v = xdist(rng);

    // ── independent dense-matmul reference (kept here permanently; NOT a
    // kernel toggle -- just this test's ground truth) ──
    auto dense_reference = [&](const float* x, std::vector<float>& y_ref) {
        std::fill(y_ref.begin(), y_ref.end(), 0.f);
        for (SIZE_TYPE b = 0; b < batch; ++b) {
            for (std::size_t r = 0; r < n_in; ++r) {
                const float xv = x[std::size_t(b) * n_in + r];
                if (xv == 0.f)
                    continue;
                const float* wr = weight_values.data() + r * n_out;
                float* yr = y_ref.data() + std::size_t(b) * n_out;
                for (std::size_t c = 0; c < n_out; ++c)
                    yr[c] += wr[c] * xv;
            }
        }
    };

    // ── correctness: every input instance, block4 forward vs reference ──
    for (int inst = 0; inst < n_instances; ++inst) {
        const float* x = x_all.data() + std::size_t(inst) * batch * n_in;
        std::vector<float> y_ref(batch * n_out, 0.f);
        dense_reference(x, y_ref);

        std::vector<float> y(batch * n_out, 0.f);
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x, batch, SIZE_TYPE(n_in),
                                                                     weights, y.data(), 4);

        double max_abs_err = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i)
            max_abs_err = std::max(max_abs_err, double(std::fabs(y[i] - y_ref[i])));
        // Accumulation order differs (block4's tile/thread-parallel sum vs
        // this reference's plain row-major sum) so bit-exactness isn't
        // expected at this size -- a tight tolerance still catches any
        // real correctness break (wrong tile paired, wrong lane, dropped
        // column, etc).
        CHECK(max_abs_err < 1e-2,
              "instance %d: block4 forward vs dense reference max abs err %.6f too large", inst,
              max_abs_err);
    }

    // ── timing: many repeated calls over a random order of the same
    // instances above (num_cpus fixed so before/after numbers are
    // comparable on the same machine) ──
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
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x, batch, SIZE_TYPE(n_in),
                                                                     weights, y_bench.data(), 4);
    }
    for (int i = 0; i < n_reps; ++i) {
        std::fill(y_bench.begin(), y_bench.end(), 0.f);
        const float* x = x_all.data() + std::size_t(order[i]) * batch * n_in;
        auto t0 = std::chrono::steady_clock::now();
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x, batch, SIZE_TYPE(n_in),
                                                                     weights, y_bench.data(), 4);
        auto t1 = std::chrono::steady_clock::now();
        call_ns[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }
    std::sort(call_ns.begin(), call_ns.end());
    const double median_ns = call_ns[std::size_t(n_reps) / 2];
    const double mean_ns = std::accumulate(call_ns.begin(), call_ns.end(), 0.0) / n_reps;
    std::printf("TIMING disldo_forward fp32 block4 (n_in=%zu n_out=%zu batch=%d tiles=%zu, "
                "%d calls): mean=%.1f ns/call median=%.1f ns/call\n",
                n_in, n_out, int(batch), weights.block4.n_tiles(), n_reps, mean_ns, median_ns);

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
