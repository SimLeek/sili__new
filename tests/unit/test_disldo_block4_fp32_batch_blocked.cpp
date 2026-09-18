// Phase 1 of the batch-blocked accumulation rollout (see
// TODO_BATCH_BLOCKING.md): disldo_forward's wide (column-partitioned)
// block4 path, at batch >= BLOCK4_BATCH_BLOCK_THRESHOLD, now transposes
// the input once and accumulates into a per-thread PRIVATE column-major
// buffer (block4_batch_accumulate, block4.hpp) instead of writing
// directly into `output` per-sample. This test's only job is to catch a
// wiring bug in that new path -- wrong column offset, wrong row mapped to
// the wrong transposed slot, remainder-batch samples dropped, a
// boundary (non-multiple-of-4) tile mishandled, etc -- by comparing
// against an independent dense-matmul reference AND against the SAME
// weights run below threshold (old scalar path), which must agree with
// each other since both compute the same mathematical result.
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
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

static void dense_reference(const float* x, SIZE_TYPE batch, std::size_t n_in, std::size_t n_out,
                            const std::vector<float>& weight_values, std::vector<float>& y_ref) {
    std::fill(y_ref.begin(), y_ref.end(), 0.f);
    for (SIZE_TYPE b = 0; b < batch; ++b) {
        for (std::size_t r = 0; r < n_in; ++r) {
            const float xv = x[std::size_t(b) * n_in + r];
            const float* wr = weight_values.data() + r * n_out;
            float* yr = y_ref.data() + std::size_t(b) * n_out;
            for (std::size_t c = 0; c < n_out; ++c)
                yr[c] += wr[c] * xv;
        }
    }
}

// n_in/n_out deliberately NOT multiples of BLOCK4_TILE(4) -- exercises
// the boundary (partial) row/col block, which the transposed path must
// handle via the same row_idx[li]=0-fallback + zero-weight masking as
// the existing scalar path. Callers below pick num_cpus/n_out to hit
// BOTH the wide and narrow dispatch (see main()'s comment) -- this
// helper itself doesn't care which, it just checks the math is right.
static void run_case(const char* label, std::size_t n_in, std::size_t n_out, SIZE_TYPE batch,
                     int num_cpus, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> idist(0.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    std::vector<float> weight_values(n_in * n_out), importance_values(n_in * n_out);
    for (auto& w : weight_values)
        w = wdist(rng);
    for (auto& im : importance_values)
        im = idist(rng);

    Weights weights;
    weights.connections.layout.rows = n_in;
    weights.connections.layout.cols = n_out;
    block4_load_dense_fp32<SIZE_TYPE, COL_TYPE>(weights, weight_values.data(),
                                                importance_values.data(), n_in, n_out);

    std::vector<float> x(std::size_t(batch) * n_in);
    for (auto& v : x)
        v = xdist(rng);

    std::vector<float> y_ref(std::size_t(batch) * n_out, 0.f);
    dense_reference(x.data(), batch, n_in, n_out, weight_values, y_ref);

    // Above threshold: exercises the new transposed/private-buffer path.
    std::vector<float> y_blocked(std::size_t(batch) * n_out, 0.f);
    disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        x.data(), batch, SIZE_TYPE(n_in), weights, y_blocked.data(), num_cpus);

    // Below threshold: same math, old scalar per-sample path -- computed
    // by feeding the SAME batch as `batch` separate size-1 calls (forces
    // the scalar branch regardless of the real batch value) and
    // concatenating, since disldo_forward's own threshold gate is on the
    // `batch` parameter itself, not something this test should reach
    // into.
    std::vector<float> y_scalar(std::size_t(batch) * n_out, 0.f);
    for (SIZE_TYPE b = 0; b < batch; ++b)
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
            x.data() + std::size_t(b) * n_in, SIZE_TYPE(1), SIZE_TYPE(n_in), weights,
            y_scalar.data() + std::size_t(b) * n_out, num_cpus);

    double max_err_vs_ref = 0.0, max_err_vs_scalar = 0.0;
    for (std::size_t i = 0; i < y_blocked.size(); ++i) {
        max_err_vs_ref = std::max(max_err_vs_ref, double(std::fabs(y_blocked[i] - y_ref[i])));
        max_err_vs_scalar =
            std::max(max_err_vs_scalar, double(std::fabs(y_blocked[i] - y_scalar[i])));
    }
    // Reassociation-only difference (block4's tile/thread-parallel sum
    // order vs a plain row-major reference/per-sample-call order) -- not
    // bit-exact, but a wiring bug (wrong column, dropped row, dropped
    // remainder sample) would blow well past this.
    CHECK(max_err_vs_ref < 1e-2, "%s: blocked vs dense-reference max abs err %.6f too large", label,
          max_err_vs_ref);
    CHECK(max_err_vs_scalar < 1e-2, "%s: blocked vs scalar-path max abs err %.6f too large", label,
          max_err_vs_scalar);
}

int main() {
    // IMPORTANT: disldo_forward picks WIDE (column-partitioned) vs
    // NARROW (tree-reduction) per call based on
    // cols_per_thread = ceil(n_out/BLOCK4_TILE) / num_cpus vs
    // COLUMN_PARTITION_MIN_COLS_PER_THREAD(12) -- Phase 1 only touched
    // the wide path's blocked branch, Phase 2 added the narrow path's.
    // An earlier version of this file picked (n_in,n_out,num_cpus)
    // combos that all worked out to cols_per_thread<12 without checking
    // the arithmetic, so it accidentally tested ONLY the narrow path the
    // whole time it claimed to test "the wide path" -- Phase 1's own
    // blocked-wide branch was actually verified only by the
    // pre-existing test_disldo_block4_fp32_wide_simd.cpp (n_out=256,
    // num_cpus=4 -> cols_per_thread=16, genuinely wide). Fixed here:
    // both groups below are picked with the real formula checked, and
    // each covers the below/at/above-threshold boundary independently.

    // ── WIDE path (cols_per_thread >= 12) ──
    // n_out=200, num_cpus=4: ceil(200/4)=50 block-cols, 50/4=12 (exact
    // boundary of the wide/narrow split itself -- also worth covering).
    run_case("WIDE n_in=61 n_out=200 batch=7 (below threshold)", 61, 200, 7, 4, 11);
    run_case("WIDE n_in=61 n_out=200 batch=8 (at threshold)", 61, 200, 8, 4, 12);
    run_case("WIDE n_in=61 n_out=200 batch=9 (above threshold)", 61, 200, 9, 4, 13);
    // n_out=256, num_cpus=4: 64 block-cols, cols_per_thread=16 -- well
    // inside the wide regime, multiple BLOCK4_BATCH_BLOCK_B(8)-wide
    // chunks plus a remainder tail, and a large exact-multiple batch.
    run_case("WIDE n_in=64 n_out=256 batch=37 (partial tail chunk)", 64, 256, 37, 4, 14);
    run_case("WIDE n_in=128 n_out=256 batch=256 (no remainder)", 128, 256, 256, 4, 15);

    // ── NARROW path (cols_per_thread < 12) ──
    // n_out=97, num_cpus=4: ceil(97/4)=25 block-cols, cols_per_thread=6.
    run_case("NARROW n_in=61 n_out=97 batch=7 (below threshold)", 61, 97, 7, 4, 1);
    run_case("NARROW n_in=61 n_out=97 batch=8 (at threshold)", 61, 97, 8, 4, 2);
    run_case("NARROW n_in=61 n_out=97 batch=9 (above threshold)", 61, 97, 9, 4, 3);
    // n_out=128, num_cpus=4: 32 block-cols, cols_per_thread=8.
    run_case("NARROW n_in=64 n_out=128 batch=37 (partial tail chunk)", 64, 128, 37, 4, 4);
    run_case("NARROW n_in=128 n_out=128 batch=256 (no remainder)", 128, 128, 256, 4, 5);
    // Boundary row/col block (n_in/n_out not multiples of BLOCK4_TILE),
    // num_cpus=6: ceil(101/4)=26 block-cols, cols_per_thread=4.
    run_case("NARROW n_in=61 n_out=101 batch=64, num_cpus=6", 61, 101, 64, 6, 6);

    if (g_fail == 0)
        std::printf("PASS: all disldo_block4_fp32_batch_blocked checks\n");
    return g_fail;
}
