// TDD (written before the implementation exists): fp4quant.hpp's
// fp4_stochastic_normal01() -- a standard-normal (mean 0, std 1) sample
// off the SAME thread-local xorshift64* state fp4_stochastic_uniform01()
// already uses (Box-Muller transform, two uniform draws per normal pair).
// Needed for the per-neuron plasticity-reset mechanism's fresh_sample()
// fan-in-scaled reinit (scale = 1/sqrt(n_in) is applied by the CALLER,
// this function only needs to produce a raw N(0,1) draw). See
// docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design.
#include "../../sili/lib/headers/fp4quant.hpp"
#include <cmath>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

static void test_mean_and_std() {
    fp4_seed_stochastic_rng(0);
    const int N = 500000;
    double sum = 0.0, sum_sq = 0.0;
    bool saw_nan_or_inf = false;
    for (int i = 0; i < N; ++i) {
        const float v = fp4_stochastic_normal01();
        if (!std::isfinite(v))
            saw_nan_or_inf = true;
        sum += double(v);
        sum_sq += double(v) * double(v);
    }
    const double mean = sum / N;
    const double var = sum_sq / N - mean * mean;
    CHECK(!saw_nan_or_inf, "fp4_stochastic_normal01 produced a NaN/Inf draw");
    CHECK(std::abs(mean) < 0.02, "mean should be ~0, got %f", mean);
    CHECK(std::abs(var - 1.0) < 0.05, "variance should be ~1, got %f", var);
}

static void test_different_seeds_differ() {
    fp4_seed_stochastic_rng(1);
    const int N = 100;
    float seq_a[N];
    for (int i = 0; i < N; ++i)
        seq_a[i] = fp4_stochastic_normal01();

    fp4_seed_stochastic_rng(2);
    bool any_diff = false;
    for (int i = 0; i < N; ++i) {
        if (fp4_stochastic_normal01() != seq_a[i]) {
            any_diff = true;
            break;
        }
    }
    CHECK(any_diff, "different seeds should not produce an identical sequence");
}

static void test_same_seed_reproducible() {
    fp4_seed_stochastic_rng(42);
    const int N = 50;
    float seq_a[N];
    for (int i = 0; i < N; ++i)
        seq_a[i] = fp4_stochastic_normal01();

    fp4_seed_stochastic_rng(42);
    for (int i = 0; i < N; ++i) {
        const float v = fp4_stochastic_normal01();
        CHECK(v == seq_a[i],
              "re-seeding with the same value should reproduce the sequence exactly");
    }
}

int main() {
    test_mean_and_std();
    test_different_seeds_differ();
    test_same_seed_reproducible();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
