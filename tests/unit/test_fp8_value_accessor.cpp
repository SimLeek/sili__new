// Correctness check for ValueAccessor<FP8BiValues> (delta_csr_types.hpp)
// -- the trait that makes FP8BiValues a drop-in VALUES_TYPE for
// SparseLinearWeightsDelta/disldo_forward/disldo_backward, same role as
// ValueAccessor<FP4BiPacked>/ValueAccessor<DeltaCSRBiValues<T>>. No
// existing test covers ValueAccessor<FP4BiPacked> directly either
// (checked -- it's exercised only via full SparseLinearLayer tests), so
// this is this trait's own first direct coverage rather than a gap
// specific to FP8.
#include "../../sili/lib/headers/delta_csr_types.hpp"
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

int main() {
    using VA = ValueAccessor<FP8BiValues>;
    static_assert(std::is_same_v<VA::value_type, float>,
                  "value_type should be float (decoded), not uint8_t");

    FP8BiValues v;
    VA::reserve(v, 8);
    VA::resize(v, 4, 0.0f, 0.0f);
    CHECK(v.weights.size() == 4 && v.importance.size() == 4, "resize didn't size both arrays");

    // set/get round-trip: within E4M3's own quantization error, not exact
    // (that's fp8_bitshift's own job to verify) -- this test is about the
    // ACCESSOR plumbing (does set() reach both arrays, does get_w vs
    // get_imp read the RIGHT one) not the codec itself.
    VA::set(v, 0, 1.5f, -2.5f);
    VA::set(v, 1, 0.0f, 100.0f);
    CHECK(std::abs(VA::get_w(v, 0) - 1.5f) < 0.05f, "get_w(0) = %f, expected ~1.5",
          double(VA::get_w(v, 0)));
    CHECK(std::abs(VA::get_imp(v, 0) - (-2.5f)) < 0.05f, "get_imp(0) = %f, expected ~-2.5",
          double(VA::get_imp(v, 0)));
    CHECK(VA::get_w(v, 1) == 0.0f, "get_w(1) = %f, expected exactly 0", double(VA::get_w(v, 1)));
    CHECK(std::abs(VA::get_imp(v, 1) - 100.0f) < 5.0f, "get_imp(1) = %f, expected ~100",
          double(VA::get_imp(v, 1)));

    // set_stochastic: statistically unbiased, same convention as
    // ValueAccessor<FP4BiPacked>::set_stochastic's own contract.
    fp4_seed_stochastic_rng(0);
    double sum_w = 0.0, sum_imp = 0.0;
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        VA::set_stochastic(v, 2, 0.31f, -7.4f);
        sum_w += double(VA::get_w(v, 2));
        sum_imp += double(VA::get_imp(v, 2));
    }
    CHECK(std::abs(sum_w / N - 0.31) < 0.02, "set_stochastic weight mean %f off target 0.31",
          sum_w / N);
    CHECK(std::abs(sum_imp / N - (-7.4)) < 0.5, "set_stochastic importance mean %f off target -7.4",
          sum_imp / N);

    // move: memmove semantics, byte-for-byte (both arrays).
    VA::resize(v, 6, 0.0f, 0.0f);
    VA::set(v, 3, 3.0f, 4.0f);
    VA::set(v, 4, 5.0f, 6.0f);
    VA::move(v, 0, 3, 2); // shift [3,4] -> [0,1]
    CHECK(v.weights[0] == v.weights[3] && v.weights[1] == v.weights[4],
          "move() didn't correctly shift the weights array");
    CHECK(v.importance[0] == v.importance[3] && v.importance[1] == v.importance[4],
          "move() didn't correctly shift the importance array");

    // projected_byte_size: 2 bytes/element (1 weight + 1 importance byte),
    // matching the docstring -- distinguishes this from
    // ValueAccessor<DeltaCSRBiValues<float>>'s 8 bytes/element (2*sizeof(float)).
    CHECK(VA::projected_byte_size(100) == 200, "projected_byte_size(100) = %zu, expected 200",
          VA::projected_byte_size(100));

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
