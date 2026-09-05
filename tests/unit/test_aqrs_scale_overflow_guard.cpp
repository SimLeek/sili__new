// AQRS scale/additive channel bulk raw-vector accessors (task #295
// follow-up / task #286 "virtual neuron" exposure): get/set_value_scale_
// raw_vector, get/set_output_scale_raw_vector, get/set_additive_u_raw_
// vector, get/set_additive_v_raw_vector on SparseLinearWeightsDelta
// (delta_csr_types.hpp). Built to let a caller (Python) read/correct the
// AQRS scale envelope in ONE call instead of n*rank individual accessor
// calls -- the real motivation: a raised scale_rank_max/additive_rank_max
// (task #295) let a real fp8 MQAR curriculum run's per-channel
// value_scale_k/output_scale_k grow unbounded (get_scale()'s combined
// envelope has no clamp anywhere), overflowing S in the forward pass and
// NaN-collapsing training. This test proves the exposure primitive
// itself is correct (round-trips, size-preserving); the actual
// clip+auto-correct policy lives in Python (sili_peridot's own
// _overflow_guard_array, sili/sparse_rnn.py).
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include <cstdio>
#include <cmath>
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
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

static const std::size_t N = 8;

static Weights make_layer() {
    Weights w;
    std::vector<SIZE_TYPE> ptrs(N + 1);
    std::vector<SIZE_TYPE> idx(N * N);
    std::vector<float> vals(N * N, 1.0f), imp(N * N, 1.0f);
    for (std::size_t r = 0; r < N; ++r) {
        ptrs[r] = SIZE_TYPE(r * N);
        for (std::size_t c = 0; c < N; ++c)
            idx[r * N + c] = SIZE_TYPE(c);
    }
    ptrs[N] = SIZE_TYPE(N * N);
    w.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, vals, imp, N, N, N * N * 2, N * N * 2);
    w.out_degree.assign(N, SIZE_TYPE(N));
    return w;
}

// Bulk value_scale/output_scale vectors are empty until the first
// set_*_raw_k touches them (lazy sizing, matches every other AQRS
// storage member's own convention).
static void test_empty_before_any_write() {
    Weights w = make_layer();
    CHECK(w.get_value_scale_raw_vector().empty(), "value_scale should start empty");
    CHECK(w.get_output_scale_raw_vector().empty(), "output_scale should start empty");
    CHECK(w.get_additive_u_raw_vector().empty(), "additive_u should start empty");
    CHECK(w.get_additive_v_raw_vector().empty(), "additive_v should start empty");
}

// Round-trip: bulk-get, mutate, bulk-set, confirm per-element accessors
// see the SAME values -- proves the bulk vector really is the live
// backing storage (a copy that silently diverged would fail this).
static void test_value_scale_round_trip() {
    Weights w = make_layer();
    w.set_scale_rank_max(4);
    w.set_scale_rank(4);
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t k = 0; k < 4; ++k)
            w.set_value_scale_raw_k(r, k, float(r * 10 + k));

    auto vs = w.get_value_scale_raw_vector();
    CHECK(vs.size() == N * 4, "expected value_scale vector size %zu, got %zu", N * 4, vs.size());
    for (auto& x : vs)
        x *= 2.0f;
    w.set_value_scale_raw_vector(vs);

    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t k = 0; k < 4; ++k) {
            float expected = float(r * 10 + k) * 2.0f;
            float got = w.get_value_scale_k(r, k);
            CHECK(std::fabs(got - expected) < 1e-4f,
                  "row=%zu k=%zu: expected %.3f after bulk round-trip, got %.3f", r, k, expected,
                  got);
        }
}

static void test_output_scale_round_trip() {
    Weights w = make_layer();
    w.set_scale_rank_max(3);
    w.set_scale_rank(3);
    for (std::size_t c = 0; c < N; ++c)
        for (std::size_t k = 0; k < 3; ++k)
            w.set_output_scale_raw_k(c, k, float(c + k) * 0.5f);

    auto os = w.get_output_scale_raw_vector();
    CHECK(os.size() == N * 3, "expected output_scale vector size %zu, got %zu", N * 3, os.size());
    for (auto& x : os)
        x += 100.0f;
    w.set_output_scale_raw_vector(os);

    for (std::size_t c = 0; c < N; ++c)
        for (std::size_t k = 0; k < 3; ++k) {
            float expected = float(c + k) * 0.5f + 100.0f;
            float got = w.get_output_scale_k(c, k);
            CHECK(std::fabs(got - expected) < 1e-4f,
                  "col=%zu k=%zu: expected %.3f after bulk round-trip, got %.3f", c, k, expected,
                  got);
        }
}

static void test_additive_u_v_round_trip() {
    Weights w = make_layer();
    w.set_additive_rank_max(5);
    w.set_additive_rank(5);
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t k = 0; k < 5; ++k)
            w.set_additive_u_raw_k(r, k, float(r) - float(k));
    for (std::size_t c = 0; c < N; ++c)
        for (std::size_t k = 0; k < 5; ++k)
            w.set_additive_v_raw_k(c, k, float(c) + float(k));

    auto au = w.get_additive_u_raw_vector();
    auto av = w.get_additive_v_raw_vector();
    CHECK(au.size() == N * 5, "expected additive_u size %zu, got %zu", N * 5, au.size());
    CHECK(av.size() == N * 5, "expected additive_v size %zu, got %zu", N * 5, av.size());
    for (auto& x : au)
        x = -x;
    for (auto& x : av)
        x = -x;
    w.set_additive_u_raw_vector(au);
    w.set_additive_v_raw_vector(av);

    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t k = 0; k < 5; ++k) {
            float expected = -(float(r) - float(k));
            CHECK(std::fabs(w.get_additive_u_k(r, k) - expected) < 1e-4f,
                  "additive_u row=%zu k=%zu mismatch after bulk round-trip", r, k);
        }
    for (std::size_t c = 0; c < N; ++c)
        for (std::size_t k = 0; k < 5; ++k) {
            float expected = -(float(c) + float(k));
            CHECK(std::fabs(w.get_additive_v_k(c, k) - expected) < 1e-4f,
                  "additive_v col=%zu k=%zu mismatch after bulk round-trip", c, k);
        }
}

// Size-preserving: a bulk setter must reject anything that isn't
// EXACTLY the current size -- this is a correction pass over existing
// trained values, not a resize/grow path (that's set_scale_rank's job,
// and a silent length change here would corrupt the row/col-major
// layout every _k accessor depends on).
static void test_size_mismatch_throws() {
    Weights w = make_layer();
    w.set_scale_rank_max(4);
    w.set_scale_rank(4);
    w.set_value_scale_raw_k(0, 0, 1.0f); // touch it so the vector is non-empty
    bool threw = false;
    try {
        w.set_value_scale_raw_vector(std::vector<float>{1.0f, 2.0f});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "set_value_scale_raw_vector should throw on size mismatch");

    threw = false;
    try {
        w.set_output_scale_raw_vector(std::vector<float>(1000, 0.0f));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "set_output_scale_raw_vector should throw on size mismatch");

    w.set_additive_rank_max(2);
    w.set_additive_rank(2);
    w.set_additive_u_raw_k(0, 0, 1.0f);
    threw = false;
    try {
        w.set_additive_u_raw_vector(std::vector<float>{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "set_additive_u_raw_vector should throw on size mismatch (empty vs non-empty)");
}

// Exact-size (no-op content) set must NOT throw -- confirms the check
// is size-only, not some other spurious rejection.
static void test_exact_size_set_does_not_throw() {
    Weights w = make_layer();
    w.set_scale_rank_max(2);
    w.set_scale_rank(2);
    w.set_value_scale_raw_k(0, 0, 1.0f);
    auto vs = w.get_value_scale_raw_vector();
    bool threw = false;
    try {
        w.set_value_scale_raw_vector(vs);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(!threw, "set_value_scale_raw_vector should NOT throw on exact-size input");
}

int main() {
    test_empty_before_any_write();
    test_value_scale_round_trip();
    test_output_scale_round_trip();
    test_additive_u_v_round_trip();
    test_size_mismatch_throws();
    test_exact_size_set_does_not_throw();

    if (g_fail == 0) {
        std::printf("All AQRS scale overflow guard exposure tests passed.\n");
    } else {
        std::printf("%d FAILURES\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
