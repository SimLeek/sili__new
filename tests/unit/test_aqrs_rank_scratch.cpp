// AQRS rank scratch (task #295): scale_rank/additive_rank used to be
// hard-capped at a compile-time SCALE_RANK_MAX/ADDITIVE_RANK_MAX=4
// (block4's SIMD backward path sized its per-rank-component accumulators
// as fixed stack arrays). Real user request: replace with persistent,
// per-instance HEAP scratch that grows to fit whatever rank is actually
// used, with an explicit reserve call that can also SHRINK (frees
// capacity a layer grew into early on and no longer needs) or preallocate
// ahead of need (avoid any reallocation during training). scale_rank_max/
// additive_rank_max become real runtime-settable policy caps, independent
// of the scratch memory itself.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
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

// scale_rank_max defaults to 4 (matches the old hardcoded constant's
// value) even though there's no longer a hard compile-time ceiling behind
// it -- confirms backward compat for any caller that never touches the
// new API.
static void test_default_cap_matches_old_constant() {
    Weights w = make_layer();
    CHECK(w.get_scale_rank_max() == 4, "default scale_rank_max should be 4, got %zu",
          w.get_scale_rank_max());
    CHECK(w.get_additive_rank_max() == 4, "default additive_rank_max should be 4, got %zu",
          w.get_additive_rank_max());
    bool threw = false;
    try {
        w.set_scale_rank(5);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw,
          "set_scale_rank(5) should still throw against the default cap=4 (backward compat)");
}

// The real point of task #295: rank can now genuinely exceed the OLD
// hardcoded compile-time limit (4), once the policy cap is raised, with
// no stack-array-overflow risk (that concern is gone -- scratch is heap,
// grows automatically).
static void test_rank_grows_past_old_hardcoded_limit() {
    Weights w = make_layer();
    w.set_scale_rank_max(8);
    w.set_scale_rank(8);
    CHECK(w.scale_rank == 8, "scale_rank should be 8 (past the old hardcoded 4), got %zu",
          w.scale_rank);
    for (std::size_t r = 0; r < N; ++r)
        for (std::size_t k = 0; k < 8; ++k)
            w.set_value_scale_raw_k(r, k, 1.0f);
    for (std::size_t c = 0; c < N; ++c)
        for (std::size_t k = 0; k < 8; ++k)
            w.set_output_scale_raw_k(c, k, 1.0f);

    std::vector<float> basis(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i)
        basis[i * N + i] = 1.0f;
    std::vector<float> y(N, 0.0f);
    disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(&basis[0], 1, SIZE_TYPE(N), w, y.data(), 2);
    // S(row,col) = sum_k value_scale_k*output_scale_k = 8*1*1 = 8, on the
    // diagonal (basis[0] = e_0), so y[0] should be exactly 8 (rank-8
    // scale envelope, all channels contributing 1.0 each).
    CHECK(std::fabs(y[0] - 8.0f) < 1e-4f,
          "rank-8 forward: expected y[0]=8.0 (sum of 8 unit channels), got %.6f", y[0]);

    std::vector<float> dy(N, 0.1f), dx(N, 0.0f), ni(N, 0.0f), ng(N, 0.0f);
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false>(
        &basis[0], 1, SIZE_TYPE(N), dy.data(), w, dx.data(), ni.data(), ng.data(), 0.01f, 2, false,
        true, 0.999f, 1e-8f, 0.9f, 0.0f, 1e30f);
    CHECK(w.scale_rank_scratch.cap_rank >= 8,
          "scratch should have grown to cover rank=8, cap_rank=%zu", w.scale_rank_scratch.cap_rank);
}

// reserve_scale_rank_scratch can explicitly SHRINK capacity back down
// (not just grow) -- the real memory-reclaim path a caller who
// overprovisioned early in training would use.
static void test_reserve_can_shrink() {
    Weights w = make_layer();
    w.set_scale_rank_max(8);
    w.set_scale_rank(8);
    w.reserve_scale_rank_scratch(4, 16, 4); // preallocate generously
    CHECK(w.scale_rank_scratch.cap_rank == 16,
          "expected cap_rank=16 after generous reserve, got %zu", w.scale_rank_scratch.cap_rank);
    w.reserve_scale_rank_scratch(2, 8, 4); // shrink back down to exactly what's in use
    CHECK(w.scale_rank_scratch.cap_rank == 8,
          "expected cap_rank=8 after shrinking reserve, got %zu", w.scale_rank_scratch.cap_rank);
    CHECK(w.scale_rank_scratch.cap_threads == 2,
          "expected cap_threads=2 after shrinking reserve, got %zu",
          w.scale_rank_scratch.cap_threads);
}

// reserve_scale_rank_scratch must refuse to shrink BELOW the rank
// currently actually in use -- that would corrupt live scratch data
// disldo_backward is reading/writing.
static void test_reserve_rejects_shrink_below_live_rank() {
    Weights w = make_layer();
    w.set_scale_rank_max(8);
    w.set_scale_rank(8);
    bool threw = false;
    try {
        w.reserve_scale_rank_scratch(4, 4, 4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw,
          "reserve_scale_rank_scratch(rank=4) should throw when scale_rank=8 is actually in use");
}

// Preallocating ahead of need (reserve BEFORE ever calling
// disldo_backward) means ensure() inside disldo_backward is a pure no-op
// -- the "allocate once at init, never touch the allocator again during
// training" mode the user specifically asked for as an available option.
static void test_preallocate_then_ensure_is_noop() {
    Weights w = make_layer();
    w.set_scale_rank_max(4);
    w.reserve_scale_rank_scratch(4, 4, 4);
    const std::size_t cap_before = w.scale_rank_scratch.cap_rank;
    const std::size_t threads_before = w.scale_rank_scratch.cap_threads;
    w.scale_rank_scratch.ensure(4, 4, 4); // exactly what disldo_backward calls internally
    CHECK(w.scale_rank_scratch.cap_rank == cap_before,
          "ensure() after a sufficient reserve should not change cap_rank");
    CHECK(w.scale_rank_scratch.cap_threads == threads_before,
          "ensure() after a sufficient reserve should not change cap_threads");
}

int main() {
    test_default_cap_matches_old_constant();
    test_rank_grows_past_old_hardcoded_limit();
    test_reserve_can_shrink();
    test_reserve_rejects_shrink_below_live_rank();
    test_preallocate_then_ensure_is_noop();

    if (g_fail == 0) {
        std::printf("All AQRS rank scratch tests passed.\n");
    } else {
        std::printf("%d FAILURES\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
