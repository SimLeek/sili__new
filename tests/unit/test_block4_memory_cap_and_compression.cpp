// Permanent regression guard for the two guarantees block4 (and the
// scattered CSR side it shares weights with) must never violate:
//   1. Memory never exceeds the configured budget during real
//      synaptogenesis -- so growth can never crash the process.
//   2. Compression genuinely shrinks real memory (not just flips an
//      is_sparse flag inside an already-dense-sized slot -- see
//      block4.hpp / TODO_DUAL_BLOCK4.md's "Real variable-length
//      compression" section for the bug this replaced).
//
// Phase 1 drives real online training (disldo_forward/backward) plus
// aggressive synaptogenesis under a TIGHT shared budget for many
// iterations, asserting the real allocated-byte totals (not just nnz)
// never exceed set_limits()'s configured caps on EITHER side.
//
// Phase 2 verifies real compression in isolation: backward touches every
// slot of every tile it processes (see block4.hpp), so a tile under
// ACTIVE training trends toward full occupancy regardless of switch_point
// -- compression's real value is for tiles NOT being actively re-touched
// (a realistic scenario: inference-heavy phases, partial-batch training,
// or connections that settled and stopped changing). This phase builds
// several low-occupancy tiles, compresses them, leaves them untouched,
// and checks the real byte totals shrank by a real, specific margin.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <random>
#include <algorithm>

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

static void check_within_budget(const Weights& weights, int iter) {
    const auto& dc = weights.connections;
    CHECK(dc.layout.total_alloc_bytes() <= dc.max_indices_bytes,
          "iter %d: scattered indices_buf allocation %zu exceeds budget %zu", iter,
          dc.layout.total_alloc_bytes(), dc.max_indices_bytes);
    const std::size_t values_bytes =
        ValueAccessor<FP4BiPacked>::projected_byte_size(dc.layout.total_alloc_elems());
    CHECK(values_bytes <= dc.max_values_bytes,
          "iter %d: scattered values allocation %zu exceeds budget %zu", iter, values_bytes,
          dc.max_values_bytes);
    CHECK(weights.block4.indices_buf.size() <= weights.block4.max_indices_bytes,
          "iter %d: block4 indices_buf allocation %zu exceeds budget %zu", iter,
          weights.block4.indices_buf.size(), weights.block4.max_indices_bytes);
    CHECK(weights.block4.total_tile_alloc_bytes() <= weights.block4.max_tile_bytes,
          "iter %d: block4 tile_data allocation %zu exceeds budget %zu", iter,
          weights.block4.total_tile_alloc_bytes(), weights.block4.max_tile_bytes);
}

static void test_memory_cap_never_exceeded_under_stress() {
    const int n_in = 48, n_out = 48;
    const std::size_t max_weights = 250;
    std::vector<SIZE_TYPE> empty_ptrs(n_in + 1, 0);
    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        empty_ptrs, {}, {}, {}, n_in, n_out, max_weights * 8 + 4096, max_weights + 64);
    weights.connections.set_limits(
        max_weights * 8 + 4096, ValueAccessor<FP4BiPacked>::projected_byte_size(max_weights + 64));
    weights.block4.init(n_in, n_out);
    weights.block4.set_limits(max_weights * 8 + 4096,
                              std::max<std::size_t>(4, max_weights / BLOCK4_TILE_SLOTS) *
                                  BLOCK4_TILE_SLOTS);
    weights.block4.switch_point = 3; // realistic compression setting, not the disabling extreme
    weights.out_degree.assign(n_out, SIZE_TYPE(0));
    weights.recompute_stats();

    fp4_seed_stochastic_rng(0);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> data_dist(-1.0f, 1.0f);

    std::vector<float> x(n_in), dy(n_out), nia(n_in, 0.0f), nga(n_out, 0.0f);
    std::size_t row = 0;
    bool hit_fatal_cap = false;
    int iters_completed = 0;

    for (int i = 0; i < 1500 && !hit_fatal_cap; ++i) {
        for (auto& v : nia)
            v = data_dist(rng);
        for (auto& v : nga)
            v = data_dist(rng);
        delta_csr_build_probes<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights, nia.data(), nga.data(),
                                                                 SIZE_TYPE(10), true);

        for (int attempt = 0; attempt < 5; ++attempt) {
            try {
                delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights, row, 0.0f,
                                                                           SIZE_TYPE(12));
                break;
            } catch (const std::bad_alloc&) {
                // The real guarantee under test: a hit here means the
                // configured budget was genuinely exhausted, not a crash
                // from an unenforced cap -- expected under a tight budget
                // with unbounded growth pressure, not a failure by itself.
                hit_fatal_cap = true;
                break;
            } catch (const std::runtime_error&) {
                try {
                    delta_csr_equalize_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights.connections,
                                                                              row);
                } catch (const std::bad_alloc&) {
                    hit_fatal_cap = true;
                    break;
                }
            }
        }
        row = (row + 1) % std::size_t(n_in);

        for (auto& v : x)
            v = data_dist(rng);
        for (auto& v : dy)
            v = data_dist(rng);
        std::vector<float> out(n_out, 0.0f), dx(n_in, 0.0f);
        disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(x.data(), 1, n_in, weights, out.data(), 2);
        disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(x.data(), 1, n_in, dy.data(), weights,
                                                          dx.data(), nia.data(), nga.data(), 0.05f,
                                                          2, false, true);

        check_within_budget(weights, i);
        ++iters_completed;
    }

    CHECK(iters_completed > 50,
          "stress loop should run for a meaningful number of iterations before any real cap is "
          "hit, got %d",
          iters_completed);
    std::printf("test_memory_cap_never_exceeded_under_stress: %d iterations, hit_fatal_cap=%s, "
                "final scattered nnz=%zu, block4 tiles=%zu, block4 live synapses=%zu\n",
                iters_completed, hit_fatal_cap ? "true" : "false", weights.connections.nnz(),
                weights.block4.n_tiles(), weights.block4.live_synapses());
}

static void test_real_compression_shrinks_measured_memory() {
    const int n_in = 64, n_out = 64;
    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        std::vector<SIZE_TYPE>(n_in + 1, 0), {}, {}, {}, n_in, n_out, 1u << 20, 1u << 20);
    weights.block4.init(n_in, n_out);
    weights.block4.switch_point = 2; // "many input majority gate" style tiles from the original ask

    // Build 40 tiles, each with exactly 2 live synapses (a realistic
    // low-occupancy case per the original ask: "24 bits for 2 parameters
    // per block, a common case for very sparse matrices"), spread across
    // distinct block-rows/cols so none of them collide or get merged.
    const int n_tiles_built = 40;
    for (int t = 0; t < n_tiles_built; ++t) {
        const uint32_t br = uint32_t(t % (n_in / BLOCK4_TILE));
        const uint32_t bc = uint32_t(t / (n_in / BLOCK4_TILE)) % (n_out / BLOCK4_TILE);
        auto h = weights.block4.get_or_create(br, bc);
        h.at(0, 0) = uint8_t(0x10 + t);
        h.at(1, 2) = uint8_t(0x20 + t);
    }
    // Compress every tile just built -- explicit, deliberate, matching
    // this codebase's "never automatic on every write" compression policy.
    for (int t = 0; t < n_tiles_built; ++t) {
        const uint32_t br = uint32_t(t % (n_in / BLOCK4_TILE));
        const uint32_t bc = uint32_t(t / (n_in / BLOCK4_TILE)) % (n_out / BLOCK4_TILE);
        weights.block4.maybe_compress(br, bc);
    }

    const std::size_t n_tiles = weights.block4.n_tiles();
    CHECK(n_tiles == std::size_t(n_tiles_built), "expected %d distinct tiles, got %zu",
          n_tiles_built, n_tiles);

    std::size_t sparse_count = 0;
    for (int t = 0; t < n_tiles_built; ++t) {
        const uint32_t br = uint32_t(t % (n_in / BLOCK4_TILE));
        const uint32_t bc = uint32_t(t / (n_in / BLOCK4_TILE)) % (n_out / BLOCK4_TILE);
        if (weights.block4.is_sparse(br, bc))
            ++sparse_count;
    }
    CHECK(sparse_count == std::size_t(n_tiles_built),
          "every 2-live tile should have compressed (switch_point=2), got %zu/%d sparse",
          sparse_count, n_tiles_built);

    const std::size_t used_bytes = weights.block4.total_tile_used_bytes();
    const std::size_t dense_equivalent_bytes = n_tiles * BLOCK4_TILE_SLOTS;
    const std::size_t expect_used = n_tiles * block4_sparse_packed_len(2); // 1+1+2 = 4 bytes/tile

    CHECK(used_bytes == expect_used,
          "compressed footprint should be exactly %zu bytes (%zu tiles * 4 bytes/tile), got %zu",
          expect_used, n_tiles, used_bytes);
    CHECK(used_bytes < dense_equivalent_bytes,
          "real compression must use fewer bytes than the dense-equivalent footprint (%zu vs %zu)",
          used_bytes, dense_equivalent_bytes);

    const double compression_ratio = double(dense_equivalent_bytes) / double(used_bytes);
    CHECK(compression_ratio >= 3.9,
          "2-live tiles (4 bytes) vs dense (16 bytes) should compress ~4x, got %.2fx",
          compression_ratio);

    // Values must survive -- compression is lossless.
    for (int t = 0; t < n_tiles_built; ++t) {
        const uint32_t br = uint32_t(t % (n_in / BLOCK4_TILE));
        const uint32_t bc = uint32_t(t / (n_in / BLOCK4_TILE)) % (n_out / BLOCK4_TILE);
        auto h = weights.block4.find(br, bc);
        CHECK(h.at(0, 0) == uint8_t(0x10 + t) && h.at(1, 2) == uint8_t(0x20 + t),
              "tile %d: values must survive real compression", t);
    }

    std::printf("test_real_compression_shrinks_measured_memory: %zu tiles, %zu bytes used "
                "(vs %zu dense-equivalent) -- %.2fx real compression\n",
                n_tiles, used_bytes, dense_equivalent_bytes, compression_ratio);
}

static void test_backward_declines_growth_gracefully_under_budget_exhaustion() {
    // Per direction: a resize DECLINED under real budget exhaustion must
    // not throw at all (unlike get_or_create's own structural-growth
    // path, which still throws std::bad_alloc as before -- see
    // ~Block4TileHandle()'s comment) -- the write is silently dropped
    // (old stored bytes kept exactly as they were) and
    // Block4Store::dropped_growth_events is incremented instead, so
    // training/inference keeps running and a caller doing its own
    // memory management can still detect this happened.
    //
    // Row headroom is now genuinely compression-aware (see block4.hpp:
    // new tiles start empty and grow via exact-shortfall-only resizing,
    // no dense-worst-case over-provisioning), so a row ends up at (or
    // very near) zero headroom naturally from ordinary use -- unlike an
    // earlier version of this test, no manual internal-state
    // construction is needed to reach this scenario; just cap the
    // budget at the current real allocation so ANY further growth is
    // genuinely impossible.
    Block4Store store;
    store.init(16, 16);
    store.switch_point = 2;
    {
        auto h = store.get_or_create(0, 0);
        h.at(0, 0) = 0x11;
        h.at(0, 1) = 0x22; // 2 live
    }
    CHECK(store.is_sparse(0, 0), "setup: tile should size itself sparse immediately");
    const std::size_t before_bytes = store.total_tile_used_bytes();
    CHECK(before_bytes == block4_sparse_packed_len(2),
          "sanity: 2-live tile should be exactly %zu bytes, got %zu", block4_sparse_packed_len(2),
          before_bytes);

    store.set_limits(std::numeric_limits<std::size_t>::max(), store.total_tile_alloc_bytes());

    bool threw = false;
    try {
        auto h = store.find(0, 0);
        h.at(0, 2) = 0x33; // 3 live > switch_point=2 -- destructor must try to grow to dense
    } catch (...) {
        threw = true;
    }
    CHECK(!threw, "declining a resize under budget exhaustion must not throw");
    CHECK(store.dropped_growth_events > 0,
          "declining the resize should have incremented dropped_growth_events, got %llu",
          static_cast<unsigned long long>(store.dropped_growth_events));

    // The write was dropped, not partially applied: this tile must
    // still read back its OLD stored bytes exactly -- slot (0,2) was
    // never actually persisted, even though the .at() assignment into
    // the handle's in-memory scratch buffer itself succeeded.
    {
        auto h = store.find(0, 0);
        CHECK(h.at(0, 0) == 0x11 && h.at(0, 1) == 0x22,
              "original values must survive a declined-growth destructor");
        CHECK(h.at(0, 2) == 0, "the declined write itself must not have been persisted");
    }
    std::printf("test_backward_declines_growth_gracefully_under_budget_exhaustion: "
                "dropped_growth_events=%llu, no exception thrown\n",
                static_cast<unsigned long long>(store.dropped_growth_events));
}

int main() {
    test_memory_cap_never_exceeded_under_stress();
    test_real_compression_shrinks_measured_memory();
    test_backward_declines_growth_gracefully_under_budget_exhaustion();

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}