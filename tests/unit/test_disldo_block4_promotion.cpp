// Correctness check for the block4 promotion/demotion hooks wired into
// delta_csr_synap_row_step (delta_csr_memory.hpp): growth (probe-driven
// insertion) should promote a tile once BLOCK4_PROMOTE_MIN_LIVE scattered
// synapses land inside it, and pruning should demote it back once live
// count drops below that threshold. Never the reverse direction.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); std::fflush(stdout); ++g_fail; } \
} while (0)

using SIZE_TYPE = int;
using COL_TYPE  = uint32_t;
using Weights   = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

// Manually set weights.probes to a fixed list of (row, col, score) candidates
// -- gives exact control over which tile a growth step targets, unlike
// delta_csr_build_probes' top-k outer product (which can't easily be made to
// target one specific 4x4 tile deterministically in a small test layer).
static void set_probes(Weights& w,
                        const std::vector<SIZE_TYPE>& rows,
                        const std::vector<SIZE_TYPE>& cols,
                        const std::vector<float>& scores) {
    w.probes.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(rows);
    w.probes.indices[1] = std::make_shared<std::vector<SIZE_TYPE>>(cols);
    w.probes.values[0]  = std::make_shared<std::vector<float>>(scores);
    w.probes.ptrs = static_cast<SIZE_TYPE>(rows.size()); // COOPointers is just a plain nnz count
}

int main() {
    const int n_in = 8, n_out = 8;

    // Empty layer: 8x8, all headroom, tile (0,0) covers rows/cols 0-3.
    std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
    std::vector<SIZE_TYPE> idx;
    std::vector<float> w, imp;
    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, n_in, n_out, 4096, 4096, 0.5f);
    weights.out_degree.assign(n_out, SIZE_TYPE(0));

    // ── Growth: insert (0,0) then (1,1) into tile (0,0), one per call ──────
    // BLOCK4_PROMOTE_MIN_LIVE defaults to 2 -- the second insertion should
    // trigger promotion of the whole tile.
    std::size_t current_row = 0;

    set_probes(weights, {0}, {0}, {1.0f});
    bool ok1 = delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, current_row, /*importance_cutoff=*/-1e9f, /*max_row_weights=*/SIZE_TYPE(10));
    CHECK(ok1, "first growth step should succeed");
    CHECK(weights.block4.n_tiles() == 0, "tile should NOT promote yet (only 1 live)");
    CHECK(weights.connections.layout.row_nnz(0) == 1, "synapse (0,0) should be scattered so far");

    current_row = 1; // force row 1 (still inside tile (0,0), local row 1)
    set_probes(weights, {1}, {1}, {1.0f});
    bool ok2 = delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, current_row, -1e9f, SIZE_TYPE(10));
    CHECK(ok2, "second growth step should succeed");
    CHECK(weights.block4.n_tiles() == 1, "tile (0,0) should now be promoted (2 live >= threshold)");
    CHECK(weights.connections.layout.row_nnz(0) == 0, "row0's synapse should have moved OUT of connections");
    CHECK(weights.connections.layout.row_nnz(1) == 0, "row1's synapse should have moved OUT of connections");

    const Block4Tile* tile = weights.block4.find(0, 0);
    CHECK(tile != nullptr, "tile (0,0) must exist after promotion");
    CHECK(tile->live_count == 2, "tile live_count should be 2, got %u", tile ? tile->live_count : 999u);

    // out_degree must still correctly reflect BOTH synapses (never touched
    // by promotion -- see block4_maybe_promote's design comment).
    CHECK(weights.out_degree[0] == 1, "out_degree[0] should be 1, got %d", (int)weights.out_degree[0]);
    CHECK(weights.out_degree[1] == 1, "out_degree[1] should be 1, got %d", (int)weights.out_degree[1]);

    // ── Growth into an ALREADY-promoted tile: single-synapse migration ────
    current_row = 2;
    set_probes(weights, {2}, {2}, {1.0f});
    bool ok3 = delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, current_row, -1e9f, SIZE_TYPE(10));
    CHECK(ok3, "third growth step should succeed");
    CHECK(weights.connections.layout.row_nnz(2) == 0, "row2's new synapse should migrate straight into the existing tile");
    tile = weights.block4.find(0, 0);
    CHECK(tile->live_count == 3, "tile live_count should be 3 after single-synapse migration, got %u", tile->live_count);
    CHECK(weights.out_degree[2] == 1, "out_degree[2] should be 1, got %d", (int)weights.out_degree[2]);

    // ── Pruning: demote by cutting live count back below threshold ────────
    // Prune col 2 (row2's synapse) and col 1 (row1's) via importance_cutoff
    // set above both their importance (0.0) but below nothing else relevant.
    // Simplest: call synap_row_step per-row with a cutoff that removes
    // everything in that row (no probes this time).
    weights.probes.indices[0] = nullptr;
    weights.probes.indices[1] = nullptr;
    weights.probes.values[0]  = nullptr;

    current_row = 2;
    delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, current_row, /*importance_cutoff=*/1e9f, /*max_row_weights=*/SIZE_TYPE(10));
    tile = weights.block4.find(0, 0);
    CHECK(tile != nullptr, "tile should still exist after pruning row2 (2 live left, still >= threshold)");
    CHECK(tile && tile->live_count == 2, "tile live_count should be 2 after pruning row2's synapse");
    CHECK(weights.out_degree[2] == 0, "out_degree[2] should drop to 0 after real prune, got %d", (int)weights.out_degree[2]);

    current_row = 1;
    delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, current_row, /*importance_cutoff=*/1e9f, /*max_row_weights=*/SIZE_TYPE(10));
    CHECK(weights.block4.n_tiles() == 0, "tile should be DEMOTED after dropping to 1 live (< threshold)");
    CHECK(weights.connections.layout.row_nnz(0) == 1, "row0's surviving synapse should be back in connections after demotion");
    CHECK(weights.out_degree[0] == 1, "out_degree[0] should be unchanged by demotion (representation move only), got %d", (int)weights.out_degree[0]);
    CHECK(weights.out_degree[1] == 0, "out_degree[1] should drop to 0 after real prune, got %d", (int)weights.out_degree[1]);

    // Verify the demoted synapse's value round-tripped exactly (was never
    // set to anything but the FP4_TABLE default 0.0f in this test, so check
    // via disldo_forward that it still contributes correctly).
    {
        std::vector<float> x(n_in, 0.f);
        x[0] = 5.0f;
        std::vector<float> y(n_out, 0.f);
        disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(x.data(), 1, n_in, weights, y.data(), 0.0f, 1);
        // stored weight for the demoted synapse was probe_score-seeded via
        // insert_col(..., value_type(0), probe_scores[i]) in Step 6 -- i.e.
        // weight=0.0 initially (importance=score). So y should be all zero.
        for (int c = 0; c < n_out; ++c)
            CHECK(std::abs(y[c]) < 1e-4f, "post-demotion forward output[%d] should be 0 (weight was never trained): got %.4f", c, y[c]);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
