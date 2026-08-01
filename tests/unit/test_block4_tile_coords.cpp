// Correctness check for Block4Store::live_tile_coords() -- confirms the
// returned (br, bc) coordinates exactly match a hand-built tile layout,
// in the same row-major order row_cursor()/row_nnz() already produce
// internally. See block4.hpp and TODO_DUAL_BLOCK4.md.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); ++g_fail; } \
} while (0)

int main() {
    // 8x8 block-rows/cols (32x32 real matrix), tiles placed at a
    // deliberately non-trivial set of (br, bc) positions -- some rows
    // with zero tiles, some with one, some with several, spanning both
    // sides of the diagonal.
    const int n_in = 32, n_out = 32;
    Block4Store store;
    store.init(n_in, n_out);
    store.set_limits(1u << 20, 1u << 20);

    const std::vector<std::pair<uint32_t, uint32_t>> expected = {
        {0, 0}, {0, 3}, {0, 7},   // row 0: several tiles, spans early->late cols
        {2, 1},                  // row 2: below diagonal (row > col)
        {4, 4},                  // row 4: on diagonal
        {5, 2}, {5, 6},          // row 5: mixed sides
        {7, 0},                  // row 7: far below diagonal
        // row 1, 3, 6: deliberately empty
    };
    for (auto [br, bc] : expected) {
        auto tile = store.get_or_create(br, bc);
        tile.at(0, 0) = 0x11; // any nonzero byte -- just needs to be live
    }

    std::vector<uint32_t> got_br, got_bc;
    store.live_tile_coords(got_br, got_bc);

    CHECK(got_br.size() == expected.size(), "count mismatch: got %zu, expected %zu",
          got_br.size(), expected.size());
    CHECK(got_bc.size() == got_br.size(), "br/bc size mismatch: %zu vs %zu",
          got_br.size(), got_bc.size());

    // Row-major order is guaranteed (matches row_cursor's own ascending
    // column-within-row contract) -- compare directly, not as sets.
    for (std::size_t i = 0; i < std::min(got_br.size(), expected.size()); ++i) {
        CHECK(got_br[i] == expected[i].first && got_bc[i] == expected[i].second,
              "position %zu: got (%u,%u), expected (%u,%u)",
              i, got_br[i], got_bc[i], expected[i].first, expected[i].second);
    }

    // n_tiles() must agree with the coordinate count -- same underlying
    // block_layout, no reason for these to ever disagree.
    CHECK(store.n_tiles() == got_br.size(),
          "n_tiles()=%zu disagrees with live_tile_coords() count=%zu",
          store.n_tiles(), got_br.size());

    // Empty store: must return empty, not garbage/uninitialized data.
    {
        Block4Store empty_store;
        empty_store.init(16, 16);
        std::vector<uint32_t> eb, ec;
        empty_store.live_tile_coords(eb, ec);
        CHECK(eb.empty() && ec.empty(), "empty store returned %zu/%zu coords, expected 0/0",
              eb.size(), ec.size());
    }

    // Erasing a tile must remove it from the reported coordinates too.
    {
        store.erase(0, 3);
        std::vector<uint32_t> br2, bc2;
        store.live_tile_coords(br2, bc2);
        CHECK(br2.size() == expected.size() - 1,
              "after erase: got %zu coords, expected %zu", br2.size(), expected.size() - 1);
        bool still_present = false;
        for (std::size_t i = 0; i < br2.size(); ++i)
            if (br2[i] == 0 && bc2[i] == 3) still_present = true;
        CHECK(!still_present, "erased tile (0,3) still reported as live");
    }

    std::printf("%s (%d total failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
