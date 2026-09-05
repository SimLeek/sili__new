// Speed cost of real block4 compression's pack/unpack/resize machinery.
// Correctness is covered separately by tests/unit/test_block4_memory_cap_and_compression.cpp
// (permanent ctest gate); this is the informational timing companion.
// Why it drives pack/unpack directly instead of a training loop, and the
// 1.00x-under-training finding that motivated that design: see
// docs/research/bench_block4_memory_and_compression.rst.
//
// Usage:
//   g++ -std=c++20 -O3 -ffast-math -march=native -fopenmp \
//     -I <repo_root> scripts/bench_block4_memory_and_compression.cpp \
//     -o bench_b4_mem
//   ./bench_b4_mem compressed
//   ./bench_b4_mem uncompressed
//
// NOT wired into ctest -- a timing report, not a pass/fail correctness
// check (matches bench_block4_vs_dense_fp4.cpp's own convention).
#include "../sili/lib/headers/delta_csr_types.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    const bool compressed = !(argc > 1 && std::strcmp(argv[1], "uncompressed") == 0);
    const int n_in = 256, n_out = 256, n_tiles = 2000, reps = 500;

    Block4Store store;
    store.init(n_in, n_out);
    store.switch_point = compressed ? 2 : 0;

    const int tiles_per_row = n_out / BLOCK4_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        const uint32_t br = uint32_t(t / tiles_per_row) % (n_in / BLOCK4_TILE);
        const uint32_t bc = uint32_t(t % tiles_per_row);
        auto h = store.get_or_create(br, bc);
        h.at(0, 0) = 0x11;
        h.at(0, 1) = 0x22; // 2 live -- exactly at the sparse threshold
    }
    if (compressed) {
        for (int t = 0; t < n_tiles; ++t) {
            const uint32_t br = uint32_t(t / tiles_per_row) % (n_in / BLOCK4_TILE);
            const uint32_t bc = uint32_t(t % tiles_per_row);
            store.maybe_compress(br, bc);
        }
    }
    const std::size_t start_used_bytes = store.total_tile_used_bytes();

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r) {
        for (int t = 0; t < n_tiles; ++t) {
            const uint32_t br = uint32_t(t / tiles_per_row) % (n_in / BLOCK4_TILE);
            const uint32_t bc = uint32_t(t % tiles_per_row);
            {
                auto h = store.find(br, bc);
                h.at(0, 2) = 0x33; // 2 -> 3 live: crosses switch_point=2 -- forces a
            } // real sparse->dense resize in the handle destructor when compressed
            {
                auto h = store.find(br, bc);
                h.at(0, 2) = 0x00; // 3 -> 2 live again
            } // dense never auto-compresses (see block4.hpp) -- explicit below
            store.maybe_compress(br, bc); // forces a real dense->sparse resize when compressed
        }
    }
    const double total_s =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    const std::size_t end_used_bytes = store.total_tile_used_bytes();
    const std::size_t dense_equiv_bytes = std::size_t(n_tiles) * BLOCK4_TILE_SLOTS;

    std::printf("mode=%s\n",
                compressed ? "compressed (switch_point=2)" : "uncompressed (switch_point=0)");
    std::printf("%d tiles x %d up/down/compress cycles: %.4f s total, %.2f ns/cycle\n", n_tiles,
                reps, total_s, total_s * 1e9 / (double(n_tiles) * reps));
    std::printf("tile bytes used: start=%zu end=%zu (dense-equivalent=%zu, %.2fx%s)\n",
                start_used_bytes, end_used_bytes, dense_equiv_bytes,
                end_used_bytes > 0 ? double(dense_equiv_bytes) / double(end_used_bytes) : 1.0,
                compressed ? "" : " -- expected ~1.00x, compression disabled");
    return 0;
}
