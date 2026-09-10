// Experiment (NOT a change to disldo_forward itself): does pairing two
// NON-ADJACENT block4 tiles -- tiles sharing the same block-row `br` but
// at DIFFERENT, possibly far-apart block-columns `bc` (a "checkered"
// sparsity pattern, i.e. real gaps between occupied tile-columns) -- into
// one 8-wide AVX2 op still pay off, the same way pairing two ADJACENT
// columns WITHIN one tile already does (disldo_forward.
// fp32_block4_avx2_column_pairing)?
//
// Investigated first (not assumed): forward's tile_br/tile_bc flat arrays
// are already collected in br-major, ascending-bc order before the
// parallel region, and forward's parallelism is per-flat-tile-index (not
// per-br), so two tiles sharing a br are ALREADY adjacent entries in that
// flat list -- no new column index needed, and grabbing both tiles'
// at_index() handles is free with the existing API. This file tests
// whether doing so is actually FASTER, as a standalone experiment before
// touching the real disldo_forward kernel (same "measure before
// committing" discipline as every other AVX2-widening decision in this
// codebase).
//
// Checkered layer: block4_load_dense_fp32's own construction loop
// (get_or_create + set_weight/set_importance per tile, proven-correct
// scale defaults), but skipping every ODD block-column -- so occupied
// tiles within a br are genuinely non-adjacent (bc=0,2,4,...), with real
// gaps, not synthetic re-indexing on fully dense data.
//
// Two hand-rolled compute functions over the SAME checkered layer,
// sharing all other overhead (tile lookup, scale fetch) so only the
// pairing strategy differs:
//   - adjacent_only: today's real per-tile design (2 columns from the
//     SAME tile combined per Block8Vec op) applied once per occupied
//     tile.
//   - cross_tile: pairs the SAME column position (lj) from TWO DIFFERENT
//     tiles (consecutive entries in the br's occupied-tile list, however
//     far apart their bc's are) into one Block8Vec op instead.
// Both are checked against real disldo_forward on the identical checkered
// weights as the correctness oracle.
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

// Builds a checkered (every-other-block-column occupied) block4 fp32
// layer, mirroring block4_load_dense_fp32's own construction pattern
// exactly except for skipping odd bc's.
static void build_checkered(Weights& weights, const std::vector<float>& weight_values,
                            const std::vector<float>& importance_values, std::size_t n_in,
                            std::size_t n_out) {
    const uint32_t block_rows = uint32_t((n_in + BLOCK4_TILE - 1) / BLOCK4_TILE);
    const uint32_t block_cols = uint32_t((n_out + BLOCK4_TILE - 1) / BLOCK4_TILE);
    weights.block4.init(n_in, n_out);
    const std::size_t idx_budget = std::size_t(block_rows) * block_cols * 16;
    const std::size_t tile_budget =
        std::size_t(block_rows) * block_cols * BLOCK4_TILE_SLOTS32_BYTES;
    weights.block4.set_limits(idx_budget, tile_budget);
    for (uint32_t br = 0; br < block_rows; ++br) {
        const std::size_t row_lo = std::size_t(br) * BLOCK4_TILE;
        const std::size_t row_hi = std::min(row_lo + BLOCK4_TILE, n_in);
        for (uint32_t bc = 0; bc < block_cols; ++bc) {
            if ((bc % 2) != 0) // checkered: only even block-columns occupied
                continue;
            const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
            const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, n_out);
            auto tile = weights.block4.get_or_create(br, bc);
            for (std::size_t row = row_lo; row < row_hi; ++row) {
                const uint32_t li = uint32_t(row - row_lo);
                for (std::size_t col = col_lo; col < col_hi; ++col) {
                    const uint32_t lj = uint32_t(col - col_lo);
                    const std::size_t idx = row * n_out + col;
                    tile.set_weight(li, lj, weight_values[idx]);
                    tile.set_importance(li, lj, importance_values[idx]);
                }
            }
        }
    }
}

// Collects (br,bc) pairs of tiles that are BOTH present and share `br`,
// walking each row's occupied-tile list and pairing consecutive entries
// -- exactly what forward's real tile_br/tile_bc flat-array walk would
// see. Odd tile out (last tile in a br with no partner) goes to `solo`.
// Tile coordinates recorded during ONE sequential cursor walk (O(1) per
// tile, elem_pos/byte_pos tracked incrementally) -- exactly like real
// disldo_forward's own tile-collection loop and forward_adjacent_only
// below. NOT looked up again afterward via raw_find (an O(row_nnz)
// linear scan per call) -- that redundant re-scan was a real confound in
// an earlier version of this experiment, not a genuine property of
// cross-tile pairing: it made cross_tile pay for 2 full linear scans per
// pair on top of the walk that already found both tiles, while
// adjacent_only paid nothing extra. Fixed by recording elem_pos/byte_pos
// once, during the same single walk that finds the pairing itself.
struct TileCoord {
    uint32_t br, bc;
    std::size_t elem_pos, byte_pos;
};
struct TilePair {
    TileCoord a, b;
};
static void collect_pairs(const Weights& weights, std::size_t n_in, std::vector<TilePair>& pairs,
                          std::vector<TileCoord>& solos) {
    const auto& BL4 = weights.block4.block_layout;
    for (std::size_t br = 0; br < BL4.rows; ++br) {
        const std::size_t n_bc = BL4.row_nnz(br);
        if (n_bc == 0)
            continue;
        auto cur = weights.block4.row_cursor(br);
        std::size_t elem_pos = BL4.elem_start[br];
        std::size_t byte_pos = weights.block4.tile_byte_start[br];
        std::vector<TileCoord> coords(n_bc);
        for (std::size_t i = 0; i < n_bc; ++i, ++elem_pos) {
            const uint32_t bc = cur.advance();
            coords[i] = {uint32_t(br), bc, elem_pos, byte_pos};
            byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);
        }
        std::size_t i = 0;
        for (; i + 1 < n_bc; i += 2)
            pairs.push_back({coords[i], coords[i + 1]});
        if (i < n_bc)
            solos.push_back(coords[i]);
    }
}

// today's real design: 2 columns WITHIN one tile combined per Block8Vec
// op (mirrors disldo_forward's process_pair exactly), applied to every
// occupied tile independently.
static void forward_adjacent_only(const Weights& weights, const float* input, SIZE_TYPE batch,
                                  std::size_t in_cols, std::size_t n_in, std::size_t n_out,
                                  float* output) {
    std::fill(output, output + std::size_t(batch) * n_out, 0.0f);
    const auto& BL4 = weights.block4.block_layout;
    for (std::size_t br = 0; br < BL4.rows; ++br) {
        const std::size_t n_bc = BL4.row_nnz(br);
        if (n_bc == 0)
            continue;
        auto cur = weights.block4.row_cursor(br);
        std::size_t elem_pos = BL4.elem_start[br];
        std::size_t byte_pos = weights.block4.tile_byte_start[br];
        for (std::size_t bk = 0; bk < n_bc; ++bk, ++elem_pos) {
            const uint32_t bc = cur.advance();
            const auto tile = weights.block4.at_index(uint32_t(br), bc, elem_pos, byte_pos);
            const uint8_t* tdata = tile.raw_data();
            byte_pos += weights.block4.tile_len_at(elem_pos, byte_pos);

            auto process_pair = [&]<uint32_t LJ0>() {
                constexpr uint32_t LJ1 = LJ0 + 1;
                const std::size_t col0 = std::size_t(bc) * BLOCK4_TILE + LJ0;
                const std::size_t col1 = std::size_t(bc) * BLOCK4_TILE + LJ1;
                const bool have0 = col0 < n_out;
                const bool have1 = col1 < n_out;
                if (!have0)
                    return;
                Block8Vec w_decoded8;
                std::memcpy(&w_decoded8, tdata + sizeof(float) * Block4Tile32::slot_index(0, LJ0),
                            sizeof(w_decoded8));
                std::size_t row_idx[BLOCK4_TILE];
                Block8Vec s8;
                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                    const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                    row_idx[li] = row < n_in ? row : 0;
                    s8[li] = row < n_in ? weights.get_scale(row, col0) : 0.0f;
                    s8[li + 4] = (row < n_in && have1) ? weights.get_scale(row, col1) : 0.0f;
                }
                const Block8Vec w8 = w_decoded8 * s8;
                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const float* in_row = input + static_cast<std::size_t>(b) * in_cols;
                    Block8Vec in8;
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const float iv = in_row[row_idx[li]];
                        in8[li] = iv;
                        in8[li + 4] = iv;
                    }
                    const Block8Vec prod = w8 * in8;
                    output[static_cast<std::size_t>(b) * n_out + col0] +=
                        prod[0] + prod[1] + prod[2] + prod[3];
                    if (have1)
                        output[static_cast<std::size_t>(b) * n_out + col1] +=
                            prod[4] + prod[5] + prod[6] + prod[7];
                }
            };
            process_pair.template operator()<0>();
            process_pair.template operator()<2>();
        }
    }
}

// experimental design: pairs the SAME column position (lj) from TWO
// DIFFERENT, non-adjacent tiles (sharing br) into one Block8Vec op.
static void forward_cross_tile(const Weights& weights, const float* input, SIZE_TYPE batch,
                               std::size_t in_cols, std::size_t n_in, std::size_t n_out,
                               float* output) {
    std::fill(output, output + std::size_t(batch) * n_out, 0.0f);
    std::vector<TilePair> pairs;
    std::vector<TileCoord> solos;
    collect_pairs(weights, n_in, pairs, solos);

    for (const auto& tp : pairs) {
        const uint32_t br = tp.a.br;
        const auto tileA = weights.block4.at_index(tp.a.br, tp.a.bc, tp.a.elem_pos, tp.a.byte_pos);
        const auto tileB = weights.block4.at_index(tp.b.br, tp.b.bc, tp.b.elem_pos, tp.b.byte_pos);
        const uint8_t* tdataA = tileA.raw_data();
        const uint8_t* tdataB = tileB.raw_data();

        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            const std::size_t colA = std::size_t(tp.a.bc) * BLOCK4_TILE + lj;
            const std::size_t colB = std::size_t(tp.b.bc) * BLOCK4_TILE + lj;
            const bool haveA = colA < n_out;
            const bool haveB = colB < n_out;
            if (!haveA)
                continue;
            Block4Vec wA, wB = block4_vec_broadcast(0.0f);
            std::memcpy(&wA, tdataA + sizeof(float) * Block4Tile32::slot_index(0, lj), sizeof(wA));
            if (haveB)
                std::memcpy(&wB, tdataB + sizeof(float) * Block4Tile32::slot_index(0, lj),
                            sizeof(wB));
            const Block8Vec w_decoded8 = block8_vec_from_lo_hi(wA, wB);

            std::size_t row_idx[BLOCK4_TILE];
            Block8Vec s8;
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                row_idx[li] = row < n_in ? row : 0;
                s8[li] = row < n_in ? weights.get_scale(row, colA) : 0.0f;
                s8[li + 4] = (row < n_in && haveB) ? weights.get_scale(row, colB) : 0.0f;
            }
            const Block8Vec w8 = w_decoded8 * s8;
            for (SIZE_TYPE b = 0; b < batch; ++b) {
                const float* in_row = input + static_cast<std::size_t>(b) * in_cols;
                Block8Vec in8;
                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                    const float iv = in_row[row_idx[li]];
                    in8[li] = iv;
                    in8[li + 4] = iv;
                }
                const Block8Vec prod = w8 * in8;
                output[static_cast<std::size_t>(b) * n_out + colA] +=
                    prod[0] + prod[1] + prod[2] + prod[3];
                if (haveB)
                    output[static_cast<std::size_t>(b) * n_out + colB] +=
                        prod[4] + prod[5] + prod[6] + prod[7];
            }
        }
    }

    // solos (no partner tile in this br): fall back to the existing
    // within-tile adjacent pairing, unchanged.
    for (const auto& s : solos) {
        const uint32_t br = s.br, bc = s.bc;
        const auto tile = weights.block4.at_index(s.br, s.bc, s.elem_pos, s.byte_pos);
        const uint8_t* tdata = tile.raw_data();
        auto process_pair = [&]<uint32_t LJ0>() {
            constexpr uint32_t LJ1 = LJ0 + 1;
            const std::size_t col0 = std::size_t(bc) * BLOCK4_TILE + LJ0;
            const std::size_t col1 = std::size_t(bc) * BLOCK4_TILE + LJ1;
            const bool have0 = col0 < n_out;
            const bool have1 = col1 < n_out;
            if (!have0)
                return;
            Block8Vec w_decoded8;
            std::memcpy(&w_decoded8, tdata + sizeof(float) * Block4Tile32::slot_index(0, LJ0),
                        sizeof(w_decoded8));
            std::size_t row_idx[BLOCK4_TILE];
            Block8Vec s8;
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
                row_idx[li] = row < n_in ? row : 0;
                s8[li] = row < n_in ? weights.get_scale(row, col0) : 0.0f;
                s8[li + 4] = (row < n_in && have1) ? weights.get_scale(row, col1) : 0.0f;
            }
            const Block8Vec w8 = w_decoded8 * s8;
            for (SIZE_TYPE b = 0; b < batch; ++b) {
                const float* in_row = input + static_cast<std::size_t>(b) * in_cols;
                Block8Vec in8;
                for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                    const float iv = in_row[row_idx[li]];
                    in8[li] = iv;
                    in8[li + 4] = iv;
                }
                const Block8Vec prod = w8 * in8;
                output[static_cast<std::size_t>(b) * n_out + col0] +=
                    prod[0] + prod[1] + prod[2] + prod[3];
                if (have1)
                    output[static_cast<std::size_t>(b) * n_out + col1] +=
                        prod[4] + prod[5] + prod[6] + prod[7];
            }
        };
        process_pair.template operator()<0>();
        process_pair.template operator()<2>();
    }
}

int main() {
    std::mt19937 rng(112233);
    std::uniform_real_distribution<float> wdist(1.0f, 4.0f);
    std::uniform_real_distribution<float> idist(0.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    // ---- correctness: small scale ----
    {
        const std::size_t n_in = 32, n_out = 32;
        const SIZE_TYPE batch = 4;
        std::vector<float> w(n_in * n_out), imp(n_in * n_out);
        for (auto& v : w)
            v = wdist(rng) * (rng() % 2 ? 1.0f : -1.0f);
        for (auto& v : imp)
            v = idist(rng);

        Weights weights;
        weights.connections.layout.rows = n_in;
        weights.connections.layout.cols = n_out;
        build_checkered(weights, w, imp, n_in, n_out);

        std::vector<float> x(batch * n_in);
        for (auto& v : x)
            v = xdist(rng);

        std::vector<float> y_oracle(batch * n_out, 0.f), y_adj(batch * n_out, 0.f),
            y_cross(batch * n_out, 0.f);
        disldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
            x.data(), batch, SIZE_TYPE(n_in), weights, y_oracle.data(), 4);
        forward_adjacent_only(weights, x.data(), batch, n_in, n_in, n_out, y_adj.data());
        forward_cross_tile(weights, x.data(), batch, n_in, n_in, n_out, y_cross.data());

        double err_adj = 0.0, err_cross = 0.0;
        for (std::size_t i = 0; i < y_oracle.size(); ++i) {
            err_adj = std::max(err_adj, double(std::fabs(y_adj[i] - y_oracle[i])));
            err_cross = std::max(err_cross, double(std::fabs(y_cross[i] - y_oracle[i])));
        }
        CHECK(err_adj < 1e-3, "adjacent_only vs disldo_forward oracle max abs err %.6f too large",
              err_adj);
        CHECK(err_cross < 1e-3, "cross_tile vs disldo_forward oracle max abs err %.6f too large",
              err_cross);
        std::printf("correctness: adjacent_only err=%.6f cross_tile err=%.6f\n", err_adj,
                    err_cross);
    }

    // ---- timing: large scale, checkered ----
    {
        const std::size_t n_in = 256, n_out = 256;
        const SIZE_TYPE batch = 8;
        std::vector<float> w(n_in * n_out), imp(n_in * n_out);
        for (auto& v : w)
            v = wdist(rng) * (rng() % 2 ? 1.0f : -1.0f);
        for (auto& v : imp)
            v = idist(rng);

        Weights weights;
        weights.connections.layout.rows = n_in;
        weights.connections.layout.cols = n_out;
        build_checkered(weights, w, imp, n_in, n_out);
        std::vector<TilePair> pairs;
        std::vector<TileCoord> solos;
        collect_pairs(weights, n_in, pairs, solos);
        std::printf("checkered layer: %zu tile-pairs, %zu solo tiles\n", pairs.size(),
                    solos.size());

        const int n_instances = 16;
        std::vector<float> x_all(std::size_t(n_instances) * batch * n_in);
        for (auto& v : x_all)
            v = xdist(rng);
        const int n_reps = 200;
        std::vector<int> order(n_reps);
        for (int i = 0; i < n_reps; ++i)
            order[i] = i % n_instances;
        std::shuffle(order.begin(), order.end(), rng);

        std::vector<float> y(batch * n_out);
        for (int i = 0; i < 20; ++i)
            forward_adjacent_only(weights, x_all.data() + std::size_t(order[i]) * batch * n_in,
                                  batch, n_in, n_in, n_out, y.data());
        std::vector<double> ns_adj(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            forward_adjacent_only(weights, x_all.data() + std::size_t(order[i]) * batch * n_in,
                                  batch, n_in, n_in, n_out, y.data());
            auto t1 = std::chrono::steady_clock::now();
            ns_adj[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }

        for (int i = 0; i < 20; ++i)
            forward_cross_tile(weights, x_all.data() + std::size_t(order[i]) * batch * n_in, batch,
                               n_in, n_in, n_out, y.data());
        std::vector<double> ns_cross(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            forward_cross_tile(weights, x_all.data() + std::size_t(order[i]) * batch * n_in, batch,
                               n_in, n_in, n_out, y.data());
            auto t1 = std::chrono::steady_clock::now();
            ns_cross[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }

        std::sort(ns_adj.begin(), ns_adj.end());
        std::sort(ns_cross.begin(), ns_cross.end());
        const double med_adj = ns_adj[std::size_t(n_reps) / 2];
        const double med_cross = ns_cross[std::size_t(n_reps) / 2];
        const double mean_adj = std::accumulate(ns_adj.begin(), ns_adj.end(), 0.0) / n_reps;
        const double mean_cross = std::accumulate(ns_cross.begin(), ns_cross.end(), 0.0) / n_reps;
        std::printf("TIMING fp32 checkered forward adjacent_only (n_in=%zu n_out=%zu batch=%d, "
                    "%d calls): mean=%.1f ns/call median=%.1f ns/call\n",
                    n_in, n_out, int(batch), n_reps, mean_adj, med_adj);
        std::printf("TIMING fp32 checkered forward cross_tile (n_in=%zu n_out=%zu batch=%d, "
                    "%d calls): mean=%.1f ns/call median=%.1f ns/call\n",
                    n_in, n_out, int(batch), n_reps, mean_cross, med_cross);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
