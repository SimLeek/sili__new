// Backward counterpart to test_disldo_block4_fp32_crosstile_forward.cpp's
// experiment: does pairing two NON-ADJACENT block4 tiles -- tiles sharing
// the same block-row `br` but at DIFFERENT block-columns `bc` (real gaps,
// checkered sparsity) -- into one 8-wide AVX2 op pay off for BACKWARD too?
//
// User's insight, confirmed against the real code before writing this:
// backward's existing row-pairing (process_row_pair_fp32 in
// linear_disldo.hpp) combines two ROWS of the SAME tile because both
// share the tile's 4 output columns (dy gather, col-indexed scale data).
// The cross-tile axis is the MIRROR IMAGE: for a FIXED row (li fixed),
// combine that SAME row's computation against TWO DIFFERENT tiles (bc0,
// bc1) sharing the same br. What's shared/per-half flips accordingly:
//   - row-level quantities (imp_scale, effective_lr, value_scale_k) are
//     now SHARED (single value, same row) instead of per-half.
//   - column-level quantities (col, out_scale_k, dy) are now PER-HALF
//     (genuinely different columns from 2 different tiles) instead of
//     shared.
//   - dx (mdx) accumulation FOLDS (sums) both tiles' contributions into
//     ONE row entry, instead of splitting into two rows.
//   - weight/importance updates write to TWO SEPARATE tiles' bytes
//     instead of one -- "the weight grads are separate anyway."
//   - mgamma (layer-wide) folds across both halves, like mdx.
//   - mcol (per-column) stays per-half (2 separate sets of 4), the
//     mirror of row-pairing's per-half mrow.
//
// Real disldo_backward serves as BOTH the correctness oracle (compare
// dx/weights after one call each, on independent copies of the same
// initial checkered weights) AND the timing baseline -- only the new
// backward_cross_tile function is hand-written here, keeping this
// experiment's own risk surface small.
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
            if ((bc % 2) != 0)
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

struct TileCoord {
    uint32_t br, bc;
    std::size_t elem_pos, byte_pos;
};
struct TilePair {
    TileCoord a, b;
};
static void collect_pairs(const Weights& weights, std::vector<TilePair>& pairs,
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

// experimental design: for each li (row) of a tile-pair sharing br,
// combine that ONE row's computation against BOTH tiles' 4 columns into
// one 8-wide op. Mirrors process_row_pair_fp32's math with the
// shared/per-half roles swapped (see file header). rank=1 only (this
// PoC uses RMSpropScalePolicy defaults, not full AQRS rank-N/gamma --
// proving the MECHANISM first, matching the same "prove it's correct and
// faster before adding every real-kernel feature" discipline as the
// forward PoC).
static void backward_cross_tile(Weights& weights, const float* input, SIZE_TYPE batch,
                                std::size_t in_cols, std::size_t n_in, std::size_t n_out,
                                const float* output_grad, float* mdx, float learning_rate) {
    std::fill(mdx, mdx + std::size_t(batch) * in_cols, 0.0f);
    std::vector<TilePair> pairs;
    std::vector<TileCoord> solos;
    collect_pairs(weights, pairs, solos);
    const bool training = (learning_rate != 0.0f);

    for (const auto& tp : pairs) {
        const uint32_t br = tp.a.br;
        const std::size_t col_baseA = std::size_t(tp.a.bc) * BLOCK4_TILE;
        const std::size_t col_baseB = std::size_t(tp.b.bc) * BLOCK4_TILE;

        // Read BOTH tiles' full 4x4 grids into local arrays FIRST, each
        // handle opened and closed before the other opens -- two
        // Block4TileHandle32s alive SIMULTANEOUSLY into the same row is a
        // real use-after-free hazard: a write-triggered resize on one
        // (sparse<->dense repack, block4_resize_tile_in_row) can memmove
        // the row's shared byte buffer out from under the other's still-
        // open byte_pos, exactly the hazard
        // disldo_backward.row_workspace_snapshot_fix documents and the
        // real kernel avoids via snapshot_row/unpack_workspace_tile.
        // Confirmed the hard way: an earlier version of this PoC held
        // tileA and tileB open across the whole li loop and segfaulted.
        float wA_grid[16], impA_grid[16], wB_grid[16], impB_grid[16];
        {
            auto tileA = weights.block4.at_index(tp.a.br, tp.a.bc, tp.a.elem_pos, tp.a.byte_pos);
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    wA_grid[li * 4 + lj] = tileA.get_weight(li, lj);
                    impA_grid[li * 4 + lj] = tileA.get_importance(li, lj);
                }
        }
        {
            auto tileB = weights.block4.at_index(tp.b.br, tp.b.bc, tp.b.elem_pos, tp.b.byte_pos);
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    wB_grid[li * 4 + lj] = tileB.get_weight(li, lj);
                    impB_grid[li * 4 + lj] = tileB.get_importance(li, lj);
                }
        }
        float wA_new[16], impA_new[16], wB_new[16], impB_new[16];
        std::memcpy(wA_new, wA_grid, sizeof(wA_new));
        std::memcpy(impA_new, impA_grid, sizeof(impA_new));
        std::memcpy(wB_new, wB_grid, sizeof(wB_new));
        std::memcpy(impB_new, impB_grid, sizeof(impB_new));

        for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
            const std::size_t row = std::size_t(br) * BLOCK4_TILE + li;
            if (row >= n_in)
                continue;
            const float imp_scale = weights.get_importance_scale(row);
            const float effective_lr = learning_rate;

            float w8[8], imp8[8], colvalid8[8];
            std::size_t colA[4], colB[4];
            for (uint32_t lj = 0; lj < 4; ++lj) {
                colA[lj] = col_baseA + lj;
                colB[lj] = col_baseB + lj;
                const bool haveA = colA[lj] < n_out;
                const bool haveB = colB[lj] < n_out;
                w8[lj] = haveA ? wA_grid[li * 4 + lj] : 0.0f;
                w8[lj + 4] = haveB ? wB_grid[li * 4 + lj] : 0.0f;
                imp8[lj] = haveA ? impA_grid[li * 4 + lj] : 0.0f;
                imp8[lj + 4] = haveB ? impB_grid[li * 4 + lj] : 0.0f;
                colvalid8[lj] = haveA ? 1.0f : 0.0f;
                colvalid8[lj + 4] = haveB ? 1.0f : 0.0f;
            }

            // combined_imp_scale is imp_scale(row) * get_output_importance_scale(col)
            // -- the SECOND factor is COLUMN-level (per-half, from tile A/B's own
            // columns respectively), not row-level. An earlier version of this PoC
            // used get_value_scale(row) here instead (a row-level quantity),
            // matching the mistaken assumption in this file's header comment that
            // ci's scale factors are entirely row-shared under cross-tile pairing.
            // That produced a ~0.0127 post-backward weight-update probe mismatch
            // against the real disldo_backward oracle -- traced by direct
            // comparison against process_row_pair_fp32's combined_imp_scale8
            // formula in linear_disldo.hpp.
            float out_imp_scale8[8];
            for (uint32_t lj = 0; lj < 4; ++lj) {
                out_imp_scale8[lj] =
                    colvalid8[lj] > 0 ? weights.get_output_importance_scale(colA[lj]) : 0.0f;
                out_imp_scale8[lj + 4] =
                    colvalid8[lj + 4] > 0 ? weights.get_output_importance_scale(colB[lj]) : 0.0f;
            }

            float S8[8], cw8[8], ci8[8], g_agg8[8] = {0}, contrib_agg8[8] = {0};
            for (uint32_t i = 0; i < 4; ++i) {
                S8[i] = colvalid8[i] > 0 ? weights.get_scale(row, colA[i]) : 0.0f;
                S8[i + 4] = colvalid8[i + 4] > 0 ? weights.get_scale(row, colB[i]) : 0.0f;
            }
            for (uint32_t i = 0; i < 8; ++i) {
                cw8[i] = w8[i];
                ci8[i] = imp8[i] * imp_scale * out_imp_scale8[i]; // combined_imp_scale,
                                                                  // rank-1-only PoC.
            }

            for (SIZE_TYPE b = 0; b < batch; ++b) {
                const float iv = input[static_cast<std::size_t>(b) * in_cols + row];
                float dyv8[8];
                for (uint32_t lj = 0; lj < 4; ++lj) {
                    dyv8[lj] = colvalid8[lj] > 0
                                   ? output_grad[static_cast<std::size_t>(b) * n_out + colA[lj]]
                                   : 0.0f;
                    dyv8[lj + 4] = colvalid8[lj + 4] > 0
                                       ? output_grad[static_cast<std::size_t>(b) * n_out + colB[lj]]
                                       : 0.0f;
                }
                float mdx_term = 0.0f;
                for (uint32_t i = 0; i < 8; ++i) {
                    const float g = dyv8[i] * iv;
                    if (training) {
                        const float contrib = cw8[i] * S8[i] * iv;
                        g_agg8[i] += g;
                        contrib_agg8[i] += contrib;
                    }
                    mdx_term += cw8[i] * S8[i] * dyv8[i];
                }
                // mdx is shaped (batch, in_cols) -- each batch sample gets
                // its OWN dx entry (mdx[b*in_cols+row]), not a single
                // accumulator shared across the batch loop. The "FOLD"
                // (both tiles' contributions sum into ONE row entry) still
                // happens -- via mdx_term's 8-wide sum above -- just
                // per-batch-sample, not across batch samples too. A real
                // bug in an earlier version of this PoC hoisted the
                // pointer outside the batch loop and accumulated every
                // batch sample's contribution into the SAME slot,
                // confirmed via a manual dense reference computation that
                // matched disldo_backward's oracle exactly while this
                // function's output didn't.
                mdx[static_cast<std::size_t>(b) * in_cols + row] += mdx_term;
            }

            if (training) {
                for (uint32_t i = 0; i < 8; ++i) {
                    ci8[i] = BoundedRMSpropSynapsePolicy<float>::update_ci(
                        ci8[i], g_agg8[i], contrib_agg8[i], 0.999f, 0.0f, 1e30f);
                    cw8[i] += BoundedRMSpropSynapsePolicy<float>::update_cw(
                        g_agg8[i], ci8[i], S8[i], effective_lr, 1e-8f, true, 1e30f, false);
                }
                for (uint32_t lj = 0; lj < 4; ++lj) {
                    if (colvalid8[lj] > 0) {
                        wA_new[li * 4 + lj] = cw8[lj];
                        impA_new[li * 4 + lj] = ci8[lj] / (imp_scale * out_imp_scale8[lj]);
                    }
                    if (colvalid8[lj + 4] > 0) {
                        wB_new[li * 4 + lj] = cw8[lj + 4];
                        impB_new[li * 4 + lj] = ci8[lj + 4] / (imp_scale * out_imp_scale8[lj + 4]);
                    }
                }
            }
        }

        if (training) {
            // Write BOTH tiles back, each handle opened and closed on its
            // own -- same one-handle-at-a-time discipline as the read
            // phase above.
            {
                auto tileA =
                    weights.block4.at_index(tp.a.br, tp.a.bc, tp.a.elem_pos, tp.a.byte_pos);
                for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                        tileA.set_weight(li, lj, wA_new[li * 4 + lj]);
                        tileA.set_importance(li, lj, impA_new[li * 4 + lj]);
                    }
            }
            {
                auto tileB =
                    weights.block4.at_index(tp.b.br, tp.b.bc, tp.b.elem_pos, tp.b.byte_pos);
                for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
                    for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                        tileB.set_weight(li, lj, wB_new[li * 4 + lj]);
                        tileB.set_importance(li, lj, impB_new[li * 4 + lj]);
                    }
            }
        }
    }

    // solos: no partner tile in this br -- fall back to per-column scalar
    // math (unchanged design, just not cross-tile-widened).
    for (const auto& s : solos) {
        auto tile = weights.block4.at_index(s.br, s.bc, s.elem_pos, s.byte_pos);
        const std::size_t col_base = std::size_t(s.bc) * BLOCK4_TILE;
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
            const std::size_t row = std::size_t(s.br) * BLOCK4_TILE + li;
            if (row >= n_in)
                continue;
            const float imp_scale = weights.get_importance_scale(row);
            for (uint32_t lj = 0; lj < 4; ++lj) {
                const std::size_t col = col_base + lj;
                if (col >= n_out)
                    continue;
                const float S = weights.get_scale(row, col);
                const float out_imp_scale = weights.get_output_importance_scale(col);
                float cw = tile.get_weight(li, lj);
                float ci = tile.get_importance(li, lj) * imp_scale * out_imp_scale;
                float g_agg = 0.0f, contrib_agg = 0.0f;
                for (SIZE_TYPE b = 0; b < batch; ++b) {
                    const float iv = input[static_cast<std::size_t>(b) * in_cols + row];
                    const float dyv = output_grad[static_cast<std::size_t>(b) * n_out + col];
                    const float g = dyv * iv;
                    if (training) {
                        g_agg += g;
                        contrib_agg += cw * S * iv;
                    }
                    mdx[static_cast<std::size_t>(b) * in_cols + row] += cw * S * dyv;
                }
                if (training) {
                    ci = BoundedRMSpropSynapsePolicy<float>::update_ci(ci, g_agg, contrib_agg,
                                                                       0.999f, 0.0f, 1e30f);
                    cw += BoundedRMSpropSynapsePolicy<float>::update_cw(g_agg, ci, S, learning_rate,
                                                                        1e-8f, true, 1e30f, false);
                    tile.set_weight(li, lj, cw);
                    tile.set_importance(li, lj, ci / (imp_scale * out_imp_scale));
                }
            }
        }
    }
}

int main() {
    std::mt19937 rng(998877);
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

        // disldo_backward (unlike disldo_forward's dc.empty() early exit)
        // touches weights.connections' DeltaCSRLayout unconditionally, so
        // it must be a REAL (if empty) delta_csr_from_absolute result --
        // just setting .layout.rows/.cols directly leaves elem_start/
        // elem_end unsized, causing an out-of-bounds vector read (an
        // std::vector assert failure in a debug build, a silent
        // segfault in -O3) inside DeltaCSRLayout::row_nnz. Confirmed via
        // gdb backtrace, not assumed. Same empty-scattered-side
        // construction the existing fp32/fp8/fp4 backward wide-simd
        // tests already use.
        Weights weights_oracle, weights_cross;
        std::vector<int> empty_ptrs(n_in + 1, 0);
        std::vector<int> empty_idx;
        std::vector<float> empty_w, empty_imp;
        weights_oracle.connections =
            delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                empty_ptrs, empty_idx, empty_w, empty_imp, n_in, n_out, 64, 64, 0.2f);
        weights_cross.connections =
            delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                empty_ptrs, empty_idx, empty_w, empty_imp, n_in, n_out, 64, 64, 0.2f);
        build_checkered(weights_oracle, w, imp, n_in, n_out);
        build_checkered(weights_cross, w, imp, n_in, n_out);

        std::vector<float> x(batch * n_in), dy(batch * n_out);
        for (auto& v : x)
            v = xdist(rng);
        for (auto& v : dy)
            v = xdist(rng);

        std::vector<float> dx_oracle(batch * n_in, 0.f), dx_cross(batch * n_in, 0.f);
        std::vector<float> nia(n_in, 0.f), nga(n_out, 0.f);
        disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE, RMSpropScalePolicy<float>,
                        false, false>(x.data(), batch, SIZE_TYPE(n_in), dy.data(), weights_oracle,
                                      dx_oracle.data(), nia.data(), nga.data(), 0.05f, 4, false,
                                      true);
        backward_cross_tile(weights_cross, x.data(), batch, n_in, n_in, n_out, dy.data(),
                            dx_cross.data(), 0.05f);

        double err_dx = 0.0;
        for (std::size_t i = 0; i < dx_oracle.size(); ++i)
            err_dx = std::max(err_dx, double(std::fabs(dx_oracle[i] - dx_cross[i])));
        CHECK(err_dx < 1e-2, "dx: cross_tile vs disldo_backward oracle max abs err %.6f too large",
              err_dx);

        // Direct raw cw/ci comparison, NOT a forward probe. disldo_backward
        // (the oracle) also runs its own unconditional per-row value_scale
        // RMSprop update (linear_disldo.hpp:3568-3648, block4's dead-row-
        // independent value_scale pass) after the per-cell weight loop --
        // this PoC's backward_cross_tile deliberately doesn't replicate that
        // (out of scope: the header comment already scopes this PoC to
        // "RMSpropScalePolicy defaults, not full AQRS rank-N/gamma", and
        // that scale-update machinery is orthogonal to the cross-tile
        // per-cell math this PoC exists to validate -- the real kernel port
        // reuses the EXISTING unmodified value_scale-update pass verbatim).
        // A forward probe run on both weight sets AFTER backward conflates
        // that expected S(row) drift with actual cw/ci correctness (an
        // earlier version of this test used a forward probe and saw a
        // spurious ~0.0127 mismatch traced exactly to this, confirmed by
        // temporarily comparing get_output_importance_scale/get_value_scale
        // between the two weight sets). Comparing raw stored cw/ci directly
        // is S-independent and is the actual claim under test.
        {
            const uint32_t block_rows = uint32_t((n_in + BLOCK4_TILE - 1) / BLOCK4_TILE);
            const uint32_t block_cols = uint32_t((n_out + BLOCK4_TILE - 1) / BLOCK4_TILE);
            double err_w = 0.0, err_i = 0.0;
            for (uint32_t br = 0; br < block_rows; ++br) {
                const std::size_t row_lo = std::size_t(br) * BLOCK4_TILE;
                const std::size_t row_hi = std::min(row_lo + BLOCK4_TILE, n_in);
                for (uint32_t bc = 0; bc < block_cols; ++bc) {
                    if ((bc % 2) != 0)
                        continue;
                    const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
                    const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, n_out);
                    auto tile_o = weights_oracle.block4.get_or_create(br, bc);
                    auto tile_c = weights_cross.block4.get_or_create(br, bc);
                    for (std::size_t row = row_lo; row < row_hi; ++row) {
                        const uint32_t li = uint32_t(row - row_lo);
                        for (std::size_t col = col_lo; col < col_hi; ++col) {
                            const uint32_t lj = uint32_t(col - col_lo);
                            err_w = std::max(err_w, double(std::fabs(tile_o.get_weight(li, lj) -
                                                                     tile_c.get_weight(li, lj))));
                            err_i =
                                std::max(err_i, double(std::fabs(tile_o.get_importance(li, lj) -
                                                                 tile_c.get_importance(li, lj))));
                        }
                    }
                }
            }
            CHECK(err_w < 1e-3,
                  "post-backward weight: cross_tile vs oracle max abs err %.6f too large", err_w);
            CHECK(err_i < 1e-3,
                  "post-backward importance: cross_tile vs oracle max abs err %.6f too large",
                  err_i);
            std::printf(
                "correctness: dx err=%.6f post-backward weight err=%.6f importance err=%.6f\n",
                err_dx, err_w, err_i);
        }
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

        Weights weights_oracle, weights_cross;
        std::vector<int> empty_ptrs(n_in + 1, 0);
        std::vector<int> empty_idx;
        std::vector<float> empty_w, empty_imp;
        weights_oracle.connections =
            delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                empty_ptrs, empty_idx, empty_w, empty_imp, n_in, n_out, 64, 64, 0.2f);
        weights_cross.connections =
            delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                empty_ptrs, empty_idx, empty_w, empty_imp, n_in, n_out, 64, 64, 0.2f);
        build_checkered(weights_oracle, w, imp, n_in, n_out);
        build_checkered(weights_cross, w, imp, n_in, n_out);

        const int n_instances = 16;
        std::vector<float> x_all(std::size_t(n_instances) * batch * n_in);
        std::vector<float> dy_all(std::size_t(n_instances) * batch * n_out);
        for (auto& v : x_all)
            v = xdist(rng);
        for (auto& v : dy_all)
            v = xdist(rng);
        const int n_reps = 200;
        std::vector<int> order(n_reps);
        for (int i = 0; i < n_reps; ++i)
            order[i] = i % n_instances;
        std::shuffle(order.begin(), order.end(), rng);

        std::vector<float> dx(batch * n_in), nia(n_in, 0.f), nga(n_out, 0.f);
        for (int i = 0; i < 20; ++i) {
            const float* x = x_all.data() + std::size_t(order[i % n_reps]) * batch * n_in;
            const float* dy = dy_all.data() + std::size_t(order[i % n_reps]) * batch * n_out;
            std::fill(dx.begin(), dx.end(), 0.f);
            std::fill(nia.begin(), nia.end(), 0.f);
            std::fill(nga.begin(), nga.end(), 0.f);
            disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE, RMSpropScalePolicy<float>,
                            false, false>(x, batch, SIZE_TYPE(n_in), dy, weights_oracle, dx.data(),
                                          nia.data(), nga.data(), 0.01f, 4, false, true);
        }
        std::vector<double> ns_oracle(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            const float* x = x_all.data() + std::size_t(order[i]) * batch * n_in;
            const float* dy = dy_all.data() + std::size_t(order[i]) * batch * n_out;
            std::fill(dx.begin(), dx.end(), 0.f);
            std::fill(nia.begin(), nia.end(), 0.f);
            std::fill(nga.begin(), nga.end(), 0.f);
            auto t0 = std::chrono::steady_clock::now();
            disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE, RMSpropScalePolicy<float>,
                            false, false>(x, batch, SIZE_TYPE(n_in), dy, weights_oracle, dx.data(),
                                          nia.data(), nga.data(), 0.01f, 4, false, true);
            auto t1 = std::chrono::steady_clock::now();
            ns_oracle[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }

        std::vector<float> dxc(batch * n_in);
        for (int i = 0; i < 20; ++i) {
            const float* x = x_all.data() + std::size_t(order[i % n_reps]) * batch * n_in;
            const float* dy = dy_all.data() + std::size_t(order[i % n_reps]) * batch * n_out;
            backward_cross_tile(weights_cross, x, batch, n_in, n_in, n_out, dy, dxc.data(), 0.01f);
        }
        std::vector<double> ns_cross(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            const float* x = x_all.data() + std::size_t(order[i]) * batch * n_in;
            const float* dy = dy_all.data() + std::size_t(order[i]) * batch * n_out;
            auto t0 = std::chrono::steady_clock::now();
            backward_cross_tile(weights_cross, x, batch, n_in, n_in, n_out, dy, dxc.data(), 0.01f);
            auto t1 = std::chrono::steady_clock::now();
            ns_cross[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }

        std::sort(ns_oracle.begin(), ns_oracle.end());
        std::sort(ns_cross.begin(), ns_cross.end());
        const double med_o = ns_oracle[std::size_t(n_reps) / 2];
        const double med_c = ns_cross[std::size_t(n_reps) / 2];
        const double mean_o = std::accumulate(ns_oracle.begin(), ns_oracle.end(), 0.0) / n_reps;
        const double mean_c = std::accumulate(ns_cross.begin(), ns_cross.end(), 0.0) / n_reps;
        std::printf("TIMING fp32 checkered backward adjacent_only/real (n_in=%zu n_out=%zu "
                    "batch=%d, %d calls): mean=%.1f ns/call median=%.1f ns/call\n",
                    n_in, n_out, int(batch), n_reps, mean_o, med_o);
        std::printf("TIMING fp32 checkered backward cross_tile (n_in=%zu n_out=%zu batch=%d, "
                    "%d calls): mean=%.1f ns/call median=%.1f ns/call\n",
                    n_in, n_out, int(batch), n_reps, mean_c, med_c);
    }

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
