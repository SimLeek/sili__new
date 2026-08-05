#pragma once
#include "fp4quant.hpp"
#include "delta_csr_types.hpp"  // uleb128_encode/decode
#include <algorithm>
#include <cstdint>
#include <vector>

// ── Block4: dense 4x4 tiles for the LOCALLY-DENSE part of a sparse matrix ──
//
// Companion to disldo (linear_disldo.hpp), not a replacement: a real layer
// is `y = disldo_forward(A, x) + block4_forward(B, x)`, where A holds the
// scattered remainder (existing SparseLinearLayer, unchanged) and B holds
// this file's dense 4x4 tiles. Rationale (see prototypes/sili_ell/
// bench_block4_dual.cpp and bench_block_kernel.cpp for the validation this
// header promotes to production code):
//   - disldo's per-synapse gather (`output[col] += w*input[row]` at a
//     data-dependent col) never auto-vectorizes on this hardware/compiler,
//     confirmed via -fopt-info-vec (GCC: "complicated access pattern").
//   - A tile stored fully DENSE (all 16 slots, zero-weighted where empty)
//     has no gather at all -- every read/write is a fixed, compile-time-
//     known offset -- and DOES auto-vectorize (confirmed, real 1.44-2.03x
//     over scalar on this CPU; hand AVX2 barely beats GCC's own SSE-width
//     auto-vectorization, since this CPU generation runs AVX2 as two
//     internal 128-bit micro-ops -- 4x4 was sized to this measured ~4x
//     real ceiling, not an assumed 8x).
//   - Real breakeven (bench_block4_dual.cpp, stabilized at a large enough
//     problem size to get past OS-scheduling noise): a tile needs >= ~10%
//     of its 16 slots live to beat disldo's own per-synapse throughput.
//   - Index bits are near-zero: within an active tile, position IS the
//     column -- only the block-level (row/4, col/4) index needs storing,
//     at the SAME ULEB128-delta scheme disldo already uses, just 1/4 the
//     cardinality. Memory per slot: exactly 1 byte (importance nibble <<
//     4 | weight nibble, same FP4 pair disldo stores) -- no per-synapse
//     index cost at all once a tile is active.
//
// Values are FP4 codes scaled by a PER-ORIGINAL-ROW value_scale (matching
// disldo's own default "per_row" calibration -- see sili_peridot's
// model/sili_block.py, _build_step_layer_from_arrays). A single global
// scale was tried first and found (not assumed) insufficient on real
// weights: one real MiniCPM5 layer's weight magnitudes spanned 0.0095 to
// 0.629 (66x) -- a global scale calibrated to the max still left the
// smallest weights far below FP4's floor (0.5), losing 43% of real
// nonzeros. Per-row narrows the range each scale has to cover to
// whatever ONE row's own weights actually span, recovering much more of
// the real distribution -- same reasoning disldo's own per-row scale is
// built on, applied here to block4's dense tiles the same way.

struct Block4Weights {
    uint32_t n_block_rows = 0, n_block_cols = 0;   // M/4, N/4 (M, N must be multiples of 4)
    std::vector<uint32_t> block_ptrs;               // size n_block_rows+1, cumulative block count
    std::vector<uint32_t> block_byte_ptrs;           // size n_block_rows+1, byte offset into block_col_bytes
    std::vector<uint8_t>  block_col_bytes;           // ULEB128-delta block-column indices
    std::vector<uint8_t>  block_data;                // 16 bytes/block: (imp<<4|w4), row-major within tile (col-major in j for the forward loop below)
    std::vector<float>    row_scale;                 // size n_block_rows*4 (one per ORIGINAL row, matching disldo's own granularity)

    uint32_t n_blocks() const { return block_ptrs.empty() ? 0 : block_ptrs.back(); }
};

// y[4*r : 4*r+4] += sum over this block-row's active blocks of W_block @ x[4*bcol : 4*bcol+4].
// Caller zeroes y first (matches disldo_forward's own "accumulated into" convention).
inline void block4_forward(const Block4Weights& bw, const float* x, float* y, int num_cpus) {
    #pragma omp parallel for schedule(static) num_threads(num_cpus)
    for (int64_t rr = 0; rr < int64_t(bw.n_block_rows); ++rr) {
        const uint32_t r = uint32_t(rr);
        const uint32_t nblocks = bw.block_ptrs[r + 1] - bw.block_ptrs[r];
        // Loaded once per block-row (4 consecutive original rows), reused
        // across every active block in this row -- one small extra load,
        // not a per-synapse cost.
        const float rs0 = bw.row_scale[r * 4 + 0], rs1 = bw.row_scale[r * 4 + 1],
                    rs2 = bw.row_scale[r * 4 + 2], rs3 = bw.row_scale[r * 4 + 3];
        float yl[4] = {0.f, 0.f, 0.f, 0.f};
        std::size_t bytepos = bw.block_byte_ptrs[r];
        uint32_t prev = 0;
        const uint8_t* data = bw.block_data.data() + std::size_t(bw.block_ptrs[r]) * 16;
        for (uint32_t b = 0; b < nblocks; ++b) {
            std::size_t dlen = 0;
            uint32_t delta = uleb128_decode<uint32_t>(bw.block_col_bytes.data() + bytepos, dlen);
            bytepos += dlen;
            const uint32_t bcol = prev + delta;
            prev = bcol;
            const uint8_t* Wb = data + std::size_t(b) * 16;   // 16 bytes, column-major: Wb[j*4+i]
            const float* xb = x + std::size_t(bcol) * 4;
            #pragma omp simd
            for (int j = 0; j < 4; ++j) {
                const float xj = xb[j];
                yl[0] += FP4_TABLE[Wb[j * 4 + 0] & 0xFu] * rs0 * xj;
                yl[1] += FP4_TABLE[Wb[j * 4 + 1] & 0xFu] * rs1 * xj;
                yl[2] += FP4_TABLE[Wb[j * 4 + 2] & 0xFu] * rs2 * xj;
                yl[3] += FP4_TABLE[Wb[j * 4 + 3] & 0xFu] * rs3 * xj;
            }
        }
        y[r * 4 + 0] += yl[0]; y[r * 4 + 1] += yl[1]; y[r * 4 + 2] += yl[2]; y[r * 4 + 3] += yl[3];
    }
}

// ── Backward: transpose + dx, then a separate weight-update pass ──
//
// Forward walks row=output (y[r] += W[r,c]*x[c]); backward-dx needs
// dx[c] += W[r,c]*dy[r], summing over ROWS for a FIXED column -- the
// opposite direction. Same structural reason Fable's own banked/packed
// codecs need dual CSR/CSC storage (see THEORY.md Lecture 1: "gather
// form... needs no atomics... is why we insist on two views"). Rather
// than scatter (which would throw away the whole reason block4 exists),
// build a transposed copy once and reuse block4_forward's exact
// validated kernel shape on it.

// Byte-for-byte 4x4 transpose of one tile: tile_bytes[j*4+i] (original
// convention, see split_for_block4) becomes tile_bytes[i*4+j] in the
// transposed structure -- a literal matrix transpose of the 16-byte tile.
inline Block4Weights transpose_block4(const Block4Weights& bw) {
    Block4Weights t;
    t.n_block_rows = bw.n_block_cols;
    t.n_block_cols = bw.n_block_rows;
    // row_scale here is keyed by ORIGINAL OUTPUT row (unchanged meaning),
    // which is now the COLUMN dimension of the transposed tile -- consumed
    // per-j (not per-i) in block4_backward_dx below.
    t.row_scale = bw.row_scale;

    struct Entry { uint32_t local_i, local_j; uint8_t byte; };
    std::vector<std::vector<Entry>> tiles(std::size_t(t.n_block_rows) * t.n_block_cols);
    for (uint32_t br = 0; br < bw.n_block_rows; ++br) {
        const uint32_t nblocks = bw.block_ptrs[br + 1] - bw.block_ptrs[br];
        std::size_t bytepos = bw.block_byte_ptrs[br];
        uint32_t prev = 0;
        const uint8_t* data = bw.block_data.data() + std::size_t(bw.block_ptrs[br]) * 16;
        for (uint32_t b = 0; b < nblocks; ++b) {
            std::size_t dlen = 0;
            uint32_t delta = uleb128_decode<uint32_t>(bw.block_col_bytes.data() + bytepos, dlen);
            bytepos += dlen;
            const uint32_t bc = prev + delta;
            prev = bc;
            const uint8_t* Wb = data + std::size_t(b) * 16;
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    const uint8_t byte = Wb[j * 4 + i];
                    if ((byte & 0xFu) == 0) continue;
                    // new tile at (new_br=bc, new_bc=br); new_local_i=j, new_local_j=i
                    tiles[std::size_t(bc) * t.n_block_cols + br].push_back({uint32_t(j), uint32_t(i), byte});
                }
            }
        }
    }

    t.block_ptrs.assign(t.n_block_rows + 1, 0);
    t.block_byte_ptrs.assign(t.n_block_rows + 1, 0);
    for (uint32_t br = 0; br < t.n_block_rows; ++br) {
        uint32_t prev_bc = 0;
        for (uint32_t bc = 0; bc < t.n_block_cols; ++bc) {
            auto& tl = tiles[std::size_t(br) * t.n_block_cols + bc];
            if (tl.empty()) continue;
            uint8_t tile_bytes[16] = {0};
            for (auto& e : tl) tile_bytes[e.local_j * 4 + e.local_i] = e.byte;
            uint8_t tmp[6];
            std::size_t dlen = uleb128_encode<uint32_t>(bc - prev_bc, tmp);
            t.block_col_bytes.insert(t.block_col_bytes.end(), tmp, tmp + dlen);
            t.block_data.insert(t.block_data.end(), tile_bytes, tile_bytes + 16);
            prev_bc = bc;
            t.block_ptrs[br + 1]++;
        }
        t.block_ptrs[br + 1] += t.block_ptrs[br];
        t.block_byte_ptrs[br + 1] = uint32_t(t.block_col_bytes.size());
    }
    return t;
}

// dx[4*r : 4*r+4] += sum over this (transposed) block-row's active blocks
// of W_block^T @ dy[4*bcol : 4*bcol+4]. bw_t must be transpose_block4()'s
// output. Same kernel shape as block4_forward, reused -- the only real
// difference is row_scale is consumed per-j here (it's keyed by the
// original OUTPUT row, which is this structure's column dimension), not
// per-i like block4_forward's own row_scale usage.
inline void block4_backward_dx(const Block4Weights& bw_t, const float* dy, float* dx, int num_cpus) {
    #pragma omp parallel for schedule(static) num_threads(num_cpus)
    for (int64_t rr = 0; rr < int64_t(bw_t.n_block_rows); ++rr) {
        const uint32_t r = uint32_t(rr);
        const uint32_t nblocks = bw_t.block_ptrs[r + 1] - bw_t.block_ptrs[r];
        float xl[4] = {0.f, 0.f, 0.f, 0.f};
        std::size_t bytepos = bw_t.block_byte_ptrs[r];
        uint32_t prev = 0;
        const uint8_t* data = bw_t.block_data.data() + std::size_t(bw_t.block_ptrs[r]) * 16;
        for (uint32_t b = 0; b < nblocks; ++b) {
            std::size_t dlen = 0;
            uint32_t delta = uleb128_decode<uint32_t>(bw_t.block_col_bytes.data() + bytepos, dlen);
            bytepos += dlen;
            const uint32_t bcol = prev + delta;
            prev = bcol;
            const uint8_t* Wb = data + std::size_t(b) * 16;
            const float* dyb = dy + std::size_t(bcol) * 4;
            const float rs0 = bw_t.row_scale[bcol * 4 + 0], rs1 = bw_t.row_scale[bcol * 4 + 1],
                        rs2 = bw_t.row_scale[bcol * 4 + 2], rs3 = bw_t.row_scale[bcol * 4 + 3];
            #pragma omp simd
            for (int j = 0; j < 4; ++j) {
                xl[j] += FP4_TABLE[Wb[0 * 4 + j] & 0xFu] * rs0 * dyb[0]
                       + FP4_TABLE[Wb[1 * 4 + j] & 0xFu] * rs1 * dyb[1]
                       + FP4_TABLE[Wb[2 * 4 + j] & 0xFu] * rs2 * dyb[2]
                       + FP4_TABLE[Wb[3 * 4 + j] & 0xFu] * rs3 * dyb[3];
            }
        }
        dx[r * 4 + 0] += xl[0]; dx[r * 4 + 1] += xl[1]; dx[r * 4 + 2] += xl[2]; dx[r * 4 + 3] += xl[3];
    }
}

// In-place weight update: w -= effective_lr*g/(1+|ci|), ci -= g*effective_lr,
// g = dy[r]*x[c] -- same damped-importance formula as disldo_backward
// (linear_disldo.hpp), same fp4_quantize_stochastic write path (matching
// the stochastic-rounding fix this project already made to disldo's own
// gradient-driven writes -- deterministic round-to-nearest at these
// small updates would silently no-op the same way disldo's did before
// that fix). Walks the ORIGINAL (row=output) structure -- no transpose
// needed for the update itself, only dx needed one.
inline void block4_weight_update(Block4Weights& bw, const float* x, const float* dy,
                                 float learning_rate, int num_cpus) {
    if (learning_rate == 0.0f) return;
    #pragma omp parallel for schedule(static) num_threads(num_cpus)
    for (int64_t rr = 0; rr < int64_t(bw.n_block_rows); ++rr) {
        const uint32_t r = uint32_t(rr);
        const uint32_t nblocks = bw.block_ptrs[r + 1] - bw.block_ptrs[r];
        const float rs[4] = {bw.row_scale[r * 4 + 0], bw.row_scale[r * 4 + 1],
                              bw.row_scale[r * 4 + 2], bw.row_scale[r * 4 + 3]};
        const float dyl[4] = {dy[r * 4 + 0], dy[r * 4 + 1], dy[r * 4 + 2], dy[r * 4 + 3]};
        std::size_t bytepos = bw.block_byte_ptrs[r];
        uint32_t prev = 0;
        uint8_t* data = bw.block_data.data() + std::size_t(bw.block_ptrs[r]) * 16;
        for (uint32_t b = 0; b < nblocks; ++b) {
            std::size_t dlen = 0;
            uint32_t delta = uleb128_decode<uint32_t>(bw.block_col_bytes.data() + bytepos, dlen);
            bytepos += dlen;
            const uint32_t bcol = prev + delta;
            prev = bcol;
            uint8_t* Wb = data + std::size_t(b) * 16;
            const float xc0 = x[bcol * 4 + 0], xc1 = x[bcol * 4 + 1],
                        xc2 = x[bcol * 4 + 2], xc3 = x[bcol * 4 + 3];
            for (int j = 0; j < 4; ++j) {
                const float xj = (j == 0) ? xc0 : (j == 1) ? xc1 : (j == 2) ? xc2 : xc3;
                for (int i = 0; i < 4; ++i) {
                    uint8_t& byte = Wb[j * 4 + i];
                    const uint8_t code = byte & 0xFu;
                    if (code == 0 && (byte >> 4) == 0) continue;   // fully sleeping slot
                    const float g = dyl[i] * xj;
                    if (g == 0.0f) continue;
                    float cw = FP4_TABLE[code] * rs[i];
                    float ci = FP4_TABLE[byte >> 4];   // importance stored unscaled (0..6 range directly, matches disldo's imp4 reuse of fp4_quantize)
                    ci -= g * learning_rate;
                    cw -= learning_rate * g / (1.0f + std::abs(ci));
                    const uint8_t new_w4 = fp4_quantize_stochastic(cw / rs[i]);
                    const uint8_t new_imp4 = fp4_quantize_stochastic(ci) & 0xFu;
                    byte = uint8_t((new_imp4 << 4) | new_w4);
                }
            }
        }
    }
}

// ── Construction: split a CSR (ptrs/indices/weights) into a locally-dense
// remainder (B, this file) and a scattered leftover (A, unchanged CSR
// arrays for the caller to build via delta_csr_from_absolute as usual) ──
//
// Policy: partition into 4x4 tiles; a tile whose live-slot fraction is
// >= min_fill_frac gets moved to B (stored fully dense, remaining slots
// zero-weighted); everything else stays in the returned "leftover" CSR
// triples for A. min_fill_frac defaults to 0.10 (bench_block4_dual.cpp's
// measured real breakeven) with a caller-settable margin since the exact
// crossover will vary by CPU/build -- re-verify before trusting a different
// default elsewhere.
struct Block4SplitResult {
    Block4Weights block4;
    std::vector<std::pair<uint32_t, uint32_t>> leftover_rc;  // (row, col) for A
    std::vector<float> leftover_w;
    std::vector<float> leftover_imp;
};

inline Block4SplitResult split_for_block4(
    uint32_t M, uint32_t N,
    const std::vector<uint32_t>& csr_ptrs,       // size M+1
    const std::vector<uint32_t>& csr_idx,        // column indices, row-sorted ascending within each row
    const std::vector<float>& csr_w,
    const std::vector<float>& csr_imp,
    float min_fill_frac = 0.10f)
{
    Block4SplitResult res;
    const uint32_t Mb = M / 4, Nb = N / 4;   // caller must ensure M, N are multiples of 4
    res.block4.n_block_rows = Mb;
    res.block4.n_block_cols = Nb;

    // Real trained-model weights are typically far smaller in magnitude
    // than FP4's smallest representable nonzero (0.5) -- the exact same
    // gap disldo's own per-row value_scale exists to close (see
    // model/sili_block.py's _build_step_layer_from_arrays in sili_peridot).
    // A first version used one GLOBAL scale and found it insufficient on
    // real weights (found the hard way, not assumed): one real MiniCPM5
    // layer's weight magnitudes spanned 0.0095-0.629 (66x), and a global
    // scale still left 43% of real nonzeros below FP4's floor even after
    // scaling. Per-row narrows this to whatever each individual row's own
    // weights span -- computed here exactly like disldo's own per-row
    // pass (max_abs per row / 6.0, applied before quantizing).
    res.block4.row_scale.assign(M, 1.0f);
    for (uint32_t row = 0; row < M; ++row) {
        float max_abs = 0.0f;
        for (uint32_t e = csr_ptrs[row]; e < csr_ptrs[row + 1]; ++e)
            max_abs = std::max(max_abs, std::abs(csr_w[e]));
        if (max_abs > 0.0f) res.block4.row_scale[row] = max_abs / 6.0f;
    }

    // Bucket every (row, col, w, imp) into its 4x4 tile.
    struct Entry { uint32_t local_i, local_j; float w, imp; uint32_t orig_row, orig_col; };
    std::vector<std::vector<Entry>> tiles(std::size_t(Mb) * Nb);
    for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t e = csr_ptrs[row]; e < csr_ptrs[row + 1]; ++e) {
            uint32_t col = csr_idx[e];
            uint32_t br = row / 4, bc = col / 4;
            tiles[std::size_t(br) * Nb + bc].push_back(
                {row % 4, col % 4, csr_w[e], csr_imp[e], row, col});
        }
    }

    res.block4.block_ptrs.assign(Mb + 1, 0);
    res.block4.block_byte_ptrs.assign(Mb + 1, 0);

    for (uint32_t br = 0; br < Mb; ++br) {
        uint32_t prev_bc = 0;
        for (uint32_t bc = 0; bc < Nb; ++bc) {
            auto& t = tiles[std::size_t(br) * Nb + bc];
            if (t.empty()) continue;
            const float fill = float(t.size()) / 16.0f;
            if (fill >= min_fill_frac) {
                uint8_t tile_bytes[16] = {0};
                for (auto& e : t) {
                    const float inv_row_scale = 1.0f / res.block4.row_scale[e.orig_row];
                    const uint8_t w4 = fp4_quantize(e.w * inv_row_scale);
                    const uint8_t imp4 = fp4_quantize(e.imp) & 0xFu;   // reuse fp4_quantize for a 4-bit importance code too
                    tile_bytes[e.local_j * 4 + e.local_i] = uint8_t((imp4 << 4) | w4);
                }
                uint8_t tmp[6];
                std::size_t dlen = uleb128_encode<uint32_t>(bc - prev_bc, tmp);
                res.block4.block_col_bytes.insert(res.block4.block_col_bytes.end(), tmp, tmp + dlen);
                res.block4.block_data.insert(res.block4.block_data.end(), tile_bytes, tile_bytes + 16);
                prev_bc = bc;
                res.block4.block_ptrs[br + 1]++;
            } else {
                for (auto& e : t) {
                    res.leftover_rc.push_back({e.orig_row, e.orig_col});
                    res.leftover_w.push_back(e.w);
                    res.leftover_imp.push_back(e.imp);
                }
            }
        }
        res.block4.block_ptrs[br + 1] += res.block4.block_ptrs[br];
        res.block4.block_byte_ptrs[br + 1] = uint32_t(res.block4.block_col_bytes.size());
    }
    return res;
}
