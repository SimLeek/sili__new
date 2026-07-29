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
// Values are UNSCALED FP4 codes (no per-row value_scale/output_scale the
// way disldo has -- see the conversation's open question on this: Fable's
// codec has the same limitation, single scalar w_scale, no per-row
// calibration). Fine for weights already in FP4's native magnitude range;
// a real deployment on small-magnitude trained weights would need this
// added before quality claims can be trusted.

struct Block4Weights {
    uint32_t n_block_rows = 0, n_block_cols = 0;   // M/4, N/4 (M, N must be multiples of 4)
    std::vector<uint32_t> block_ptrs;               // size n_block_rows+1, cumulative block count
    std::vector<uint32_t> block_byte_ptrs;           // size n_block_rows+1, byte offset into block_col_bytes
    std::vector<uint8_t>  block_col_bytes;           // ULEB128-delta block-column indices
    std::vector<uint8_t>  block_data;                // 16 bytes/block: (imp<<4|w4), row-major within tile (col-major in j for the forward loop below)
    float w_scale = 1.0f;

    uint32_t n_blocks() const { return block_ptrs.empty() ? 0 : block_ptrs.back(); }
};

// y[4*r : 4*r+4] += sum over this block-row's active blocks of W_block @ x[4*bcol : 4*bcol+4].
// Caller zeroes y first (matches disldo_forward's own "accumulated into" convention).
inline void block4_forward(const Block4Weights& bw, const float* x, float* y,
                           int num_cpus, float w_scale_override = -1.0f) {
    const float ws = (w_scale_override >= 0.0f) ? w_scale_override : bw.w_scale;
    #pragma omp parallel for schedule(static) num_threads(num_cpus)
    for (int64_t rr = 0; rr < int64_t(bw.n_block_rows); ++rr) {
        const uint32_t r = uint32_t(rr);
        const uint32_t nblocks = bw.block_ptrs[r + 1] - bw.block_ptrs[r];
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
                for (int i = 0; i < 4; ++i)
                    yl[i] += FP4_TABLE[Wb[j * 4 + i] & 0xFu] * ws * xj;
            }
        }
        y[r * 4 + 0] += yl[0]; y[r * 4 + 1] += yl[1]; y[r * 4 + 2] += yl[2]; y[r * 4 + 3] += yl[3];
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
                    const uint8_t w4 = fp4_quantize(e.w);
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
