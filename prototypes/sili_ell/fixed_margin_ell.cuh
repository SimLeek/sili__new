// fixed_margin_ell.cuh
// ===========================================================================
// Doubly-regular sparse weights: every row owns exactly R synapses, every
// column exactly C, with M*R == N*C. CPU (OpenMP) + CUDA kernels.
// Safe to include from plain C++; CUDA kernels only appear under nvcc.
//
// Glossary, since the terms outran the code:
//   ELL / ELLPACK : sparse format holding a FIXED number of entries per row
//                   as dense [rows x width] arrays (indices + values).
//                   CSR with a constant row count IS ELL: row_ptr[i] = i*R
//                   becomes closed form and vanishes. Transposed, the same
//                   trick gives the column copy (CSC -> [N x C]).
//   margins       : the per-row and per-column nnz counts (R and C).
//   biregular     : constant on both margins. The invariant of this file.
//   curveball     : margin-preserving rewiring: two rows trade columns
//                   drawn from the symmetric difference of their adjacency
//                   lists. trade_pair() below commits one traded edge.
//
// Structural init is a wrapped diagonal, G disjoint sheets:
//
//        j 0 1 2 3 4 5 6 7 8
//      i +------------------
//      0 | X X X . . . . . .
//      1 | . . . X X X . . .        M=6 N=9 R=3 C=2, one sheet.
//      2 | . . . . . . X X X        The stripe never truncates at the
//      3 | X X X . . . . . .        edge, it wraps the torus.
//      4 | . . . X X X . . .        (donuts: also sparse, also evenly
//      5 | . . . . . . X X X         distributed, also load-balanced)
//
// Slots are stable for the lifetime of the matrix. Growth, death, and
// relocation all reuse slots in place: no compaction, no re-sort, ever.
// ===========================================================================

#pragma once
#include <cstdint>
#include <cassert>
#include <vector>

#if defined(__CUDACC__)
#define FME_HD __host__ __device__
#else
#define FME_HD
#endif

namespace fme {

// Silent = structurally placed, not yet grown. The sign bit is the mark.
// Kernels skip silent slots; their weights are kept at zero at the only
// two places that could touch them (init and trade_pair), so there is no
// bulk zeroing pass, on-thread or off.
constexpr int32_t SILENT_BIT = (int32_t)0x80000000;
FME_HD inline int32_t id_of  (int32_t v) { return v & 0x7fffffff; }
FME_HD inline bool    is_live(int32_t v) { return v >= 0; }

// All arrays are caller-allocated (host or device as appropriate).
// int32 slot ids cap total nnz at 2^31; at a 2M-synapse scale that is
// three decades of headroom.
struct FixedMarginELL {
    int M, N, R, C;    // M*R == N*C; R and C multiples of warp/SIMD width

    // Row copy ("CSR minus row_ptr"). Backward traverses this, and it is
    // the authoritative copy: weight updates land here.
    int32_t* colidx;   // [M*R] input index per row slot (may carry SILENT_BIT)
    float*   wr;       // [M*R] weights in row order

    // Column copy ("CSC minus col_ptr"). Forward push traverses this.
    // Follows wr through the sync cursor.
    int32_t* rowidx;   // [N*C] output index per column slot
    float*   wc;       // [N*C] weights in column order

    // Slot permutation linking the two copies. Structure edits patch both
    // copies immediately; only VALUES ride the cursor.
    int32_t* r2c;      // [M*R] row slot k mirrors column slot r2c[k]
    int32_t* c2r;      // [N*C] inverse
    int32_t  cursor;   // next column slot ell_sync_window will refresh
};

// ---------------------------------------------------------------------------
// Structural init: G sheets, each an offset wrapped diagonal. Sheet g owns
// row slots [g*R/G, (g+1)*R/G) with columns offset by g*(N/G), so sheets
// are disjoint within every row. Sheets g >= liveSheets are placed SILENT:
// growth later is clearing sign bits, not allocating.
//
// Per-sheet margin proof: k = i*(R/G) + u enumerates 0 .. M*R/G - 1 exactly
// once, so k % N hits every column (M*R)/(G*N) = C/G times; adding a
// constant offset mod N changes nothing. Sheets are a starting point --
// score-driven trades will scramble the structure under the invariant.
// ---------------------------------------------------------------------------
inline void init_wrapped_diagonal(FixedMarginELL& W, int G, int liveSheets)
{
    assert((int64_t)W.M * W.R == (int64_t)W.N * W.C);
    assert(W.R <= W.N && W.C <= W.M);
    assert(G >= 1 && W.R % G == 0 && W.C % G == 0);
    const int rs = W.R / G;              // row slots per sheet
    assert(rs <= W.N / G);               // arcs fit disjointly in a row
    for (int i = 0; i < W.M; ++i)
        for (int g = 0; g < G; ++g)
            for (int u = 0; u < rs; ++u) {
                const int64_t k = (int64_t)i * rs + u;
                const int32_t j =
                    (int32_t)((k + (int64_t)g * (W.N / G)) % W.N);
                W.colidx[(int64_t)i * W.R + g * rs + u] =
                    (g < liveSheets) ? j : (j | SILENT_BIT);
            }
}

// ---------------------------------------------------------------------------
// One counting pass builds the column copy and both permutations. fill[j]
// landing at exactly C for every j IS the margin invariant; the assert
// catches any broken construction loudly. Weights assumed zero-initialized
// (or copy wr if warm-starting).
// ---------------------------------------------------------------------------
inline void build_column_mirror(FixedMarginELL& W)
{
    std::vector<int32_t> fill(W.N, 0);
    for (int64_t k = 0; k < (int64_t)W.M * W.R; ++k) {
        const int32_t v = W.colidx[k];
        const int32_t j = id_of(v);
        const int32_t q = j * W.C + fill[j]++;
        const int32_t i = (int32_t)(k / W.R);
        W.rowidx[q] = is_live(v) ? i : (i | SILENT_BIT);
        W.r2c[k] = q;
        W.c2r[q] = (int32_t)k;
        W.wc[q]  = W.wr[k];
    }
    for (int j = 0; j < W.N; ++j) assert(fill[j] == W.C);
    W.cursor = 0;
}

// ---------------------------------------------------------------------------
// Commit one traded edge of a curveball move: row i gives column jA and
// receives jB; row ip does the reverse. The caller's set ops guarantee jB
// is not already in row i and jA not in row ip -- which is exactly what
// "drawn from the symmetric difference" gives you for free (your parallel
// sorted-COO merge is the machine that computes it).
//
// Eight index writes, four weight writes, zero moves. Every slot stays
// where it is, only its label changes, so both copies and the permutation
// stay exact and the margins never break, even transiently. Shown for live
// slots; to relocate silent capacity, OR SILENT_BIT back onto the four
// patched indices. Run trades while the sync cursor is parked, or accept
// that any value crossing a trade mid-flight gets reset to zero anyway.
// ---------------------------------------------------------------------------
inline void trade_pair(FixedMarginELL& W, int i, int sA, int ip, int sB)
{
    const int64_t kA = (int64_t)i  * W.R + sA;
    const int64_t kB = (int64_t)ip * W.R + sB;
    const int32_t qA = W.r2c[kA];        // slot inside column jA's block
    const int32_t qB = W.r2c[kB];        // slot inside column jB's block
    const int32_t jA = id_of(W.colidx[kA]);
    const int32_t jB = id_of(W.colidx[kB]);
    (void)jA; (void)jB;
    W.colidx[kA] = id_of(W.colidx[kB]);  // rows swap column ownership
    W.colidx[kB] = jA;
    W.rowidx[qA] = ip;                   // columns swap row ownership,
    W.rowidx[qB] = i;                    // slots stay in their column
    W.r2c[kA] = qB;  W.c2r[qB] = (int32_t)kA;
    W.r2c[kB] = qA;  W.c2r[qA] = (int32_t)kB;
    W.wr[kA] = W.wr[kB] = 0.f;           // relocated synapses are newborn
    W.wc[qA] = W.wc[qB] = 0.f;
}

#if defined(__CUDACC__)
// ===========================================================================
// CUDA kernels. Launch configs in comments. Kernels are batch-1 vectors;
// for batches add blockIdx.y and stride in/out by N and M.
// ===========================================================================

// ---------------------------------------------------------------------------
// Pull / gather. One warp per left-hand row; lane L covers slots L, L+32,
// ... so index and weight loads are fully coalesced, and width % 32 == 0
// means each row block is an exact number of 128-byte sectors with no tail.
//
// The SAME kernel runs both directions, because the two copies are the
// same shape with the sides swapped:
//   forward  y  = W x    : ell_gather(colidx, wr, x,  y,  M, R)
//   backward dx = W^T dy : ell_gather(rowidx, wc, dy, dx, N, C)
// Launch: <<<(rows*32 + 255)/256, 256>>>
// ---------------------------------------------------------------------------
__global__ void ell_gather(const int32_t* __restrict__ idx,
                           const float*   __restrict__ w,
                           const float*   __restrict__ in,
                           float*         __restrict__ out,
                           int rows, int width)
{
    const int row  = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int64_t base = (int64_t)row * width;
    float acc = 0.f;
    for (int s = lane; s < width; s += 32) {
        const int32_t v  = idx[base + s];
        const float   xv = in[id_of(v)];
        acc += is_live(v) ? w[base + s] * xv : 0.f;
    }
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

// ---------------------------------------------------------------------------
// Event-driven forward through the column copy ("weights as CSC"): only
// inputs that fired do work, and equal C means every active warp does
// IDENTICAL work -- zero load imbalance by construction, not by scheduling.
// active[a] = input index, xa[a] = its value. out pre-zeroed (or carrying
// decayed state; that is your integrate-and-fire's business, not this
// kernel's). Scattered atomics are inherent to push; L2 atomics on modern
// parts take it fine.
// Launch: <<<(nActive*32 + 255)/256, 256>>>
// ---------------------------------------------------------------------------
__global__ void ell_push_forward(const int32_t* __restrict__ rowidx,
                                 const float*   __restrict__ wc,
                                 const int32_t* __restrict__ active,
                                 const float*   __restrict__ xa,
                                 float* __restrict__ out,
                                 int nActive, int C)
{
    const int a    = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (a >= nActive) return;
    const int64_t base = (int64_t)active[a] * C;
    const float   xv   = xa[a];
    for (int t = lane; t < C; t += 32) {
        const int32_t v = rowidx[base + t];
        if (is_live(v)) atomicAdd(&out[id_of(v)], wc[base + t] * xv);
    }
}

// ---------------------------------------------------------------------------
// Event-driven backward through the row copy ("weights as CSR"), fused
// with the weight update. For each output row with nonzero grad:
//     dx[j]     += w * dy[i]              (scatter)
//     wr[i, s]  -= lr * dy[i] * x[j]      (swap this line for your rule:
//                                          traces, energy terms, etc.)
// This is why wr is the authoritative copy: updates land here once per
// synapse per step, and wc follows through the cursor. activeOut rows are
// distinct, so wr writes are race-free; only dx needs atomics.
// Launch: <<<(nActive*32 + 255)/256, 256>>>
// ---------------------------------------------------------------------------
__global__ void ell_push_backward(const int32_t* __restrict__ colidx,
                                  float*         __restrict__ wr,
                                  const int32_t* __restrict__ activeOut,
                                  const float*   __restrict__ dyA,
                                  const float*   __restrict__ x,
                                  float* __restrict__ dx,
                                  float lr, int nActive, int R)
{
    const int a    = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (a >= nActive) return;
    const int     i    = activeOut[a];
    const float   g    = dyA[a];
    const int64_t base = (int64_t)i * R;
    for (int s = lane; s < R; s += 32) {
        const int32_t v = colidx[base + s];
        if (!is_live(v)) continue;
        const int32_t j = id_of(v);
        const float   w = wr[base + s];
        atomicAdd(&dx[j], w * g);
        wr[base + s] = w - lr * g * x[j];
    }
}

// ---------------------------------------------------------------------------
// Stepwise wr -> wc: refresh a window of column slots through c2r.
// Gathered reads, coalesced writes, and the entire sync state is one
// integer. Clamps at NC; wrap the cursor on the host:
//     W.cursor = (start + count >= NC) ? 0 : start + count;
// Launch: <<<(count + 255)/256, 256>>>
// ---------------------------------------------------------------------------
__global__ void ell_sync_window(const int32_t* __restrict__ c2r,
                                const float*   __restrict__ wr,
                                float* __restrict__ wc,
                                int32_t start, int32_t count, int32_t NC)
{
    const int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t q = start + t;
    if (t >= count || q >= NC) return;
    wc[q] = wr[c2r[q]];
}

#endif // __CUDACC__

// ===========================================================================
// CPU (OpenMP) versions. schedule(static) is optimal here BECAUSE the
// margins are constant: every row is identical work, so the static split
// is already perfectly balanced. Same symmetry as the GPU side:
//   forward  : ell_gather_cpu(colidx, wr, x,  y,  M, R)
//   backward : ell_gather_cpu(rowidx, wc, dy, dx, N, C)
// ===========================================================================

inline void ell_gather_cpu(const int32_t* idx, const float* w,
                           const float* in, float* out,
                           int rows, int width)
{
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        const int64_t base = (int64_t)row * width;
        float acc = 0.f;
        #pragma omp simd reduction(+:acc)
        for (int s = 0; s < width; ++s) {
            const int32_t v    = idx[base + s];
            const float   live = (v >= 0) ? 1.f : 0.f;    // branchless mask
            acc += live * w[base + s] * in[v & 0x7fffffff];
        }
        out[row] = acc;
    }
    // The gather in[...] is the entire cost; AVX2/AVX-512 vgatherdps eats
    // it 8 or 16 lanes at a time when the compiler is in the mood.
}

inline void ell_push_forward_cpu(const int32_t* rowidx, const float* wc,
                                 const int32_t* active, const float* xa,
                                 float* out, int nActive, int C)
{
    // At low activity a serial pass often wins (no atomics). Parallel
    // alternative to atomics: private out per thread, reduce after.
    #pragma omp parallel for schedule(static)
    for (int a = 0; a < nActive; ++a) {
        const int64_t base = (int64_t)active[a] * C;
        const float   xv   = xa[a];
        for (int t = 0; t < C; ++t) {
            const int32_t v = rowidx[base + t];
            if (v < 0) continue;
            #pragma omp atomic
            out[v] += wc[base + t] * xv;
        }
    }
}

inline void ell_push_backward_cpu(const int32_t* colidx, float* wr,
                                  const int32_t* activeOut, const float* dyA,
                                  const float* x, float* dx,
                                  float lr, int nActive, int R)
{
    // activeOut rows are distinct -> wr writes race-free; dx needs atomics.
    #pragma omp parallel for schedule(static)
    for (int a = 0; a < nActive; ++a) {
        const int     i    = activeOut[a];
        const float   g    = dyA[a];
        const int64_t base = (int64_t)i * R;
        for (int s = 0; s < R; ++s) {
            const int32_t v = colidx[base + s];
            if (v < 0) continue;
            const int32_t j = v & 0x7fffffff;
            const float   w = wr[base + s];
            #pragma omp atomic
            dx[j] += w * g;
            wr[base + s] = w - lr * g * x[j];
        }
    }
}

inline void ell_sync_window_cpu(const int32_t* c2r, const float* wr,
                                float* wc, int32_t start, int32_t count,
                                int32_t NC)
{
    const int32_t end = (start + count < NC) ? start + count : NC;
    #pragma omp parallel for schedule(static)
    for (int32_t q = start; q < end; ++q) wc[q] = wr[c2r[q]];
}

} // namespace fme
