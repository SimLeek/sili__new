#version 450
// banked_ell.comp.glsl
// ===========================================================================
// Doubly-regular sparse weights at 32 bits per parameter, everything O(1).
// Vulkan GLSL, no subgroup intrinsics, no float atomics, no varints.
//
// Budget, per synapse:
//              was (fixed_margin_ell.cuh)        now
//   weights    2 x fp32            = 64 bits     fp4 + 4b importance =  8
//   indices    2 x int32           = 64 bits     12b pi + 12b inv    = 24
//   perms      r2c + c2r           = 64 bits     closed form         =  0
//   sync       wc copy + cursor                  gone (one weight store)
//   total      192 bits                          32 bits
//
// The structure: banked / quasi-cyclic.
//   i = u*C + p     u = i >> C_LOG    p = i & (C-1)    p: row residue
//   j = v*R + s     v = j >> R_LOG    s = j & (R-1)    s: column residue
// Invariant: for every residue pair (p, s), the synapses between row
// class p and column class s form a PERFECT MATCHING between u in [0,S)
// and v in [0,S), where S = M/C = N/R (equal because M*R == N*C).
// The whole fabric is R*C independent permutations of [0,S):
//   pi [i*R + s] = v     row view:    where slot s of row i points
//   inv[j*C + p] = u     column view: who occupies slot p of column j
// Slot addresses are arithmetic, so nothing is searched and nothing is
// synced: forward, backward, and updates all address the single weight
// array directly.
//
// Banks are interleaved residue classes (bank s = columns s, s+R, s+2R,
// ...), so "one synapse per bank per row" still allows a row to fully
// connect any R consecutive inputs. Honest constraints:
//   - a row cannot hold two synapses whose columns agree mod R
//   - trades act within one (p, s) plane: partner rows agree mod C,
//     partner columns agree mod R
// Reachable family: (S!)^(R*C) wirings. Not a small room.
//
// Word layouts (LOG_S = 12 shown; fetch/store handle straddles):
//
//   pi_bits / inv_bits: 12-bit fields packed tight
//     word t:   [ f2 low bits | field f1  | field f0  ]
//                31         24 23       12 11        0
//
//   syn: one byte per synapse, column-major, 4 per word
//     [ imp | w4 ][ imp | w4 ][ imp | w4 ][ imp | w4 ]
//        byte 3      byte 2      byte 1      byte 0
//
// w4 code 0 decodes to exactly 0.0 and doubles as the silent mark:
// silent capacity contributes nothing (no branch required) and the
// update stage skips bytes with imp == 0, so preallocated slots sleep
// until grown. Growth is writing one byte.
//
// pi and inv are S-entry permutations, R*C of them, packed shoulder to
// shoulder. Rewiring is a transposition. FPGA people call this routing,
// LDPC people call it quasi-cyclic, the synapses just call it home.
//
// Stages -- compile one per pipeline:
//   glslangValidator -V --target-env vulkan1.1 -DSTAGE_INIT -o init.spv banked_ell.comp.glsl
//   STAGE_INIT      build shifted-identity planes, zero syn
//   STAGE_FORWARD   push: one workgroup (size C) per active input
//   STAGE_DX        push: one workgroup (size R) per active output grad
//   STAGE_WUPDATE   weight + importance update, per active input column
//   STAGE_TRADE     commit a batch of curveball transpositions
//   STAGE_TOFLOAT   fixed-point accumulator -> float, and clear
//
// Portability notes: every invocation is an independent little machine
// (no subgroup ops), so this runs identically at subgroup size 8..64
// and maps to an FPGA as C or R parallel lanes with banked BRAM.
// Accumulators are integer fixed point via atomicAdd(int) -- core GLSL,
// deterministic, and exactly what a DSP block wants anyway.
// ===========================================================================

layout(constant_id = 0) const uint LOG_S    = 12u; // S = 1 << LOG_S
layout(constant_id = 1) const uint C_LOG    = 5u;  // C = 1 << C_LOG per column
layout(constant_id = 2) const uint R_LOG    = 5u;  // R = 1 << R_LOG per row
layout(constant_id = 3) const int  FIX_SHIFT = 14; // accumulator fixed point
layout(constant_id = 5) const float W_SCALE = 0.0625; // codebook scale

// Derived: M = S*C outputs, N = S*R inputs. Pad real sizes up to these.
// Padded inputs never enter active lists, so their columns never fire
// and never train; padded outputs accumulate values nobody reads.
// Cheapest kind of wrong.

#define S_SIZE (1u << LOG_S)
#define S_MASK (S_SIZE - 1u)
#define C_SZ   (1u << C_LOG)
#define R_SZ   (1u << R_LOG)

layout(push_constant) uniform Push {
    uint  count;  // active list length, or trade count
    float lr;
    uint  seed;   // stochastic rounding salt, change per step
} pc;

// fp4 codebook, E2M1-flavored: value = W_LUT[code] * W_SCALE.
// Code 0 MUST be 0.0 (it is the silent mark). Codebook is yours to swap.
const float W_LUT[16] = float[16](
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
);
// Codes in ascending value order, for stochastic rounding brackets.
const uint SORT_CODE[16] = uint[16](
    15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u, 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u
);

// ---------------------------------------------------------------------------
// Buffers per stage. Bindings: 0 pi, 1 inv, 2 syn, 3 list, 4 values, 5 aux.
// ---------------------------------------------------------------------------
#if defined(STAGE_INIT) || defined(STAGE_DX) || defined(STAGE_TRADE)
layout(std430, binding = 0) buffer PiB { uint pi_bits[]; };
#endif
#if defined(STAGE_INIT) || defined(STAGE_FORWARD) || defined(STAGE_WUPDATE) || defined(STAGE_TRADE)
layout(std430, binding = 1) buffer InvB { uint inv_bits[]; };
#endif
#if defined(STAGE_INIT) || defined(STAGE_FORWARD) || defined(STAGE_DX) || defined(STAGE_WUPDATE) || defined(STAGE_TRADE)
layout(std430, binding = 2) buffer SynB { uint syn[]; };
#endif

// ---------------------------------------------------------------------------
// Packed 12-bit field access. Reads may straddle two words; stores in the
// TRADE stage use atomicAnd/atomicOr so trades sharing a word (but never
// a field -- the round builder guarantees disjoint (p,s,u)) compose.
// ---------------------------------------------------------------------------
#if defined(STAGE_DX) || defined(STAGE_TRADE)
uint pi_get(uint f) {
    uint bit = f * LOG_S;
    uint w = bit >> 5u, sh = bit & 31u;
    uint v = pi_bits[w] >> sh;
    if (sh + LOG_S > 32u) v |= pi_bits[w + 1u] << (32u - sh);
    return v & S_MASK;
}
#endif
#if defined(STAGE_FORWARD) || defined(STAGE_WUPDATE) || defined(STAGE_TRADE)
uint inv_get(uint f) {
    uint bit = f * LOG_S;
    uint w = bit >> 5u, sh = bit & 31u;
    uint v = inv_bits[w] >> sh;
    if (sh + LOG_S > 32u) v |= inv_bits[w + 1u] << (32u - sh);
    return v & S_MASK;
}
#endif

// ===========================================================================
#ifdef STAGE_INIT
// Shifted-identity plane (p, s): v = (u + p + s*C) mod S. Its inverse:
// u = (v - p - s*C) mod S. Any per-plane shifts work; trades will take
// the structure wherever the importance scores want it to go.
// Each invocation owns whole OUTPUT WORDS (packed fields race otherwise)
// and computes every field overlapping its word.
// Dispatch: ceil(max(bit_words(nnz), nnz/4) / 256) groups.
// ===========================================================================
layout(local_size_x = 256) in;

uint pi_value(uint k) {
    uint i = k >> R_LOG;
    uint s = k & (R_SZ - 1u);
    uint u = i >> C_LOG;
    uint p = i & (C_SZ - 1u);
    return (u + p + s * C_SZ) & S_MASK;
}
uint inv_value(uint q) {
    uint j = q >> C_LOG;
    uint p = q & (C_SZ - 1u);
    uint v = j >> R_LOG;
    uint s = j & (R_SZ - 1u);
    return (v - p - s * C_SZ) & S_MASK;
}

void main() {
    uint w = gl_GlobalInvocationID.x;
    uint nnz = S_SIZE << (C_LOG + R_LOG);          // S*C == S*R slot counts match
    if (w < (nnz >> 2u)) syn[w] = 0u;              // w4=0, imp=0: silent
    uint nWords = (nnz * LOG_S + 31u) >> 5u;
    if (w >= nWords) return;
    uint firstBit = w << 5u;
    uint accPi = 0u, accInv = 0u;
    for (uint f = firstBit / LOG_S; f < nnz && f * LOG_S < firstBit + 32u; ++f) {
        int sh = int(f * LOG_S) - int(firstBit);
        if (sh >= 0) {
            accPi  |= pi_value(f)  << uint(sh);
            accInv |= inv_value(f) << uint(sh);
        } else {
            accPi  |= pi_value(f)  >> uint(-sh);
            accInv |= inv_value(f) >> uint(-sh);
        }
    }
    pi_bits[w]  = accPi;
    inv_bits[w] = accInv;
}
#endif

// ===========================================================================
#ifdef STAGE_FORWARD
// Event-driven forward, "weights as the column view". One workgroup of
// size C per active input j; invocation p handles column slot p:
//   i = (inv[j*C+p] << C_LOG) | p;   out[i] += w * x[j]
// Contiguous inv and syn reads, uniform control flow, silent slots
// contribute zero without branching.
// Dispatch: pc.count groups. Host sets local size spec (id 4) = C.
// ===========================================================================
layout(local_size_x_id = 4) in;
layout(std430, binding = 3) readonly buffer ActL { uint  act_idx[]; };
layout(std430, binding = 4) readonly buffer ActV { float act_val[]; };
layout(std430, binding = 5) buffer OutA { int out_acc[]; };   // M entries

void main() {
    uint a = gl_WorkGroupID.x;
    if (a >= pc.count) return;
    uint  j  = act_idx[a];
    float xv = act_val[a];
    uint  p  = gl_LocalInvocationID.x;
    uint  q  = (j << C_LOG) + p;
    uint  u  = inv_get(q);
    uint  i  = (u << C_LOG) | p;
    uint  code = (syn[q >> 2u] >> ((q & 3u) * 8u)) & 15u;
    float w  = W_LUT[code] * W_SCALE;
    int contrib = int(round(w * xv * float(1 << FIX_SHIFT)));
    if (contrib != 0) atomicAdd(out_acc[i], contrib);
}
#endif

// ===========================================================================
#ifdef STAGE_DX
// Event-driven backward, "weights as the row view". One workgroup of
// size R per active output grad row i; invocation s handles row slot s:
//   j = (pi[i*R+s] << R_LOG) | s;  q = j*C + (i mod C);  dx[j] += w * dy[i]
// pi reads contiguous; the weight read is the one scattered access in
// the whole system, and it is read-only. This stage never writes syn.
// Dispatch: pc.count groups. Host sets local size spec (id 4) = R.
// ===========================================================================
layout(local_size_x_id = 4) in;
layout(std430, binding = 3) readonly buffer ActL { uint  act_idx[]; }; // active rows
layout(std430, binding = 4) readonly buffer ActV { float act_val[]; }; // their dy
layout(std430, binding = 5) buffer DxA { int dx_acc[]; };             // N entries

void main() {
    uint a = gl_WorkGroupID.x;
    if (a >= pc.count) return;
    uint  i = act_idx[a];
    float g = act_val[a];
    uint  s = gl_LocalInvocationID.x;
    uint  k = (i << R_LOG) + s;
    uint  v = pi_get(k);
    uint  j = (v << R_LOG) | s;
    uint  q = (j << C_LOG) + (i & (C_SZ - 1u));
    uint  code = (syn[q >> 2u] >> ((q & 3u) * 8u)) & 15u;
    float w = W_LUT[code] * W_SCALE;
    int contrib = int(round(w * g * float(1 << FIX_SHIFT)));
    if (contrib != 0) atomicAdd(dx_acc[j], contrib);
}
#endif

// ===========================================================================
#ifdef STAGE_WUPDATE
// Weight + importance update, driven by ACTIVE INPUT columns (grad-like
// rules give zero update where x[j] == 0, so only active columns need
// touching). One invocation per active column owns that column's C/4
// syn words outright: whole-word read-modify-write, no atomics, no
// races, because column blocks are word aligned (C multiple of 4).
// Updates land here once per synapse per step; there is no second copy
// to reconcile, which is the entire retirement plan of the sync cursor.
// Dispatch: ceil(pc.count / 64) groups.
// ===========================================================================
layout(local_size_x = 64) in;
layout(std430, binding = 3) readonly buffer ActL { uint  act_idx[]; };
layout(std430, binding = 4) readonly buffer ActV { float act_val[]; };
layout(std430, binding = 5) readonly buffer DyB  { float dy[]; };     // M entries

uint wang(uint x) {
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x ^= x >> 4u;
    x *= 0x27d4eb2du;
    x ^= x >> 15u;
    return x;
}

// Stochastic rounding to the codebook: land on the bracketing code
// above or below with probability proportional to proximity, so tiny
// fp4-invisible gradients still move weights in expectation. This is
// what makes 4-bit ONLINE training breathe.
uint quantize(float t, uint rnd) {
    float lo = W_LUT[SORT_CODE[0]];
    if (t <= lo) return SORT_CODE[0];
    for (uint z = 1u; z < 16u; ++z) {
        float hi = W_LUT[SORT_CODE[z]];
        if (t <= hi) {
            float fr = (t - lo) / max(hi - lo, 1e-9);
            return (float(rnd & 0xffffu) * (1.0 / 65536.0) < fr)
                 ? SORT_CODE[z] : SORT_CODE[z - 1u];
        }
        lo = hi;
    }
    return SORT_CODE[15];
}

void main() {
    uint a = gl_GlobalInvocationID.x;
    if (a >= pc.count) return;
    uint  j  = act_idx[a];
    float xv = act_val[a];
    uint  qbase = j << C_LOG;
    for (uint ww = 0u; ww < (C_SZ >> 2u); ++ww) {
        uint word = syn[(qbase >> 2u) + ww];
        uint outw = 0u;
        for (uint b = 0u; b < 4u; ++b) {
            uint byv  = (word >> (b * 8u)) & 255u;
            uint code = byv & 15u;
            uint imp  = byv >> 4u;
            if (imp == 0u && code == 0u) {        // sleeping capacity
                outw |= byv << (b * 8u);
                continue;
            }
            uint q = qbase + (ww << 2u) + b;
            uint u = inv_get(q);
            uint i = (u << C_LOG) | (q & (C_SZ - 1u));
            float g    = dy[i];
            float wcur = W_LUT[code] * W_SCALE;
            float wnew = wcur - pc.lr * g * xv;
            uint  nc   = quantize(wnew / W_SCALE, wang(q ^ pc.seed));
            // Importance: bump when the step actually crossed a code
            // boundary. Swap for your rule; decay belongs in a periodic
            // pass or at trade-selection time (your bottom-k machinery).
            if (nc != code && imp < 15u) imp += 1u;
            outw |= ((imp << 4u) | nc) << (b * 8u);
        }
        syn[(qbase >> 2u) + ww] = outw;
    }
}
#endif

// ===========================================================================
#ifdef STAGE_TRADE
// Commit curveball transpositions. Trade record (p, s, u1, u2): within
// plane (p, s), rows i1 = u1*C + p and i2 = u2*C + p swap their columns
// j1, j2 (both congruent to s mod R). Four packed-field writes, two
// weight bytes zeroed, O(1), margins never break even transiently.
// The round builder guarantees records touch disjoint (p, s, u), so
// atomicAnd/atomicOr on shared straddle words compose safely. Run this
// stage with a barrier before/after; do not overlap with training.
// Selection (importance bottom-k, symmetric-difference set ops) stays
// in your existing machinery; this stage only commits.
// Dispatch: ceil(pc.count / 64) groups.
// ===========================================================================
layout(local_size_x = 64) in;
layout(std430, binding = 3) readonly buffer TrL { uvec4 trades[]; };

void pi_put(uint f, uint val) {
    uint bit = f * LOG_S;
    uint w = bit >> 5u, sh = bit & 31u;
    atomicAnd(pi_bits[w], ~(S_MASK << sh));
    atomicOr (pi_bits[w], val << sh);
    if (sh + LOG_S > 32u) {
        uint hb = sh + LOG_S - 32u;               // bits spilling into w+1
        atomicAnd(pi_bits[w + 1u], ~((1u << hb) - 1u));
        atomicOr (pi_bits[w + 1u], val >> (32u - sh));
    }
}
void inv_put(uint f, uint val) {
    uint bit = f * LOG_S;
    uint w = bit >> 5u, sh = bit & 31u;
    atomicAnd(inv_bits[w], ~(S_MASK << sh));
    atomicOr (inv_bits[w], val << sh);
    if (sh + LOG_S > 32u) {
        uint hb = sh + LOG_S - 32u;
        atomicAnd(inv_bits[w + 1u], ~((1u << hb) - 1u));
        atomicOr (inv_bits[w + 1u], val >> (32u - sh));
    }
}
void syn_zero(uint q) {
    atomicAnd(syn[q >> 2u], ~(255u << ((q & 3u) * 8u)));
}

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= pc.count) return;
    uvec4 tr = trades[t];
    uint p = tr.x, s = tr.y, u1 = tr.z, u2 = tr.w;
    uint i1 = (u1 << C_LOG) | p;
    uint i2 = (u2 << C_LOG) | p;
    uint k1 = (i1 << R_LOG) + s;
    uint k2 = (i2 << R_LOG) + s;
    uint v1 = pi_get(k1);
    uint v2 = pi_get(k2);
    uint j1 = (v1 << R_LOG) | s;
    uint j2 = (v2 << R_LOG) | s;
    pi_put(k1, v2);
    pi_put(k2, v1);
    inv_put((j1 << C_LOG) + p, u2);
    inv_put((j2 << C_LOG) + p, u1);
    syn_zero((j1 << C_LOG) + p);                  // relocated synapses are
    syn_zero((j2 << C_LOG) + p);                  // newborn (and silent
}                                                 // until imp is bumped)
#endif

// ===========================================================================
#ifdef STAGE_TOFLOAT
// Fixed point -> float, clearing the accumulator for the next step.
// Dispatch over the accumulator length (binding 3 in, 4 out).
// ===========================================================================
layout(local_size_x = 256) in;
layout(std430, binding = 3) buffer AccB { int   acc[]; };
layout(std430, binding = 4) buffer OutF { float outf[]; };

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= pc.count) return;
    outf[t] = float(acc[t]) * (1.0 / float(1 << FIX_SHIFT));
    acc[t]  = 0;
}
#endif
