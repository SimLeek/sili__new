#version 450
// dual_codec_ell.comp.glsl
// ===========================================================================
// Gather-mode sparse training kernels, atomic-free on every training pass.
// One weight store (row-major), two index views, codec chosen per layer at
// pipeline-compile time:
//
//   -DCODEC_BANKED   closed-form quasi-cyclic planes, 2*LOG_S+8 bits/param
//   -DCODEC_PACKED   anchor + fixed-width deltas in groups of 32,
//                    measured bits/param (wins when spans are clustered)
//
// Stages (compile one per pipeline):
//   -DSTAGE_FORWARD   y[i]  = sum_s w[i,s] * x[col(i,s)]     workgroup/row
//   -DSTAGE_DX        dx[j] = sum_p w[row(j,p),j] * dy[...]  workgroup/col
//   -DSTAGE_WUPDATE   w -= lr * dy[i] * x[j], active-dy rows, row-exclusive
//   -DSTAGE_TRADE     banked only: curveball transpositions (barriered)
//
// No atomics in FORWARD / DX / WUPDATE: single writer per output element,
// single owner per weight word. TRADE uses masked atomicAnd/atomicOr on
// packed fields and never overlaps training.
//
// Relabeling (both codecs share the code path; packed sets a* = 1):
//   stored col key jp = (aC * j) & NMASK      j = (aCinv * jp) & NMASK
//   stored row key ip = (aR * i) & MMASK      i = (aRinv * ip) & MMASK
// BANKED splits keys as   jp = [ s : R_LOG | v : LOG_S ]
//                         ip = [ p : C_LOG | u : LOG_S ]
// Banks come from the HIGH bits of the multiplicative hash, so any input
// stride (rasters included) lands in pseudorandom banks. Invariant: the
// synapses of plane (p, s) form a perfect matching u <-> v; weight slot
// of (ip, s) is ip*R + s; of (jp, p) is jp*C + p. Nothing is searched.
//
// PACKED layout per direction (groups of 32 sorted ids):
//   hdr[g]  = anchor << 5 | bitwidth        (anchor <= 27 bits)
//   off[g]  = bit offset of the group's deltas in stream[]
//   stream  = tightly packed fixed-width deltas from the group anchor
//   slotback[q] (column view only, R_LOG-bit fields): the row-slot s of
//   the synapse at column slot q, so DX can address the row-major weight
//   without searching. Costs R_LOG bits/param; delete it and binary
//   search the row group instead if those bits matter more than the hop.
//
// syn[] is one byte per synapse, [imp:4 | w4:4], row-major in STORED row
// order (ip for banked, i for packed). Code 0 decodes to 0.0 and marks
// silence; silent slots are skipped before any gather, so x / dy / dx
// only ever need their real extents (dx sized by push.nReal guard).
//
// Bindings by stage (ro unless noted):
//   FORWARD  banked: 0 pi_bits          packed: 0 hdr, 1 off, 6 stream
//            both:   2 syn, 3 x, 4 xbits, 5 y (wo)
//   DX       banked: 0 inv_bits         packed: 0 hdr, 1 off, 6 stream,
//                                               7 slotback
//            both:   2 syn, 3 dy, 4 dybits, 5 dx (wo)
//   WUPDATE  banked: 0 pi_bits          packed: 0 hdr, 1 off, 6 stream
//            both:   2 syn (rw), 3 act_rows, 4 act_dy, 8 x, 9 xbits
//   TRADE    banked: 0 pi_bits (rw), 1 inv_bits (rw), 2 syn (rw), 3 trades
//
// Dispatch: FORWARD count = real output rows; DX count = column keys
// (banked: 1 << (LOG_S + R_LOG), packed: real columns); WUPDATE count =
// active rows / 64; TRADE count = trades / 64. Local sizes: FORWARD = R,
// DX = C via spec id 4 (power of two, <= 128); WUPDATE, TRADE fixed 64.
// ===========================================================================

layout(constant_id = 0) const uint LOG_S  = 12u;
layout(constant_id = 1) const uint C_LOG  = 5u;
layout(constant_id = 2) const uint R_LOG  = 5u;
layout(constant_id = 5) const float W_SCALE = 0.0625;

#define R_SZ   (1u << R_LOG)
#define C_SZ   (1u << C_LOG)
#define S_MASK ((1u << LOG_S) - 1u)
#define KC     (LOG_S + R_LOG)
#define KR     (LOG_S + C_LOG)
#define NMASK  ((1u << KC) - 1u)
#define MMASK  ((1u << KR) - 1u)

layout(push_constant) uniform Push {
    uint  count;   // rows / column keys / active rows / trades
    float lr;
    uint  seed;
    uint  aC;      // packed layers: all four multipliers are 1
    uint  aCinv;
    uint  aR;
    uint  aRinv;
    uint  nReal;   // real column count, guards padded dx writes
    float lrImp;   // 0: legacy importance bump; >0: stochastic importance
} pc;

const float W_LUT[16] = float[16](
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
);
const uint SORT_CODE[16] = uint[16](
    15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u, 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u
);

// Generic bit-field fetch (fields never span more than two words).
#define DEF_FETCH(fn, arr) \
uint fn(uint bitpos, uint width) { \
    uint w = bitpos >> 5u, sh = bitpos & 31u; \
    uint v = arr[w] >> sh; \
    if (sh + width > 32u) v |= arr[w + 1u] << (32u - sh); \
    return v & ((1u << width) - 1u); \
}

// ===========================================================================
#ifdef STAGE_FORWARD
layout(local_size_x_id = 4) in;                      // = R_SZ
#ifdef CODEC_BANKED
layout(std430, binding = 0) readonly buffer B0 { uint pi_bits[]; };
DEF_FETCH(pi_f, pi_bits)
#else
layout(std430, binding = 0) readonly buffer B0 { uint col_hdr[]; };
layout(std430, binding = 1) readonly buffer B1 { uint col_off[]; };
layout(std430, binding = 6) readonly buffer B6 { uint col_stream[]; };
DEF_FETCH(cs_f, col_stream)
#endif
layout(std430, binding = 2) readonly buffer B2 { uint  syn[]; };
layout(std430, binding = 3) readonly buffer B3 { float x[]; };
layout(std430, binding = 4) readonly buffer B4 { uint  xbits[]; };
layout(std430, binding = 5) writeonly buffer B5 { float y[]; };

shared float red[128];

void main() {
    uint i = gl_WorkGroupID.x;                       // real output row
    uint s = gl_LocalInvocationID.x;
    float part = 0.0;
    if (i < pc.count) {
#ifdef CODEC_BANKED
        uint rowkey = (pc.aR * i) & MMASK;
        uint k = rowkey * R_SZ + s;
        uint code = (syn[k >> 2u] >> ((k & 3u) * 8u)) & 15u;
        if (code != 0u) {
            uint v  = pi_f(k * LOG_S, LOG_S);
            uint jp = (s << LOG_S) | v;
            uint j  = (pc.aCinv * jp) & NMASK;
#else
        uint k = i * R_SZ + s;
        uint code = (syn[k >> 2u] >> ((k & 3u) * 8u)) & 15u;
        if (code != 0u) {
            uint gb  = i * (R_SZ >> 5u) + (s >> 5u);
            uint hdr = col_hdr[gb];
            uint wd  = hdr & 31u;
            uint j   = (hdr >> 5u) + cs_f(col_off[gb] + (s & 31u) * wd, wd);
#endif
            if (((xbits[j >> 5u] >> (j & 31u)) & 1u) != 0u)
                part = W_LUT[code] * W_SCALE * x[j];
        }
    }
    uint lid = gl_LocalInvocationID.x;
    red[lid] = part;
    barrier();
    for (uint h = gl_WorkGroupSize.x >> 1u; h > 0u; h >>= 1u) {
        if (lid < h) red[lid] += red[lid + h];
        barrier();
    }
    if (lid == 0u && i < pc.count) y[i] = red[0];
}
#endif

// ===========================================================================
#ifdef STAGE_DX
layout(local_size_x_id = 4) in;                      // = C_SZ
#ifdef CODEC_BANKED
layout(std430, binding = 0) readonly buffer B0 { uint inv_bits[]; };
DEF_FETCH(inv_f, inv_bits)
#else
layout(std430, binding = 0) readonly buffer B0 { uint row_hdr[]; };
layout(std430, binding = 1) readonly buffer B1 { uint row_off[]; };
layout(std430, binding = 6) readonly buffer B6 { uint row_stream[]; };
layout(std430, binding = 7) readonly buffer B7 { uint slotback[]; };
DEF_FETCH(rs_f, row_stream)
DEF_FETCH(sb_f, slotback)
#endif
layout(std430, binding = 2) readonly buffer B2 { uint  syn[]; };
layout(std430, binding = 3) readonly buffer B3 { float dy[]; };
layout(std430, binding = 4) readonly buffer B4 { uint  dybits[]; };
layout(std430, binding = 5) writeonly buffer B5 { float dx[]; };

shared float red[128];

void main() {
    uint jc = gl_WorkGroupID.x;                      // column key
    uint p  = gl_LocalInvocationID.x;
    float part = 0.0;
    uint jr = jc;
    if (jc < pc.count) {
        uint q = jc * C_SZ + p;
#ifdef CODEC_BANKED
        jr = (pc.aCinv * jc) & NMASK;                // real column
        uint u  = inv_f(q * LOG_S, LOG_S);
        uint ip = (p << LOG_S) | u;
        uint k  = ip * R_SZ + (jc >> LOG_S);
        uint code = (syn[k >> 2u] >> ((k & 3u) * 8u)) & 15u;
        if (code != 0u) {
            uint i = (pc.aRinv * ip) & MMASK;
#else
        uint gb  = jc * (C_SZ >> 5u) + (p >> 5u);
        uint hdr = row_hdr[gb];
        uint wd  = hdr & 31u;
        uint i   = (hdr >> 5u) + rs_f(row_off[gb] + (p & 31u) * wd, wd);
        uint k   = i * R_SZ + sb_f(q * R_LOG, R_LOG);
        uint code = (syn[k >> 2u] >> ((k & 3u) * 8u)) & 15u;
        if (code != 0u) {
#endif
            if (((dybits[i >> 5u] >> (i & 31u)) & 1u) != 0u)
                part = W_LUT[code] * W_SCALE * dy[i];
        }
    }
    uint lid = gl_LocalInvocationID.x;
    red[lid] = part;
    barrier();
    for (uint h = gl_WorkGroupSize.x >> 1u; h > 0u; h >>= 1u) {
        if (lid < h) red[lid] += red[lid + h];
        barrier();
    }
    if (lid == 0u && jc < pc.count && jr < pc.nReal) dx[jr] = red[0];
}
#endif

// ===========================================================================
#ifdef STAGE_WUPDATE
layout(local_size_x = 64) in;
#ifdef CODEC_BANKED
layout(std430, binding = 0) readonly buffer B0 { uint pi_bits[]; };
DEF_FETCH(pi_f, pi_bits)
#else
layout(std430, binding = 0) readonly buffer B0 { uint col_hdr[]; };
layout(std430, binding = 1) readonly buffer B1 { uint col_off[]; };
layout(std430, binding = 6) readonly buffer B6 { uint col_stream[]; };
DEF_FETCH(cs_f, col_stream)
#endif
layout(std430, binding = 2) buffer B2 { uint syn[]; };
layout(std430, binding = 3) readonly buffer B3 { uint  act_rows[]; };
layout(std430, binding = 4) readonly buffer B4 { float act_dy[]; };
layout(std430, binding = 8) readonly buffer B8 { float x[]; };
layout(std430, binding = 9) readonly buffer B9 { uint  xbits[]; };

uint wang(uint h) {
    h = (h ^ 61u) ^ (h >> 16u);
    h *= 9u;
    h ^= h >> 4u;
    h *= 0x27d4eb2du;
    h ^= h >> 15u;
    return h;
}

// Stochastic rounding: land on a bracketing code with probability
// proportional to proximity, so sub-LSB gradients still move weights
// in expectation. This is what lets fp4 train online.
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
    uint  i = act_rows[a];                            // real row
    float g = act_dy[a];
#ifdef CODEC_BANKED
    uint rowkey = (pc.aR * i) & MMASK;
#else
    uint rowkey = i;
#endif
    uint kbase = rowkey * R_SZ;
    for (uint ww = 0u; ww < (R_SZ >> 2u); ++ww) {
        uint word = syn[(kbase >> 2u) + ww];
        uint outw = 0u;
        for (uint b = 0u; b < 4u; ++b) {
            uint byv  = (word >> (b * 8u)) & 255u;
            uint code = byv & 15u;
            uint imp  = byv >> 4u;
            uint s    = (ww << 2u) + b;
            if (imp == 0u && code == 0u) {            // sleeping capacity
                outw |= byv << (b * 8u);
                continue;
            }
#ifdef CODEC_BANKED
            uint k  = kbase + s;
            uint v  = pi_f(k * LOG_S, LOG_S);
            uint jp = (s << LOG_S) | v;
            uint j  = (pc.aCinv * jp) & NMASK;
#else
            uint gb  = i * (R_SZ >> 5u) + (s >> 5u);
            uint hdr = col_hdr[gb];
            uint wd  = hdr & 31u;
            uint j   = (hdr >> 5u) + cs_f(col_off[gb] + (s & 31u) * wd, wd);
#endif
            float xv = (((xbits[j >> 5u] >> (j & 31u)) & 1u) != 0u)
                     ? x[j] : 0.0;
            float upd = pc.lr * g * xv;
            if (pc.lrImp == 0.0) {                    // legacy importance
                if (upd == 0.0) {                     // zero step: pass the
                    outw |= byv << (b * 8u);          // byte through; re-
                    continue;                         // rounding a parked
                }                                     // weight random-walks it
                float wnew = W_LUT[code] * W_SCALE - upd;
                uint  nc   = quantize(wnew / W_SCALE, wang((kbase + s) ^ pc.seed));
                if (nc != code && imp < 15u) imp += 1u;
                outw |= ((imp << 4u) | nc) << (b * 8u);
            } else {                                  // stochastic importance:
                if (xv == 0.0) {                      // one signed dithered
                    outw |= byv << (b * 8u);          // step of
                    continue;                         // lrImp*|x|*(|w|-|dy|)
                }
                uint nc = code;
                if (upd != 0.0) {
                    float wnew = W_LUT[code] * W_SCALE - upd;
                    nc = quantize(wnew / W_SCALE, wang((kbase + s) ^ pc.seed));
                }
                float d  = pc.lrImp * (abs(W_LUT[code] * W_SCALE * xv) - abs(g * xv));
                float ad = abs(d);
                uint  rn = wang((kbase + s) ^ pc.seed ^ 0x51ed270bu);
                uint  inc = uint(ad);
                if (float(rn & 0xffffu) * (1.0 / 65536.0) < ad - float(inc)) inc += 1u;
                int   iv = int(imp) + ((d >= 0.0) ? int(inc) : -int(inc));
                int   lb = (nc != 0u) ? 1 : 0;
                imp = uint(clamp(iv, lb, 15));
                outw |= ((imp << 4u) | nc) << (b * 8u);
            }
        }
        syn[(kbase >> 2u) + ww] = outw;
    }
}
#endif

// ===========================================================================
#ifdef STAGE_TRADE
// Banked only. Packed layers rewire on the CPU by re-encoding groups into
// their interleaved slack space, exactly as before. Record (p, s, u1, u2):
// rows ip1, ip2 of plane (p, s) swap partners. Disjoint (p, s, u) per
// batch; masked atomics let straddling neighbors coexist. Barrier this
// stage against training; it is the only atomic user left in the file.
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer B0 { uint pi_bits[]; };
layout(std430, binding = 1) buffer B1 { uint inv_bits[]; };
layout(std430, binding = 2) buffer B2 { uint syn[]; };
layout(std430, binding = 3) readonly buffer B3 { uvec4 trades[]; };
DEF_FETCH(pi_f, pi_bits)

void pi_put(uint f, uint val) {
    uint bit = f * LOG_S, w = bit >> 5u, sh = bit & 31u;
    atomicAnd(pi_bits[w], ~(S_MASK << sh));
    atomicOr (pi_bits[w], val << sh);
    if (sh + LOG_S > 32u) {
        uint hb = sh + LOG_S - 32u;
        atomicAnd(pi_bits[w + 1u], ~((1u << hb) - 1u));
        atomicOr (pi_bits[w + 1u], val >> (32u - sh));
    }
}
void inv_put(uint f, uint val) {
    uint bit = f * LOG_S, w = bit >> 5u, sh = bit & 31u;
    atomicAnd(inv_bits[w], ~(S_MASK << sh));
    atomicOr (inv_bits[w], val << sh);
    if (sh + LOG_S > 32u) {
        uint hb = sh + LOG_S - 32u;
        atomicAnd(inv_bits[w + 1u], ~((1u << hb) - 1u));
        atomicOr (inv_bits[w + 1u], val >> (32u - sh));
    }
}
void syn_zero(uint k) {
    atomicAnd(syn[k >> 2u], ~(255u << ((k & 3u) * 8u)));
}

void main() {
    uint t = gl_GlobalInvocationID.x;
    if (t >= pc.count) return;
    uvec4 tr = trades[t];
    uint p = tr.x, s = tr.y, u1 = tr.z, u2 = tr.w;
    uint ip1 = (p << LOG_S) | u1;
    uint ip2 = (p << LOG_S) | u2;
    uint k1 = ip1 * R_SZ + s;
    uint k2 = ip2 * R_SZ + s;
    uint v1 = pi_f(k1 * LOG_S, LOG_S);
    uint v2 = pi_f(k2 * LOG_S, LOG_S);
    pi_put(k1, v2);
    pi_put(k2, v1);
    uint jp1 = (s << LOG_S) | v1;
    uint jp2 = (s << LOG_S) | v2;
    inv_put(jp1 * C_SZ + p, u2);
    inv_put(jp2 * C_SZ + p, u1);
    syn_zero(k1);                                     // relocated synapses
    syn_zero(k2);                                     // are newborn
}
#endif
