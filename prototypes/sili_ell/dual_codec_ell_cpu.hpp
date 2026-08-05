// dual_codec_ell_cpu.hpp
// ===========================================================================
// CPU translation of dual_codec_ell.comp.glsl, one function per stage per
// codec, mapped as the GPU code would be:
//
//   workgroup dimension  ->  #pragma omp parallel for  (rows / columns)
//   warp lanes           ->  #pragma omp simd          (slot inner loop)
//   shared-mem reduction ->  simd reduction(+:acc)
//   if (code != 0) gather -> branchless select of the gather INDEX
//                            (js = code ? j : 0), so buffers keep their
//                            real extents and the vectorizer keeps its
//                            straight-line body
//
// One deliberate improvement over the GLSL fetch: bit fields are read with
// a byte-aligned unaligned 32-bit load, (load32(base + bit/8) >> (bit&7)),
// which holds any field of width <= 25 bits and removes the two-word
// straddle branch entirely. All bit buffers built by the controller carry
// at least one word of slack, so the 4-byte load never runs off the end.
//
// Same math, same wang hash, same stochastic quantizer as the shader, so
// a scalar reference that recomputes slot addresses independently must
// match update results BYTE-EXACTLY. That is what the harness checks.
// ===========================================================================
#pragma once
#include "sparse_format_controller.hpp"
#include <cstring>
#include <array>

namespace sfc {

static const float W_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
static const uint32_t SORT_ORD[16] = {
    15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u, 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u
};

inline uint32_t wang(uint32_t h) {
    h = (h ^ 61u) ^ (h >> 16);
    h *= 9u;
    h ^= h >> 4;
    h *= 0x27d4eb2du;
    h ^= h >> 15;
    return h;
}

// Independent salt for importance dithering: reusing the weight
// rounder's draw would correlate weight hops with importance bumps.
constexpr uint32_t IMP_SALT = 0x51ed270bu;

// Unbiased stochastic step for a saturating 4-bit counter: integer part
// plus Bernoulli on the fraction, so E[imp'] = clamp(imp + d) for any
// magnitude of d. lb is the lower clamp (1 for live-coded synapses, 0
// when the weight code is zero, which lets drained zombies silence
// themselves at byte 0x00).
inline uint32_t imp_step(uint32_t imp, float d, uint32_t rnd, uint32_t lb) {
    const float ad = std::fabs(d);
    uint32_t inc = uint32_t(ad);
    const float frac = ad - float(inc);
    inc += (float(rnd & 0xffffu) * (1.0f / 65536.0f) < frac) ? 1u : 0u;
    int v = int(imp) + ((d >= 0.f) ? int(inc) : -int(inc));
    if (v < int(lb)) v = int(lb);
    if (v > 15) v = 15;
    return uint32_t(v);
}
inline uint32_t quantize(float t, uint32_t rnd) {
    float lo = W_LUT[SORT_ORD[0]];
    if (t <= lo) return SORT_ORD[0];
    for (uint32_t z = 1; z < 16; ++z) {
        float hi = W_LUT[SORT_ORD[z]];
        if (t <= hi) {
            float fr = (t - lo) / std::max(hi - lo, 1e-9f);
            return (float(rnd & 0xffffu) * (1.0f / 65536.0f) < fr)
                 ? SORT_ORD[z] : SORT_ORD[z - 1];
        }
        lo = hi;
    }
    return SORT_ORD[15];
}

// Branchless bit-field fetch, width <= 25. Buffers carry >= 1 slack word.
inline uint32_t bgetb(const uint8_t* b, uint64_t bit, uint32_t width) {
    uint32_t v;
    std::memcpy(&v, b + (bit >> 3), 4);
    return (v >> (bit & 7u)) & ((1u << width) - 1u);
}

// ===========================================================================
// BANKED kernels
// ===========================================================================

// STAGE_FORWARD: y[i] = sum_s w[i,s] * x[col(i,s)], rows = real row count.
inline void forward_banked_cpu(const BankedLayer& L, uint32_t rows,
                               const float* x, const uint32_t* xbits,
                               float* y, float w_scale = 0.0625f) {
    const uint32_t R = L.R(), LOG_S = L.LOG_S;
    const uint32_t NM = (1u << L.KC()) - 1u, MM = (1u << L.KR()) - 1u;
    const uint32_t aR = L.aR, aCinv = L.aCinv;
    const uint8_t* pib = reinterpret_cast<const uint8_t*>(L.pi_bits.data());
    const uint8_t* synp = L.syn.data();
    #pragma omp parallel for schedule(static)
    for (int64_t ii = 0; ii < int64_t(rows); ++ii) {
        const uint32_t i = uint32_t(ii);
        const uint64_t kbase = uint64_t((aR * i) & MM) * R;
        float acc = 0.f;
        #pragma omp simd reduction(+:acc)
        for (uint32_t s = 0; s < R; ++s) {
            const uint64_t k = kbase + s;
            const uint32_t code = synp[k] & 15u;
            const uint32_t v = bgetb(pib, k * LOG_S, LOG_S);
            const uint32_t j = (aCinv * ((s << LOG_S) | v)) & NM;
            const uint32_t js = (code != 0u) ? j : 0u;
            const float gate = (code != 0u) ? 1.f : 0.f;
            const float xb = float((xbits[js >> 5] >> (js & 31u)) & 1u);
            acc += gate * xb * (W_LUT[code] * w_scale) * x[js];
        }
        y[i] = acc;
    }
}

// STAGE_DX: dx[j] = sum_p w[row(j,p), j] * dy[row(j,p)].
// nkeys = 1 << KC (all stored column keys); dx sized nReal.
inline void dx_banked_cpu(const BankedLayer& L, uint32_t nkeys, uint32_t nReal,
                          const float* dy, const uint32_t* dybits,
                          float* dx, float w_scale = 0.0625f) {
    const uint32_t R = L.R(), C = L.C(), LOG_S = L.LOG_S;
    const uint32_t NM = (1u << L.KC()) - 1u, MM = (1u << L.KR()) - 1u;
    const uint32_t aCinv = L.aCinv, aRinv = L.aRinv;
    const uint8_t* invb = reinterpret_cast<const uint8_t*>(L.inv_bits.data());
    const uint8_t* synp = L.syn.data();
    #pragma omp parallel for schedule(static)
    for (int64_t jj = 0; jj < int64_t(nkeys); ++jj) {
        const uint32_t jc = uint32_t(jj);
        const uint64_t qbase = uint64_t(jc) * C;
        const uint32_t sb = jc >> LOG_S;
        float acc = 0.f;
        #pragma omp simd reduction(+:acc)
        for (uint32_t p = 0; p < C; ++p) {
            const uint32_t u = bgetb(invb, (qbase + p) * LOG_S, LOG_S);
            const uint32_t ip = (p << LOG_S) | u;
            const uint64_t k = uint64_t(ip) * R + sb;
            const uint32_t code = synp[k] & 15u;
            const uint32_t i = (aRinv * ip) & MM;
            const uint32_t is = (code != 0u) ? i : 0u;
            const float gate = (code != 0u) ? 1.f : 0.f;
            const float db = float((dybits[is >> 5] >> (is & 31u)) & 1u);
            acc += gate * db * (W_LUT[code] * w_scale) * dy[is];
        }
        const uint32_t jr = (aCinv * jc) & NM;
        if (jr < nReal) dx[jr] = acc;
    }
}

// STAGE_WUPDATE: active-dy rows own their weight bytes exclusively, so
// the outer parallel loop is race-free without atomics. Inner loop stays
// scalar: the quantizer's bracket search does not vectorize and this
// pass is not the hot one.
// imp_lr == 0: legacy bump-on-code-change importance (byte-exact with
// the original harnesses). imp_lr > 0: stochastic importance -- one
// signed dithered step of imp_lr * |x| * (|w| - |dy|) per touched
// synapse, the fused equivalent of "contribution*lr at forward,
// gradient*lr countering at backward". Weight bytes never re-round on a
// zero step in either mode.
inline void wupdate_banked_cpu(BankedLayer& L, const uint32_t* act_rows,
                               const float* act_dy, uint32_t nact,
                               const float* x, const uint32_t* xbits,
                               float lr, uint32_t seed,
                               float w_scale = 0.0625f, float imp_lr = 0.f) {
    const uint32_t R = L.R(), LOG_S = L.LOG_S;
    const uint32_t NM = (1u << L.KC()) - 1u, MM = (1u << L.KR()) - 1u;
    const uint32_t aR = L.aR, aCinv = L.aCinv;
    const uint8_t* pib = reinterpret_cast<const uint8_t*>(L.pi_bits.data());
    #pragma omp parallel for schedule(static)
    for (int64_t a = 0; a < int64_t(nact); ++a) {
        const uint32_t i = act_rows[a];
        const float g = act_dy[a];
        const uint64_t kbase = uint64_t((aR * i) & MM) * R;
        for (uint32_t s = 0; s < R; ++s) {
            const uint64_t k = kbase + s;
            uint8_t byv = L.syn[size_t(k)];
            uint32_t code = byv & 15u, imp = byv >> 4;
            if (imp == 0u && code == 0u) continue;          // sleeping
            const uint32_t v = bgetb(pib, k * LOG_S, LOG_S);
            const uint32_t j = (aCinv * ((s << LOG_S) | v)) & NM;
            const float xv = ((xbits[j >> 5] >> (j & 31u)) & 1u) ? x[j] : 0.f;
            const float upd = lr * g * xv;
            if (imp_lr == 0.f) {
                if (upd == 0.f) continue;  // zero step: do not re-round
                const float wnew = W_LUT[code] * w_scale - upd;
                const uint32_t nc = quantize(wnew / w_scale,
                                             wang(uint32_t(k) ^ seed));
                if (nc != code && imp < 15u) imp += 1u;
                L.syn[size_t(k)] = uint8_t((imp << 4) | nc);
            } else {
                if (xv == 0.f) continue;   // both terms vanish
                uint32_t nc = code;
                if (upd != 0.f) {
                    const float wnew = W_LUT[code] * w_scale - upd;
                    nc = quantize(wnew / w_scale, wang(uint32_t(k) ^ seed));
                }
                const float d = imp_lr * (std::fabs(W_LUT[code] * w_scale * xv)
                                        - std::fabs(g * xv));
                imp = imp_step(imp, d, wang(uint32_t(k) ^ seed ^ IMP_SALT),
                               (nc != 0u) ? 1u : 0u);
                L.syn[size_t(k)] = uint8_t((imp << 4) | nc);
            }
        }
    }
}

// STAGE_TRADE, serial (trades are rare and barriered on the GPU too).
struct Trade { uint32_t p, s, u1, u2; };
inline void trade_banked_cpu(BankedLayer& L, const Trade* tr, size_t n) {
    const uint32_t R = L.R(), C = L.C(), LOG_S = L.LOG_S;
    for (size_t t = 0; t < n; ++t) {
        const uint32_t p = tr[t].p, s = tr[t].s;
        const uint32_t ip1 = (p << LOG_S) | tr[t].u1;
        const uint32_t ip2 = (p << LOG_S) | tr[t].u2;
        const uint64_t k1 = uint64_t(ip1) * R + s;
        const uint64_t k2 = uint64_t(ip2) * R + s;
        const uint32_t v1 = L.fget(L.pi_bits, k1);
        const uint32_t v2 = L.fget(L.pi_bits, k2);
        L.fset(L.pi_bits, k1, v2);
        L.fset(L.pi_bits, k2, v1);
        const uint32_t jp1 = (s << LOG_S) | v1;
        const uint32_t jp2 = (s << LOG_S) | v2;
        L.fset(L.inv_bits, uint64_t(jp1) * C + p, tr[t].u2);
        L.fset(L.inv_bits, uint64_t(jp2) * C + p, tr[t].u1);
        L.syn[size_t(k1)] = 0;
        L.syn[size_t(k2)] = 0;
    }
}

// ===========================================================================
// PACKED kernels + builder
// ===========================================================================
struct PackedGpu {
    uint32_t M_real = 0, N_real = 0, R = 0, C = 0, R_LOG = 0;
    PackedView rowv;                    // per row: sorted column ids, padded
    PackedView colv;                    // per col: sorted row ids, padded
    std::vector<uint32_t> slotback;     // R_LOG-bit field per column slot
    std::vector<uint8_t>  syn;          // (M_real + 1) * R, sentinel row silent
};

// kept must satisfy <= R live per row and <= C live per column, which the
// banked contests guarantee for round-trips; asserts catch anything else.
// Row lists pad by repeating the last id (span stays small); column lists
// pad with sentinel row M_real whose weight bytes are all silent.
inline PackedGpu build_packed(const std::vector<Syn>& kept,
                              uint32_t M_real, uint32_t N_real,
                              uint32_t R, uint32_t C, uint32_t R_LOG) {
    PackedGpu P;
    P.M_real = M_real; P.N_real = N_real; P.R = R; P.C = C; P.R_LOG = R_LOG;
    P.syn.assign(size_t(M_real + 1) * R, 0u);
    std::vector<std::vector<std::pair<uint32_t, uint8_t>>> rows(M_real);
    for (const auto& s : kept) rows[s.i].push_back({s.j, s.b});
    std::vector<std::vector<uint32_t>> rlists(M_real);
    for (uint32_t i = 0; i < M_real; ++i) {
        auto& L = rows[i];
        std::sort(L.begin(), L.end());
        assert(L.size() <= R);
        auto& out = rlists[i];
        for (uint32_t s = 0; s < uint32_t(L.size()); ++s) {
            out.push_back(L[s].first);
            P.syn[size_t(i) * R + s] = L[s].second;
        }
        uint32_t padj = out.empty() ? 0u : out.back();
        while (out.size() < R) out.push_back(padj);
    }
    P.rowv = pack_view(rlists, R);

    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> cols(N_real);
    for (uint32_t i = 0; i < M_real; ++i)
        for (uint32_t s = 0; s < uint32_t(rows[i].size()); ++s)
            cols[rows[i][s].first].push_back({i, s});
    std::vector<std::vector<uint32_t>> clists(N_real);
    BitWriter sb;
    for (uint32_t j = 0; j < N_real; ++j) {
        auto& L = cols[j];
        std::sort(L.begin(), L.end());
        assert(L.size() <= C);
        auto& out = clists[j];
        for (auto& e : L) out.push_back(e.first);
        while (out.size() < C) out.push_back(M_real);        // sentinel
        for (uint32_t p = 0; p < C; ++p)
            sb.put(p < L.size() ? L[p].second : 0u, R_LOG);
    }
    P.colv = pack_view(clists, C);
    P.slotback = std::move(sb.w);
    P.slotback.push_back(0u);                                // fetch slack
    return P;
}

inline void forward_packed_cpu(const PackedGpu& P, const float* x,
                               const uint32_t* xbits, float* y,
                               float w_scale = 0.0625f) {
    const uint32_t R = P.R;
    const uint32_t gsz = (R < GROUP) ? R : GROUP, gcount = R / gsz;
    const uint8_t* strm = reinterpret_cast<const uint8_t*>(P.rowv.stream.data());
    const uint8_t* synp = P.syn.data();
    #pragma omp parallel for schedule(static)
    for (int64_t ii = 0; ii < int64_t(P.M_real); ++ii) {
        const uint32_t i = uint32_t(ii);
        float acc = 0.f;
        for (uint32_t g = 0; g < gcount; ++g) {
            const uint32_t gb = i * gcount + g;
            const uint32_t hdr = P.rowv.hdr[gb], wd = hdr & 31u;
            const uint32_t anchor = hdr >> 5;
            const uint64_t obase = P.rowv.off[gb];
            const uint64_t kbase = uint64_t(i) * R + g * gsz;
            #pragma omp simd reduction(+:acc)
            for (uint32_t t = 0; t < gsz; ++t) {
                const uint32_t code = synp[kbase + t] & 15u;
                const uint32_t j = anchor + bgetb(strm, obase + uint64_t(t) * wd, wd);
                const uint32_t js = (code != 0u) ? j : 0u;
                const float gate = (code != 0u) ? 1.f : 0.f;
                const float xb = float((xbits[js >> 5] >> (js & 31u)) & 1u);
                acc += gate * xb * (W_LUT[code] * w_scale) * x[js];
            }
        }
        y[i] = acc;
    }
}

inline void dx_packed_cpu(const PackedGpu& P, const float* dy,
                          const uint32_t* dybits, float* dx,
                          float w_scale = 0.0625f) {
    const uint32_t R = P.R, C = P.C, R_LOG = P.R_LOG;
    const uint32_t gsz = (C < GROUP) ? C : GROUP, gcount = C / gsz;
    const uint8_t* strm = reinterpret_cast<const uint8_t*>(P.colv.stream.data());
    const uint8_t* sbk = reinterpret_cast<const uint8_t*>(P.slotback.data());
    const uint8_t* synp = P.syn.data();
    #pragma omp parallel for schedule(static)
    for (int64_t jj = 0; jj < int64_t(P.N_real); ++jj) {
        const uint32_t j = uint32_t(jj);
        float acc = 0.f;
        for (uint32_t g = 0; g < gcount; ++g) {
            const uint32_t gb = j * gcount + g;
            const uint32_t hdr = P.colv.hdr[gb], wd = hdr & 31u;
            const uint32_t anchor = hdr >> 5;
            const uint64_t obase = P.colv.off[gb];
            const uint64_t qbase = uint64_t(j) * C + g * gsz;
            #pragma omp simd reduction(+:acc)
            for (uint32_t t = 0; t < gsz; ++t) {
                const uint32_t i = anchor + bgetb(strm, obase + uint64_t(t) * wd, wd);
                const uint32_t s = bgetb(sbk, (qbase + t) * R_LOG, R_LOG);
                const uint32_t code = synp[uint64_t(i) * R + s] & 15u;
                const uint32_t is = (code != 0u) ? i : 0u;
                const float gate = (code != 0u) ? 1.f : 0.f;
                const float db = float((dybits[is >> 5] >> (is & 31u)) & 1u);
                acc += gate * db * (W_LUT[code] * w_scale) * dy[is];
            }
        }
        dx[j] = acc;
    }
}

inline void wupdate_packed_cpu(PackedGpu& P, const uint32_t* act_rows,
                               const float* act_dy, uint32_t nact,
                               const float* x, const uint32_t* xbits,
                               float lr, uint32_t seed,
                               float w_scale = 0.0625f, float imp_lr = 0.f) {
    const uint32_t R = P.R;
    const uint8_t* strm = reinterpret_cast<const uint8_t*>(P.rowv.stream.data());
    const uint32_t gsz = (R < GROUP) ? R : GROUP, gcount = R / gsz;
    #pragma omp parallel for schedule(static)
    for (int64_t a = 0; a < int64_t(nact); ++a) {
        const uint32_t i = act_rows[a];
        const float g = act_dy[a];
        for (uint32_t s = 0; s < R; ++s) {
            const uint64_t k = uint64_t(i) * R + s;
            uint8_t byv = P.syn[size_t(k)];
            uint32_t code = byv & 15u, imp = byv >> 4;
            if (imp == 0u && code == 0u) continue;
            const uint32_t gb = i * gcount + s / gsz;
            const uint32_t hdr = P.rowv.hdr[gb], wd = hdr & 31u;
            const uint32_t j = (hdr >> 5)
                + bgetb(strm, uint64_t(P.rowv.off[gb]) + uint64_t(s % gsz) * wd, wd);
            const float xv = ((xbits[j >> 5] >> (j & 31u)) & 1u) ? x[j] : 0.f;
            const float upd = lr * g * xv;
            if (imp_lr == 0.f) {
                if (upd == 0.f) continue;
                const float wnew = W_LUT[code] * w_scale - upd;
                const uint32_t nc = quantize(wnew / w_scale,
                                             wang(uint32_t(k) ^ seed));
                if (nc != code && imp < 15u) imp += 1u;
                P.syn[size_t(k)] = uint8_t((imp << 4) | nc);
            } else {
                if (xv == 0.f) continue;
                uint32_t nc = code;
                if (upd != 0.f) {
                    const float wnew = W_LUT[code] * w_scale - upd;
                    nc = quantize(wnew / w_scale, wang(uint32_t(k) ^ seed));
                }
                const float d = imp_lr * (std::fabs(W_LUT[code] * w_scale * xv)
                                        - std::fabs(g * xv));
                imp = imp_step(imp, d, wang(uint32_t(k) ^ seed ^ IMP_SALT),
                               (nc != 0u) ? 1u : 0u);
                P.syn[size_t(k)] = uint8_t((imp << 4) | nc);
            }
        }
    }
}

} // namespace sfc
