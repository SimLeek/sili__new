// succinct_inverse.hpp
// ===========================================================================
// Going below 32 bits per slot: drop the stored inverse (inv_bits, LOG_S
// bits/slot) and answer inverse queries from pi alone.
//
// A permutation pi on [0, S) decomposes into cycles, and the predecessor
// pi^{-1}(v) is one forward-walk around v's cycle away. Walking a whole
// cycle is O(S); the classic fix (Munro, Raman, Raman, Rao 2003) marks
// every t-th element along each cycle and stores, for each marked
// element, a pointer jumping t steps BACKWARD along its cycle. A query
// then walks forward at most ~t steps to a mark, jumps back once, and
// walks forward at most t more steps to land on the predecessor: O(t)
// pi-fetches, all plane-local (a plane's pi entries span ~S*R*LOG_S/8
// bytes, cache-warm).
//
// Cost accounting per slot, replacing the LOG_S-bit inverse field:
//     marker bitmap        1 bit
//     rank counters        16/64 = 0.25 bits
//     shortcut pointers    LOG_S / t bits
// At LOG_S = 12, t = 6: 3.25 bits instead of 12, so the banked layer
// drops from 2*LOG_S + 8 = 32 to LOG_S + 3.25 + 8 = 23.25 bits/slot.
// The price is paid only by passes that query the inverse (backward-dx
// and the growth probe): O(t) dependent fetches instead of 1.
//
// Trades edit pi and therefore invalidate a plane's cycle structure.
// Shortcuts are ACCELERATORS, not truth: a lookup on a dirty plane that
// exceeds its step budget falls back to a plain cycle walk (correct,
// O(cycle)), and finalize_trades() rebuilds shortcuts for dirty planes
// in O(S) per plane, amortized over a trade batch. pi itself is always
// current, so correctness never depends on rebuild timing.
//
// This structure REPLACES inv_bits at run time; BankedLayer's inv_bits
// can be freed (or never materialized) once this is built. The trade
// kernel's inv writes become unnecessary for layers running succinct.
// ===========================================================================
#pragma once
#include "synapto.hpp"

namespace sfc {

struct SuccinctInv {
    uint32_t LOG_S = 0, R_LOG = 0, C_LOG = 0, t = 6;
    uint32_t S = 0, R = 0, C = 0, nplanes = 0, scap = 0, wpp = 0;
    // mark:  nplanes * (S/64) words, bit u set iff u is a shortcut site
    // rankc: per word, number of marks in earlier words of the plane
    // sc:    packed LOG_S-bit fields, scap reserved per plane, rank order
    std::vector<uint64_t> mark;
    std::vector<uint16_t> rankc;
    std::vector<uint32_t> sc_bits;
    std::vector<uint8_t>  dirty;
    uint64_t overflow = 0;   // cycles that ran out of shortcut budget

    double bits_per_slot() const {
        double slots = double(nplanes) * S;
        return (double(mark.size()) * 64 + double(rankc.size()) * 16
              + double(nplanes) * scap * LOG_S) / slots;
    }
};

inline uint32_t pi_at(const BankedLayer& L, uint32_t p, uint32_t s, uint32_t u) {
    const uint64_t k = (uint64_t((p << L.LOG_S) | u)) * L.R() + s;
    const uint64_t bit = k * L.LOG_S;
    const uint8_t* b = reinterpret_cast<const uint8_t*>(L.pi_bits.data());
    uint32_t v;
    std::memcpy(&v, b + (bit >> 3), 4);
    return (v >> (bit & 7u)) & ((1u << L.LOG_S) - 1u);
}

inline bool si_marked(const SuccinctInv& si, uint32_t P, uint32_t x) {
    return (si.mark[size_t(P) * si.wpp + (x >> 6)] >> (x & 63u)) & 1u;
}
inline uint32_t si_shortcut(const SuccinctInv& si, uint32_t P, uint32_t x) {
    const size_t wbase = size_t(P) * si.wpp;
    const uint64_t w = si.mark[wbase + (x >> 6)];
    const uint32_t r = si.rankc[wbase + (x >> 6)]
        + uint32_t(__builtin_popcountll(w & ((1ull << (x & 63u)) - 1ull)));
    const uint64_t bit = (uint64_t(P) * si.scap + r) * si.LOG_S;
    return bits_get(si.sc_bits.data(), bit, si.LOG_S);
}

// Rebuild one plane's marks, ranks, and shortcuts from pi. O(S).
inline void si_rebuild_plane(const BankedLayer& L, SuccinctInv& si,
                             uint32_t p, uint32_t s) {
    const uint32_t S = si.S, P = p * si.R + s;
    static thread_local std::vector<uint8_t>  vis, mk;
    static thread_local std::vector<uint16_t> val;
    static thread_local std::vector<uint32_t> cyc;
    vis.assign(S, 0); mk.assign(S, 0); val.assign(S, 0);
    uint32_t placed = 0;
    for (uint32_t start = 0; start < S; ++start) {
        if (vis[start]) continue;
        cyc.clear();
        uint32_t x = start;
        do { vis[x] = 1; cyc.push_back(x); x = pi_at(L, p, s, x); }
        while (x != start);
        const uint32_t len = uint32_t(cyc.size());
        if (len <= si.t) continue;   // resolved by a short forward walk
        // Marks at 0, t, 2t, ...: every inter-mark gap is <= t, so any
        // query reaches a mark within t-1 forward steps and the back
        // jump lands strictly BEHIND the target. (Marking t-1, 2t-1,...
        // leaves a wrap gap of up to 2t-1 and breaks the bound; that
        // bug shipped in the first version of this file.)
        for (uint32_t j = 0; j < len; j += si.t) {
            if (placed == si.scap) { ++si.overflow; break; }
            mk[cyc[j]] = 1;
            val[cyc[j]] = uint16_t(cyc[(j + len - si.t) % len]);
            ++placed;
        }
    }
    const size_t wbase = size_t(P) * si.wpp;
    uint32_t run = 0, r = 0;
    for (uint32_t w = 0; w < si.wpp; ++w) {
        uint64_t bits = 0;
        si.rankc[wbase + w] = uint16_t(run);
        for (uint32_t b = 0; b < 64 && (w * 64 + b) < S; ++b)
            if (mk[w * 64 + b]) { bits |= 1ull << b; ++run; }
        si.mark[wbase + w] = bits;
    }
    for (uint32_t u = 0; u < S; ++u)
        if (mk[u]) {
            const uint64_t bit = (uint64_t(P) * si.scap + r) * si.LOG_S;
            // in-place field write (single-threaded rebuild path)
            uint64_t wd = bit >> 5, sh = bit & 31;
            uint32_t m = (1u << si.LOG_S) - 1u, v = val[u];
            si.sc_bits[size_t(wd)] =
                (si.sc_bits[size_t(wd)] & ~(m << sh)) | (v << sh);
            if (sh + si.LOG_S > 32)
                si.sc_bits[size_t(wd) + 1] =
                    (si.sc_bits[size_t(wd) + 1] & ~((1u << (sh + si.LOG_S - 32)) - 1u))
                    | (v >> (32 - sh));
            ++r;
        }
    si.dirty[P] = 0;
}

inline SuccinctInv build_succinct_inv(const BankedLayer& L, uint32_t t) {
    SuccinctInv si;
    si.LOG_S = L.LOG_S; si.R_LOG = L.R_LOG; si.C_LOG = L.C_LOG; si.t = t;
    si.S = L.S(); si.R = L.R(); si.C = L.C();
    si.nplanes = si.R * si.C;
    si.scap = (si.S + t - 1) / t + 16;   // ceil(S/t) + slack for cycle count
    si.wpp = (si.S + 63) / 64;
    si.mark.assign(size_t(si.nplanes) * si.wpp, 0);
    si.rankc.assign(size_t(si.nplanes) * si.wpp, 0);
    si.sc_bits.assign((uint64_t(si.nplanes) * si.scap * si.LOG_S + 31) / 32 + 1, 0);
    si.dirty.assign(si.nplanes, 0);
    for (uint32_t p = 0; p < si.C; ++p)
        for (uint32_t s = 0; s < si.R; ++s)
            si_rebuild_plane(L, si, p, s);
    return si;
}

// pi^{-1}(v) within plane (p, s). Clean plane: <= ~3t pi-fetches.
// Dirty (or pathological) plane: falls back to a plain cycle walk.
// stats pointers are optional and must be null in parallel callers.
inline uint32_t si_inv(const BankedLayer& L, const SuccinctInv& si,
                       uint32_t p, uint32_t s, uint32_t v,
                       uint64_t* hops = nullptr, uint64_t* fallbacks = nullptr) {
    const uint32_t P = p * si.R + s;
    if (!si.dirty[P]) {
        uint32_t x = v;
        bool jumped = false;
        const uint32_t cap = 3 * si.t + 4;
        for (uint32_t step = 0; step < cap; ++step) {
            const uint32_t nx = pi_at(L, p, s, x);
            if (hops) ++*hops;
            if (nx == v) return x;
            if (!jumped && si_marked(si, P, x)) {
                x = si_shortcut(si, P, x);
                jumped = true;
            } else {
                x = nx;
            }
        }
    }
    if (fallbacks) ++*fallbacks;
    uint32_t x = v;
    for (uint32_t i = 0; i < si.S; ++i) {
        const uint32_t nx = pi_at(L, p, s, x);
        if (nx == v) return x;
        x = nx;
    }
    assert(false && "pi is not a permutation");
    return 0;
}

inline void mark_trade_dirty(SuccinctInv& si, const Trade& tr) {
    si.dirty[tr.p * si.R + tr.s] = 1;
}
inline void finalize_trades(const BankedLayer& L, SuccinctInv& si) {
    for (uint32_t p = 0; p < si.C; ++p)
        for (uint32_t s = 0; s < si.R; ++s)
            if (si.dirty[p * si.R + s]) si_rebuild_plane(L, si, p, s);
}

// Backward-dx through the succinct inverse: identical math to
// dx_banked_cpu with the inv_bits fetch replaced by si_inv. Inner loop
// is scalar (variable-length walks do not vectorize); the outer loop
// still parallelizes, and this is the ONLY pass that pays the O(t).
inline void dx_banked_si_cpu(const BankedLayer& L, const SuccinctInv& si,
                             uint32_t nkeys, uint32_t nReal,
                             const float* dy, const uint32_t* dybits,
                             float* dx, float w_scale = 0.0625f) {
    const uint32_t R = L.R(), C = L.C(), LOG_S = L.LOG_S;
    const uint32_t NM = (1u << L.KC()) - 1u, MM = (1u << L.KR()) - 1u;
    const uint32_t aCinv = L.aCinv, aRinv = L.aRinv;
    const uint8_t* synp = L.syn.data();
    #pragma omp parallel for schedule(static)
    for (int64_t jj = 0; jj < int64_t(nkeys); ++jj) {
        const uint32_t jc = uint32_t(jj);
        const uint32_t sb = jc >> LOG_S, v = jc & (si.S - 1u);
        float acc = 0.f;
        for (uint32_t p = 0; p < C; ++p) {
            const uint32_t u = si_inv(L, si, p, sb, v);
            const uint32_t ip = (p << LOG_S) | u;
            const uint64_t k = uint64_t(ip) * R + sb;
            const uint32_t code = synp[size_t(k)] & 15u;
            if (code == 0u) continue;
            const uint32_t i = (aRinv * ip) & MM;
            if (((dybits[i >> 5] >> (i & 31u)) & 1u) == 0u) continue;
            acc += W_LUT[code] * w_scale * dy[i];
        }
        const uint32_t jr = (aCinv * jc) & NM;
        if (jr < nReal) dx[jr] = acc;
    }
}

} // namespace sfc
