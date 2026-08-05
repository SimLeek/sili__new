// sparse_format_controller.hpp
// ===========================================================================
// Runtime structure selection between two index codecs for doubly-sparse
// trainable layers:
//
//   PACKED  anchor + fixed-width deltas in groups of 32 (both views),
//           plus an R_LOG-bit slot backpointer on the column view.
//           Wins when spans are clustered. Rewires by group re-encode.
//   BANKED  quasi-cyclic permutation planes behind multiplicative-hash
//           relabeling. Constant 2*LOG_S + 8 bits/param, closed-form
//           addressing, O(1) trades. Wins when spans are large.
//
// The controller compares TOTAL stored bits per parameter with hysteresis
// and a dwell count, and gates the packed -> banked direction on an
// importance-weighted demotion audit (search over candidate multipliers).
// banked -> packed is lossless, so only one direction carries risk.
// Demoted synapses are not deleted; they exit as a regrow queue for the
// synaptogenesis intake, which exists precisely to place synapses well.
//
// The arrays produced here are byte-identical to the SSBO contents the
// dual_codec_ell.comp.glsl kernels expect. Layers drift between
// neighborhoods; this file is just the moving van.
//
// ASCII only, header-only, no dependencies beyond the standard library.
// ===========================================================================
#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cassert>
#include <random>

namespace sfc {

static constexpr uint32_t GROUP = 32;

// Neutral interchange form. b = (imp << 4) | w4, code 0 == silent.
struct Syn {
    uint32_t i, j;
    uint8_t  b;
};

// ---------------------------------------------------------------------------
// Bit stream helpers (fields never span more than two words).
// ---------------------------------------------------------------------------
struct BitWriter {
    std::vector<uint32_t> w;
    uint64_t bits = 0;
    void put(uint32_t v, uint32_t n) {
        if (n == 0) return;
        uint64_t word = bits >> 5, sh = bits & 31;
        if (word + 2 > w.size()) w.resize(word + 2, 0u);
        w[word] |= v << sh;
        if (sh + n > 32) w[word + 1] |= v >> (32 - sh);
        bits += n;
    }
};
inline uint32_t bits_get(const uint32_t* a, uint64_t pos, uint32_t n) {
    if (n == 0) return 0;
    uint64_t word = pos >> 5, sh = pos & 31;
    uint32_t v = a[word] >> sh;
    if (sh + n > 32) v |= a[word + 1] << (32 - sh);
    return v & ((n < 32) ? ((1u << n) - 1u) : 0xffffffffu);
}

inline uint32_t modinv_pow2(uint32_t a, uint32_t bitsN) {
    assert(a & 1u);
    uint32_t x = 1;                       // Newton: x <- x * (2 - a*x)
    for (int it = 0; it < 6; ++it) x *= 2u - a * x;
    return (bitsN >= 32) ? x : (x & ((1u << bitsN) - 1u));
}
inline uint32_t bit_width(uint32_t v) {
    uint32_t n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}

// ---------------------------------------------------------------------------
// Packed codec, one direction (row-major or column-major).
// hdr[g] = anchor << 5 | width; off[g] = bit offset; stream = deltas.
// ---------------------------------------------------------------------------
struct PackedView {
    uint32_t rows = 0, slots = 0;                  // slots per row (R or C)
    std::vector<uint32_t> hdr, off, stream;
    uint64_t index_bits() const {
        return uint64_t(stream.size()) * 32u + uint64_t(hdr.size()) * 64u;
    }
};

// lists[r] must be sorted ascending; lists[r].size() == slots (pad the
// tail of ragged rows by repeating the last id with a silent weight).
inline PackedView pack_view(const std::vector<std::vector<uint32_t>>& lists,
                            uint32_t slots) {
    PackedView v;
    v.rows = uint32_t(lists.size());
    v.slots = slots;
    const uint32_t gsz = (slots < GROUP) ? slots : GROUP;
    BitWriter bw;
    for (const auto& L : lists) {
        assert(L.size() == slots);
        for (uint32_t g = 0; g < slots; g += gsz) {
            uint32_t anchor = L[g], span = 0;
            for (uint32_t t = 0; t < gsz; ++t) span = std::max(span, L[g + t] - anchor);
            uint32_t wd = bit_width(span);
            assert(wd <= 25);   // byte-aligned unaligned-load fetch bound
            v.hdr.push_back((anchor << 5) | wd);
            v.off.push_back(uint32_t(bw.bits));
            for (uint32_t t = 0; t < gsz; ++t) bw.put(L[g + t] - anchor, wd);
        }
    }
    v.stream = std::move(bw.w);
    if (v.stream.empty()) v.stream.push_back(0u);
    return v;
}
inline uint32_t packed_id(const PackedView& v, uint32_t r, uint32_t s) {
    const uint32_t gsz = (v.slots < GROUP) ? v.slots : GROUP;
    uint32_t g = r * (v.slots / gsz) + (s / gsz);
    uint32_t wd = v.hdr[g] & 31u;
    return (v.hdr[g] >> 5) + bits_get(v.stream.data(),
                                      uint64_t(v.off[g]) + (s % gsz) * wd, wd);
}

// ---------------------------------------------------------------------------
// Banked layer. pi/inv are LOG_S-bit fields; syn is row-major in STORED
// row order (ip). Multipliers hash real ids to stored keys; banks are the
// HIGH bits of the hash, so strides land pseudorandomly.
// ---------------------------------------------------------------------------
struct BankedLayer {
    uint32_t LOG_S = 12, R_LOG = 5, C_LOG = 5;
    uint32_t aC = 1, aCinv = 1, aR = 1, aRinv = 1;
    std::vector<uint32_t> pi_bits, inv_bits;
    std::vector<uint8_t>  syn;
    uint32_t R() const { return 1u << R_LOG; }
    uint32_t C() const { return 1u << C_LOG; }
    uint32_t S() const { return 1u << LOG_S; }
    uint32_t KC() const { return LOG_S + R_LOG; }
    uint32_t KR() const { return LOG_S + C_LOG; }
    uint32_t fget(const std::vector<uint32_t>& a, uint64_t f) const {
        return bits_get(a.data(), f * LOG_S, LOG_S);
    }
    void fset(std::vector<uint32_t>& a, uint64_t f, uint32_t val) {
        uint64_t bit = f * LOG_S, w = bit >> 5, sh = bit & 31;
        uint32_t m = (1u << LOG_S) - 1u;
        a[w] = (a[w] & ~(m << sh)) | (val << sh);
        if (sh + LOG_S > 32) {
            uint32_t hb = uint32_t(sh) + LOG_S - 32;
            a[w + 1] = (a[w + 1] & ~((1u << hb) - 1u)) | (val >> (32 - sh));
        }
    }
};

// Both costs are per STORED SLOT, so underfilled layers compare fairly;
// multiply either by capacity/live for bits per live synapse.
inline double banked_bits_per_param(uint32_t LOG_S) { return 2.0 * LOG_S + 8.0; }
inline double packed_bits_per_param(const PackedView& rowv, const PackedView& colv,
                                    uint32_t R_LOG) {
    double rs = double(rowv.rows) * rowv.slots;
    double cs = double(colv.rows) * colv.slots;
    return double(rowv.index_bits()) / rs
         + double(colv.index_bits()) / cs
         + double(R_LOG)   // slot backpointer on the column view
         + 8.0;            // weight + importance byte
}

// ---------------------------------------------------------------------------
// Audition: pick the multiplier that minimizes importance-weighted
// demotion when synapses are bucketed by the high bits of (a * id).
// Structured layers (strides, rasters) often have a much better a than
// the blind-hash expectation; this is where the search earns its keep.
// ---------------------------------------------------------------------------
struct Audition {
    uint32_t a = 1;
    double   demoted_frac = 1.0;   // importance-weighted
};

// owner_of / banked_of select the axis: for the column contest,
// owner = i, hashed id = j; for the row contest, owner = j, hashed id = i.
inline Audition audition(const std::vector<Syn>& syns, bool bank_columns,
                         uint32_t key_bits, uint32_t bank_log,
                         uint32_t owner_count, int candidates,
                         uint64_t rng_seed) {
    std::vector<std::vector<uint32_t>> owned(owner_count);
    for (uint32_t t = 0; t < syns.size(); ++t)
        owned[bank_columns ? syns[t].i : syns[t].j].push_back(t);

    std::mt19937_64 rng(rng_seed);
    std::vector<uint32_t> cand;
    cand.push_back(2654435769u);                       // golden ratio, classic
    while (int(cand.size()) < candidates)
        cand.push_back(uint32_t(rng()) | 1u);

    uint32_t mask = (key_bits >= 32) ? 0xffffffffu : ((1u << key_bits) - 1u);
    uint32_t shift = key_bits - bank_log;
    double total_imp = 0.0;
    for (const auto& s : syns) total_imp += double(s.b >> 4) + 1.0;

    Audition best;
    std::vector<double> bank_sum(1u << bank_log), bank_max(1u << bank_log);
    for (uint32_t a : cand) {
        double demoted = 0.0;
        for (const auto& L : owned) {
            std::fill(bank_sum.begin(), bank_sum.end(), 0.0);
            std::fill(bank_max.begin(), bank_max.end(), 0.0);
            for (uint32_t t : L) {
                uint32_t id = bank_columns ? syns[t].j : syns[t].i;
                uint32_t bk = ((a * id) & mask) >> shift;
                double imp = double(syns[t].b >> 4) + 1.0;
                bank_sum[bk] += imp;
                bank_max[bk] = std::max(bank_max[bk], imp);
            }
            for (uint32_t bk = 0; bk < (1u << bank_log); ++bk)
                demoted += bank_sum[bk] - bank_max[bk];
        }
        double frac = (total_imp > 0.0) ? demoted / total_imp : 0.0;
        if (frac < best.demoted_frac) { best.a = a; best.demoted_frac = frac; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// packed -> banked. Keeps the importance winner of every (row, colbank)
// and (col, rowbank) contest; losers exit via regrow. Planes are then
// completed to full permutations with silent filler.
// ---------------------------------------------------------------------------
struct ToBankedResult {
    BankedLayer layer;
    std::vector<Syn> regrow;
    double demoted_frac = 0.0;
};

inline ToBankedResult to_banked(std::vector<Syn> syns,
                                uint32_t LOG_S, uint32_t R_LOG, uint32_t C_LOG,
                                uint32_t M_real, uint32_t N_real,
                                int candidates = 32, uint64_t seed = 1) {
    ToBankedResult res;
    BankedLayer& L = res.layer;
    L.LOG_S = LOG_S; L.R_LOG = R_LOG; L.C_LOG = C_LOG;
    Audition ac = audition(syns, true,  L.KC(), R_LOG, M_real, candidates, seed);
    Audition ar = audition(syns, false, L.KR(), C_LOG, N_real, candidates, seed + 1);
    L.aC = ac.a; L.aCinv = modinv_pow2(ac.a, L.KC());
    L.aR = ar.a; L.aRinv = modinv_pow2(ar.a, L.KR());

    const uint32_t NM = (1u << L.KC()) - 1u, MM = (1u << L.KR()) - 1u;
    auto jp_of = [&](uint32_t j) { return (L.aC * j) & NM; };
    auto ip_of = [&](uint32_t i) { return (L.aR * i) & MM; };

    // Contest 1: per (row, colbank) keep max importance.
    auto imp_of = [](const Syn& s) { return double(s.b >> 4) + 1.0; };
    std::stable_sort(syns.begin(), syns.end(), [&](const Syn& x, const Syn& y) {
        uint32_t kx = (x.i << R_LOG) | (jp_of(x.j) >> LOG_S);
        uint32_t ky = (y.i << R_LOG) | (jp_of(y.j) >> LOG_S);
        if (kx != ky) return kx < ky;
        return imp_of(x) > imp_of(y);
    });
    std::vector<Syn> keep;
    keep.reserve(syns.size());
    uint32_t last = 0xffffffffu;
    for (const auto& s : syns) {
        uint32_t k = (s.i << R_LOG) | (jp_of(s.j) >> LOG_S);
        if (k == last) res.regrow.push_back(s);
        else { keep.push_back(s); last = k; }
    }
    // Contest 2: per (column, rowbank) keep max importance.
    std::stable_sort(keep.begin(), keep.end(), [&](const Syn& x, const Syn& y) {
        uint64_t kx = (uint64_t(x.j) << C_LOG) | (ip_of(x.i) >> LOG_S);
        uint64_t ky = (uint64_t(y.j) << C_LOG) | (ip_of(y.i) >> LOG_S);
        if (kx != ky) return kx < ky;
        return imp_of(x) > imp_of(y);
    });
    std::vector<Syn> fin;
    fin.reserve(keep.size());
    uint64_t last2 = ~0ull;
    for (const auto& s : keep) {
        uint64_t k = (uint64_t(s.j) << C_LOG) | (ip_of(s.i) >> LOG_S);
        if (k == last2) res.regrow.push_back(s);
        else { fin.push_back(s); last2 = k; }
    }
    double timp = 0, dimp = 0;
    for (const auto& s : fin) timp += imp_of(s);
    for (const auto& s : res.regrow) dimp += imp_of(s);
    res.demoted_frac = (timp + dimp > 0) ? dimp / (timp + dimp) : 0.0;

    // Build planes, then complete each to a full permutation.
    const uint32_t S = L.S(), R = L.R(), C = L.C();
    uint64_t nslots = uint64_t(S) * C * R;
    L.pi_bits.assign(size_t((nslots * LOG_S + 31) / 32 + 1), 0u);
    L.inv_bits.assign(size_t((nslots * LOG_S + 31) / 32 + 1), 0u);
    L.syn.assign(size_t(nslots), 0u);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> plane(size_t(C) * R);
    std::vector<const Syn*> plane_syn(size_t(nslots), nullptr);
    for (const auto& s : fin) {
        uint32_t ip = ip_of(s.i), jp = jp_of(s.j);
        uint32_t p = ip >> LOG_S, u = ip & (S - 1);
        uint32_t sb = jp >> LOG_S, v = jp & (S - 1);
        plane[size_t(p) * R + sb].push_back({u, v});
        uint64_t k = uint64_t(ip) * R + sb;
        L.fset(L.pi_bits, k, v);
        L.fset(L.inv_bits, uint64_t(jp) * C + p, u);
        L.syn[size_t(k)] = s.b;
    }
    std::vector<uint8_t> used_u(S), used_v(S);
    for (uint32_t p = 0; p < C; ++p)
        for (uint32_t sb = 0; sb < R; ++sb) {
            std::fill(used_u.begin(), used_u.end(), 0);
            std::fill(used_v.begin(), used_v.end(), 0);
            for (auto& uv : plane[size_t(p) * R + sb]) {
                used_u[uv.first] = 1;
                used_v[uv.second] = 1;
            }
            uint32_t v = 0;
            for (uint32_t u = 0; u < S; ++u) {
                if (used_u[u]) continue;
                while (used_v[v]) ++v;                 // silent filler pairing
                uint32_t ip = (p << LOG_S) | u, jp = (sb << LOG_S) | v;
                L.fset(L.pi_bits, uint64_t(ip) * R + sb, v);
                L.fset(L.inv_bits, uint64_t(jp) * C + p, u);
                ++v;
            }
        }
    return res;
}

// ---------------------------------------------------------------------------
// banked -> interchange (lossless), and on to packed views + slotback.
// ---------------------------------------------------------------------------
inline std::vector<Syn> to_syns(const BankedLayer& L,
                                uint32_t M_real, uint32_t N_real) {
    std::vector<Syn> out;
    const uint32_t R = L.R(), S = L.S();
    for (uint64_t k = 0; k < L.syn.size(); ++k) {
        if (L.syn[size_t(k)] == 0) continue;
        uint32_t ip = uint32_t(k >> L.R_LOG), sb = uint32_t(k & (R - 1));
        uint32_t v = L.fget(L.pi_bits, k);
        uint32_t jp = (sb << L.LOG_S) | v;
        uint32_t i = (L.aRinv * ip) & ((1u << L.KR()) - 1u);
        uint32_t j = (L.aCinv * jp) & ((1u << L.KC()) - 1u);
        if (i < M_real && j < N_real) out.push_back({i, j, L.syn[size_t(k)]});
        (void)S;
    }
    return out;
}

// Sampled estimate of what the packed row-view would cost per index if
// this banked layer converted down. Cheap enough to run on a timer.
inline double estimate_packed_row_bits(const BankedLayer& L, uint32_t M_real,
                                       uint32_t N_real, int sample_rows,
                                       uint64_t seed) {
    std::mt19937_64 rng(seed);
    const uint32_t R = L.R();
    uint64_t bits = 0, groups = 0;
    std::vector<uint32_t> cols;
    for (int t = 0; t < sample_rows; ++t) {
        uint32_t i = uint32_t(rng() % M_real);
        uint32_t ip = (L.aR * i) & ((1u << L.KR()) - 1u);
        cols.clear();
        for (uint32_t sb = 0; sb < R; ++sb) {
            uint64_t k = uint64_t(ip) * R + sb;
            if (L.syn[size_t(k)] == 0) continue;
            uint32_t jp = (sb << L.LOG_S) | L.fget(L.pi_bits, k);
            uint32_t j = (L.aCinv * jp) & ((1u << L.KC()) - 1u);
            if (j < N_real) cols.push_back(j);
        }
        if (cols.empty()) continue;
        std::sort(cols.begin(), cols.end());
        for (size_t g = 0; g < cols.size(); g += GROUP) {
            size_t e = std::min(cols.size(), g + GROUP);
            uint32_t wd = bit_width(cols[e - 1] - cols[g]);
            bits += uint64_t(wd) * GROUP + 64;          // deltas + hdr + off
            groups += 1;
        }
    }
    uint64_t n = groups * GROUP;
    return n ? double(bits) / double(n) : 1e9;
}

// ---------------------------------------------------------------------------
// The decision. Hysteresis on total bits/param, dwell so one noisy
// evaluation cannot flip a layer, demotion gate on the risky direction.
// ---------------------------------------------------------------------------
enum class Fmt { Packed, Banked };

struct FormatStats {
    double packed_total_bits;   // measured (packed) or estimated (banked)
    double demoted_frac;        // from audition; only used going up
};

struct Controller {
    double margin      = 2.0;   // bits of hysteresis around banked cost
    double demote_max  = 0.05;  // max importance-weighted demotion
    int    dwell       = 4;     // consecutive agreeing evaluations
    uint32_t LOG_S     = 12;
    int above = 0, below = 0;

    Fmt decide(Fmt cur, const FormatStats& s) {
        double bb = banked_bits_per_param(LOG_S);
        if (cur == Fmt::Packed) {
            below = 0;
            bool worse = s.packed_total_bits > bb + margin;
            above = worse ? above + 1 : 0;
            if (above >= dwell && s.demoted_frac <= demote_max) {
                above = 0;
                return Fmt::Banked;
            }
            return Fmt::Packed;
        } else {
            above = 0;
            bool better = s.packed_total_bits < bb - margin;
            below = better ? below + 1 : 0;
            if (below >= dwell) {
                below = 0;
                return Fmt::Packed;      // lossless direction, no gate
            }
            return Fmt::Banked;
        }
    }
};

} // namespace sfc
