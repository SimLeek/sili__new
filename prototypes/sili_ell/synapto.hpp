// synapto.hpp
// ===========================================================================
// Structural grow / prune MECHANISM for the banked codec, kept strictly
// policy-free: importance decides WHO, outer products propose WHAT, this
// file only knows HOW. Everything here is O(1) per synapse except the
// capacity rebuilds, which are offline passes through the interchange
// form.
//
//   probe_grow(L, i, j)   two byte reads at closed-form slots; reports
//                         Free / AlreadyPresent / which live synapse
//                         blocks (row-bank or column-bank occupant),
//                         so eviction becomes an explicit policy call.
//   commit_grow(...)      a transposition among SILENT slots (zero
//                         semantic effect) + one byte write.
//   prune(...)            one byte write; wiring stays, so re-growing
//                         the same pair later needs no trade at all.
//   assemble_banked(...)  build from placed synapses with a HARD assert
//                         on bank collisions (no contests, no demotion).
//   expand_banked(...)    R_LOG/C_LOG + n rebuild with the same
//                         multipliers. Banks are the high bits of the
//                         hash, so raising R_LOG SPLITS banks: colliding-
//                         after implies colliding-before, hence expansion
//                         is provably demotion-free. assemble's assert
//                         re-proves it at runtime on every call.
//   shrink                is just to_banked at smaller logs: merging
//                         banks can collide, so it goes through the
//                         normal importance contests. No new code.
//
// GPU mapping: probe + dedup stay host-side with candidate selection;
// a batch of commits is a STAGE_TRADE record list plus a byte scatter.
// ===========================================================================
#pragma once
#include "dual_codec_ell_cpu.hpp"

namespace sfc {

enum class GrowStatus { Free, AlreadyPresent, RowBankBusy, ColBankBusy, BothBusy };

struct GrowProbe {
    GrowStatus st;
    uint64_t k;            // target row slot for (i, j)
    uint64_t k2;           // slot of column j's current row-residue holder
    uint32_t row_evict_j;  // valid when row bank busy: j currently at k
    uint32_t col_evict_i;  // valid when col bank busy: i currently at k2
};

inline GrowProbe probe_grow(const BankedLayer& L, uint32_t i, uint32_t j) {
    const uint32_t R = L.R(), C = L.C(), LOG_S = L.LOG_S;
    const uint32_t NM = (1u << L.KC()) - 1u, MM = (1u << L.KR()) - 1u;
    const uint32_t ip = (L.aR * i) & MM;
    const uint32_t jp = (L.aC * j) & NM;
    const uint32_t p = ip >> LOG_S, s = jp >> LOG_S;
    GrowProbe pr;
    pr.k = uint64_t(ip) * R + s;
    const uint32_t u2 = L.fget(L.inv_bits, uint64_t(jp) * C + p);
    pr.k2 = (uint64_t((p << LOG_S) | u2)) * R + s;
    const bool rb = L.syn[size_t(pr.k)] != 0;
    const bool cb = L.syn[size_t(pr.k2)] != 0;
    const uint32_t vcur = L.fget(L.pi_bits, pr.k);
    pr.row_evict_j = (L.aCinv * ((s << LOG_S) | vcur)) & NM;
    pr.col_evict_i = (L.aRinv * ((p << LOG_S) | u2)) & MM;
    if (rb && pr.row_evict_j == j) { pr.st = GrowStatus::AlreadyPresent; return pr; }
    pr.st = (rb && cb) ? GrowStatus::BothBusy
          : rb ? GrowStatus::RowBankBusy
          : cb ? GrowStatus::ColBankBusy
          : GrowStatus::Free;
    return pr;
}

// Requires a Free probe. If the slot is already wired to j (a pruned
// ancestor, or filler luck), the trade is skipped entirely.
inline bool commit_grow(BankedLayer& L, uint32_t i, uint32_t j, uint8_t byte) {
    GrowProbe pr = probe_grow(L, i, j);
    if (pr.st != GrowStatus::Free) return false;
    const uint32_t LOG_S = L.LOG_S;
    const uint32_t jp = (L.aC * j) & ((1u << L.KC()) - 1u);
    const uint32_t vt = jp & (L.S() - 1u), s = jp >> LOG_S;
    if (L.fget(L.pi_bits, pr.k) != vt) {
        const uint32_t ip = uint32_t(pr.k >> L.R_LOG);
        const uint32_t ip2 = uint32_t(pr.k2 >> L.R_LOG);
        Trade t{ip >> LOG_S, s, ip & (L.S() - 1u), ip2 & (L.S() - 1u)};
        trade_banked_cpu(L, &t, 1);          // silent-silent transposition
    }
    L.syn[size_t(pr.k)] = byte;
    return true;
}

// Returns true if (i, j) was live and is now silent. Wiring untouched.
inline bool prune(BankedLayer& L, uint32_t i, uint32_t j) {
    GrowProbe pr = probe_grow(L, i, j);
    if (pr.st != GrowStatus::AlreadyPresent) return false;
    L.syn[size_t(pr.k)] = 0;
    return true;
}

inline uint64_t live_count(const BankedLayer& L) {
    uint64_t n = 0;
    for (uint8_t b : L.syn) n += (b != 0);
    return n;
}

// ---------------------------------------------------------------------------
// Assemble from already-placed synapses with FIXED multipliers and a hard
// assert on bank collisions. Used for empty init and for expansion, where
// collisions are impossible by the bank-splitting argument; the assert
// re-proves that on every call instead of trusting the theorem.
// ---------------------------------------------------------------------------
inline BankedLayer assemble_banked(const std::vector<Syn>& placed,
                                   uint32_t LOG_S, uint32_t R_LOG, uint32_t C_LOG,
                                   uint32_t aC, uint32_t aR) {
    BankedLayer L;
    L.LOG_S = LOG_S; L.R_LOG = R_LOG; L.C_LOG = C_LOG;
    L.aC = aC; L.aCinv = modinv_pow2(aC, L.KC());
    L.aR = aR; L.aRinv = modinv_pow2(aR, L.KR());
    const uint32_t S = L.S(), R = L.R(), C = L.C();
    const uint64_t nslots = uint64_t(S) * C * R;
    L.pi_bits.assign(size_t((nslots * LOG_S + 31) / 32 + 1), 0u);
    L.inv_bits.assign(size_t((nslots * LOG_S + 31) / 32 + 1), 0u);
    L.syn.assign(size_t(nslots), 0u);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> plane(size_t(C) * R);
    const uint32_t NM = (1u << L.KC()) - 1u, MM = (1u << L.KR()) - 1u;
    for (const auto& sy : placed) {
        const uint32_t ip = (aR * sy.i) & MM, jp = (aC * sy.j) & NM;
        const uint32_t p = ip >> LOG_S, u = ip & (S - 1);
        const uint32_t sb = jp >> LOG_S, v = jp & (S - 1);
        plane[size_t(p) * R + sb].push_back({u, v});
        const uint64_t k = uint64_t(ip) * R + sb;
        L.fset(L.pi_bits, k, v);
        L.fset(L.inv_bits, uint64_t(jp) * C + p, u);
        L.syn[size_t(k)] = sy.b;
    }
    std::vector<uint8_t> used_u(S), used_v(S);
    for (uint32_t p = 0; p < C; ++p)
        for (uint32_t sb = 0; sb < R; ++sb) {
            std::fill(used_u.begin(), used_u.end(), 0);
            std::fill(used_v.begin(), used_v.end(), 0);
            for (auto& uv : plane[size_t(p) * R + sb]) {
                assert(!used_u[uv.first] && "row-bank collision in assemble");
                assert(!used_v[uv.second] && "col-bank collision in assemble");
                used_u[uv.first] = 1;
                used_v[uv.second] = 1;
            }
            uint32_t v = 0;
            for (uint32_t u = 0; u < S; ++u) {
                if (used_u[u]) continue;
                while (used_v[v]) ++v;
                const uint32_t ip = (p << LOG_S) | u, jp = (sb << LOG_S) | v;
                L.fset(L.pi_bits, uint64_t(ip) * R + sb, v);
                L.fset(L.inv_bits, uint64_t(jp) * C + p, u);
                ++v;
            }
        }
    return L;
}

// Demotion-free capacity growth: same multipliers, more bank bits.
inline BankedLayer expand_banked(const BankedLayer& L, uint32_t addR, uint32_t addC,
                                 uint32_t M_real, uint32_t N_real) {
    std::vector<Syn> keep = to_syns(L, M_real, N_real);
    return assemble_banked(keep, L.LOG_S, L.R_LOG + addR, L.C_LOG + addC,
                           L.aC, L.aR);
}

// ===========================================================================
// Packed-codec mechanism. Same lifecycle, different cost profile: packed
// has NO bank constraint -- any (i, j) fits while row i and column j
// have live headroom, which is the entire blocking rule. Pruning is in
// place; growth commits by batched re-encode (in sili, the interleaved-
// slack uleb encoder; here, a rebuild); capacity change is a rebuild at
// the new R or C and is trivially lossless in BOTH directions, because
// with no banks there is nothing to collide and nothing to contest.
// ===========================================================================
inline uint32_t packed_slot_of(const PackedGpu& P, uint32_t i, uint32_t j) {
    for (uint32_t s = 0; s < P.R; ++s)
        if (P.syn[size_t(uint64_t(i) * P.R + s)] != 0 &&
            packed_id(P.rowv, i, s) == j) return s;
    return P.R;                                   // not live
}
inline bool prune_packed(PackedGpu& P, uint32_t i, uint32_t j) {
    const uint32_t s = packed_slot_of(P, i, j);
    if (s == P.R) return false;
    P.syn[size_t(uint64_t(i) * P.R + s)] = 0;     // tombstone id stays until
    return true;                                  // the next re-encode
}
inline std::vector<Syn> packed_to_syns(const PackedGpu& P) {
    std::vector<Syn> out;
    for (uint32_t i = 0; i < P.M_real; ++i)
        for (uint32_t s = 0; s < P.R; ++s) {
            const uint8_t b = P.syn[size_t(uint64_t(i) * P.R + s)];
            if (b) out.push_back({i, packed_id(P.rowv, i, s), b});
        }
    return out;
}
inline uint64_t packed_live_count(const PackedGpu& P) {
    uint64_t n = 0;
    for (uint32_t i = 0; i < P.M_real; ++i)
        for (uint32_t s = 0; s < P.R; ++s)
            n += (P.syn[size_t(uint64_t(i) * P.R + s)] != 0);
    return n;
}
// Batched growth commit and/or capacity change: current live plus adds,
// re-encoded at (newR, newC). Caller enforces capacity beforehand;
// build_packed's asserts re-check it.
inline PackedGpu regrow_packed(const PackedGpu& P, const std::vector<Syn>& adds,
                               uint32_t newR, uint32_t newC, uint32_t newR_LOG) {
    std::vector<Syn> all = packed_to_syns(P);
    all.insert(all.end(), adds.begin(), adds.end());
    return build_packed(all, P.M_real, P.N_real, newR, newC, newR_LOG);
}

} // namespace sfc
