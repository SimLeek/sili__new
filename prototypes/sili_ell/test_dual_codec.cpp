// test_dual_codec.cpp -- correctness + benchmark harness for the CPU
// translation of the dual-codec kernels. Small config gets exhaustive
// checks against independent scalar references; large config gets timed.
#include "dual_codec_ell_cpu.hpp"
#include <cstdio>
#include <map>
#include <set>
#include <omp.h>

using namespace sfc;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; \
    std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static uint32_t slot_of(const BankedLayer& L, uint32_t i, uint32_t j) {
    uint32_t ip = (L.aR * i) & ((1u << L.KR()) - 1u);
    uint32_t jp = (L.aC * j) & ((1u << L.KC()) - 1u);
    return ip * L.R() + (jp >> L.LOG_S);
}

static void audit_inverse(const BankedLayer& L, const char* tag) {
    const uint32_t R = L.R(), C = L.C(), S = L.S();
    for (uint32_t ip = 0; ip < (1u << L.KR()); ++ip)
        for (uint32_t s = 0; s < R; ++s) {
            uint32_t v = L.fget(L.pi_bits, uint64_t(ip) * R + s);
            uint32_t jp = (s << L.LOG_S) | v;
            uint32_t u2 = L.fget(L.inv_bits, uint64_t(jp) * C + (ip >> L.LOG_S));
            CHECK(u2 == (ip & (S - 1)), "%s pi/inv mismatch ip=%u s=%u", tag, ip, s);
        }
    for (uint32_t jp = 0; jp < (1u << L.KC()); ++jp)
        for (uint32_t p = 0; p < C; ++p) {
            uint32_t u = L.fget(L.inv_bits, uint64_t(jp) * C + p);
            uint32_t ip = (p << L.LOG_S) | u;
            uint32_t v2 = L.fget(L.pi_bits, uint64_t(ip) * R + (jp >> L.LOG_S));
            CHECK(v2 == (jp & (S - 1)), "%s inv/pi mismatch jp=%u p=%u", tag, jp, p);
        }
}

int main() {
    std::printf("threads=%d\n", omp_get_max_threads());
    const float SC = 0.0625f;

    // ---------------- small config: exhaustive correctness ----------------
    const uint32_t LOG_S = 6, R_LOG = 3, C_LOG = 3;
    const uint32_t M = 400, N = 450, R = 1u << R_LOG, C = 1u << C_LOG;
    std::mt19937_64 rng(12345);
    std::set<uint64_t> seen;
    std::vector<Syn> syns;
    while (syns.size() < 1600) {
        uint32_t i = uint32_t(rng() % M), j = uint32_t(rng() % N);
        if (!seen.insert((uint64_t(i) << 32) | j).second) continue;
        uint8_t b = uint8_t((1 + rng() % 15) << 4 | (1 + rng() % 15));
        syns.push_back({i, j, b});
    }

    ToBankedResult tb = to_banked(syns, LOG_S, R_LOG, C_LOG, M, N, 32, 7);
    BankedLayer& L = tb.layer;
    std::printf("to_banked: kept=%zu regrow=%zu demoted_frac=%.4f aC=%u aR=%u\n",
                syns.size() - tb.regrow.size(), tb.regrow.size(),
                tb.demoted_frac, L.aC, L.aR);
    CHECK(L.aCinv * L.aC % (1u << L.KC()) == 1u, "aCinv wrong");
    CHECK(L.aRinv * L.aR % (1u << L.KR()) == 1u, "aRinv wrong");
    audit_inverse(L, "post-convert");

    std::vector<Syn> kept = to_syns(L, M, N);
    CHECK(kept.size() + tb.regrow.size() == syns.size(),
          "kept+regrow=%zu != input=%zu", kept.size() + tb.regrow.size(), syns.size());
    {   // every kept synapse must be an input synapse with its byte intact
        std::map<uint64_t, uint8_t> in;
        for (auto& s : syns) in[(uint64_t(s.i) << 32) | s.j] = s.b;
        for (auto& s : kept) {
            auto it = in.find((uint64_t(s.i) << 32) | s.j);
            CHECK(it != in.end() && it->second == s.b, "kept synapse corrupted");
        }
    }

    // inputs
    std::vector<float> x(N), dy(M);
    std::vector<uint32_t> xb((N + 31) / 32, 0), db((M + 31) / 32, 0);
    for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 500.f;
    for (auto& v : dy) v = float(int(rng() % 2001) - 1000) / 500.f;
    for (uint32_t j = 0; j < N; ++j) if (rng() % 10 < 7) xb[j >> 5] |= 1u << (j & 31);
    for (uint32_t i = 0; i < M; ++i) if (rng() % 10 < 6) db[i >> 5] |= 1u << (i & 31);

    // references from the kept list
    std::vector<float> yref(M, 0.f), dxref(N, 0.f);
    for (auto& s : kept) {
        float w = W_LUT[s.b & 15] * SC;
        if ((xb[s.j >> 5] >> (s.j & 31)) & 1u) yref[s.i] += w * x[s.j];
        if ((db[s.i >> 5] >> (s.i & 31)) & 1u) dxref[s.j] += w * dy[s.i];
    }

    std::vector<float> y(M, -1.f), dx(N, -1.f);
    forward_banked_cpu(L, M, x.data(), xb.data(), y.data(), SC);
    for (uint32_t i = 0; i < M; ++i)
        CHECK(std::abs(y[i] - yref[i]) <= 1e-4f * (1.f + std::abs(yref[i])),
              "fwd banked i=%u got %f want %f", i, y[i], yref[i]);

    dx_banked_cpu(L, 1u << L.KC(), N, dy.data(), db.data(), dx.data(), SC);
    for (uint32_t j = 0; j < N; ++j)
        CHECK(std::abs(dx[j] - dxref[j]) <= 1e-4f * (1.f + std::abs(dxref[j])),
              "dx banked j=%u got %f want %f", j, dx[j], dxref[j]);

    PackedGpu P = build_packed(kept, M, N, R, C, R_LOG);
    std::fill(y.begin(), y.end(), -1.f);
    std::fill(dx.begin(), dx.end(), -1.f);
    forward_packed_cpu(P, x.data(), xb.data(), y.data(), SC);
    for (uint32_t i = 0; i < M; ++i)
        CHECK(std::abs(y[i] - yref[i]) <= 1e-4f * (1.f + std::abs(yref[i])),
              "fwd packed i=%u got %f want %f", i, y[i], yref[i]);
    dx_packed_cpu(P, dy.data(), db.data(), dx.data(), SC);
    for (uint32_t j = 0; j < N; ++j)
        CHECK(std::abs(dx[j] - dxref[j]) <= 1e-4f * (1.f + std::abs(dxref[j])),
              "dx packed j=%u got %f want %f", j, dx[j], dxref[j]);

    // ---- update: byte-exact against independent slot addressing ----
    const float LR = 0.11f;
    const uint32_t SEED = 0xC0FFEEu;
    std::vector<uint32_t> act;
    std::vector<float> actg;
    for (uint32_t i = 0; i < M; i += 7) { act.push_back(i); actg.push_back(dy[i]); }

    {   // banked
        std::vector<uint8_t> before = L.syn;
        std::map<uint64_t, uint8_t> expect;
        std::set<uint32_t> actset(act.begin(), act.end());
        for (auto& s : kept) {
            if (!actset.count(s.i)) continue;
            float g = dy[s.i];
            float xv = ((xb[s.j >> 5] >> (s.j & 31)) & 1u) ? x[s.j] : 0.f;
            float upd = LR * g * xv;
            if (upd == 0.f) continue;
            uint32_t k = slot_of(L, s.i, s.j);
            uint32_t code = s.b & 15u, imp = s.b >> 4;
            float wnew = W_LUT[code] * SC - upd;
            uint32_t nc = quantize(wnew / SC, wang(k ^ SEED));
            if (nc != code && imp < 15u) imp += 1u;
            expect[k] = uint8_t((imp << 4) | nc);
        }
        wupdate_banked_cpu(L, act.data(), actg.data(), uint32_t(act.size()),
                           x.data(), xb.data(), LR, SEED, SC);
        for (uint64_t k = 0; k < L.syn.size(); ++k) {
            auto it = expect.find(k);
            uint8_t want = (it != expect.end()) ? it->second : before[size_t(k)];
            CHECK(L.syn[size_t(k)] == want, "wupd banked k=%llu got %u want %u",
                  (unsigned long long)k, L.syn[size_t(k)], want);
        }
    }
    audit_inverse(L, "post-update");

    {   // packed (fresh copy, its own slot addressing)
        PackedGpu Q = build_packed(kept, M, N, R, C, R_LOG);
        std::vector<uint8_t> before = Q.syn;
        // recompute row slot positions the way the builder does
        std::vector<std::vector<std::pair<uint32_t, uint8_t>>> rows(M);
        for (auto& s : kept) rows[s.i].push_back({s.j, s.b});
        std::map<uint64_t, uint8_t> expect;
        std::set<uint32_t> actset(act.begin(), act.end());
        for (uint32_t i = 0; i < M; ++i) {
            std::sort(rows[i].begin(), rows[i].end());
            if (!actset.count(i)) continue;
            for (uint32_t s = 0; s < uint32_t(rows[i].size()); ++s) {
                uint32_t j = rows[i][s].first;
                uint8_t b = rows[i][s].second;
                float xv = ((xb[j >> 5] >> (j & 31)) & 1u) ? x[j] : 0.f;
                float upd = LR * dy[i] * xv;
                if (upd == 0.f) continue;
                uint64_t k = uint64_t(i) * R + s;
                uint32_t code = b & 15u, imp = b >> 4;
                float wnew = W_LUT[code] * SC - upd;
                uint32_t nc = quantize(wnew / SC, wang(uint32_t(k) ^ SEED));
                if (nc != code && imp < 15u) imp += 1u;
                expect[k] = uint8_t((imp << 4) | nc);
            }
        }
        wupdate_packed_cpu(Q, act.data(), actg.data(), uint32_t(act.size()),
                           x.data(), xb.data(), LR, SEED, SC);
        for (uint64_t k = 0; k < Q.syn.size(); ++k) {
            auto it = expect.find(k);
            uint8_t want = (it != expect.end()) ? it->second : before[size_t(k)];
            CHECK(Q.syn[size_t(k)] == want, "wupd packed k=%llu got %u want %u",
                  (unsigned long long)k, Q.syn[size_t(k)], want);
        }
    }

    // ---- trades: invariants hold, traded slots die, bystanders live ----
    {
        std::vector<Syn> beforeS = to_syns(L, M, N);
        std::vector<std::pair<uint32_t, uint32_t>> planes;
        for (uint32_t p = 0; p < C; ++p)
            for (uint32_t s = 0; s < R; ++s) planes.push_back({p, s});
        std::shuffle(planes.begin(), planes.end(), rng);
        std::vector<Trade> trades;
        std::set<uint64_t> tradedk;
        for (uint32_t t = 0; t < 40; ++t) {
            uint32_t u1 = uint32_t(rng() % L.S()), u2 = uint32_t(rng() % L.S());
            if (u1 == u2) u2 = (u2 + 1) % L.S();
            trades.push_back({planes[t].first, planes[t].second, u1, u2});
            for (uint32_t u : {u1, u2}) {
                uint32_t ip = (planes[t].first << LOG_S) | u;
                tradedk.insert(uint64_t(ip) * R + planes[t].second);
            }
        }
        trade_banked_cpu(L, trades.data(), trades.size());
        audit_inverse(L, "post-trade");
        for (uint64_t k : tradedk)
            CHECK(L.syn[size_t(k)] == 0, "traded slot %llu not silent",
                  (unsigned long long)k);
        std::set<uint64_t> afterSet;
        for (auto& s : to_syns(L, M, N))
            afterSet.insert((uint64_t(s.i) << 32) | s.j);
        for (auto& s : beforeS) {
            bool traded = tradedk.count(slot_of(L, s.i, s.j)) != 0;
            bool present = afterSet.count((uint64_t(s.i) << 32) | s.j) != 0;
            CHECK(present == !traded, "bystander wrong i=%u j=%u", s.i, s.j);
        }
    }

    std::printf("small-config checks: %s (%d failures)\n\n",
                g_fail ? "FAIL" : "PASS", g_fail);

    // ---- structured-graph audition: 3x3 RGB receptive fields on a
    // 210x160x3 raster (strides 0,3,6,480,483,486,960,963,966). With
    // identity banking a whole patch lands in one bank and most of it
    // would demote; the multiplier search should scramble that back to
    // the random-graph baseline. ----
    {
        const uint32_t sN = 100800, sLOG = 12, sRL = 5;
        const uint32_t offs[9] = {0, 3, 6, 480, 483, 486, 960, 963, 966};
        std::vector<Syn> ss;
        for (uint32_t i = 0; i < 20000; ++i) {
            uint32_t base = uint32_t(rng() % (sN - 967));
            for (uint32_t o : offs) ss.push_back({i, base + o, 0x11});
        }
        auto demoted_for = [&](uint32_t a) {
            uint32_t mask = (1u << (sLOG + sRL)) - 1u;
            uint64_t dem = 0;
            uint32_t cnt[32];
            for (uint32_t i = 0; i < 20000; ++i) {
                for (auto& c : cnt) c = 0;
                for (uint32_t t = 0; t < 9; ++t)
                    cnt[((a * ss[size_t(i) * 9 + t].j) & mask) >> sLOG]++;
                for (auto c : cnt) if (c > 1) dem += c - 1;
            }
            return double(dem) / double(ss.size());
        };
        Audition sa = audition(ss, true, sLOG + sRL, sRL, 20000, 48, 77);
        double d1 = demoted_for(1u), db = demoted_for(sa.a);
        std::printf("stride demo: demoted identity=%.1f%%  audition-best=%.1f%%\n\n",
                    100.0 * d1, 100.0 * db);
        CHECK(db < 0.20 && d1 > 0.5, "audition did not beat identity: %f vs %f",
              db, d1);
    }

    // ---------------- large config: benchmark + controller ----------------
    const uint32_t bLOG_S = 12, bRL = 5, bCL = 5;
    const uint32_t bM = 100000, bN = 100800, bR = 32, bC = 32;
    std::vector<Syn> big;
    big.reserve(size_t(bM) * 18);
    {
        std::set<uint32_t> rowseen;
        for (uint32_t i = 0; i < bM; ++i) {
            rowseen.clear();
            while (rowseen.size() < 18) rowseen.insert(uint32_t(rng() % bN));
            for (uint32_t j : rowseen)
                big.push_back({i, j, uint8_t((1 + rng() % 15) << 4 | (1 + rng() % 15))});
        }
    }
    double t0 = omp_get_wtime();
    ToBankedResult bt = to_banked(big, bLOG_S, bRL, bCL, bM, bN, 32, 9);
    double t1 = omp_get_wtime();
    std::vector<Syn> bkept = to_syns(bt.layer, bM, bN);
    std::printf("big to_banked: %.2fs  in=%zu kept=%zu demoted=%.3f%%\n",
                t1 - t0, big.size(), bkept.size(), 100.0 * bt.demoted_frac);

    PackedGpu BP = build_packed(bkept, bM, bN, bR, bC, bRL);
    double ppb = packed_bits_per_param(BP.rowv, BP.colv, bRL);
    double bpb = banked_bits_per_param(bLOG_S);
    double est = estimate_packed_row_bits(bt.layer, bM, bN, 2000, 42);
    std::printf("bits per stored slot: packed=%.1f banked=%.1f (sampled row-index est %.1f b/idx)\n",
                ppb, bpb, est);
    Controller ctl;
    ctl.LOG_S = bLOG_S;
    Fmt f = Fmt::Packed;
    for (int e = 0; e < 5; ++e)
        f = ctl.decide(f, {ppb, bt.demoted_frac});
    std::printf("controller after 5 evals: %s\n", f == Fmt::Banked ? "Banked" : "Packed");

    std::vector<float> bx(bN), bdy(bM), by(bM), bdx(bN);
    std::vector<uint32_t> bxb((bN + 31) / 32, 0), bdb((bM + 31) / 32, 0);
    for (auto& v : bx) v = 0.5f;
    for (auto& v : bdy) v = 0.25f;
    for (uint32_t j = 0; j < bN; ++j) if (rng() % 100 < 30) bxb[j >> 5] |= 1u << (j & 31);
    for (uint32_t i = 0; i < bM; ++i) if (rng() % 100 < 5) bdb[i >> 5] |= 1u << (i & 31);

    auto bench = [&](const char* name, auto&& fn, double work) {
        fn();                                      // warm
        double best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            double a = omp_get_wtime();
            fn();
            best = std::min(best, omp_get_wtime() - a);
        }
        std::printf("%-16s %8.3f ms   %7.1f Msyn/s\n", name, best * 1e3,
                    work / best * 1e-6);
    };
    double fw = double(bM) * bR, dxw = double(1u << (bLOG_S + bRL)) * bC;
    bench("fwd banked", [&] { forward_banked_cpu(bt.layer, bM, bx.data(), bxb.data(), by.data()); }, fw);
    bench("fwd packed", [&] { forward_packed_cpu(BP, bx.data(), bxb.data(), by.data()); }, fw);
    bench("dx  banked", [&] { dx_banked_cpu(bt.layer, 1u << (bLOG_S + bRL), bN, bdy.data(), bdb.data(), bdx.data()); }, dxw);
    bench("dx  packed", [&] { dx_packed_cpu(BP, bdy.data(), bdb.data(), bdx.data()); }, double(bN) * bC);
    // Cross-check BEFORE any updates: the codecs hash different slot ids
    // into the stochastic rounder, so identical training legitimately
    // diverges the two representations. Determinism holds within a codec,
    // not across codecs.
    forward_banked_cpu(bt.layer, bM, bx.data(), bxb.data(), by.data());
    std::vector<float> by2(bM);
    forward_packed_cpu(BP, bx.data(), bxb.data(), by2.data());
    int bad = 0;
    for (uint32_t i = 0; i < bM; ++i)
        if (std::abs(by[i] - by2[i]) > 1e-4f * (1.f + std::abs(by[i]))) ++bad;
    CHECK(bad == 0, "big cross-check mismatches: %d", bad);

    std::vector<uint32_t> bact;
    std::vector<float> bactg;
    for (uint32_t i = 0; i < bM; i += 20) { bact.push_back(i); bactg.push_back(0.25f); }
    bench("wupd banked", [&] { wupdate_banked_cpu(bt.layer, bact.data(), bactg.data(), uint32_t(bact.size()), bx.data(), bxb.data(), 0.01f, 1u); }, double(bact.size()) * bR);
    bench("wupd packed", [&] { wupdate_packed_cpu(BP, bact.data(), bactg.data(), uint32_t(bact.size()), bx.data(), bxb.data(), 0.01f, 1u); }, double(bact.size()) * bR);

    std::printf("\nTOTAL: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
