// test_succinct.cpp -- correctness and cost measurement for the succinct
// inverse. Small config: exhaustive equality vs the stored inverse,
// exact dx parity, trade/dirty/fallback/rebuild lifecycle. Large config:
// t-sweep of bits per slot, average hops, and dx throughput vs the full
// 32-bit layout.
#include "succinct_inverse.hpp"
#include <cstdio>
#include <set>
#include <omp.h>

using namespace sfc;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; \
    std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// A uniformly random member of the banked family with given fill.
static BankedLayer random_family(uint32_t LOG_S, uint32_t RL, uint32_t CL,
                                 double fill, std::mt19937_64& rng) {
    BankedLayer L;
    L.LOG_S = LOG_S; L.R_LOG = RL; L.C_LOG = CL;
    L.aC = L.aR = 1u;
    L.aCinv = modinv_pow2(1u, L.KC());
    L.aRinv = modinv_pow2(1u, L.KR());
    const uint32_t S = L.S(), R = L.R(), C = L.C();
    const uint64_t nslots = uint64_t(S) * C * R;
    L.pi_bits.assign(size_t((nslots * LOG_S + 31) / 32 + 1), 0u);
    L.inv_bits.assign(size_t((nslots * LOG_S + 31) / 32 + 1), 0u);
    L.syn.assign(size_t(nslots), 0u);
    std::vector<uint32_t> perm(S);
    for (uint32_t p = 0; p < C; ++p)
        for (uint32_t s = 0; s < R; ++s) {
            for (uint32_t u = 0; u < S; ++u) perm[u] = u;
            std::shuffle(perm.begin(), perm.end(), rng);
            for (uint32_t u = 0; u < S; ++u) {
                const uint32_t v = perm[u];
                L.fset(L.pi_bits, (uint64_t((p << LOG_S) | u)) * R + s, v);
                L.fset(L.inv_bits, (uint64_t((s << LOG_S) | v)) * C + p, u);
            }
        }
    const uint32_t thresh = uint32_t(fill * 1000.0);
    for (uint64_t k = 0; k < nslots; ++k)
        if (rng() % 1000 < thresh)
            L.syn[size_t(k)] = uint8_t((1 + rng() % 15) << 4 | (1 + rng() % 15));
    return L;
}

static void check_all_inverses(const BankedLayer& L, const SuccinctInv& si,
                               const char* tag, uint64_t* hops = nullptr,
                               uint64_t* fbs = nullptr) {
    const uint32_t C = L.C(), S = L.S();
    for (uint32_t jp = 0; jp < (1u << L.KC()); ++jp)
        for (uint32_t p = 0; p < C; ++p) {
            const uint32_t want = L.fget(L.inv_bits, uint64_t(jp) * C + p);
            const uint32_t got = si_inv(L, si, p, jp >> L.LOG_S,
                                        jp & (S - 1u), hops, fbs);
            if (got != want) {
                CHECK(false, "%s si_inv mismatch jp=%u p=%u got=%u want=%u",
                      tag, jp, p, got, want);
                return;
            }
        }
}

int main() {
    std::printf("threads=%d\n", omp_get_max_threads());
    std::mt19937_64 rng(20260728);

    // ------------------- small config: exhaustive -------------------
    {
        const uint32_t LOG_S = 6, RL = 3, CL = 3, M = 300, N = 350;
        std::set<uint64_t> seen;
        std::vector<Syn> syns;
        while (syns.size() < 1500) {
            uint32_t i = uint32_t(rng() % M), j = uint32_t(rng() % N);
            if (!seen.insert((uint64_t(i) << 32) | j).second) continue;
            syns.push_back({i, j, uint8_t((1 + rng() % 15) << 4 | (1 + rng() % 15))});
        }
        ToBankedResult tb = to_banked(syns, LOG_S, RL, CL, M, N, 16, 3);
        BankedLayer& L = tb.layer;

        SuccinctInv si = build_succinct_inv(L, 4);
        uint64_t hops = 0, fbs = 0;
        check_all_inverses(L, si, "clean", &hops, &fbs);
        const uint64_t nq = uint64_t(1u << L.KC()) * L.C();
        std::printf("small clean: %.2f avg hops (t=%u), fallbacks=%llu\n",
                    double(hops) / double(nq), si.t, (unsigned long long)fbs);
        CHECK(fbs == 0, "fallbacks on a clean structure");
        CHECK(double(hops) / double(nq) <= si.t + 3.0, "hops above expectation");

        // exact dx parity, full inverse vs succinct
        std::vector<float> dy(M), dx1(N, -1.f), dx2(N, -1.f);
        std::vector<uint32_t> db((M + 31) / 32, 0);
        for (auto& v : dy) v = float(int(rng() % 2001) - 1000) / 500.f;
        for (uint32_t i = 0; i < M; ++i)
            if (rng() % 10 < 6) db[i >> 5] |= 1u << (i & 31);
        dx_banked_cpu(L, 1u << L.KC(), N, dy.data(), db.data(), dx1.data());
        dx_banked_si_cpu(L, si, 1u << L.KC(), N, dy.data(), db.data(), dx2.data());
        for (uint32_t j = 0; j < N; ++j)
            CHECK(std::abs(dx1[j] - dx2[j]) <= 1e-5f * (1.f + std::abs(dx1[j])),
                  "dx parity j=%u %f vs %f", j, dx1[j], dx2[j]);

        // trades: correctness through dirty (fallback) and after rebuild
        std::vector<std::pair<uint32_t, uint32_t>> planes;
        for (uint32_t p = 0; p < L.C(); ++p)
            for (uint32_t s = 0; s < L.R(); ++s) planes.push_back({p, s});
        std::shuffle(planes.begin(), planes.end(), rng);
        for (uint32_t t = 0; t < 25; ++t) {
            uint32_t u1 = uint32_t(rng() % L.S()), u2 = uint32_t(rng() % L.S());
            if (u1 == u2) u2 = (u2 + 1) % L.S();
            Trade tr{planes[t].first, planes[t].second, u1, u2};
            trade_banked_cpu(L, &tr, 1);
            mark_trade_dirty(si, tr);
        }
        uint64_t fb2 = 0;
        check_all_inverses(L, si, "dirty", nullptr, &fb2);
        std::printf("small dirty: fallbacks=%llu (expected > 0)\n",
                    (unsigned long long)fb2);
        CHECK(fb2 > 0, "dirty planes never fell back -- test not exercising it");

        finalize_trades(L, si);
        uint64_t fb3 = 0;
        check_all_inverses(L, si, "rebuilt", nullptr, &fb3);
        CHECK(fb3 == 0, "fallbacks after rebuild");
        std::printf("small config: %s\n\n", g_fail ? "FAIL" : "PASS");
    }

    // -------------- large config: t sweep, bits and speed --------------
    {
        const uint32_t LOG_S = 12, RL = 5, CL = 5;
        BankedLayer L = random_family(LOG_S, RL, CL, 0.35, rng);
        const uint32_t nkeys = 1u << L.KC();
        const uint32_t NR = nkeys;             // identity multipliers
        std::vector<float> dy(1u << L.KR()), dx1(NR), dx2(NR);
        std::vector<uint32_t> db(((1u << L.KR()) + 31) / 32, 0xffffffffu);
        for (auto& v : dy) v = 0.25f;

        auto bench = [&](auto&& fn) {
            fn();
            double best = 1e9;
            for (int rep = 0; rep < 5; ++rep) {
                double a = omp_get_wtime();
                fn();
                best = std::min(best, omp_get_wtime() - a);
            }
            return best;
        };
        const double work = double(nkeys) * L.C();
        double tfull = bench([&] {
            dx_banked_cpu(L, nkeys, NR, dy.data(), db.data(), dx1.data());
        });
        std::printf("layout            bits/slot   dx ms   Msyn/s   avg hops\n");
        std::printf("full inverse         32.00  %7.2f  %7.1f       1.00\n",
                    tfull * 1e3, work / tfull * 1e-6);

        for (uint32_t t : {4u, 6u, 8u, 12u}) {
            double b0 = omp_get_wtime();
            SuccinctInv si = build_succinct_inv(L, t);
            double bt = omp_get_wtime() - b0;
            double ts = bench([&] {
                dx_banked_si_cpu(L, si, nkeys, NR, dy.data(), db.data(), dx2.data());
            });
            bool eq = true;
            for (uint32_t j = 0; j < NR; ++j)
                if (std::abs(dx1[j] - dx2[j]) > 1e-5f * (1.f + std::abs(dx1[j]))) { eq = false; break; }
            CHECK(eq, "large dx parity failed at t=%u", t);
            // sampled hop count (sequential, stats enabled)
            uint64_t hops = 0, fbs = 0;
            for (int q = 0; q < 200000; ++q) {
                uint32_t jp = uint32_t(rng() % nkeys);
                uint32_t p = uint32_t(rng() % L.C());
                si_inv(L, si, p, jp >> LOG_S, jp & (L.S() - 1u), &hops, &fbs);
            }
            CHECK(fbs == 0, "fallbacks on clean large structure t=%u", t);
            double bps = double(LOG_S) + 8.0 + si.bits_per_slot();
            std::printf("succinct t=%-2u        %5.2f  %7.2f  %7.1f      %5.2f"
                        "   (build %.0f ms)\n",
                        t, bps, ts * 1e3, work / ts * 1e-6,
                        double(hops) / 200000.0, bt * 1e3);
        }
    }

    std::printf("\nTOTAL: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
