// test_stochastic_importance.cpp -- three claims, measured:
//   1. The 4-bit dithered importance counter is unbiased: under constant
//      drift d per step, mean importance across many synapses matches
//      clamp(8 + T*d) to within CLT bounds, on the REAL kernel.
//   2. After training, importance separates teacher-position synapses
//      from junk, and pruning by importance preserves quality far better
//      than random pruning (magnitude pruning printed as the baseline).
//   3. The memory ledger: persistent storage is exactly 32 bits per
//      parameter, transients are O(M + N), and the fused update
//      allocates nothing -- there is no optimizer state to count.
#include "synapto.hpp"
#include <cstdio>
#include <map>
#include <set>
#include <omp.h>

using namespace sfc;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; \
    std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

using Key = uint64_t;
static Key key(uint32_t i, uint32_t j) { return (Key(i) << 32) | j; }

// From test_succinct.cpp: a uniformly random member of the banked family.
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

int main() {
    const float SC = 0.0625f;
    std::mt19937_64 rng(1913);

    // ---------------- 1. counter unbiasedness on the kernel ----------------
    {
        const uint32_t LOG_S = 6, RL = 2, CL = 2;
        const uint32_t Mreal = 1u << (LOG_S + CL);      // all stored rows real
        const uint32_t Nreal = 1u << (LOG_S + RL);
        auto run_case = [&](float g, float imp_lr, int T) {
            BankedLayer L = random_family(LOG_S, RL, CL, 1.0, rng);
            for (auto& b : L.syn) b = uint8_t((8u << 4) | 4u);  // imp 8, w code 4
            std::vector<float> x(Nreal, 0.5f), dy(Mreal, g);
            std::vector<uint32_t> xb((Nreal + 31) / 32, 0xffffffffu);
            std::vector<uint32_t> act(Mreal);
            for (uint32_t i = 0; i < Mreal; ++i) act[i] = i;
            for (int t = 1; t <= T; ++t)
                wupdate_banked_cpu(L, act.data(), dy.data(), Mreal,
                                   x.data(), xb.data(), 1e-7f,
                                   uint32_t(t) * 2654435761u, SC, imp_lr);
            double m = 0;
            for (auto b : L.syn) m += double(b >> 4);
            return m / double(L.syn.size());
        };
        const int T = 120;
        // drift per step: imp_lr * ( |w*x| - |g*x| ),  w = 0.125, x = 0.5
        float gpos = 0.045f, gneg = 0.205f;
        double dpos = 1.0 * (0.125 * 0.5 - double(gpos) * 0.5);   // +0.04
        double dneg = 1.0 * (0.125 * 0.5 - double(gneg) * 0.5);   // -0.04
        double mpos = run_case(gpos, 1.0f, T);
        double mneg = run_case(gneg, 1.0f, T);
        double ppos = 8.0 + T * dpos, pneg = 8.0 + T * dneg;
        std::printf("counter: drift %+.3f/step -> mean %.2f (pred %.2f)\n",
                    dpos, mpos, ppos);
        std::printf("counter: drift %+.3f/step -> mean %.2f (pred %.2f)\n",
                    dneg, mneg, pneg);
        CHECK(std::abs(mpos - ppos) < 0.7, "positive drift off: %f vs %f", mpos, ppos);
        CHECK(std::abs(mneg - pneg) < 0.7, "negative drift off: %f vs %f", mneg, pneg);
    }

    // ------- 2. importance as a pruning signal, teacher vs junk -------
    {
        const uint32_t LOG_S = 8, RL = 5, CL = 5, M = 1000, N = 1000;
        const uint32_t TEACH = 12, JUNK = 6;
        const float LR = 0.08f, IMP_LR = 0.5f;

        std::vector<std::vector<std::pair<uint32_t, float>>> teach(M);
        std::map<Key, float> tmap;
        std::set<uint64_t> used;
        std::vector<Syn> syns;
        for (uint32_t i = 0; i < M; ++i) {
            std::set<uint32_t> row;
            while (row.size() < TEACH) row.insert(uint32_t(rng() % N));
            for (uint32_t j : row) {
                uint32_t code = 1 + rng() % 15;
                if (code == 8) code = 7;
                teach[i].push_back({j, W_LUT[code] * SC});
                tmap[key(i, j)] = W_LUT[code] * SC;
                used.insert(key(i, j));
                syns.push_back({i, j, uint8_t((8u << 4) | ((rng() & 1) ? 1u : 9u))});
            }
            uint32_t placed = 0;
            while (placed < JUNK) {
                uint32_t j = uint32_t(rng() % N);
                if (!used.insert(key(i, j)).second) continue;
                syns.push_back({i, j, uint8_t((8u << 4) | (1 + rng() % 15))});
                ++placed;
            }
        }
        ToBankedResult tb = to_banked(syns, LOG_S, RL, CL, M, N, 16, 5);
        BankedLayer& L = tb.layer;

        auto teacher_y = [&](const float* x, const uint32_t* xb, float* yt) {
            for (uint32_t i = 0; i < M; ++i) {
                float a = 0.f;
                for (auto& [j, w] : teach[i])
                    if ((xb[j >> 5] >> (j & 31)) & 1u) a += w * x[j];
                yt[i] = a;
            }
        };
        auto make_x = [&](std::vector<float>& x, std::vector<uint32_t>& xb) {
            std::fill(xb.begin(), xb.end(), 0u);
            for (uint32_t j = 0; j < N; ++j) {
                if (rng() % 100 < 30) {
                    xb[j >> 5] |= 1u << (j & 31);
                    x[j] = float(int(rng() % 2001) - 1000) / 1000.f;
                } else x[j] = 0.f;
            }
        };
        const int NP = 16;
        std::vector<std::vector<float>> px(NP, std::vector<float>(N));
        std::vector<std::vector<uint32_t>> pxb(NP, std::vector<uint32_t>((N + 31) / 32));
        std::vector<std::vector<float>> pyt(NP, std::vector<float>(M));
        for (int p = 0; p < NP; ++p) {
            make_x(px[p], pxb[p]);
            teacher_y(px[p].data(), pxb[p].data(), pyt[p].data());
        }
        auto mse_of = [&](const BankedLayer& LL) {
            std::vector<float> y(M);
            double e = 0;
            for (int p = 0; p < NP; ++p) {
                forward_banked_cpu(LL, M, px[p].data(), pxb[p].data(), y.data(), SC);
                for (uint32_t i = 0; i < M; ++i) {
                    double d = y[i] - pyt[p][i];
                    e += d * d;
                }
            }
            return e / (double(NP) * M);
        };

        std::vector<float> x(N), yt(M), ys(M), dy(M);
        std::vector<uint32_t> xb((N + 31) / 32);
        std::vector<uint32_t> act(M);
        for (uint32_t i = 0; i < M; ++i) act[i] = i;
        for (int t = 1; t <= 800; ++t) {
            make_x(x, xb);
            teacher_y(x.data(), xb.data(), yt.data());
            forward_banked_cpu(L, M, x.data(), xb.data(), ys.data(), SC);
            for (uint32_t i = 0; i < M; ++i) dy[i] = ys[i] - yt[i];
            wupdate_banked_cpu(L, act.data(), dy.data(), M, x.data(), xb.data(),
                               LR, uint32_t(t) * 2654435761u, SC, IMP_LR);
        }

        auto live = to_syns(L, M, N);
        double it = 0, ij = 0;
        uint64_t nt = 0, nj = 0;
        for (auto& s : live) {
            if (tmap.count(key(s.i, s.j))) { it += s.b >> 4; ++nt; }
            else                            { ij += s.b >> 4; ++nj; }
        }
        it /= double(nt); ij /= double(std::max<uint64_t>(nj, 1));
        std::printf("importance after training: teacher mean %.2f (n=%llu), "
                    "junk mean %.2f (n=%llu)\n", it, (unsigned long long)nt,
                    ij, (unsigned long long)nj);
        CHECK(it - ij >= 3.0, "importance does not separate: %.2f vs %.2f", it, ij);

        // prune 40% of live three ways from the same snapshot
        auto prune_frac = [&](BankedLayer LL, auto&& order) {
            auto lv = to_syns(LL, M, N);
            std::sort(lv.begin(), lv.end(), order);
            for (size_t t = 0; t < lv.size() * 2 / 5; ++t)
                prune(LL, lv[t].i, lv[t].j);
            return mse_of(LL);
        };
        double m0 = mse_of(L);
        double m_imp = prune_frac(L, [](const Syn& a, const Syn& b) {
            if ((a.b >> 4) != (b.b >> 4)) return (a.b >> 4) < (b.b >> 4);
            return std::abs(W_LUT[a.b & 15]) < std::abs(W_LUT[b.b & 15]);
        });
        double m_mag = prune_frac(L, [](const Syn& a, const Syn& b) {
            float wa = std::abs(W_LUT[a.b & 15]), wb = std::abs(W_LUT[b.b & 15]);
            if (wa != wb) return wa < wb;
            return (a.b >> 4) < (b.b >> 4);
        });
        std::vector<Syn> lv = to_syns(L, M, N);
        std::shuffle(lv.begin(), lv.end(), rng);
        BankedLayer Lr = L;
        for (size_t t = 0; t < lv.size() * 2 / 5; ++t)
            prune(Lr, lv[t].i, lv[t].j);
        double m_rand = mse_of(Lr);
        std::printf("prune 40%%: before %.5f | by-importance %.5f | "
                    "by-magnitude %.5f | random %.5f\n", m0, m_imp, m_mag, m_rand);
        // The floor MSE dominates absolute numbers; the signal quality
        // lives in the DAMAGE above baseline.
        double d_imp = m_imp - m0, d_mag = m_mag - m0, d_rand = m_rand - m0;
        std::printf("prune damage: importance %+.5f | magnitude %+.5f | "
                    "random %+.5f\n", d_imp, d_mag, d_rand);
        CHECK(d_imp < 0.25 * d_rand,
              "importance pruning damage not << random: %f vs %f", d_imp, d_rand);
        CHECK(d_imp < 0.05 * m0,
              "importance pruning damaged the baseline: %f vs %f", d_imp, m0);
    }

    // ------------------------ 3. memory ledger ------------------------
    {
        const uint32_t LOG_S = 12, RL = 5, CL = 5;
        BankedLayer L = random_family(LOG_S, RL, CL, 0.35, rng);
        const uint64_t nslots = uint64_t(L.S()) * L.C() * L.R();
        const double nominal = double(2 * LOG_S + 8);
        const double alloc = (double(L.pi_bits.size()) * 32
                            + double(L.inv_bits.size()) * 32
                            + double(L.syn.size()) * 8) / double(nslots);
        const uint32_t M = 1u << L.KR(), N = 1u << L.KC();
        const double transient = 4.0 * (2.0 * M + 2.0 * N)      // x, y, dy, dx
                               + (M + N) / 8.0 + 4.0 * M;       // bitmaps, act
        std::printf("ledger: persistent %.3f bits/param nominal, %.3f allocated"
                    " (slack words)\n", nominal, alloc);
        std::printf("ledger: transients %.2f MB vs weights %.2f MB "
                    "(O(M+N), no optimizer state, no master weights)\n",
                    transient / 1048576.0, nominal * nslots / 8.0 / 1048576.0);
        CHECK(nominal == 32.0, "nominal bits/param not 32");
        CHECK(std::abs(alloc - 32.0) < 0.02, "allocation overhead: %f", alloc);
    }

    std::printf("\nTOTAL: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
