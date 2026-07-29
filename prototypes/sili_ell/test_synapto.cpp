// test_synapto.cpp -- structural grow/prune mechanism tests plus a
// quality-under-churn experiment: a capacity-starved student learns a
// sparse teacher, plateaus against the bank wall, expands capacity
// (demotion-free), improves further, then contracts and degrades
// gracefully. Policy in this harness is deliberately dumb (magnitude
// pruning, outer-product growth); the real importance system replaces it
// through the same probe/commit seam.
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

static void audit(const BankedLayer& L, const char* tag) {
    const uint32_t R = L.R(), C = L.C(), S = L.S();
    for (uint32_t ip = 0; ip < (1u << L.KR()); ++ip)
        for (uint32_t s = 0; s < R; ++s) {
            uint32_t v = L.fget(L.pi_bits, uint64_t(ip) * R + s);
            uint32_t u2 = L.fget(L.inv_bits,
                                 uint64_t((s << L.LOG_S) | v) * C + (ip >> L.LOG_S));
            CHECK(u2 == (ip & (S - 1)), "%s audit ip=%u s=%u", tag, ip, s);
        }
}

using Key = uint64_t;
static Key key(uint32_t i, uint32_t j) { return (Key(i) << 32) | j; }

int main() {
    const float SC = 0.0625f;
    std::mt19937_64 rng(4242);

    // ================= Phase A: mechanism exactness =================
    {
        const uint32_t LOG_S = 6, RL = 3, CL = 3, M = 300, N = 350;
        BankedLayer L = assemble_banked({}, LOG_S, RL, CL, 2654435769u, 2654435769u);
        audit(L, "empty");
        CHECK(live_count(L) == 0, "empty layer not empty");

        std::map<Key, uint8_t> acc;      // ground-truth live set
        int nfree = 0, nrow = 0, ncol = 0, nboth = 0, ndup = 0;
        std::vector<std::pair<uint32_t, uint32_t>> rowblocked;
        for (int t = 0; t < 800; ++t) {
            uint32_t i = uint32_t(rng() % M), j = uint32_t(rng() % N);
            uint8_t b = uint8_t((1 + rng() % 15) << 4 | (1 + rng() % 15));
            GrowProbe pr = probe_grow(L, i, j);
            switch (pr.st) {
            case GrowStatus::Free:
                CHECK(commit_grow(L, i, j, b), "commit after Free failed");
                acc[key(i, j)] = b;
                ++nfree;
                break;
            case GrowStatus::AlreadyPresent:
                CHECK(acc.count(key(i, j)) == 1, "AlreadyPresent but not in set");
                ++ndup;
                break;
            case GrowStatus::RowBankBusy:
                CHECK(acc.count(key(i, pr.row_evict_j)) == 1,
                      "reported row blocker (%u,%u) not live", i, pr.row_evict_j);
                rowblocked.push_back({i, j});
                ++nrow;
                break;
            case GrowStatus::ColBankBusy:
                CHECK(acc.count(key(pr.col_evict_i, j)) == 1,
                      "reported col blocker (%u,%u) not live", pr.col_evict_i, j);
                ++ncol;
                break;
            case GrowStatus::BothBusy:
                ++nboth;
                break;
            }
        }
        std::printf("phase A grow: free=%d dup=%d rowblk=%d colblk=%d both=%d live=%llu\n",
                    nfree, ndup, nrow, ncol, nboth,
                    (unsigned long long)live_count(L));
        audit(L, "post-grow");
        CHECK(live_count(L) == acc.size(), "live count mismatch");

        {   // stored set == ground truth, byte-exact
            auto s = to_syns(L, M, N);
            CHECK(s.size() == acc.size(), "to_syns size");
            for (auto& sy : s) {
                auto it = acc.find(key(sy.i, sy.j));
                CHECK(it != acc.end() && it->second == sy.b, "stored synapse wrong");
            }
        }

        // eviction path: prune the blocker, retry, must succeed
        int retried = 0;
        for (auto& [i, j] : rowblocked) {
            if (retried == 8) break;
            GrowProbe pr = probe_grow(L, i, j);
            if (pr.st != GrowStatus::RowBankBusy) continue;   // world changed
            CHECK(prune(L, i, pr.row_evict_j), "prune blocker failed");
            acc.erase(key(i, pr.row_evict_j));
            CHECK(commit_grow(L, i, j, 0x92), "grow after eviction failed");
            acc[key(i, j)] = 0x92;
            ++retried;
        }
        std::printf("phase A evict-retry: %d succeeded\n", retried);
        audit(L, "post-evict");

        // prune every other live synapse
        {
            int idx = 0;
            std::vector<Key> victims;
            for (auto& [k, b] : acc) if ((idx++ & 1) == 0) victims.push_back(k);
            for (Key k : victims) {
                CHECK(prune(L, uint32_t(k >> 32), uint32_t(k)), "prune failed");
                acc.erase(k);
            }
            CHECK(live_count(L) == acc.size(), "post-prune count");
        }

        // forward against the ground-truth set
        std::vector<float> x(N), y(M, -1.f), yr(M, 0.f);
        std::vector<uint32_t> xb((N + 31) / 32, 0xffffffffu);
        for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 500.f;
        for (auto& [k, b] : acc)
            yr[uint32_t(k >> 32)] += W_LUT[b & 15] * SC * x[uint32_t(k)];
        forward_banked_cpu(L, M, x.data(), xb.data(), y.data(), SC);
        for (uint32_t i = 0; i < M; ++i)
            CHECK(std::abs(y[i] - yr[i]) <= 1e-4f * (1.f + std::abs(yr[i])),
                  "phase A fwd i=%u", i);

        // demotion-free expansion: same set, same outputs
        BankedLayer L2 = expand_banked(L, 1, 0, M, N);
        audit(L2, "expanded");
        CHECK(live_count(L2) == acc.size(), "expansion changed live count");
        std::vector<float> y2(M, -1.f);
        forward_banked_cpu(L2, M, x.data(), xb.data(), y2.data(), SC);
        for (uint32_t i = 0; i < M; ++i)
            CHECK(std::abs(y2[i] - yr[i]) <= 1e-4f * (1.f + std::abs(yr[i])),
                  "expanded fwd i=%u", i);
        std::printf("phase A: %s\n\n", g_fail ? "FAIL" : "PASS");
    }

    // ============ Phase B: quality under structural churn ============
    {
        const uint32_t LOG_S = 8, CL = 4;
        uint32_t RL = 3;                          // start capacity-starved
        const uint32_t M = 1000, N = 1000, TEACH = 12;
        const float LR = 0.08f;

        // on-grid sparse teacher
        std::vector<std::vector<std::pair<uint32_t, float>>> teach(M);
        {
            std::set<uint32_t> row;
            for (uint32_t i = 0; i < M; ++i) {
                row.clear();
                while (row.size() < TEACH) row.insert(uint32_t(rng() % N));
                for (uint32_t j : row) {
                    uint32_t code = 1 + rng() % 15;
                    if (code == 8) code = 7;
                    teach[i].push_back({j, W_LUT[code] * SC});
                }
            }
        }
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

        // fixed probes for MSE
        const int NP = 16;
        std::vector<std::vector<float>> px(NP, std::vector<float>(N));
        std::vector<std::vector<uint32_t>> pxb(NP, std::vector<uint32_t>((N + 31) / 32));
        std::vector<std::vector<float>> pyt(NP, std::vector<float>(M));
        for (int p = 0; p < NP; ++p) {
            make_x(px[p], pxb[p]);
            teacher_y(px[p].data(), pxb[p].data(), pyt[p].data());
        }

        BankedLayer L = assemble_banked({}, LOG_S, RL, CL, 2654435769u, 2654435769u);
        auto mse = [&]() {
            std::vector<float> y(M);
            double e = 0;
            for (int p = 0; p < NP; ++p) {
                forward_banked_cpu(L, M, px[p].data(), pxb[p].data(), y.data(), SC);
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
        std::vector<uint32_t> dball((M + 31) / 32, 0xffffffffu);
        for (uint32_t i = 0; i < M; ++i) act[i] = i;
        uint32_t step = 1;

        auto train = [&](int steps) {
            for (int t = 0; t < steps; ++t, ++step) {
                make_x(x, xb);
                teacher_y(x.data(), xb.data(), yt.data());
                forward_banked_cpu(L, M, x.data(), xb.data(), ys.data(), SC);
                for (uint32_t i = 0; i < M; ++i) dy[i] = ys[i] - yt[i];
                wupdate_banked_cpu(L, act.data(), dy.data(), M,
                                   x.data(), xb.data(), LR, step * 2654435761u, SC);
            }
        };
        // Growth scores accumulate e*x over a window: E[e_i x_j] at an
        // absent slot is proportional to the missing weight, so the top
        // of this matrix IS the gradient's wish list. (A single sample
        // is rank-1 noise and grows junk; ask me how I know.)
        std::vector<float> score;
        auto grow_cycle = [&](int budget, int& blocked_row, int& blocked_col) {
            score.assign(size_t(M) * N, 0.f);
            const int TS = 32;
            for (int t = 0; t < TS; ++t) {
                make_x(x, xb);
                teacher_y(x.data(), xb.data(), yt.data());
                forward_banked_cpu(L, M, x.data(), xb.data(), ys.data(), SC);
                for (uint32_t i = 0; i < M; ++i) {
                    float e = ys[i] - yt[i];
                    if (e == 0.f) continue;
                    float* row = score.data() + size_t(i) * N;
                    for (uint32_t j = 0; j < N; ++j)
                        row[j] += e * x[j];        // x is 0 when inactive
                }
            }
            std::vector<std::pair<float, Key>> cand;
            cand.reserve(size_t(M) * N / 8);
            for (uint32_t i = 0; i < M; ++i)
                for (uint32_t j = 0; j < N; ++j) {
                    float sc = score[size_t(i) * N + j];
                    if (sc != 0.f) cand.push_back({-std::abs(sc), key(i, j)});
                }
            size_t topn = std::min(cand.size(), size_t(budget) * 6);
            std::nth_element(cand.begin(), cand.begin() + topn, cand.end());
            std::sort(cand.begin(), cand.begin() + topn);
            int grown = 0;
            for (size_t t = 0; t < topn && grown < budget; ++t) {
                uint32_t i = uint32_t(cand[t].second >> 32);
                uint32_t j = uint32_t(cand[t].second);
                GrowProbe pr = probe_grow(L, i, j);
                if (pr.st == GrowStatus::AlreadyPresent) continue;
                if (pr.st != GrowStatus::Free) {
                    blocked_row += (pr.st != GrowStatus::ColBankBusy);
                    blocked_col += (pr.st != GrowStatus::RowBankBusy);
                    continue;
                }
                float sc = score[size_t(i) * N + j];
                uint8_t code = (sc > 0.f) ? 9 : 1;   // small, gradient sign
                commit_grow(L, i, j, uint8_t(0x80 | code));
                ++grown;
            }
            return grown;
        };

        std::map<Key, float> tmap;
        double tmass = 0;
        for (uint32_t i = 0; i < M; ++i)
            for (auto& [j, w] : teach[i]) { tmap[key(i, j)] = w; tmass += w * w; }
        auto coverage = [&]() {
            double cvr = 0;
            for (auto& sy : to_syns(L, M, N)) {
                auto it = tmap.find(key(sy.i, sy.j));
                if (it != tmap.end()) cvr += it->second * it->second;
            }
            return cvr / tmass;
        };
        auto zombie_prune = [&]() {          // weight learned to zero: cull;
            int z = 0;                       // wiring stays, regrow is free
            for (auto& sy : to_syns(L, M, N))
                if ((sy.b & 7u) == 0u) { prune(L, sy.i, sy.j); ++z; }
            return z;
        };

        const int PRE = 16, POST = 11, HOLD = 4, CONTRACT = 12;
        const int TOT = PRE + 1 + POST + HOLD + CONTRACT;
        std::printf("cycle  live  RL  mse       grown zomb rowblk colblk\n");
        double mse0 = 0, mse_plateau = 0, mse_grown = 0, mse_hold = 0, best = 1e9;
        int blocked_before = 0;
        double unblocked_frac = 0.0;
        for (int c = 0; c < TOT; ++c) {
            int br = 0, bc = 0, gr = 0, zb = 0;
            if (c < PRE) {
                train(30); zb = zombie_prune(); gr = grow_cycle(400, br, bc);
            } else if (c == PRE) {
                // Frozen-list unblocking measurement across the expansion:
                // teacher positions row-blocked before must be ~half freed
                // after, because raising R_LOG SPLITS banks.
                std::vector<Key> blocked;
                for (auto& [tk, tw] : tmap) {
                    GrowProbe pr = probe_grow(L, uint32_t(tk >> 32), uint32_t(tk));
                    if (pr.st == GrowStatus::RowBankBusy || pr.st == GrowStatus::BothBusy)
                        blocked.push_back(tk);
                }
                blocked_before = int(blocked.size());
                uint64_t lv = live_count(L);
                double m_before = mse();
                L = expand_banked(L, 1, 0, M, N);
                RL = 4;
                CHECK(live_count(L) == lv, "expansion changed live");
                CHECK(std::abs(mse() - m_before) < 1e-9 + 0.02 * m_before,
                      "expansion changed function");
                audit(L, "phaseB-expand");
                int freed = 0;
                for (Key tk : blocked) {
                    GrowProbe pr = probe_grow(L, uint32_t(tk >> 32), uint32_t(tk));
                    if (pr.st == GrowStatus::Free || pr.st == GrowStatus::ColBankBusy)
                        ++freed;
                }
                unblocked_frac = blocked_before ? double(freed) / blocked_before : 1.0;
                std::printf("-- expand: %d teacher slots row-blocked, %.0f%% freed by the bank split --\n",
                            blocked_before, 100.0 * unblocked_frac);
                train(30); zb = zombie_prune(); gr = grow_cycle(400, br, bc);
            } else if (c < PRE + 1 + POST) {
                train(30); zb = zombie_prune(); gr = grow_cycle(400, br, bc);
            } else if (c < PRE + 1 + POST + HOLD) {
                train(30);
            } else {
                auto live = to_syns(L, M, N);
                std::sort(live.begin(), live.end(), [&](const Syn& a2, const Syn& b2) {
                    float wa = std::abs(W_LUT[a2.b & 15]), wb = std::abs(W_LUT[b2.b & 15]);
                    if (wa != wb) return wa < wb;
                    return (a2.b >> 4) < (b2.b >> 4);
                });
                size_t np = live.size() / 12;
                for (size_t t = 0; t < np; ++t) prune(L, live[t].i, live[t].j);
                train(30);
            }
            double m = mse();
            best = std::min(best, m);
            if (c == 0) mse0 = m;
            if (c == PRE - 1) mse_plateau = m;
            if (c == PRE + POST) mse_grown = m;
            if (c == PRE + POST + HOLD) mse_hold = m;
            std::printf("%5d %5llu %3u  %.6f %5d %4d %6d %6d\n", c,
                        (unsigned long long)live_count(L), RL, m, gr, zb, br, bc);
        }
        double mse_end = mse();
        audit(L, "phaseB-end");
        std::printf("teacher-mass coverage at end: %.1f%%\n", 100.0 * coverage());

        CHECK(mse_plateau < 0.75 * mse0,
              "insufficient learning while starved: %f vs %f", mse_plateau, mse0);
        CHECK(blocked_before > 3000,
              "starvation not observed (pigeonhole says >= 4/row): %d", blocked_before);
        CHECK(unblocked_frac > 0.25,
              "bank split freed too few blocked slots: %.2f", unblocked_frac);
        CHECK(mse_grown < 0.9 * mse_plateau,
              "post-expansion growth did not improve quality: %f vs %f",
              mse_grown, mse_plateau);
        CHECK(mse_hold < 1.3 * mse_grown, "hold unstable: %f vs %f", mse_hold, mse_grown);
        CHECK(mse_end < 1.6 * mse_hold,
              "contraction not graceful: %f vs %f", mse_end, mse_hold);
        std::printf("mse: start=%.5f plateau=%.5f grown=%.5f hold=%.5f end=%.5f best=%.5f\n",
                    mse0, mse_plateau, mse_grown, mse_hold, mse_end, best);
    }

    std::printf("\nTOTAL: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
