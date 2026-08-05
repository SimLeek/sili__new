// test_synapto_packed.cpp -- the same quality-under-churn experiment as
// test_synapto.cpp phase B, running on the PACKED codec's mechanism:
// in-place pruning, batched re-encode growth, rebuild-based capacity
// change. Same teacher, same policy code, same schedule shape. The
// codec-specific predictions differ and are asserted: blocking is raw
// row/column capacity (no banks), the pre-expansion wall is HARD (rows
// saturate at exactly R live), and expansion unblocks ~all previously
// row-blocked positions rather than banked's one half.
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

int main() {
    const float SC = 0.0625f;
    const float LR = 0.08f;
    std::mt19937_64 rng(90210);

    const uint32_t M = 1000, N = 1000, TEACH = 12, C = 16, C_LOG = 4;
    uint32_t R = 8, R_LOG = 3;                    // start capacity-starved

    // on-grid sparse teacher (identical construction to the banked test)
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
    std::map<Key, float> tmap;
    for (uint32_t i = 0; i < M; ++i)
        for (auto& [j, w] : teach[i]) tmap[key(i, j)] = w;

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

    PackedGpu P = build_packed({}, M, N, R, C, R_LOG);
    auto mse = [&]() {
        std::vector<float> y(M);
        double e = 0;
        for (int p = 0; p < NP; ++p) {
            forward_packed_cpu(P, px[p].data(), pxb[p].data(), y.data(), SC);
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
    uint32_t step = 1;

    auto train = [&](int steps) {
        for (int t = 0; t < steps; ++t, ++step) {
            make_x(x, xb);
            teacher_y(x.data(), xb.data(), yt.data());
            forward_packed_cpu(P, x.data(), xb.data(), ys.data(), SC);
            for (uint32_t i = 0; i < M; ++i) dy[i] = ys[i] - yt[i];
            wupdate_packed_cpu(P, act.data(), dy.data(), M,
                               x.data(), xb.data(), LR, step * 2654435761u, SC);
        }
    };
    auto zombie_prune = [&]() {
        int z = 0;
        for (auto& sy : packed_to_syns(P))
            if ((sy.b & 7u) == 0u) { prune_packed(P, sy.i, sy.j); ++z; }
        return z;
    };
    std::vector<float> score;
    // Same accumulated-correlation growth policy as the banked test;
    // only the blocking rule and the commit differ.
    auto grow_cycle = [&](int budget, int& blocked_row, int& blocked_col) {
        score.assign(size_t(M) * N, 0.f);
        const int TS = 32;
        for (int t = 0; t < TS; ++t) {
            make_x(x, xb);
            teacher_y(x.data(), xb.data(), yt.data());
            forward_packed_cpu(P, x.data(), xb.data(), ys.data(), SC);
            for (uint32_t i = 0; i < M; ++i) {
                float e = ys[i] - yt[i];
                if (e == 0.f) continue;
                float* row = score.data() + size_t(i) * N;
                for (uint32_t j = 0; j < N; ++j) row[j] += e * x[j];
            }
        }
        // capacity state
        std::vector<uint32_t> rl(M, 0), cl(N, 0);
        auto live = packed_to_syns(P);
        for (auto& sy : live) { ++rl[sy.i]; ++cl[sy.j]; }
        std::set<Key> present;
        for (auto& sy : live) present.insert(key(sy.i, sy.j));

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
        std::vector<Syn> adds;
        for (size_t t = 0; t < topn && int(adds.size()) < budget; ++t) {
            uint32_t i = uint32_t(cand[t].second >> 32);
            uint32_t j = uint32_t(cand[t].second);
            if (present.count(cand[t].second)) continue;
            if (rl[i] == R) { ++blocked_row; continue; }
            if (cl[j] == C) { ++blocked_col; continue; }
            float sc = score[size_t(i) * N + j];
            uint8_t code = (sc > 0.f) ? 9 : 1;
            adds.push_back({i, j, uint8_t(0x80 | code)});
            ++rl[i]; ++cl[j];
            present.insert(cand[t].second);
        }
        if (!adds.empty()) P = regrow_packed(P, adds, R, C, R_LOG);
        return int(adds.size());
    };

    const int PRE = 16, POST = 11, HOLD = 4, CONTRACT = 12;
    const int TOT = PRE + 1 + POST + HOLD + CONTRACT;
    std::printf("cycle  live  R   mse       grown zomb rowblk colblk\n");
    double mse0 = 0, mse_plateau = 0, mse_grown = 0, mse_hold = 0, best = 1e9;
    double mse_c3 = 0;
    int blocked_before = 0;
    double unblocked_frac = 0.0;
    for (int c = 0; c < TOT; ++c) {
        int br = 0, bc = 0, gr = 0, zb = 0;
        if (c < PRE) {
            train(30); zb = zombie_prune(); gr = grow_cycle(400, br, bc);
        } else if (c == PRE) {
            // Frozen-list unblocking across expansion. Packed blocks only
            // on raw capacity, so raising R should free essentially ALL
            // row-blocked teacher positions (banked freed one half).
            std::vector<uint32_t> rl(M, 0), cl(N, 0);
            auto live = packed_to_syns(P);
            for (auto& sy : live) { ++rl[sy.i]; ++cl[sy.j]; }
            std::set<Key> present;
            for (auto& sy : live) present.insert(key(sy.i, sy.j));
            std::vector<Key> blocked;
            for (auto& [tk, tw] : tmap) {
                uint32_t i = uint32_t(tk >> 32), j = uint32_t(tk);
                if (present.count(tk)) continue;
                if (rl[i] == R) blocked.push_back(tk);
            }
            blocked_before = int(blocked.size());
            uint64_t lv = packed_live_count(P);
            double m_before = mse();
            P = regrow_packed(P, {}, 16, C, 4);       // R 8 -> 16, lossless
            R = 16; R_LOG = 4;
            CHECK(packed_live_count(P) == lv, "expansion changed live");
            CHECK(std::abs(mse() - m_before) < 1e-9 + 0.02 * m_before,
                  "expansion changed function");
            int freed = 0;
            for (Key tk : blocked) {
                uint32_t j = uint32_t(tk);
                if (cl[j] < C) ++freed;               // only col-full remains
            }
            unblocked_frac = blocked_before ? double(freed) / blocked_before : 1.0;
            std::printf("-- expand: %d teacher slots row-blocked, %.0f%% freed"
                        " (no banks: capacity was the only wall) --\n",
                        blocked_before, 100.0 * unblocked_frac);
            train(30); zb = zombie_prune(); gr = grow_cycle(400, br, bc);
        } else if (c < PRE + 1 + POST) {
            train(30); zb = zombie_prune(); gr = grow_cycle(400, br, bc);
        } else if (c < PRE + 1 + POST + HOLD) {
            train(30);
        } else {
            auto live = packed_to_syns(P);
            std::sort(live.begin(), live.end(), [&](const Syn& a2, const Syn& b2) {
                float wa = std::abs(W_LUT[a2.b & 15]), wb = std::abs(W_LUT[b2.b & 15]);
                if (wa != wb) return wa < wb;
                return (a2.b >> 4) < (b2.b >> 4);
            });
            size_t np = live.size() / 12;
            for (size_t t = 0; t < np; ++t) prune_packed(P, live[t].i, live[t].j);
            train(30);
        }
        double m = mse();
        best = std::min(best, m);
        if (c == 0) mse0 = m;
        if (c == PRE - 1) mse_plateau = m;
        if (c == PRE + POST) mse_grown = m;
        if (c == PRE + POST + HOLD) mse_hold = m;
        if (c == PRE + 1 + POST + HOLD + 2) mse_c3 = m;
        std::printf("%5d %5llu %3u  %.6f %5d %4d %6d %6d\n", c,
                    (unsigned long long)packed_live_count(P), R, m, gr, zb, br, bc);
    }
    double mse_end = mse();

    CHECK(mse_plateau < 0.75 * mse0,
          "insufficient learning while starved: %f vs %f", mse_plateau, mse0);
    // Packed's wall is HARD capacity: only saturated rows block, so far
    // fewer teacher slots are blocked at the boundary than under banked's
    // collision wall (which bites at every occupancy). The pigeonhole
    // bound applies per SATURATED row only.
    CHECK(blocked_before > 500,
          "starvation not observed at the capacity wall: %d", blocked_before);
    CHECK(unblocked_frac > 0.9,
          "packed expansion should free ~all row blocks: %.2f", unblocked_frac);
    CHECK(mse_grown < 0.9 * mse_plateau,
          "post-expansion growth did not improve quality: %f vs %f",
          mse_grown, mse_plateau);
    CHECK(mse_hold < 1.3 * mse_grown, "hold unstable: %f vs %f", mse_hold, mse_grown);
    CHECK(mse_c3 < 1.3 * mse_hold,
          "early contraction not graceful: %f vs %f", mse_c3, mse_hold);
    CHECK(mse_end < 0.5 * mse0,
          "deep contraction lost too much: %f vs start %f", mse_end, mse0);
    std::printf("mse: start=%.5f plateau=%.5f grown=%.5f hold=%.5f end=%.5f best=%.5f\n",
                mse0, mse_plateau, mse_grown, mse_hold, mse_end, best);

    std::printf("\nTOTAL: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
