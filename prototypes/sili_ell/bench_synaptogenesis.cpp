// bench_synaptogenesis.cpp
// ===========================================================================
// Structural growth/pruning speed: sili's delta_csr_row_insert_col /
// delta_csr_row_remove_col (the real per-synapse primitives synap_row_step
// is built on -- see delta_csr_memory.hpp) vs Fable's banked probe_grow+
// commit_grow / prune (synapto.hpp) vs Fable's packed prune_packed /
// regrow_packed (batched, no O(1) single-insert primitive exists for packed
// -- growth is "batched re-encode... a rebuild", per synapto.hpp's own
// comment, so its cost is reported honestly as an amortized-per-synapse
// number from a real batch, not pretended to be O(1)).
//
// Same M/N/density/construction as bench_vs_disldo.cpp, same shared
// (i,j,weight) generation so all three systems start from an identical
// initial graph before any structural change is applied.
// ===========================================================================
#include "../../sili/lib/headers/linear_disldo.hpp"
#include "sparse_format_controller.hpp"
#include "dual_codec_ell_cpu.hpp"
#include "synapto.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

using namespace sfc;

static uint8_t fable_code_for(float sili_quantized_value) {
    uint8_t best = 0;
    float best_err = std::abs(sili_quantized_value - W_LUT[0]);
    for (uint8_t i = 1; i < 16; ++i) {
        float err = std::abs(sili_quantized_value - W_LUT[i]);
        if (err < best_err) { best_err = err; best = i; }
    }
    return best;
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    const uint32_t M = 4096, N = 4096;
    const uint32_t target_nnz_per_row = 200;
    const uint32_t LOG_S = 4, R_LOG = 8, C_LOG = 8;
    const uint32_t N_TRIALS = 2000;   // single-synapse ops to average over

    std::mt19937 rng(42);
    std::vector<std::vector<uint32_t>> row_cols(M);
    {
        std::set<uint32_t> seen;
        std::uniform_int_distribution<uint32_t> colpick(0, N - 1);
        for (uint32_t i = 0; i < M; ++i) {
            seen.clear();
            while (seen.size() < target_nnz_per_row) seen.insert(colpick(rng));
            for (uint32_t j : seen) row_cols[i].push_back(j);
            std::sort(row_cols[i].begin(), row_cols[i].end());
        }
    }
    const size_t total_syn = size_t(M) * target_nnz_per_row;

    struct SynW { uint32_t i, j; float w; uint32_t imp_nibble; };
    std::vector<SynW> all;
    all.reserve(total_syn);
    std::mt19937 wrng(7);
    std::uniform_real_distribution<float> wval(-6.0f, 6.0f), ival(0.0f, 1.0f);
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t j : row_cols[i]) {
            const uint8_t sili_code = fp4_quantize(wval(wrng));
            const uint32_t imp_nibble = 1u + uint32_t(ival(wrng) * 14.0f);
            all.push_back({i, j, FP4_TABLE[sili_code], imp_nibble});
        }

    std::vector<Syn> syns;
    syns.reserve(total_syn);
    for (auto& s : all) syns.push_back({s.i, s.j, uint8_t((s.imp_nibble << 4) | fable_code_for(s.w))});

    // ---- sili: disldo CSR, rows=inputs (see bench_vs_disldo.cpp's orientation note) ----
    std::vector<std::vector<std::pair<uint32_t, SynW*>>> by_input(N);
    for (auto& s : all) by_input[s.j].push_back({s.i, &s});
    for (auto& lst : by_input) std::sort(lst.begin(), lst.end(),
        [](auto& a, auto& b) { return a.first < b.first; });
    std::vector<int> csr_ptrs(N + 1, 0);
    std::vector<int> csr_idx;
    std::vector<float> csr_w, csr_imp;
    csr_idx.reserve(total_syn); csr_w.reserve(total_syn); csr_imp.reserve(total_syn);
    for (uint32_t j = 0; j < N; ++j) {
        for (auto& [i, sp] : by_input[j]) {
            csr_idx.push_back(int(i)); csr_w.push_back(sp->w); csr_imp.push_back(float(sp->imp_nibble) / 15.0f);
        }
        csr_ptrs[j + 1] = int(csr_idx.size());
    }
    SparseLinearWeightsDelta<int, FP4BiPacked, uint32_t> weights;
    // Real, generous headroom (0.3 = 30% -- disldo's normal synaptogenesis
    // convention reserves ~20%, per delta_csr_from_absolute's own default;
    // used 0.3 here for enough single-insert room across N_TRIALS random rows).
    weights.connections = delta_csr_from_absolute<int, FP4BiPacked, uint32_t>(
        csr_ptrs, csr_idx, csr_w, csr_imp, N, M,
        std::size_t(total_syn * 1.3) * 8 + 4096, std::size_t(total_syn * 1.3) + 64, 0.3f);
    weights.recompute_stats();
    std::printf("sili: %zu live synapses, rows=inputs=%u cols=outputs=%u, 30%% index+elem headroom reserved\n",
                weights.connections.nnz(), N, M);

    // ---- Fable banked ----
    double tb0 = now_s();
    ToBankedResult tb = to_banked(syns, LOG_S, R_LOG, C_LOG, M, N, 32, 9);
    std::printf("banked: to_banked() %.3fs, live=%zu (demoted %.1f%%, bank capacity is the only headroom concept it has)\n",
                now_s() - tb0, size_t((1.0 - tb.demoted_frac) * total_syn), 100.0 * tb.demoted_frac);

    // ---- Fable packed ----
    uint32_t R = ((target_nnz_per_row + GROUP - 1) / GROUP) * GROUP;
    std::vector<uint32_t> col_degree(N, 0);
    for (auto& s : all) col_degree[s.j]++;
    uint32_t maxcd = *std::max_element(col_degree.begin(), col_degree.end());
    uint32_t C = 1u; while (C < maxcd) C <<= 1;
    uint32_t pR_LOG = 0; while ((1u << pR_LOG) < R) ++pR_LOG;
    PackedGpu P = build_packed(syns, M, N, R, C, pR_LOG);
    std::printf("packed: live=%zu, R=%u C=%u (0%% headroom -- packed has no bank/hash constraint so this snapshot needs none for correctness, but growth still needs a rebuild, see below)\n",
                packed_live_count(P), R, C);

    // ---- Candidate (i,j) pairs for insertion: not already present ----
    std::set<std::pair<uint32_t,uint32_t>> existing;
    for (auto& s : all) existing.insert({s.i, s.j});
    std::vector<std::pair<uint32_t,uint32_t>> new_pairs;
    std::mt19937 prng(123);
    std::uniform_int_distribution<uint32_t> ripick(0, M - 1), rjpick(0, N - 1);
    while (new_pairs.size() < N_TRIALS) {
        uint32_t i = ripick(prng), j = rjpick(prng);
        if (existing.count({i, j}) == 0) { new_pairs.push_back({i, j}); existing.insert({i, j}); }
    }
    // ---- Candidate (i,j) pairs for removal: currently live ----
    std::vector<std::pair<uint32_t,uint32_t>> remove_pairs;
    {
        std::vector<uint32_t> perm(all.size());
        for (size_t k = 0; k < perm.size(); ++k) perm[k] = uint32_t(k);
        std::shuffle(perm.begin(), perm.end(), prng);
        for (uint32_t t = 0; t < N_TRIALS; ++t) remove_pairs.push_back({all[perm[t]].i, all[perm[t]].j});
    }

    // =================== INSERT ===================
    // sili: delta_csr_row_insert_col -- ROW is the input j (matches CSR
    // orientation), inserting output i as the stored "column".
    {
        double t0 = now_s();
        int ok = 0;
        for (auto& [i, j] : new_pairs)
            ok += delta_csr_row_insert_col<int, FP4BiPacked, uint32_t>(
                weights.connections, j, i, 1.0f, 0.5f) ? 1 : 0;
        double dt = now_s() - t0;
        std::printf("\nsili insert (delta_csr_row_insert_col): %d/%u ok, %.2f us/op total, %.4f us/op avg\n",
                    ok, N_TRIALS, dt * 1e6, dt * 1e6 / N_TRIALS);
    }
    {
        double t0 = now_s();
        int ok = 0;
        for (auto& [i, j] : new_pairs) {
            GrowProbe pr = probe_grow(tb.layer, i, j);
            if (pr.st == GrowStatus::Free)
                ok += commit_grow(tb.layer, i, j, uint8_t(0x11)) ? 1 : 0;
        }
        double dt = now_s() - t0;
        std::printf("banked insert (probe_grow+commit_grow): %d/%u ok (rest blocked by bank collision -- a REAL, not incidental, limitation), %.4f us/op avg over attempted\n",
                    ok, N_TRIALS, dt * 1e6 / N_TRIALS);
    }
    {
        // packed has no O(1) single-insert primitive -- growth is a batched
        // rebuild (synapto.hpp's regrow_packed). Measure the batch honestly,
        // report amortized cost, not pretend it's incremental.
        std::vector<Syn> adds;
        for (auto& [i, j] : new_pairs) adds.push_back({i, j, 0x11});
        double t0 = now_s();
        PackedGpu P2 = regrow_packed(P, adds, R, C, pR_LOG);
        double dt = now_s() - t0;
        std::printf("packed insert (regrow_packed, BATCHED rebuild of all %zu live + %u new): %.2f ms total, %.4f us/synapse amortized (NOT O(1) per-insert -- whole-layer rebuild cost)\n",
                    packed_live_count(P), N_TRIALS, dt * 1e3, dt * 1e6 / N_TRIALS);
    }

    // =================== REMOVE ===================
    {
        double t0 = now_s();
        int ok = 0;
        for (auto& [i, j] : remove_pairs)
            ok += delta_csr_row_remove_col<int, FP4BiPacked, uint32_t>(weights.connections, j, i) ? 1 : 0;
        double dt = now_s() - t0;
        std::printf("\nsili remove (delta_csr_row_remove_col): %d/%u ok, %.4f us/op avg\n",
                    ok, N_TRIALS, dt * 1e6 / N_TRIALS);
    }
    {
        double t0 = now_s();
        int ok = 0;
        for (auto& [i, j] : remove_pairs) ok += prune(tb.layer, i, j) ? 1 : 0;
        double dt = now_s() - t0;
        std::printf("banked remove (prune): %d/%u ok, %.4f us/op avg\n", ok, N_TRIALS, dt * 1e6 / N_TRIALS);
    }
    {
        double t0 = now_s();
        int ok = 0;
        for (auto& [i, j] : remove_pairs) ok += prune_packed(P, i, j) ? 1 : 0;
        double dt = now_s() - t0;
        std::printf("packed remove (prune_packed, LINEAR scan of row's R=%u slots per call): %d/%u ok, %.4f us/op avg\n",
                    R, ok, N_TRIALS, dt * 1e6 / N_TRIALS);
    }

    return 0;
}
