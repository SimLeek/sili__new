// bench_packed_vs_disldo.cpp
// ===========================================================================
// Same methodology as bench_vs_disldo.cpp (banked), but against Fable's
// PACKED codec instead: forward_packed_cpu / dx_packed_cpu / wupdate_packed_cpu
// (dual_codec_ell_cpu.hpp), built via build_packed (sparse_format_controller.hpp).
// Packed has NO hash-bank collision constraint (synapto.hpp: "any (i,j) fits
// while row i and column j have live headroom, which is the entire blocking
// rule") -- so unlike banked, there is no demotion to account for as long as
// R/C capacity actually covers the realized row/column degrees. Same
// M/N/density/RNG seeds as the banked benchmark for a direct comparison.
//
// Orientation note (see bench_vs_disldo.cpp for the full explanation): disldo's
// CSR is indexed by INPUT row ("CSR of W^T"), Fable's Syn{i,j} convention is
// indexed by OUTPUT row ("CSR of W"). Built separately here, from one shared
// canonical (i,j,weight) list, so both systems compute the same logical W.
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

template <class F>
double bench_ms(F&& fn, int reps = 20) {
    fn();  // warm
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_s();
        fn();
        best = std::min(best, now_s() - t0);
    }
    return best * 1e3;
}

int main() {
    const uint32_t M = 4096, N = 4096;
    const uint32_t target_nnz_per_row = 200;    // same as bench_vs_disldo.cpp, same seeds
    const int NUM_CPUS = 8;

    std::mt19937 rng(42);
    std::vector<std::vector<uint32_t>> row_cols(M);   // output i -> sorted input js
    std::vector<uint32_t> col_degree(N, 0);
    {
        std::set<uint32_t> seen;
        std::uniform_int_distribution<uint32_t> colpick(0, N - 1);
        for (uint32_t i = 0; i < M; ++i) {
            seen.clear();
            while (seen.size() < target_nnz_per_row) seen.insert(colpick(rng));
            for (uint32_t j : seen) { row_cols[i].push_back(j); col_degree[j]++; }
            std::sort(row_cols[i].begin(), row_cols[i].end());
        }
    }
    const size_t total_syn = size_t(M) * target_nnz_per_row;
    const uint32_t max_col_degree = *std::max_element(col_degree.begin(), col_degree.end());
    std::printf("problem: M=%u N=%u target_nnz/row=%u total_syn=%zu density=%.2f%% realized_max_col_degree=%u\n",
                M, N, target_nnz_per_row, total_syn,
                100.0 * target_nnz_per_row / N, max_col_degree);

    // ---- One canonical (i, j, weight, importance) list, shared by both systems ----
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

    // ---- Fable: Syn{i,j} matches all[] directly (row = output) ----
    std::vector<Syn> syns;
    syns.reserve(total_syn);
    for (auto& s : all)
        syns.push_back({s.i, s.j, uint8_t((s.imp_nibble << 4) | fable_code_for(s.w))});

    // ---- disldo: CSR rows = N (inputs), grouped by j, storing output i as "column" ----
    std::vector<std::vector<std::pair<uint32_t, SynW*>>> by_input(N);
    for (auto& s : all) by_input[s.j].push_back({s.i, &s});
    for (auto& lst : by_input) std::sort(lst.begin(), lst.end(),
        [](auto& a, auto& b) { return a.first < b.first; });

    std::vector<int> csr_ptrs(N + 1, 0);
    std::vector<int> csr_idx;                 // output indices i
    std::vector<float> csr_w, csr_imp;
    csr_idx.reserve(total_syn); csr_w.reserve(total_syn); csr_imp.reserve(total_syn);
    for (uint32_t j = 0; j < N; ++j) {
        for (auto& [i, sp] : by_input[j]) {
            csr_idx.push_back(int(i));
            csr_w.push_back(sp->w);
            csr_imp.push_back(float(sp->imp_nibble) / 15.0f);
        }
        csr_ptrs[j + 1] = int(csr_idx.size());
    }
    SparseLinearWeightsDelta<int, FP4BiPacked, uint32_t> weights;
    weights.connections = delta_csr_from_absolute<int, FP4BiPacked, uint32_t>(
        csr_ptrs, csr_idx, csr_w, csr_imp, N, M,
        total_syn * 8 + 4096, total_syn + 64, 0.2f);
    weights.recompute_stats();
    weights.out_degree.assign(M, 0);
    for (int i : csr_idx) weights.out_degree[size_t(i)]++;
    double sili_nnz = double(weights.connections.nnz());
    std::printf("sili: real nnz stored = %.0f (delta-CSR, rows=inputs=%u, cols=outputs=%u)\n", sili_nnz, N, M);

    // ---- Build Fable's PackedGpu from the SAME (i,j) pattern ----
    // R, C must be multiples of GROUP (32) -- pack_view groups in chunks of
    // 32 and reads a full group unconditionally; a non-multiple leaves the
    // last group reading past a row's real slot count (found by a hard
    // crash on the first cut of this file: `wd <= 25` tripped on garbage
    // past-the-end data, not an actual encoding-width problem).
    uint32_t R = ((target_nnz_per_row + GROUP - 1) / GROUP) * GROUP;
    uint32_t C = 1u; while (C < max_col_degree) C <<= 1;
    uint32_t R_LOG = 0; while ((1u << R_LOG) < R) ++R_LOG;
    std::printf("packed: R=%u (exact row fit) C=%u (next-pow2 >= realized max col degree %u, R_LOG=%u)\n",
                R, C, max_col_degree, R_LOG);
    double t0 = now_s();
    PackedGpu P = build_packed(syns, M, N, R, C, R_LOG);
    double packed_build_s = now_s() - t0;
    std::printf("packed: build_packed() %.3fs, live=%zu (== total_syn, no demotion by design)\n",
                packed_build_s, packed_live_count(P));

    // ---- Correctness cross-check ----
    std::vector<float> x(N), y_disldo(M, 0.f), y_packed(M, 0.f), y_ref(M, 0.f);
    std::uniform_real_distribution<float> xval(-1.0f, 1.0f);
    for (auto& v : x) v = xval(rng);
    std::vector<uint32_t> xbits((N + 31) / 32, 0xffffffffu);

    disldo_forward<int, FP4BiPacked, uint32_t>(x.data(), 1, N, weights, y_disldo.data(), 0.0f, NUM_CPUS);
    forward_packed_cpu(P, x.data(), xbits.data(), y_packed.data(), 1.0f);
    for (auto& s : all) y_ref[s.i] += s.w * x[s.j];
    double max_err_disldo = 0, max_err_packed = 0;
    for (uint32_t i = 0; i < M; ++i) {
        max_err_disldo = std::max(max_err_disldo, double(std::abs(y_disldo[i] - y_ref[i])));
        max_err_packed = std::max(max_err_packed, double(std::abs(y_packed[i] - y_ref[i])));
    }
    std::printf("correctness: max|disldo-ref|=%.4f (should be ~0), max|packed-ref|=%.4f (should be ~0, no demotion)\n",
                max_err_disldo, max_err_packed);

    // ---- Speed: forward ----
    double disldo_fwd_ms = bench_ms([&] {
        std::fill(y_disldo.begin(), y_disldo.end(), 0.f);
        disldo_forward<int, FP4BiPacked, uint32_t>(x.data(), 1, N, weights, y_disldo.data(), 0.0f, NUM_CPUS);
    });
    double packed_fwd_ms = bench_ms([&] {
        forward_packed_cpu(P, x.data(), xbits.data(), y_packed.data(), 1.0f);
    });
    std::printf("\n--- FORWARD (batch=1) ---\n");
    std::printf("disldo_forward   %8.4f ms   %8.1f Msyn/s (real nnz=%.0f)\n",
                disldo_fwd_ms, sili_nnz / (disldo_fwd_ms * 1e-3) * 1e-6, sili_nnz);
    std::printf("forward_packed   %8.4f ms   %8.1f Msyn/s (exact live nnz, no padding waste)\n",
                packed_fwd_ms, sili_nnz / (packed_fwd_ms * 1e-3) * 1e-6);
    std::printf("speedup (disldo_ms / packed_ms): %.2fx\n", disldo_fwd_ms / packed_fwd_ms);

    // ---- Speed: backward ----
    std::vector<float> dy(M), dx_disldo(N, 0.f), dx_packed(N, 0.f);
    std::vector<float> neuron_in_accum(N, 0.f), neuron_grad_accum(M, 0.f);
    for (auto& v : dy) v = xval(rng);
    std::vector<uint32_t> dybits((M + 31) / 32, 0xffffffffu);

    double disldo_bwd_ms = bench_ms([&] {
        std::fill(dx_disldo.begin(), dx_disldo.end(), 0.f);
        disldo_backward<int, FP4BiPacked, uint32_t>(
            x.data(), 1, N, dy.data(), weights, dx_disldo.data(),
            neuron_in_accum.data(), neuron_grad_accum.data(), 0.001f, NUM_CPUS, false);
    });
    double packed_bwd_ms = bench_ms([&] {
        dx_packed_cpu(P, dy.data(), dybits.data(), dx_packed.data(), 1.0f);
    });
    std::printf("\n--- BACKWARD dx (batch=1) ---\n");
    std::printf("disldo_backward  %8.4f ms   %8.1f Msyn/s\n",
                disldo_bwd_ms, sili_nnz / (disldo_bwd_ms * 1e-3) * 1e-6);
    std::printf("dx_packed        %8.4f ms   %8.1f Msyn/s (iterates real N=%u x C=%u, not an inflated key space)\n",
                packed_bwd_ms, (double(N) * C) / (packed_bwd_ms * 1e-3) * 1e-6, N, C);
    std::printf("speedup (disldo_ms / packed_ms): %.2fx\n", disldo_bwd_ms / packed_bwd_ms);

    // ---- Weight update ----
    std::vector<uint32_t> act_rows(M);
    for (uint32_t i = 0; i < M; ++i) act_rows[i] = i;
    double packed_wupd_ms = bench_ms([&] {
        wupdate_packed_cpu(P, act_rows.data(), dy.data(), M, x.data(), xbits.data(),
                           0.001f, 1u, 1.0f, 0.0f);
    });
    std::printf("\n--- WEIGHT UPDATE ---\n");
    std::printf("wupdate_packed   %8.4f ms   %8.1f Msyn/s\n",
                packed_wupd_ms, sili_nnz / (packed_wupd_ms * 1e-3) * 1e-6);
    std::printf("combined speedup (disldo_bwd_ms / (packed_dx_ms+wupd_ms)): %.2fx\n",
                disldo_bwd_ms / (packed_bwd_ms + packed_wupd_ms));

    // ---- Memory ----
    std::size_t sili_idx_bytes = weights.connections.indices_buf.size();
    std::size_t sili_val_bytes = size_t(sili_nnz);
    double sili_bits_per_syn = 8.0 * double(sili_idx_bytes + sili_val_bytes) / sili_nnz;
    double packed_bits_per_syn = packed_bits_per_param(P.rowv, P.colv, R_LOG);
    std::printf("\n--- MEMORY (real bytes actually stored, THIS density/pattern) ---\n");
    std::printf("sili   (DeltaCSR, ULEB128 idx + FP4BiPacked val): %.2f bits/synapse (%zu idx-bytes + %zu val-bytes over %.0f live)\n",
                sili_bits_per_syn, sili_idx_bytes, sili_val_bytes, sili_nnz);
    std::printf("packed (group-varint deltas both views + slot backpointer + weight/imp byte): %.2f bits/synapse (no demotion, exact live count == capacity here)\n",
                packed_bits_per_syn);
    std::printf("delta (packed - sili): %.2f bits/synapse %s with packed at this density/capacity choice\n",
                packed_bits_per_syn - sili_bits_per_syn,
                packed_bits_per_syn > sili_bits_per_syn ? "more" : "less");

    return 0;
}
