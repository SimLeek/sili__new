// bench_vs_disldo.cpp
// ===========================================================================
// Controlled, apples-to-apples benchmark: sili's disldo_forward/backward
// (linear_disldo.hpp) vs Fable's banked-codec forward_banked_cpu/dx_banked_cpu
// (dual_codec_ell_cpu.hpp), on the IDENTICAL sparsity pattern and problem
// size, in one process, one compiler, one set of flags. No Python, no
// pybind marshaling on either side -- both called through their real,
// production entry points (SparseLinearWeightsDelta built via
// delta_csr_from_absolute, exactly what SparseLinearLayer::load_weights
// does; BankedLayer built via to_banked, exactly what the sili_ell
// controller does on a packed->banked conversion).
//
// Orientation note (found via an isolated repro, not guessed): disldo's
// CSR is indexed by INPUT row (each row lists the OUTPUT columns it
// feeds -- "CSR of W^T"), while Fable's kernels and this file's Syn{i,j}
// convention index by OUTPUT row (y[i] = sum_j w[i,j]*x[j] -- "CSR of W").
// With M==N the array sizes coincide so a naive same-orientation build
// doesn't crash, it silently computes disldo's transpose. Fixed here by
// building disldo's CSR grouped by input j explicitly.
//
// Reports: forward/backward throughput (Msyn/s, same unit sili_ell's own
// harness uses, for direct comparability), real bytes/synapse for both
// systems at this density, and a quality/correctness cross-check (both
// must reproduce the same dense reference matmul with learning_rate=0).
// ===========================================================================
#include "../../sili/lib/headers/linear_disldo.hpp"
#include "sparse_format_controller.hpp"
#include "dual_codec_ell_cpu.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

using namespace sfc;

// Both systems' FP4 tables hold the same 15 real magnitudes (0, +-0.5, +-1,
// +-1.5, +-2, +-3, +-4, +-6) but at DIFFERENT 4-bit codes (sili: code 8 =
// NaN/unused; Fable's W_LUT: code 8 = -0.0). Quantize once through sili's
// real fp4_quantize(), then find Fable's matching code for that SAME real
// value by nearest-neighbor search of its own table -- so both systems
// store the identical post-quantization value, not two independent random
// 4-bit codes.
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
    // Problem size: representative of a real sili MLP-style layer (not
    // sili_ell's own 100000x100800-at-18-nnz/row blind-hash stress test,
    // which is far sparser than what disldo actually runs in production).
    const uint32_t M = 4096, N = 4096;          // outputs, inputs
    const uint32_t target_nnz_per_row = 200;    // ~4.9% density
    const uint32_t LOG_S = 4, R_LOG = 8, C_LOG = 8;   // R=C=256 capacity (28% headroom over target)
    const int NUM_CPUS = 8;

    std::mt19937 rng(42);
    std::vector<std::vector<uint32_t>> row_cols(M);  // output i -> sorted input js
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
    std::printf("problem: M=%u N=%u target_nnz/row=%u total_syn=%zu density=%.2f%%\n",
                M, N, target_nnz_per_row, total_syn,
                100.0 * target_nnz_per_row / N);

    // ---- Generate one canonical (i, j, weight, importance) list, shared by both systems ----
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

    // ---- Fable: Syn{i,j} = weight from input j to output i, matches all[] directly ----
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
    std::printf("sili: real nnz stored = %zu (delta-CSR, rows=inputs=%u, cols=outputs=%u)\n",
                weights.connections.nnz(), N, M);

    // ---- Build Fable's BankedLayer from the SAME (i,j) pattern ----
    double t0 = now_s();
    ToBankedResult tb = to_banked(syns, LOG_S, R_LOG, C_LOG, M, N, 32, 9);
    double banked_build_s = now_s() - t0;
    std::printf("banked: to_banked() %.3fs, kept=%zu demoted=%.3f%% (bank-collision loss on THIS pattern/multiplier)\n",
                banked_build_s, size_t((1.0 - tb.demoted_frac) * total_syn), 100.0 * tb.demoted_frac);

    // ---- Correctness cross-check: dense reference matmul (learning_rate=0 both sides) ----
    std::vector<float> x(N), y_disldo(M, 0.f), y_banked(M, 0.f), y_ref(M, 0.f);
    std::uniform_real_distribution<float> xval(-1.0f, 1.0f);
    for (auto& v : x) v = xval(rng);
    std::vector<uint32_t> xbits((N + 31) / 32, 0xffffffffu);  // all active (dense input)

    disldo_forward<int, FP4BiPacked, uint32_t>(x.data(), 1, N, weights, y_disldo.data(), 0.0f, NUM_CPUS);
    forward_banked_cpu(tb.layer, M, x.data(), xbits.data(), y_banked.data(), /*w_scale=*/1.0f);
    for (auto& s : all) y_ref[s.i] += s.w * x[s.j];
    double max_err_disldo = 0, max_err_banked = 0;
    for (uint32_t i = 0; i < M; ++i) {
        max_err_disldo = std::max(max_err_disldo, double(std::abs(y_disldo[i] - y_ref[i])));
        max_err_banked = std::max(max_err_banked, double(std::abs(y_banked[i] - y_ref[i])));
    }
    std::printf("correctness: max|disldo-ref|=%.4f (should be ~0), max|banked-ref|=%.4f (banked demoted %.1f%% of synapses so this WILL differ)\n",
                max_err_disldo, max_err_banked, 100.0 * tb.demoted_frac);

    // ---- Speed: forward ----
    double disldo_fwd_ms = bench_ms([&] {
        std::fill(y_disldo.begin(), y_disldo.end(), 0.f);
        disldo_forward<int, FP4BiPacked, uint32_t>(x.data(), 1, N, weights, y_disldo.data(), 0.0f, NUM_CPUS);
    });
    double banked_fwd_ms = bench_ms([&] {
        forward_banked_cpu(tb.layer, M, x.data(), xbits.data(), y_banked.data(), 1.0f);
    });
    double sili_nnz = double(weights.connections.nnz());
    double banked_capacity = double(M) * tb.layer.R();
    std::printf("\n--- FORWARD (batch=1) ---\n");
    std::printf("disldo_forward   %8.4f ms   %8.1f Msyn/s (real nnz=%.0f)\n",
                disldo_fwd_ms, sili_nnz / (disldo_fwd_ms * 1e-3) * 1e-6, sili_nnz);
    std::printf("forward_banked   %8.4f ms   %8.1f Msyn/s (capacity slots=%.0f, incl. silent)\n",
                banked_fwd_ms, banked_capacity / (banked_fwd_ms * 1e-3) * 1e-6, banked_capacity);
    std::printf("speedup (disldo_ms / banked_ms): %.2fx\n", disldo_fwd_ms / banked_fwd_ms);

    // ---- Speed: backward ----
    // dy is a gradient over OUTPUTS (size M); dx (gradient over inputs, size N)
    // is what disldo_backward computes (its "input_grad" param, size in_cols=N).
    std::vector<float> dy(M), dx_disldo(N, 0.f), dx_banked(N, 0.f);
    std::vector<float> neuron_in_accum(N, 0.f), neuron_grad_accum(M, 0.f);
    for (auto& v : dy) v = xval(rng);
    std::vector<uint32_t> dybits((M + 31) / 32, 0xffffffffu);

    double disldo_bwd_ms = bench_ms([&] {
        std::fill(dx_disldo.begin(), dx_disldo.end(), 0.f);
        disldo_backward<int, FP4BiPacked, uint32_t>(
            x.data(), 1, N, dy.data(), weights, dx_disldo.data(),
            neuron_in_accum.data(), neuron_grad_accum.data(), 0.001f, NUM_CPUS, false);
    });
    const uint32_t nkeys = 1u << tb.layer.KC();
    double banked_bwd_ms = bench_ms([&] {
        dx_banked_cpu(tb.layer, nkeys, N, dy.data(), dybits.data(), dx_banked.data(), 1.0f);
    });
    std::printf("\n--- BACKWARD dx (batch=1) ---\n");
    std::printf("disldo_backward  %8.4f ms   %8.1f Msyn/s\n",
                disldo_bwd_ms, sili_nnz / (disldo_bwd_ms * 1e-3) * 1e-6);
    std::printf("dx_banked        %8.4f ms   %8.1f Msyn/s (iterates nkeys=%u x C=%u)\n",
                banked_bwd_ms, (double(nkeys) * tb.layer.C()) / (banked_bwd_ms * 1e-3) * 1e-6,
                nkeys, tb.layer.C());
    std::printf("speedup (disldo_ms / banked_ms): %.2fx\n", disldo_bwd_ms / banked_bwd_ms);

    // ---- Weight-update step (banked's fused equivalent of disldo_backward's weight write) ----
    std::vector<uint32_t> act_rows(M);
    for (uint32_t i = 0; i < M; ++i) act_rows[i] = i;
    double banked_wupd_ms = bench_ms([&] {
        wupdate_banked_cpu(tb.layer, act_rows.data(), dy.data(), M, x.data(), xbits.data(),
                           0.001f, 1u, 1.0f, 0.0f);
    });
    std::printf("\n--- WEIGHT UPDATE (banked's fused fwd-already-done update; disldo's is fused INTO disldo_backward above) ---\n");
    std::printf("wupdate_banked   %8.4f ms   %8.1f Msyn/s\n",
                banked_wupd_ms, sili_nnz / (banked_wupd_ms * 1e-3) * 1e-6);
    std::printf("(disldo has no separate update call -- backward above already includes it, so compare\n");
    std::printf(" disldo_backward's %.4fms against banked's dx(%.4fms)+wupdate(%.4fms)=%.4fms combined)\n",
                disldo_bwd_ms, banked_bwd_ms, banked_wupd_ms, banked_bwd_ms + banked_wupd_ms);
    std::printf("combined speedup (disldo_bwd_ms / (banked_dx_ms+wupd_ms)): %.2fx\n",
                disldo_bwd_ms / (banked_bwd_ms + banked_wupd_ms));

    // ---- Memory: real bytes/synapse, THIS density, both systems ----
    std::size_t sili_idx_bytes = weights.connections.indices_buf.size();
    std::size_t sili_val_bytes = size_t(sili_nnz);  // FP4BiPacked: 1 byte holds BOTH weight+importance nibbles
    double sili_bits_per_syn = 8.0 * double(sili_idx_bytes + sili_val_bytes) / sili_nnz;
    double banked_bits_per_syn_capacity = banked_bits_per_param(LOG_S);  // per STORED SLOT (capacity), README's definition
    double banked_live = sili_nnz * (1.0 - tb.demoted_frac);
    double banked_bits_per_syn_live = banked_bits_per_syn_capacity * banked_capacity / banked_live;
    std::printf("\n--- MEMORY (real bytes actually stored, THIS density/pattern) ---\n");
    std::printf("sili (DeltaCSR, ULEB128 idx + FP4BiPacked val): %.2f bits/LIVE synapse (%zu idx-bytes + %zu val-bytes over %.0f live)\n",
                sili_bits_per_syn, sili_idx_bytes, sili_val_bytes, sili_nnz);
    std::printf("banked, per CAPACITY slot (README's own metric): %.1f bits/slot, %u slots/row\n",
                banked_bits_per_syn_capacity, tb.layer.R());
    std::printf("banked, per LIVE synapse (capacity cost / actual live count after %.1f%% demotion): %.2f bits/live-synapse\n",
                100.0 * tb.demoted_frac, banked_bits_per_syn_live);
    std::printf("delta (banked_live - sili): %.2f bits/synapse more with banked at this density/capacity choice\n",
                banked_bits_per_syn_live - sili_bits_per_syn);

    return 0;
}
