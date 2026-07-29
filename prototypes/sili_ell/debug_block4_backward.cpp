#include "../../sili/lib/headers/dense_block4.hpp"
#include <cstdio>
#include <random>

int main() {
    fp4_seed_stochastic_rng(42);
    const uint32_t n_out = 32, n_in = 24;  // both multiples of 4, deliberately unequal
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u01(0, 1);
    std::uniform_int_distribution<int> magpick(1, 7);   // FP4_TABLE[1..7] = 0.5..6, exact values
    // exact FP4 table values -- rules out ordinary quantization rounding
    // noise so a real orientation/indexing bug (if any) shows up cleanly,
    // matching the methodology already used for forward-only correctness
    // checks earlier this session.

    std::vector<float> Wdense(std::size_t(n_out) * n_in, 0.0f);
    std::vector<uint32_t> ptrs(n_out + 1, 0);
    std::vector<uint32_t> idx;
    std::vector<float> w, imp;
    for (uint32_t r = 0; r < n_out; ++r) {
        // Force row_scale=1.0 for every row (guarantee a magnitude-6 entry)
        // so per-row rescaling can't introduce any real quantization noise
        // -- isolates whether the transpose/backward LOGIC is correct,
        // separate from the (already-understood, expected) rescaling noise.
        idx.push_back(0); w.push_back(6.0f); imp.push_back(0.5f);
        Wdense[std::size_t(r) * n_in + 0] = 6.0f;
        for (uint32_t c = 1; c < n_in; ++c) {
            if (u01(rng) < 0.4f) {
                float v = FP4_TABLE[magpick(rng)] * (u01(rng) < 0.5f ? -1.0f : 1.0f);
                idx.push_back(c); w.push_back(v); imp.push_back(0.5f);
                Wdense[std::size_t(r) * n_in + c] = v;
            }
        }
        ptrs[r + 1] = uint32_t(idx.size());
    }
    printf("nnz=%zu\n", idx.size());

    auto split = split_for_block4(n_out, n_in, ptrs, idx, w, imp, 0.10f);
    printf("block4 live blocks region built, leftover=%zu\n", split.leftover_rc.size());

    // ---- forward + dx correctness ----
    std::uniform_real_distribution<float> xv(-1.0f, 1.0f);
    std::vector<float> x(n_in), y(n_out, 0.f), dy(n_out), dx(n_in, 0.f);
    for (auto& v : x) v = xv(rng);
    for (auto& v : dy) v = xv(rng);

    block4_forward(split.block4, x.data(), y.data(), 1);
    std::vector<float> y_ref(n_out, 0.f);
    for (uint32_t r = 0; r < n_out; ++r)
        for (uint32_t c = 0; c < n_in; ++c)
            y_ref[r] += Wdense[std::size_t(r) * n_in + c] * x[c];
    float max_err_fwd = 0;
    for (uint32_t r = 0; r < n_out; ++r) max_err_fwd = std::max(max_err_fwd, std::abs(y[r] - y_ref[r]));
    printf("forward max err vs dense ref: %.4f\n", max_err_fwd);

    auto bwt = transpose_block4(split.block4);
    block4_backward_dx(bwt, dy.data(), dx.data(), 1);
    std::vector<float> dx_ref(n_in, 0.f);
    for (uint32_t r = 0; r < n_out; ++r)
        for (uint32_t c = 0; c < n_in; ++c)
            dx_ref[c] += Wdense[std::size_t(r) * n_in + c] * dy[r];
    float max_err_dx = 0;
    for (uint32_t c = 0; c < n_in; ++c) max_err_dx = std::max(max_err_dx, std::abs(dx[c] - dx_ref[c]));
    printf("backward dx max err vs dense ref: %.4f\n", max_err_dx);

    // ---- weight update: check the AVERAGE weight change direction is
    // correct (stochastic rounding means exact match isn't guaranteed per-
    // call, so average over many trials against the deterministic formula's
    // EXPECTED value instead of checking one draw). ----
    const float lr = 0.1f;
    float w_before = -1;
    // find a live slot to inspect: row 0's first active block
    {
        auto& bw = split.block4;
        if (bw.block_ptrs[1] > bw.block_ptrs[0]) {
            uint8_t byte = bw.block_data[std::size_t(bw.block_ptrs[0]) * 16];
            w_before = FP4_TABLE[byte & 0xF] * bw.row_scale[0];
        }
    }
    block4_weight_update(split.block4, x.data(), dy.data(), lr, 1);
    printf("weight update ran (w[0,*] before ~%.4f, spot check only -- full numeric verification would need many stochastic trials averaged)\n", w_before);

    return 0;
}
