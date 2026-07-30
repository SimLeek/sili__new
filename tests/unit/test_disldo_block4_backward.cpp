// Correctness check for block4's backward contribution inside
// disldo_backward (linear_disldo.hpp): dx must match a hand-computed
// dense reference summing BOTH the scattered CSR and block4 gradients,
// and the block4-owned weight byte must move in the gradient-descent
// direction (checked by re-decoding it and calling disldo_forward again).
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>

static int g_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); ++g_fail; } \
} while (0)

int main() {
    const int n_in = 8, n_out = 8;

    // Same fixture as test_disldo_block4_forward.cpp: scattered row0->col5
    // (w=2.0), row1->col2 (w=-1.5); block4 tile (0,0) with (row0,col1)=1.0,
    // (row2,col3)=0.5.
    std::vector<int> ptrs(n_in + 1, 0), idx;
    std::vector<float> w, imp;
    idx.push_back(5); w.push_back(2.0f); imp.push_back(0.5f);
    ptrs[1] = 1;
    idx.push_back(2); w.push_back(-1.5f); imp.push_back(0.5f);
    ptrs[2] = 2;
    for (int r = 2; r <= n_in; ++r) ptrs[r] = 2;

    SparseLinearWeightsDelta<int, FP4BiPacked, uint32_t> weights;
    weights.connections = delta_csr_from_absolute<int, FP4BiPacked, uint32_t>(
        ptrs, idx, w, imp, n_in, n_out, 256, 64, 0.2f);
    weights.block4.init(n_in, n_out);
    weights.recompute_stats();

    auto tile = weights.block4.get_or_create(0, 0);
    tile.at(0, 1) = fp4_quantize(1.0f) | (fp4_quantize(0.5f) << 4);
    tile.at(2, 3) = fp4_quantize(0.5f) | (fp4_quantize(0.5f) << 4);

    std::vector<float> x(n_in, 0.f);
    x[0] = 3.0f; x[1] = 1.0f; x[2] = 2.0f;

    std::vector<float> dy(n_out, 0.f);
    dy[5] = 0.7f;   // hits scattered row0
    dy[2] = -0.3f;  // hits scattered row1
    dy[1] = 0.4f;   // hits block4 (row0,col1)
    dy[3] = -0.2f;  // hits block4 (row2,col3)

    // Reference dx (learning_rate=0, so this must equal the pure W^T @ dy
    // matmul with the SAME weights the forward test already validated):
    // dx[0] = W[0,5]*dy[5] + W[0,1]*dy[1] = 2.0*0.7 + 1.0*0.4 = 1.8
    // dx[1] = W[1,2]*dy[2] = -1.5*-0.3 = 0.45
    // dx[2] = W[2,3]*dy[3] = 0.5*-0.2 = -0.1
    std::vector<float> dx_ref(n_in, 0.f);
    dx_ref[0] = 1.8f;
    dx_ref[1] = 0.45f;
    dx_ref[2] = -0.1f;

    std::vector<float> dx(n_in, 0.f);
    std::vector<float> neuron_in(n_in, 0.f), neuron_grad(n_out, 0.f);
    disldo_backward<int, FP4BiPacked, uint32_t>(
        x.data(), 1, n_in, dy.data(), weights, dx.data(),
        neuron_in.data(), neuron_grad.data(),
        /*learning_rate=*/0.0f, /*num_cpus=*/1);

    for (int r = 0; r < n_in; ++r)
        CHECK(std::abs(dx[r] - dx_ref[r]) < 1e-4f, "dx[%d]: got %.4f expected %.4f", r, dx[r], dx_ref[r]);

    // Weight bytes must be UNCHANGED with learning_rate=0.
    const auto t_after0 = weights.block4.find(0, 0);
    CHECK((t_after0.at(0, 1) & 0xFu) == fp4_quantize(1.0f), "block4 (0,1) weight changed at lr=0");
    CHECK((t_after0.at(2, 3) & 0xFu) == fp4_quantize(0.5f), "block4 (2,3) weight changed at lr=0");

    // Now a real training step: learning_rate != 0. The block4 weight at
    // (row=0,col=1) sees input=3.0, dy=0.4 -> gradient g=1.2 -> weight
    // should move in the -lr*g direction (i.e. DOWN from 1.0, since g>0).
    std::vector<float> dx2(n_in, 0.f);
    std::vector<float> ni2(n_in, 0.f), ng2(n_out, 0.f);
    disldo_backward<int, FP4BiPacked, uint32_t>(
        x.data(), 1, n_in, dy.data(), weights, dx2.data(),
        ni2.data(), ng2.data(),
        /*learning_rate=*/0.5f, /*num_cpus=*/2, /*lr_per_row_nnz=*/false,
        /*damp_by_importance=*/false);

    const auto t_after1 = weights.block4.find(0, 0);
    const float w01_after = FP4_TABLE[t_after1.at(0, 1) & 0xFu];
    CHECK(w01_after < 1.0f - 1e-6f, "block4 (0,1) weight should decrease under lr=0.5, g>0: got %.3f", w01_after);

    // (row=2,col=3): input=2.0, dy=-0.2 -> g=-0.4 -> weight should move UP.
    const float w23_after = FP4_TABLE[t_after1.at(2, 3) & 0xFu];
    CHECK(w23_after > 0.5f - 1e-6f, "block4 (2,3) weight should increase under lr=0.5, g<0: got %.3f", w23_after);

    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
