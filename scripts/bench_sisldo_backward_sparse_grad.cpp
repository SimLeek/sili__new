// Real-kernel A/B for Phase 7's t_col_grad/t_gamma_grad persistent-
// buffer fix in disldo_backward_sparse_grad (sisldo_ops.hpp) -- see
// TODO_BATCH_BLOCKING.md. Small, batch-independent buffer (out_cols/
// rank-scaled, like disldo_backward's Phase 6 group buffers) -- expect
// a small effect, verified rather than assumed.
#include "../sili/lib/headers/delta_csr_memory.hpp"
#include "../sili/lib/headers/sisldo_ops.hpp"
#include "../sili/lib/headers/csr.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using VT = DeltaCSRBiValues<float>;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, VT, COL_TYPE>;

int main(int argc, char** argv) {
    const int num_cpus = argc > 1 ? std::atoi(argv[1]) : 4;
    const std::size_t n_in = 288, n_out = 288;
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> idist(0.1f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    std::vector<float> wv(n_in * n_out), imp(n_in * n_out);
    for (auto& w : wv)
        w = wdist(rng);
    for (auto& v : imp)
        v = idist(rng);

    Weights weights;
    std::vector<SIZE_TYPE> ptrs(n_in + 1, 0);
    std::vector<SIZE_TYPE> idx;
    std::vector<float> scattered_wv, scattered_imp;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, VT, COL_TYPE>(
        ptrs, idx, scattered_wv, scattered_imp, n_in, n_out, std::size_t(64), std::size_t(64));
    block4_load_dense_fp32<SIZE_TYPE, COL_TYPE>(weights, wv.data(), imp.data(), n_in, n_out);
    weights.out_degree.assign(n_out, SIZE_TYPE(n_in));
    weights.set_scale_rank_max(1);
    weights.set_scale_rank(1);
    weights.output_scale_is_trainable = true;

    std::printf("# disldo_backward_sparse_grad real-kernel bench: n_in=%zu n_out=%zu num_cpus=%d\n",
                n_in, n_out, num_cpus);

    for (int batch : {1, 8, 64, 256, 1024}) {
        std::vector<float> input(std::size_t(batch) * n_in);
        for (auto& v : input)
            v = xdist(rng);
        std::vector<SIZE_TYPE> dy_ptrs(std::size_t(batch) + 1, 0);
        std::vector<SIZE_TYPE> dy_idx;
        std::vector<float> dy_vals;
        for (int b = 0; b < batch; ++b) {
            for (SIZE_TYPE c = 0; c < SIZE_TYPE(n_out); ++c) {
                dy_idx.push_back(c);
                dy_vals.push_back(xdist(rng));
            }
            dy_ptrs[std::size_t(b) + 1] = dy_ptrs[std::size_t(b)] + SIZE_TYPE(n_out);
        }
        auto dy_csr = make_csr_input<SIZE_TYPE, float>(SIZE_TYPE(batch), SIZE_TYPE(n_out), dy_ptrs,
                                                       dy_idx, dy_vals);

        std::vector<float> dx(std::size_t(batch) * n_in), nia(n_in), nga(n_out);
        const int n_warmup = 3;
        for (int i = 0; i < n_warmup; ++i) {
            std::fill(dx.begin(), dx.end(), 0.f);
            std::fill(nia.begin(), nia.end(), 0.f);
            std::fill(nga.begin(), nga.end(), 0.f);
            disldo_backward_sparse_grad<SIZE_TYPE, VT, COL_TYPE, RMSpropScalePolicy<float>, true>(
                input.data(), SIZE_TYPE(batch), weights, dy_csr, dx.data(), nia.data(), nga.data(),
                0.001f, num_cpus, true);
        }
        const int n_reps = std::max(5, 2000 / std::max(batch, 1));
        std::vector<double> call_ns(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            std::fill(dx.begin(), dx.end(), 0.f);
            std::fill(nia.begin(), nia.end(), 0.f);
            std::fill(nga.begin(), nga.end(), 0.f);
            auto t0 = std::chrono::steady_clock::now();
            disldo_backward_sparse_grad<SIZE_TYPE, VT, COL_TYPE, RMSpropScalePolicy<float>, true>(
                input.data(), SIZE_TYPE(batch), weights, dy_csr, dx.data(), nia.data(), nga.data(),
                0.001f, num_cpus, true);
            auto t1 = std::chrono::steady_clock::now();
            call_ns[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        std::sort(call_ns.begin(), call_ns.end());
        const double median_ns = call_ns[std::size_t(n_reps) / 2];
        const double mean_ns = std::accumulate(call_ns.begin(), call_ns.end(), 0.0) / n_reps;
        std::printf("batch=%5d  reps=%5d  mean=%12.1f ns/call  median=%12.1f ns/call\n", batch,
                    n_reps, mean_ns, median_ns);
    }
    return 0;
}
