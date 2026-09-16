// Real-kernel A/B for Phase 6's group_dx/group_col_grad persistent-
// buffer fix (see TODO_BATCH_BLOCKING.md). Small, batch-independent
// buffer -- expect a small effect, verified rather than assumed. Needs
// num_cpus high enough to span >1 CCX group (see
// TODO_BATCH_BLOCKING.md's Phase 6 entry -- confirmed num_groups=2 at
// num_cpus=8/16 on arch-sandbox), and OMP_PROC_BIND=true
// OMP_PLACES=cores set for the group-aware path to trigger correctly.
#include "../sili/lib/headers/delta_csr_memory.hpp"
#include "../sili/lib/headers/linear_disldo.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

int main(int argc, char** argv) {
    const int num_cpus = argc > 1 ? std::atoi(argv[1]) : 8;
    const std::size_t n_in = 288, n_out = 288;
    std::mt19937 rng(21);
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
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        ptrs, idx, scattered_wv, scattered_imp, n_in, n_out, std::size_t(64), std::size_t(64));
    block4_load_dense_fp32<SIZE_TYPE, COL_TYPE>(weights, wv.data(), imp.data(), n_in, n_out);
    weights.out_degree.assign(n_out, SIZE_TYPE(n_in));
    weights.set_scale_rank_max(1);
    weights.set_scale_rank(1);
    weights.output_scale_is_trainable = true;

    std::printf("# disldo_backward group-reduction real-kernel bench: n_in=%zu n_out=%zu "
                "num_cpus=%d\n",
                n_in, n_out, num_cpus);

    for (int batch : {1, 8, 64, 256, 1024}) {
        std::vector<float> input(std::size_t(batch) * n_in),
            output_grad(std::size_t(batch) * n_out);
        for (auto& v : input)
            v = xdist(rng);
        for (auto& v : output_grad)
            v = xdist(rng);
        std::vector<float> input_grad(std::size_t(batch) * n_in);
        std::vector<float> neuron_input_accum(n_in), neuron_grad_accum(n_out);

        const int n_warmup = 3;
        for (int i = 0; i < n_warmup; ++i) {
            std::fill(input_grad.begin(), input_grad.end(), 0.f);
            std::fill(neuron_input_accum.begin(), neuron_input_accum.end(), 0.f);
            std::fill(neuron_grad_accum.begin(), neuron_grad_accum.end(), 0.f);
            disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                input.data(), SIZE_TYPE(batch), SIZE_TYPE(n_in), output_grad.data(), weights,
                input_grad.data(), neuron_input_accum.data(), neuron_grad_accum.data(), 0.001f,
                num_cpus);
        }
        const int n_reps = std::max(5, 2000 / std::max(batch, 1));
        std::vector<double> call_ns(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            std::fill(input_grad.begin(), input_grad.end(), 0.f);
            std::fill(neuron_input_accum.begin(), neuron_input_accum.end(), 0.f);
            std::fill(neuron_grad_accum.begin(), neuron_grad_accum.end(), 0.f);
            auto t0 = std::chrono::steady_clock::now();
            disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
                input.data(), SIZE_TYPE(batch), SIZE_TYPE(n_in), output_grad.data(), weights,
                input_grad.data(), neuron_input_accum.data(), neuron_grad_accum.data(), 0.001f,
                num_cpus);
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
