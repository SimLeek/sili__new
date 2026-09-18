// DIDLDO kernel bench, real linear_didldo.hpp implementation (MKL) -- see
// TODO_BATCH_BLOCKING.md's "Also queued -> DIDLDO/SIDLDO" for the full
// writeup/measurements this once justified. Previously a standalone
// proof-of-concept with its own hand-rolled didldo_forward/didldo_backward
// (scipy's bundled OpenBLAS, vanilla RMSprop) predating linear_didldo.hpp's
// real implementation -- rewritten to call the actual production kernels
// (same header bench_sidldo_forward.cpp/bench_sidldo_backward.cpp already
// use) so this perf-gates real code, not a stale duplicate that drifted
// from it (see docs/research/sparse_rnn.rst:
// sparse_rnn.didldo_same_optimizer_as_disldo for why that duplicate's
// optimizer formula was wrong).
#include "../../sili/lib/headers/linear_didldo.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char** argv) {
    int n_in = argc > 1 ? std::atoi(argv[1]) : 288;
    int n_out = argc > 2 ? std::atoi(argv[2]) : 288;
    int num_cpus = argc > 3 ? std::atoi(argv[3]) : 8;
    int reps = argc > 4 ? std::atoi(argv[4]) : 30;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<int> batches = {1, 32, 64, 128, 256, 512, 1024};
    DenseLinearWeights weights;
    weights.resize(std::size_t(n_in), std::size_t(n_out));
    for (auto& v : weights.w)
        v = dist(rng);
    const std::vector<float> w0 =
        weights.w; // reset point -- every batch starts from the same weights

    // bwd0/bwdX match test_perf_matrix_vs_torch.py's op set (fwd+grad,
    // fwd+grad+step) for an apples-to-apples ms comparison.
    std::printf("n_in=%d n_out=%d num_cpus=%d\n", n_in, n_out, num_cpus);
    std::printf("%6s %14s %14s %14s\n", "batch", "fwd_ms", "bwd0_ms", "bwdX_ms");

    for (int batch : batches) {
        std::vector<float> x(std::size_t(batch) * n_in), y(std::size_t(batch) * n_out),
            dy(std::size_t(batch) * n_out), dx(std::size_t(batch) * n_in);
        for (auto& v : x)
            v = dist(rng);
        for (auto& v : dy)
            v = dist(rng);

        // warm-up must exercise every op the timed loop calls (incl.
        // lr!=0) or the first real call pays one-time spin-up cost.
        weights.w = w0;
        for (int i = 0; i < 5; ++i) {
            didldo_forward(x.data(), weights, y.data(), batch, num_cpus);
            didldo_backward(x.data(), dy.data(), weights, dx.data(), batch, num_cpus, 1e-3f);
        }

        double t_fwd = 0, t_bwd0 = 0, t_bwdX = 0;
        for (int r = 0; r < reps; ++r) {
            double t0 = now_ms();
            didldo_forward(x.data(), weights, y.data(), batch, num_cpus);
            double t1 = now_ms();
            t_fwd += t1 - t0;

            t0 = now_ms();
            didldo_forward(x.data(), weights, y.data(), batch, num_cpus);
            didldo_backward(x.data(), dy.data(), weights, dx.data(), batch, num_cpus, 0.0f);
            t1 = now_ms();
            t_bwd0 += t1 - t0;

            t0 = now_ms();
            didldo_forward(x.data(), weights, y.data(), batch, num_cpus);
            didldo_backward(x.data(), dy.data(), weights, dx.data(), batch, num_cpus, 1e-3f);
            t1 = now_ms();
            t_bwdX += t1 - t0;
        }
        std::printf("%6d %14.4f %14.4f %14.4f\n", batch, t_fwd / reps, t_bwd0 / reps,
                    t_bwdX / reps);
    }
    return 0;
}
