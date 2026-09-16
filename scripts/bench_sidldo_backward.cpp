// SIDLDO backward bench (TODO_BATCH_BLOCKING.md "Also queued") -- times
// sidldo_backward (dense x, sparse dy union-gather, no high-batch
// variant yet) across the batch range, for direct comparison against
// sisldo_ms/torch_ms bwd0/bwdX already recorded in the Block4 Bench
// dataset at matching (n, density) points.
#include "../sili/lib/headers/linear_sidldo.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char** argv) {
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 288;
    const int n_out = argc > 2 ? std::atoi(argv[2]) : 288;

    std::mt19937 rng(21);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    DenseLinearWeights weights;
    weights.resize(std::size_t(n_in), std::size_t(n_out));
    for (auto& v : weights.w)
        v = dist(rng);

    std::vector<float> densities = {0.005f, 0.05f, 0.1f, 0.2f, 0.5f};
    std::vector<int> batches = {1, 4, 8, 16, 32, 63, 64, 128, 256, 512, 1024};

    std::printf("n_in=%d n_out=%d\n", n_in, n_out);
    std::printf("%10s %6s %14s %14s\n", "density", "batch", "bwd0_ms", "bwdX_ms");

    for (float density : densities) {
        for (int batch : batches) {
            std::vector<int> col_pool(n_out);
            std::iota(col_pool.begin(), col_pool.end(), 0);
            const std::size_t k = std::max<std::size_t>(1, std::size_t(density * float(n_out)));
            std::vector<int> dy_ptrs(std::size_t(batch) + 1, 0);
            std::vector<int> dy_idx;
            std::vector<float> dy_val;
            for (int b = 0; b < batch; ++b) {
                std::shuffle(col_pool.begin(), col_pool.end(), rng);
                std::vector<int> cols(col_pool.begin(), col_pool.begin() + std::ptrdiff_t(k));
                for (int c : cols) {
                    dy_idx.push_back(c);
                    dy_val.push_back(dist(rng));
                }
                dy_ptrs[std::size_t(b) + 1] = dy_ptrs[std::size_t(b)] + int(cols.size());
            }
            std::vector<float> x(std::size_t(batch) * n_in);
            for (auto& v : x)
                v = dist(rng);

            SidldoBackwardScratch scratch;
            std::vector<float> dx(std::size_t(batch) * n_in);

            // bwd0 (lr=0, grad only)
            for (int i = 0; i < 10; ++i)
                sidldo_backward(x.data(), dy_ptrs.data(), dy_idx.data(), dy_val.data(), batch,
                                weights, scratch, dx.data(), 0.0f);
            int reps = batch >= 256 ? 20 : 100;
            double t = 0;
            for (int r = 0; r < reps; ++r) {
                double t0 = now_ms();
                sidldo_backward(x.data(), dy_ptrs.data(), dy_idx.data(), dy_val.data(), batch,
                                weights, scratch, dx.data(), 0.0f);
                t += now_ms() - t0;
            }
            double t_bwd0 = t / reps;

            // bwdX (lr!=0, grad+update) -- persistent square_avg across
            // reps, same as every other backward bench in this arc.
            for (int i = 0; i < 10; ++i)
                sidldo_backward(x.data(), dy_ptrs.data(), dy_idx.data(), dy_val.data(), batch,
                                weights, scratch, dx.data(), 1e-3f);
            t = 0;
            for (int r = 0; r < reps; ++r) {
                double t0 = now_ms();
                sidldo_backward(x.data(), dy_ptrs.data(), dy_idx.data(), dy_val.data(), batch,
                                weights, scratch, dx.data(), 1e-3f);
                t += now_ms() - t0;
            }
            double t_bwdX = t / reps;

            std::printf("%10.3f %6d %14.4f %14.4f\n", density, batch, t_bwd0, t_bwdX);
        }
    }
    return 0;
}
