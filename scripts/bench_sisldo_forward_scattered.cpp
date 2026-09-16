// Real-kernel A/B harness for Phase 5 (see TODO_BATCH_BLOCKING.md):
// times sisldo_forward's SCATTERED (non-block4) branch specifically --
// its reduction was already tree-based (not fully serial, unlike the
// block4 branch), so the win here is expected to be smaller; verified
// rather than assumed.
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
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

int main(int argc, char** argv) {
    const int num_cpus = argc > 1 ? std::atoi(argv[1]) : 4;
    const std::size_t n_in = 288, n_out = 288;
    const float weight_density = 0.1f, input_density = 0.1f;
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> idist(0.0f, 1.0f);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);

    std::vector<SIZE_TYPE> wptrs(n_in + 1, 0);
    std::vector<SIZE_TYPE> widx;
    std::vector<float> wvals, wimp;
    const std::size_t wk = std::max<std::size_t>(1, std::size_t(weight_density * float(n_out)));
    std::vector<SIZE_TYPE> wcol_pool(n_out);
    std::iota(wcol_pool.begin(), wcol_pool.end(), SIZE_TYPE(0));
    for (std::size_t r = 0; r < n_in; ++r) {
        std::shuffle(wcol_pool.begin(), wcol_pool.end(), rng);
        std::vector<SIZE_TYPE> row_cols(wcol_pool.begin(), wcol_pool.begin() + std::ptrdiff_t(wk));
        std::sort(row_cols.begin(), row_cols.end());
        for (SIZE_TYPE c : row_cols) {
            widx.push_back(c);
            wvals.push_back(wdist(rng));
            wimp.push_back(idist(rng));
        }
        wptrs[r + 1] = wptrs[r] + SIZE_TYPE(row_cols.size());
    }

    Weights weights;
    weights.connections = delta_csr_from_absolute<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        wptrs, widx, wvals, wimp, n_in, n_out, 1u << 20, 1u << 20, 0.2f);
    weights.recompute_stats();

    std::printf("# sisldo_forward SCATTERED real-kernel bench: n_in=%zu n_out=%zu num_cpus=%d "
                "weight_density=%.2f input_density=%.2f nnz=%zu block4_tiles=%zu\n",
                n_in, n_out, num_cpus, double(weight_density), double(input_density), widx.size(),
                weights.block4.n_tiles());

    const std::size_t k = std::max<std::size_t>(1, std::size_t(input_density * float(n_in)));
    for (int batch : {1, 4, 8, 16, 32, 64, 256, 1024}) {
        std::vector<SIZE_TYPE> ptrs(std::size_t(batch) + 1, 0);
        std::vector<SIZE_TYPE> idx;
        std::vector<float> vals;
        std::vector<SIZE_TYPE> col_pool(n_in);
        std::iota(col_pool.begin(), col_pool.end(), SIZE_TYPE(0));
        for (int b = 0; b < batch; ++b) {
            std::shuffle(col_pool.begin(), col_pool.end(), rng);
            std::vector<SIZE_TYPE> row_cols(col_pool.begin(), col_pool.begin() + std::ptrdiff_t(k));
            std::sort(row_cols.begin(), row_cols.end());
            for (SIZE_TYPE c : row_cols) {
                idx.push_back(c);
                vals.push_back(xdist(rng));
            }
            ptrs[std::size_t(b) + 1] = ptrs[std::size_t(b)] + SIZE_TYPE(row_cols.size());
        }
        auto x_sparse =
            make_csr_input<SIZE_TYPE, float>(SIZE_TYPE(batch), SIZE_TYPE(n_in), ptrs, idx, vals);

        std::vector<float> y(std::size_t(batch) * n_out);
        const int n_warmup = 5;
        for (int i = 0; i < n_warmup; ++i) {
            std::fill(y.begin(), y.end(), 0.f);
            sisldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x_sparse, weights,
                                                                         y.data(), num_cpus);
        }
        const int n_reps = std::max(10, 20000 / std::max(batch, 1));
        std::vector<double> call_ns(n_reps);
        for (int i = 0; i < n_reps; ++i) {
            std::fill(y.begin(), y.end(), 0.f);
            auto t0 = std::chrono::steady_clock::now();
            sisldo_forward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(x_sparse, weights,
                                                                         y.data(), num_cpus);
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
