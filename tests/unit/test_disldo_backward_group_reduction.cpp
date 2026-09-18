// Correctness check for Phase 6's group_dx/group_col_grad persistent-
// scratch fix (TODO_BATCH_BLOCKING.md) -- specifically the num_groups>1
// branch, which only triggers on real dual-CCX hardware at high enough
// num_cpus (sili_topology::current_thread_group() must actually return >1
// distinct values across threads). Compares disldo_backward's dx output
// at num_cpus=1 (trivially num_groups=1) against several higher num_cpus
// values -- was a manual eyeball-the-printed-sum script
// (scripts/verify_disldo_backward_group_reduction.cpp), converted to a
// real assertion per the scripts/ triage (see commit history).
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

using SIZE_TYPE = int;
using COL_TYPE = uint32_t;
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>;

static Weights make_layer(std::size_t n_in, std::size_t n_out, std::mt19937& rng) {
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> idist(0.1f, 1.0f);
    std::vector<float> wv(n_in * n_out), imp(n_in * n_out);
    for (auto& w : wv)
        w = wdist(rng);
    for (auto& i : imp)
        i = idist(rng);
    Weights weights;
    // Properly-initialized EMPTY scattered CSR (not just .rows/.cols set by
    // hand) -- disldo_backward's row_nnz() walk needs real elem_start/
    // byte_start vectors even for a pure-block4 layer, unlike disldo_forward
    // which short-circuits on dc.empty() before touching them.
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
    return weights;
}

static std::vector<float> run_at(int num_cpus, std::size_t n_in, std::size_t n_out, SIZE_TYPE batch,
                                 const std::vector<float>& input,
                                 const std::vector<float>& output_grad, std::mt19937 layer_seed) {
    Weights weights = make_layer(n_in, n_out, layer_seed);
    std::vector<float> input_grad(std::size_t(batch) * n_in, 0.f);
    std::vector<float> neuron_input_accum(n_in, 0.f), neuron_grad_accum(n_out, 0.f);
    disldo_backward<SIZE_TYPE, DeltaCSRBiValues<float>, COL_TYPE>(
        input.data(), batch, SIZE_TYPE(n_in), output_grad.data(), weights, input_grad.data(),
        neuron_input_accum.data(), neuron_grad_accum.data(), 0.001f, num_cpus);
    return input_grad;
}

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}

int main() {
    // Deliberately not multiples of BLOCK4_TILE, matching the original
    // manual script's shape choice (catches boundary-tile bugs the group-
    // reduction fix's own row/tile bookkeeping could introduce).
    const std::size_t n_in = 289, n_out = 271;
    const SIZE_TYPE batch = 64;
    std::mt19937 seed_rng(99);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::vector<float> input(std::size_t(batch) * n_in), output_grad(std::size_t(batch) * n_out);
    for (auto& v : input)
        v = xdist(seed_rng);
    for (auto& v : output_grad)
        v = xdist(seed_rng);

    // num_cpus=1 is the reference: trivially num_groups=1, already covered
    // by the existing ctest suite elsewhere -- every higher num_cpus value
    // must match it regardless of whether THIS machine's topology actually
    // has >1 CCX group (the fix must be correct either way; dual-CCX
    // hardware just exercises the num_groups>1 branch specifically).
    const auto ref = run_at(1, n_in, n_out, batch, input, output_grad, std::mt19937(42));
    for (int num_cpus : {4, 8, 16}) {
        const auto grad =
            run_at(num_cpus, n_in, n_out, batch, input, output_grad, std::mt19937(42));
        const double err = max_abs_diff(grad, ref);
        CHECK(err < 1e-3,
              "num_cpus=%d input_grad diverges from num_cpus=1 reference, max abs err %.6f",
              num_cpus, err);
    }

    if (g_fail == 0)
        std::printf("PASS: all disldo_backward_group_reduction checks\n");
    return g_fail;
}
