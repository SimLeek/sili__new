// Sentinel-policy dispatch test: proves disldo_backward's SynapsePolicyT
// template parameter genuinely reaches EVERY one of its 8 real call sites
// (scattered scalar + block4/SIMD), not just the ones remembered by hand.
// Direct motivation: this session's earlier square-then-sum revert missed
// 2 SIMD sites and was only caught because test_block4_scattered_
// divergence.cpp happened to exercise them -- this test exists so a future
// change to the ci-update template doesn't need to get that lucky again.
//
// SentinelSynapsePolicy::update_cw ignores ci/eff_lr/eps entirely and
// always returns a HUGE signed delta (opposite sign of g, like a real
// descent step but 1e6x larger than anything real training at these
// g/eff_lr magnitudes could ever produce) -- deterministic rounding
// (StochasticRounding=false) then saturates every touched weight to
// FP4's own max representable magnitude (6.0, see FP4_TABLE) with a sign
// matching -g. If EVERY touched weight in BOTH the scattered-CSR arm and
// the block4 arm ends up at exactly +-6.0, that proves the template
// parameter dispatched correctly everywhere -- if even one of the 8 call
// sites had been missed (still using the real RMSprop formula), that
// specific weight would show a normal small delta instead of saturating,
// exactly the class of bug this test is built to catch.
#include "../../sili/lib/headers/delta_csr_memory.hpp"
#include "../../sili/lib/headers/linear_disldo.hpp"
#include <cstdio>
#include <cmath>
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
using Weights = SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE>;

// Test-only policy -- deliberately NOT in delta_csr_types.hpp alongside the
// real Plain/Bounded policies (matches NoScalePolicy's own "diagnostic
// only" precedent for the ScalePolicy family): its whole job is to make
// dispatch failures loud and easy to spot, not to be a real optimizer.
template <typename VALUE_TYPE> struct SentinelSynapsePolicy {
    // ci itself isn't what this test checks (packed FP4 importance storage
    // makes an exact-value assertion on it awkward) -- reuse the real
    // Plain formula so ci stays finite/sane and doesn't itself become the
    // reason something looks wrong.
    static VALUE_TYPE update_ci(VALUE_TYPE ci, VALUE_TYPE g, VALUE_TYPE contrib, VALUE_TYPE beta2,
                                VALUE_TYPE min_decay_frac, VALUE_TYPE max_ci) {
        return PlainRMSpropSynapsePolicy<VALUE_TYPE>::update_ci(ci, g, contrib, beta2,
                                                                min_decay_frac, max_ci);
    }
    static VALUE_TYPE update_cw(VALUE_TYPE g, VALUE_TYPE /*ci*/, VALUE_TYPE /*S*/,
                                VALUE_TYPE /*eff_lr*/, VALUE_TYPE /*eps*/,
                                bool /*damp_by_importance*/, VALUE_TYPE /*max_abs_delta*/,
                                bool /*scale_invariant*/ = false) {
        return g >= VALUE_TYPE(0) ? VALUE_TYPE(-1e6) : VALUE_TYPE(1e6);
    }
};

template <> struct SentinelSynapsePolicy<Block4Vec> {
    static Block4Vec update_ci(Block4Vec ci, Block4Vec g, Block4Vec contrib, Block4Vec beta2,
                               Block4Vec min_decay_frac, Block4Vec max_ci) {
        return PlainRMSpropSynapsePolicy<Block4Vec>::update_ci(ci, g, contrib, beta2,
                                                               min_decay_frac, max_ci);
    }
    static Block4Vec update_cw(Block4Vec g, Block4Vec /*ci*/, Block4Vec /*S*/, Block4Vec /*eff_lr*/,
                               Block4Vec /*eps*/, bool /*damp_by_importance*/,
                               Block4Vec /*max_abs_delta*/, bool /*scale_invariant*/ = false) {
        Block4Vec result;
        for (int i = 0; i < SILI_BLOCK4_TILE_SIZE; ++i)
            result[i] = g[i] >= 0.0f ? -1e6f : 1e6f;
        return result;
    }
};

int main() {
    // n_out=6 is deliberately NOT a multiple of BLOCK4_TILE(4) -- forces
    // disldo_backward's "full_tile_cols" SIMD branch for the first
    // tile-column (cols 0-3) AND its scalar boundary-tile fallback branch
    // for the second, partial tile-column (cols 4-5) in the SAME run.
    // n_in=8, n_out=8 alone (both exact multiples of 4) was checked first
    // and only ever exercised the SIMD branch -- confirmed directly it
    // silently never reached the boundary-fallback sites at all, which
    // would have let a real dispatch bug there go undetected by this test.
    const std::size_t n_in = 8, n_out = 6;

    std::vector<float> dense_w(n_in * n_out);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        dense_w[i] = 1.0f + 0.07f * float(i % 7);
    std::vector<uint8_t> weight_codes(n_in * n_out), importance_codes(n_in * n_out, 0);
    for (std::size_t i = 0; i < dense_w.size(); ++i)
        weight_codes[i] = fp4_quantize(dense_w[i]);

    std::vector<float> input(n_in), dy(n_out);
    for (std::size_t r = 0; r < n_in; ++r)
        input[r] = 0.3f + 0.1f * float(r);
    for (std::size_t c = 0; c < n_out; ++c)
        dy[c] = -0.2f + 0.05f * float(c);

    // ── Arm A: everything in scattered CSR, block4 left empty ─────────────
    Weights weights_a;
    {
        std::vector<SIZE_TYPE> ptrs(n_in + 1);
        std::vector<SIZE_TYPE> idx(n_in * n_out);
        std::vector<float> w(n_in * n_out), imp(n_in * n_out, 0.0f);
        for (std::size_t r = 0; r < n_in; ++r) {
            ptrs[r] = SIZE_TYPE(r * n_out);
            for (std::size_t c = 0; c < n_out; ++c) {
                idx[r * n_out + c] = SIZE_TYPE(c);
                w[r * n_out + c] = FP4_TABLE[weight_codes[r * n_out + c] & 0x0Fu];
            }
        }
        ptrs[n_in] = SIZE_TYPE(n_in * n_out);
        weights_a.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            ptrs, idx, w, imp, n_in, n_out, n_in * n_out * 2, n_in * n_out * 2);
    }
    weights_a.out_degree.assign(n_out, SIZE_TYPE(n_in));

    // ── Arm B: everything in block4, scattered CSR left empty ─────────────
    Weights weights_b;
    {
        std::vector<SIZE_TYPE> ptrs_empty(n_in + 1, 0);
        std::vector<SIZE_TYPE> idx_empty;
        std::vector<float> w_empty, imp_empty;
        weights_b.connections = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
            ptrs_empty, idx_empty, w_empty, imp_empty, n_in, n_out, 0, 0);
    }
    weights_b.out_degree.assign(n_out, SIZE_TYPE(n_in));
    weights_b.block4.init(n_in, n_out);
    weights_b.block4.set_limits(n_in * n_out * 2, n_in * n_out * 2);
    for (std::size_t r = 0; r < n_in; ++r) {
        for (std::size_t c = 0; c < n_out; ++c) {
            const uint32_t br = uint32_t(r / 4), bc = uint32_t(c / 4);
            const uint32_t li = uint32_t(r % 4), lj = uint32_t(c % 4);
            auto tile = weights_b.block4.get_or_create(br, bc);
            tile.at(li, lj) = weight_codes[r * n_out + c] | (importance_codes[r * n_out + c] << 4);
        }
    }
    // Ceiling division -- n_out=6 has a partial (boundary) tile-column at
    // bc=1 that plain integer division (n_out/4=1) would silently skip,
    // leaving that tile never compressed/finalized.
    for (uint32_t br = 0; br < uint32_t((n_in + 3) / 4); ++br)
        for (uint32_t bc = 0; bc < uint32_t((n_out + 3) / 4); ++bc)
            weights_b.block4.maybe_compress(br, bc);

    // ── Backward with the sentinel policy on both arms ─────────────────────
    std::vector<float> dx_a(n_in, 0.0f), dx_b(n_in, 0.0f);
    std::vector<float> ni_a(n_in, 0.0f), ng_a(n_out, 0.0f);
    std::vector<float> ni_b(n_in, 0.0f), ng_b(n_out, 0.0f);
    const float lr = 0.05f;
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false,
                    SentinelSynapsePolicy>(input.data(), 1, SIZE_TYPE(n_in), dy.data(), weights_a,
                                           dx_a.data(), ni_a.data(), ng_a.data(), lr, 1);
    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE, RMSpropScalePolicy<float>, false, false,
                    SentinelSynapsePolicy>(input.data(), 1, SIZE_TYPE(n_in), dy.data(), weights_b,
                                           dx_b.data(), ni_b.data(), ng_b.data(), lr, 1);

    // ── Every touched weight in BOTH arms must have saturated to +-6.0 ─────
    for (std::size_t r = 0; r < n_in; ++r) {
        auto cursor = weights_a.connections.row_cursor(r);
        const auto& L = weights_a.connections.layout;
        const std::size_t row_nnz = L.row_nnz(r);
        for (std::size_t e = 0; e < row_nnz; ++e) {
            const COL_TYPE col = cursor.advance();
            const std::size_t c = std::size_t(col);
            const std::size_t vb = L.elem_start[r] + e;
            const float w_a = ValueAccessor<FP4BiPacked>::get_w(weights_a.connections.values, vb);

            const uint32_t br = uint32_t(r / 4), bc = uint32_t(c / 4);
            const uint32_t li = uint32_t(r % 4), lj = uint32_t(c % 4);
            auto tile = weights_b.block4.find(br, bc);
            const float w_b = FP4_TABLE[tile.at(li, lj) & 0x0Fu];

            CHECK(std::abs(std::abs(w_a) - 6.0f) < 1e-3f,
                  "scattered weight[%zu][%zu] did NOT saturate to the sentinel's +-6.0 -- "
                  "got %.6f (SynapsePolicyT may not be reaching this scattered call site)",
                  r, c, w_a);
            CHECK(std::abs(std::abs(w_b) - 6.0f) < 1e-3f,
                  "block4 weight[%zu][%zu] did NOT saturate to the sentinel's +-6.0 -- "
                  "got %.6f (SynapsePolicyT may not be reaching this block4/SIMD call site)",
                  r, c, w_b);
        }
    }

    if (g_fail == 0) {
        std::printf("test_synapse_policy_dispatch: all sites dispatch correctly (%zu synapses x 2 "
                    "arms checked)\n",
                    n_in * n_out);
        return 0;
    }
    std::printf("test_synapse_policy_dispatch: %d failure(s)\n", g_fail);
    return 1;
}
