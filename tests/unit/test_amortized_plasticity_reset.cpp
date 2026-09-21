// TDD (written before the implementation exists): apply_amortized_plasticity_step
// (delta_csr_types.hpp) -- the scattered-CSR half of the per-neuron
// utility-based plasticity reset (Continual-Backprop-inspired, see
// docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design for
// the full derivation and the several rounds of direct correction that
// shaped this design: (1) top-K-by-importance, not bottom-K-by-utility,
// selection -- a mathematical worst-case-timing flaw showed the original
// bottom-K design could NEVER detect a genuinely frozen-and-now-wrong
// column; (2) a LOCAL per-column gradient-activity deviation gate, not a
// global loss signal; (3) engine-safety correction -- that gate is
// derived from col_importance's own per-CYCLE delta, not a real
// backward-kernel hook (too risky, SIMD-vectorized hot path); (4) an
// asymmetric EMA catchup rate so genuine improvement is recognized
// quickly while a spike isn't instantly absorbed; (5) a SEPARATE dead-pool
// (bottom-K by importance*|weight|, the ORIGINAL formula, honestly
// rescoped to what it actually detects) since gradient-magnitude-based
// deviation alone can't distinguish "optimal, low gradient" from "dead,
// low gradient."
#include "../../sili/lib/headers/delta_csr_types.hpp"
#include <cmath>
#include <cstdio>
#include <functional>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

using VT = DeltaCSRBiValues<float>;
using Conn = DeltaCSRWeights<std::size_t, VT, uint32_t>;

// Builds a fully-dense n_in x n_out scattered CSR layer (every (i,j) a
// live synapse), all weights/importance set via the callback.
static Conn make_dense_conn(std::size_t n_in, std::size_t n_out,
                            const std::function<float(std::size_t, std::size_t)>& w_fn,
                            const std::function<float(std::size_t, std::size_t)>& imp_fn) {
    Conn conn;
    conn.layout.rows = n_in;
    conn.layout.cols = n_out;
    conn.layout.elem_start.assign(n_in + 1, 0);
    conn.layout.elem_end.assign(n_in, 0);
    conn.layout.byte_start.assign(n_in + 1, 0);
    conn.layout.byte_end.assign(n_in, 0);
    for (std::size_t r = 0; r < n_in; ++r) {
        conn.layout.elem_start[r] = r * n_out;
        conn.layout.elem_end[r] = r * n_out + n_out;
    }
    conn.layout.elem_start[n_in] = n_in * n_out;
    conn.layout.total_nnz = n_in * n_out;

    // Build the delta-encoded indices buffer: each row's columns are
    // 0,1,2,...,n_out-1 (uleb128 delta-encoded, delta=1 for col>0, delta=col for col=0).
    conn.indices_buf.clear();
    for (std::size_t r = 0; r < n_in; ++r) {
        conn.layout.byte_start[r] = conn.indices_buf.size();
        uint32_t prev = 0;
        for (std::size_t c = 0; c < n_out; ++c) {
            const uint32_t delta = static_cast<uint32_t>(c) - prev;
            uint32_t v = delta;
            do {
                uint8_t byte = v & 0x7F;
                v >>= 7;
                if (v)
                    byte |= 0x80;
                conn.indices_buf.push_back(byte);
            } while (v);
            prev = static_cast<uint32_t>(c);
        }
        conn.layout.byte_end[r] = conn.indices_buf.size();
    }
    conn.layout.byte_start[n_in] = conn.indices_buf.size();

    ValueAccessor<VT>::resize(conn.values, n_in * n_out);
    for (std::size_t r = 0; r < n_in; ++r) {
        for (std::size_t c = 0; c < n_out; ++c) {
            const std::size_t vb = r * n_out + c;
            ValueAccessor<VT>::set_live(conn.values, vb, w_fn(r, c), imp_fn(r, c));
        }
    }
    return conn;
}

static void test_col_importance_accumulates_from_known_values() {
    // 2 inputs x 3 outputs, importance varies by column only.
    auto conn = make_dense_conn(
        2, 3, [](std::size_t, std::size_t) { return 0.1f; },
        [](std::size_t, std::size_t c) {
            return static_cast<float>(c) + 1.0f;
        }); // col j -> importance j+1

    PlasticityState st;
    PlasticityCellCursor st_cur;
    auto r = apply_amortized_plasticity_step(conn, 3, st, st_cur, /*chunk_size=*/6, /*eta=*/0.0f,
                                             /*eta_slow=*/0.99f, /*eta_slow_catchup=*/0.95f,
                                             /*eta_fast=*/0.5f, /*blend=*/0.1f,
                                             /*reset_fraction=*/0.0f, /*dead_fraction=*/0.0f,
                                             /*k=*/0.5f);
    CHECK(r.cycle_complete, "6 cells, chunk_size=6 should complete in one call");
    // eta=0.0 (weight on OLD value) means col_importance == the last-seen raw importance value
    // exactly.
    CHECK(std::abs(st.col_importance[0] - 1.0f) < 1e-6f, "col 0 importance should be 1.0, got %f",
          st.col_importance[0]);
    CHECK(std::abs(st.col_importance[1] - 2.0f) < 1e-6f, "col 1 importance should be 2.0, got %f",
          st.col_importance[1]);
    CHECK(std::abs(st.col_importance[2] - 3.0f) < 1e-6f, "col 2 importance should be 3.0, got %f",
          st.col_importance[2]);
}

static void test_frozen_pool_selects_highest_importance_mature_columns() {
    // 2x4, importance strictly increasing by column: col 3 is highest.
    auto conn = make_dense_conn(
        2, 4, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t c) { return static_cast<float>(c); });
    PlasticityState st;
    PlasticityCellCursor st_cur;
    // reset_fraction=0.25 of 4 mature columns -> top_n = round(4*0.25) = 1
    // (nothing eligible cycle 1: all col_age=0 < maturity 1). Run 2 cycles.
    apply_amortized_plasticity_step(conn, 4, st, st_cur, 8, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.25f,
                                    0.0f, 0.5f);
    CHECK(st.col_reset_active[3] == 0,
          "cycle 1: nothing should be mature yet (col_age starts at 0)");
    auto r2 = apply_amortized_plasticity_step(conn, 4, st, st_cur, 8, 0.0f, 0.99f, 0.95f, 0.5f,
                                              0.1f, 0.25f, 0.0f, 0.5f);
    CHECK(r2.cycle_complete, "cycle 2 should also complete (8 cells, chunk_size=8)");
    CHECK(st.col_reset_active[3] == 1, "col 3 (highest importance) should be the frozen pool pick");
    CHECK(r2.n_reset_this_cycle == 1, "exactly 1 column should be selected, got %zu",
          r2.n_reset_this_cycle);
}

static void test_dead_pool_excludes_frozen_pool_picks() {
    // 2x4: col 3 has both highest importance (frozen candidate) AND
    // lowest importance*|weight| (would ALSO be a dead candidate if not
    // excluded) -- weight[col 3] set to 0 so col_util_dead[3]=0, the
    // global minimum, while col_importance[3] is also the maximum.
    auto conn = make_dense_conn(
        2, 4, [](std::size_t, std::size_t c) { return c == 3 ? 0.0f : 1.0f; },
        [](std::size_t, std::size_t c) { return static_cast<float>(c); });
    PlasticityState st;
    PlasticityCellCursor st_cur;
    apply_amortized_plasticity_step(conn, 4, st, st_cur, 8, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.25f,
                                    0.25f, 0.5f);
    auto r2 = apply_amortized_plasticity_step(conn, 4, st, st_cur, 8, 0.0f, 0.99f, 0.95f, 0.5f,
                                              0.1f, 0.25f, 0.25f, 0.5f);
    CHECK(st.col_reset_active[3] == 1, "col 3 should be the frozen pick (highest importance)");
    CHECK(st.col_dead_active[3] == 0,
          "col 3 must NOT also be in the dead pool despite qualifying by col_util_dead alone");
    CHECK(r2.n_dead_this_cycle == 1, "exactly 1 OTHER column should fill the dead pool, got %zu",
          r2.n_dead_this_cycle);
}

static void test_frozen_blend_is_gated_dead_blend_is_full_rate() {
    // Force col_reset_active/col_dead_active directly (bypass selection)
    // to isolate the blend-application logic itself.
    auto conn = make_dense_conn(
        1, 2, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t) { return 1.0f; });
    PlasticityState st;
    PlasticityCellCursor st_cur;
    st.ensure_sized(2);
    st.col_reset_active[0] = 1;
    st.col_plasticity_boost[0] = 0.5f; // gated: effective strength = blend*0.5
    st.col_dead_active[1] = 1;         // ungated: effective strength = blend (full)

    apply_amortized_plasticity_step(conn, 2, st, st_cur, 2, 0.0f, 0.99f, 0.95f, 0.5f,
                                    /*blend=*/0.2f, 0.0f, 0.0f, 0.5f);
    const float w0 = ValueAccessor<VT>::get_w(conn.values, 0); // col 0, gated
    const float w1 = ValueAccessor<VT>::get_w(conn.values, 1); // col 1, ungated
    // strength0 = 0.2*0.5 = 0.1 -> w0 = 0.9*1.0 + 0.1*fresh (fresh is bounded, |fresh|<10 in
    // practice) strength1 = 0.2       -> w1 = 0.8*1.0 + 0.2*fresh Isolate the DETERMINISTIC
    // (non-fresh) component: w = (1-strength)*w_orig + strength*fresh, so |w - (1-strength)*w_orig|
    // should be small relative to a much larger jump if strength were 1.0.
    CHECK(std::abs(w0 - 0.9f) < 5.0f, "gated blend sanity: w0=%f (loose bound, RNG-dependent)", w0);
    CHECK(std::abs(w1 - 0.8f) < 5.0f, "full-rate blend sanity: w1=%f (loose bound, RNG-dependent)",
          w1);
    // The REAL check: gated strength (0.1) must move LESS than full-rate (0.2) on average.
    // Deterministic check instead: importance shrinks by exactly (1-strength) each (no RNG there).
    const float imp0 = ValueAccessor<VT>::get_imp(conn.values, 0);
    const float imp1 = ValueAccessor<VT>::get_imp(conn.values, 1);
    CHECK(std::abs(imp0 - 0.9f) < 1e-5f,
          "gated importance should shrink by exactly (1-0.1)=0.9, got %f", imp0);
    CHECK(std::abs(imp1 - 0.8f) < 1e-5f,
          "full-rate importance should shrink by exactly (1-0.2)=0.8, got %f", imp1);
}

static void test_maturity_gate_excludes_just_reset_column() {
    auto conn = make_dense_conn(
        1, 2, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t c) { return c == 0 ? 10.0f : 1.0f; });
    PlasticityState st;
    PlasticityCellCursor st_cur;
    // Cycle 1: nothing mature yet.
    apply_amortized_plasticity_step(conn, 2, st, st_cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.5f,
                                    0.0f, 0.5f);
    // Cycle 2: col 0 (highest importance) becomes mature and gets selected+reset (age->0).
    apply_amortized_plasticity_step(conn, 2, st, st_cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.5f,
                                    0.0f, 0.5f);
    CHECK(st.col_age[0] == 0, "col 0 should have just been reset, age=0");
    // Cycle 3: col 0's age is now 0 (< maturity 1) -- must NOT be immediately reselected
    // even though its importance is still (likely) highest.
    auto r3 = apply_amortized_plasticity_step(conn, 2, st, st_cur, 2, 0.0f, 0.99f, 0.95f, 0.5f,
                                              0.1f, 0.5f, 0.0f, 0.5f);
    CHECK(st.col_reset_active[0] == 0, "just-reset column must not be immediately re-eligible");
    (void)r3;
}

static void test_empty_layer() {
    Conn conn;
    conn.layout.rows = 0;
    PlasticityState st;
    PlasticityCellCursor st_cur;
    auto r = apply_amortized_plasticity_step(conn, 0, st, st_cur, 4, 0.99f, 0.99f, 0.95f, 0.5f,
                                             0.1f, 0.01f, 0.01f, 0.5f);
    CHECK(r.cycle_complete, "empty layer should report cycle_complete=true immediately");
    CHECK(r.n_reset_this_cycle == 0 && r.n_dead_this_cycle == 0,
          "empty layer should reset nothing");
}

static void test_asymmetric_catchup_rate() {
    // Directly exercise the col_grad_slow asymmetric-rate branch by
    // constructing a scenario where col_importance drops sharply between
    // two cycles (simulating genuine improvement) vs rises sharply
    // (simulating a real problem) and checking which rate each used.
    auto conn_drop = make_dense_conn(
        1, 1, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t) { return 10.0f; });
    PlasticityState st_drop;
    PlasticityCellCursor st_drop_cur;
    apply_amortized_plasticity_step(
        conn_drop, 1, st_drop, st_drop_cur, 1, 0.0f, 0.99f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f,
        0.5f); // cycle 1: col_importance -> 10, delta=10-0=10 (rise, slow rate)
    // Manually drop the underlying importance to simulate genuine improvement next cycle.
    ValueAccessor<VT>::set_live(conn_drop.values, 0, 1.0f, 1.0f);
    apply_amortized_plasticity_step(
        conn_drop, 1, st_drop, st_drop_cur, 1, 0.0f, 0.99f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f,
        0.5f); // cycle 2: col_importance -> 1, delta=1-10=-9 (drop, catchup rate)
    // col_grad_slow after cycle 2 should reflect the FAST catchup rate (0.5) applied to delta=-9,
    // not the slow rate (0.99): slow_after = 0.5*slow_before + 0.5*(-9).
    // slow_before (after cycle1, rise, rate=0.99): 0.99*0 + 0.01*10 = 0.1
    const float expected_slow_after_cycle1 =
        0.01f * 10.0f; // eta_slow=0.99 -> weight (1-eta_slow)=0.01
    const float expected_slow_after_cycle2 =
        0.5f * expected_slow_after_cycle1 + 0.5f * (-9.0f); // catchup rate 0.5 applied
    CHECK(std::abs(st_drop.col_grad_slow[0] - expected_slow_after_cycle2) < 1e-4f,
          "col_grad_slow should use the FAST catchup rate on a sustained drop, expected %f got %f",
          expected_slow_after_cycle2, st_drop.col_grad_slow[0]);
}

int main() {
    test_col_importance_accumulates_from_known_values();
    test_frozen_pool_selects_highest_importance_mature_columns();
    test_dead_pool_excludes_frozen_pool_picks();
    test_frozen_blend_is_gated_dead_blend_is_full_rate();
    test_maturity_gate_excludes_just_reset_column();
    test_empty_layer();
    test_asymmetric_catchup_rate();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
