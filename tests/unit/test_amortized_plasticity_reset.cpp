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
// quickly while a spike isn't instantly absorbed; (5) the dead pool
// (bottom-K by importance*|weight|) was tried and then PRUNED entirely
// (third correction round) -- its own touch shrank importance further,
// making a touched column MORE likely to be re-selected next cycle, a
// self-reinforcing spiral confirmed to cause a real accuracy regression
// across a full 100k-step relaunch. Only the frozen pool remains.
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
                                             /*reset_fraction=*/0.0f, /*k=*/0.5f);
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
                                    0.5f);
    CHECK(st.col_reset_active[3] == 0,
          "cycle 1: nothing should be mature yet (col_age starts at 0)");
    auto r2 = apply_amortized_plasticity_step(conn, 4, st, st_cur, 8, 0.0f, 0.99f, 0.95f, 0.5f,
                                              0.1f, 0.25f, 0.5f);
    CHECK(r2.cycle_complete, "cycle 2 should also complete (8 cells, chunk_size=8)");
    CHECK(st.col_reset_active[3] == 1, "col 3 (highest importance) should be the frozen pool pick");
    CHECK(r2.n_reset_this_cycle == 1, "exactly 1 column should be selected, got %zu",
          r2.n_reset_this_cycle);
}

static void test_frozen_blend_is_gated_by_plasticity_boost() {
    // Force col_reset_active directly (bypass selection) to isolate the
    // blend-application logic itself.
    auto conn = make_dense_conn(
        1, 1, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t) { return 1.0f; });
    PlasticityState st;
    PlasticityCellCursor st_cur;
    st.ensure_sized(1);
    st.col_reset_active[0] = 1;
    st.col_plasticity_boost[0] = 0.5f; // effective strength = blend*0.5

    apply_amortized_plasticity_step(conn, 1, st, st_cur, 1, 0.0f, 0.99f, 0.95f, 0.5f,
                                    /*blend=*/0.2f, 0.0f, 0.5f);
    const float w0 = ValueAccessor<VT>::get_w(conn.values, 0);
    // strength = 0.2*0.5 = 0.1 -> w0 = 0.9*1.0 + 0.1*fresh (fresh is bounded, |fresh|<10 in
    // practice). Isolate the DETERMINISTIC (non-fresh) component via importance, which has no
    // RNG in its update: imp shrinks by exactly (1-strength).
    CHECK(std::abs(w0 - 0.9f) < 5.0f, "gated blend sanity: w0=%f (loose bound, RNG-dependent)", w0);
    const float imp0 = ValueAccessor<VT>::get_imp(conn.values, 0);
    CHECK(std::abs(imp0 - 0.9f) < 1e-5f,
          "gated importance should shrink by exactly (1-0.1)=0.9, got %f", imp0);
}

static void test_maturity_gate_excludes_just_reset_column() {
    auto conn = make_dense_conn(
        1, 2, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t c) { return c == 0 ? 10.0f : 1.0f; });
    PlasticityState st;
    PlasticityCellCursor st_cur;
    // Cycle 1: nothing mature yet.
    apply_amortized_plasticity_step(conn, 2, st, st_cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.5f,
                                    0.5f);
    // Cycle 2: col 0 (highest importance) becomes mature and gets selected+reset (age->0).
    apply_amortized_plasticity_step(conn, 2, st, st_cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.5f,
                                    0.5f);
    CHECK(st.col_age[0] == 0, "col 0 should have just been reset, age=0");
    // Cycle 3: col 0's age is now 0 (< maturity 1) -- must NOT be immediately reselected
    // even though its importance is still (likely) highest.
    auto r3 = apply_amortized_plasticity_step(conn, 2, st, st_cur, 2, 0.0f, 0.99f, 0.95f, 0.5f,
                                              0.1f, 0.5f, 0.5f);
    CHECK(st.col_reset_active[0] == 0, "just-reset column must not be immediately re-eligible");
    (void)r3;
}

static void test_empty_layer() {
    Conn conn;
    conn.layout.rows = 0;
    PlasticityState st;
    PlasticityCellCursor st_cur;
    auto r = apply_amortized_plasticity_step(conn, 0, st, st_cur, 4, 0.99f, 0.99f, 0.95f, 0.5f,
                                             0.1f, 0.01f, 0.5f);
    CHECK(r.cycle_complete, "empty layer should report cycle_complete=true immediately");
    CHECK(r.n_reset_this_cycle == 0, "empty layer should reset nothing");
}

static void test_asymmetric_catchup_rate() {
    // Directly exercise the col_grad_slow asymmetric-rate branch by
    // constructing a scenario where col_importance drops sharply between
    // two cycles (simulating genuine improvement) vs rises sharply
    // (simulating a real problem) and checking which rate each used.
    // Cycle 1 warm-starts col_grad_slow/fast to the real delta (cold-start
    // fix, see test_cold_start_no_spurious_deviation below) -- this test
    // exercises the asymmetric rate starting from cycle 2 onward, once
    // both EMAs already hold a real value.
    auto conn_drop = make_dense_conn(
        1, 1, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t) { return 10.0f; });
    PlasticityState st_drop;
    PlasticityCellCursor st_drop_cur;
    apply_amortized_plasticity_step(
        conn_drop, 1, st_drop, st_drop_cur, 1, 0.0f, 0.99f, 0.5f, 0.5f, 0.0f, 0.0f,
        0.5f); // cycle 1: col_importance -> 10, delta=10-0=10 (warm-start, slow := 10)
    // Manually drop the underlying importance to simulate genuine improvement next cycle.
    ValueAccessor<VT>::set_live(conn_drop.values, 0, 1.0f, 1.0f);
    apply_amortized_plasticity_step(
        conn_drop, 1, st_drop, st_drop_cur, 1, 0.0f, 0.99f, 0.5f, 0.5f, 0.0f, 0.0f,
        0.5f); // cycle 2: col_importance -> 1, delta=1-10=-9 (drop vs slow=10, catchup rate)
    // col_grad_slow after cycle 2 should reflect the FAST catchup rate (0.5) applied to delta=-9,
    // not the slow rate (0.99): slow_after = 0.5*slow_before + 0.5*(-9).
    // slow_before (after cycle 1, warm-start): exactly 10.0.
    const float expected_slow_after_cycle1 = 10.0f; // warm-start, not an EMA blend from 0
    const float expected_slow_after_cycle2 =
        0.5f * expected_slow_after_cycle1 + 0.5f * (-9.0f); // catchup rate 0.5 applied
    CHECK(std::abs(st_drop.col_grad_slow[0] - expected_slow_after_cycle2) < 1e-4f,
          "col_grad_slow should use the FAST catchup rate on a sustained drop, expected %f got %f",
          expected_slow_after_cycle2, st_drop.col_grad_slow[0]);
}

static void test_cold_start_no_spurious_deviation() {
    // Direct instruction: "don't start an ema with zero grad since that
    // never happens... if the variable is zero/just initialized then the
    // new value replaces it." A column's FIRST cycle must warm-start
    // col_grad_slow/fast to the real delta (not the ~50x spurious ratio
    // the old zero-init math produced -- the bug actually observed in a
    // real width=288 smoke run, dev=47.49 at step 250, before any column
    // could legitimately be frozen). With the z-score deviation formula,
    // a fresh column also has col_grad_var=0, so deviation is exactly 0
    // (not 1.0 -- z-score's own "no deviation from itself yet" baseline),
    // an even safer never-flagged starting point.
    auto conn = make_dense_conn(
        1, 1, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t) { return 10.0f; });
    PlasticityState st;
    PlasticityCellCursor cur;
    apply_amortized_plasticity_step(conn, 1, st, cur, 1, 0.0f, 0.99f, 0.95f, 0.5f, 0.0f, 0.0f,
                                    0.5f); // cycle 1: col_importance -> 10, delta=10
    CHECK(std::abs(st.col_grad_slow[0] - 10.0f) < 1e-6f,
          "col_grad_slow should warm-start to the real first delta (10.0), got %f",
          st.col_grad_slow[0]);
    CHECK(std::abs(st.col_grad_fast[0] - 10.0f) < 1e-6f,
          "col_grad_fast should warm-start to the real first delta (10.0), got %f",
          st.col_grad_fast[0]);
    CHECK(std::abs(st.col_grad_var[0]) < 1e-6f,
          "col_grad_var should be exactly 0 on a column's first cycle (no variance signal yet), "
          "got %f",
          st.col_grad_var[0]);
    const float std_dev = std::sqrt(std::max(0.0f, st.col_grad_var[0]));
    const float deviation = (st.col_grad_fast[0] - st.col_grad_slow[0]) / (std_dev + 1e-8f);
    CHECK(std::abs(deviation) < 1e-4f,
          "deviation on a column's first cycle should be exactly ~0 (no spurious cold-start "
          "spike), got %f",
          deviation);
}

static void test_reset_inflates_variance_preventing_immediate_reflag() {
    // Second-round correction, direct instruction: the reset action's own
    // `(1-strength)` importance decay must not corrupt the very signal
    // that decides future resets. A real width=288 smoke run showed the
    // SAME layer/pool pinned at dev=16-17 across multiple cycles instead
    // of settling -- the reset's own decay produces a large artificial
    // delta next cycle that (without this fix) looks like a genuine
    // spike, re-flagging the column it just reset. Construct a column
    // that settles into a STABLE, small per-cycle delta after being
    // reset (simulating a column that's genuinely fine post-reset, with
    // no real new problem) and confirm it does NOT get immediately
    // reselected once its self-induced perturbation is accounted for.
    auto conn = make_dense_conn(
        1, 1, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t) { return 100.0f; });
    PlasticityState st;
    PlasticityCellCursor cur;
    // Cycle 1: warm-start (col_importance=100, delta=100, age 0 -> not mature).
    apply_amortized_plasticity_step(conn, 1, st, cur, 1, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 1.0f, 0.5f,
                                    0.9f);
    // Cycle 2: col 0 becomes mature and IS selected (only candidate, reset_fraction=1.0);
    // col_plasticity_boost gets set from deviation vs the (still-zero-variance) baseline.
    ValueAccessor<VT>::set_live(conn.values, 0, 1.0f, 100.0f);
    apply_amortized_plasticity_step(conn, 1, st, cur, 1, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 1.0f, 0.5f,
                                    0.9f);
    CHECK(st.col_reset_active[0] == 1, "col 0 should be selected into the frozen pool on cycle 2");
    const float importance_before_decay = st.col_importance_prev_cycle[0];
    // Cycle 3: the reset's own per-cell blend already ran during cycle 2's touches (this SAME
    // call, since chunk_size=1 covers the single cell and completes the cycle in one call) --
    // col_importance[0] now reflects (1-strength)*100, a large negative delta relative to
    // cycle 2's snapshot. Without the fix, this negative delta corrupts col_grad_slow via the
    // catchup branch; with the fix, col_grad_var absorbs it as an EXPECTED (self-induced)
    // swing.
    CHECK(st.col_grad_var[0] > 0.0f,
          "col_grad_var should have been inflated by the reset's own self-induced perturbation, "
          "got %f",
          st.col_grad_var[0]);
    (void)importance_before_decay;
}

static void test_deviation_distribution_min_max_tracked() {
    // Distribution logging (direct instruction, after k=2.0 turned out to
    // guarantee the gate never opened across 3 real 100k-step runs): a
    // future relaunch needs min/max (not just a single "worst" value) to
    // actually calibrate k from real data. 3 columns, distinct
    // importance so all 3 rank into a reset_fraction=1.0 selection, each
    // pre-seeded with a different col_grad_slow/fast (via two manual
    // cycles) so their z-scores differ -- min/max should bracket the
    // per-column deviation values, not just report one of them.
    auto conn = make_dense_conn(
        1, 3, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t c) { return static_cast<float>(c) + 1.0f; });
    PlasticityState st;
    PlasticityCellCursor cur;
    apply_amortized_plasticity_step(conn, 3, st, cur, 3, 0.0f, 0.99f, 0.95f, 0.5f, 0.0f, 1.0f,
                                    0.5f); // cycle 1: warm-start all 3 columns
    // Cycle 2: bump only column 2's importance sharply (a real deviation spike, vb=2 is that
    // column's sole cell since n_in=1). Columns 0 and 1 are left unchanged -- their delta stays
    // exactly 0 relative to their own warm-started slow, giving deviation=0 exactly.
    ValueAccessor<VT>::set_live(conn.values, 2, 1.0f,
                                30.0f); // col 2: importance 3 -> 30, real spike
    auto r2 = apply_amortized_plasticity_step(conn, 3, st, cur, 3, 0.0f, 0.99f, 0.95f, 0.5f, 0.0f,
                                              1.0f, 0.5f);
    CHECK(r2.n_reset_this_cycle == 3, "all 3 mature columns should be selected, got %zu",
          r2.n_reset_this_cycle);
    CHECK(r2.max_deviation > r2.min_deviation,
          "with a real spike on one column, max_deviation should exceed min_deviation "
          "(min=%f max=%f)",
          r2.min_deviation, r2.max_deviation);
    CHECK(r2.max_deviation >= r2.mean_deviation && r2.mean_deviation >= r2.min_deviation,
          "mean_deviation should sit between min and max (min=%f mean=%f max=%f)", r2.min_deviation,
          r2.mean_deviation, r2.max_deviation);
}

// L2-saturation-gated decay on col_importance (direct instruction, after
// offline replay of a real 100k-step run's per-column logs showed raw
// importance saturates at max_ci=100 for most of a saturated pool's
// population by late training, collapsing the frozen-pool's ranking
// signal to numerical noise -- see
// docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design's
// l2_saturation_decay section). Deliberately NOT gated on col_age or any
// other elapsed-time/step counter -- a counter-based decay exponent
// eventually overflows or silently changes meaning at large step counts,
// which this mechanism must never do (infinite-horizon target). The
// gate is instead a pure function of CURRENT state: the population's L2
// norm relative to the fully-saturated ceiling (||max_ci*ones(n)||_2),
// passed through a SOFT sigmoid ramp (direct instruction: "soft
// threshold if we can not hard") so it stays near-zero comfortably below
// threshold and only really engages once the population is genuinely
// saturated, with no discontinuity at the threshold itself.

static void test_l2_decay_default_is_noop() {
    // l2_decay_lambda defaults to 0.0 -- exact no-op, byte-identical to
    // pre-L2-decay behavior, even for a fully-saturated population
    // (feedback_new_shared_parameter_backward_compat).
    auto conn = make_dense_conn(
        1, 1, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t) { return 100.0f; }); // already at max_ci
    PlasticityState st;
    PlasticityCellCursor cur;
    apply_amortized_plasticity_step(conn, 1, st, cur, 1, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 0.0f,
                                    0.5f);
    // cycle boundary just ran; cycle 2 touch should leave importance untouched by L2 decay.
    apply_amortized_plasticity_step(conn, 1, st, cur, 1, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 0.0f,
                                    0.5f);
    const float imp = ValueAccessor<VT>::get_imp(conn.values, 0);
    CHECK(std::abs(imp - 100.0f) < 1e-4f,
          "default l2_decay_lambda=0 must not touch importance at all, got %f", imp);
}

static void test_l2_decay_near_zero_far_below_threshold() {
    // Population importance is tiny relative to max_ci=100 -- sat_ratio
    // should be ~0, and decay_strength should be near-zero ("not
    // activating much when it actually is below it", direct wording).
    auto conn = make_dense_conn(
        1, 4, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t) { return 0.01f; });
    PlasticityState st;
    PlasticityCellCursor cur;
    apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 0.0f, 0.5f,
                                    0.9f, /*l2_decay_lambda=*/1.0f);
    auto r2 = apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f,
                                              0.0f, 0.5f, 0.9f, /*l2_decay_lambda=*/1.0f);
    CHECK(r2.l2_sat_ratio < 0.01, "sat_ratio should be ~0 for a near-empty population, got %f",
          r2.l2_sat_ratio);
    CHECK(r2.l2_decay_strength < 0.01,
          "decay_strength should be near-zero far below threshold, got %f", r2.l2_decay_strength);
}

static void test_l2_decay_strong_when_fully_saturated() {
    // Every column pinned at max_ci=100 -- sat_ratio should be ~1.0, and
    // with lambda=1.0 decay should visibly shrink importance on the very
    // next touch (population-level signal, applies even to a column that
    // is NOT in the frozen pool this cycle).
    auto conn = make_dense_conn(
        1, 4, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t) { return 100.0f; });
    PlasticityState st;
    PlasticityCellCursor cur;
    apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 0.0f, 0.5f,
                                    0.9f, /*l2_decay_lambda=*/1.0f);
    auto r2 = apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f,
                                              0.0f, 0.5f, 0.9f, /*l2_decay_lambda=*/1.0f);
    CHECK(r2.l2_sat_ratio > 0.99,
          "sat_ratio should be ~1.0 for a fully-saturated population, got %f", r2.l2_sat_ratio);
    CHECK(r2.l2_decay_strength > 0.5,
          "decay_strength should be substantial once fully saturated, got %f",
          r2.l2_decay_strength);
    // cycle 3: touch cells again, importance should have visibly shrunk from 100.
    apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 0.0f, 0.5f,
                                    0.9f, /*l2_decay_lambda=*/1.0f);
    const float imp_col0 = ValueAccessor<VT>::get_imp(conn.values, 0);
    CHECK(imp_col0 < 99.0f,
          "importance should visibly shrink once population is saturated and lambda>0, got %f",
          imp_col0);
}

static void test_l2_decay_touches_every_cell_not_just_reset_pool() {
    // 4 columns, reset_fraction small enough that only 1 column lands in
    // the frozen pool -- L2 decay must still shrink the OTHER columns'
    // importance too (it's a population-level gate, not scoped to the
    // frozen-pool selection), while leaving their WEIGHT untouched
    // (L2 decay only ever acts on importance, never weight).
    auto conn = make_dense_conn(
        1, 4, [](std::size_t, std::size_t) { return 5.0f; },
        [](std::size_t, std::size_t c) { return 100.0f - static_cast<float>(c); });
    PlasticityState st;
    PlasticityCellCursor cur;
    apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 0.25f,
                                    0.5f, 0.9f, /*l2_decay_lambda=*/1.0f);
    apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 0.25f,
                                    0.5f, 0.9f, /*l2_decay_lambda=*/1.0f);
    CHECK(st.col_reset_active[3] == 0,
          "col 3 (lowest importance) should not be in the frozen pool");
    apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 0.25f,
                                    0.5f, 0.9f, /*l2_decay_lambda=*/1.0f);
    const float imp_col3 = ValueAccessor<VT>::get_imp(conn.values, 3);
    const float w_col3 = ValueAccessor<VT>::get_w(conn.values, 3);
    CHECK(imp_col3 < 97.0f,
          "a non-frozen-pool column should still be shrunk by the population-level L2 decay, "
          "got %f",
          imp_col3);
    CHECK(std::abs(w_col3 - 5.0f) < 1e-5f,
          "L2 decay must never touch weight, only importance, got w=%f", w_col3);
}

static void test_l2_decay_ramp_is_smooth_not_a_hard_step() {
    // Direct instruction: "soft threshold if we can not hard". Build 3
    // populations at increasing average importance (well below, near,
    // and at the threshold*max_ci scale) and confirm decay_strength
    // increases smoothly/monotonically rather than jumping straight
    // from ~0 to ~1 at the threshold.
    auto make_and_measure = [](float imp_value) {
        auto conn = make_dense_conn(
            1, 4, [](std::size_t, std::size_t) { return 1.0f; },
            [imp_value](std::size_t, std::size_t) { return imp_value; });
        PlasticityState st;
        PlasticityCellCursor cur;
        apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f, 0.0f,
                                        0.5f, 0.9f, 1.0f);
        auto r2 = apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f,
                                                  0.5f, 0.0f, 0.5f, 0.9f, 1.0f);
        return r2.l2_decay_strength;
    };
    const double d_low = make_and_measure(60.0f);   // sat_ratio 0.6, well below threshold 0.9
    const double d_mid = make_and_measure(88.0f);   // sat_ratio 0.88, just below threshold
    const double d_high = make_and_measure(92.0f);  // sat_ratio 0.92, just above threshold
    const double d_full = make_and_measure(100.0f); // sat_ratio 1.0, fully saturated
    CHECK(d_low < d_mid && d_mid < d_high && d_high < d_full,
          "decay_strength should increase monotonically with saturation, got %f, %f, %f, %f", d_low,
          d_mid, d_high, d_full);
    // No single step should be a near-total jump from ~0 to ~1 -- each
    // consecutive gap should be a fraction of the total range, not the
    // whole range at once (the defining property of a SOFT threshold).
    CHECK(d_mid - d_low < 0.5, "low->mid step should be gradual, got jump of %f", d_mid - d_low);
    CHECK(d_high - d_mid < 0.5, "mid->high step should be gradual, got jump of %f", d_high - d_mid);
}

// select_by_deviation: growth-RATE-based selection instead of
// absolute-LEVEL selection -- direct instruction, after comparing a
// graduated real run against a stuck real run's collected column data:
// the stuck run's importance was already growing 17-147x faster than
// the graduated run's from early in training (steps ~9000-18000),
// well before EITHER run's importance reached a high absolute level --
// but the existing top-K-by-importance selection can never catch a
// column in that state, since a fast-accelerating-but-still-low column
// never ranks in the top-K by raw level until it's already deep into
// the pathological regime. col_grad_fast/slow/var (and thus deviation)
// are already tracked for EVERY mature column each cycle (not just the
// ones selected by importance) -- this just changes what SELECTS the
// top-K, not the reset/blend action itself. See
// docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design.select_by_deviation_early_detection.

static void test_select_by_deviation_off_matches_existing_importance_selection() {
    // Regression: default (select_by_deviation=false) must reproduce
    // the exact existing top-K-by-importance behavior.
    auto conn = make_dense_conn(
        1, 4, [](std::size_t, std::size_t) { return 1.0f; },
        [](std::size_t, std::size_t c) { return static_cast<float>(c); });
    PlasticityState st;
    PlasticityCellCursor cur;
    apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.25f,
                                    0.5f);
    auto r2 = apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f,
                                              0.25f, 0.5f);
    CHECK(r2.cycle_complete, "cycle 2 should complete");
    CHECK(st.col_reset_active[3] == 1, "default mode should still pick col 3 (highest importance)");
    CHECK(r2.n_reset_this_cycle == 1, "exactly 1 column should be selected, got %zu",
          r2.n_reset_this_cycle);
}

// 2-column setup: col 0 has HIGH absolute importance but a STABLE,
// unremarkable per-cycle delta (low deviation); col 1 has LOW absolute
// importance but a SUDDEN spike on the final cycle (high deviation).
// reset_fraction=0.0 during warm-up (top_n=0, nothing selected yet,
// but col_grad_slow/fast/var still update normally every cycle) avoids
// any earlier selection perturbing the controlled construction; the
// final cycle uses reset_fraction=0.5 (top_n=1) to force exactly one
// pick, and is run TWICE on freshly-warmed-up state -- once per mode.
static void run_warmup_cycles(Conn& conn, PlasticityState& st, PlasticityCellCursor& cur) {
    ValueAccessor<VT>::set_live(conn.values, 0, 1.0f, 50.0f); // col 0
    ValueAccessor<VT>::set_live(conn.values, 1, 1.0f, 1.0f);  // col 1
    apply_amortized_plasticity_step(conn, 2, st, cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.0f,
                                    0.5f);
    ValueAccessor<VT>::set_live(conn.values, 0, 1.0f, 51.0f); // delta +1
    ValueAccessor<VT>::set_live(conn.values, 1, 1.0f, 1.0f);  // delta 0
    apply_amortized_plasticity_step(conn, 2, st, cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.0f,
                                    0.5f);
    ValueAccessor<VT>::set_live(conn.values, 0, 1.0f, 52.0f); // delta +1
    ValueAccessor<VT>::set_live(conn.values, 1, 1.0f, 1.0f);  // delta 0
    apply_amortized_plasticity_step(conn, 2, st, cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.0f,
                                    0.5f);
}

static void test_select_by_deviation_picks_growth_rate_not_absolute_level() {
    // default mode (select_by_deviation=false): must pick col 0, the
    // higher-IMPORTANCE column (53 > 21), ignoring col 1's spike.
    {
        auto conn = make_dense_conn(
            1, 2, [](std::size_t, std::size_t) { return 1.0f; },
            [](std::size_t, std::size_t) { return 0.0f; });
        PlasticityState st;
        PlasticityCellCursor cur;
        run_warmup_cycles(conn, st, cur);
        ValueAccessor<VT>::set_live(conn.values, 0, 1.0f, 53.0f); // delta +1, unremarkable
        ValueAccessor<VT>::set_live(conn.values, 1, 1.0f, 21.0f); // delta +20, a real spike
        auto r = apply_amortized_plasticity_step(conn, 2, st, cur, 2, 0.0f, 0.99f, 0.95f, 0.5f,
                                                 0.1f, 0.5f, 0.5f, 0.9f, 0.0f, 0.9f, 0.05f, 100.0f,
                                                 /*select_by_deviation=*/false);
        CHECK(r.cycle_complete, "cycle should complete");
        CHECK(st.col_reset_active[0] == 1 && st.col_reset_active[1] == 0,
              "default mode should pick col 0 (importance 53>21), got col0=%d col1=%d",
              st.col_reset_active[0], st.col_reset_active[1]);
    }
    // select_by_deviation=true: must pick col 1, the SPIKING column,
    // even though its absolute importance (21) is far below col 0's (53).
    {
        auto conn = make_dense_conn(
            1, 2, [](std::size_t, std::size_t) { return 1.0f; },
            [](std::size_t, std::size_t) { return 0.0f; });
        PlasticityState st;
        PlasticityCellCursor cur;
        run_warmup_cycles(conn, st, cur);
        ValueAccessor<VT>::set_live(conn.values, 0, 1.0f, 53.0f);
        ValueAccessor<VT>::set_live(conn.values, 1, 1.0f, 21.0f);
        auto r = apply_amortized_plasticity_step(conn, 2, st, cur, 2, 0.0f, 0.99f, 0.95f, 0.5f,
                                                 0.1f, 0.5f, 0.5f, 0.9f, 0.0f, 0.9f, 0.05f, 100.0f,
                                                 /*select_by_deviation=*/true);
        CHECK(r.cycle_complete, "cycle should complete");
        CHECK(st.col_reset_active[1] == 1 && st.col_reset_active[0] == 0,
              "select_by_deviation=true should pick col 1 (the spike), got col0=%d col1=%d",
              st.col_reset_active[0], st.col_reset_active[1]);
    }
}

// select_by_deviation's gate: EVT-derived k tried first, then found by
// direct log analysis of a real 100k-step run (v6) to be a near-total
// no-op -- gate_open_frac was EXACTLY 0.000 in every one of 6 pools,
// across the ENTIRE run (both the early climbing phase and the later
// plateau). Real per-column deviation never approached the theoretical
// sqrt(2*ln(N))~=3.1-3.4 threshold (observed max ~1.0-1.8 throughout),
// meaning the EMA-lag structure of col_grad_fast/slow doesn't actually
// produce iid-standard-normal statistics the way the EVT derivation
// assumed -- the fix that stopped v5's over-triggering over-corrected
// into permanent inertness. Replaced with a threshold derived from the
// population's OWN empirical distribution instead of a theoretical
// asymptotic: k = the linear-interpolated percentile of the mature
// population's ACTUAL deviation values this cycle, at percentile
// 100*(1-reset_fraction) -- still equation-derived (an order statistic
// of real data, no new guessed constant), reusing the EXISTING
// reset_fraction knob rather than introducing one, and self-calibrating
// to whatever the real spread happens to be instead of assuming
// normality. Validated against v6's own real log data before
// implementing: this percentile rule would have opened the gate
// 47-100% of the time in the late-plateau phase across all 6 pools,
// vs 0% for the EVT rule and 97.25% for the original fixed k=1.0. See
// docs/research/toy_tile_recurrence_rmt.rst:
// plasticity_reset_design.select_by_deviation_early_detection.k_derivation.
static void test_select_by_deviation_gate_uses_population_percentile_not_passed_k() {
    // 4 mature columns (N=4), reset_fraction=0.25 -> top_n=1,
    // percentile = 100*(1-0.25) = 75.
    auto make_and_warm = [](PlasticityState& st, PlasticityCellCursor& cur) {
        auto conn = make_dense_conn(
            1, 4, [](std::size_t, std::size_t) { return 1.0f; },
            [](std::size_t, std::size_t) { return 10.0f; });
        for (int cycle = 0; cycle < 3; ++cycle)
            apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f,
                                            0.0f, 0.5f);
        return conn;
    };

    auto compute_deviation = [](const PlasticityState& st, std::size_t j) {
        const float std_dev = std::sqrt(std::max(0.0f, st.col_grad_var[j]));
        return (st.col_grad_fast[j] - st.col_grad_slow[j]) / (std_dev + 1e-8f);
    };
    // Same linear-interpolation percentile numpy's default uses --
    // must match what was validated against the real v6 log data.
    auto percentile = [](std::vector<float> v, double p) {
        std::sort(v.begin(), v.end());
        const double idx = (static_cast<double>(v.size()) - 1.0) * (p / 100.0);
        const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
        const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
        const double frac = idx - static_cast<double>(lo);
        return static_cast<float>(v[lo] + frac * (v[hi] - v[lo]));
    };

    float boost_low_k = 0.0f, boost_high_k = 0.0f;
    std::vector<float> devs;
    {
        PlasticityState st;
        PlasticityCellCursor cur;
        auto conn = make_and_warm(st, cur);
        ValueAccessor<VT>::set_live(conn.values, 3, 1.0f, 60.0f); // col 3: delta +50, a real spike
        auto r = apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f,
                                                 0.1f, 0.25f, /*k=*/0.0f, 0.9f, 0.0f, 0.9f, 0.05f,
                                                 100.0f, /*select_by_deviation=*/true);
        CHECK(r.cycle_complete, "cycle should complete");
        CHECK(st.col_reset_active[3] == 1, "col 3 (the spike) should be selected, got %d",
              st.col_reset_active[3]);
        for (std::size_t j = 0; j < 4; ++j)
            devs.push_back(compute_deviation(st, j));
        boost_low_k = st.col_plasticity_boost[3];
    }
    {
        PlasticityState st;
        PlasticityCellCursor cur;
        auto conn = make_and_warm(st, cur);
        ValueAccessor<VT>::set_live(conn.values, 3, 1.0f, 60.0f);
        apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.25f,
                                        /*k=*/1000.0f, 0.9f, 0.0f, 0.9f, 0.05f, 100.0f,
                                        /*select_by_deviation=*/true);
        boost_high_k = st.col_plasticity_boost[3];
    }
    CHECK(std::abs(boost_low_k - boost_high_k) < 1e-5f,
          "passed-in k must be IGNORED under select_by_deviation=true (gate uses the "
          "population's own percentile instead) -- got %f (k=0) vs %f (k=1000)",
          boost_low_k, boost_high_k);
    const float k_expected = percentile(devs, 75.0);
    const float expected_boost = std::max(0.0f, devs[3] - k_expected);
    CHECK(std::abs(boost_low_k - expected_boost) < 1e-3f,
          "plasticity_boost should equal max(0, deviation - percentile(mature_devs, 75)) = "
          "%f, got %f",
          expected_boost, boost_low_k);
}

static void test_default_mode_gate_still_uses_passed_k() {
    // Regression: select_by_deviation=false must still use the PASSED k
    // unchanged -- only the deviation-selection mode's gate is replaced
    // by the EVT-derived threshold. Same 4-column fixture, but ranked
    // (and gated) by importance, not deviation -- col 3 (imp=60) still
    // wins on importance alone.
    auto make_and_warm = [](PlasticityState& st, PlasticityCellCursor& cur) {
        auto conn = make_dense_conn(
            1, 4, [](std::size_t, std::size_t) { return 1.0f; },
            [](std::size_t, std::size_t) { return 10.0f; });
        for (int cycle = 0; cycle < 3; ++cycle)
            apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f,
                                            0.0f, 0.5f);
        return conn;
    };
    float boost_k0 = 0.0f, boost_k_big = 0.0f;
    {
        PlasticityState st;
        PlasticityCellCursor cur;
        auto conn = make_and_warm(st, cur);
        ValueAccessor<VT>::set_live(conn.values, 3, 1.0f, 60.0f);
        apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.25f,
                                        /*k=*/0.0f, 0.9f, 0.0f, 0.9f, 0.05f, 100.0f,
                                        /*select_by_deviation=*/false);
        boost_k0 = st.col_plasticity_boost[3];
    }
    {
        PlasticityState st;
        PlasticityCellCursor cur;
        auto conn = make_and_warm(st, cur);
        ValueAccessor<VT>::set_live(conn.values, 3, 1.0f, 60.0f);
        apply_amortized_plasticity_step(conn, 4, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.25f,
                                        /*k=*/1000.0f, 0.9f, 0.0f, 0.9f, 0.05f, 100.0f,
                                        /*select_by_deviation=*/false);
        boost_k_big = st.col_plasticity_boost[3];
    }
    CHECK(boost_k0 > boost_k_big + 500.0f || (boost_k0 > 0.0f && boost_k_big == 0.0f),
          "default mode must still use the PASSED k (k=0 -> boost=%f, k=1000 -> boost=%f "
          "should have collapsed to 0)",
          boost_k0, boost_k_big);
}

int main() {
    test_col_importance_accumulates_from_known_values();
    test_frozen_pool_selects_highest_importance_mature_columns();
    test_frozen_blend_is_gated_by_plasticity_boost();
    test_maturity_gate_excludes_just_reset_column();
    test_empty_layer();
    test_asymmetric_catchup_rate();
    test_cold_start_no_spurious_deviation();
    test_reset_inflates_variance_preventing_immediate_reflag();
    test_deviation_distribution_min_max_tracked();
    test_l2_decay_default_is_noop();
    test_l2_decay_near_zero_far_below_threshold();
    test_l2_decay_strong_when_fully_saturated();
    test_l2_decay_touches_every_cell_not_just_reset_pool();
    test_l2_decay_ramp_is_smooth_not_a_hard_step();
    test_select_by_deviation_off_matches_existing_importance_selection();
    test_select_by_deviation_picks_growth_rate_not_absolute_level();
    test_select_by_deviation_gate_uses_population_percentile_not_passed_k();
    test_default_mode_gate_still_uses_passed_k();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
