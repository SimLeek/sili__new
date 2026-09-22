// TDD (written before the implementation exists): apply_amortized_block4_plasticity_step
// (block4_plasticity_TODO_DELETE.hpp) -- the block4 half of the
// per-neuron utility-based plasticity reset. Mirrors
// test_amortized_plasticity_reset.cpp's scattered coverage exactly
// (same design, same several rounds of correction, including the dead
// pool's later removal -- see that file's own header comment and
// docs/research/toy_tile_recurrence_rmt.rst:plasticity_reset_design),
// plus a bit-equivalence check against the scattered path for the same
// logical values (col_importance accumulation is a per-cell EMA with no
// cross-cell reduction, so the two storage formats must agree exactly,
// matching this project's established bit-equivalence testing
// convention for every scattered/block4 pair).
#include "../../sili/lib/headers/block4_plasticity_TODO_DELETE.hpp"
#include <cmath>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

static void test_col_importance_accumulates_from_known_values() {
    // 8x8 (2x2 tiles), all weight=0.1, importance varies by ABSOLUTE column.
    Block4Store32 store;
    store.init(8, 8);
    store.switch_point = 0; // force dense tiles, simplest case
    for (uint32_t br = 0; br < 2; ++br) {
        for (uint32_t bc = 0; bc < 2; ++bc) {
            auto h = store.get_or_create(br, bc);
            for (uint32_t li = 0; li < 4; ++li) {
                for (uint32_t lj = 0; lj < 4; ++lj) {
                    const uint32_t col = bc * 4 + lj;
                    h.set_weight(li, lj, 0.1f);
                    h.set_importance(li, lj, static_cast<float>(col) + 1.0f);
                }
            }
        }
    }
    PlasticityState st;
    Block4PlasticityCursor cur;
    auto r = apply_amortized_block4_plasticity_step(
        store, 8, st, cur, /*chunk_size=*/4, /*eta=*/0.0f, 0.99f, 0.95f, 0.5f, 0.1f, 0.0f, 0.5f);
    CHECK(r.cycle_complete, "4 tiles, chunk_size=4 should complete in one call");
    for (std::size_t j = 0; j < 8; ++j)
        CHECK(std::abs(st.col_importance[j] - (static_cast<float>(j) + 1.0f)) < 1e-6f,
              "col %zu importance should be %f, got %f", j, static_cast<float>(j) + 1.0f,
              st.col_importance[j]);
}

static void test_frozen_pool_selects_highest_importance_mature_columns() {
    Block4Store32 store;
    store.init(8, 8);
    store.switch_point = 0;
    for (uint32_t br = 0; br < 2; ++br) {
        for (uint32_t bc = 0; bc < 2; ++bc) {
            auto h = store.get_or_create(br, bc);
            for (uint32_t li = 0; li < 4; ++li) {
                for (uint32_t lj = 0; lj < 4; ++lj) {
                    const uint32_t col = bc * 4 + lj;
                    h.set_weight(li, lj, 1.0f);
                    h.set_importance(li, lj, static_cast<float>(col)); // col 7 highest
                }
            }
        }
    }
    PlasticityState st;
    Block4PlasticityCursor cur;
    apply_amortized_block4_plasticity_step(store, 8, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f,
                                           /*reset_fraction=*/0.125f, 0.5f);
    CHECK(st.col_reset_active[7] == 0, "cycle 1: nothing mature yet");
    auto r2 = apply_amortized_block4_plasticity_step(store, 8, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f,
                                                     0.1f, 0.125f, 0.5f);
    CHECK(r2.cycle_complete, "cycle 2 should complete");
    CHECK(st.col_reset_active[7] == 1, "col 7 (highest importance) should be the frozen pick");
    CHECK(r2.n_reset_this_cycle == 1, "exactly 1 column should be selected, got %zu",
          r2.n_reset_this_cycle);
}

static void test_frozen_blend_gated_and_unflagged_untouched() {
    Block4Store32 store;
    store.init(4, 8); // 1x2 tiles
    store.switch_point = 0;
    {
        auto h = store.get_or_create(0, 0);
        for (uint32_t li = 0; li < 4; ++li)
            for (uint32_t lj = 0; lj < 4; ++lj) {
                h.set_weight(li, lj, 1.0f);
                h.set_importance(li, lj, 1.0f);
            }
    }
    {
        auto h = store.get_or_create(0, 1);
        for (uint32_t li = 0; li < 4; ++li)
            for (uint32_t lj = 0; lj < 4; ++lj) {
                h.set_weight(li, lj, 1.0f);
                h.set_importance(li, lj, 1.0f);
            }
    }
    PlasticityState st;
    Block4PlasticityCursor cur;
    st.ensure_sized(8);
    st.col_reset_active[0] = 1; // column 0 (tile bc=0, lj=0): gated
    st.col_plasticity_boost[0] = 0.5f;
    // column 1 (bc=0, lj=1): unflagged -- must stay byte-identical.

    apply_amortized_block4_plasticity_step(store, 8, st, cur, 2, 0.0f, 0.99f, 0.95f, 0.5f,
                                           /*blend=*/0.2f, 0.0f, 0.5f);
    auto h0 = store.find(0, 0);
    // Deterministic component (importance has no RNG term): gated strength = 0.2*0.5=0.1.
    CHECK(std::abs(h0.get_importance(0, 0) - 0.9f) < 1e-5f,
          "gated col 0 importance should shrink by (1-0.1)=0.9, got %f", h0.get_importance(0, 0));
    CHECK(h0.get_weight(0, 1) == 1.0f && h0.get_importance(0, 1) == 1.0f,
          "untouched column (not selected) must be byte-identical");
}

static void test_maturity_gate_excludes_just_reset_column() {
    Block4Store32 store;
    store.init(4, 8);
    store.switch_point = 0;
    for (uint32_t bc = 0; bc < 2; ++bc) {
        auto h = store.get_or_create(0, bc);
        for (uint32_t li = 0; li < 4; ++li)
            for (uint32_t lj = 0; lj < 4; ++lj) {
                const uint32_t col = bc * 4 + lj;
                h.set_weight(li, lj, 1.0f);
                h.set_importance(li, lj, col == 0 ? 10.0f : 1.0f);
            }
    }
    PlasticityState st;
    Block4PlasticityCursor cur;
    apply_amortized_block4_plasticity_step(store, 8, st, cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f,
                                           0.125f, 0.5f);
    apply_amortized_block4_plasticity_step(store, 8, st, cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f,
                                           0.125f, 0.5f);
    CHECK(st.col_age[0] == 0, "col 0 should have just been reset, age=0");
    apply_amortized_block4_plasticity_step(store, 8, st, cur, 2, 0.0f, 0.99f, 0.95f, 0.5f, 0.1f,
                                           0.125f, 0.5f);
    CHECK(st.col_reset_active[0] == 0, "just-reset column must not be immediately re-eligible");
}

static void test_empty_store() {
    Block4Store32 store;
    store.init(8, 8);
    PlasticityState st;
    Block4PlasticityCursor cur;
    auto r = apply_amortized_block4_plasticity_step(store, 8, st, cur, 4, 0.99f, 0.99f, 0.95f, 0.5f,
                                                    0.1f, 0.01f, 0.5f);
    CHECK(r.cycle_complete, "empty store should report cycle_complete=true immediately");
    CHECK(r.n_reset_this_cycle == 0, "empty store should reset nothing");
}

static void test_bit_equivalence_vs_scattered() {
    // Same logical (row,col)->(weight,importance) values via block4 vs
    // scattered -- col_importance accumulation must match bit-for-bit
    // (per-cell EMA, no cross-cell reduction -- see the scattered test
    // file's own bit-equivalence precedent).
    Block4Store32 store;
    store.init(4, 4); // 1x1 tile
    store.switch_point = 0;
    const float weights_in[4] = {4.0f, -2.5f, 0.125f, 17.0f};
    const float imps_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    {
        auto h = store.get_or_create(0, 0);
        for (uint32_t lj = 0; lj < 4; ++lj) {
            h.set_weight(0, lj, weights_in[lj]);
            h.set_importance(0, lj, imps_in[lj]);
        }
    }
    PlasticityState st_block4;
    Block4PlasticityCursor cur_block4;
    apply_amortized_block4_plasticity_step(store, 4, st_block4, cur_block4, 1, 0.7f, 0.99f, 0.95f,
                                           0.5f, 0.1f, 0.0f, 0.5f);

    // Scattered equivalent: single row (r=0), 4 columns.
    using VT = DeltaCSRBiValues<float>;
    using Conn = DeltaCSRWeights<std::size_t, VT, uint32_t>;
    Conn conn;
    conn.layout.rows = 1;
    conn.layout.cols = 4;
    conn.layout.elem_start = {0, 4};
    conn.layout.elem_end = {4};
    conn.layout.byte_start = {0, 0};
    conn.layout.byte_end = {0};
    conn.layout.total_nnz = 4;
    conn.indices_buf = {0, 1, 1, 1}; // deltas: col0=0, then +1 each
    ValueAccessor<VT>::resize(conn.values, 4);
    for (std::size_t c = 0; c < 4; ++c)
        ValueAccessor<VT>::set_live(conn.values, c, weights_in[c], imps_in[c]);
    PlasticityState st_scattered;
    PlasticityCellCursor cur_scattered;
    apply_amortized_plasticity_step(conn, 4, st_scattered, cur_scattered, 4, 0.7f, 0.99f, 0.95f,
                                    0.5f, 0.1f, 0.0f, 0.5f);

    for (std::size_t j = 0; j < 4; ++j) {
        CHECK(st_block4.col_importance[j] == st_scattered.col_importance[j],
              "bit-equivalence failed on col_importance[%zu]: block4=%.9g scattered=%.9g", j,
              double(st_block4.col_importance[j]), double(st_scattered.col_importance[j]));
    }
}

// L2-saturation-gated decay -- block4 mirror of the scattered coverage
// in test_amortized_plasticity_reset.cpp (see that file's header
// comment for the full derivation). Same shared cycle-boundary function
// computes l2_sat_ratio/l2_decay_strength; only the per-cell application
// differs (block4's tile handle vs scattered's ValueAccessor).

static Block4Store32 make_uniform_store(float w_value, float imp_value) {
    Block4Store32 store;
    store.init(8, 8);
    store.switch_point = 0; // force dense tiles, simplest case
    for (uint32_t br = 0; br < 2; ++br) {
        for (uint32_t bc = 0; bc < 2; ++bc) {
            auto h = store.get_or_create(br, bc);
            for (uint32_t li = 0; li < 4; ++li) {
                for (uint32_t lj = 0; lj < 4; ++lj) {
                    h.set_weight(li, lj, w_value);
                    h.set_importance(li, lj, imp_value);
                }
            }
        }
    }
    return store;
}

static void test_l2_decay_default_is_noop_block4() {
    auto store = make_uniform_store(1.0f, 100.0f); // already at max_ci
    PlasticityState st;
    Block4PlasticityCursor cur;
    apply_amortized_block4_plasticity_step(store, 8, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f,
                                           0.0f, 0.5f);
    apply_amortized_block4_plasticity_step(store, 8, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f,
                                           0.0f, 0.5f);
    auto h = store.get_or_create(0, 0);
    const float imp = h.get_importance(0, 0);
    CHECK(std::abs(imp - 100.0f) < 1e-4f,
          "default l2_decay_lambda=0 must not touch block4 importance at all, got %f", imp);
}

static void test_l2_decay_strong_when_fully_saturated_block4() {
    auto store = make_uniform_store(5.0f, 100.0f);
    PlasticityState st;
    Block4PlasticityCursor cur;
    apply_amortized_block4_plasticity_step(store, 8, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f,
                                           0.0f, 0.5f, 0.9f, /*l2_decay_lambda=*/1.0f);
    auto r2 = apply_amortized_block4_plasticity_step(store, 8, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f,
                                                     0.5f, 0.0f, 0.5f, 0.9f,
                                                     /*l2_decay_lambda=*/1.0f);
    CHECK(r2.l2_sat_ratio > 0.99,
          "sat_ratio should be ~1.0 for a fully-saturated block4 population, got %f",
          r2.l2_sat_ratio);
    apply_amortized_block4_plasticity_step(store, 8, st, cur, 4, 0.0f, 0.99f, 0.95f, 0.5f, 0.5f,
                                           0.0f, 0.5f, 0.9f, /*l2_decay_lambda=*/1.0f);
    auto h = store.get_or_create(0, 0);
    const float imp = h.get_importance(0, 0);
    const float w = h.get_weight(0, 0);
    CHECK(imp < 99.0f,
          "block4 importance should visibly shrink once population is saturated and lambda>0, "
          "got %f",
          imp);
    CHECK(std::abs(w - 5.0f) < 1e-5f, "L2 decay must never touch block4 weight, got w=%f", w);
}

int main() {
    test_col_importance_accumulates_from_known_values();
    test_frozen_pool_selects_highest_importance_mature_columns();
    test_frozen_blend_gated_and_unflagged_untouched();
    test_maturity_gate_excludes_just_reset_column();
    test_empty_store();
    test_bit_equivalence_vs_scattered();
    test_l2_decay_default_is_noop_block4();
    test_l2_decay_strong_when_fully_saturated_block4();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
