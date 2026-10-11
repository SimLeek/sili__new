// apply_amortized_block4_l2_init (block4_l2_init_TODO_DELETE.hpp) --
// block4 mirror of test_amortized_l2_init.cpp's scattered coverage
// (see that file's header comment for the full L2 Init derivation and
// citation).
#include "../../sili/lib/headers/block4_l2_init_TODO_DELETE.hpp"
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

struct Block4L2InitTestState {
    std::vector<float> initial_weight;
    std::vector<uint8_t> captured;
};

static void test_first_touch_only_captures_no_modification() {
    Block4Store32 store;
    store.init(4, 4); // single 4x4 tile
    store.switch_point = 0;
    {
        auto h = store.get_or_create(0, 0);
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj)
                h.set_weight(li, lj, static_cast<float>(li * BLOCK4_TILE + lj + 1));
    }

    Block4L2InitTestState st;
    Block4L2InitCursor cur;
    auto r =
        apply_amortized_block4_l2_init<float>(store, cur, st.initial_weight, st.captured, 1, 0.5f);
    CHECK(r.cycle_complete, "single tile, chunk_size=1 should complete the cycle");
    CHECK(r.n_touched == 16, "should touch all 16 cells of the tile, got %zu", r.n_touched);
    auto h2 = store.get_or_create(0, 0);
    for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
            const float expected = static_cast<float>(li * BLOCK4_TILE + lj + 1);
            CHECK(std::abs(h2.get_weight(li, lj) - expected) < 1e-6f,
                  "first touch must not modify weight at (%u,%u)", li, lj);
        }
}

static void test_second_touch_pulls_toward_captured_initial() {
    Block4Store32 store;
    store.init(4, 4);
    store.switch_point = 0;
    {
        auto h = store.get_or_create(0, 0);
        for (uint32_t li = 0; li < BLOCK4_TILE; ++li)
            for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj)
                h.set_weight(li, lj, 1.0f);
    }
    Block4L2InitTestState st;
    Block4L2InitCursor cur;
    apply_amortized_block4_l2_init<float>(store, cur, st.initial_weight, st.captured, 1, 0.5f);

    {
        auto h = store.get_or_create(0, 0);
        h.set_weight(0, 0, 5.0f); // simulate real training moving cell (0,0) away from init
    }
    apply_amortized_block4_l2_init<float>(store, cur, st.initial_weight, st.captured, 1, 0.5f);

    auto h = store.get_or_create(0, 0);
    // new_w = w + rate*(initial - w) = 5 + 0.5*(1-5) = 3.0
    CHECK(std::abs(h.get_weight(0, 0) - 3.0f) < 1e-6f,
          "second touch should pull (0,0) halfway back to initial (1.0) from 5.0 -> 3.0, got %f",
          h.get_weight(0, 0));
    // an untouched-by-us cell (unchanged since capture) should be pulled toward ITSELF (no-op).
    CHECK(std::abs(h.get_weight(1, 1) - 1.0f) < 1e-6f,
          "a cell still at its initial value should stay there, got %f", h.get_weight(1, 1));
}

static void test_empty_store_completes_immediately() {
    Block4Store32 store;
    store.init(4, 4);
    Block4L2InitTestState st;
    Block4L2InitCursor cur;
    auto r =
        apply_amortized_block4_l2_init<float>(store, cur, st.initial_weight, st.captured, 4, 0.5f);
    CHECK(r.cycle_complete, "empty store should report cycle_complete=true immediately");
    CHECK(r.n_touched == 0, "empty store should touch nothing, got %zu", r.n_touched);
}

int main() {
    test_first_touch_only_captures_no_modification();
    test_second_touch_pulls_toward_captured_initial();
    test_empty_store_completes_immediately();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
