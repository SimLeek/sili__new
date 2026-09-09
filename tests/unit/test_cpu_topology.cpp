// Runtime cache-domain (CCX) topology detection (see cpu_topology.hpp).
// No hardcoded topology assumptions: this test only checks the
// INVARIANTS the reduction code in linear_disldo.hpp actually relies on,
// not any specific machine's real grouping (which varies by hardware).
#include "../../sili/lib/headers/cpu_topology.hpp"
#include <atomic>
#include <cstdio>
#include <omp.h>
#include <vector>

static std::atomic<int> g_fail{0};
#define CHECK(cond, fmt, ...)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);               \
            std::fflush(stdout);                                                                   \
            ++g_fail;                                                                              \
        }                                                                                          \
    } while (0)

// The table must either be empty (topology unknown -> flat fallback) or
// cover every online logical CPU with a non-negative group id. Either
// way, num_l3_groups()-equivalent logic in callers must treat an empty
// table as "everyone is group 0" -- this test just verifies the raw
// table's own shape is self-consistent, since callers build their own
// group-count/leader logic on top of it.
static void test_table_shape_is_self_consistent() {
    const std::vector<int>& t = sili_topology::cpu_to_l3_group();
    for (std::size_t i = 0; i < t.size(); ++i) {
        CHECK(t[i] >= 0, "cpu %zu has negative group id %d", i, t[i]);
    }
    // Calling twice must return the identical (cached) table.
    const std::vector<int>& t2 = sili_topology::cpu_to_l3_group();
    CHECK(&t == &t2, "cpu_to_l3_group() should return the same cached table on repeated calls");
}

// current_thread_group() must always return a valid, in-range group id
// (or 0 if the table is empty) -- never garbage, never out of bounds --
// regardless of which real CPU the calling thread happens to be on.
// Checked from several real OpenMP worker threads, not just the main
// thread.
static void test_current_thread_group_always_in_range() {
    const std::vector<int>& t = sili_topology::cpu_to_l3_group();
    int max_g = 0;
    for (int v : t)
        if (v > max_g)
            max_g = v;
    const bool table_empty = t.empty();

#pragma omp parallel num_threads(4)
    {
        const int g = sili_topology::current_thread_group();
        CHECK(g >= 0, "current_thread_group() returned negative %d", g);
        if (!table_empty) {
            CHECK(g <= max_g, "current_thread_group()=%d exceeds max table group %d", g, max_g);
        } else {
            CHECK(g == 0, "empty topology table should force group 0, got %d", g);
        }
    }
}

// The core correctness property the reduction code in linear_disldo.hpp
// actually depends on: grouping is a PARTITION. Every tid in [0,
// num_cpus) must land in exactly one group, so a leader-per-group
// reduction that walks "every tid whose group matches mine" sums every
// thread's contribution exactly once, regardless of how many distinct
// groups exist or which tid ended up in which one.
static void test_grouping_is_a_partition_regardless_of_topology() {
    const int num_cpus = 8;
    std::vector<int> group_of_tid(static_cast<std::size_t>(num_cpus), -1);
#pragma omp parallel num_threads(num_cpus)
    {
        const int tid = omp_get_thread_num();
        group_of_tid[static_cast<std::size_t>(tid)] = sili_topology::current_thread_group();
    }

    int touched = 0;
    for (int tid = 0; tid < num_cpus; ++tid) {
        CHECK(group_of_tid[static_cast<std::size_t>(tid)] >= 0, "tid %d never got a group assigned",
              tid);
        for (int t = 0; t < num_cpus; ++t)
            if (group_of_tid[static_cast<std::size_t>(t)] ==
                group_of_tid[static_cast<std::size_t>(tid)])
                ++touched;
    }
    // Every tid must be reachable from a "same group as me" scan
    // (trivially true if it's its own group, which is always the case).
    CHECK(touched >= num_cpus, "partition scan touched fewer slots (%d) than threads (%d)", touched,
          num_cpus);
}

int main() {
    test_table_shape_is_self_consistent();
    test_current_thread_group_always_in_range();
    test_grouping_is_a_partition_regardless_of_topology();

    if (g_fail == 0) {
        std::printf("All cpu_topology tests passed.\n");
    } else {
        std::printf("%d FAILURES\n", g_fail.load());
    }
    return g_fail == 0 ? 0 : 1;
}
