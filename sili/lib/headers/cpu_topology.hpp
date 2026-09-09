#pragma once
// See disldo_backward.ccx_aware_reduction in docs/research/linear_disldo.rst.
#include <algorithm>
#include <cstddef>
#include <vector>

#if defined(__linux__)
#include <fstream>
#include <sched.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#define SILI_HAS_CPU_TOPOLOGY 1
#else
#define SILI_HAS_CPU_TOPOLOGY 0
#endif

namespace sili_topology {

// logical CPU id -> cache-group id. Empty means "topology unknown".
inline const std::vector<int>& cpu_to_l3_group() {
    static const std::vector<int> table = [] {
        std::vector<int> t;
#if SILI_HAS_CPU_TOPOLOGY
        const long n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n <= 0)
            return t;
        t.assign(static_cast<std::size_t>(n), 0);
        std::unordered_map<std::string, int> group_of_set;
        int next_group = 0;
        for (long cpu = 0; cpu < n; ++cpu) {
            const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                                     "/cache/index3/shared_cpu_list";
            std::ifstream f(path);
            if (!f.is_open()) {
                t.clear();
                return t;
            }
            std::string line;
            std::getline(f, line);
            const auto it = group_of_set.find(line);
            int g;
            if (it == group_of_set.end()) {
                g = next_group++;
                group_of_set.emplace(line, g);
            } else {
                g = it->second;
            }
            t[static_cast<std::size_t>(cpu)] = g;
        }
#endif
        return t;
    }();
    return table;
}

// Group id for the calling thread's current CPU (cheap vDSO call).
inline int current_thread_group() {
#if SILI_HAS_CPU_TOPOLOGY
    const std::vector<int>& t = cpu_to_l3_group();
    if (t.empty())
        return 0;
    const int cpu = sched_getcpu();
    if (cpu < 0 || static_cast<std::size_t>(cpu) >= t.size())
        return 0;
    return t[static_cast<std::size_t>(cpu)];
#else
    return 0;
#endif
}

} // namespace sili_topology
