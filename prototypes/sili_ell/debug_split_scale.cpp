#include "../../sili/lib/headers/dense_block4.hpp"
#include <cstdio>
#include <random>

int main() {
    // Reproduce gate_proj's real scale: n_out=4608, n_in=1536, density~0.78
    const uint32_t n_out = 4608, n_in = 1536;
    const float density = 0.78f;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u01(0, 1), wv(-6, 6);

    std::vector<uint32_t> ptrs(n_out + 1, 0);
    std::vector<uint32_t> idx;
    std::vector<float> w, imp;
    idx.reserve(std::size_t(n_out) * n_in * density);
    for (uint32_t r = 0; r < n_out; ++r) {
        for (uint32_t c = 0; c < n_in; ++c) {
            if (u01(rng) < density) {
                idx.push_back(c);
                w.push_back(wv(rng));
                imp.push_back(0.5f);
            }
        }
        ptrs[r + 1] = uint32_t(idx.size());
    }
    printf("built CSR: nnz=%zu (expected ~%.0f)\n", idx.size(), double(n_out) * n_in * density);

    auto split = split_for_block4(n_out, n_in, ptrs, idx, w, imp, 0.10f);
    std::size_t b_nnz = 0;
    for (uint8_t b : split.block4.block_data) b_nnz += ((b & 0xFu) != 0);
    printf("block4 capacity slots: %zu\n", split.block4.block_data.size());
    printf("block4 live: %zu, leftover: %zu, sum: %zu, expected: %zu\n",
           b_nnz, split.leftover_rc.size(), b_nnz + split.leftover_rc.size(), idx.size());
    printf("n_blocks total: %u (Mb=%u Nb=%u Mb*Nb=%u)\n",
           split.block4.n_blocks(), split.block4.n_block_rows, split.block4.n_block_cols,
           split.block4.n_block_rows * split.block4.n_block_cols);
    return 0;
}
