#include "../../sili/lib/headers/dense_block4.hpp"
#include <cstdio>
#include <random>

int main() {
    std::mt19937 rng(0);
    const uint32_t n_in = 64, n_out = 32;
    std::vector<uint32_t> ptrs = {0};
    std::vector<uint32_t> idx;
    std::vector<float> w, imp;
    std::uniform_real_distribution<float> wv(-6, 6), impv(0, 1);
    for (uint32_t r = 0; r < n_in; ++r) {
        std::vector<uint32_t> cols;
        for (uint32_t c = 0; c < n_out; ++c) cols.push_back(c);
        std::shuffle(cols.begin(), cols.end(), rng);
        cols.resize(uint32_t(n_out * 0.3f));
        std::sort(cols.begin(), cols.end());
        for (uint32_t c : cols) { idx.push_back(c); w.push_back(wv(rng)); imp.push_back(impv(rng)); }
        ptrs.push_back(uint32_t(idx.size()));
    }
    printf("total input entries: %zu\n", idx.size());

    auto split = split_for_block4(n_in, n_out, ptrs, idx, w, imp, 0.10f);
    std::size_t b_nnz = 0;
    for (uint8_t b : split.block4.block_data) b_nnz += ((b & 0xFu) != 0);
    printf("block4 live: %zu, leftover: %zu, total accounted: %zu (should be %zu)\n",
           b_nnz, split.leftover_rc.size(), b_nnz + split.leftover_rc.size(), idx.size());

    // Reconstruct dense reference from block4 + leftover and compare against original CSR.
    std::vector<float> Wdense(std::size_t(n_in) * n_out, 0.0f);
    for (uint32_t r = 0; r < n_in; ++r)
        for (uint32_t e = ptrs[r]; e < ptrs[r+1]; ++e)
            Wdense[std::size_t(r) * n_out + idx[e]] = w[e];

    std::vector<float> Wrecon(std::size_t(n_in) * n_out, 0.0f);
    // from block4
    {
        const auto& bw = split.block4;
        for (uint32_t br = 0; br < bw.n_block_rows; ++br) {
            uint32_t nblocks = bw.block_ptrs[br+1] - bw.block_ptrs[br];
            std::size_t bytepos = bw.block_byte_ptrs[br];
            uint32_t prev = 0;
            const uint8_t* data = bw.block_data.data() + std::size_t(bw.block_ptrs[br]) * 16;
            for (uint32_t b = 0; b < nblocks; ++b) {
                std::size_t dlen = 0;
                uint32_t delta = uleb128_decode<uint32_t>(bw.block_col_bytes.data() + bytepos, dlen);
                bytepos += dlen;
                uint32_t bcol = prev + delta; prev = bcol;
                const uint8_t* Wb = data + std::size_t(b) * 16;
                for (int lj = 0; lj < 4; ++lj)
                    for (int li = 0; li < 4; ++li) {
                        uint8_t code = Wb[lj*4+li] & 0xF;
                        if (code == 0) continue;
                        uint32_t row = br*4+li, col = bcol*4+lj;
                        Wrecon[std::size_t(row)*n_out+col] = FP4_TABLE[code];
                    }
            }
        }
    }
    for (std::size_t k = 0; k < split.leftover_rc.size(); ++k) {
        auto [row, col] = split.leftover_rc[k];
        Wrecon[std::size_t(row)*n_out+col] = split.leftover_w[k];
    }

    int mismatches = 0;
    for (std::size_t i = 0; i < Wdense.size(); ++i) {
        bool orig_nonzero = Wdense[i] != 0.0f;
        bool recon_nonzero = Wrecon[i] != 0.0f;
        if (orig_nonzero != recon_nonzero) {
            if (mismatches < 10) printf("mismatch at flat idx %zu (row=%zu col=%zu): orig=%.3f recon=%.3f\n",
                                        i, i/n_out, i%n_out, Wdense[i], Wrecon[i]);
            mismatches++;
        }
    }
    printf("mismatches: %d\n", mismatches);
    return 0;
}
