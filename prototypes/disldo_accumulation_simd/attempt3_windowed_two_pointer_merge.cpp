// Corrected again per user feedback: no local buffer for the whole
// window (that was the bug last time -- redundant scatter-then-merge
// pass). Instead: direct two-pointer merge between window positions and
// the sorted synapse list, filling each 8-wide SIMD chunk on the fly
// (tiny, register-sized, reused per chunk -- NOT a per-window scratch
// buffer) and adding straight into mo[]. No block-sparse reformatting,
// no encode-time format change -- just a smarter runtime merge over the
// EXISTING sorted column list. Also tests a density-threshold dispatch
// (dense-merge vs plain scalar) decided ONCE per group from window_len/G,
// not a per-element branch, per the user's "statistical measure outside
// the loop" suggestion.

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

constexpr std::size_t G = 32;

void accumulate_scalar(const float* w_per_synapse, const uint32_t* cols,
                        std::size_t n_synapses, float input_r, float* mo) {
    for (std::size_t e = 0; e < n_synapses; ++e) mo[cols[e]] += w_per_synapse[e] * input_r;
}

// Direct two-pointer merge, no scratch window buffer -- one small
// (8-float, register/stack resident) chunk buffer reused per SIMD chunk.
// "SIMD if" per user's point: the window's FIRST and LAST chunks are
// guaranteed non-empty (win_start/win_end ARE active columns by
// definition), but an INTERIOR chunk can be entirely empty if there's a
// local gap -- a cheap vectorized all-zero check (compare + movemask)
// skips the load-add-store for those, avoiding wasted memory traffic on
// chunks that wouldn't change anything, without a per-lane scalar branch.
void accumulate_dense_merge(const uint32_t* cols, const float* w, std::size_t syn_start, std::size_t syn_end,
                             uint32_t win_start, uint32_t win_end, float input_r, float* mo) {
    const std::size_t win_len = win_end - win_start + 1;
    std::size_t syn_idx = syn_start;
    std::size_t pos = 0;
    while (pos < win_len) {
        const std::size_t chunk_len = std::min((std::size_t)8, win_len - pos);
        alignas(32) float chunk[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const uint32_t chunk_base = win_start + (uint32_t)pos;
        bool any_active = false;
        while (syn_idx < syn_end && cols[syn_idx] < chunk_base + chunk_len) {
            chunk[cols[syn_idx] - chunk_base] = w[syn_idx] * input_r;
            any_active = true;
            ++syn_idx;
        }
        if (any_active) {
            if (chunk_len == 8) {
                __m256 cur = _mm256_loadu_ps(mo + chunk_base);
                __m256 add = _mm256_load_ps(chunk);
                _mm256_storeu_ps(mo + chunk_base, _mm256_add_ps(cur, add));
            } else {
                const int rem = (int)chunk_len;
                __m256i mask = _mm256_setr_epi32(
                    rem > 0 ? -1 : 0, rem > 1 ? -1 : 0, rem > 2 ? -1 : 0, rem > 3 ? -1 : 0,
                    rem > 4 ? -1 : 0, rem > 5 ? -1 : 0, rem > 6 ? -1 : 0, rem > 7 ? -1 : 0);
                __m256 cur = _mm256_maskload_ps(mo + chunk_base, mask);
                __m256 add = _mm256_load_ps(chunk);
                _mm256_maskstore_ps(mo + chunk_base, mask, _mm256_add_ps(cur, add));
            }
        }
        pos += chunk_len;
    }
}

// Per-group dispatch: dense-merge if the group's window isn't too sparse
// (decided once per group of G, not per element), else fall back to
// plain scalar pointer-walk for that group.
void accumulate_hybrid(const float* w_per_synapse, const uint32_t* cols,
                        std::size_t n_synapses, float input_r, float* mo, double threshold_ratio) {
    std::size_t i = 0;
    while (i < n_synapses) {
        const std::size_t g = std::min(G, n_synapses - i);
        const uint32_t win_start = cols[i];
        const uint32_t win_end   = cols[i + g - 1];
        const std::size_t win_len = win_end - win_start + 1;
        const double ratio = (double)win_len / (double)g; // "empty slots per active" proxy

        if (ratio <= threshold_ratio) {
            accumulate_dense_merge(cols, w_per_synapse, i, i + g, win_start, win_end, input_r, mo);
        } else {
            for (std::size_t k = 0; k < g; ++k) mo[cols[i + k]] += w_per_synapse[i + k] * input_r;
        }
        i += g;
    }
}

int main() {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);

    for (double density : {0.67, 0.75, 0.90}) {
        for (std::size_t n_out : {size_t(1536), size_t(4096)}) {
            const std::size_t n_synapses = (std::size_t)(n_out * density);
            std::vector<uint32_t> cols;
            {
                std::vector<uint32_t> all_cols(n_out);
                for (uint32_t c = 0; c < n_out; ++c) all_cols[c] = c;
                std::shuffle(all_cols.begin(), all_cols.end(), rng);
                all_cols.resize(n_synapses);
                std::sort(all_cols.begin(), all_cols.end());
                cols = all_cols;
            }
            std::vector<float> w(n_synapses);
            for (auto& v : w) v = wdist(rng);
            const float input_r = 0.37f;

            std::vector<float> mo_ref(n_out, 0.0f), mo_hybrid(n_out, 0.0f);
            accumulate_scalar(w.data(), cols.data(), n_synapses, input_r, mo_ref.data());
            accumulate_hybrid(w.data(), cols.data(), n_synapses, input_r, mo_hybrid.data(), 6.0);

            bool correct = true;
            for (std::size_t i = 0; i < n_out; ++i)
                if (std::abs(mo_ref[i] - mo_hybrid[i]) > 1e-4f) { correct = false; break; }

            const int REPS = 3000;
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < REPS; ++r) {
                std::fill(mo_ref.begin(), mo_ref.end(), 0.0f);
                accumulate_scalar(w.data(), cols.data(), n_synapses, input_r, mo_ref.data());
            }
            auto t1 = std::chrono::steady_clock::now();
            double ms_scalar = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

            t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < REPS; ++r) {
                std::fill(mo_hybrid.begin(), mo_hybrid.end(), 0.0f);
                accumulate_hybrid(w.data(), cols.data(), n_synapses, input_r, mo_hybrid.data(), 6.0);
            }
            t1 = std::chrono::steady_clock::now();
            double ms_hybrid = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

            printf("density=%.2f n_out=%5zu n_syn=%5zu  correct=%s  scalar=%.5fms hybrid=%.5fms(%.2fx)\n",
                   density, n_out, n_synapses, correct ? "yes" : "NO", ms_scalar, ms_hybrid, ms_scalar / ms_hybrid);
        }
    }
    return 0;
}
