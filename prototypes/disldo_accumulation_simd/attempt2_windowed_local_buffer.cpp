// Corrected prototype per user feedback: NO batch parallelism (batch=1
// almost always in sili's real/primary use -- the earlier gather-across-
// batch prototype targeted the wrong axis, since it only looked good at
// the batch=20-50 scale sili_peridot's B6/B7 conversion happens to use,
// not sili's actual typical online/recurrent workload).
//
// Real insight: the accumulate into mo[] is NOT a true random scatter.
// CSR columns within a row are monotonically sorted (both old ULEB128
// and the new FOR encoding preserve this), and at real-checkpoint
// density (67-90%), a decoded GROUP of G columns clusters into a small
// window rather than spanning the whole output width. So: decode a
// group -> find its [min, max] column window -> scatter the group's few
// contributions into a small LOCAL stack buffer at that window (cheap,
// on-stack, no random global memory access) -> do ONE contiguous
// vectorized load-add-store against mo[] over that window (AVX2 masked
// load/store handles the ragged tail, no AVX-512/scatter needed at all).

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

constexpr std::size_t G = 32; // FOR group size for this prototype

// Scalar baseline: batch=1 shape of the CURRENT real disldo_forward inner
// loop (mo[col] += w*iv, one row's worth of synapses).
void accumulate_scalar(const float* w_per_synapse, const uint32_t* cols,
                        std::size_t n_synapses, float input_r, float* mo) {
    for (std::size_t e = 0; e < n_synapses; ++e) {
        mo[cols[e]] += w_per_synapse[e] * input_r;
    }
}

// Windowed SIMD: process synapses in groups of G (matching FOR's own
// group size -- columns within a group are sorted, so [min,max] is just
// the first/last element), scatter into a small local buffer, one
// contiguous masked load-add-store per group.
void accumulate_windowed_simd(const float* w_per_synapse, const uint32_t* cols,
                               std::size_t n_synapses, float input_r, float* mo,
                               std::size_t& max_window_seen, double& sum_window_seen) {
    std::size_t i = 0;
    alignas(32) float local_buf[4096]; // generous cap; real window sizes measured below
    while (i < n_synapses) {
        const std::size_t g = std::min(G, n_synapses - i);
        const uint32_t win_start = cols[i];
        const uint32_t win_end   = cols[i + g - 1];
        const std::size_t win_len = win_end - win_start + 1;

        max_window_seen = std::max(max_window_seen, win_len);
        sum_window_seen += (double)win_len;

        if (win_len > 4096) {
            // Fallback for a pathologically sparse group (shouldn't happen
            // at real checkpoint density, but stay correct regardless).
            for (std::size_t k = 0; k < g; ++k) mo[cols[i + k]] += w_per_synapse[i + k] * input_r;
            i += g;
            continue;
        }

        std::memset(local_buf, 0, win_len * sizeof(float));
        for (std::size_t k = 0; k < g; ++k)
            local_buf[cols[i + k] - win_start] += w_per_synapse[i + k] * input_r;

        std::size_t k = 0;
        for (; k + 8 <= win_len; k += 8) {
            __m256 cur = _mm256_loadu_ps(mo + win_start + k);
            __m256 add = _mm256_loadu_ps(local_buf + k);
            _mm256_storeu_ps(mo + win_start + k, _mm256_add_ps(cur, add));
        }
        if (k < win_len) {
            const int rem = (int)(win_len - k);
            __m256i mask = _mm256_setr_epi32(
                rem > 0 ? -1 : 0, rem > 1 ? -1 : 0, rem > 2 ? -1 : 0, rem > 3 ? -1 : 0,
                rem > 4 ? -1 : 0, rem > 5 ? -1 : 0, rem > 6 ? -1 : 0, rem > 7 ? -1 : 0);
            __m256 cur = _mm256_maskload_ps(mo + win_start + k, mask);
            __m256 add = _mm256_maskload_ps(local_buf + k, mask);
            _mm256_maskstore_ps(mo + win_start + k, mask, _mm256_add_ps(cur, add));
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
            cols.reserve(n_synapses);
            {
                std::vector<uint32_t> all_cols(n_out);
                for (uint32_t c = 0; c < n_out; ++c) all_cols[c] = c;
                std::shuffle(all_cols.begin(), all_cols.end(), rng);
                all_cols.resize(n_synapses);
                std::sort(all_cols.begin(), all_cols.end()); // CSR columns are sorted
                cols = all_cols;
            }
            std::vector<float> w(n_synapses);
            for (auto& v : w) v = wdist(rng);
            const float input_r = 0.37f;

            std::vector<float> mo_ref(n_out, 0.0f), mo_win(n_out, 0.0f);
            accumulate_scalar(w.data(), cols.data(), n_synapses, input_r, mo_ref.data());
            std::size_t max_window = 0; double sum_window = 0;
            accumulate_windowed_simd(w.data(), cols.data(), n_synapses, input_r, mo_win.data(), max_window, sum_window);

            bool correct = true;
            for (std::size_t i = 0; i < n_out; ++i)
                if (std::abs(mo_ref[i] - mo_win[i]) > 1e-4f) { correct = false; break; }

            const int REPS = 2000;
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < REPS; ++r) {
                std::fill(mo_ref.begin(), mo_ref.end(), 0.0f);
                accumulate_scalar(w.data(), cols.data(), n_synapses, input_r, mo_ref.data());
            }
            auto t1 = std::chrono::steady_clock::now();
            double ms_scalar = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

            std::size_t mw = 0; double sw = 0;
            t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < REPS; ++r) {
                std::fill(mo_win.begin(), mo_win.end(), 0.0f);
                mw = 0; sw = 0;
                accumulate_windowed_simd(w.data(), cols.data(), n_synapses, input_r, mo_win.data(), mw, sw);
            }
            t1 = std::chrono::steady_clock::now();
            double ms_win = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

            const std::size_t n_groups = (n_synapses + G - 1) / G;
            printf("density=%.2f n_out=%5zu n_syn=%5zu  correct=%s  scalar=%.5fms windowed=%.5fms(%.2fx)  "
                   "avg_window=%.1f max_window=%zu (group_size=%zu)\n",
                   density, n_out, n_synapses, correct ? "yes" : "NO", ms_scalar, ms_win, ms_scalar / ms_win,
                   sw / n_groups, mw, G);
        }
    }
    return 0;
}
