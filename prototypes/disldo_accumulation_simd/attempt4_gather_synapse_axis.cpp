// Same proven technique as the earlier (correctness-verified, 1.3-1.84x)
// gather prototype, but applied to the correct axis per user feedback:
// batch=1 (sili's real/primary use case, not the batch=20-50 that
// sili_peridot's B6/B7 conversion happens to use), vectorizing across 8
// SYNAPSES at once instead of 8 batch elements. This axis always has
// real width (n_row synapses per row) regardless of batch size.
//
// Per synapse group of 8: gather 8 arbitrary mo[] values (AVX2 gather,
// available on this hardware), compute 8 contributions via one SIMD
// multiply (w[] and input_r are trivially vectorizable -- no merge
// logic needed, cols don't need to be examined at all for this part),
// add, write back via 8 individual scalar stores (no scatter instruction
// on this hardware, but that's fine -- stores don't need to be one op).

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

void accumulate_scalar(const float* w_per_synapse, const uint32_t* cols,
                        std::size_t n_synapses, float input_r, float* mo) {
    for (std::size_t e = 0; e < n_synapses; ++e) mo[cols[e]] += w_per_synapse[e] * input_r;
}

void accumulate_gather_synapse(const float* w_per_synapse, const uint32_t* cols,
                                std::size_t n_synapses, float input_r, float* mo) {
    const __m256 iv = _mm256_set1_ps(input_r);
    std::size_t e = 0;
    for (; e + 8 <= n_synapses; e += 8) {
        __m256i col_v = _mm256_loadu_si256((const __m256i*)(cols + e));
        __m256  w_v   = _mm256_loadu_ps(w_per_synapse + e);
        __m256  contrib = _mm256_mul_ps(w_v, iv);

        __m256 cur = _mm256_i32gather_ps(mo, col_v, 4);
        __m256 result = _mm256_add_ps(cur, contrib);

        alignas(32) int32_t col_arr[8];
        alignas(32) float result_arr[8];
        _mm256_store_si256((__m256i*)col_arr, col_v);
        _mm256_storeu_ps(result_arr, result);
        for (int k = 0; k < 8; ++k) mo[col_arr[k]] = result_arr[k];
    }
    for (; e < n_synapses; ++e) mo[cols[e]] += w_per_synapse[e] * input_r; // tail
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

            std::vector<float> mo_ref(n_out, 0.0f), mo_gather(n_out, 0.0f);
            accumulate_scalar(w.data(), cols.data(), n_synapses, input_r, mo_ref.data());
            accumulate_gather_synapse(w.data(), cols.data(), n_synapses, input_r, mo_gather.data());

            bool correct = true;
            for (std::size_t i = 0; i < n_out; ++i)
                if (std::abs(mo_ref[i] - mo_gather[i]) > 1e-4f) { correct = false; break; }

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
                std::fill(mo_gather.begin(), mo_gather.end(), 0.0f);
                accumulate_gather_synapse(w.data(), cols.data(), n_synapses, input_r, mo_gather.data());
            }
            t1 = std::chrono::steady_clock::now();
            double ms_gather = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

            printf("density=%.2f n_out=%5zu n_syn=%5zu  correct=%s  scalar=%.5fms gather=%.5fms(%.2fx)\n",
                   density, n_out, n_synapses, correct ? "yes" : "NO", ms_scalar, ms_gather, ms_scalar / ms_gather);
        }
    }
    return 0;
}
