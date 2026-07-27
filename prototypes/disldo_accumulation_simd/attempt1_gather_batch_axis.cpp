// Prototype: can disldo_forward's inner batch-accumulation loop be
// vectorized on THIS hardware (AVX2 + BMI2, no AVX-512, so gather is
// available but scatter is not)? Per user's point: gather the current
// mo[] values at 8 batch-strided positions, add 8 contributions in one
// vectorized op, write back via individual scalar stores (no hardware
// scatter needed for that part -- stores don't have to be a single
// instruction). Also removes the `if (iv==0) continue` branch (a sparse-
// input optimization, not required for correctness -- multiplying by a
// true zero and adding it is a no-op either way) since that branch is
// one of the two "unsupported control flow" reasons GCC gave for failing
// to auto-vectorize this loop.
//
// Matches disldo_forward's real shape: OUTER loop over synapses (each
// with its own fixed `col`), INNER loop over batch -- vectorizing across
// batch (fixed col per synapse) means the 8 target addresses are
// mo[(b+0..7)*n_out + col], a regular strided-but-not-contiguous pattern
// (stride n_out), exactly what gather is for.

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

// Scalar baseline -- matches the CURRENT real code's structure exactly
// (branch included).
void accumulate_scalar_branchy(const float* w_per_synapse, const uint32_t* col_per_synapse,
                                std::size_t n_synapses, const float* input, std::size_t batch,
                                std::size_t n_out, float* mo) {
    for (std::size_t e = 0; e < n_synapses; ++e) {
        const float w = w_per_synapse[e];
        const uint32_t col = col_per_synapse[e];
        for (std::size_t b = 0; b < batch; ++b) {
            const float iv = input[b];
            if (iv == 0.0f) continue;
            const float contrib = w * iv;
            mo[b * n_out + col] += contrib;
        }
    }
}

// Branch-free scalar (removes the `if` -- still fully scalar, isolates
// how much the branch itself costs vs. the gather/SIMD question).
void accumulate_scalar_branchfree(const float* w_per_synapse, const uint32_t* col_per_synapse,
                                   std::size_t n_synapses, const float* input, std::size_t batch,
                                   std::size_t n_out, float* mo) {
    for (std::size_t e = 0; e < n_synapses; ++e) {
        const float w = w_per_synapse[e];
        const uint32_t col = col_per_synapse[e];
        for (std::size_t b = 0; b < batch; ++b) {
            mo[b * n_out + col] += w * input[b];
        }
    }
}

// Gather (read) + SIMD add + scalar store-back (write) -- 8-wide across
// the batch dimension, fixed synapse/col per outer iteration.
void accumulate_gather_simd(const float* w_per_synapse, const uint32_t* col_per_synapse,
                             std::size_t n_synapses, const float* input, std::size_t batch,
                             std::size_t n_out, float* mo) {
    const __m256i stride = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); // batch offsets 0..7
    const __m256i n_out_v = _mm256_set1_epi32(static_cast<int>(n_out));

    for (std::size_t e = 0; e < n_synapses; ++e) {
        const __m256 w = _mm256_set1_ps(w_per_synapse[e]);
        const uint32_t col = col_per_synapse[e];
        const __m256i col_v = _mm256_set1_epi32(static_cast<int>(col));

        std::size_t b = 0;
        for (; b + 8 <= batch; b += 8) {
            // addresses (as element indices): (b+0..7)*n_out + col
            __m256i b_idx = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(b)), stride);
            __m256i addr  = _mm256_add_epi32(_mm256_mullo_epi32(b_idx, n_out_v), col_v);

            __m256 iv = _mm256_loadu_ps(input + b); // input is contiguous, no gather needed here
            __m256 contrib = _mm256_mul_ps(w, iv);

            __m256 cur = _mm256_i32gather_ps(mo, addr, 4); // gather: available on AVX2
            __m256 result = _mm256_add_ps(cur, contrib);

            alignas(32) int32_t addr_arr[8];
            alignas(32) float result_arr[8];
            _mm256_store_si256((__m256i*)addr_arr, addr);
            _mm256_storeu_ps(result_arr, result);
            for (int k = 0; k < 8; ++k) mo[addr_arr[k]] = result_arr[k]; // no hardware scatter -- individual stores
        }
        for (; b < batch; ++b) mo[b * n_out + col] += w_per_synapse[e] * input[b]; // tail
    }
}

int main() {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> ivdist(-1.0f, 1.0f);

    for (std::size_t n_out : {size_t(1536), size_t(4096)}) {
        for (std::size_t batch : {size_t(20), size_t(50)}) {
            for (std::size_t n_synapses : {size_t(1000), size_t(4000)}) {
                std::vector<float> w(n_synapses);
                std::vector<uint32_t> cols(n_synapses);
                std::uniform_int_distribution<uint32_t> coldist(0, (uint32_t)n_out - 1);
                for (std::size_t i = 0; i < n_synapses; ++i) { w[i] = wdist(rng); cols[i] = coldist(rng); }
                std::vector<float> input(batch);
                for (auto& v : input) v = ivdist(rng);

                std::vector<float> mo_ref(batch * n_out, 0.0f), mo_bf(batch * n_out, 0.0f), mo_simd(batch * n_out, 0.0f);

                accumulate_scalar_branchy(w.data(), cols.data(), n_synapses, input.data(), batch, n_out, mo_ref.data());
                accumulate_scalar_branchfree(w.data(), cols.data(), n_synapses, input.data(), batch, n_out, mo_bf.data());
                accumulate_gather_simd(w.data(), cols.data(), n_synapses, input.data(), batch, n_out, mo_simd.data());

                bool correct_bf = true, correct_simd = true;
                for (std::size_t i = 0; i < mo_ref.size(); ++i) {
                    if (std::abs(mo_ref[i] - mo_bf[i]) > 1e-4f) correct_bf = false;
                    if (std::abs(mo_ref[i] - mo_simd[i]) > 1e-3f) correct_simd = false; // slightly looser: SIMD sum order differs
                }

                const int REPS = 200;
                auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < REPS; ++r) {
                    std::fill(mo_ref.begin(), mo_ref.end(), 0.0f);
                    accumulate_scalar_branchy(w.data(), cols.data(), n_synapses, input.data(), batch, n_out, mo_ref.data());
                }
                auto t1 = std::chrono::steady_clock::now();
                double ms_branchy = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

                t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < REPS; ++r) {
                    std::fill(mo_bf.begin(), mo_bf.end(), 0.0f);
                    accumulate_scalar_branchfree(w.data(), cols.data(), n_synapses, input.data(), batch, n_out, mo_bf.data());
                }
                t1 = std::chrono::steady_clock::now();
                double ms_bf = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

                t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < REPS; ++r) {
                    std::fill(mo_simd.begin(), mo_simd.end(), 0.0f);
                    accumulate_gather_simd(w.data(), cols.data(), n_synapses, input.data(), batch, n_out, mo_simd.data());
                }
                t1 = std::chrono::steady_clock::now();
                double ms_simd = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

                printf("n_out=%5zu batch=%3zu n_syn=%5zu  correct_bf=%s correct_simd=%s  "
                       "branchy=%.4fms branchfree=%.4fms(%.2fx) gather_simd=%.4fms(%.2fx)\n",
                       n_out, batch, n_synapses, correct_bf ? "yes" : "NO", correct_simd ? "yes" : "NO",
                       ms_branchy, ms_bf, ms_branchy / ms_bf, ms_simd, ms_branchy / ms_simd);
            }
        }
    }
    return 0;
}
