// Attempt 7: Frame-of-Reference (FOR) style group encoding. Per user's
// insight -- instead of encoding delta-from-PREVIOUS-element (which
// forces a prefix sum to reconstruct absolute values, even a "local"
// one within a group), encode OFFSET-FROM-GROUP-START. Absolute values
// are monotonic increasing (deltas are all positive), so within a group
// every offset is independently known relative to the group's own start
// -- decode becomes ONE broadcast-add per group, no shift-add prefix-sum
// tree at all (v3/v5 still had that). The group's own LAST offset,
// already computed by that same add, is exactly the next group's start
// -- the cross-group dependency collapses to reading one already-
// computed vector lane, not a separate reduction.
//
// Tradeoff: offsets can be LARGER than raw deltas (an offset is the sum
// of up to G individual deltas), so groups may need a wider tier more
// often than v3's per-delta tiers did. Tested across G=8/16/32 to see
// where the accounting nets out for real-distribution data.

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

template <size_t G> struct ForCodec {

    // ---- Encoder ---- deltas: raw per-value deltas (NOT cumulative).
    static std::vector<uint8_t> encode(const std::vector<uint32_t>& deltas) {
        std::vector<uint8_t> out;
        out.reserve(deltas.size() * 2);
        size_t n = deltas.size();
        uint64_t cum = 0; // absolute value just before the current group
        for (size_t i = 0; i < n; i += G) {
            size_t g = std::min(G, n - i);
            uint64_t group_start = cum;
            uint32_t offsets[G] = {0};
            uint64_t running = 0;
            for (size_t k = 0; k < g; ++k) {
                running += deltas[i + k];
                offsets[k] = (uint32_t)running; // offset from group_start, monotonic increasing
            }
            for (size_t k = g; k < G; ++k)
                offsets[k] = (g > 0) ? offsets[g - 1] : 0; // pad, harmless
            uint32_t maxv = offsets[G - 1];
            uint8_t tier = (maxv < 256) ? 0 : (maxv < 65536) ? 1 : 2; // 1/2/4 bytes
            int width = (tier == 0) ? 1 : (tier == 1) ? 2 : 4;
            out.push_back(tier);
            for (size_t k = 0; k < G; ++k)
                for (int b = 0; b < width; ++b)
                    out.push_back((uint8_t)(offsets[k] >> (8 * b)));
            cum = group_start + running; // = absolute value of the group's last element
        }
        return out;
    }

    static inline __m256i widen_tier(const uint8_t* p, uint8_t tier,
                                     size_t half /*0 or 1, for G=16*/) {
        if (tier == 0) {
            __m128i chunk = _mm_loadl_epi64((const __m128i*)(p + half * 8));
            return _mm256_cvtepu8_epi32(chunk);
        } else if (tier == 1) {
            __m128i chunk = _mm_loadu_si128((const __m128i*)(p + half * 16));
            return _mm256_cvtepu16_epi32(chunk);
        } else {
            return _mm256_loadu_si256((const __m256i*)(p + half * 32));
        }
    }

    // Decode: ONE broadcast-add per 8-wide chunk, no prefix sum. G may be
    // 8, 16, or 32 -- processed as G/8 independent 8-wide chunks, each just
    // offsets+carry (carry is the SAME scalar for every chunk in a group,
    // since all offsets in a group share one group_start).
    static void decode(const uint8_t* buf, size_t n_groups, uint32_t* out_cols,
                       uint64_t* tier_counts /* [3] */) {
        size_t pos = 0;
        uint32_t carry = 0;
        constexpr size_t chunks = G / 8;
        for (size_t gi = 0; gi < n_groups; ++gi) {
            uint8_t tier = buf[pos++];
            tier_counts[tier]++;
            int width = (tier == 0) ? 1 : (tier == 1) ? 2 : 4;
            __m256i v_carry = _mm256_set1_epi32((int)carry);
            for (size_t c = 0; c < chunks; ++c) {
                __m256i offsets = widen_tier(buf + pos, tier, c);
                __m256i result = _mm256_add_epi32(offsets, v_carry); // <-- the whole "prefix sum"
                _mm256_storeu_si256((__m256i*)(out_cols + gi * G + c * 8), result);
            }
            pos += G * width;
            carry = out_cols[gi * G + G - 1]; // last element of this group == next group's start
        }
    }

}; // ForCodec

inline uint32_t uleb128_decode_scalar(const uint8_t* buf, size_t& pos) {
    uint32_t result = 0;
    int shift = 0;
    while (true) {
        uint8_t byte = buf[pos++];
        result |= (uint32_t)(byte & 0x7Fu) << shift;
        if (!(byte & 0x80u))
            break;
        shift += 7;
    }
    return result;
}
std::vector<uint8_t> encode_uleb128(const std::vector<uint32_t>& deltas) {
    std::vector<uint8_t> buf;
    for (uint32_t v : deltas) {
        while (v >= 0x80) {
            buf.push_back((uint8_t)((v & 0x7F) | 0x80));
            v >>= 7;
        }
        buf.push_back((uint8_t)v);
    }
    return buf;
}
void decode_uleb128_scalar(const uint8_t* buf, size_t n, uint32_t* out_cols) {
    size_t pos = 0;
    uint32_t cur = 0;
    for (size_t i = 0; i < n; ++i) {
        cur += uleb128_decode_scalar(buf, pos);
        out_cols[i] = cur;
    }
}

std::vector<uint32_t> make_raw_deltas(size_t n, double pct_multibyte, unsigned seed) {
    std::mt19937 rng(seed);
    std::geometric_distribution<int> small_delta(0.4); // median ~1, matches real checkpoint data
    std::uniform_int_distribution<int> big_delta(200, 5000);
    std::bernoulli_distribution is_multibyte(pct_multibyte);
    std::vector<uint32_t> deltas(n);
    for (size_t i = 0; i < n; ++i)
        deltas[i] = is_multibyte(rng) ? (uint32_t)big_delta(rng) : (uint32_t)(1 + small_delta(rng));
    return deltas;
}

template <size_t G> void run_bench(double pct_multibyte, size_t N) {
    size_t n_groups = (N + G - 1) / G;
    size_t N_padded = n_groups * G;
    auto deltas = make_raw_deltas(N, pct_multibyte, 42);
    auto uleb_buf = encode_uleb128(deltas);
    auto for_buf = ForCodec<G>::encode(deltas);

    std::vector<uint32_t> out_ref(N_padded), out_for(N_padded);
    decode_uleb128_scalar(uleb_buf.data(), N, out_ref.data());
    uint64_t tier_counts[3] = {0, 0, 0};
    ForCodec<G>::decode(for_buf.data(), n_groups, out_for.data(), tier_counts);

    bool correct = true;
    for (size_t i = 0; i < N; ++i)
        if (out_ref[i] != out_for[i]) {
            correct = false;
            break;
        }

    const int REPS = 5000;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < REPS; ++r)
        decode_uleb128_scalar(uleb_buf.data(), N, out_ref.data());
    auto t1 = std::chrono::steady_clock::now();
    double ms_scalar = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < REPS; ++r) {
        uint64_t tc[3] = {0, 0, 0};
        ForCodec<G>::decode(for_buf.data(), n_groups, out_for.data(), tc);
    }
    t1 = std::chrono::steady_clock::now();
    double ms_for = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

    double total_tiers = (double)(tier_counts[0] + tier_counts[1] + tier_counts[2]);
    printf("G=%2zu pct_mb=%.3f N=%5zu  correct=%s  uleb=%6zuB for=%6zuB(%.2fx)  "
           "scalar=%.5fms  for=%.5fms  speedup=%.2fx  tier0/1/2=%.0f%%/%.0f%%/%.0f%%\n",
           G, pct_multibyte, N, correct ? "yes" : "NO", uleb_buf.size(), for_buf.size(),
           (double)for_buf.size() / uleb_buf.size(), ms_scalar, ms_for, ms_scalar / ms_for,
           100.0 * tier_counts[0] / total_tiers, 100.0 * tier_counts[1] / total_tiers,
           100.0 * tier_counts[2] / total_tiers);
}

int main() {
    for (double pct_mb : {0.0, 0.001, 0.01}) {
        for (size_t N : {size_t(100), size_t(500), size_t(1000), size_t(1200), size_t(2000),
                         size_t(4000), size_t(8000)}) {
            run_bench<8>(pct_mb, N);
            run_bench<16>(pct_mb, N);
            run_bench<32>(pct_mb, N);
            run_bench<64>(pct_mb, N);
        }
    }
    return 0;
}
