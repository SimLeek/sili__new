#include "../../sili/lib/headers/delta_csr_for.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <random>
#include <vector>

// Frame-of-reference delta encoding -- see delta_csr_for.hpp's own
// header comment for the design (offset-from-group-start instead of
// delta-from-previous, eliminating the prefix-sum dependency that
// capped every ULEB128-based SIMD attempt at ~2.5x) and
// prototypes/for_delta_encoding/ for the benchmark this was validated
// with before landing here.

namespace {

// Scalar ULEB128 reference decode, independent of delta_csr_for.hpp,
// used only to build an independent "ground truth" cumulative-sum
// result to compare FOR decode against.
std::vector<uint32_t> cumulative_from_deltas(const std::vector<uint32_t>& deltas) {
    std::vector<uint32_t> out(deltas.size());
    uint64_t cur = 0;
    for (std::size_t i = 0; i < deltas.size(); ++i) {
        cur += deltas[i];
        out[i] = static_cast<uint32_t>(cur);
    }
    return out;
}

std::vector<uint32_t> random_small_deltas(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::geometric_distribution<int> small(0.4); // median ~1, matches real checkpoint data
    std::vector<uint32_t> deltas(n);
    for (std::size_t i = 0; i < n; ++i) deltas[i] = 1 + small(rng);
    return deltas;
}

} // namespace

TEMPLATE_TEST_CASE_SIG("for_encode_row/for_decode_row round-trip on random small deltas",
                       "[delta_csr_for]", ((std::size_t G), G), 8, 16, 32, 64) {
    for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(7), std::size_t(8),
                          std::size_t(9), std::size_t(G), std::size_t(G + 1), std::size_t(200)}) {
        auto deltas = random_small_deltas(n, 42 + (unsigned)n);
        auto expected = cumulative_from_deltas(deltas);

        std::vector<uint8_t> encoded;
        for_encode_row<G>(deltas.data(), n, encoded);

        std::vector<uint32_t> decoded(n);
        for_decode_row<G>(encoded.data(), n, decoded.data());

        CAPTURE(G, n);
        CHECK(decoded == expected);
    }
}

TEMPLATE_TEST_CASE_SIG("for_encode_row/for_decode_row handles multi-byte (tier escalation) deltas",
                       "[delta_csr_for]", ((std::size_t G), G), 8, 16, 32, 64) {
    // Mix of tiny and large deltas -- forces some groups into tier 1
    // (2-byte) and, with large enough values, tier 2 (4-byte).
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> small(1, 3);
    std::uniform_int_distribution<int> big(1000, 100000);
    std::bernoulli_distribution is_big(0.1);

    for (std::size_t n : {std::size_t(50), std::size_t(500)}) {
        std::vector<uint32_t> deltas(n);
        for (std::size_t i = 0; i < n; ++i)
            deltas[i] = is_big(rng) ? (uint32_t)big(rng) : (uint32_t)small(rng);
        auto expected = cumulative_from_deltas(deltas);

        std::vector<uint8_t> encoded;
        for_encode_row<G>(deltas.data(), n, encoded);

        std::vector<uint32_t> decoded(n);
        for_decode_row<G>(encoded.data(), n, decoded.data());

        CAPTURE(G, n);
        CHECK(decoded == expected);
    }
}

TEST_CASE("for_tier_for_max picks the smallest tier that fits", "[delta_csr_for]") {
    CHECK(for_tier_for_max(0)     == ForWidthTier::Byte1);
    CHECK(for_tier_for_max(255)   == ForWidthTier::Byte1);
    CHECK(for_tier_for_max(256)   == ForWidthTier::Byte2);
    CHECK(for_tier_for_max(65535) == ForWidthTier::Byte2);
    CHECK(for_tier_for_max(65536) == ForWidthTier::Byte4);
}

TEST_CASE("for_*_row_dispatch matches the equivalent templated call for every group size",
         "[delta_csr_for]") {
    auto deltas = random_small_deltas(300, 99);
    auto expected = cumulative_from_deltas(deltas);

    for (auto g : {ForGroupSize::G8, ForGroupSize::G16, ForGroupSize::G32, ForGroupSize::G64}) {
        std::vector<uint8_t> encoded;
        for_encode_row_dispatch(deltas.data(), deltas.size(), encoded, g);

        std::vector<uint32_t> decoded(deltas.size());
        for_decode_row_dispatch(encoded.data(), deltas.size(), decoded.data(), g);

        CAPTURE((int)g);
        CHECK(decoded == expected);
    }
}

TEST_CASE("for_encode_row/for_decode_row on an empty row", "[delta_csr_for]") {
    std::vector<uint32_t> deltas;
    std::vector<uint8_t> encoded;
    for_encode_row<32>(deltas.data(), 0, encoded);
    CHECK(encoded.empty());

    std::vector<uint32_t> decoded;
    for_decode_row<32>(encoded.data(), 0, decoded.data()); // no-op, must not read/write anything
}
