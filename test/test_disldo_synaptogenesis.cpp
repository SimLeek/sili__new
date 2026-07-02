#include "../sili/lib/headers/linear_disldo.hpp"
#include "../sili/lib/headers/sparse_struct.hpp"
#include "tests_main.hpp"
#include <catch2/catch_all.hpp>
#include <cstddef>
#include <vector>

// ── disldo_forward / disldo_backward ─────────────────────────────────────────
//
// Generic-over-VALUES_TYPE DISLDO (dense-input) forward/backward. Previously
// unwritten: disldo_forward/backward existed but used plain float32 +
// absolute CSR, never DeltaCSRLayout or FP4BiPacked at all -- see conversation.

TEST_CASE("disldo_forward matches dense reference", "[disldo][forward]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    // 3 inputs, 4 outputs: row0->col0(1.5), row0->col2(-2.0), row1->col1(0.5), row2->col3(3.0)
    std::vector<SIZE_TYPE> ptrs = {0, 2, 3, 4};
    std::vector<SIZE_TYPE> idx  = {0, 2, 1, 3};
    std::vector<float>     w    = {1.5f, -2.0f, 0.5f, 3.0f};
    std::vector<float>     imp  = {0.0f, 0.0f, 0.0f, 0.0f};

    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(3), std::size_t(4), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(4, SIZE_TYPE(0));

    std::vector<float> input = {1.0f, 0.6f, 2.0f};
    std::vector<float> output(4, 0.0f);

    disldo_forward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        input.data(), SIZE_TYPE(1), SIZE_TYPE(3), weights, output.data(), 0.0f, 1);

    // Dense reference: out[c] = sum_r W[r,c] * input[r]
    std::vector<float> expected = {1.5f, 0.3f, -2.0f, 6.0f};
    CHECK_VECTOR_ALMOST_EQUAL(output, expected, 1e-4f);
}

TEST_CASE("disldo_backward dx matches dense reference, weights unchanged at lr=0",
         "[disldo][backward]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    std::vector<SIZE_TYPE> ptrs = {0, 2, 3, 4};
    std::vector<SIZE_TYPE> idx  = {0, 2, 1, 3};
    std::vector<float>     w    = {1.5f, -2.0f, 0.5f, 3.0f};
    std::vector<float>     imp  = {0.0f, 0.0f, 0.0f, 0.0f};

    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(3), std::size_t(4), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(4, SIZE_TYPE(0));

    std::vector<float> input = {1.0f, 0.6f, 2.0f};
    std::vector<float> dy    = {0.3f, -0.5f, 0.2f, 0.4f};
    std::vector<float> dx(3, 0.0f);
    std::vector<float> in_acc(3, 0.0f), gr_acc(4, 0.0f);

    disldo_backward<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        input.data(), SIZE_TYPE(1), SIZE_TYPE(3), dy.data(),
        weights, dx.data(), in_acc.data(), gr_acc.data(), 0.0f, 1);

    // Dense reference: dx[r] = sum_c W[r,c] * dy[c]
    // dx[0] = 1.5*0.3 + -2.0*0.2 = 0.05, dx[1] = 0.5*-0.5 = -0.25, dx[2] = 3.0*0.4 = 1.2
    std::vector<float> expected_dx = {0.05f, -0.25f, 1.2f};
    CHECK_VECTOR_ALMOST_EQUAL(dx, expected_dx, 1e-4f);

    CHECK(in_acc[0] == Catch::Approx(1.0f));
    CHECK(in_acc[1] == Catch::Approx(0.6f));
    CHECK(gr_acc[0] == Catch::Approx(0.3f));

    auto cur = weights.connections.row_cursor(0);
    cur.advance();
    const float w0 = ValueAccessor<FP4BiPacked>::get_w(
        weights.connections.values, weights.connections.layout.elem_start[0]);
    CHECK(w0 == Catch::Approx(1.5f));
}

// ── delta_csr_row_rebuild ─────────────────────────────────────────────────────
//
// Regression tests for two bugs found by direct execution (see conversation):
// (1) blank_fraction typo zeroed growth headroom on every freshly-imported row
// (2) nnz_delta read row_nnz() AFTER elem_end was already overwritten, so
//     total_nnz never changed even on a successful rebuild.

TEST_CASE("delta_csr_row_rebuild actually grows total_nnz on success", "[synaptogenesis]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    std::vector<SIZE_TYPE> ptrs = {0, 2, 3, 4};
    std::vector<SIZE_TYPE> idx  = {0, 2, 1, 3};
    std::vector<float>     w    = {1.5f, -2.0f, 0.5f, 3.0f};
    std::vector<float>     imp  = {0.0f, 0.0f, 0.0f, 0.0f};

    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(3), std::size_t(4), std::size_t(64), std::size_t(64));

    REQUIRE(dc.nnz() == 4);

    // Row 0 currently has cols {0,2}. Rebuild with an added col=1 -- fits
    // within the 20% growth headroom reserved by delta_csr_from_absolute.
    std::vector<COL_TYPE> new_cols = {0, 1, 2};
    std::vector<float>    new_w    = {1.5f, 0.0f, -2.0f};
    std::vector<float>    new_imp  = {0.0f, 20.0f, 0.0f};

    const bool ok = delta_csr_row_rebuild<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        dc, 0, new_cols, new_w, new_imp);

    REQUIRE(ok);
    CHECK(dc.nnz() == 5);              // was silently staying at 4 before the fix
    CHECK(dc.layout.row_nnz(0) == 3);
}

TEST_CASE("delta_csr_row_rebuild fails cleanly when growth exceeds reserved headroom",
         "[synaptogenesis]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    // Rows with only 1 synapse each get minimal headroom (by design --
    // growth is meant to be incremental across many synaptogenesis calls,
    // not one large jump on a near-empty row).
    std::vector<SIZE_TYPE> ptrs = {0, 1, 2};
    std::vector<SIZE_TYPE> idx  = {1, 3};
    std::vector<float>     w    = {0.5f, 3.0f};
    std::vector<float>     imp  = {0.0f, 0.0f};

    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(2), std::size_t(4), std::size_t(64), std::size_t(64));

    const std::size_t nnz_before = dc.nnz();

    // Attempt to grow row 0 from 1 to 4 synapses in one step -- should
    // legitimately fail (not enough reserved headroom), not silently no-op.
    std::vector<COL_TYPE> too_many = {0, 1, 2, 3};
    std::vector<float>    tw(4, 1.0f), ti(4, 0.0f);

    const bool ok = delta_csr_row_rebuild<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        dc, 0, too_many, tw, ti);

    CHECK_FALSE(ok);
    CHECK(dc.nnz() == nnz_before);   // must be unchanged, not partially applied
}

// ── delta_csr_build_probes ────────────────────────────────────────────────────

TEST_CASE("delta_csr_build_probes: outer product, top-k, excludes existing",
         "[synaptogenesis][build_probes]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    std::vector<SIZE_TYPE> ptrs = {0, 2, 3, 4};
    std::vector<SIZE_TYPE> idx  = {0, 2, 1, 3};
    std::vector<float>     w    = {1.5f, -2.0f, 0.5f, 3.0f};
    std::vector<float>     imp  = {0.0f, 0.0f, 0.0f, 0.0f};
    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(3), std::size_t(4), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(4, SIZE_TYPE(0));

    std::vector<float> input_accum = {5.0f, 0.1f, 3.0f};
    std::vector<float> grad_accum  = {0.1f, 4.0f, 0.1f, 0.1f};

    delta_csr_build_probes<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, input_accum.data(), grad_accum.data(), SIZE_TYPE(3));

    REQUIRE(weights.probes.nnz() > 0);

    // No probe may duplicate an existing connection.
    auto& pr = *weights.probes.indices[0];
    auto& pc = *weights.probes.indices[1];
    for (std::size_t i = 0; i < pr.size(); ++i) {
        auto cur = weights.connections.row_cursor(static_cast<std::size_t>(pr[i]));
        while (!cur.at_end())
            REQUIRE(cur.advance() != static_cast<COL_TYPE>(pc[i]));
    }

    // The globally hottest pair (row0 x col1: 5.0*4.0=20) must be present.
    bool found_hottest = false;
    auto& pv = *weights.probes.values[0];
    for (std::size_t i = 0; i < pr.size(); ++i)
        if (pr[i] == 0 && pc[i] == 1) { found_hottest = true; CHECK(pv[i] == Catch::Approx(20.0f)); }
    CHECK(found_hottest);
}

TEST_CASE("delta_csr_build_probes: per_row finds candidates global mode starves",
         "[synaptogenesis][build_probes][per_row]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    // Row 0 already connects to the 3 globally-hottest outputs (cols 0,1,2).
    // Global top-k=3 mode has nothing left to offer row0. Per-row mode looks
    // past those and finds row0 has genuinely new candidates among cols 3,4.
    std::vector<SIZE_TYPE> ptrs = {0, 3, 3};
    std::vector<SIZE_TYPE> idx  = {0, 1, 2};
    std::vector<float>     w    = {1.0f, 1.0f, 1.0f};
    std::vector<float>     imp  = {0.0f, 0.0f, 0.0f};
    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(2), std::size_t(5), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights_global;
    weights_global.connections = dc;
    weights_global.out_degree.assign(5, SIZE_TYPE(0));
    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights_perrow;
    weights_perrow.connections = dc;
    weights_perrow.out_degree.assign(5, SIZE_TYPE(0));

    std::vector<float> input_accum = {5.0f, 5.0f};
    std::vector<float> grad_accum  = {10.0f, 9.0f, 8.0f, 1.0f, 0.5f};

    delta_csr_build_probes<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights_global, input_accum.data(), grad_accum.data(), SIZE_TYPE(3), false);
    delta_csr_build_probes<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights_perrow, input_accum.data(), grad_accum.data(), SIZE_TYPE(3), true);

    auto count_row0 = [](auto& w) {
        int n = 0;
        for (auto r : *w.probes.indices[0]) if (r == 0) ++n;
        return n;
    };

    CHECK(count_row0(weights_global) == 0);   // starved -- confirmed limitation of global mode
    CHECK(count_row0(weights_perrow) > 0);    // per_row finds cols 3,4
    CHECK(weights_perrow.probes.nnz() >= weights_global.probes.nnz());
}

// ── End-to-end: build_probes + synap_row_step grows real connections ─────────

// ── compact() / expand_headroom() ─────────────────────────────────────────────

TEST_CASE("compact() is lossless and shrinks reserved headroom", "[memory]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    std::vector<SIZE_TYPE> ptrs = {0, 3, 5, 6};
    std::vector<SIZE_TYPE> idx  = {0, 5, 12, 2, 9, 20};
    std::vector<float>     w    = {1.5f, -2.0f, 3.0f, 0.5f, -0.5f, 6.0f};
    std::vector<float>     imp  = {0.1f, -0.2f, 0.3f, 0.0f, -0.4f, 0.5f};
    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(3), std::size_t(25), std::size_t(4096), std::size_t(4096));

    const std::size_t bytes_before = dc.layout.total_alloc_bytes();
    const std::size_t elems_before = dc.layout.total_alloc_elems();

    auto compacted = compact<SIZE_TYPE, FP4BiPacked, COL_TYPE>(dc);

    CHECK(compacted.layout.total_alloc_bytes() <= bytes_before);
    CHECK(compacted.layout.total_alloc_elems() <= elems_before);
    REQUIRE(compacted.nnz() == dc.nnz());

    for (std::size_t r = 0; r < 3; ++r) {
        auto c1 = dc.row_cursor(r);
        auto c2 = compacted.row_cursor(r);
        std::size_t k = 0;
        while (!c1.at_end() && !c2.at_end()) {
            REQUIRE(c1.advance() == c2.advance());
            CHECK(ValueAccessor<FP4BiPacked>::get_w(dc.values, dc.layout.elem_start[r]+k) ==
                 ValueAccessor<FP4BiPacked>::get_w(compacted.values, compacted.layout.elem_start[r]+k));
            ++k;
        }
        REQUIRE(c1.at_end() == c2.at_end());
    }

    // Idempotent.
    auto compacted2 = compact<SIZE_TYPE, FP4BiPacked, COL_TYPE>(compacted);
    CHECK(compacted2.layout.total_alloc_bytes() == compacted.layout.total_alloc_bytes());
}

TEST_CASE("synap_row_step throws (not silently no-ops) when compact() removed all headroom",
         "[memory][synaptogenesis][regression]") {
    // Regression test for the exact failure mode reported in conversation:
    // training silently stops improving after compact(), with no error.
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    std::vector<SIZE_TYPE> ptrs = {0, 2, 3, 4};
    std::vector<SIZE_TYPE> idx  = {0, 2, 1, 3};
    std::vector<float>     w    = {1.5f, -2.0f, 0.5f, 3.0f};
    std::vector<float>     imp  = {0.0f, 0.0f, 0.0f, 0.0f};
    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(3), std::size_t(4), std::size_t(4096), std::size_t(4096));

    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights;
    weights.connections = compact<SIZE_TYPE, FP4BiPacked, COL_TYPE>(dc);
    weights.out_degree.assign(4, SIZE_TYPE(0));

    std::vector<float> input_accum = {5.0f, 0.1f, 3.0f};
    std::vector<float> grad_accum  = {0.1f, 4.0f, 0.1f, 0.1f};
    delta_csr_build_probes<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, input_accum.data(), grad_accum.data(), SIZE_TYPE(3));
    REQUIRE(weights.probes.nnz() > 0);

    std::size_t current_row = 0;
    REQUIRE_THROWS_AS(
        (delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights, current_row, 0.0f, SIZE_TYPE(10))),
        std::runtime_error);
}

TEST_CASE("expand_headroom() restores growth after compact(), synaptogenesis works again",
         "[memory][synaptogenesis][regression]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    std::vector<SIZE_TYPE> ptrs = {0, 2, 3, 4};
    std::vector<SIZE_TYPE> idx  = {0, 2, 1, 3};
    std::vector<float>     w    = {1.5f, -2.0f, 0.5f, 3.0f};
    std::vector<float>     imp  = {0.0f, 0.0f, 0.0f, 0.0f};
    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(3), std::size_t(4), std::size_t(4096), std::size_t(4096));

    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights;
    weights.connections = compact<SIZE_TYPE, FP4BiPacked, COL_TYPE>(dc);
    weights.out_degree.assign(4, SIZE_TYPE(0));

    weights.connections = expand_headroom<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights.connections, 0.5f);

    // Lossless.
    auto c1 = dc.row_cursor(0);
    auto c2 = weights.connections.row_cursor(0);
    REQUIRE(c1.advance() == c2.advance());
    REQUIRE(c1.advance() == c2.advance());

    std::vector<float> input_accum = {5.0f, 0.1f, 3.0f};
    std::vector<float> grad_accum  = {0.1f, 4.0f, 0.1f, 0.1f};
    delta_csr_build_probes<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, input_accum.data(), grad_accum.data(), SIZE_TYPE(3));

    const std::size_t nnz_before = weights.connections.nnz();
    std::size_t current_row = 0;
    CHECK_NOTHROW(
        (delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(weights, current_row, 0.0f, SIZE_TYPE(10))));
    CHECK(weights.connections.nnz() > nnz_before);
}

TEST_CASE("synaptogenesis end-to-end: probes generated and applied grow nnz",
         "[synaptogenesis][integration]") {
    using SIZE_TYPE = int;
    using COL_TYPE  = uint32_t;

    std::vector<SIZE_TYPE> ptrs = {0, 2, 3, 4};
    std::vector<SIZE_TYPE> idx  = {0, 2, 1, 3};
    std::vector<float>     w    = {1.5f, -2.0f, 0.5f, 3.0f};
    std::vector<float>     imp  = {0.0f, 0.0f, 0.0f, 0.0f};
    auto dc = delta_csr_from_absolute<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        ptrs, idx, w, imp, std::size_t(3), std::size_t(4), std::size_t(64), std::size_t(64));

    SparseLinearWeightsDelta<SIZE_TYPE, FP4BiPacked, COL_TYPE> weights;
    weights.connections = dc;
    weights.out_degree.assign(4, SIZE_TYPE(0));

    const std::size_t nnz_before = weights.connections.nnz();

    std::vector<float> input_accum = {5.0f, 0.1f, 3.0f};
    std::vector<float> grad_accum  = {0.1f, 4.0f, 0.1f, 0.1f};
    delta_csr_build_probes<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
        weights, input_accum.data(), grad_accum.data(), SIZE_TYPE(3));

    REQUIRE(weights.probes.nnz() > 0);

    // Not every row necessarily has enough reserved headroom to accommodate
    // whatever this step's merge produces in one call (see the two
    // regression tests above) -- that's expected, not a bug here. This
    // test's actual intent is "confirm real growth happens somewhere over
    // several steps," not "every row succeeds in one step" -- tolerate a
    // per-row throw and move on, same as a real caller should.
    std::size_t current_row = 0;
    for (int i = 0; i < 3; ++i) {
        try {
            delta_csr_synap_row_step<SIZE_TYPE, FP4BiPacked, COL_TYPE>(
                weights, current_row, 0.0f, SIZE_TYPE(10));
        } catch (const std::runtime_error&) {
            // Expected for rows without enough headroom for this much
            // growth in one step -- continue to the next row.
        }
    }

    // Strict > , not >= : this must be a real, verified size increase (see
    // conversation for why a >= here would have masked the nnz_delta bug).
    CHECK(weights.connections.nnz() > nnz_before);
}
