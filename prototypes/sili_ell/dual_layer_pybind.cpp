// dual_layer_pybind.cpp
// ===========================================================================
// Minimal, prototype-stage pybind11 binding for the dual-matrix design
// (y = disldo_forward(A) + block4_forward(B)) so sili_peridot can benchmark
// it against real converted-model weights. Deliberately separate from
// sili._cpu (the production extension) -- this is still a speed prototype
// on a feature branch, not yet merged/stabilized. Forward-only: no
// backward/training here yet (see conversation -- that's the next step,
// this unblocks the requested per-layer speedup benchmark first).
// ===========================================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../../sili/lib/headers/linear_disldo.hpp"
#include "../../sili/lib/headers/dense_block4.hpp"

namespace py = pybind11;

struct DualLayer {
    using S = uint32_t;
    SparseLinearWeightsDelta<int, FP4BiPacked, uint32_t> a_weights;   // leftover, disldo path
    Block4Weights b_weights;                                          // locally-dense, block4 path
    uint32_t n_in, n_out;
    int num_cpus;
    bool has_a = false;

    py::array_t<float> forward(py::array_t<float> x) {
        auto xbuf = x.request();
        const float* xp = static_cast<const float*>(xbuf.ptr);
        std::vector<float> y(n_out, 0.0f);

        if (has_a) {
            disldo_forward<int, FP4BiPacked, uint32_t>(
                xp, 1, n_in, a_weights, y.data(), 0.0f, num_cpus);
        }
        block4_forward(b_weights, xp, y.data(), num_cpus);

        py::array_t<float> result(n_out);
        std::memcpy(result.request().ptr, y.data(), n_out * sizeof(float));
        return result;
    }

    std::size_t a_nnz() const { return has_a ? a_weights.connections.nnz() : 0; }
    std::size_t b_nnz() const {
        std::size_t n = 0;
        for (uint8_t b : b_weights.block_data) n += ((b & 0xFu) != 0);
        return n;
    }
    std::size_t b_capacity_slots() const { return b_weights.block_data.size(); }
    std::size_t a_index_bytes() const { return has_a ? a_weights.connections.indices_buf.size() : 0; }
    std::size_t b_index_bytes() const {
        return b_weights.block_col_bytes.size() + b_weights.block_data.size();  // block deltas + 1 byte/slot (weight+imp packed, no separate index)
    }
};

// Build a DualLayer from a CSR in the NATURAL orientation (row = output,
// col = input -- matching PyTorch's own nn.Linear.weight layout,
// [out_features, in_features], and block4_forward's own internal
// convention: y[r*4+i] is a fixed WRITE indexed by block-row, x[c*4+j] is
// the READ indexed by block-col -- row must be output for that write to
// land in-bounds). disldo wants the OPPOSITE orientation (its CSR is
// "CSR of W^T", indexed by INPUT row -- see conversation's orientation
// note, found the hard way earlier this session), so the leftover/A path
// is built via an explicit transpose below, not by reusing the input
// ptrs/idx directly.
//
// A REAL bug this fixes (found via AddressSanitizer, not by inspection --
// see conversation): an earlier version passed (n_in, n_out) straight
// into split_for_block4 assuming disldo's row=input orientation, while
// block4_forward's write (`y[r*4+i]`) needs row=output. With n_in != n_out
// this silently wrote past y's real allocation (heap-buffer-overflow,
// n_out=32 but n_block_rows computed as n_in/4=16) -- corrupted heap
// metadata that only crashed later, at a much-delayed, confusing point
// (interpreter shutdown), exactly the kind of bug ASan exists for.
//
// Returns a unique_ptr, constructed via `new` and never copied/moved after:
// FP4BiPacked holds a SELF-REFERENTIAL pointer (_lanes[i]._dp = &this->_data,
// see fp4quant.hpp) that's only safe across a move if every enclosing type
// in the chain (SparseLinearWeightsDelta -> DeltaCSRWeights -> FP4BiPacked)
// correctly propagates move semantics -- returning DualLayer BY VALUE across
// the pybind11 boundary crashed with "free(): invalid size" separately from
// the bug above. Constructing once at its final heap address sidesteps the
// whole question rather than chasing the exact failure point further.
std::unique_ptr<DualLayer> build_dual_layer(
    py::array_t<uint32_t> ptrs, py::array_t<uint32_t> idx,
    py::array_t<float> weights, py::array_t<float> importance,
    uint32_t n_in, uint32_t n_out, float min_fill_frac, int num_cpus)
{
    auto layer_ptr = std::make_unique<DualLayer>();
    DualLayer& layer = *layer_ptr;
    layer.n_in = n_in; layer.n_out = n_out; layer.num_cpus = num_cpus;

    auto pbuf = ptrs.request(); auto ibuf = idx.request();
    auto wbuf = weights.request(); auto impbuf = importance.request();
    std::vector<uint32_t> vptrs((uint32_t*)pbuf.ptr, (uint32_t*)pbuf.ptr + pbuf.shape[0]);
    std::vector<uint32_t> vidx((uint32_t*)ibuf.ptr, (uint32_t*)ibuf.ptr + ibuf.shape[0]);
    std::vector<float> vw((float*)wbuf.ptr, (float*)wbuf.ptr + wbuf.shape[0]);
    std::vector<float> vimp((float*)impbuf.ptr, (float*)impbuf.ptr + impbuf.shape[0]);

    // Input ptrs/idx: rows = n_out (natural orientation), "columns" = n_in.
    uint32_t n_in_pad = ((n_in + 3) / 4) * 4;
    uint32_t n_out_pad = ((n_out + 3) / 4) * 4;

    auto split = split_for_block4(n_out_pad, n_in_pad, vptrs, vidx, vw, vimp, min_fill_frac);
    layer.b_weights = std::move(split.block4);

    if (!split.leftover_rc.empty()) {
        // Transpose leftover (row=output, col=input) -> disldo's own
        // orientation (row=input, col=output), row-sorted per input row.
        std::vector<std::vector<std::pair<uint32_t, std::size_t>>> by_input_row(n_in_pad);
        for (std::size_t k = 0; k < split.leftover_rc.size(); ++k) {
            auto [out_row, in_col] = split.leftover_rc[k];
            by_input_row[in_col].push_back({out_row, k});   // key: input row, value: (output col, data idx)
        }
        for (auto& r : by_input_row) std::sort(r.begin(), r.end());

        std::vector<int> a_ptrs(n_in_pad + 1, 0);
        std::vector<int> a_idx;
        std::vector<float> a_w, a_imp;
        a_idx.reserve(split.leftover_rc.size());
        for (uint32_t in_row = 0; in_row < n_in_pad; ++in_row) {
            for (auto& [out_col, k] : by_input_row[in_row]) {
                a_idx.push_back(int(out_col));
                a_w.push_back(split.leftover_w[k]);
                a_imp.push_back(split.leftover_imp[k]);
            }
            a_ptrs[in_row + 1] = int(a_idx.size());
        }
        // Per-row value_scale calibration, matching disldo's own default
        // (model/sili_block.py's _build_step_layer_from_arrays in
        // sili_peridot) -- a REAL gap found by A/B-testing quantization
        // error against a standalone, properly-calibrated disldo layer on
        // real weights: embed_tokens (the tensor with the largest leftover
        // fraction, 3.5%) showed 2.07x WORSE relative error than plain
        // disldo on identical weights before this fix, because leftover
        // entries were quantized raw/unscaled while disldo's own
        // comparison layer got calibrated. Without this, the dual-matrix
        // design's A path is a strictly worse disldo than disldo itself,
        // which isn't a fair test of the design.
        std::vector<float> a_row_scale(n_in_pad, 1.0f);
        for (uint32_t in_row = 0; in_row < n_in_pad; ++in_row) {
            float max_abs = 0.0f;
            for (int e = a_ptrs[in_row]; e < a_ptrs[in_row + 1]; ++e)
                max_abs = std::max(max_abs, std::abs(a_w[std::size_t(e)]));
            if (max_abs > 0.0f) {
                a_row_scale[in_row] = max_abs / 6.0f;
                for (int e = a_ptrs[in_row]; e < a_ptrs[in_row + 1]; ++e)
                    a_w[std::size_t(e)] /= a_row_scale[in_row];
            }
        }
        layer.a_weights.connections = delta_csr_from_absolute<int, FP4BiPacked, uint32_t>(
            a_ptrs, a_idx, a_w, a_imp, n_in_pad, n_out_pad,
            a_idx.size() * 8 + 4096, a_idx.size() + 64, 0.2f);
        layer.a_weights.recompute_stats();
        for (uint32_t in_row = 0; in_row < n_in_pad; ++in_row)
            if (a_row_scale[in_row] != 1.0f)
                layer.a_weights.set_value_scale_raw(in_row, a_row_scale[in_row]);
        layer.has_a = true;
    }
    return layer_ptr;
}

PYBIND11_MODULE(dual_layer_proto, m) {
    py::class_<DualLayer, std::unique_ptr<DualLayer>>(m, "DualLayer")
        .def("forward", &DualLayer::forward)
        .def("a_nnz", &DualLayer::a_nnz)
        .def("b_nnz", &DualLayer::b_nnz)
        .def("b_capacity_slots", &DualLayer::b_capacity_slots)
        .def("a_index_bytes", &DualLayer::a_index_bytes)
        .def("b_index_bytes", &DualLayer::b_index_bytes);
    m.def("build_dual_layer", &build_dual_layer,
          py::arg("ptrs"), py::arg("idx"), py::arg("weights"), py::arg("importance"),
          py::arg("n_in"), py::arg("n_out"), py::arg("min_fill_frac") = 0.10f,
          py::arg("num_cpus") = 4);
}
