#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <sys/types.h>
#include <vector>

#include "linear_sisldo.hpp"
#include "linear_disldo.hpp"
#include "csr.hpp"
#include "loss.hpp"
#include "fp4quant.hpp"

namespace py = pybind11;

// ── SISLDOLayer ───────────────────────────────────────────────────────────────
// Sparse Input, Sparse Linear, Dense Output layer.
//
// Lifetime:
//   weights owns all connection data via shared_ptr<vector>.
//   neuron_input_accum / neuron_grad_accum accumulate across training steps;
//   caller zeros them after each neurogenesis cycle via zero_accum().
//   output_buf holds the last forward output for the energy wrapper to read
//   without a copy — invalidated on the next forward() call.

class SISLDOLayerV {
public:
    using S = int;
    using V = float;

    SparseLinearWeightsV<S, V> weights;
    std::vector<V>            neuron_input_accum;
    std::vector<V>            neuron_grad_accum;
    std::vector<V>            output_buf;
    int                       num_cpus;

    SISLDOLayerV(S n_inputs, S n_outputs, S max_weights, int cpus = 4)
        : num_cpus(cpus)
    {
        weights.connections.rows    = n_inputs;
        weights.connections.cols    = n_outputs;
        weights.connections.ptrs[0] = std::make_shared<std::vector<S>>(n_inputs + 1, S(0));
        weights.connections.indices[0] = std::make_shared<std::vector<S>>();
        weights.connections.values[0]  = std::make_shared<std::vector<V>>();
        weights.connections.values[1]  = std::make_shared<std::vector<V>>();
        weights.probes.rows = n_inputs;
        weights.probes.cols = n_outputs;

        reserve_connections(weights.connections, max_weights);

        neuron_input_accum.assign(n_inputs,  V(0));
        neuron_grad_accum .assign(n_outputs, V(0));
        weights.out_degree.assign(n_outputs, S(0));
    }

    // ── Scalar properties ─────────────────────────────────────────────────────

    S n_inputs()  const { return weights.connections.rows; }
    S n_outputs() const { return weights.connections.cols; }
    S nnz()       const { return weights.connections.nnz(); }

    // ── Input construction ───────────────────────────────────────────────────

    CSRInput<S, V> numpy_to_sparse_input(
        py::array_t<S> ptrs,
        py::array_t<S> indices,
        py::array_t<V> values,
        S batch, S cols)
    {
        auto pb = ptrs.request(), ib = indices.request(), vb = values.request();
        const S nnz_in = (S)ib.size;
        CSRInput<S, V> csr;
        csr.rows       = batch;
        csr.cols       = cols;
        csr.ptrs[0]    = std::make_shared<std::vector<S>>((S*)pb.ptr, (S*)pb.ptr + batch + 1);
        csr.indices[0] = std::make_shared<std::vector<S>>((S*)ib.ptr, (S*)ib.ptr + nnz_in);
        csr.values[0]  = std::make_shared<std::vector<V>>((V*)vb.ptr, (V*)vb.ptr + nnz_in);
        return csr;
    }

    void load_weights(py::array_t<S> ptrs, py::array_t<S> indices,
                  py::array_t<V> vals, py::array_t<V> imp) {
        auto pb = ptrs.request(), ib = indices.request(),
            vb = vals.request(), impb = imp.request();
        const S rows = weights.connections.rows;
        const S cols = weights.connections.cols;
        weights = make_weights_v<S, V>(
            rows, cols,
            std::vector<S>((S*)pb.ptr,   (S*)pb.ptr   + pb.size),
            std::vector<S>((S*)ib.ptr,   (S*)ib.ptr   + ib.size),
            std::vector<V>((V*)vb.ptr,   (V*)vb.ptr   + vb.size),
            std::vector<V>((V*)impb.ptr, (V*)impb.ptr + impb.size));
    }

    // ── Forward (use module-level dense_to_csr to prepare input first) ─────────

    py::array_t<V> forward_sparse(
        py::array_t<S> ptrs, py::array_t<S> indices, py::array_t<V> values,
        S batch, V learning_rate)
    {
        output_buf.assign(batch * n_outputs(), V(0));
        auto input_csr = numpy_to_sparse_input(ptrs, indices, values, batch, n_inputs());
        sisldo_forward(input_csr, weights, output_buf.data(), learning_rate, num_cpus);

        return py::array_t<V>(
            {(py::ssize_t)batch, (py::ssize_t)n_outputs()},
            {(py::ssize_t)(n_outputs() * sizeof(V)), (py::ssize_t)sizeof(V)},
            output_buf.data(), py::cast(this));
    }

    // ── Backward ─────────────────────────────────────────────────────────────
    // dy:           dense [batch, n_outputs] — weight gradient kernel
    // dy_sparse_*:  sparse dy — input gradient kernel
    //               if all outputs are active, pass dy converted to a single CSR row

    py::array_t<V> backward(
        py::array_t<S> x_ptrs,
        py::array_t<S> x_indices,
        py::array_t<V> x_values,
        py::array_t<V> dy,
        py::array_t<S> dy_sparse_ptrs,
        py::array_t<S> dy_sparse_indices,
        py::array_t<V> dy_sparse_values,
        V learning_rate,
        S batch, S cols)
    {
        auto dybuf    = dy.request();

        std::vector<V> dx(batch * n_inputs(), V(0));
        auto input_csr    = numpy_to_sparse_input(x_ptrs, x_indices, x_values, batch, cols);
        auto out_grad_csr = numpy_to_sparse_input(
            dy_sparse_ptrs, dy_sparse_indices, dy_sparse_values, batch, n_outputs());

        sisldo_backward(
            input_csr, weights, out_grad_csr,
            dx.data(), (V*)dybuf.ptr,
            neuron_input_accum.data(), neuron_grad_accum.data(), learning_rate,
            num_cpus);

        py::array_t<V> result({(py::ssize_t)batch, (py::ssize_t)n_inputs()});
        std::copy(dx.begin(), dx.end(), (V*)result.request().ptr);
        return result;
    }

    // ── Optimization ─────────────────────────────────────────────────────────

    void decay_importance(V rate) {
        sisldo_decay_importance(weights, rate, num_cpus);
    }

    // ── Neurogenesis ──────────────────────────────────────────────────────────

    void build_probes(S k) {
        genesis_build_probes(
            weights,
            neuron_input_accum.data(), neuron_grad_accum.data(),
            n_inputs(), n_outputs(), k, num_cpus);
    }

    void optim_synaptogenesis(V learning_rate, V importance_beta, S max_weights) {
        sisldo_optim_synaptogenesis(
            weights, learning_rate, importance_beta, max_weights, num_cpus);
    }

    void zero_accum() {
        std::fill(neuron_input_accum.begin(), neuron_input_accum.end(), V(0));
        std::fill(neuron_grad_accum .begin(), neuron_grad_accum .end(), V(0));
    }

    // ── Zero-copy numpy views ─────────────────────────────────────────────────

    py::array_t<V> get_neuron_input_accum() {
        return py::array_t<V>({(py::ssize_t)n_inputs()},  {sizeof(V)},
                              neuron_input_accum.data(), py::cast(this));
    }
    py::array_t<V> get_neuron_grad_accum() {
        return py::array_t<V>({(py::ssize_t)n_outputs()}, {sizeof(V)},
                              neuron_grad_accum.data(), py::cast(this));
    }
    py::array_t<V> get_output_buf() {
        return py::array_t<V>({(py::ssize_t)output_buf.size()}, {sizeof(V)},
                              output_buf.data(), py::cast(this));
    }
    py::array_t<V> get_weights_vals() {
        return py::array_t<V>({(py::ssize_t)nnz()}, {sizeof(V)},
                              weights.connections.values[0]->data(), py::cast(this));
    }
    py::array_t<V> get_importance() {
        return py::array_t<V>({(py::ssize_t)nnz()}, {sizeof(V)},
                              weights.connections.values[1]->data(), py::cast(this));
    }
    py::array_t<S> get_indices() {
        return py::array_t<S>({(py::ssize_t)nnz()}, {sizeof(S)},
                              weights.connections.indices[0]->data(), py::cast(this));
    }
    py::array_t<S> get_ptrs() {
        return py::array_t<S>({(py::ssize_t)(n_inputs() + 1)}, {sizeof(S)},
                              weights.connections.ptrs[0]->data(), py::cast(this));
    }
};




// ─────────────────────────────────────────────────────────────────────────────
//  Module
// ─────────────────────────────────────────────────────────────────────────────


// ── SparseLinearLayer ───────────────────────────────────────────────────────────────
// Rewritten (see conversation) to use SparseLinearWeightsDelta<S,FP4BiPacked,
// COL_TYPE> -- the delta-CSR/generic-ValueAccessor generation, matching what
// disldo_forward/disldo_backward now require. Previously used
// SparseLinearWeights<S,V> (absolute CSR, no delta encoding, no FP4 packing
// despite the class existing alongside an FP4-packed SISLDOLayer) and had NO
// synaptogenesis methods at all -- build_probes/synap_row_step below are new.

class SparseLinearLayer {
public:
    using S = int;
    using V = float;
    using COL_TYPE = uint32_t;

    SparseLinearWeightsDelta<S, FP4BiPacked, COL_TYPE> weights;
    std::vector<V>            neuron_input_accum;
    std::vector<V>            neuron_grad_accum;
    std::vector<V>            output_buf;
    int                       num_cpus;

    // Last dense input — stored for backward.
    std::vector<V> _last_input;
    S              _last_batch = 0;
    S              _last_cols  = 0;

    SparseLinearLayer(S n_inputs, S n_outputs, S max_weights, int cpus = 4)
        : num_cpus(cpus)
    {
        std::vector<S> empty_ptrs(static_cast<std::size_t>(n_inputs) + 1, S(0));
        std::vector<S> empty_idx;
        std::vector<V> empty_w, empty_imp;
        // Budget is generous headroom (reserve, not allocate-and-init) --
        // 8 bytes/synapse covers worst-case ULEB128 (5) + value byte (1) + margin.
        weights.connections = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
            empty_ptrs, empty_idx, empty_w, empty_imp,
            static_cast<std::size_t>(n_inputs), static_cast<std::size_t>(n_outputs),
            static_cast<std::size_t>(max_weights) * 8 + 4096,
            static_cast<std::size_t>(max_weights) + 64);
        weights.probes.rows = n_inputs;
        weights.probes.cols = n_outputs;
        neuron_input_accum.assign(n_inputs,  V(0));
        neuron_grad_accum .assign(n_outputs, V(0));
        weights.out_degree.assign(n_outputs, S(0));
    }

    S n_inputs()  const { return static_cast<S>(weights.connections.layout.rows); }
    S n_outputs() const { return static_cast<S>(weights.connections.layout.cols); }
    S nnz()       const { return static_cast<S>(weights.connections.nnz()); }

    // ── Forward ───────────────────────────────────────────────────────────────

    py::array_t<V> forward(py::array_t<V> x, V learning_rate = 0.01) {
        auto xbuf     = x.request();
        _last_batch   = (xbuf.ndim == 2) ? (S)xbuf.shape[0] : 1;
        _last_cols    = (xbuf.ndim == 2) ? (S)xbuf.shape[1] : (S)xbuf.shape[0];

        const V* src  = (V*)xbuf.ptr;
        _last_input.assign(src, src + _last_batch * _last_cols);

        output_buf.assign(_last_batch * n_outputs(), V(0));
        disldo_forward<S, FP4BiPacked, COL_TYPE>(src, _last_batch, _last_cols, weights,
                       output_buf.data(), learning_rate, num_cpus);

        return py::array_t<V>(
            {(py::ssize_t)_last_batch, (py::ssize_t)n_outputs()},
            {(py::ssize_t)(n_outputs() * sizeof(V)), (py::ssize_t)sizeof(V)},
            output_buf.data(), py::cast(this));
    }

    // ── Backward ─────────────────────────────────────────────────────────────

    py::array_t<V> backward(py::array_t<V> dy, V learning_rate) {
        auto dybuf = dy.request();
        std::vector<V> dx(_last_batch * _last_cols, V(0));
        disldo_backward<S, FP4BiPacked, COL_TYPE>(
            _last_input.data(), _last_batch, _last_cols,
            (V*)dybuf.ptr, weights,
            dx.data(),
            neuron_input_accum.data(), neuron_grad_accum.data(),
            learning_rate,
            num_cpus);
        py::array_t<V> result({(py::ssize_t)_last_batch, (py::ssize_t)_last_cols});
        std::copy(dx.begin(), dx.end(), (V*)result.request().ptr);
        return result;
    }

    // ── Synaptogenesis (NEW — see class comment) ────────────────────────────────

    void build_probes(S k, bool per_row = false) {
        delta_csr_build_probes<S, FP4BiPacked, COL_TYPE>(
            weights, neuron_input_accum.data(), neuron_grad_accum.data(), k, per_row);
    }
    bool synap_row_step(S current_row, V importance_cutoff, S max_row_weights) {
        std::size_t row = static_cast<std::size_t>(current_row);
        return delta_csr_synap_row_step<S, FP4BiPacked, COL_TYPE>(
            weights, row, importance_cutoff, max_row_weights);
    }

    // Repack in place: every row occupies exactly its active bytes/elements,
    // zero inter-row blank space (see compact() in sparse_struct.hpp for the
    // full rationale). Call before saving/measuring a freshly converted or
    // long-since-pruned model. Zeroes growth headroom -- call
    // weights.connections.reserve_indices()/reserve_values() again after if
    // this model is about to resume training rather than just be deployed.
    void compact() {
        weights.connections = ::compact<S, FP4BiPacked, COL_TYPE>(weights.connections);
    }

    void zero_accum() {
        std::fill(neuron_input_accum.begin(), neuron_input_accum.end(), V(0));
        std::fill(neuron_grad_accum .begin(), neuron_grad_accum .end(), V(0));
    }

    // ptrs/indices/vals: standard absolute CSR + true float weights (NOT
    // pre-packed FP4 bytes — delta_csr_from_absolute quantizes internally,
    // unlike the old make_weights(FP4BiPacked(raw_bytes)) contract).
    void load_weights(py::array_t<S> ptrs, py::array_t<S> indices,
                      py::array_t<V> vals) {
        auto pb=ptrs.request(), ib=indices.request(), vb=vals.request();
        const std::size_t rows = weights.connections.layout.rows;
        const std::size_t cols = weights.connections.layout.cols;
        std::vector<S> p((S*)pb.ptr, (S*)pb.ptr + pb.size);
        std::vector<S> idx((S*)ib.ptr, (S*)ib.ptr + ib.size);
        std::vector<V> w((V*)vb.ptr, (V*)vb.ptr + vb.size);
        std::vector<V> imp(w.size(), V(0));
        weights.connections = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
            p, idx, w, imp, rows, cols,
            idx.size() * 8 + 4096, idx.size() + 64);
    }

    // ── Zero-copy numpy views ────────────────────────────────────────────────
    py::array_t<V> get_neuron_input_accum() {
        return py::array_t<V>({(py::ssize_t)n_inputs()}, {sizeof(V)},
                              neuron_input_accum.data(), py::cast(this)); }
    py::array_t<V> get_neuron_grad_accum() {
        return py::array_t<V>({(py::ssize_t)n_outputs()}, {sizeof(V)},
                              neuron_grad_accum.data(), py::cast(this)); }

    // NOTE: no longer zero-copy (delta-CSR has no plain float array to view
    // directly) — materializes absolute CSR + float weights/importance via
    // delta_csr_to_absolute on each call. O(nnz), not O(1) like before.
    py::array_t<V> get_weights_vals() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_to_absolute<S, FP4BiPacked, COL_TYPE>(weights.connections, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)ow.size());
        std::copy(ow.begin(), ow.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<V> get_importance() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_to_absolute<S, FP4BiPacked, COL_TYPE>(weights.connections, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)oimp.size());
        std::copy(oimp.begin(), oimp.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_indices() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_to_absolute<S, FP4BiPacked, COL_TYPE>(weights.connections, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)oi.size());
        std::copy(oi.begin(), oi.end(), (S*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_ptrs() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_to_absolute<S, FP4BiPacked, COL_TYPE>(weights.connections, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)op.size());
        std::copy(op.begin(), op.end(), (S*)result.request().ptr);
        return result;
    }
};

// ── DISLDOLayerV ──────────────────────────────────────────────────────────────
// Same rewrite as SparseLinearLayer, VALUES_TYPE=DeltaCSRBiValues<float> instead of
// FP4BiPacked — the exact same disldo_forward/backward/build_probes/
// synap_row_step functions, generic via ValueAccessor, no separate
// implementation needed. This is the concrete realization of "run_tests_4_bit
// and run_tests_32_bit should use the same functions" (see conversation).

class DISLDOLayerV {
public:
    using S = int;
    using V = float;
    using COL_TYPE = uint32_t;
    using VT = DeltaCSRBiValues<V>;

    SparseLinearWeightsDelta<S, VT, COL_TYPE> weights;
    std::vector<V>            neuron_input_accum;
    std::vector<V>            neuron_grad_accum;
    std::vector<V>            output_buf;
    int                       num_cpus;

    std::vector<V> _last_input;
    S              _last_batch = 0;
    S              _last_cols  = 0;

    DISLDOLayerV(S n_inputs, S n_outputs, S max_weights, int cpus = 4)
        : num_cpus(cpus)
    {
        std::vector<S> empty_ptrs(static_cast<std::size_t>(n_inputs) + 1, S(0));
        std::vector<S> empty_idx;
        std::vector<V> empty_w, empty_imp;
        weights.connections = delta_csr_from_absolute<S, VT, COL_TYPE>(
            empty_ptrs, empty_idx, empty_w, empty_imp,
            static_cast<std::size_t>(n_inputs), static_cast<std::size_t>(n_outputs),
            static_cast<std::size_t>(max_weights) * 8 + 4096,
            static_cast<std::size_t>(max_weights) + 64);
        weights.probes.rows = n_inputs;
        weights.probes.cols = n_outputs;
        neuron_input_accum.assign(n_inputs,  V(0));
        neuron_grad_accum .assign(n_outputs, V(0));
        weights.out_degree.assign(n_outputs, S(0));
    }

    S n_inputs()  const { return static_cast<S>(weights.connections.layout.rows); }
    S n_outputs() const { return static_cast<S>(weights.connections.layout.cols); }
    S nnz()       const { return static_cast<S>(weights.connections.nnz()); }

    py::array_t<V> forward(py::array_t<V> x, V learning_rate = 0.01) {
        auto xbuf     = x.request();
        _last_batch   = (xbuf.ndim == 2) ? (S)xbuf.shape[0] : 1;
        _last_cols    = (xbuf.ndim == 2) ? (S)xbuf.shape[1] : (S)xbuf.shape[0];

        const V* src  = (V*)xbuf.ptr;
        _last_input.assign(src, src + _last_batch * _last_cols);

        output_buf.assign(_last_batch * n_outputs(), V(0));
        disldo_forward<S, VT, COL_TYPE>(src, _last_batch, _last_cols, weights,
                       output_buf.data(), learning_rate, num_cpus);

        return py::array_t<V>(
            {(py::ssize_t)_last_batch, (py::ssize_t)n_outputs()},
            {(py::ssize_t)(n_outputs() * sizeof(V)), (py::ssize_t)sizeof(V)},
            output_buf.data(), py::cast(this));
    }

    py::array_t<V> backward(py::array_t<V> dy, V learning_rate) {
        auto dybuf = dy.request();
        std::vector<V> dx(_last_batch * _last_cols, V(0));
        disldo_backward<S, VT, COL_TYPE>(
            _last_input.data(), _last_batch, _last_cols,
            (V*)dybuf.ptr, weights,
            dx.data(),
            neuron_input_accum.data(), neuron_grad_accum.data(),
            learning_rate,
            num_cpus);
        py::array_t<V> result({(py::ssize_t)_last_batch, (py::ssize_t)_last_cols});
        std::copy(dx.begin(), dx.end(), (V*)result.request().ptr);
        return result;
    }

    void build_probes(S k, bool per_row = false) {
        delta_csr_build_probes<S, VT, COL_TYPE>(
            weights, neuron_input_accum.data(), neuron_grad_accum.data(), k, per_row);
    }
    bool synap_row_step(S current_row, V importance_cutoff, S max_row_weights) {
        std::size_t row = static_cast<std::size_t>(current_row);
        return delta_csr_synap_row_step<S, VT, COL_TYPE>(
            weights, row, importance_cutoff, max_row_weights);
    }
    void zero_accum() {
        std::fill(neuron_input_accum.begin(), neuron_input_accum.end(), V(0));
        std::fill(neuron_grad_accum .begin(), neuron_grad_accum .end(), V(0));
    }
    void load_weights(py::array_t<S> ptrs, py::array_t<S> indices,
                      py::array_t<V> vals,  py::array_t<V> imp) {
        auto pb=ptrs.request(), ib=indices.request(),
             vb=vals.request(), impb=imp.request();
        const std::size_t rows = weights.connections.layout.rows;
        const std::size_t cols = weights.connections.layout.cols;
        std::vector<S> p((S*)pb.ptr, (S*)pb.ptr + pb.size);
        std::vector<S> idx((S*)ib.ptr, (S*)ib.ptr + ib.size);
        std::vector<V> w((V*)vb.ptr, (V*)vb.ptr + vb.size);
        std::vector<V> imp_v((V*)impb.ptr, (V*)impb.ptr + impb.size);
        weights.connections = delta_csr_from_absolute<S, VT, COL_TYPE>(
            p, idx, w, imp_v, rows, cols,
            idx.size() * 8 + 4096, idx.size() + 64);
    }

    // ── Zero-copy numpy views ────────────────────────────────────────────────
    py::array_t<V> get_neuron_input_accum() {
        return py::array_t<V>({(py::ssize_t)n_inputs()}, {sizeof(V)},
                              neuron_input_accum.data(), py::cast(this)); }
    py::array_t<V> get_neuron_grad_accum() {
        return py::array_t<V>({(py::ssize_t)n_outputs()}, {sizeof(V)},
                              neuron_grad_accum.data(), py::cast(this)); }
    py::array_t<V> get_weights_vals() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_to_absolute<S, VT, COL_TYPE>(weights.connections, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)ow.size());
        std::copy(ow.begin(), ow.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<V> get_importance() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_to_absolute<S, VT, COL_TYPE>(weights.connections, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)oimp.size());
        std::copy(oimp.begin(), oimp.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_indices() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_to_absolute<S, VT, COL_TYPE>(weights.connections, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)oi.size());
        std::copy(oi.begin(), oi.end(), (S*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_ptrs() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_to_absolute<S, VT, COL_TYPE>(weights.connections, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)op.size());
        std::copy(op.begin(), op.end(), (S*)result.request().ptr);
        return result;
    }
};

PYBIND11_MODULE(_cpu, m)
{

    // ── SparseLinearLayer ───────────────────────────────────────────────────────────

    py::class_<SparseLinearLayer>(m, "SparseLinearLayer")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("forward",              &SparseLinearLayer::forward,
             py::arg("x"), py::arg("learning_rate") = 0.01f)
        .def("backward",             &SparseLinearLayer::backward,
             py::arg("dy"), py::arg("learning_rate"))
        .def("build_probes",         &SparseLinearLayer::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayer::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"))
        .def("compact",              &SparseLinearLayer::compact,
             "Repack in place: every row occupies exactly its active bytes/elements,\n"
             "zero inter-row blank space. Call before saving/measuring a freshly\n"
             "converted model. Zeroes growth headroom.")
        .def("zero_accum",           &SparseLinearLayer::zero_accum)
        .def_property_readonly("neuron_input_accum", &SparseLinearLayer::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &SparseLinearLayer::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &SparseLinearLayer::get_weights_vals)
        .def_property_readonly("importance",         &SparseLinearLayer::get_importance)
        .def_property_readonly("indices",            &SparseLinearLayer::get_indices)
        .def_property_readonly("ptrs",               &SparseLinearLayer::get_ptrs)
        .def("load_weights",        &SparseLinearLayer::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"))
        .def_property_readonly("out_degree", [](const SparseLinearLayer& self) {
            return py::array_t<SparseLinearLayer::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(SparseLinearLayer::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &SparseLinearLayer::num_cpus)
        .def_property_readonly("n_inputs",  &SparseLinearLayer::n_inputs)
        .def_property_readonly("n_outputs", &SparseLinearLayer::n_outputs)
        .def_property_readonly("nnz",       &SparseLinearLayer::nnz);

    // DISLDOLayerV: identical API surface to SparseLinearLayer, DeltaCSRBiValues<float>
    // (32-bit) instead of FP4BiPacked -- same disldo_forward/backward/
    // build_probes/synap_row_step functions, generic via ValueAccessor.
    // Was never registered with pybind at all before this (see conversation).
    py::class_<DISLDOLayerV>(m, "DISLDOLayerV")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("forward",              &DISLDOLayerV::forward,
             py::arg("x"), py::arg("learning_rate") = 0.01f)
        .def("backward",             &DISLDOLayerV::backward,
             py::arg("dy"), py::arg("learning_rate"))
        .def("build_probes",         &DISLDOLayerV::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &DISLDOLayerV::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"))
        .def("zero_accum",           &DISLDOLayerV::zero_accum)
        .def_property_readonly("neuron_input_accum", &DISLDOLayerV::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &DISLDOLayerV::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &DISLDOLayerV::get_weights_vals)
        .def_property_readonly("importance",         &DISLDOLayerV::get_importance)
        .def_property_readonly("indices",            &DISLDOLayerV::get_indices)
        .def_property_readonly("ptrs",               &DISLDOLayerV::get_ptrs)
        .def("load_weights",        &DISLDOLayerV::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"), py::arg("importance"))
        .def_property_readonly("out_degree", [](const DISLDOLayerV& self) {
            return py::array_t<DISLDOLayerV::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(DISLDOLayerV::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &DISLDOLayerV::num_cpus)
        .def_property_readonly("n_inputs",  &DISLDOLayerV::n_inputs)
        .def_property_readonly("n_outputs", &DISLDOLayerV::n_outputs)
        .def_property_readonly("nnz",       &DISLDOLayerV::nnz);

    // ── CSR construction utilities ────────────────────────────────────────────
    //
    // dense_to_csr: convert dense float32 array to CSR, keeping only |v| > threshold.
    // This is the correct way to sparsify activations for SISLDO — NOT a dense-to-dense
    // roundtrip. Use this on energy dynamics output before passing to forward_sparse.

    m.def("dense_to_csr",
        [](py::array_t<float> x, float threshold) -> py::tuple {
            auto buf   = x.request();
            int batch  = (buf.ndim == 2) ? (int)buf.shape[0] : 1;
            int cols   = (buf.ndim == 2) ? (int)buf.shape[1] : (int)buf.shape[0];
            float* src = (float*)buf.ptr;

            std::vector<int>   ptrs(batch + 1, 0);
            std::vector<int>   indices;
            std::vector<float> values;
            indices.reserve(batch * cols / 8);
            values .reserve(batch * cols / 8);

            for (int b = 0; b < batch; ++b) {
                ptrs[b] = (int)indices.size();
                for (int c = 0; c < cols; ++c) {
                    float v = src[b * cols + c];
                    if (v > threshold || v < -threshold) {
                        indices.push_back(c);
                        values .push_back(v);
                    }
                }
            }
            ptrs[batch] = (int)indices.size();

            py::array_t<int>   out_ptrs   ({batch + 1});
            py::array_t<int>   out_indices({(py::ssize_t)indices.size()});
            py::array_t<float> out_values ({(py::ssize_t)values .size()});

            std::copy(ptrs.begin(),    ptrs.end(),
                      (int*)  out_ptrs   .request().ptr);
            std::copy(indices.begin(), indices.end(),
                      (int*)  out_indices.request().ptr);
            std::copy(values .begin(), values .end(),
                      (float*)out_values .request().ptr);

            return py::make_tuple(out_ptrs, out_indices, out_values);
        },
        py::arg("x"), py::arg("threshold") = 1e-4f,
        "Convert dense float32 [batch,cols] or [cols] to (ptrs,indices,values) CSR.\n"
        "Keeps only entries where |v| > threshold. Use this to sparsify activations\n"
        "before passing to SISLDOLayer.forward_sparse().");

    m.def("dense_to_top_k_csr",
        [](py::array_t<float> x, int k, int num_threads) -> py::tuple {
            auto buf = x.request();
            int rows = (buf.ndim == 2) ? buf.shape[0] : 1;
            int cols = (buf.ndim == 2) ? buf.shape[1] : buf.shape[0];
            float* src = (float*)buf.ptr;

            // Call your optimized exact-k function
            auto csr = top_k_csr<int, float>(src, rows, cols, k, num_threads);

            // The exact number of non-zeros returned
            int nnz = csr.indices[0]->size();

            py::array_t<int>   out_ptrs({(py::ssize_t)(rows + 1)});
            py::array_t<int>   out_indices({(py::ssize_t)nnz});
            py::array_t<float> out_values({(py::ssize_t)nnz});

            std::copy(csr.ptrs[0]->begin(), csr.ptrs[0]->end(), (int*)out_ptrs.request().ptr);
            std::copy(csr.indices[0]->begin(), csr.indices[0]->end(), (int*)out_indices.request().ptr);
            std::copy(csr.values[0]->begin(), csr.values[0]->end(), (float*)out_values.request().ptr);

            return py::make_tuple(out_ptrs, out_indices, out_values);
        },
        py::arg("x"), py::arg("k"), py::arg("num_threads") = 4,
        "Exact top-k sparsity conversion for forward and backward passes."
    );

    // ── make_csr_input ────────────────────────────────────────────────────────
    m.def("make_csr_input",
        [](int rows, int cols,
           py::array_t<int>   ptrs,
           py::array_t<int>   indices,
           py::array_t<float> values) {
            auto pb = ptrs.request(), ib = indices.request(), vb = values.request();
            return make_csr_input<int, float>(
                rows, cols,
                std::vector<int>  ((int*)  pb.ptr, (int*)  pb.ptr + pb.size),
                std::vector<int>  ((int*)  ib.ptr, (int*)  ib.ptr + ib.size),
                std::vector<float>((float*)vb.ptr, (float*)vb.ptr + vb.size));
        },
        py::arg("rows"), py::arg("cols"),
        py::arg("ptrs"), py::arg("indices"), py::arg("values"));

    m.def("make_weights",
        [](int rows, int cols,
           py::array_t<int>   ptrs,
           py::array_t<int>   indices,
           py::array_t<float> values,
           py::array_t<float> grads,
           py::array_t<float> importance) {
            auto pb   = ptrs.request(),   ib  = indices.request();
            auto vb   = values.request(), gb  = grads.request();
            auto impb = importance.request();
            return make_weights<int, float>(
                rows, cols,
                std::vector<int>  ((int*)  pb.ptr,   (int*)  pb.ptr   + pb.size),
                std::vector<int>  ((int*)  ib.ptr,   (int*)  ib.ptr   + ib.size),
                std::vector<float>((float*)vb.ptr,   (float*)vb.ptr   + vb.size),
                std::vector<float>((float*)gb.ptr,   (float*)gb.ptr   + gb.size),
                std::vector<float>((float*)impb.ptr, (float*)impb.ptr + impb.size));
        },
        py::arg("rows"), py::arg("cols"),
        py::arg("ptrs"), py::arg("indices"),
        py::arg("values"), py::arg("grads"), py::arg("importance"));

    // ── Loss functions ────────────────────────────────────────────────────────

    m.def("mse_loss",
        [](py::array_t<float> output, py::array_t<float> desired) {
            auto ob = output.request(), db = desired.request();
            if (ob.size != db.size) throw std::runtime_error("mse_loss: size mismatch");
            return mse_loss((float*)ob.ptr, (float*)db.ptr, (size_t)ob.size);
        },
        py::arg("output"), py::arg("desired"),
        "(1/n) * sum((desired - output)^2)");

    m.def("mse_grad",
        [](py::array_t<float> output, py::array_t<float> desired) {
            auto ob = output.request(), db = desired.request();
            if (ob.size != db.size) throw std::runtime_error("mse_grad: size mismatch");
            py::array_t<float> grad((py::ssize_t)ob.size);
            mse_grad((float*)ob.ptr, (float*)db.ptr,
                     (float*)grad.request().ptr, (size_t)ob.size);
            return grad;
        },
        py::arg("output"), py::arg("desired"),
        "-2 * (desired - output) / n");

    m.def("mse_loss_parallel",
        [](py::array_t<float> output, py::array_t<float> desired, int num_cpus) {
            auto ob = output.request(), db = desired.request();
            if (ob.size != db.size) throw std::runtime_error("mse_loss_parallel: size mismatch");
            return mse_loss_parallel((float*)ob.ptr, (float*)db.ptr, (size_t)ob.size, num_cpus);
        },
        py::arg("output"), py::arg("desired"), py::arg("num_cpus") = 4);

    m.def("mse_grad_parallel",
        [](py::array_t<float> output, py::array_t<float> desired, int num_cpus) {
            auto ob = output.request(), db = desired.request();
            if (ob.size != db.size) throw std::runtime_error("mse_grad_parallel: size mismatch");
            py::array_t<float> grad((py::ssize_t)ob.size);
            mse_grad_parallel((float*)ob.ptr, (float*)db.ptr,
                              (float*)grad.request().ptr, (size_t)ob.size, num_cpus);
            return grad;
        },
        py::arg("output"), py::arg("desired"), py::arg("num_cpus") = 4);
}