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
#include "hoyer_sparsify.hpp"
#include "fp4quant.hpp"
#include "attention.hpp"

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
// COL_TYPE> -- the delta-CSR/generic-ValueAccessor generation. Previously
// (as DISLDOLayer) used SparseLinearWeights<S,V> (absolute CSR, no delta
// encoding, no FP4 packing) and had NO synaptogenesis at all --
// build_probes/synap_row_step were added along with the rewiring.
//
// CORRECTION (see conversation): renaming DISLDOLayer -> SparseLinearLayer
// and dropping the old SISLDOLayer (non-V) together left this class
// DISLDO-only (dense input) -- SISLDO's sparse-input forward/backward
// (genuinely different code, not just a naming variant: different kernel
// structure, and legitimately needed when input activations really are
// sparse, e.g. after a top-k/threshold step run BETWEEN layers -- that
// sparsification is intentionally a separate operation, not something this
// class decides for itself) had no surviving implementation anywhere in
// this repo. forward_sparse uses delta_csr_forward (already existed in
// this file, generic over VALUES_TYPE); backward_sparse uses
// delta_csr_backward_sparse_grad (dense input, sparse gradient -- see
// forward_sparse/backward_sparse comment below for why these are NOT
// mirror images of each other). Both verified correct against
// hand-computed references before wiring up -- see conversation.

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
    // Budget established at construction -- used by load_weights to avoid
    // allocating a smaller limit that would then be exceeded by the per-row
    // headroom calculation inside delta_csr_from_absolute, which was
    // corrupting the heap (load_weights was passing idx.size()*8+4096 which
    // could be SMALLER than the actual bytes written by indices_buf.assign).
    std::size_t    _idx_budget_bytes = 4096;
    std::size_t    _val_budget_nnz   = 64;

    SparseLinearLayer(S n_inputs, S n_outputs, S max_weights, int cpus = 4)
        : num_cpus(cpus),
          _idx_budget_bytes(static_cast<std::size_t>(max_weights) * 8 + 4096),
          _val_budget_nnz  (static_cast<std::size_t>(max_weights) + 64)
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
        weights.recompute_stats();
        weights.probes.rows = n_inputs;
        weights.probes.cols = n_outputs;
        neuron_input_accum.assign(n_inputs,  V(0));
        neuron_grad_accum .assign(n_outputs, V(0));
        weights.out_degree.assign(n_outputs, S(0));
    }

    S n_inputs()  const { return static_cast<S>(weights.connections.layout.rows); }
    S n_outputs() const { return static_cast<S>(weights.connections.layout.cols); }
    S nnz()       const { return static_cast<S>(weights.connections.nnz()); }

    // ── Forward (dense input — DISLDO) ──────────────────────────────────────────

    py::array_t<V> forward_dense(py::array_t<V> x, V learning_rate = 0.01) {
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

    // ── Backward (dense input — DISLDO) ─────────────────────────────────────────

    py::array_t<V> backward_dense(py::array_t<V> dy, V learning_rate, bool lr_per_row_nnz = false) {
        auto dybuf = dy.request();
        std::vector<V> dx(_last_batch * _last_cols, V(0));
        disldo_backward<S, FP4BiPacked, COL_TYPE>(
            _last_input.data(), _last_batch, _last_cols,
            (V*)dybuf.ptr, weights,
            dx.data(),
            neuron_input_accum.data(), neuron_grad_accum.data(),
            learning_rate,
            num_cpus, lr_per_row_nnz);
        py::array_t<V> result({(py::ssize_t)_last_batch, (py::ssize_t)_last_cols});
        std::copy(dx.begin(), dx.end(), (V*)result.request().ptr);
        return result;
    }

    // ── forward_sparse / backward_sparse ─────────────────────────────────────────
    // NOT mirror images of each other (see conversation) -- forward's
    // bottleneck is the ACTIVATIONS, backward's is the GRADIENT, and these
    // are independent axes:
    //   forward_sparse:  SPARSE input (CSR)   -- skips inactive input rows.
    //   backward_sparse: DENSE input, SPARSE gradient (CSR) -- deliberately
    //                     NOT sparse input. dx[r] = sum_c W[r,c]*dy[c]
    //                     depends only on weights and the gradient, not on
    //                     input[r] -- a row whose own activation was zero
    //                     still gets a correct dx, correctly telling the
    //                     upstream layer "you should have fired more here."
    //                     Sparse input would skip that row entirely,
    //                     permanently losing that correction path -- only
    //                     "fired & shouldn't have" would ever get fixed,
    //                     never "didn't fire & should have". The weight
    //                     update still scales with the true (dense) input
    //                     value, so it stays appropriately small for rows
    //                     that didn't fire -- no separate handling needed.
    // Use *_sparse when the relevant side (activations for forward, gradient
    // for backward) is ACTUALLY sparse (e.g. after a top-k/threshold
    // sparsification step run between layers -- that decision belongs
    // outside this class, see class comment) -- meaningless/wasteful to
    // force through here otherwise. No bare forward()/backward() on this
    // class deliberately -- see TODO.md for the planned auto-dispatching
    // version and why a bare name isn't here yet: an unqualified default
    // risks everyone reaching for dense out of habit and losing a real
    // 10-100x speedup when the relevant side actually is sparse.

    CSRInput<S, V> _numpy_to_csr_input(py::array_t<S> ptrs, py::array_t<S> indices,
                                       py::array_t<V> values, S batch, S cols) {
        auto pb = ptrs.request(), ib = indices.request(), vb = values.request();
        const S nz = (S)ib.size;
        CSRInput<S, V> csr;
        csr.rows = batch; csr.cols = cols;
        csr.ptrs[0]    = std::make_shared<std::vector<S>>((S*)pb.ptr, (S*)pb.ptr + batch + 1);
        csr.indices[0] = std::make_shared<std::vector<S>>((S*)ib.ptr, (S*)ib.ptr + nz);
        csr.values[0]  = std::make_shared<std::vector<V>>((V*)vb.ptr, (V*)vb.ptr + nz);
        return csr;
    }

    py::array_t<V> forward_sparse(
        py::array_t<S> ptrs, py::array_t<S> indices, py::array_t<V> values,
        S batch, V learning_rate = 0.0f)
    {
        auto input = _numpy_to_csr_input(ptrs, indices, values, batch, n_inputs());
        output_buf.assign(batch * n_outputs(), V(0));
        delta_csr_forward<S, FP4BiPacked, COL_TYPE>(
            input, weights, output_buf.data(), learning_rate, num_cpus);
        return py::array_t<V>(
            {(py::ssize_t)batch, (py::ssize_t)n_outputs()},
            {(py::ssize_t)(n_outputs() * sizeof(V)), (py::ssize_t)sizeof(V)},
            output_buf.data(), py::cast(this));
    }

    py::array_t<V> backward_sparse(
        py::array_t<V> x,   // DENSE input -- see class comment for why
        py::array_t<S> dy_ptrs, py::array_t<S> dy_indices, py::array_t<V> dy_values,
        S batch, V learning_rate = 0.01f, bool lr_per_row_nnz = false)
    {
        auto xbuf = x.request();
        auto out_grad = _numpy_to_csr_input(dy_ptrs, dy_indices, dy_values, batch, n_outputs());
        std::vector<V> dx(batch * n_inputs(), V(0));
        delta_csr_backward_sparse_grad<S, FP4BiPacked, COL_TYPE>(
            (V*)xbuf.ptr, batch, weights, out_grad, dx.data(),
            neuron_input_accum.data(), neuron_grad_accum.data(), learning_rate, num_cpus, lr_per_row_nnz);
        py::array_t<V> result({(py::ssize_t)batch, (py::ssize_t)n_inputs()});
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

    // Stateful convenience wrapper around synap_row_step: advances an
    // internal row cursor automatically (wraps via % n_inputs, matching
    // delta_csr_synap_row_step's own semantics), so callers doing a "one
    // step per call, many calls over time" synaptogenesis sweep don't need
    // to track the row index themselves -- synap_row_step (above) stays
    // available for callers who want explicit control instead. Separate
    // cursor from equalizer_step below -- they serve different purposes
    // (synaptogenesis vs. memory rebalancing) and may reasonably progress
    // at different paces.
    S      _synap_row = 0;
    bool synap_step(V importance_cutoff, S max_row_weights) {
        std::size_t row = static_cast<std::size_t>(_synap_row);
        const bool did = delta_csr_synap_row_step<S, FP4BiPacked, COL_TYPE>(
            weights, row, importance_cutoff, max_row_weights);
        _synap_row = static_cast<S>(row);
        return did;
    }

    // Stateful convenience wrapper around delta_csr_equalize_step (memory
    // rebalancing -- redistributes blank space between neighboring rows'
    // territory so growth headroom stays reasonably even across the
    // layer). Own internal cursor, separate from synap_step's.
    S      _equalize_row = 0;
    void equalizer_step() {
        std::size_t row = static_cast<std::size_t>(_equalize_row);
        delta_csr_equalize_step<S, FP4BiPacked, COL_TYPE>(weights.connections, row);
        _equalize_row = static_cast<S>(row);
    }

    // One-time setup: grow each row until it has at least target_elems
    // elements of reserved space. Unlike equalizer_step() (which only
    // redistributes the existing pool), this ADDS memory for rows below
    // the target. Call once with max_row_weights before starting
    // synaptogenesis; subsequent cycles use the staggered equalizer_step()
    // for ongoing redistribution.
    // Last row handled separately (no memmove needed -- nothing follows it).
    // equalize_to_capacity(target_elems_per_row, target_bytes_per_row=0)
    // Grow each row until it has at least target_elems_per_row elements and
    // target_bytes_per_row index bytes of reserved space.
    //
    // target_bytes_per_row = 0 (default): derive as target_elems * uleb128_max
    // (worst-case encoding; safe for any column range but wastes memory for
    // models where column deltas are typically small). Pass an explicit byte
    // count when you know the typical encoding -- e.g. 100 connections at
    // 2 bytes each = 204 bytes, vs the default 100*5+4 = 504 bytes.
    //
    // This is a ONE-TIME CONSTRUCTION CALL. Call it from from_descriptor with
    // the max_row_weights and bytes_per_row for this specific layer. After
    // this, the pool is fixed and equalizer_step() only redistributes within
    // it -- it never grows further.
    //
    // KEY SUBTLETY: delta_csr_shift_row updates elem_start[r+1..rows] and
    // (after the fix applied to it) byte_end/elem_end for shifted rows. A
    // bulk loop over all rows is therefore safe: no stale _end arrays remain.
    void equalize_to_capacity(int target_elems_per_row, int target_bytes_per_row = 0) {
        const std::size_t tgt_e = static_cast<std::size_t>(target_elems_per_row);
        const std::size_t tgt_b = (target_bytes_per_row > 0)
            ? static_cast<std::size_t>(target_bytes_per_row)
            : tgt_e * uleb128_max_bytes<COL_TYPE>() + 4;
        auto& dc  = weights.connections;
        const std::size_t rows = dc.layout.rows;

        for (std::size_t r = 0; r + 1 < rows; ++r) {
            auto& L = dc.layout;
            const std::size_t cur_b = L.row_alloc_bytes(r);
            const std::size_t cur_e = L.row_alloc_elems(r);
            const std::size_t use_b = std::max(cur_b, tgt_b);
            const std::size_t use_e = std::max(cur_e, tgt_e);
            if (use_b == cur_b && use_e == cur_e) continue;

            const std::ptrdiff_t bd =
                static_cast<std::ptrdiff_t>(use_b) - static_cast<std::ptrdiff_t>(cur_b);
            const std::ptrdiff_t ed =
                static_cast<std::ptrdiff_t>(use_e) - static_cast<std::ptrdiff_t>(cur_e);

            delta_csr_shift_row<S, FP4BiPacked, COL_TYPE>(dc, r, use_b, use_e);
            // delta_csr_shift_row now updates byte_end and elem_end for all
            // rows r+1..rows-1 -- no additional fixup needed here.
        }

        // Last row: no rows follow it, so no memmove is needed -- just
        // extend the flat buffers and update the end markers.
        if (rows > 0) {
            auto& L = dc.layout;
            const std::size_t r   = rows - 1;
            const std::size_t cur_b = L.row_alloc_bytes(r);
            const std::size_t cur_e = L.row_alloc_elems(r);
            const std::size_t use_b = std::max(cur_b, tgt_b);
            const std::size_t use_e = std::max(cur_e, tgt_e);
            if (use_b > cur_b) {
                dc.indices_buf.resize(dc.indices_buf.size() + (use_b - cur_b), uint8_t(0));
                L.byte_start[rows] = L.byte_start[r] + use_b;
            }
            if (use_e > cur_e) {
                const std::size_t new_total = L.elem_start[rows] + (use_e - cur_e);
                ValueAccessor<FP4BiPacked>::resize(dc.values, new_total);
                L.elem_start[rows] = new_total;
            }
        }
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

    // Opposite of compact(): restores growth headroom, normalized to exactly
    // blank_fraction of current content (not "at least" -- see expand() in
    // sparse_struct.hpp). Call before resuming synaptogenesis on a layer
    // that's been compact()ed -- synap_row_step now throws a catchable
    // exception rather than silently doing nothing if headroom is missing.
    void expand_headroom(float blank_fraction = 0.2f) {
        weights.connections = ::expand_headroom<S, FP4BiPacked, COL_TYPE>(weights.connections, blank_fraction);
    }

    // Like expand_headroom() but guarantees each row has headroom for at
    // least min_nnz_per_row connections. Required for grow-back after prune:
    // a row pruned to 2 connections with plain expand_headroom() gets only
    // ~2.4 connections of headroom, then fails when synap_step tries to grow
    // back toward max_row_weights (which may be much larger).
    void expand_headroom_to(int min_nnz_per_row, float blank_fraction = 0.2f) {
        weights.connections = ::expand_headroom_to<S, FP4BiPacked, COL_TYPE>(
            weights.connections,
            static_cast<std::size_t>(min_nnz_per_row),
            blank_fraction);
    }

    // Per-ROW scale applied to stored importance/weight to get true units
    // before any arithmetic -- see SparseLinearWeightsDelta's own comment
    // (delta_csr_types.hpp) for the full motivation, including why per-row
    // rather than per-layer. Default 1.0 for any row not yet touched,
    // exact backward compat.
    V get_importance_scale(S row) const { return weights.get_importance_scale(static_cast<std::size_t>(row)); }
    V get_value_scale(S row)      const { return weights.get_value_scale(static_cast<std::size_t>(row)); }

    // Change ONE row's scale mid-training without corrupting that row's
    // existing stored data -- see SparseLinearWeightsDelta::
    // rescale_importance_row/rescale_value_row for what this actually does
    // (re-reads at the OLD per-row scale, re-encodes at the NEW one). Do
    // not just assign get_*_scale()'s value directly -- that would
    // silently reinterpret existing stored values as if they'd always
    // been at the new scale.
    void rescale_importance_row(S row, V new_scale) {
        weights.rescale_importance_row(static_cast<std::size_t>(row), new_scale);
    }
    void rescale_value_row(S row, V new_scale) {
        weights.rescale_value_row(static_cast<std::size_t>(row), new_scale);
    }

    // Bulk convenience: set EVERY row to the same new_scale -- backward-
    // compatible interface with the original per-layer-scalar design.
    void rescale_importance(V new_scale) {
        weights.rescale_importance(new_scale);
    }
    void rescale_value(V new_scale) {
        weights.rescale_value(new_scale);
    }

    // Running L1/L2/max stats for the STORED (quantized) importance/value
    // distribution, maintained incrementally at O(1) per synapse touched --
    // see SparseLinearWeightsDelta's own comment (delta_csr_types.hpp) for
    // the full design (including the max_abs monotonic-bound limitation).
    // Underpins a Python-side adaptive rescaling policy, not built here --
    // see refactoring_todo.md/TODO.md.
    V get_value_l1()           const { return static_cast<V>(weights.value_l1); }
    V get_value_l2_sq()        const { return static_cast<V>(weights.value_l2_sq); }
    V get_value_max_abs()      const { return weights.value_max_abs; }
    V get_importance_l1()      const { return static_cast<V>(weights.importance_l1); }
    V get_importance_l2_sq()   const { return static_cast<V>(weights.importance_l2_sq); }
    V get_importance_max_abs() const { return weights.importance_max_abs; }
    V hoyer_value()            const { return weights.hoyer_value(); }
    V hoyer_importance()       const { return weights.hoyer_importance(); }

    // Recompute all six stats above from scratch -- O(nnz). Gives an EXACT
    // max_abs (unlike the incrementally-maintained monotonic bound); call
    // when that distinction matters, not routinely.
    void recompute_stats() { weights.recompute_stats(); }

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
        // Use the budget already established at construction (max_indices_bytes
        // / max_values_bytes on the existing layout), not a recalculated smaller
        // one from idx.size(). Using idx.size()*8+4096 here created a new
        // DeltaCSRWeights with a SMALLER limit than the layer was constructed
        // with, causing dc.indices_buf.assign(L.byte_start[rows]) to write
        // beyond max_indices_bytes and corrupt the heap.
        // Use the budget established at construction (see _idx_budget_bytes
        // comment above): idx.size()*8+4096 can be SMALLER than the actual
        // bytes written by indices_buf.assign(L.byte_start[rows]) due to
        // per-row headroom, corrupting the heap when a second layer existed.
        const std::size_t idx_budget = std::max(_idx_budget_bytes, idx.size() * 8 + 4096);
        const std::size_t val_budget = std::max(_val_budget_nnz,   idx.size() + 64);
        weights.connections = delta_csr_from_absolute<S, FP4BiPacked, COL_TYPE>(
            p, idx, w, imp, rows, cols, idx_budget, val_budget);
        weights.recompute_stats();
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
    std::size_t    _idx_budget_bytes = 4096;
    std::size_t    _val_budget_nnz   = 64;

    DISLDOLayerV(S n_inputs, S n_outputs, S max_weights, int cpus = 4)
        : num_cpus(cpus),
          _idx_budget_bytes(static_cast<std::size_t>(max_weights) * 8 + 4096),
          _val_budget_nnz  (static_cast<std::size_t>(max_weights) + 64)
    {
        std::vector<S> empty_ptrs(static_cast<std::size_t>(n_inputs) + 1, S(0));
        std::vector<S> empty_idx;
        std::vector<V> empty_w, empty_imp;
        weights.connections = delta_csr_from_absolute<S, VT, COL_TYPE>(
            empty_ptrs, empty_idx, empty_w, empty_imp,
            static_cast<std::size_t>(n_inputs), static_cast<std::size_t>(n_outputs),
            static_cast<std::size_t>(max_weights) * 8 + 4096,
            static_cast<std::size_t>(max_weights) + 64);
        weights.recompute_stats();
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
            std::max(_idx_budget_bytes, idx.size() * 8 + 4096),
            std::max(_val_budget_nnz,   idx.size() + 64));
        weights.recompute_stats();
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
        .def("forward_dense",        &SparseLinearLayer::forward_dense,
             py::arg("x"), py::arg("learning_rate") = 0.01f)
        .def("backward_dense",       &SparseLinearLayer::backward_dense,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false)
        .def("forward_sparse",       &SparseLinearLayer::forward_sparse,
             py::arg("ptrs"), py::arg("indices"), py::arg("values"),
             py::arg("batch"), py::arg("learning_rate") = 0.0f)
        .def("backward_sparse",      &SparseLinearLayer::backward_sparse,
             py::arg("x"),
             py::arg("dy_ptrs"), py::arg("dy_indices"), py::arg("dy_values"),
             py::arg("batch"), py::arg("learning_rate") = 0.01f, py::arg("lr_per_row_nnz") = false)
        .def("build_probes",         &SparseLinearLayer::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayer::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"))
        .def("synap_step",           &SparseLinearLayer::synap_step,
             py::arg("importance_cutoff"), py::arg("max_row_weights"),
             "Stateful convenience wrapper around synap_row_step -- advances an\n"
             "internal row cursor automatically, so a caller doing repeated\n"
             "one-step-per-call synaptogenesis sweeps doesn't need to track the\n"
             "row index itself. Use synap_row_step directly for explicit control.")
        .def("equalizer_step",       &SparseLinearLayer::equalizer_step,
             "One row of staggered memory redistribution -- call once per\n"
             "synaptogenesis cycle. REDISTRIBUTES the existing pool; does NOT\n"
             "add new memory. Use equalize_to_capacity() first to ensure the\n"
             "pool is large enough for max_row_weights connections per row.")
        .def("equalize_to_capacity", &SparseLinearLayer::equalize_to_capacity,
             py::arg("target_elems_per_row"),
             py::arg("target_bytes_per_row") = 0,
             "One-time construction call: grow each row to at least\n"
             "target_elems_per_row elements and target_bytes_per_row index\n"
             "bytes of reserved space. target_bytes_per_row=0 (default)\n"
             "derives bytes as target_elems * uleb128_max (5) + 4 (safe\n"
             "worst-case). Pass an explicit byte count for efficiency:\n"
             "e.g. 100 connections with 2-byte deltas = 204 bytes, vs\n"
             "the default 100*5+4=504. After this call the pool is fixed;\n"
             "equalizer_step() only redistributes within it, never grows.")
        .def("compact",              &SparseLinearLayer::compact,
             "Repack in place: every row occupies exactly its active bytes/elements,\n"
             "zero inter-row blank space. Call before saving/measuring a freshly\n"
             "converted model. Zeroes growth headroom.")
        .def("expand_headroom",      &SparseLinearLayer::expand_headroom,
             py::arg("blank_fraction") = 0.2f,
             "Restore per-row growth headroom (proportional to current nnz).\n"
             "WARNING: use expand_headroom_to(max_row_weights) after a prune\n"
             "cycle -- plain expand_headroom allocates headroom proportional\n"
             "to CURRENT nnz, leaving no room to grow back to max_row_weights.")
        .def("expand_headroom_to",   &SparseLinearLayer::expand_headroom_to,
             py::arg("min_nnz_per_row"), py::arg("blank_fraction") = 0.2f,
             "Like expand_headroom() but guarantees each row has headroom for\n"
             "at least min_nnz_per_row connections. Call with max_row_weights\n"
             "before synaptogenesis after a prune cycle.")
        .def("get_importance_scale", &SparseLinearLayer::get_importance_scale,
             py::arg("row"),
             "Per-ROW scale applied to that row's stored importance to get true\n"
             "units. Default 1.0 for any row not yet touched, exact backward\n"
             "compat. Use rescale_importance_row()/rescale_importance() to\n"
             "change it, never assign directly.")
        .def("get_value_scale",      &SparseLinearLayer::get_value_scale,
             py::arg("row"),
             "Same as get_importance_scale() but for stored weight values.")
        .def("set_value_scale_raw",
             [](SparseLinearLayer& self, int row, float scale) {
                 self.weights.set_value_scale_raw(
                     static_cast<std::size_t>(row), scale);
             },
             py::arg("row"), py::arg("scale"),
             "Set value_scale[row] directly WITHOUT re-encoding stored weights.\n"
             "Use this after pre-scaling weights before load_weights() -- calling\n"
             "rescale_value_row() after a pre-scaled load would double-encode.\n"
             "Typical pattern:\n"
             "  row_scale = max_abs / FP4_MAX\n"
             "  layer.load_weights(ptrs, idx, vals / row_scale)  # pre-scaled\n"
             "  layer.set_value_scale_raw(r, row_scale)          # set metadata only")
        .def("set_importance_scale_raw",
             [](SparseLinearLayer& self, int row, float scale) {
                 self.weights.set_importance_scale_raw(
                     static_cast<std::size_t>(row), scale);
             },
             py::arg("row"), py::arg("scale"),
             "Same as set_value_scale_raw() but for importance.")
        .def("rescale_importance_row", &SparseLinearLayer::rescale_importance_row,
             py::arg("row"), py::arg("new_scale"),
             "Change ONE row's importance scale mid-training without corrupting\n"
             "that row's existing stored data -- re-reads at the OLD per-row\n"
             "scale, re-encodes at the NEW one.")
        .def("rescale_value_row",    &SparseLinearLayer::rescale_value_row,
             py::arg("row"), py::arg("new_scale"),
             "Same as rescale_importance_row() but for stored weight values.")
        .def("rescale_importance",   &SparseLinearLayer::rescale_importance,
             py::arg("new_scale"),
             "Bulk convenience: set EVERY row's importance scale to the same\n"
             "value. Backward-compatible interface with the original\n"
             "per-layer-scalar design.")
        .def("rescale_value",        &SparseLinearLayer::rescale_value,
             py::arg("new_scale"),
             "Same as rescale_importance() but for stored weight values.")
        .def_property_readonly("value_l1",           &SparseLinearLayer::get_value_l1)
        .def_property_readonly("value_l2_sq",        &SparseLinearLayer::get_value_l2_sq)
        .def_property_readonly("value_max_abs",      &SparseLinearLayer::get_value_max_abs)
        .def_property_readonly("importance_l1",      &SparseLinearLayer::get_importance_l1)
        .def_property_readonly("importance_l2_sq",   &SparseLinearLayer::get_importance_l2_sq)
        .def_property_readonly("importance_max_abs", &SparseLinearLayer::get_importance_max_abs)
        .def("hoyer_value",          &SparseLinearLayer::hoyer_value,
             "Hoyer's sparsity measure on the STORED weight distribution, in\n"
             "[0,1] -- 0 means values spread evenly across FP4's representable\n"
             "range, 1 means concentrated (e.g. mostly zero, or mostly clustered\n"
             "at one magnitude). O(1), from running stats -- see value_max_abs\n"
             "for a cheap complementary saturation check this doesn't catch.")
        .def("hoyer_importance",     &SparseLinearLayer::hoyer_importance,
             "Same as hoyer_value() but for the STORED importance distribution.")
        .def("recompute_stats",      &SparseLinearLayer::recompute_stats,
             "Recompute value_l1/l2_sq/max_abs and importance_l1/l2_sq/max_abs\n"
             "from scratch, O(nnz). Gives an exact max_abs (the incrementally-\n"
             "maintained one is a monotonic upper bound, not a live exact max).")
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
        .def_property_readonly("nnz",       &SparseLinearLayer::nnz)
        .def_property_readonly("last_input",
            [](const SparseLinearLayer& self) -> py::object {
                if (self._last_input.empty()) return py::none();
                // Return a zero-copy numpy view of the stored last input.
                // Shape [_last_batch, _last_cols] -- needed by backward_sparse
                // which requires the explicit forward input (can't retrieve it
                // from inside the kernel the way backward_dense does via the
                // stored member).
                return py::array_t<float>(
                    {(py::ssize_t)self._last_batch, (py::ssize_t)self._last_cols},
                    {(py::ssize_t)self._last_cols * (py::ssize_t)sizeof(float), (py::ssize_t)sizeof(float)},
                    self._last_input.data(), py::cast(&self));
            },
            "Dense input from the most recent forward_dense/forward_sparse call.\n"
            "Shape [batch, n_inputs]. None if no forward pass has been run yet.\n"
            "Used by backward_sparse which requires the explicit forward input.");

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

    // ── hoyer_sparsify ────────────────────────────────────────────────────────
    // NOT wired into an automatic dense/sparse dispatch (see TODO.md) -- this
    // is the standalone Hoyer's-Sparsity-Measure operation, exposed so its
    // actual behavior on real data can be explored/tested from Python before
    // deciding on dispatch thresholds. Returns diagnostics (hoyer_score,
    // k_estimate, l1/l2 norms) alongside the CSR result, since the point is
    // to make the not-obvious behavior actually inspectable.
    m.def("hoyer_sparsify",
        [](py::array_t<float> x) -> py::dict {
            auto buf   = x.request();
            const std::size_t rows = (buf.ndim == 2) ? (std::size_t)buf.shape[0] : 1;
            const std::size_t cols = (buf.ndim == 2) ? (std::size_t)buf.shape[1] : (std::size_t)buf.shape[0];
            float* src = (float*)buf.ptr;

            auto per_row = hoyer_sparsify_per_batch<float>(src, rows, cols);

            std::vector<int>   ptrs(rows + 1, 0);
            std::vector<int>   indices;
            std::vector<float> values;
            py::array_t<float> hoyer_scores({(py::ssize_t)rows});
            py::array_t<int>   k_estimates ({(py::ssize_t)rows});
            py::array_t<float> l1_norms    ({(py::ssize_t)rows});
            py::array_t<float> l2_norms    ({(py::ssize_t)rows});

            float* hs = (float*)hoyer_scores.request().ptr;
            int*   ke = (int*)  k_estimates.request().ptr;
            float* l1 = (float*)l1_norms.request().ptr;
            float* l2 = (float*)l2_norms.request().ptr;

            for (std::size_t r = 0; r < rows; ++r) {
                ptrs[r] = (int)indices.size();
                for (std::size_t j = 0; j < per_row[r].indices.size(); ++j) {
                    indices.push_back(per_row[r].indices[j]);
                    values .push_back(per_row[r].values[j]);
                }
                hs[r] = per_row[r].hoyer_score;
                ke[r] = per_row[r].k_estimate;
                l1[r] = per_row[r].l1_norm;
                l2[r] = per_row[r].l2_norm;
            }
            ptrs[rows] = (int)indices.size();

            py::array_t<int>   out_ptrs   ({(py::ssize_t)(rows + 1)});
            py::array_t<int>   out_indices({(py::ssize_t)indices.size()});
            py::array_t<float> out_values ({(py::ssize_t)values.size()});
            std::copy(ptrs.begin(),    ptrs.end(),    (int*)  out_ptrs.request().ptr);
            std::copy(indices.begin(), indices.end(), (int*)  out_indices.request().ptr);
            std::copy(values.begin(),  values.end(),  (float*)out_values.request().ptr);

            py::dict result;
            result["ptrs"]         = out_ptrs;
            result["indices"]      = out_indices;
            result["values"]       = out_values;
            result["hoyer_score"]  = hoyer_scores;
            result["k_estimate"]   = k_estimates;
            result["l1_norm"]      = l1_norms;
            result["l2_norm"]      = l2_norms;
            return result;
        },
        py::arg("x"),
        "Hoyer's Sparsity Measure top-k sparsification, per row.\n"
        "hoyer(x) = (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1), in [0,1].\n"
        "k_estimate = (||x||_1/||x||_2)^2 -- exact for a vector with exactly\n"
        "k nonzero entries of equal magnitude, a principled estimate of the\n"
        "'effective' significant-element count otherwise. Returns CSR\n"
        "(ptrs, indices, values) using k_estimate as the per-row top-k, plus\n"
        "the diagnostics (hoyer_score, k_estimate, l1_norm, l2_norm) so the\n"
        "not-obvious behavior can actually be inspected.\n"
        "NOTE: 'row' here means one row of the [batch,cols] array, i.e. one\n"
        "BATCH SAMPLE -- not a weight-matrix row (DeltaCSRWeights elsewhere\n"
        "uses 'row' for input neuron, a different axis). This per-sample\n"
        "granularity is for CONSTRUCTING an accurate sparse representation\n"
        "once you've already decided to route a batch through the sparse\n"
        "path -- it is NOT the right thing to base that routing decision on\n"
        "(forward_dense/forward_sparse are each called once for the WHOLE\n"
        "batch, so a per-sample answer isn't actionable there). For the\n"
        "actual dense-vs-sparse routing decision, use hoyer_score().\n"
        "Not wired into any automatic dispatch yet -- see TODO.md.");

    // ── hoyer_score ─────────────────────────────────────────────────────
    m.def("hoyer_score",
        [](py::array_t<float> x) -> py::dict {
            auto buf   = x.request();
            const std::size_t rows = (buf.ndim == 2) ? (std::size_t)buf.shape[0] : 1;
            const std::size_t cols = (buf.ndim == 2) ? (std::size_t)buf.shape[1] : (std::size_t)buf.shape[0];
            float* src = (float*)buf.ptr;

            auto agg = hoyer_score<float>(src, rows, cols);

            py::dict result;
            result["hoyer_score"] = agg.hoyer_score;
            result["k_estimate"]  = agg.k_estimate;
            result["l1_norm"]     = agg.l1_norm;
            result["l2_norm"]     = agg.l2_norm;
            result["n_total"]     = (int)(rows * cols);
            return result;
        },
        py::arg("x"),
        "Batch-level aggregate Hoyer's measure -- the quantity a dense-vs-\n"
        "sparse ROUTING decision should actually use, computed over the\n"
        "WHOLE flattened batch (all rows*cols elements together), since\n"
        "forward_dense/forward_sparse are each invoked once for the entire\n"
        "batch in a single call, not once per sample. Returns hoyer_score\n"
        "(threshold this to decide forward_dense vs forward_sparse),\n"
        "k_estimate, l1_norm, l2_norm, n_total. Does not return indices/\n"
        "values -- for constructing the actual sparse CSR once routing has\n"
        "decided 'sparse', use hoyer_sparsify() instead. Not wired into any\n"
        "automatic dispatch yet -- see TODO.md.");

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

    // ── Attention ops (ported from sparse_linear_ops.hpp) ────────────────────
    // All three take numpy [T, d] Q/K/V arrays (float32) and return a
    // numpy [T, d] output. The Python-facing names match exactly what
    // multimodal_sparse_rnn.py calls (sparse_attention, sparse_banded_attention).

    m.def("sparse_attention",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           std::size_t top_k, int num_cpus) {
            auto qb=q.request(), kb=k.request(), vb=v.request();
            const std::size_t T = qb.shape[0], d = qb.shape[1];
            py::array_t<float> out({(py::ssize_t)T, (py::ssize_t)d});
            auto ob = out.request();
            std::fill((float*)ob.ptr, (float*)ob.ptr + T*d, 0.0f);
            sparse_attention_forward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (float*)ob.ptr, T, d, top_k, num_cpus);
            return out;
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("top_k") = 0, py::arg("num_cpus") = 4,
        "Global top-k sparse attention. Q/K/V are [T, d] float32 numpy arrays.\n"
        "top_k=0 -> use sqrt(T). Returns [T, d] output.");

    m.def("sparse_banded_attention",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           std::size_t half_bandwidth, std::size_t inner_k, int num_cpus) {
            auto qb=q.request(), kb=k.request(), vb=v.request();
            const std::size_t T = qb.shape[0], K = kb.shape[0], d = qb.shape[1];
            py::array_t<float> out({(py::ssize_t)T, (py::ssize_t)d});
            auto ob = out.request();
            std::fill((float*)ob.ptr, (float*)ob.ptr + T*d, 0.0f);
            sparse_banded_attention_forward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (float*)ob.ptr, T, K, d, half_bandwidth, inner_k, num_cpus);
            return out;
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("half_bandwidth"), py::arg("inner_k") = 0, py::arg("num_cpus") = 4,
        "Banded sparse attention. Q/K/V are [T, d] float32 numpy arrays.\n"
        "inner_k=0 -> use all keys in the band (dense banded). Returns [T, d] output.");

    m.def("banded_attention",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           std::size_t half_bandwidth, int num_cpus) {
            auto qb=q.request(), kb=k.request(), vb=v.request();
            const std::size_t T = qb.shape[0], K = kb.shape[0], d = qb.shape[1];
            py::array_t<float> out({(py::ssize_t)T, (py::ssize_t)d});
            auto ob = out.request();
            std::fill((float*)ob.ptr, (float*)ob.ptr + T*d, 0.0f);
            banded_attention_forward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (float*)ob.ptr, T, K, d, half_bandwidth, num_cpus);
            return out;
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("half_bandwidth"), py::arg("num_cpus") = 4,
        "Dense banded attention. Q/K/V are [T, d] float32 numpy arrays.\n"
        "Returns [T, d] output.");

    m.def("banded_attention_backward",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           py::array_t<float> dO,
           std::size_t half_bandwidth, int num_cpus) {
            auto qb=q.request(), kb=k.request(), vb=v.request(), dob=dO.request();
            const std::size_t T = qb.shape[0], K = kb.shape[0], d = qb.shape[1];
            py::array_t<float> dQ({(py::ssize_t)T, (py::ssize_t)d});
            py::array_t<float> dK({(py::ssize_t)K, (py::ssize_t)d});
            py::array_t<float> dV({(py::ssize_t)K, (py::ssize_t)d});
            auto dqb=dQ.request(), dkb=dK.request(), dvb=dV.request();
            std::fill((float*)dqb.ptr, (float*)dqb.ptr + T*d, 0.0f);
            std::fill((float*)dkb.ptr, (float*)dkb.ptr + K*d, 0.0f);
            std::fill((float*)dvb.ptr, (float*)dvb.ptr + K*d, 0.0f);
            banded_attention_backward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (const float*)dob.ptr,
                (float*)dqb.ptr, (float*)dkb.ptr, (float*)dvb.ptr,
                T, K, d, half_bandwidth, num_cpus);
            return py::make_tuple(dQ, dK, dV);
        },
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"),
        py::arg("half_bandwidth"), py::arg("num_cpus") = 4,
        "Backward pass for banded_attention. Returns (dQ, dK, dV) each [T or K, d].");

    m.def("sparse_banded_attention_backward",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           py::array_t<float> dO,
           std::size_t half_bandwidth, std::size_t inner_k, int num_cpus) {
            auto qb=q.request(), kb=k.request(), vb=v.request(), dob=dO.request();
            const std::size_t T = qb.shape[0], K = kb.shape[0], d = qb.shape[1];
            py::array_t<float> dQ({(py::ssize_t)T, (py::ssize_t)d});
            py::array_t<float> dK({(py::ssize_t)K, (py::ssize_t)d});
            py::array_t<float> dV({(py::ssize_t)K, (py::ssize_t)d});
            auto dqb=dQ.request(), dkb=dK.request(), dvb=dV.request();
            std::fill((float*)dqb.ptr, (float*)dqb.ptr + T*d, 0.0f);
            std::fill((float*)dkb.ptr, (float*)dkb.ptr + K*d, 0.0f);
            std::fill((float*)dvb.ptr, (float*)dvb.ptr + K*d, 0.0f);
            sparse_banded_attention_backward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (const float*)dob.ptr,
                (float*)dqb.ptr, (float*)dkb.ptr, (float*)dvb.ptr,
                T, K, d, half_bandwidth, inner_k, num_cpus);
            return py::make_tuple(dQ, dK, dV);
        },
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"),
        py::arg("half_bandwidth"), py::arg("inner_k") = 0, py::arg("num_cpus") = 4,
        "Backward pass for sparse_banded_attention. Returns (dQ, dK, dV).");

    m.def("sparse_attention_backward",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           py::array_t<float> dO,
           std::size_t top_k, int num_cpus) {
            auto qb=q.request(), kb=k.request(), vb=v.request(), dob=dO.request();
            const std::size_t T = qb.shape[0], d = qb.shape[1];
            py::array_t<float> dQ({(py::ssize_t)T, (py::ssize_t)d});
            py::array_t<float> dK({(py::ssize_t)T, (py::ssize_t)d});
            py::array_t<float> dV({(py::ssize_t)T, (py::ssize_t)d});
            auto dqb=dQ.request(), dkb=dK.request(), dvb=dV.request();
            std::fill((float*)dqb.ptr, (float*)dqb.ptr + T*d, 0.0f);
            std::fill((float*)dkb.ptr, (float*)dkb.ptr + T*d, 0.0f);
            std::fill((float*)dvb.ptr, (float*)dvb.ptr + T*d, 0.0f);
            sparse_attention_backward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (const float*)dob.ptr,
                (float*)dqb.ptr, (float*)dkb.ptr, (float*)dvb.ptr,
                T, d, top_k, num_cpus);
            return py::make_tuple(dQ, dK, dV);
        },
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"),
        py::arg("top_k") = 0, py::arg("num_cpus") = 4,
        "Backward pass for sparse_attention. Returns (dQ, dK, dV).");

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