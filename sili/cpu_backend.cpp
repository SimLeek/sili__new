#include <algorithm>
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

// ── Block4View ───────────────────────────────────────────────────────────────
// Thin, non-owning wrapper exposing a layer's Block4Store to Python as
// `layer.block4.tiles` / `layer.block4.synapses` -- purely observational
// (see block4.hpp: block4's whole point is to be invisible to callers
// otherwise). Holds a raw pointer into the owning layer's `weights.block4`;
// SparseLinearLayer::block4()/DISLDOLayerV::block4() bind this with
// py::keep_alive<0, 1>() so the layer can't be freed while a Block4View
// referencing it is still alive in Python.
class Block4View {
public:
    explicit Block4View(Block4Store& s) : store(&s) {}
    std::size_t tiles()    const { return store->n_tiles(); }
    std::size_t synapses() const { return store->live_synapses(); }
    // Per-store compression parameter (see block4.hpp: Block4Store::switch_point) --
    // the only knob in Block4View that isn't purely observational. A tile with
    // <= switch_point live synapses is eligible to be packed into the
    // sparse-encoded 16-byte layout instead of staying fully dense; 0 disables
    // compression entirely.
    uint32_t get_switch_point() const { return store->switch_point; }
    void set_switch_point(uint32_t v) { store->switch_point = v; }
    // How many times a backward/promotion value-update to an EXISTING
    // tile couldn't be persisted because growing its storage would have
    // exceeded the store's own memory budget -- see block4.hpp:
    // Block4Store::dropped_growth_events / ~Block4TileHandle(). By
    // design this is declined, not thrown, so training/inference keeps
    // running; a caller doing its own memory management can poll this to
    // detect and react (equalizer_step()/expand_headroom(), or just
    // accepting some updates are being dropped under the current budget).
    std::uint64_t dropped_growth_events() const { return store->dropped_growth_events; }
    // How many times a row's write-back in merge_row_workspace couldn't
    // fit its full content within the row's existing headroom even
    // after eviction ran (see block4.hpp: Block4Store::
    // merge_row_workspace's own comment for the real bug this
    // replaced -- an unconditional write-back that silently corrupted
    // the next row). Same declined-not-thrown philosophy as
    // dropped_growth_events: the row's trailing tiles keep their
    // pre-call content and training keeps running. Watch this counter
    // (and _bytes below) to know when this layer's max_row_weights
    // genuinely needs expand_headroom_to() with a bigger budget.
    std::uint64_t row_merge_overflow_events() const { return store->row_merge_overflow_events; }
    std::uint64_t row_merge_overflow_bytes_dropped() const { return store->row_merge_overflow_bytes_dropped; }
    // Real, live memory accounting -- see block4.hpp's
    // total_tile_used_bytes()/total_tile_alloc_bytes(): used_bytes sums
    // each row's ACTUAL packed content (sparse-vs-dense per tile, exactly
    // what merge_row_workspace itself would compute), alloc_bytes is the
    // tile_data buffer's current capacity (used_bytes plus whatever
    // per-row headroom slack -- e.g. block4_expand_headroom's
    // blank_fraction -- is currently reserved but not yet live). The gap
    // between them is the real cost of allowing rows room to grow; watch
    // it to confirm that cost stays small relative to a fully-dense
    // layer's n_in*n_out bytes, not silently approaching it.
    std::size_t used_bytes()  const { return store->total_tile_used_bytes(); }
    std::size_t alloc_bytes() const { return store->total_tile_alloc_bytes(); }
private:
    Block4Store* store;
};

// FP8 counterpart to Block4View, for Block4Store8 -- same fields/role,
// separate class since Block4View is hardcoded to Block4Store& (see
// delta_csr_types.hpp's Block4StoreFor<VALUES_TYPE> trait for why
// SparseLinearWeightsDelta.block4 is Block4Store8 specifically for
// FP8BiValues).
class Block4View8 {
public:
    explicit Block4View8(Block4Store8& s) : store(&s) {}
    std::size_t tiles()    const { return store->n_tiles(); }
    std::size_t synapses() const { return store->live_synapses(); }
    uint32_t get_switch_point() const { return store->switch_point; }
    void set_switch_point(uint32_t v) { store->switch_point = v; }
    std::uint64_t dropped_growth_events() const { return store->dropped_growth_events; }
    std::uint64_t row_merge_overflow_events() const { return store->row_merge_overflow_events; }
    std::uint64_t row_merge_overflow_bytes_dropped() const { return store->row_merge_overflow_bytes_dropped; }
private:
    Block4Store8* store;
};

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
        S batch)
    {
        output_buf.assign(batch * n_outputs(), V(0));
        auto input_csr = numpy_to_sparse_input(ptrs, indices, values, batch, n_inputs());
        sisldo_forward_trivalues(input_csr, weights, output_buf.data(), num_cpus);

        // COPY output_buf out, not a view into it -- output_buf is a
        // member reused (assign()'d over) by every future forward call
        // on this same object; a caller holding a PREVIOUS call's
        // returned array alive (e.g. comparing before/after a training
        // step, or ordinary Tensor-graph code keeping intermediates
        // around for backprop) would silently see it change out from
        // under them once this layer is called again. See
        // JOURNAL.md/sili_peridot's tile-recurrence work for how this
        // was found -- a real, reproducible, previously-unknown bug.
        py::array_t<V> result({(py::ssize_t)batch, (py::ssize_t)n_outputs()});
        std::copy(output_buf.begin(), output_buf.end(), result.mutable_data());
        return result;
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

        sisldo_backward_trivalues(
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
// this repo. forward_sparse uses sisldo_forward (already existed in
// this file, generic over VALUES_TYPE); backward_sparse uses
// disldo_backward_sparse_grad (dense input, sparse gradient -- see
// forward_sparse/backward_sparse comment below for why these are NOT
// mirror images of each other). Both verified correct against
// hand-computed references before wiring up -- see conversation.

// Templated over ScalePolicy/DeferredScaleWrite, same pattern as
// SparseLinearLayer8Impl below -- SparseLinearLayer (default policy,
// current behavior, unchanged) and SparseLinearLayerResync (the FP4
// counterpart of SparseLinearLayer8Resync's stale value_scale/
// output_scale fix -- see disldo_backward's DeferredScaleWrite docstring
// in linear_disldo.hpp) share this one implementation.
template <typename ScalePolicy = RMSpropScalePolicy<float>, bool DeferredScaleWrite = false,
          bool StochasticRounding = true>
class SparseLinearLayerImpl {
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

    SparseLinearLayerImpl(S n_inputs, S n_outputs, S max_weights, int cpus = 4)
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
        // Real, enforced cap -- without this, DeltaCSRWeights::
        // max_indices_bytes/max_values_bytes sit at their default
        // SIZE_MAX, so reserve_indices()/reserve_values() (called by
        // expand_headroom() during synaptogenesis) never actually
        // throw, and max_weights only bounds the INITIAL allocation
        // above, not later growth. Matches the exact budget just
        // reserved (values in bytes, via projected_byte_size -- see
        // delta_csr_from_absolute's own "values_bytes" parameter, which
        // is actually an nnz count despite the name, same units
        // reserve_values() takes).
        weights.connections.set_limits(
            _idx_budget_bytes,
            ValueAccessor<FP4BiPacked>::projected_byte_size(_val_budget_nnz));
        weights.block4.init(static_cast<std::size_t>(n_inputs), static_cast<std::size_t>(n_outputs));
        // Real, enforced cap for block4 -- see block4.hpp's
        // block4_row_shift comment for the bug this closes (block4
        // grew unboundedly, confirmed reaching ~1.9x max_weights with
        // zero resistance in a stress test). Deliberately a SEPARATE
        // budget from weights.connections.set_limits() above, not a
        // shared one (per direction) -- sized the same way: indices
        // mirrors the scattered side's own formula (block4's own
        // uleb128 tile-position index, same encoding, smaller
        // coordinate range so this is conservative/safe not tight).
        //
        // Tile data budget: max_weights / BLOCK4_TILE_SLOTS tiles, NOT
        // max_weights tiles and NOT max_weights/BLOCK4_PROMOTE_MIN_LIVE
        // tiles either -- both tried and measured too generous. A tile
        // holds 2-16 live synapses; dividing by BLOCK4_PROMOTE_MIN_LIVE
        // (2, assuming worst-case MINIMAL fill) sounds conservative but
        // is backwards: once promoted, a tile is treated as fully dense
        // by forward/backward (every slot touched on every call, see
        // TODO_DUAL_BLOCK4.md's switch_point findings), so REAL tiles
        // trend toward MAXIMUM fill under training, not minimum.
        // Confirmed, not hypothetical: at max_weights=200, the /2
        // formula (100-tile budget) still let block4 reach 381 real
        // synapses across only 47 tiles (avg ~8.1/tile, nowhere near
        // the 100-tile cap) before this fix. Dividing by
        // BLOCK4_TILE_SLOTS (16, max occupancy) instead keeps the
        // worst-case synapse ceiling at max_weights exactly (every
        // budgeted tile fully packed) -- the correct assumption given
        // how tiles actually fill in practice.
        // Floor of 4 tiles regardless of the formula above -- for a
        // small max_weights (< BLOCK4_TILE_SLOTS), max_weights/16
        // rounds down to 0, which would make block4 promotion
        // impossible even for a legitimately tiny test/toy layer. 4
        // tiles is a small, arbitrary-but-reasonable floor (matches
        // this codebase's own small-layer test fixtures, e.g.
        // test_disldo_block4_promotion.cpp's single-tile usage).
        weights.block4.set_limits(
            static_cast<std::size_t>(max_weights) * 8 + 4096,
            std::max<std::size_t>(4, static_cast<std::size_t>(max_weights) / BLOCK4_TILE_SLOTS) * BLOCK4_TILE_SLOTS);
        weights.recompute_stats();
        weights.probes.rows = n_inputs;
        weights.probes.cols = n_outputs;
        neuron_input_accum.assign(n_inputs,  V(0));
        neuron_grad_accum .assign(n_outputs, V(0));
        weights.out_degree.assign(n_outputs, S(0));
    }

    S n_inputs()  const { return static_cast<S>(weights.connections.layout.rows); }
    S n_outputs() const { return static_cast<S>(weights.connections.layout.cols); }
    S nnz()       const { return static_cast<S>(weights.connections.nnz() + weights.block4.live_synapses()); }
    // Purely observational (block4's whole point is to be invisible to
    // callers otherwise) -- see Block4View, bound as layer.block4.
    Block4View block4() { return Block4View(weights.block4); }

    // Rank of the value_scale/output_scale factorization -- see
    // scale_rank's own docstring, delta_csr_types.hpp. Default 1
    // (original behavior); set BEFORE any training call touches
    // value_scale/output_scale, since changing rank after synapses
    // already have per-component scale data stored would silently
    // reinterpret that data under a different row-major stride. Only
    // meaningful for disldo_forward/disldo_backward's SCATTERED-CSR
    // path -- block4's own forward/backward remain rank-1-only (see
    // scale_rank's own docstring for the tracked gap this implies).
    std::size_t get_scale_rank() const { return weights.scale_rank; }
    void set_scale_rank(std::size_t rank) {
        if (rank == 0) throw std::invalid_argument("scale_rank must be >= 1");
        if (rank > decltype(weights)::SCALE_RANK_MAX)
            throw std::invalid_argument("scale_rank exceeds SCALE_RANK_MAX (block4's SIMD backward path uses fixed-size stack arrays sized to it)");
        weights.scale_rank = rank;
    }

    // ── Forward (dense input — DISLDO) ──────────────────────────────────────────

    py::array_t<V> forward_dense(py::array_t<V> x) {
        auto xbuf     = x.request();
        _last_batch   = (xbuf.ndim == 2) ? (S)xbuf.shape[0] : 1;
        _last_cols    = (xbuf.ndim == 2) ? (S)xbuf.shape[1] : (S)xbuf.shape[0];

        const V* src  = (V*)xbuf.ptr;
        _last_input.assign(src, src + _last_batch * _last_cols);

        output_buf.assign(_last_batch * n_outputs(), V(0));
        disldo_forward<S, FP4BiPacked, COL_TYPE>(src, _last_batch, _last_cols, weights,
                       output_buf.data(), num_cpus);

        // COPY, not a view -- see forward_sparse's own comment above for
        // why (output_buf is reused/overwritten by every future call).
        py::array_t<V> result({(py::ssize_t)_last_batch, (py::ssize_t)n_outputs()});
        std::copy(output_buf.begin(), output_buf.end(), result.mutable_data());
        return result;
    }

    // ── Backward (dense input — DISLDO) ─────────────────────────────────────────

    py::array_t<V> backward_dense(py::array_t<V> dy, V learning_rate, bool lr_per_row_nnz = false,
                                  bool damp_by_importance = true, V beta2 = 0.999f, V eps = 1e-8f) {
        auto dybuf = dy.request();
        std::vector<V> dx(_last_batch * _last_cols, V(0));
        // beta1=0.9 (function default), min_decay_frac left at its own true
        // no-op default (0.0), max_abs_delta=2.0 -- BoundedRMSpropSynapsePolicy's
        // now-tuned production default, in RAW (pre-lr-multiply) units (see
        // update_cw's own docstring, delta_csr_types.hpp, for why it's
        // raw-space; 2.0 reproduces the exact validated behavior at the
        // tuning sweep's own lr=0.05 and generalizes correctly to other lr).
        disldo_backward<S, FP4BiPacked, COL_TYPE, ScalePolicy, DeferredScaleWrite, StochasticRounding>(
            _last_input.data(), _last_batch, _last_cols,
            (V*)dybuf.ptr, weights,
            dx.data(),
            neuron_input_accum.data(), neuron_grad_accum.data(),
            learning_rate,
            num_cpus, lr_per_row_nnz, damp_by_importance, beta2, eps,
            0.9f, 0.0f, 2.0f);
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
        S batch)
    {
        auto input = _numpy_to_csr_input(ptrs, indices, values, batch, n_inputs());
        output_buf.assign(batch * n_outputs(), V(0));
        sisldo_forward<S, FP4BiPacked, COL_TYPE>(
            input, weights, output_buf.data(), num_cpus);
        // COPY, not a view -- see SISLDOLayerV::forward_sparse's own
        // comment for why (output_buf is reused/overwritten by every
        // future call on this same object).
        py::array_t<V> result({(py::ssize_t)batch, (py::ssize_t)n_outputs()});
        std::copy(output_buf.begin(), output_buf.end(), result.mutable_data());
        return result;
    }

    py::array_t<V> backward_sparse(
        py::array_t<V> x,   // DENSE input -- see class comment for why
        py::array_t<S> dy_ptrs, py::array_t<S> dy_indices, py::array_t<V> dy_values,
        S batch, V learning_rate = 0.01f, bool lr_per_row_nnz = false,
        bool damp_by_importance = true, V beta2 = 0.999f, V eps = 1e-8f)
    {
        auto xbuf = x.request();
        auto out_grad = _numpy_to_csr_input(dy_ptrs, dy_indices, dy_values, batch, n_outputs());
        std::vector<V> dx(batch * n_inputs(), V(0));
        disldo_backward_sparse_grad<S, FP4BiPacked, COL_TYPE>(
            (V*)xbuf.ptr, batch, weights, out_grad, dx.data(),
            neuron_input_accum.data(), neuron_grad_accum.data(), learning_rate, num_cpus, lr_per_row_nnz,
            damp_by_importance, beta2, eps);
        py::array_t<V> result({(py::ssize_t)batch, (py::ssize_t)n_inputs()});
        std::copy(dx.begin(), dx.end(), (V*)result.request().ptr);
        return result;
    }

    // ── Synaptogenesis (NEW — see class comment) ────────────────────────────────

    void build_probes(S k, bool per_row = false) {
        delta_csr_build_probes<S, FP4BiPacked, COL_TYPE>(
            weights, neuron_input_accum.data(), neuron_grad_accum.data(), k, per_row);
    }
    bool synap_row_step(S current_row, V importance_cutoff, S max_row_weights, S max_prune_per_step = S(8),
                        V importance_eps = V(1e-3)) {
        std::size_t row = static_cast<std::size_t>(current_row);
        return delta_csr_synap_row_step<S, FP4BiPacked, COL_TYPE>(
            weights, row, importance_cutoff, max_row_weights, max_prune_per_step, importance_eps);
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
    bool synap_step(V importance_cutoff, S max_row_weights, S max_prune_per_step = S(8),
                    V importance_eps = V(1e-3)) {
        std::size_t row = static_cast<std::size_t>(_synap_row);
        const bool did = delta_csr_synap_row_step<S, FP4BiPacked, COL_TYPE>(
            weights, row, importance_cutoff, max_row_weights, max_prune_per_step, importance_eps);
        _synap_row = static_cast<S>(row);
        return did;
    }

    // Stateful convenience wrapper around delta_csr_equalize_step (memory
    // rebalancing -- redistributes blank space between neighboring rows'
    // territory so growth headroom stays reasonably even across the
    // layer). Own internal cursor, separate from synap_step's. Also
    // steps block4's own row-headroom equalizer (Block4Store::
    // equalize_step, own independent cursor) -- same idea, applied to
    // block4's tile_data allocation: without this, bytes a row's tile
    // freed by compressing (see block4.hpp: new tiles start empty, sized
    // to real content, not dense-worst-case) stay permanently locked to
    // that one row, unusable anywhere else in the matrix.
    S      _equalize_row = 0;
    S      _block4_equalize_row = 0;
    void equalizer_step() {
        std::size_t row = static_cast<std::size_t>(_equalize_row);
        delta_csr_equalize_step<S, FP4BiPacked, COL_TYPE>(weights.connections, row);
        _equalize_row = static_cast<S>(row);

        std::size_t b4_row = static_cast<std::size_t>(_block4_equalize_row);
        weights.block4.equalize_step(b4_row);
        _block4_equalize_row = static_cast<S>(b4_row);
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
        // compact()/expand_headroom() rebuild via delta_csr_from_absolute
        // internally, which returns a BRAND NEW DeltaCSRWeights -- the
        // limits set at construction don't carry over onto it, silently
        // reverting to unbounded (SIZE_MAX) otherwise. Re-apply every
        // time this member gets reassigned wholesale.
        weights.connections.set_limits(
            _idx_budget_bytes,
            ValueAccessor<FP4BiPacked>::projected_byte_size(_val_budget_nnz));
    }

    // Opposite of compact(): restores growth headroom, normalized to exactly
    // blank_fraction of current content (not "at least" -- see expand() in
    // sparse_struct.hpp). Call before resuming synaptogenesis on a layer
    // that's been compact()ed -- synap_row_step now throws a catchable
    // exception rather than silently doing nothing if headroom is missing.
    void expand_headroom(float blank_fraction = 0.2f) {
        weights.connections = ::expand_headroom<S, FP4BiPacked, COL_TYPE>(weights.connections, blank_fraction);
        // See compact()'s identical comment -- re-apply the cap after
        // every wholesale reassignment of weights.connections.
        weights.connections.set_limits(
            _idx_budget_bytes,
            ValueAccessor<FP4BiPacked>::projected_byte_size(_val_budget_nnz));
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
        // See compact()'s identical comment. NOTE (real, not yet fully
        // closed gap): if min_nnz_per_row * n_rows exceeds the layer's
        // own construction-time budget, THIS call can still allocate
        // past _idx_budget_bytes/_val_budget_nnz once (the internal
        // delta_csr_from_absolute doesn't know about those numbers) --
        // re-applying the cap here doesn't undo that, only prevents any
        // FURTHER growth beyond it afterward. Calling this with a
        // min_nnz_per_row inconsistent with max_weights is still a real
        // way to exceed the intended budget; not fixed here.
        weights.connections.set_limits(
            _idx_budget_bytes,
            ValueAccessor<FP4BiPacked>::projected_byte_size(_val_budget_nnz));
    }

    // Per-ROW scale applied to stored importance/weight to get true units
    // before any arithmetic -- see SparseLinearWeightsDelta's own comment
    // (delta_csr_types.hpp) for the full motivation, including why per-row
    // rather than per-layer. Default 1.0 for any row not yet touched,
    // exact backward compat.
    V get_importance_scale(S row) const { return weights.get_importance_scale(static_cast<std::size_t>(row)); }
    V get_value_scale(S row)      const { return weights.get_value_scale(static_cast<std::size_t>(row)); }
    V get_output_scale(S col)     const { return weights.get_output_scale(static_cast<std::size_t>(col)); }
    V get_value_scale_importance(S row)  const { return weights.get_value_scale_importance(static_cast<std::size_t>(row)); }
    V get_output_scale_importance(S col) const { return weights.get_output_scale_importance(static_cast<std::size_t>(col)); }
    V get_output_importance_scale(S col) const { return weights.get_output_importance_scale(static_cast<std::size_t>(col)); }

    // Per-component (rank>1) accessors -- see scale_rank's own docstring.
    V get_value_scale_k(S row, S k)  const { return weights.get_value_scale_k(static_cast<std::size_t>(row), static_cast<std::size_t>(k)); }
    V get_output_scale_k(S col, S k) const { return weights.get_output_scale_k(static_cast<std::size_t>(col), static_cast<std::size_t>(k)); }
    void set_value_scale_raw_k(S row, S k, V v)  { weights.set_value_scale_raw_k(static_cast<std::size_t>(row), static_cast<std::size_t>(k), v); }
    void set_output_scale_raw_k(S col, S k, V v) { weights.set_output_scale_raw_k(static_cast<std::size_t>(col), static_cast<std::size_t>(k), v); }
    // Combined rank-N scale S(row,col) -- THE quantity Hadamard-multiplied
    // against quant in both forward and backward.
    V get_scale(S row, S col) const { return weights.get_scale(static_cast<std::size_t>(row), static_cast<std::size_t>(col)); }

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
        // If the loaded checkpoint genuinely has more weights than this
        // layer was originally constructed for, idx_budget/val_budget
        // above already grew past _idx_budget_bytes/_val_budget_nnz to
        // fit it (existing behavior, not new) -- update the STORED
        // budget to match so it becomes the real floor for future
        // set_limits() calls (compact()/expand_headroom()) instead of
        // silently shrinking back down to the old, now-too-small one.
        _idx_budget_bytes = idx_budget;
        _val_budget_nnz   = val_budget;
        weights.connections.set_limits(
            _idx_budget_bytes,
            ValueAccessor<FP4BiPacked>::projected_byte_size(_val_budget_nnz));
        weights.recompute_stats();
        // load_weights replaces .connections wholesale, so out_degree
        // (needed by output_scale's own gradient, disldo_backward) must be
        // rebuilt from the new indices -- it isn't touched otherwise and
        // would silently stay at the constructor's all-zero value.
        weights.out_degree.assign(cols, S(0));
        for (S c : idx) ++weights.out_degree[c];
    }

    // Bulk LOADING (not quantization) of already-quantized dense weight/
    // importance codes directly into block4 -- see block4_load_dense's own
    // docstring (delta_csr_memory.hpp) for the full loading-vs-quantization
    // design rationale. weight_codes/importance_codes are row-major n_in x
    // n_out uint8 arrays (already-decided FP4 codes 0-15) -- produce them
    // via fp4_quantize_array() for simple deterministic rounding, or any
    // other scheme; this method does no quantization itself.
    void load_dense_codes(py::array_t<uint8_t> weight_codes, py::array_t<uint8_t> importance_codes) {
        auto wb = weight_codes.request(), ib = importance_codes.request();
        const std::size_t rows = weights.connections.layout.rows;
        const std::size_t cols = weights.connections.layout.cols;
        block4_load_dense<S, FP4BiPacked, COL_TYPE>(
            weights, (const uint8_t*)wb.ptr, (const uint8_t*)ib.ptr, rows, cols);
        // Every row connects to every column -- out_degree[c] = rows for
        // every c, matching load_weights()'s own "rebuild from what was
        // just loaded" precedent (needed for output_scale's gradient, see
        // disldo_backward's out_degree normalization).
        weights.out_degree.assign(cols, S(rows));
    }

    // block4-side counterpart to expand_headroom()/compact() above (which
    // only ever touch weights.connections/scattered CSR -- see
    // block4_expand_headroom's own docstring, delta_csr_memory.hpp, for
    // why block4 needed its own separate pair). expand_block4_headroom
    // gives every block4 row blank_fraction slack to grow into (called
    // automatically by load_dense_codes' underlying block4_load_dense, but
    // exposed directly too for a layer whose block4 content changed some
    // other way, e.g. after compact_block4()). FP4-only, matching
    // block4_expand_headroom's own scope.
    void expand_block4_headroom(float blank_fraction = 0.2f) {
        block4_expand_headroom<S, FP4BiPacked, COL_TYPE>(weights, blank_fraction);
    }
    // Opposite: shrinks every block4 row's tile-byte headroom back down to
    // exactly its current live content, zero slack -- call once a layer's
    // block4 content is done growing (e.g. post-pruning, or a plateaued
    // layer) to reclaim the blank_fraction slack. Call
    // expand_block4_headroom() again afterward before resuming training
    // that needs block4 rows to grow further.
    void compact_block4() {
        block4_compact<S, FP4BiPacked, COL_TYPE>(weights);
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
        delta_csr_combined_to_absolute<S, FP4BiPacked, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)ow.size());
        std::copy(ow.begin(), ow.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<V> get_importance() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_combined_to_absolute<S, FP4BiPacked, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)oimp.size());
        std::copy(oimp.begin(), oimp.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_indices() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_combined_to_absolute<S, FP4BiPacked, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)oi.size());
        std::copy(oi.begin(), oi.end(), (S*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_ptrs() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_combined_to_absolute<S, FP4BiPacked, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)op.size());
        std::copy(op.begin(), op.end(), (S*)result.request().ptr);
        return result;
    }
};

// Concrete instantiations -- SparseLinearLayer is the default (current)
// behavior, unchanged; SparseLinearLayerResync applies the DeferredScaleWrite
// fix (value_scale/output_scale stay consistent with stored codes) that
// SparseLinearLayer8Resync already validated for FP8 -- see
// sili_peridot/JOURNAL.md for why plain FP4 needed this too (true_multi_digit
// real-FP4 collapse to chance vs a fp32-shadow control that succeeded).
using SparseLinearLayer       = SparseLinearLayerImpl<>;
using SparseLinearLayerResync = SparseLinearLayerImpl<RMSpropScalePolicy<float>, true>;
// value_scale/output_scale forced to their init value (1.0) forever, never
// trained -- direct real-hardware test of "zero trained scale" (matches
// sili_peridot's fixed_digit_residual_quantize philosophy: no scale to go
// stale because nothing is separately learned).
using SparseLinearLayerNoScale = SparseLinearLayerImpl<NoScalePolicy<float>, false>;
// Single-variable isolation: SAME RMSprop scale handling as the plain
// default (ScalePolicy/DeferredScaleWrite unchanged), only the weight/
// importance store's rounding changes -- deterministic nearest-neighbour
// (fp4_quantize) instead of stochastic dithered rounding
// (fp4_quantize_stochastic). Tests whether real FP4's per-step rounding
// noise, not any value_scale mechanism, explains true_multi_digit's
// collapse to chance vs the deterministic-rounding fp32-shadow control
// that succeeded -- see sili_peridot/JOURNAL.md.
using SparseLinearLayerDeterministic = SparseLinearLayerImpl<RMSpropScalePolicy<float>, false, false>;
// Full 2x2: deterministic-rounding counterparts of Resync and NoScale too,
// now that the machinery exists -- cheap to add, completes the matrix
// (stochastic/deterministic x plain-scale/resync/noscale) instead of only
// isolating rounding against the plain baseline.
using SparseLinearLayerResyncDeterministic  = SparseLinearLayerImpl<RMSpropScalePolicy<float>, true, false>;
using SparseLinearLayerNoScaleDeterministic = SparseLinearLayerImpl<NoScalePolicy<float>, false, false>;

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
        // Real, enforced cap -- see SparseLinearLayer's identical fix
        // for why (max_indices_bytes/max_values_bytes default to
        // SIZE_MAX otherwise, making reserve_indices/reserve_values a
        // no-op check).
        weights.connections.set_limits(
            _idx_budget_bytes,
            ValueAccessor<VT>::projected_byte_size(_val_budget_nnz));
        weights.recompute_stats();
        weights.probes.rows = n_inputs;
        weights.probes.cols = n_outputs;
        neuron_input_accum.assign(n_inputs,  V(0));
        neuron_grad_accum .assign(n_outputs, V(0));
        weights.out_degree.assign(n_outputs, S(0));
    }

    S n_inputs()  const { return static_cast<S>(weights.connections.layout.rows); }
    S n_outputs() const { return static_cast<S>(weights.connections.layout.cols); }
    S nnz()       const { return static_cast<S>(weights.connections.nnz() + weights.block4.live_synapses()); }
    Block4View block4() { return Block4View(weights.block4); }

    py::array_t<V> forward(py::array_t<V> x) {
        auto xbuf     = x.request();
        _last_batch   = (xbuf.ndim == 2) ? (S)xbuf.shape[0] : 1;
        _last_cols    = (xbuf.ndim == 2) ? (S)xbuf.shape[1] : (S)xbuf.shape[0];

        const V* src  = (V*)xbuf.ptr;
        _last_input.assign(src, src + _last_batch * _last_cols);

        output_buf.assign(_last_batch * n_outputs(), V(0));
        disldo_forward<S, VT, COL_TYPE>(src, _last_batch, _last_cols, weights,
                       output_buf.data(), num_cpus);

        // COPY, not a view -- see SparseLinearLayer::forward_dense's own
        // comment for why (output_buf is reused/overwritten by every
        // future call on this same object).
        py::array_t<V> result({(py::ssize_t)_last_batch, (py::ssize_t)n_outputs()});
        std::copy(output_buf.begin(), output_buf.end(), result.mutable_data());
        return result;
    }

    py::array_t<V> backward(py::array_t<V> dy, V learning_rate, bool lr_per_row_nnz = false,
                             bool damp_by_importance = true, V beta2 = 0.999f, V eps = 1e-8f) {
        auto dybuf = dy.request();
        std::vector<V> dx(_last_batch * _last_cols, V(0));
        // See SparseLinearLayer::backward_dense's identical comment on these
        // trailing 3 args (BoundedRMSpropSynapsePolicy's tuned production default).
        disldo_backward<S, VT, COL_TYPE>(
            _last_input.data(), _last_batch, _last_cols,
            (V*)dybuf.ptr, weights,
            dx.data(),
            neuron_input_accum.data(), neuron_grad_accum.data(),
            learning_rate,
            num_cpus, lr_per_row_nnz, damp_by_importance, beta2, eps,
            0.9f, 0.0f, 2.0f);
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
        const std::size_t idx_budget = std::max(_idx_budget_bytes, idx.size() * 8 + 4096);
        const std::size_t val_budget = std::max(_val_budget_nnz,   idx.size() + 64);
        weights.connections = delta_csr_from_absolute<S, VT, COL_TYPE>(
            p, idx, w, imp_v, rows, cols, idx_budget, val_budget);
        // See SparseLinearLayer::load_weights' identical fix/comment.
        _idx_budget_bytes = idx_budget;
        _val_budget_nnz   = val_budget;
        weights.connections.set_limits(
            _idx_budget_bytes,
            ValueAccessor<VT>::projected_byte_size(_val_budget_nnz));
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
        delta_csr_combined_to_absolute<S, VT, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)ow.size());
        std::copy(ow.begin(), ow.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<V> get_importance() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_combined_to_absolute<S, VT, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)oimp.size());
        std::copy(oimp.begin(), oimp.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_indices() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_combined_to_absolute<S, VT, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)oi.size());
        std::copy(oi.begin(), oi.end(), (S*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_ptrs() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_combined_to_absolute<S, VT, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)op.size());
        std::copy(op.begin(), op.end(), (S*)result.request().ptr);
        return result;
    }
};

// ── SparseLinearLayer8 ──────────────────────────────────────────────────────
// FP8 (OCP MX E4M3, fp8quant.hpp) real quantized storage -- the "alt, not
// replace" FP8 sibling to SparseLinearLayer's production FP4, requested
// after sili_peridot's toy-model quantization sweep validated 8-bit +
// rank-1 scale (weight AND importance both quantized) as consistently,
// substantially better than native FP4 across three task families (see
// sili_peridot's JOURNAL.md). Same disldo_forward/disldo_backward/
// build_probes/synap_row_step functions as SparseLinearLayer/DISLDOLayerV,
// generic via ValueAccessor<FP8BiValues> -- no separate kernel needed for
// the scattered path.
//
// SCOPE, stated plainly rather than silently: this class currently covers
// the SCATTERED path only, matching DISLDOLayerV's own scope exactly
// (weights.block4 stays default-constructed/empty -- never initialized,
// so disldo_forward/disldo_backward's `if (weights.block4.n_tiles() > 0)`
// guards never fire and block4's FP4-specific decode path is never
// touched). Full block4 promotion support (matching SparseLinearLayer's
// combined scattered+dense-tile-SIMD feature parity, per direct
// instruction) is real, substantial follow-up work -- Block4Tile/
// Block4TileHandle's dense-tile storage and accessor API are hardcoded
// to FP4's 1-byte nibble-packed (weight,importance) layout throughout
// (confirmed by reading block4.hpp's Block4TileHandle::at()/tile_data
// indexing directly), and E4M3 needs 2 full bytes/slot -- a real
// structural addition to block4.hpp (new Block4Tile8/Block4Store8 types)
// plus new FP8-dispatch branches inside disldo_forward/disldo_backward's
// existing block4 code sections, not a template-parameter swap. Not
// started here; this class's own docstring/PR should be updated when
// that lands rather than silently claiming feature parity it doesn't
// have yet.
// Templated over ScalePolicy/DeferredScaleWrite (see delta_csr_types.hpp's
// ScalePolicy docstring and linear_disldo.hpp's disldo_backward) so the
// three real, compiled variants below (SparseLinearLayer8 = today's exact
// behavior, SparseLinearLayer8Resync, SparseLinearLayer8AdaMax) share ONE
// implementation instead of three hand-copied classes -- only backward()'s
// disldo_backward<...> call site actually differs between them.
template <typename ScalePolicy = RMSpropScalePolicy<float>, bool DeferredScaleWrite = false>
class SparseLinearLayer8Impl {
public:
    using S = int;
    using V = float;
    using COL_TYPE = uint32_t;
    using VT = FP8BiValues;

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

    SparseLinearLayer8Impl(S n_inputs, S n_outputs, S max_weights, int cpus = 4)
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
        weights.connections.set_limits(
            _idx_budget_bytes,
            ValueAccessor<VT>::projected_byte_size(_val_budget_nnz));
        weights.block4.init(static_cast<std::size_t>(n_inputs), static_cast<std::size_t>(n_outputs));
        // Same budget-sizing convention as SparseLinearLayer's (FP4)
        // constructor -- tile COUNT budget identical (max_weights synapses
        // at max fill of BLOCK4_TILE_SLOTS=16/tile), but multiplied by
        // BLOCK4_TILE_SLOTS8_BYTES (32, a dense FP8 tile's real byte size)
        // instead of FP4's 16, since that's what set_limits' second
        // argument actually budgets (tile_data bytes, not tile count).
        weights.block4.set_limits(
            static_cast<std::size_t>(max_weights) * 8 + 4096,
            std::max<std::size_t>(4, static_cast<std::size_t>(max_weights) / BLOCK4_TILE_SLOTS) * BLOCK4_TILE_SLOTS8_BYTES);
        weights.recompute_stats();
        weights.probes.rows = n_inputs;
        weights.probes.cols = n_outputs;
        neuron_input_accum.assign(n_inputs,  V(0));
        neuron_grad_accum .assign(n_outputs, V(0));
        weights.out_degree.assign(n_outputs, S(0));
    }

    S n_inputs()  const { return static_cast<S>(weights.connections.layout.rows); }
    S n_outputs() const { return static_cast<S>(weights.connections.layout.cols); }
    S nnz()       const { return static_cast<S>(weights.connections.nnz() + weights.block4.live_synapses()); }
    Block4View8 block4() { return Block4View8(weights.block4); }

    // Per-ROW/per-COLUMN (rank-1) scale -- true_w = stored_w *
    // value_scale[row] * output_scale[col], SAME mechanism SparseLinearLayer
    // (FP4) already uses (SparseLinearWeightsDelta's value_scale/
    // output_scale members are VALUES_TYPE-agnostic, not FP4-specific --
    // checked directly in delta_csr_types.hpp before exposing these here).
    // This is the concrete "rank-1 scale" half of the "8-bit + rank-1,
    // weight+importance quantized" scheme validated in sili_peridot's
    // toy-model sweep -- output_scale only becomes gradient-trainable in
    // backward() once set_output_scale_raw has been called at least once
    // (see SparseLinearLayer's own identical set_output_scale_raw docstring).
    V get_value_scale(S row)  const { return weights.get_value_scale(static_cast<std::size_t>(row)); }
    V get_output_scale(S col) const { return weights.get_output_scale(static_cast<std::size_t>(col)); }
    void set_value_scale_raw(S row, V scale)  { weights.set_value_scale_raw(static_cast<std::size_t>(row), scale); }
    void set_output_scale_raw(S col, V scale) { weights.set_output_scale_raw(static_cast<std::size_t>(col), scale); }

    py::array_t<V> forward(py::array_t<V> x) {
        auto xbuf     = x.request();
        _last_batch   = (xbuf.ndim == 2) ? (S)xbuf.shape[0] : 1;
        _last_cols    = (xbuf.ndim == 2) ? (S)xbuf.shape[1] : (S)xbuf.shape[0];

        const V* src  = (V*)xbuf.ptr;
        _last_input.assign(src, src + _last_batch * _last_cols);

        output_buf.assign(_last_batch * n_outputs(), V(0));
        disldo_forward<S, VT, COL_TYPE>(src, _last_batch, _last_cols, weights,
                       output_buf.data(), num_cpus);

        py::array_t<V> result({(py::ssize_t)_last_batch, (py::ssize_t)n_outputs()});
        std::copy(output_buf.begin(), output_buf.end(), result.mutable_data());
        return result;
    }

    py::array_t<V> backward(py::array_t<V> dy, V learning_rate, bool lr_per_row_nnz = false,
                             bool damp_by_importance = true, V beta2 = 0.999f, V eps = 1e-8f) {
        auto dybuf = dy.request();
        std::vector<V> dx(_last_batch * _last_cols, V(0));
        // See SparseLinearLayer::backward_dense's identical comment on these
        // trailing 3 args (BoundedRMSpropSynapsePolicy's tuned production default).
        disldo_backward<S, VT, COL_TYPE, ScalePolicy, DeferredScaleWrite>(
            _last_input.data(), _last_batch, _last_cols,
            (V*)dybuf.ptr, weights,
            dx.data(),
            neuron_input_accum.data(), neuron_grad_accum.data(),
            learning_rate,
            num_cpus, lr_per_row_nnz, damp_by_importance, beta2, eps,
            0.9f, 0.0f, 2.0f);
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
        const std::size_t idx_budget = std::max(_idx_budget_bytes, idx.size() * 8 + 4096);
        const std::size_t val_budget = std::max(_val_budget_nnz,   idx.size() + 64);
        weights.connections = delta_csr_from_absolute<S, VT, COL_TYPE>(
            p, idx, w, imp_v, rows, cols, idx_budget, val_budget);
        _idx_budget_bytes = idx_budget;
        _val_budget_nnz   = val_budget;
        weights.connections.set_limits(
            _idx_budget_bytes,
            ValueAccessor<VT>::projected_byte_size(_val_budget_nnz));
        weights.recompute_stats();
    }

    // See SparseLinearLayerImpl::load_dense_codes' docstring (this file) --
    // same loading-only contract, FP8 (E4M3) codes instead of FP4 nibbles.
    void load_dense_codes(py::array_t<uint8_t> weight_codes, py::array_t<uint8_t> importance_codes) {
        auto wb = weight_codes.request(), ib = importance_codes.request();
        const std::size_t rows = weights.connections.layout.rows;
        const std::size_t cols = weights.connections.layout.cols;
        block4_load_dense<S, VT, COL_TYPE>(
            weights, (const uint8_t*)wb.ptr, (const uint8_t*)ib.ptr, rows, cols);
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
        delta_csr_combined_to_absolute<S, VT, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)ow.size());
        std::copy(ow.begin(), ow.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<V> get_importance() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_combined_to_absolute<S, VT, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<V> result((py::ssize_t)oimp.size());
        std::copy(oimp.begin(), oimp.end(), (V*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_indices() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_combined_to_absolute<S, VT, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)oi.size());
        std::copy(oi.begin(), oi.end(), (S*)result.request().ptr);
        return result;
    }
    py::array_t<S> get_ptrs() {
        std::vector<S> op, oi; std::vector<V> ow, oimp;
        delta_csr_combined_to_absolute<S, VT, COL_TYPE>(weights, op, oi, ow, oimp);
        py::array_t<S> result((py::ssize_t)op.size());
        std::copy(op.begin(), op.end(), (S*)result.request().ptr);
        return result;
    }
};

// Real, compiled variants -- see SparseLinearLayer8Impl's own docstring.
// SparseLinearLayer8 is EXACTLY today's behavior (both defaults), the other
// two are the real (not fake-quantize-simulated) test of whether fixing
// value_scale/output_scale's staleness (Resync) and/or its update rule
// (AdaMax) closes the out-of-context gap found in sili_peridot's toy
// simulation -- see delta_csr_types.hpp's ScalePolicy docstring.
using SparseLinearLayer8       = SparseLinearLayer8Impl<>;
using SparseLinearLayer8Resync = SparseLinearLayer8Impl<RMSpropScalePolicy<float>, true>;
using SparseLinearLayer8AdaMax = SparseLinearLayer8Impl<AdaMaxScalePolicy<float>, true>;

PYBIND11_MODULE(_cpu, m)
{
    // ── Block4View ────────────────────────────────────────────────────────────
    py::class_<Block4View>(m, "Block4View")
        .def_property_readonly("tiles",    &Block4View::tiles,
             "Number of block4 tiles currently promoted -- purely observational.")
        .def_property_readonly("synapses", &Block4View::synapses,
             "Number of synapses currently living in block4 (subset of nnz) --"
             " purely observational.")
        .def_property("switch_point", &Block4View::get_switch_point, &Block4View::set_switch_point,
             "Tiles with <= switch_point live synapses may be packed into the"
             " sparse-encoded tile layout instead of staying fully dense (see"
             " block4.hpp: Block4Store::switch_point, Block4Store::maybe_compress)."
             " 0 disables compression entirely. Default 10.")
        .def_property_readonly("dropped_growth_events", &Block4View::dropped_growth_events,
             "Count of backward/promotion value-updates to an existing tile that"
             " couldn't be persisted because growing its storage would have"
             " exceeded the memory budget -- declined (with the tile keeping its"
             " old value), not thrown, so training keeps running. Nonzero means"
             " some updates are being silently dropped under the current budget.")
        .def_property_readonly("row_merge_overflow_events", &Block4View::row_merge_overflow_events,
             "Count of backward() calls where a row's block4 write-back couldn't"
             " fit its full content within the row's existing headroom, even"
             " after evicting every low-importance synapse it could (each"
             " distinct block-column touched needs >=1 byte of structural"
             " overhead that eviction alone can't shrink away). Declined, not"
             " thrown -- the row's trailing tiles keep their pre-call content"
             " and training keeps running. Nonzero means this layer's"
             " max_row_weights genuinely needs expand_headroom_to() with a"
             " bigger budget.")
        .def_property_readonly("row_merge_overflow_bytes_dropped",
             &Block4View::row_merge_overflow_bytes_dropped,
             "Cumulative bytes of intended row content dropped by"
             " row_merge_overflow_events, across every occurrence.")
        .def_property_readonly("used_bytes", &Block4View::used_bytes,
             "Real bytes of ACTUAL tile content currently live (sparse-vs-dense"
             " per tile, exactly what merge_row_workspace itself would compute)."
             " Compare against n_in*n_out to see the real compression ratio at"
             " the current sparsity level.")
        .def_property_readonly("alloc_bytes", &Block4View::alloc_bytes,
             "Current capacity of the tile_data buffer -- used_bytes plus"
             " whatever per-row growth headroom is currently reserved but not"
             " yet live (e.g. block4_expand_headroom's blank_fraction slack)."
             " The gap (alloc_bytes - used_bytes) is the real memory cost of"
             " allowing rows room to grow.");

    // ── Block4View8 (FP8 counterpart, real bug found+fixed: this class was
    //    never registered here, so SparseLinearLayer8.block4 raised
    //    "Unregistered type" from Python despite compiling fine in C++) ──
    py::class_<Block4View8>(m, "Block4View8")
        .def_property_readonly("tiles",    &Block4View8::tiles,
             "Number of block4 tiles currently promoted -- purely observational.")
        .def_property_readonly("synapses", &Block4View8::synapses,
             "Number of synapses currently living in block4 (subset of nnz) --"
             " purely observational.")
        .def_property("switch_point", &Block4View8::get_switch_point, &Block4View8::set_switch_point,
             "Tiles with <= switch_point live synapses may be packed into the"
             " sparse-encoded tile layout instead of staying fully dense (see"
             " block4.hpp: Block4Store8::switch_point, Block4Store8::maybe_compress)."
             " 0 disables compression entirely. Default BLOCK4_SPARSE_MAX_COUNT8 (12).")
        .def_property_readonly("dropped_growth_events", &Block4View8::dropped_growth_events,
             "Count of backward/promotion value-updates to an existing tile that"
             " couldn't be persisted because growing its storage would have"
             " exceeded the memory budget -- declined (with the tile keeping its"
             " old value), not thrown, so training keeps running. Nonzero means"
             " some updates are being silently dropped under the current budget.")
        .def_property_readonly("row_merge_overflow_events", &Block4View8::row_merge_overflow_events,
             "Count of backward() calls where a row's block4 write-back couldn't"
             " fit its full content within the row's existing headroom, even"
             " after evicting every low-importance synapse it could. Declined,"
             " not thrown -- the row's trailing tiles keep their pre-call"
             " content and training keeps running. Nonzero means this layer's"
             " max_row_weights genuinely needs expand_headroom_to() with a"
             " bigger budget.")
        .def_property_readonly("row_merge_overflow_bytes_dropped",
             &Block4View8::row_merge_overflow_bytes_dropped,
             "Cumulative bytes of intended row content dropped by"
             " row_merge_overflow_events, across every occurrence.");

    // ── SparseLinearLayer ───────────────────────────────────────────────────────────

    py::class_<SparseLinearLayer>(m, "SparseLinearLayer")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("get_scale_rank",       &SparseLinearLayer::get_scale_rank)
        .def("set_scale_rank",       &SparseLinearLayer::set_scale_rank, py::arg("rank"))
        .def("forward_dense",        &SparseLinearLayer::forward_dense,
             py::arg("x"))
        .def("backward_dense",       &SparseLinearLayer::backward_dense,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
             "damp_by_importance=True (default): weight update divided by\n"
             "(sqrt(importance)+eps), where importance is a decayed EMA of\n"
             "g^2 (RMSprop-style) -- a per-synapse adaptive-learning-rate\n"
             "effect. False: raw update, no damping -- importance is still\n"
             "tracked identically either way, only its use to shape the\n"
             "weight step is toggled. For A/B-testing whether the damping\n"
             "itself helps optimization, not for production use. beta2/eps\n"
             "only affect the True case.")
        .def("forward_sparse",       &SparseLinearLayer::forward_sparse,
             py::arg("ptrs"), py::arg("indices"), py::arg("values"),
             py::arg("batch"))
        .def("backward_sparse",      &SparseLinearLayer::backward_sparse,
             py::arg("x"),
             py::arg("dy_ptrs"), py::arg("dy_indices"), py::arg("dy_values"),
             py::arg("batch"), py::arg("learning_rate") = 0.01f, py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("build_probes",         &SparseLinearLayer::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayer::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"),
             py::arg("max_prune_per_step") = 8, py::arg("importance_eps") = 1e-3f)
        .def("synap_step",           &SparseLinearLayer::synap_step,
             py::arg("importance_cutoff"), py::arg("max_row_weights"), py::arg("max_prune_per_step") = 8,
             py::arg("importance_eps") = 1e-3f,
             "Stateful convenience wrapper around synap_row_step -- advances an\n"
             "internal row cursor automatically, so a caller doing repeated\n"
             "one-step-per-call synaptogenesis sweeps doesn't need to track the\n"
             "row index itself. Use synap_row_step directly for explicit control.")
        .def("equalizer_step",       &SparseLinearLayer::equalizer_step,
             "One row of staggered memory redistribution -- call once per\n"
             "synaptogenesis cycle. REDISTRIBUTES the existing pool; does NOT\n"
             "add new memory. Use equalize_to_capacity() first to ensure the\n"
             "pool is large enough for max_row_weights connections per row.\n"
             "Also steps block4's own tile-storage row-headroom equalizer\n"
             "(own independent cursor) -- lets bytes a row's tiles freed by\n"
             "compressing become usable by growth elsewhere in the matrix.")
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
        .def("get_value_scale_k",    &SparseLinearLayer::get_value_scale_k, py::arg("row"), py::arg("k"))
        .def("get_output_scale_k",   &SparseLinearLayer::get_output_scale_k, py::arg("col"), py::arg("k"))
        .def("set_value_scale_raw_k",  &SparseLinearLayer::set_value_scale_raw_k, py::arg("row"), py::arg("k"), py::arg("v"))
        .def("set_output_scale_raw_k", &SparseLinearLayer::set_output_scale_raw_k, py::arg("col"), py::arg("k"), py::arg("v"))
        .def("get_scale",            &SparseLinearLayer::get_scale, py::arg("row"), py::arg("col"),
             "Combined rank-N scale S(row,col) = sum_k value_scale_k(row,k)*"
             "output_scale_k(col,k) -- see scale_rank/set_scale_rank.")
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
        .def("get_output_scale",     &SparseLinearLayer::get_output_scale,
             py::arg("col"),
             "Per-COLUMN (per-output) counterpart to get_value_scale() -- true_w =\n"
             "stored_w * value_scale[row] * output_scale[col]. Default 1.0 for any\n"
             "column not yet touched, exact backward compat with every caller that\n"
             "never sets it.")
        .def("set_output_scale_raw",
             [](SparseLinearLayer& self, int col, float scale) {
                 self.weights.set_output_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Set output_scale[col] directly WITHOUT re-encoding stored weights --\n"
             "same convention as set_value_scale_raw(), but per-output instead of\n"
             "per-input. Calling this at least once makes output_scale\n"
             "gradient-trainable in backward_dense(), like value_scale.")
        .def("get_value_scale_importance",  &SparseLinearLayer::get_value_scale_importance,
             py::arg("row"),
             "Per-row importance backing value_scale's own gradient step, same\n"
             "damping role as a synapse's importance value. Default 0.")
        .def("get_output_scale_importance", &SparseLinearLayer::get_output_scale_importance,
             py::arg("col"),
             "Per-column counterpart for output_scale. Default 0.")
        .def("get_output_importance_scale", &SparseLinearLayer::get_output_importance_scale,
             py::arg("col"),
             "Per-COLUMN counterpart to get_importance_scale() -- true_imp =\n"
             "stored_imp * importance_scale[row] * output_importance_scale[col].\n"
             "Default 1.0, same convention as get_output_scale().")
        .def("set_output_importance_scale_raw",
             [](SparseLinearLayer& self, int col, float scale) {
                 self.weights.set_output_importance_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Same as set_importance_scale_raw() but per-output instead of\n"
             "per-input -- see set_output_scale_raw() for the analogous pattern.")
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
        .def("load_dense_codes",    &SparseLinearLayer::load_dense_codes,
             py::arg("weight_codes"), py::arg("importance_codes"),
             "Bulk-load already-quantized dense FP4 codes directly into block4,\n"
             "bypassing scattered CSR entirely. See block4_load_dense's docstring\n"
             "(delta_csr_memory.hpp) -- loading only, no quantization performed\n"
             "here; produce codes via fp4_quantize_array() or any other scheme.")
        .def("expand_block4_headroom", &SparseLinearLayer::expand_block4_headroom,
             py::arg("blank_fraction") = 0.2f,
             "block4-side counterpart to expand_headroom() -- that only ever\n"
             "touches scattered CSR. Gives every block4 row blank_fraction slack\n"
             "to grow into.")
        .def("compact_block4", &SparseLinearLayer::compact_block4,
             "Opposite of expand_block4_headroom(): shrinks every block4 row's\n"
             "headroom back down to exactly its current live content.")
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
        .def_property_readonly("block4",    &SparseLinearLayer::block4,
             py::keep_alive<0, 1>(),
             "Purely observational view onto this layer's block4 storage --"
             " layer.block4.tiles / layer.block4.synapses.")
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


    // ── SparseLinearLayerResync ──────────────────────────────────────────────────
    // Same API as SparseLinearLayer, DeferredScaleWrite=true fix (see class-level
    // comment above SparseLinearLayerImpl).

    py::class_<SparseLinearLayerResync>(m, "SparseLinearLayerResync")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("forward_dense",        &SparseLinearLayerResync::forward_dense,
             py::arg("x"))
        .def("backward_dense",       &SparseLinearLayerResync::backward_dense,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
             "damp_by_importance=True (default): weight update divided by\n"
             "(sqrt(importance)+eps), where importance is a decayed EMA of\n"
             "g^2 (RMSprop-style) -- a per-synapse adaptive-learning-rate\n"
             "effect. False: raw update, no damping -- importance is still\n"
             "tracked identically either way, only its use to shape the\n"
             "weight step is toggled. For A/B-testing whether the damping\n"
             "itself helps optimization, not for production use. beta2/eps\n"
             "only affect the True case.")
        .def("forward_sparse",       &SparseLinearLayerResync::forward_sparse,
             py::arg("ptrs"), py::arg("indices"), py::arg("values"),
             py::arg("batch"))
        .def("backward_sparse",      &SparseLinearLayerResync::backward_sparse,
             py::arg("x"),
             py::arg("dy_ptrs"), py::arg("dy_indices"), py::arg("dy_values"),
             py::arg("batch"), py::arg("learning_rate") = 0.01f, py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("build_probes",         &SparseLinearLayerResync::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayerResync::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"),
             py::arg("max_prune_per_step") = 8, py::arg("importance_eps") = 1e-3f)
        .def("synap_step",           &SparseLinearLayerResync::synap_step,
             py::arg("importance_cutoff"), py::arg("max_row_weights"), py::arg("max_prune_per_step") = 8,
             py::arg("importance_eps") = 1e-3f,
             "Stateful convenience wrapper around synap_row_step -- advances an\n"
             "internal row cursor automatically, so a caller doing repeated\n"
             "one-step-per-call synaptogenesis sweeps doesn't need to track the\n"
             "row index itself. Use synap_row_step directly for explicit control.")
        .def("equalizer_step",       &SparseLinearLayerResync::equalizer_step,
             "One row of staggered memory redistribution -- call once per\n"
             "synaptogenesis cycle. REDISTRIBUTES the existing pool; does NOT\n"
             "add new memory. Use equalize_to_capacity() first to ensure the\n"
             "pool is large enough for max_row_weights connections per row.\n"
             "Also steps block4's own tile-storage row-headroom equalizer\n"
             "(own independent cursor) -- lets bytes a row's tiles freed by\n"
             "compressing become usable by growth elsewhere in the matrix.")
        .def("equalize_to_capacity", &SparseLinearLayerResync::equalize_to_capacity,
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
        .def("compact",              &SparseLinearLayerResync::compact,
             "Repack in place: every row occupies exactly its active bytes/elements,\n"
             "zero inter-row blank space. Call before saving/measuring a freshly\n"
             "converted model. Zeroes growth headroom.")
        .def("expand_headroom",      &SparseLinearLayerResync::expand_headroom,
             py::arg("blank_fraction") = 0.2f,
             "Restore per-row growth headroom (proportional to current nnz).\n"
             "WARNING: use expand_headroom_to(max_row_weights) after a prune\n"
             "cycle -- plain expand_headroom allocates headroom proportional\n"
             "to CURRENT nnz, leaving no room to grow back to max_row_weights.")
        .def("expand_headroom_to",   &SparseLinearLayerResync::expand_headroom_to,
             py::arg("min_nnz_per_row"), py::arg("blank_fraction") = 0.2f,
             "Like expand_headroom() but guarantees each row has headroom for\n"
             "at least min_nnz_per_row connections. Call with max_row_weights\n"
             "before synaptogenesis after a prune cycle.")
        .def("get_importance_scale", &SparseLinearLayerResync::get_importance_scale,
             py::arg("row"),
             "Per-ROW scale applied to that row's stored importance to get true\n"
             "units. Default 1.0 for any row not yet touched, exact backward\n"
             "compat. Use rescale_importance_row()/rescale_importance() to\n"
             "change it, never assign directly.")
        .def("get_value_scale",      &SparseLinearLayerResync::get_value_scale,
             py::arg("row"),
             "Same as get_importance_scale() but for stored weight values.")
        .def("set_value_scale_raw",
             [](SparseLinearLayerResync& self, int row, float scale) {
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
             [](SparseLinearLayerResync& self, int row, float scale) {
                 self.weights.set_importance_scale_raw(
                     static_cast<std::size_t>(row), scale);
             },
             py::arg("row"), py::arg("scale"),
             "Same as set_value_scale_raw() but for importance.")
        .def("get_output_scale",     &SparseLinearLayerResync::get_output_scale,
             py::arg("col"),
             "Per-COLUMN (per-output) counterpart to get_value_scale() -- true_w =\n"
             "stored_w * value_scale[row] * output_scale[col]. Default 1.0 for any\n"
             "column not yet touched, exact backward compat with every caller that\n"
             "never sets it.")
        .def("set_output_scale_raw",
             [](SparseLinearLayerResync& self, int col, float scale) {
                 self.weights.set_output_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Set output_scale[col] directly WITHOUT re-encoding stored weights --\n"
             "same convention as set_value_scale_raw(), but per-output instead of\n"
             "per-input. Calling this at least once makes output_scale\n"
             "gradient-trainable in backward_dense(), like value_scale.")
        .def("get_value_scale_importance",  &SparseLinearLayerResync::get_value_scale_importance,
             py::arg("row"),
             "Per-row importance backing value_scale's own gradient step, same\n"
             "damping role as a synapse's importance value. Default 0.")
        .def("get_output_scale_importance", &SparseLinearLayerResync::get_output_scale_importance,
             py::arg("col"),
             "Per-column counterpart for output_scale. Default 0.")
        .def("get_output_importance_scale", &SparseLinearLayerResync::get_output_importance_scale,
             py::arg("col"),
             "Per-COLUMN counterpart to get_importance_scale() -- true_imp =\n"
             "stored_imp * importance_scale[row] * output_importance_scale[col].\n"
             "Default 1.0, same convention as get_output_scale().")
        .def("set_output_importance_scale_raw",
             [](SparseLinearLayerResync& self, int col, float scale) {
                 self.weights.set_output_importance_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Same as set_importance_scale_raw() but per-output instead of\n"
             "per-input -- see set_output_scale_raw() for the analogous pattern.")
        .def("rescale_importance_row", &SparseLinearLayerResync::rescale_importance_row,
             py::arg("row"), py::arg("new_scale"),
             "Change ONE row's importance scale mid-training without corrupting\n"
             "that row's existing stored data -- re-reads at the OLD per-row\n"
             "scale, re-encodes at the NEW one.")
        .def("rescale_value_row",    &SparseLinearLayerResync::rescale_value_row,
             py::arg("row"), py::arg("new_scale"),
             "Same as rescale_importance_row() but for stored weight values.")
        .def("rescale_importance",   &SparseLinearLayerResync::rescale_importance,
             py::arg("new_scale"),
             "Bulk convenience: set EVERY row's importance scale to the same\n"
             "value. Backward-compatible interface with the original\n"
             "per-layer-scalar design.")
        .def("rescale_value",        &SparseLinearLayerResync::rescale_value,
             py::arg("new_scale"),
             "Same as rescale_importance() but for stored weight values.")
        .def_property_readonly("value_l1",           &SparseLinearLayerResync::get_value_l1)
        .def_property_readonly("value_l2_sq",        &SparseLinearLayerResync::get_value_l2_sq)
        .def_property_readonly("value_max_abs",      &SparseLinearLayerResync::get_value_max_abs)
        .def_property_readonly("importance_l1",      &SparseLinearLayerResync::get_importance_l1)
        .def_property_readonly("importance_l2_sq",   &SparseLinearLayerResync::get_importance_l2_sq)
        .def_property_readonly("importance_max_abs", &SparseLinearLayerResync::get_importance_max_abs)
        .def("hoyer_value",          &SparseLinearLayerResync::hoyer_value,
             "Hoyer's sparsity measure on the STORED weight distribution, in\n"
             "[0,1] -- 0 means values spread evenly across FP4's representable\n"
             "range, 1 means concentrated (e.g. mostly zero, or mostly clustered\n"
             "at one magnitude). O(1), from running stats -- see value_max_abs\n"
             "for a cheap complementary saturation check this doesn't catch.")
        .def("hoyer_importance",     &SparseLinearLayerResync::hoyer_importance,
             "Same as hoyer_value() but for the STORED importance distribution.")
        .def("recompute_stats",      &SparseLinearLayerResync::recompute_stats,
             "Recompute value_l1/l2_sq/max_abs and importance_l1/l2_sq/max_abs\n"
             "from scratch, O(nnz). Gives an exact max_abs (the incrementally-\n"
             "maintained one is a monotonic upper bound, not a live exact max).")
        .def("zero_accum",           &SparseLinearLayerResync::zero_accum)
        .def_property_readonly("neuron_input_accum", &SparseLinearLayerResync::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &SparseLinearLayerResync::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &SparseLinearLayerResync::get_weights_vals)
        .def_property_readonly("importance",         &SparseLinearLayerResync::get_importance)
        .def_property_readonly("indices",            &SparseLinearLayerResync::get_indices)
        .def_property_readonly("ptrs",               &SparseLinearLayerResync::get_ptrs)
        .def("load_weights",        &SparseLinearLayerResync::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"))
        .def("load_dense_codes",    &SparseLinearLayerResync::load_dense_codes,
             py::arg("weight_codes"), py::arg("importance_codes"))
        .def_property_readonly("out_degree", [](const SparseLinearLayerResync& self) {
            return py::array_t<SparseLinearLayerResync::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(SparseLinearLayerResync::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &SparseLinearLayerResync::num_cpus)
        .def_property_readonly("n_inputs",  &SparseLinearLayerResync::n_inputs)
        .def_property_readonly("n_outputs", &SparseLinearLayerResync::n_outputs)
        .def_property_readonly("nnz",       &SparseLinearLayerResync::nnz)
        .def_property_readonly("block4",    &SparseLinearLayerResync::block4,
             py::keep_alive<0, 1>(),
             "Purely observational view onto this layer's block4 storage --"
             " layer.block4.tiles / layer.block4.synapses.")
        .def_property_readonly("last_input",
            [](const SparseLinearLayerResync& self) -> py::object {
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


    // ── SparseLinearLayerNoScale ─────────────────────────────────────────────────
    // Same API as SparseLinearLayer, value_scale/output_scale permanently 1.0,
    // never trained -- direct test of the "zero trained scale" hypothesis on
    // real FP4 hardware storage (see NoScalePolicy comment, delta_csr_types.hpp).

    py::class_<SparseLinearLayerNoScale>(m, "SparseLinearLayerNoScale")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("forward_dense",        &SparseLinearLayerNoScale::forward_dense,
             py::arg("x"))
        .def("backward_dense",       &SparseLinearLayerNoScale::backward_dense,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
             "damp_by_importance=True (default): weight update divided by\n"
             "(sqrt(importance)+eps), where importance is a decayed EMA of\n"
             "g^2 (RMSprop-style) -- a per-synapse adaptive-learning-rate\n"
             "effect. False: raw update, no damping -- importance is still\n"
             "tracked identically either way, only its use to shape the\n"
             "weight step is toggled. For A/B-testing whether the damping\n"
             "itself helps optimization, not for production use. beta2/eps\n"
             "only affect the True case.")
        .def("forward_sparse",       &SparseLinearLayerNoScale::forward_sparse,
             py::arg("ptrs"), py::arg("indices"), py::arg("values"),
             py::arg("batch"))
        .def("backward_sparse",      &SparseLinearLayerNoScale::backward_sparse,
             py::arg("x"),
             py::arg("dy_ptrs"), py::arg("dy_indices"), py::arg("dy_values"),
             py::arg("batch"), py::arg("learning_rate") = 0.01f, py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("build_probes",         &SparseLinearLayerNoScale::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayerNoScale::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"),
             py::arg("max_prune_per_step") = 8, py::arg("importance_eps") = 1e-3f)
        .def("synap_step",           &SparseLinearLayerNoScale::synap_step,
             py::arg("importance_cutoff"), py::arg("max_row_weights"), py::arg("max_prune_per_step") = 8,
             py::arg("importance_eps") = 1e-3f,
             "Stateful convenience wrapper around synap_row_step -- advances an\n"
             "internal row cursor automatically, so a caller doing repeated\n"
             "one-step-per-call synaptogenesis sweeps doesn't need to track the\n"
             "row index itself. Use synap_row_step directly for explicit control.")
        .def("equalizer_step",       &SparseLinearLayerNoScale::equalizer_step,
             "One row of staggered memory redistribution -- call once per\n"
             "synaptogenesis cycle. REDISTRIBUTES the existing pool; does NOT\n"
             "add new memory. Use equalize_to_capacity() first to ensure the\n"
             "pool is large enough for max_row_weights connections per row.\n"
             "Also steps block4's own tile-storage row-headroom equalizer\n"
             "(own independent cursor) -- lets bytes a row's tiles freed by\n"
             "compressing become usable by growth elsewhere in the matrix.")
        .def("equalize_to_capacity", &SparseLinearLayerNoScale::equalize_to_capacity,
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
        .def("compact",              &SparseLinearLayerNoScale::compact,
             "Repack in place: every row occupies exactly its active bytes/elements,\n"
             "zero inter-row blank space. Call before saving/measuring a freshly\n"
             "converted model. Zeroes growth headroom.")
        .def("expand_headroom",      &SparseLinearLayerNoScale::expand_headroom,
             py::arg("blank_fraction") = 0.2f,
             "Restore per-row growth headroom (proportional to current nnz).\n"
             "WARNING: use expand_headroom_to(max_row_weights) after a prune\n"
             "cycle -- plain expand_headroom allocates headroom proportional\n"
             "to CURRENT nnz, leaving no room to grow back to max_row_weights.")
        .def("expand_headroom_to",   &SparseLinearLayerNoScale::expand_headroom_to,
             py::arg("min_nnz_per_row"), py::arg("blank_fraction") = 0.2f,
             "Like expand_headroom() but guarantees each row has headroom for\n"
             "at least min_nnz_per_row connections. Call with max_row_weights\n"
             "before synaptogenesis after a prune cycle.")
        .def("get_importance_scale", &SparseLinearLayerNoScale::get_importance_scale,
             py::arg("row"),
             "Per-ROW scale applied to that row's stored importance to get true\n"
             "units. Default 1.0 for any row not yet touched, exact backward\n"
             "compat. Use rescale_importance_row()/rescale_importance() to\n"
             "change it, never assign directly.")
        .def("get_value_scale",      &SparseLinearLayerNoScale::get_value_scale,
             py::arg("row"),
             "Same as get_importance_scale() but for stored weight values.")
        .def("set_value_scale_raw",
             [](SparseLinearLayerNoScale& self, int row, float scale) {
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
             [](SparseLinearLayerNoScale& self, int row, float scale) {
                 self.weights.set_importance_scale_raw(
                     static_cast<std::size_t>(row), scale);
             },
             py::arg("row"), py::arg("scale"),
             "Same as set_value_scale_raw() but for importance.")
        .def("get_output_scale",     &SparseLinearLayerNoScale::get_output_scale,
             py::arg("col"),
             "Per-COLUMN (per-output) counterpart to get_value_scale() -- true_w =\n"
             "stored_w * value_scale[row] * output_scale[col]. Default 1.0 for any\n"
             "column not yet touched, exact backward compat with every caller that\n"
             "never sets it.")
        .def("set_output_scale_raw",
             [](SparseLinearLayerNoScale& self, int col, float scale) {
                 self.weights.set_output_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Set output_scale[col] directly WITHOUT re-encoding stored weights --\n"
             "same convention as set_value_scale_raw(), but per-output instead of\n"
             "per-input. Calling this at least once makes output_scale\n"
             "gradient-trainable in backward_dense(), like value_scale.")
        .def("get_value_scale_importance",  &SparseLinearLayerNoScale::get_value_scale_importance,
             py::arg("row"),
             "Per-row importance backing value_scale's own gradient step, same\n"
             "damping role as a synapse's importance value. Default 0.")
        .def("get_output_scale_importance", &SparseLinearLayerNoScale::get_output_scale_importance,
             py::arg("col"),
             "Per-column counterpart for output_scale. Default 0.")
        .def("get_output_importance_scale", &SparseLinearLayerNoScale::get_output_importance_scale,
             py::arg("col"),
             "Per-COLUMN counterpart to get_importance_scale() -- true_imp =\n"
             "stored_imp * importance_scale[row] * output_importance_scale[col].\n"
             "Default 1.0, same convention as get_output_scale().")
        .def("set_output_importance_scale_raw",
             [](SparseLinearLayerNoScale& self, int col, float scale) {
                 self.weights.set_output_importance_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Same as set_importance_scale_raw() but per-output instead of\n"
             "per-input -- see set_output_scale_raw() for the analogous pattern.")
        .def("rescale_importance_row", &SparseLinearLayerNoScale::rescale_importance_row,
             py::arg("row"), py::arg("new_scale"),
             "Change ONE row's importance scale mid-training without corrupting\n"
             "that row's existing stored data -- re-reads at the OLD per-row\n"
             "scale, re-encodes at the NEW one.")
        .def("rescale_value_row",    &SparseLinearLayerNoScale::rescale_value_row,
             py::arg("row"), py::arg("new_scale"),
             "Same as rescale_importance_row() but for stored weight values.")
        .def("rescale_importance",   &SparseLinearLayerNoScale::rescale_importance,
             py::arg("new_scale"),
             "Bulk convenience: set EVERY row's importance scale to the same\n"
             "value. Backward-compatible interface with the original\n"
             "per-layer-scalar design.")
        .def("rescale_value",        &SparseLinearLayerNoScale::rescale_value,
             py::arg("new_scale"),
             "Same as rescale_importance() but for stored weight values.")
        .def_property_readonly("value_l1",           &SparseLinearLayerNoScale::get_value_l1)
        .def_property_readonly("value_l2_sq",        &SparseLinearLayerNoScale::get_value_l2_sq)
        .def_property_readonly("value_max_abs",      &SparseLinearLayerNoScale::get_value_max_abs)
        .def_property_readonly("importance_l1",      &SparseLinearLayerNoScale::get_importance_l1)
        .def_property_readonly("importance_l2_sq",   &SparseLinearLayerNoScale::get_importance_l2_sq)
        .def_property_readonly("importance_max_abs", &SparseLinearLayerNoScale::get_importance_max_abs)
        .def("hoyer_value",          &SparseLinearLayerNoScale::hoyer_value,
             "Hoyer's sparsity measure on the STORED weight distribution, in\n"
             "[0,1] -- 0 means values spread evenly across FP4's representable\n"
             "range, 1 means concentrated (e.g. mostly zero, or mostly clustered\n"
             "at one magnitude). O(1), from running stats -- see value_max_abs\n"
             "for a cheap complementary saturation check this doesn't catch.")
        .def("hoyer_importance",     &SparseLinearLayerNoScale::hoyer_importance,
             "Same as hoyer_value() but for the STORED importance distribution.")
        .def("recompute_stats",      &SparseLinearLayerNoScale::recompute_stats,
             "Recompute value_l1/l2_sq/max_abs and importance_l1/l2_sq/max_abs\n"
             "from scratch, O(nnz). Gives an exact max_abs (the incrementally-\n"
             "maintained one is a monotonic upper bound, not a live exact max).")
        .def("zero_accum",           &SparseLinearLayerNoScale::zero_accum)
        .def_property_readonly("neuron_input_accum", &SparseLinearLayerNoScale::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &SparseLinearLayerNoScale::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &SparseLinearLayerNoScale::get_weights_vals)
        .def_property_readonly("importance",         &SparseLinearLayerNoScale::get_importance)
        .def_property_readonly("indices",            &SparseLinearLayerNoScale::get_indices)
        .def_property_readonly("ptrs",               &SparseLinearLayerNoScale::get_ptrs)
        .def("load_weights",        &SparseLinearLayerNoScale::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"))
        .def("load_dense_codes",    &SparseLinearLayerNoScale::load_dense_codes,
             py::arg("weight_codes"), py::arg("importance_codes"))
        .def_property_readonly("out_degree", [](const SparseLinearLayerNoScale& self) {
            return py::array_t<SparseLinearLayerNoScale::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(SparseLinearLayerNoScale::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &SparseLinearLayerNoScale::num_cpus)
        .def_property_readonly("n_inputs",  &SparseLinearLayerNoScale::n_inputs)
        .def_property_readonly("n_outputs", &SparseLinearLayerNoScale::n_outputs)
        .def_property_readonly("nnz",       &SparseLinearLayerNoScale::nnz)
        .def_property_readonly("block4",    &SparseLinearLayerNoScale::block4,
             py::keep_alive<0, 1>(),
             "Purely observational view onto this layer's block4 storage --"
             " layer.block4.tiles / layer.block4.synapses.")
        .def_property_readonly("last_input",
            [](const SparseLinearLayerNoScale& self) -> py::object {
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



    // ── SparseLinearLayerDeterministic ───────────────────────────────────────────
    // Same API as SparseLinearLayer, StochasticRounding=false only -- see
    // StochasticRounding's docstring on disldo_backward, linear_disldo.hpp.

    py::class_<SparseLinearLayerDeterministic>(m, "SparseLinearLayerDeterministic")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("get_scale_rank",       &SparseLinearLayerDeterministic::get_scale_rank)
        .def("set_scale_rank",       &SparseLinearLayerDeterministic::set_scale_rank, py::arg("rank"))
        .def("forward_dense",        &SparseLinearLayerDeterministic::forward_dense,
             py::arg("x"))
        .def("backward_dense",       &SparseLinearLayerDeterministic::backward_dense,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
             "damp_by_importance=True (default): weight update divided by\n"
             "(sqrt(importance)+eps), where importance is a decayed EMA of\n"
             "g^2 (RMSprop-style) -- a per-synapse adaptive-learning-rate\n"
             "effect. False: raw update, no damping -- importance is still\n"
             "tracked identically either way, only its use to shape the\n"
             "weight step is toggled. For A/B-testing whether the damping\n"
             "itself helps optimization, not for production use. beta2/eps\n"
             "only affect the True case.")
        .def("forward_sparse",       &SparseLinearLayerDeterministic::forward_sparse,
             py::arg("ptrs"), py::arg("indices"), py::arg("values"),
             py::arg("batch"))
        .def("backward_sparse",      &SparseLinearLayerDeterministic::backward_sparse,
             py::arg("x"),
             py::arg("dy_ptrs"), py::arg("dy_indices"), py::arg("dy_values"),
             py::arg("batch"), py::arg("learning_rate") = 0.01f, py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("build_probes",         &SparseLinearLayerDeterministic::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayerDeterministic::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"),
             py::arg("max_prune_per_step") = 8, py::arg("importance_eps") = 1e-3f)
        .def("synap_step",           &SparseLinearLayerDeterministic::synap_step,
             py::arg("importance_cutoff"), py::arg("max_row_weights"), py::arg("max_prune_per_step") = 8,
             py::arg("importance_eps") = 1e-3f,
             "Stateful convenience wrapper around synap_row_step -- advances an\n"
             "internal row cursor automatically, so a caller doing repeated\n"
             "one-step-per-call synaptogenesis sweeps doesn't need to track the\n"
             "row index itself. Use synap_row_step directly for explicit control.")
        .def("equalizer_step",       &SparseLinearLayerDeterministic::equalizer_step,
             "One row of staggered memory redistribution -- call once per\n"
             "synaptogenesis cycle. REDISTRIBUTES the existing pool; does NOT\n"
             "add new memory. Use equalize_to_capacity() first to ensure the\n"
             "pool is large enough for max_row_weights connections per row.\n"
             "Also steps block4's own tile-storage row-headroom equalizer\n"
             "(own independent cursor) -- lets bytes a row's tiles freed by\n"
             "compressing become usable by growth elsewhere in the matrix.")
        .def("equalize_to_capacity", &SparseLinearLayerDeterministic::equalize_to_capacity,
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
        .def("compact",              &SparseLinearLayerDeterministic::compact,
             "Repack in place: every row occupies exactly its active bytes/elements,\n"
             "zero inter-row blank space. Call before saving/measuring a freshly\n"
             "converted model. Zeroes growth headroom.")
        .def("expand_headroom",      &SparseLinearLayerDeterministic::expand_headroom,
             py::arg("blank_fraction") = 0.2f,
             "Restore per-row growth headroom (proportional to current nnz).\n"
             "WARNING: use expand_headroom_to(max_row_weights) after a prune\n"
             "cycle -- plain expand_headroom allocates headroom proportional\n"
             "to CURRENT nnz, leaving no room to grow back to max_row_weights.")
        .def("expand_headroom_to",   &SparseLinearLayerDeterministic::expand_headroom_to,
             py::arg("min_nnz_per_row"), py::arg("blank_fraction") = 0.2f,
             "Like expand_headroom() but guarantees each row has headroom for\n"
             "at least min_nnz_per_row connections. Call with max_row_weights\n"
             "before synaptogenesis after a prune cycle.")
        .def("get_importance_scale", &SparseLinearLayerDeterministic::get_importance_scale,
             py::arg("row"),
             "Per-ROW scale applied to that row's stored importance to get true\n"
             "units. Default 1.0 for any row not yet touched, exact backward\n"
             "compat. Use rescale_importance_row()/rescale_importance() to\n"
             "change it, never assign directly.")
        .def("get_value_scale",      &SparseLinearLayerDeterministic::get_value_scale,
             py::arg("row"),
             "Same as get_importance_scale() but for stored weight values.")
        .def("get_value_scale_k",    &SparseLinearLayerDeterministic::get_value_scale_k, py::arg("row"), py::arg("k"))
        .def("get_output_scale_k",   &SparseLinearLayerDeterministic::get_output_scale_k, py::arg("col"), py::arg("k"))
        .def("set_value_scale_raw_k",  &SparseLinearLayerDeterministic::set_value_scale_raw_k, py::arg("row"), py::arg("k"), py::arg("v"))
        .def("set_output_scale_raw_k", &SparseLinearLayerDeterministic::set_output_scale_raw_k, py::arg("col"), py::arg("k"), py::arg("v"))
        .def("get_scale",            &SparseLinearLayerDeterministic::get_scale, py::arg("row"), py::arg("col"),
             "Combined rank-N scale S(row,col) = sum_k value_scale_k(row,k)*"
             "output_scale_k(col,k) -- see scale_rank/set_scale_rank.")
        .def("set_value_scale_raw",
             [](SparseLinearLayerDeterministic& self, int row, float scale) {
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
             [](SparseLinearLayerDeterministic& self, int row, float scale) {
                 self.weights.set_importance_scale_raw(
                     static_cast<std::size_t>(row), scale);
             },
             py::arg("row"), py::arg("scale"),
             "Same as set_value_scale_raw() but for importance.")
        .def("get_output_scale",     &SparseLinearLayerDeterministic::get_output_scale,
             py::arg("col"),
             "Per-COLUMN (per-output) counterpart to get_value_scale() -- true_w =\n"
             "stored_w * value_scale[row] * output_scale[col]. Default 1.0 for any\n"
             "column not yet touched, exact backward compat with every caller that\n"
             "never sets it.")
        .def("set_output_scale_raw",
             [](SparseLinearLayerDeterministic& self, int col, float scale) {
                 self.weights.set_output_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Set output_scale[col] directly WITHOUT re-encoding stored weights --\n"
             "same convention as set_value_scale_raw(), but per-output instead of\n"
             "per-input. Calling this at least once makes output_scale\n"
             "gradient-trainable in backward_dense(), like value_scale.")
        .def("get_value_scale_importance",  &SparseLinearLayerDeterministic::get_value_scale_importance,
             py::arg("row"),
             "Per-row importance backing value_scale's own gradient step, same\n"
             "damping role as a synapse's importance value. Default 0.")
        .def("get_output_scale_importance", &SparseLinearLayerDeterministic::get_output_scale_importance,
             py::arg("col"),
             "Per-column counterpart for output_scale. Default 0.")
        .def("get_output_importance_scale", &SparseLinearLayerDeterministic::get_output_importance_scale,
             py::arg("col"),
             "Per-COLUMN counterpart to get_importance_scale() -- true_imp =\n"
             "stored_imp * importance_scale[row] * output_importance_scale[col].\n"
             "Default 1.0, same convention as get_output_scale().")
        .def("set_output_importance_scale_raw",
             [](SparseLinearLayerDeterministic& self, int col, float scale) {
                 self.weights.set_output_importance_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Same as set_importance_scale_raw() but per-output instead of\n"
             "per-input -- see set_output_scale_raw() for the analogous pattern.")
        .def("rescale_importance_row", &SparseLinearLayerDeterministic::rescale_importance_row,
             py::arg("row"), py::arg("new_scale"),
             "Change ONE row's importance scale mid-training without corrupting\n"
             "that row's existing stored data -- re-reads at the OLD per-row\n"
             "scale, re-encodes at the NEW one.")
        .def("rescale_value_row",    &SparseLinearLayerDeterministic::rescale_value_row,
             py::arg("row"), py::arg("new_scale"),
             "Same as rescale_importance_row() but for stored weight values.")
        .def("rescale_importance",   &SparseLinearLayerDeterministic::rescale_importance,
             py::arg("new_scale"),
             "Bulk convenience: set EVERY row's importance scale to the same\n"
             "value. Backward-compatible interface with the original\n"
             "per-layer-scalar design.")
        .def("rescale_value",        &SparseLinearLayerDeterministic::rescale_value,
             py::arg("new_scale"),
             "Same as rescale_importance() but for stored weight values.")
        .def_property_readonly("value_l1",           &SparseLinearLayerDeterministic::get_value_l1)
        .def_property_readonly("value_l2_sq",        &SparseLinearLayerDeterministic::get_value_l2_sq)
        .def_property_readonly("value_max_abs",      &SparseLinearLayerDeterministic::get_value_max_abs)
        .def_property_readonly("importance_l1",      &SparseLinearLayerDeterministic::get_importance_l1)
        .def_property_readonly("importance_l2_sq",   &SparseLinearLayerDeterministic::get_importance_l2_sq)
        .def_property_readonly("importance_max_abs", &SparseLinearLayerDeterministic::get_importance_max_abs)
        .def("hoyer_value",          &SparseLinearLayerDeterministic::hoyer_value,
             "Hoyer's sparsity measure on the STORED weight distribution, in\n"
             "[0,1] -- 0 means values spread evenly across FP4's representable\n"
             "range, 1 means concentrated (e.g. mostly zero, or mostly clustered\n"
             "at one magnitude). O(1), from running stats -- see value_max_abs\n"
             "for a cheap complementary saturation check this doesn't catch.")
        .def("hoyer_importance",     &SparseLinearLayerDeterministic::hoyer_importance,
             "Same as hoyer_value() but for the STORED importance distribution.")
        .def("recompute_stats",      &SparseLinearLayerDeterministic::recompute_stats,
             "Recompute value_l1/l2_sq/max_abs and importance_l1/l2_sq/max_abs\n"
             "from scratch, O(nnz). Gives an exact max_abs (the incrementally-\n"
             "maintained one is a monotonic upper bound, not a live exact max).")
        .def("zero_accum",           &SparseLinearLayerDeterministic::zero_accum)
        .def_property_readonly("neuron_input_accum", &SparseLinearLayerDeterministic::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &SparseLinearLayerDeterministic::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &SparseLinearLayerDeterministic::get_weights_vals)
        .def_property_readonly("importance",         &SparseLinearLayerDeterministic::get_importance)
        .def_property_readonly("indices",            &SparseLinearLayerDeterministic::get_indices)
        .def_property_readonly("ptrs",               &SparseLinearLayerDeterministic::get_ptrs)
        .def("load_weights",        &SparseLinearLayerDeterministic::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"))
        .def("load_dense_codes",    &SparseLinearLayerDeterministic::load_dense_codes,
             py::arg("weight_codes"), py::arg("importance_codes"))
        .def("expand_block4_headroom", &SparseLinearLayerDeterministic::expand_block4_headroom,
             py::arg("blank_fraction") = 0.2f,
             "block4-side counterpart to expand_headroom() -- that only ever\n"
             "touches scattered CSR. Gives every block4 row blank_fraction slack\n"
             "to grow into.")
        .def("compact_block4", &SparseLinearLayerDeterministic::compact_block4,
             "Opposite of expand_block4_headroom(): shrinks every block4 row's\n"
             "headroom back down to exactly its current live content.")
        .def_property_readonly("out_degree", [](const SparseLinearLayerDeterministic& self) {
            return py::array_t<SparseLinearLayerDeterministic::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(SparseLinearLayerDeterministic::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &SparseLinearLayerDeterministic::num_cpus)
        .def_property_readonly("n_inputs",  &SparseLinearLayerDeterministic::n_inputs)
        .def_property_readonly("n_outputs", &SparseLinearLayerDeterministic::n_outputs)
        .def_property_readonly("nnz",       &SparseLinearLayerDeterministic::nnz)
        .def_property_readonly("block4",    &SparseLinearLayerDeterministic::block4,
             py::keep_alive<0, 1>(),
             "Purely observational view onto this layer's block4 storage --"
             " layer.block4.tiles / layer.block4.synapses.")
        .def_property_readonly("last_input",
            [](const SparseLinearLayerDeterministic& self) -> py::object {
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



    // ── SparseLinearLayerResyncDeterministic ─────────────────────────────────────
    // DeferredScaleWrite=true + StochasticRounding=false together.

    py::class_<SparseLinearLayerResyncDeterministic>(m, "SparseLinearLayerResyncDeterministic")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("forward_dense",        &SparseLinearLayerResyncDeterministic::forward_dense,
             py::arg("x"))
        .def("backward_dense",       &SparseLinearLayerResyncDeterministic::backward_dense,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
             "damp_by_importance=True (default): weight update divided by\n"
             "(sqrt(importance)+eps), where importance is a decayed EMA of\n"
             "g^2 (RMSprop-style) -- a per-synapse adaptive-learning-rate\n"
             "effect. False: raw update, no damping -- importance is still\n"
             "tracked identically either way, only its use to shape the\n"
             "weight step is toggled. For A/B-testing whether the damping\n"
             "itself helps optimization, not for production use. beta2/eps\n"
             "only affect the True case.")
        .def("forward_sparse",       &SparseLinearLayerResyncDeterministic::forward_sparse,
             py::arg("ptrs"), py::arg("indices"), py::arg("values"),
             py::arg("batch"))
        .def("backward_sparse",      &SparseLinearLayerResyncDeterministic::backward_sparse,
             py::arg("x"),
             py::arg("dy_ptrs"), py::arg("dy_indices"), py::arg("dy_values"),
             py::arg("batch"), py::arg("learning_rate") = 0.01f, py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("build_probes",         &SparseLinearLayerResyncDeterministic::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayerResyncDeterministic::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"),
             py::arg("max_prune_per_step") = 8, py::arg("importance_eps") = 1e-3f)
        .def("synap_step",           &SparseLinearLayerResyncDeterministic::synap_step,
             py::arg("importance_cutoff"), py::arg("max_row_weights"), py::arg("max_prune_per_step") = 8,
             py::arg("importance_eps") = 1e-3f,
             "Stateful convenience wrapper around synap_row_step -- advances an\n"
             "internal row cursor automatically, so a caller doing repeated\n"
             "one-step-per-call synaptogenesis sweeps doesn't need to track the\n"
             "row index itself. Use synap_row_step directly for explicit control.")
        .def("equalizer_step",       &SparseLinearLayerResyncDeterministic::equalizer_step,
             "One row of staggered memory redistribution -- call once per\n"
             "synaptogenesis cycle. REDISTRIBUTES the existing pool; does NOT\n"
             "add new memory. Use equalize_to_capacity() first to ensure the\n"
             "pool is large enough for max_row_weights connections per row.\n"
             "Also steps block4's own tile-storage row-headroom equalizer\n"
             "(own independent cursor) -- lets bytes a row's tiles freed by\n"
             "compressing become usable by growth elsewhere in the matrix.")
        .def("equalize_to_capacity", &SparseLinearLayerResyncDeterministic::equalize_to_capacity,
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
        .def("compact",              &SparseLinearLayerResyncDeterministic::compact,
             "Repack in place: every row occupies exactly its active bytes/elements,\n"
             "zero inter-row blank space. Call before saving/measuring a freshly\n"
             "converted model. Zeroes growth headroom.")
        .def("expand_headroom",      &SparseLinearLayerResyncDeterministic::expand_headroom,
             py::arg("blank_fraction") = 0.2f,
             "Restore per-row growth headroom (proportional to current nnz).\n"
             "WARNING: use expand_headroom_to(max_row_weights) after a prune\n"
             "cycle -- plain expand_headroom allocates headroom proportional\n"
             "to CURRENT nnz, leaving no room to grow back to max_row_weights.")
        .def("expand_headroom_to",   &SparseLinearLayerResyncDeterministic::expand_headroom_to,
             py::arg("min_nnz_per_row"), py::arg("blank_fraction") = 0.2f,
             "Like expand_headroom() but guarantees each row has headroom for\n"
             "at least min_nnz_per_row connections. Call with max_row_weights\n"
             "before synaptogenesis after a prune cycle.")
        .def("get_importance_scale", &SparseLinearLayerResyncDeterministic::get_importance_scale,
             py::arg("row"),
             "Per-ROW scale applied to that row's stored importance to get true\n"
             "units. Default 1.0 for any row not yet touched, exact backward\n"
             "compat. Use rescale_importance_row()/rescale_importance() to\n"
             "change it, never assign directly.")
        .def("get_value_scale",      &SparseLinearLayerResyncDeterministic::get_value_scale,
             py::arg("row"),
             "Same as get_importance_scale() but for stored weight values.")
        .def("set_value_scale_raw",
             [](SparseLinearLayerResyncDeterministic& self, int row, float scale) {
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
             [](SparseLinearLayerResyncDeterministic& self, int row, float scale) {
                 self.weights.set_importance_scale_raw(
                     static_cast<std::size_t>(row), scale);
             },
             py::arg("row"), py::arg("scale"),
             "Same as set_value_scale_raw() but for importance.")
        .def("get_output_scale",     &SparseLinearLayerResyncDeterministic::get_output_scale,
             py::arg("col"),
             "Per-COLUMN (per-output) counterpart to get_value_scale() -- true_w =\n"
             "stored_w * value_scale[row] * output_scale[col]. Default 1.0 for any\n"
             "column not yet touched, exact backward compat with every caller that\n"
             "never sets it.")
        .def("set_output_scale_raw",
             [](SparseLinearLayerResyncDeterministic& self, int col, float scale) {
                 self.weights.set_output_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Set output_scale[col] directly WITHOUT re-encoding stored weights --\n"
             "same convention as set_value_scale_raw(), but per-output instead of\n"
             "per-input. Calling this at least once makes output_scale\n"
             "gradient-trainable in backward_dense(), like value_scale.")
        .def("get_value_scale_importance",  &SparseLinearLayerResyncDeterministic::get_value_scale_importance,
             py::arg("row"),
             "Per-row importance backing value_scale's own gradient step, same\n"
             "damping role as a synapse's importance value. Default 0.")
        .def("get_output_scale_importance", &SparseLinearLayerResyncDeterministic::get_output_scale_importance,
             py::arg("col"),
             "Per-column counterpart for output_scale. Default 0.")
        .def("get_output_importance_scale", &SparseLinearLayerResyncDeterministic::get_output_importance_scale,
             py::arg("col"),
             "Per-COLUMN counterpart to get_importance_scale() -- true_imp =\n"
             "stored_imp * importance_scale[row] * output_importance_scale[col].\n"
             "Default 1.0, same convention as get_output_scale().")
        .def("set_output_importance_scale_raw",
             [](SparseLinearLayerResyncDeterministic& self, int col, float scale) {
                 self.weights.set_output_importance_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Same as set_importance_scale_raw() but per-output instead of\n"
             "per-input -- see set_output_scale_raw() for the analogous pattern.")
        .def("rescale_importance_row", &SparseLinearLayerResyncDeterministic::rescale_importance_row,
             py::arg("row"), py::arg("new_scale"),
             "Change ONE row's importance scale mid-training without corrupting\n"
             "that row's existing stored data -- re-reads at the OLD per-row\n"
             "scale, re-encodes at the NEW one.")
        .def("rescale_value_row",    &SparseLinearLayerResyncDeterministic::rescale_value_row,
             py::arg("row"), py::arg("new_scale"),
             "Same as rescale_importance_row() but for stored weight values.")
        .def("rescale_importance",   &SparseLinearLayerResyncDeterministic::rescale_importance,
             py::arg("new_scale"),
             "Bulk convenience: set EVERY row's importance scale to the same\n"
             "value. Backward-compatible interface with the original\n"
             "per-layer-scalar design.")
        .def("rescale_value",        &SparseLinearLayerResyncDeterministic::rescale_value,
             py::arg("new_scale"),
             "Same as rescale_importance() but for stored weight values.")
        .def_property_readonly("value_l1",           &SparseLinearLayerResyncDeterministic::get_value_l1)
        .def_property_readonly("value_l2_sq",        &SparseLinearLayerResyncDeterministic::get_value_l2_sq)
        .def_property_readonly("value_max_abs",      &SparseLinearLayerResyncDeterministic::get_value_max_abs)
        .def_property_readonly("importance_l1",      &SparseLinearLayerResyncDeterministic::get_importance_l1)
        .def_property_readonly("importance_l2_sq",   &SparseLinearLayerResyncDeterministic::get_importance_l2_sq)
        .def_property_readonly("importance_max_abs", &SparseLinearLayerResyncDeterministic::get_importance_max_abs)
        .def("hoyer_value",          &SparseLinearLayerResyncDeterministic::hoyer_value,
             "Hoyer's sparsity measure on the STORED weight distribution, in\n"
             "[0,1] -- 0 means values spread evenly across FP4's representable\n"
             "range, 1 means concentrated (e.g. mostly zero, or mostly clustered\n"
             "at one magnitude). O(1), from running stats -- see value_max_abs\n"
             "for a cheap complementary saturation check this doesn't catch.")
        .def("hoyer_importance",     &SparseLinearLayerResyncDeterministic::hoyer_importance,
             "Same as hoyer_value() but for the STORED importance distribution.")
        .def("recompute_stats",      &SparseLinearLayerResyncDeterministic::recompute_stats,
             "Recompute value_l1/l2_sq/max_abs and importance_l1/l2_sq/max_abs\n"
             "from scratch, O(nnz). Gives an exact max_abs (the incrementally-\n"
             "maintained one is a monotonic upper bound, not a live exact max).")
        .def("zero_accum",           &SparseLinearLayerResyncDeterministic::zero_accum)
        .def_property_readonly("neuron_input_accum", &SparseLinearLayerResyncDeterministic::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &SparseLinearLayerResyncDeterministic::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &SparseLinearLayerResyncDeterministic::get_weights_vals)
        .def_property_readonly("importance",         &SparseLinearLayerResyncDeterministic::get_importance)
        .def_property_readonly("indices",            &SparseLinearLayerResyncDeterministic::get_indices)
        .def_property_readonly("ptrs",               &SparseLinearLayerResyncDeterministic::get_ptrs)
        .def("load_weights",        &SparseLinearLayerResyncDeterministic::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"))
        .def("load_dense_codes",    &SparseLinearLayerResyncDeterministic::load_dense_codes,
             py::arg("weight_codes"), py::arg("importance_codes"))
        .def_property_readonly("out_degree", [](const SparseLinearLayerResyncDeterministic& self) {
            return py::array_t<SparseLinearLayerResyncDeterministic::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(SparseLinearLayerResyncDeterministic::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &SparseLinearLayerResyncDeterministic::num_cpus)
        .def_property_readonly("n_inputs",  &SparseLinearLayerResyncDeterministic::n_inputs)
        .def_property_readonly("n_outputs", &SparseLinearLayerResyncDeterministic::n_outputs)
        .def_property_readonly("nnz",       &SparseLinearLayerResyncDeterministic::nnz)
        .def_property_readonly("block4",    &SparseLinearLayerResyncDeterministic::block4,
             py::keep_alive<0, 1>(),
             "Purely observational view onto this layer's block4 storage --"
             " layer.block4.tiles / layer.block4.synapses.")
        .def_property_readonly("last_input",
            [](const SparseLinearLayerResyncDeterministic& self) -> py::object {
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


    // ── SparseLinearLayerNoScaleDeterministic ────────────────────────────────────
    // NoScalePolicy + StochasticRounding=false together.

    py::class_<SparseLinearLayerNoScaleDeterministic>(m, "SparseLinearLayerNoScaleDeterministic")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("forward_dense",        &SparseLinearLayerNoScaleDeterministic::forward_dense,
             py::arg("x"))
        .def("backward_dense",       &SparseLinearLayerNoScaleDeterministic::backward_dense,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
             "damp_by_importance=True (default): weight update divided by\n"
             "(sqrt(importance)+eps), where importance is a decayed EMA of\n"
             "g^2 (RMSprop-style) -- a per-synapse adaptive-learning-rate\n"
             "effect. False: raw update, no damping -- importance is still\n"
             "tracked identically either way, only its use to shape the\n"
             "weight step is toggled. For A/B-testing whether the damping\n"
             "itself helps optimization, not for production use. beta2/eps\n"
             "only affect the True case.")
        .def("forward_sparse",       &SparseLinearLayerNoScaleDeterministic::forward_sparse,
             py::arg("ptrs"), py::arg("indices"), py::arg("values"),
             py::arg("batch"))
        .def("backward_sparse",      &SparseLinearLayerNoScaleDeterministic::backward_sparse,
             py::arg("x"),
             py::arg("dy_ptrs"), py::arg("dy_indices"), py::arg("dy_values"),
             py::arg("batch"), py::arg("learning_rate") = 0.01f, py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("build_probes",         &SparseLinearLayerNoScaleDeterministic::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayerNoScaleDeterministic::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"),
             py::arg("max_prune_per_step") = 8, py::arg("importance_eps") = 1e-3f)
        .def("synap_step",           &SparseLinearLayerNoScaleDeterministic::synap_step,
             py::arg("importance_cutoff"), py::arg("max_row_weights"), py::arg("max_prune_per_step") = 8,
             py::arg("importance_eps") = 1e-3f,
             "Stateful convenience wrapper around synap_row_step -- advances an\n"
             "internal row cursor automatically, so a caller doing repeated\n"
             "one-step-per-call synaptogenesis sweeps doesn't need to track the\n"
             "row index itself. Use synap_row_step directly for explicit control.")
        .def("equalizer_step",       &SparseLinearLayerNoScaleDeterministic::equalizer_step,
             "One row of staggered memory redistribution -- call once per\n"
             "synaptogenesis cycle. REDISTRIBUTES the existing pool; does NOT\n"
             "add new memory. Use equalize_to_capacity() first to ensure the\n"
             "pool is large enough for max_row_weights connections per row.\n"
             "Also steps block4's own tile-storage row-headroom equalizer\n"
             "(own independent cursor) -- lets bytes a row's tiles freed by\n"
             "compressing become usable by growth elsewhere in the matrix.")
        .def("equalize_to_capacity", &SparseLinearLayerNoScaleDeterministic::equalize_to_capacity,
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
        .def("compact",              &SparseLinearLayerNoScaleDeterministic::compact,
             "Repack in place: every row occupies exactly its active bytes/elements,\n"
             "zero inter-row blank space. Call before saving/measuring a freshly\n"
             "converted model. Zeroes growth headroom.")
        .def("expand_headroom",      &SparseLinearLayerNoScaleDeterministic::expand_headroom,
             py::arg("blank_fraction") = 0.2f,
             "Restore per-row growth headroom (proportional to current nnz).\n"
             "WARNING: use expand_headroom_to(max_row_weights) after a prune\n"
             "cycle -- plain expand_headroom allocates headroom proportional\n"
             "to CURRENT nnz, leaving no room to grow back to max_row_weights.")
        .def("expand_headroom_to",   &SparseLinearLayerNoScaleDeterministic::expand_headroom_to,
             py::arg("min_nnz_per_row"), py::arg("blank_fraction") = 0.2f,
             "Like expand_headroom() but guarantees each row has headroom for\n"
             "at least min_nnz_per_row connections. Call with max_row_weights\n"
             "before synaptogenesis after a prune cycle.")
        .def("get_importance_scale", &SparseLinearLayerNoScaleDeterministic::get_importance_scale,
             py::arg("row"),
             "Per-ROW scale applied to that row's stored importance to get true\n"
             "units. Default 1.0 for any row not yet touched, exact backward\n"
             "compat. Use rescale_importance_row()/rescale_importance() to\n"
             "change it, never assign directly.")
        .def("get_value_scale",      &SparseLinearLayerNoScaleDeterministic::get_value_scale,
             py::arg("row"),
             "Same as get_importance_scale() but for stored weight values.")
        .def("set_value_scale_raw",
             [](SparseLinearLayerNoScaleDeterministic& self, int row, float scale) {
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
             [](SparseLinearLayerNoScaleDeterministic& self, int row, float scale) {
                 self.weights.set_importance_scale_raw(
                     static_cast<std::size_t>(row), scale);
             },
             py::arg("row"), py::arg("scale"),
             "Same as set_value_scale_raw() but for importance.")
        .def("get_output_scale",     &SparseLinearLayerNoScaleDeterministic::get_output_scale,
             py::arg("col"),
             "Per-COLUMN (per-output) counterpart to get_value_scale() -- true_w =\n"
             "stored_w * value_scale[row] * output_scale[col]. Default 1.0 for any\n"
             "column not yet touched, exact backward compat with every caller that\n"
             "never sets it.")
        .def("set_output_scale_raw",
             [](SparseLinearLayerNoScaleDeterministic& self, int col, float scale) {
                 self.weights.set_output_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Set output_scale[col] directly WITHOUT re-encoding stored weights --\n"
             "same convention as set_value_scale_raw(), but per-output instead of\n"
             "per-input. Calling this at least once makes output_scale\n"
             "gradient-trainable in backward_dense(), like value_scale.")
        .def("get_value_scale_importance",  &SparseLinearLayerNoScaleDeterministic::get_value_scale_importance,
             py::arg("row"),
             "Per-row importance backing value_scale's own gradient step, same\n"
             "damping role as a synapse's importance value. Default 0.")
        .def("get_output_scale_importance", &SparseLinearLayerNoScaleDeterministic::get_output_scale_importance,
             py::arg("col"),
             "Per-column counterpart for output_scale. Default 0.")
        .def("get_output_importance_scale", &SparseLinearLayerNoScaleDeterministic::get_output_importance_scale,
             py::arg("col"),
             "Per-COLUMN counterpart to get_importance_scale() -- true_imp =\n"
             "stored_imp * importance_scale[row] * output_importance_scale[col].\n"
             "Default 1.0, same convention as get_output_scale().")
        .def("set_output_importance_scale_raw",
             [](SparseLinearLayerNoScaleDeterministic& self, int col, float scale) {
                 self.weights.set_output_importance_scale_raw(
                     static_cast<std::size_t>(col), scale);
             },
             py::arg("col"), py::arg("scale"),
             "Same as set_importance_scale_raw() but per-output instead of\n"
             "per-input -- see set_output_scale_raw() for the analogous pattern.")
        .def("rescale_importance_row", &SparseLinearLayerNoScaleDeterministic::rescale_importance_row,
             py::arg("row"), py::arg("new_scale"),
             "Change ONE row's importance scale mid-training without corrupting\n"
             "that row's existing stored data -- re-reads at the OLD per-row\n"
             "scale, re-encodes at the NEW one.")
        .def("rescale_value_row",    &SparseLinearLayerNoScaleDeterministic::rescale_value_row,
             py::arg("row"), py::arg("new_scale"),
             "Same as rescale_importance_row() but for stored weight values.")
        .def("rescale_importance",   &SparseLinearLayerNoScaleDeterministic::rescale_importance,
             py::arg("new_scale"),
             "Bulk convenience: set EVERY row's importance scale to the same\n"
             "value. Backward-compatible interface with the original\n"
             "per-layer-scalar design.")
        .def("rescale_value",        &SparseLinearLayerNoScaleDeterministic::rescale_value,
             py::arg("new_scale"),
             "Same as rescale_importance() but for stored weight values.")
        .def_property_readonly("value_l1",           &SparseLinearLayerNoScaleDeterministic::get_value_l1)
        .def_property_readonly("value_l2_sq",        &SparseLinearLayerNoScaleDeterministic::get_value_l2_sq)
        .def_property_readonly("value_max_abs",      &SparseLinearLayerNoScaleDeterministic::get_value_max_abs)
        .def_property_readonly("importance_l1",      &SparseLinearLayerNoScaleDeterministic::get_importance_l1)
        .def_property_readonly("importance_l2_sq",   &SparseLinearLayerNoScaleDeterministic::get_importance_l2_sq)
        .def_property_readonly("importance_max_abs", &SparseLinearLayerNoScaleDeterministic::get_importance_max_abs)
        .def("hoyer_value",          &SparseLinearLayerNoScaleDeterministic::hoyer_value,
             "Hoyer's sparsity measure on the STORED weight distribution, in\n"
             "[0,1] -- 0 means values spread evenly across FP4's representable\n"
             "range, 1 means concentrated (e.g. mostly zero, or mostly clustered\n"
             "at one magnitude). O(1), from running stats -- see value_max_abs\n"
             "for a cheap complementary saturation check this doesn't catch.")
        .def("hoyer_importance",     &SparseLinearLayerNoScaleDeterministic::hoyer_importance,
             "Same as hoyer_value() but for the STORED importance distribution.")
        .def("recompute_stats",      &SparseLinearLayerNoScaleDeterministic::recompute_stats,
             "Recompute value_l1/l2_sq/max_abs and importance_l1/l2_sq/max_abs\n"
             "from scratch, O(nnz). Gives an exact max_abs (the incrementally-\n"
             "maintained one is a monotonic upper bound, not a live exact max).")
        .def("zero_accum",           &SparseLinearLayerNoScaleDeterministic::zero_accum)
        .def_property_readonly("neuron_input_accum", &SparseLinearLayerNoScaleDeterministic::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &SparseLinearLayerNoScaleDeterministic::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &SparseLinearLayerNoScaleDeterministic::get_weights_vals)
        .def_property_readonly("importance",         &SparseLinearLayerNoScaleDeterministic::get_importance)
        .def_property_readonly("indices",            &SparseLinearLayerNoScaleDeterministic::get_indices)
        .def_property_readonly("ptrs",               &SparseLinearLayerNoScaleDeterministic::get_ptrs)
        .def("load_weights",        &SparseLinearLayerNoScaleDeterministic::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"))
        .def("load_dense_codes",    &SparseLinearLayerNoScaleDeterministic::load_dense_codes,
             py::arg("weight_codes"), py::arg("importance_codes"))
        .def_property_readonly("out_degree", [](const SparseLinearLayerNoScaleDeterministic& self) {
            return py::array_t<SparseLinearLayerNoScaleDeterministic::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(SparseLinearLayerNoScaleDeterministic::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &SparseLinearLayerNoScaleDeterministic::num_cpus)
        .def_property_readonly("n_inputs",  &SparseLinearLayerNoScaleDeterministic::n_inputs)
        .def_property_readonly("n_outputs", &SparseLinearLayerNoScaleDeterministic::n_outputs)
        .def_property_readonly("nnz",       &SparseLinearLayerNoScaleDeterministic::nnz)
        .def_property_readonly("block4",    &SparseLinearLayerNoScaleDeterministic::block4,
             py::keep_alive<0, 1>(),
             "Purely observational view onto this layer's block4 storage --"
             " layer.block4.tiles / layer.block4.synapses.")
        .def_property_readonly("last_input",
            [](const SparseLinearLayerNoScaleDeterministic& self) -> py::object {
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
             py::arg("x"))
        .def("backward",             &DISLDOLayerV::backward,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
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
        .def_property_readonly("nnz",       &DISLDOLayerV::nnz)
        .def_property_readonly("block4",    &DISLDOLayerV::block4, py::keep_alive<0, 1>());

    // ── SparseLinearLayer8 ───────────────────────────────────────────────────
    py::class_<SparseLinearLayer8>(m, "SparseLinearLayer8")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("forward",              &SparseLinearLayer8::forward,
             py::arg("x"))
        .def("backward",             &SparseLinearLayer8::backward,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("build_probes",         &SparseLinearLayer8::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayer8::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"))
        .def("zero_accum",           &SparseLinearLayer8::zero_accum)
        .def_property_readonly("neuron_input_accum", &SparseLinearLayer8::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &SparseLinearLayer8::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &SparseLinearLayer8::get_weights_vals)
        .def_property_readonly("importance",         &SparseLinearLayer8::get_importance)
        .def_property_readonly("indices",            &SparseLinearLayer8::get_indices)
        .def_property_readonly("ptrs",               &SparseLinearLayer8::get_ptrs)
        .def("load_weights",        &SparseLinearLayer8::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"), py::arg("importance"))
        .def("load_dense_codes",    &SparseLinearLayer8::load_dense_codes,
             py::arg("weight_codes"), py::arg("importance_codes"),
             "Bulk-load already-quantized dense FP8 (E4M3) codes directly into\n"
             "block4, bypassing scattered CSR entirely. See block4_load_dense's\n"
             "docstring (delta_csr_memory.hpp) -- loading only, no quantization\n"
             "performed here; produce codes via fp8_quantize_array() or any other\n"
             "scheme.")
        .def("get_value_scale",      &SparseLinearLayer8::get_value_scale,
             py::arg("row"),
             "Per-ROW scale -- true_w = stored_w * value_scale[row] * output_scale[col].\n"
             "Default 1.0 for any row not yet touched. Same VALUES_TYPE-agnostic\n"
             "mechanism SparseLinearLayer (FP4) uses.")
        .def("set_value_scale_raw",  &SparseLinearLayer8::set_value_scale_raw,
             py::arg("row"), py::arg("scale"),
             "Set value_scale[row] directly WITHOUT re-encoding stored weights --\n"
             "same convention as SparseLinearLayer::set_value_scale_raw.")
        .def("get_output_scale",     &SparseLinearLayer8::get_output_scale,
             py::arg("col"),
             "Per-COLUMN (rank-1) counterpart to get_value_scale(). Default 1.0 for\n"
             "any column not yet touched.")
        .def("set_output_scale_raw", &SparseLinearLayer8::set_output_scale_raw,
             py::arg("col"), py::arg("scale"),
             "Set output_scale[col] directly. Calling this at least once makes\n"
             "output_scale gradient-trainable in backward(), like value_scale --\n"
             "this is what makes the row+col scale genuinely rank-1, matching the\n"
             "scheme validated in sili_peridot's toy-model quantization sweep.")
        .def_property_readonly("out_degree", [](const SparseLinearLayer8& self) {
            return py::array_t<SparseLinearLayer8::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(SparseLinearLayer8::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &SparseLinearLayer8::num_cpus)
        .def_property_readonly("n_inputs",  &SparseLinearLayer8::n_inputs)
        .def_property_readonly("n_outputs", &SparseLinearLayer8::n_outputs)
        .def_property_readonly("nnz",       &SparseLinearLayer8::nnz)
        .def_property_readonly("block4",    &SparseLinearLayer8::block4, py::keep_alive<0, 1>());

    // SparseLinearLayer8Resync: exact SparseLinearLayer8 API, but the real
    // DeferredScaleWrite fix (value_scale/output_scale stay consistent with
    // stored codes -- see SparseLinearLayer8Impl/ScalePolicy docstrings).
    py::class_<SparseLinearLayer8Resync>(m, "SparseLinearLayer8Resync")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("forward",              &SparseLinearLayer8Resync::forward,
             py::arg("x"))
        .def("backward",             &SparseLinearLayer8Resync::backward,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("build_probes",         &SparseLinearLayer8Resync::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayer8Resync::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"))
        .def("zero_accum",           &SparseLinearLayer8Resync::zero_accum)
        .def_property_readonly("neuron_input_accum", &SparseLinearLayer8Resync::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &SparseLinearLayer8Resync::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &SparseLinearLayer8Resync::get_weights_vals)
        .def_property_readonly("importance",         &SparseLinearLayer8Resync::get_importance)
        .def_property_readonly("indices",            &SparseLinearLayer8Resync::get_indices)
        .def_property_readonly("ptrs",               &SparseLinearLayer8Resync::get_ptrs)
        .def("load_weights",        &SparseLinearLayer8Resync::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"), py::arg("importance"))
        .def("load_dense_codes",    &SparseLinearLayer8Resync::load_dense_codes,
             py::arg("weight_codes"), py::arg("importance_codes"))
        .def("get_value_scale",      &SparseLinearLayer8Resync::get_value_scale,
             py::arg("row"),
             "Per-ROW scale -- true_w = stored_w * value_scale[row] * output_scale[col].\n"
             "Default 1.0 for any row not yet touched. Same VALUES_TYPE-agnostic\n"
             "mechanism SparseLinearLayer (FP4) uses.")
        .def("set_value_scale_raw",  &SparseLinearLayer8Resync::set_value_scale_raw,
             py::arg("row"), py::arg("scale"),
             "Set value_scale[row] directly WITHOUT re-encoding stored weights --\n"
             "same convention as SparseLinearLayer::set_value_scale_raw.")
        .def("get_output_scale",     &SparseLinearLayer8Resync::get_output_scale,
             py::arg("col"),
             "Per-COLUMN (rank-1) counterpart to get_value_scale(). Default 1.0 for\n"
             "any column not yet touched.")
        .def("set_output_scale_raw", &SparseLinearLayer8Resync::set_output_scale_raw,
             py::arg("col"), py::arg("scale"),
             "Set output_scale[col] directly. Calling this at least once makes\n"
             "output_scale gradient-trainable in backward(), like value_scale --\n"
             "this is what makes the row+col scale genuinely rank-1, matching the\n"
             "scheme validated in sili_peridot's toy-model quantization sweep.")
        .def_property_readonly("out_degree", [](const SparseLinearLayer8Resync& self) {
            return py::array_t<SparseLinearLayer8Resync::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(SparseLinearLayer8Resync::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &SparseLinearLayer8Resync::num_cpus)
        .def_property_readonly("n_inputs",  &SparseLinearLayer8Resync::n_inputs)
        .def_property_readonly("n_outputs", &SparseLinearLayer8Resync::n_outputs)
        .def_property_readonly("nnz",       &SparseLinearLayer8Resync::nnz)
        .def_property_readonly("block4",    &SparseLinearLayer8Resync::block4, py::keep_alive<0, 1>());

    // SparseLinearLayer8AdaMax: exact SparseLinearLayer8 API, but with the
    // AdaMax-style decayed-running-max scale update (also with the
    // DeferredScaleWrite fix -- AdaMax's growth-is-instant safety property
    // only matters if the stored code actually reflects it).
    py::class_<SparseLinearLayer8AdaMax>(m, "SparseLinearLayer8AdaMax")
        .def(py::init<int, int, int, int>(),
             py::arg("n_inputs"), py::arg("n_outputs"), py::arg("max_weights"),
             py::arg("num_cpus") = 4)
        .def("forward",              &SparseLinearLayer8AdaMax::forward,
             py::arg("x"))
        .def("backward",             &SparseLinearLayer8AdaMax::backward,
             py::arg("dy"), py::arg("learning_rate"), py::arg("lr_per_row_nnz") = false,
             py::arg("damp_by_importance") = true, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f)
        .def("build_probes",         &SparseLinearLayer8AdaMax::build_probes,
             py::arg("k"), py::arg("per_row") = false)
        .def("synap_row_step",       &SparseLinearLayer8AdaMax::synap_row_step,
             py::arg("current_row"), py::arg("importance_cutoff"), py::arg("max_row_weights"))
        .def("zero_accum",           &SparseLinearLayer8AdaMax::zero_accum)
        .def_property_readonly("neuron_input_accum", &SparseLinearLayer8AdaMax::get_neuron_input_accum)
        .def_property_readonly("neuron_grad_accum",  &SparseLinearLayer8AdaMax::get_neuron_grad_accum)
        .def_property_readonly("weights_vals",       &SparseLinearLayer8AdaMax::get_weights_vals)
        .def_property_readonly("importance",         &SparseLinearLayer8AdaMax::get_importance)
        .def_property_readonly("indices",            &SparseLinearLayer8AdaMax::get_indices)
        .def_property_readonly("ptrs",               &SparseLinearLayer8AdaMax::get_ptrs)
        .def("load_weights",        &SparseLinearLayer8AdaMax::load_weights,
             py::arg("ptrs"), py::arg("indices"), py::arg("weights"), py::arg("importance"))
        .def("load_dense_codes",    &SparseLinearLayer8AdaMax::load_dense_codes,
             py::arg("weight_codes"), py::arg("importance_codes"))
        .def("get_value_scale",      &SparseLinearLayer8AdaMax::get_value_scale,
             py::arg("row"),
             "Per-ROW scale -- true_w = stored_w * value_scale[row] * output_scale[col].\n"
             "Default 1.0 for any row not yet touched. Same VALUES_TYPE-agnostic\n"
             "mechanism SparseLinearLayer (FP4) uses.")
        .def("set_value_scale_raw",  &SparseLinearLayer8AdaMax::set_value_scale_raw,
             py::arg("row"), py::arg("scale"),
             "Set value_scale[row] directly WITHOUT re-encoding stored weights --\n"
             "same convention as SparseLinearLayer::set_value_scale_raw.")
        .def("get_output_scale",     &SparseLinearLayer8AdaMax::get_output_scale,
             py::arg("col"),
             "Per-COLUMN (rank-1) counterpart to get_value_scale(). Default 1.0 for\n"
             "any column not yet touched.")
        .def("set_output_scale_raw", &SparseLinearLayer8AdaMax::set_output_scale_raw,
             py::arg("col"), py::arg("scale"),
             "Set output_scale[col] directly. Calling this at least once makes\n"
             "output_scale gradient-trainable in backward(), like value_scale --\n"
             "this is what makes the row+col scale genuinely rank-1, matching the\n"
             "scheme validated in sili_peridot's toy-model quantization sweep.")
        .def_property_readonly("out_degree", [](const SparseLinearLayer8AdaMax& self) {
            return py::array_t<SparseLinearLayer8AdaMax::S>(
                {(py::ssize_t)self.weights.out_degree.size()},
                {sizeof(SparseLinearLayer8AdaMax::S)},
                self.weights.out_degree.data(),
                py::cast(&self));
        })
        .def_readonly ("num_cpus",  &SparseLinearLayer8AdaMax::num_cpus)
        .def_property_readonly("n_inputs",  &SparseLinearLayer8AdaMax::n_inputs)
        .def_property_readonly("n_outputs", &SparseLinearLayer8AdaMax::n_outputs)
        .def_property_readonly("nnz",       &SparseLinearLayer8AdaMax::nnz)
        .def_property_readonly("block4",    &SparseLinearLayer8AdaMax::block4, py::keep_alive<0, 1>());

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

    // ── Bulk quantize-array utilities ───────────────────────────────────────
    //
    // Standalone, layer-independent elementwise float32->code conversion --
    // deliberately kept SEPARATE from block4_load_dense (delta_csr_memory.hpp)
    // and its "load_dense_codes" pybind bindings below: loading (placing
    // already-decided codes into storage) and quantization (deciding what
    // code represents a float) are different concerns. This is the simple
    // deterministic round-to-nearest scheme (reuses the existing scalar
    // fp4_quantize/fp8_quantize codecs, unchanged); a caller wanting a
    // smarter scheme (rank-1 scale fit, residual decomposition, etc.) can
    // produce codes some other way and hand them directly to
    // load_dense_codes instead of calling these.
    m.def("fp4_quantize_array",
        [](py::array_t<float> vals) -> py::array_t<uint8_t> {
            auto buf = vals.request();
            py::array_t<uint8_t> out(buf.size);
            const float* src = (const float*)buf.ptr;
            uint8_t* dst = (uint8_t*)out.request().ptr;
            for (py::ssize_t i = 0; i < buf.size; ++i) dst[i] = fp4_quantize(src[i]);
            return out;
        },
        py::arg("vals"),
        "Deterministic FP4 (E2M1) quantize, elementwise -- same scalar\n"
        "fp4_quantize() semantics every insertion path in this codebase\n"
        "already uses, just applied over a whole array at once. Output\n"
        "codes (0-15) are the RAW magnitude+sign code, not scaled by any\n"
        "row/col value_scale -- pre-divide by scale yourself before calling\n"
        "this if you want one, same convention as load_weights().");

    m.def("fp8_quantize_array",
        [](py::array_t<float> vals) -> py::array_t<uint8_t> {
            auto buf = vals.request();
            py::array_t<uint8_t> out(buf.size);
            const float* src = (const float*)buf.ptr;
            uint8_t* dst = (uint8_t*)out.request().ptr;
            for (py::ssize_t i = 0; i < buf.size; ++i) dst[i] = fp8_quantize(src[i]);
            return out;
        },
        py::arg("vals"),
        "Deterministic FP8 (E4M3) quantize, elementwise -- FP8 counterpart\n"
        "to fp4_quantize_array(), same scale convention (pre-divide yourself).");

    m.def("seed_fp4_stochastic_rng", &fp4_seed_stochastic_rng, py::arg("seed"),
        "Reseed the CALLING thread's FP4 stochastic-rounding RNG (see "
        "fp4quant.hpp's fp4_quantize_stochastic -- used by disldo_forward's "
        "importance update and disldo_backward's weight+importance update, "
        "NOT by construction/loading/compact, which stay deterministic). "
        "For single-threaded test reproducibility, same precedent as pinning "
        "np.random.seed(0) for EnergyDynamics' own unseeded exploration "
        "noise -- does not control a real (OpenMP-parallel) training run's "
        "outcome, only this one thread's RNG state.");

    // ── csr_union ─────────────────────────────────────────────────────────────
    // OpenMP-parallel replacement for sili.sparse_rnn.csr_union's Python loop
    // (see csr.hpp's csr_union<>) -- construction/loading-time CSR merge only.
    m.def("csr_union",
        [](py::array_t<int> ptrs_a, py::array_t<int> idx_a, py::array_t<float> vals_a,
           py::array_t<int> ptrs_b, py::array_t<int> idx_b, py::array_t<float> vals_b,
           int n_rows, std::string prefer, int num_cpus) -> py::tuple {
            auto to_vec_i = [](py::array_t<int>& a) {
                auto buf = a.request();
                return std::vector<int>((int*)buf.ptr, (int*)buf.ptr + buf.size);
            };
            auto to_vec_f = [](py::array_t<float>& a) {
                auto buf = a.request();
                return std::vector<float>((float*)buf.ptr, (float*)buf.ptr + buf.size);
            };
            int prefer_code = (prefer == "a") ? 0 : (prefer == "b") ? 1
                            : (prefer == "sum") ? 2 : throw std::invalid_argument(
                                "prefer must be 'a', 'b', or 'sum'");

            std::vector<int>   out_ptrs, out_idx;
            std::vector<float> out_vals;
            csr_union<int, float>(
                to_vec_i(ptrs_a), to_vec_i(idx_a), to_vec_f(vals_a),
                to_vec_i(ptrs_b), to_vec_i(idx_b), to_vec_f(vals_b),
                n_rows, prefer_code, num_cpus,
                out_ptrs, out_idx, out_vals);

            py::array_t<int>   ret_ptrs ({(py::ssize_t)out_ptrs.size()});
            py::array_t<int>   ret_idx  ({(py::ssize_t)out_idx.size()});
            py::array_t<float> ret_vals ({(py::ssize_t)out_vals.size()});
            std::copy(out_ptrs.begin(), out_ptrs.end(), (int*)  ret_ptrs.request().ptr);
            std::copy(out_idx.begin(),  out_idx.end(),  (int*)  ret_idx .request().ptr);
            std::copy(out_vals.begin(), out_vals.end(), (float*)ret_vals.request().ptr);
            return py::make_tuple(ret_ptrs, ret_idx, ret_vals);
        },
        py::arg("ptrs_a"), py::arg("idx_a"), py::arg("vals_a"),
        py::arg("ptrs_b"), py::arg("idx_b"), py::arg("vals_b"),
        py::arg("n_rows"), py::arg("prefer") = "a", py::arg("num_cpus") = 4,
        "Merge two same-shape CSRs into the union of their nonzero positions.\n"
        "prefer: 'a' keeps A's value on overlap, 'b' keeps B's, 'sum' adds them.\n"
        "Construction/loading time only -- not used in forward/backward."
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

    // NOTE: a standalone m.def("make_weights", ...) binding used to live here,
    // calling make_weights<int,float> with a stale 7-arg signature (rows,
    // cols, ptrs, indices, values, grads, importance -- a separate
    // backprop-adjacent grads array that no longer exists as a concept)
    // against the current 5-arg definition. It never compiled against the
    // current headers (invalid use of incomplete type in pybind11's lambda
    // signature deduction, since the 7-arg lambda's call to the 5-arg
    // template was ill-formed). Confirmed unused from Python (no caller
    // anywhere in the repo) and its purpose -- constructing a usable CSR
    // weight layer from a few plain vectors for testing -- is already
    // served by delta_csr_from_absolute, used throughout the C++ test suite
    // and cpu_backend.cpp itself. Removed rather than patched; see TODO.md's
    // former "Correctness, needs a real fix eventually" entry (now resolved).

    // ── Attention ops (ported from sparse_linear_ops.hpp) ────────────────────
    // All three take numpy [T, d] Q/K/V arrays (float32) and return a
    // numpy [T, d] output. The Python-facing names match exactly what
    // multimodal_sparse_rnn.py calls (sparse_attention, sparse_banded_attention).

    m.def("sparse_attention",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           std::size_t top_k, int num_cpus, bool causal) {
            auto qb=q.request(), kb=k.request(), vb=v.request();
            const std::size_t T = qb.shape[0], d = qb.shape[1];
            py::array_t<float> out({(py::ssize_t)T, (py::ssize_t)d});
            auto ob = out.request();
            std::fill((float*)ob.ptr, (float*)ob.ptr + T*d, 0.0f);
            sparse_attention_forward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (float*)ob.ptr, T, d, top_k, num_cpus, causal);
            return out;
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("top_k") = 0, py::arg("num_cpus") = 4, py::arg("causal") = false,
        "Global top-k sparse attention. Q/K/V are [T, d] float32 numpy arrays.\n"
        "top_k=0 -> use sqrt(T). causal=True masks any selected (query, key)\n"
        "pair where the key's sequence position is after the query's.\n"
        "Returns [T, d] output.");

    m.def("sparse_banded_attention",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           std::size_t half_bandwidth, std::size_t inner_k, int num_cpus, bool causal) {
            auto qb=q.request(), kb=k.request(), vb=v.request();
            const std::size_t T = qb.shape[0], K = kb.shape[0], d = qb.shape[1];
            py::array_t<float> out({(py::ssize_t)T, (py::ssize_t)d});
            auto ob = out.request();
            std::fill((float*)ob.ptr, (float*)ob.ptr + T*d, 0.0f);
            sparse_banded_attention_forward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (float*)ob.ptr, T, K, d, half_bandwidth, inner_k, num_cpus, causal);
            return out;
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("half_bandwidth"), py::arg("inner_k") = 0, py::arg("num_cpus") = 4,
        py::arg("causal") = false,
        "Banded sparse attention. Q/K/V are [T, d] float32 numpy arrays.\n"
        "inner_k=0 -> use all keys in the band (dense banded). causal=True\n"
        "requires T == K and clamps each query's band so it never selects a\n"
        "key past its own position. Returns [T, d] output.");

    m.def("banded_attention",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           std::size_t half_bandwidth, int num_cpus, bool causal) {
            auto qb=q.request(), kb=k.request(), vb=v.request();
            const std::size_t T = qb.shape[0], K = kb.shape[0], d = qb.shape[1];
            py::array_t<float> out({(py::ssize_t)T, (py::ssize_t)d});
            auto ob = out.request();
            std::fill((float*)ob.ptr, (float*)ob.ptr + T*d, 0.0f);
            banded_attention_forward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (float*)ob.ptr, T, K, d, half_bandwidth, num_cpus, causal);
            return out;
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("half_bandwidth"), py::arg("num_cpus") = 4, py::arg("causal") = false,
        "Dense banded attention. Q/K/V are [T, d] float32 numpy arrays.\n"
        "causal=True requires T == K and clamps each query's band so it\n"
        "never sees a key past its own position (autoregressive self-attn).\n"
        "Returns [T, d] output.");

    m.def("banded_attention_backward",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           py::array_t<float> dO,
           std::size_t half_bandwidth, int num_cpus, bool causal) {
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
                T, K, d, half_bandwidth, num_cpus, causal);
            return py::make_tuple(dQ, dK, dV);
        },
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"),
        py::arg("half_bandwidth"), py::arg("num_cpus") = 4, py::arg("causal") = false,
        "Backward pass for banded_attention. causal must match the forward\n"
        "call. Returns (dQ, dK, dV) each [T or K, d].");

    m.def("gaussian_attention",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           py::array_t<float> centers, py::array_t<float> sigmas,
           int num_cpus, bool causal) {
            auto qb=q.request(), kb=k.request(), vb=v.request();
            auto cb=centers.request(), sb=sigmas.request();
            const std::size_t T = qb.shape[0], K = kb.shape[0], d = qb.shape[1];
            py::array_t<float> out({(py::ssize_t)T, (py::ssize_t)d});
            auto ob = out.request();
            std::fill((float*)ob.ptr, (float*)ob.ptr + T*d, 0.0f);
            gaussian_attention_forward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (float*)ob.ptr, T, K, d,
                (const float*)cb.ptr, (const float*)sb.ptr, num_cpus, causal);
            return out;
        },
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("centers"), py::arg("sigmas"), py::arg("num_cpus") = 4, py::arg("causal") = false,
        "Full (every query x every key) attention with a learnable per-query\n"
        "Gaussian log-bias: score[q,j] = Q[q].K[j]*scale - (j-centers[q])^2/\n"
        "(2*sigmas[q]^2), then softmax as usual. Q/K/V are [T or K, d] float32\n"
        "numpy arrays; centers/sigmas are [T]. sigmas must be strictly positive\n"
        "-- callers should store an unconstrained log_sigma and exponentiate\n"
        "before calling in (see sili.tensor.exp), not pass raw trainable sigma\n"
        "directly. Returns [T, d] output.");

    m.def("gaussian_attention_backward",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           py::array_t<float> dO, py::array_t<float> centers, py::array_t<float> sigmas,
           int num_cpus, bool causal) {
            auto qb=q.request(), kb=k.request(), vb=v.request(), dob=dO.request();
            auto cb=centers.request(), sb=sigmas.request();
            const std::size_t T = qb.shape[0], K = kb.shape[0], d = qb.shape[1];
            py::array_t<float> dQ({(py::ssize_t)T, (py::ssize_t)d});
            py::array_t<float> dK({(py::ssize_t)K, (py::ssize_t)d});
            py::array_t<float> dV({(py::ssize_t)K, (py::ssize_t)d});
            py::array_t<float> dCenters({(py::ssize_t)T});
            py::array_t<float> dSigmas({(py::ssize_t)T});
            auto dqb=dQ.request(), dkb=dK.request(), dvb=dV.request();
            auto dcb=dCenters.request(), dsb=dSigmas.request();
            std::fill((float*)dqb.ptr, (float*)dqb.ptr + T*d, 0.0f);
            std::fill((float*)dkb.ptr, (float*)dkb.ptr + K*d, 0.0f);
            std::fill((float*)dvb.ptr, (float*)dvb.ptr + K*d, 0.0f);
            std::fill((float*)dcb.ptr, (float*)dcb.ptr + T, 0.0f);
            std::fill((float*)dsb.ptr, (float*)dsb.ptr + T, 0.0f);
            gaussian_attention_backward(
                (const float*)qb.ptr, (const float*)kb.ptr, (const float*)vb.ptr,
                (const float*)dob.ptr,
                (float*)dqb.ptr, (float*)dkb.ptr, (float*)dvb.ptr,
                (float*)dcb.ptr, (float*)dsb.ptr,
                T, K, d, (const float*)cb.ptr, (const float*)sb.ptr, num_cpus, causal);
            return py::make_tuple(dQ, dK, dV, dCenters, dSigmas);
        },
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"),
        py::arg("centers"), py::arg("sigmas"), py::arg("num_cpus") = 4, py::arg("causal") = false,
        "Backward pass for gaussian_attention. centers/sigmas must match the\n"
        "forward call. Returns (dQ, dK, dV, dCenters, dSigmas).");

    m.def("sparse_banded_attention_backward",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           py::array_t<float> dO,
           std::size_t half_bandwidth, std::size_t inner_k, int num_cpus, bool causal) {
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
                T, K, d, half_bandwidth, inner_k, num_cpus, causal);
            return py::make_tuple(dQ, dK, dV);
        },
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"),
        py::arg("half_bandwidth"), py::arg("inner_k") = 0, py::arg("num_cpus") = 4,
        py::arg("causal") = false,
        "Backward pass for sparse_banded_attention. causal must match the\n"
        "forward call. Returns (dQ, dK, dV).");

    m.def("sparse_attention_backward",
        [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
           py::array_t<float> dO,
           std::size_t top_k, int num_cpus, bool causal) {
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
                T, d, top_k, num_cpus, causal);
            return py::make_tuple(dQ, dK, dV);
        },
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"),
        py::arg("top_k") = 0, py::arg("num_cpus") = 4, py::arg("causal") = false,
        "Backward pass for sparse_attention. causal must match the forward\n"
        "call. Returns (dQ, dK, dV).");

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