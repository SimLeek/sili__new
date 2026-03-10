#include "csr.hpp"
#include "coo.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

//SISLDO: Sparse Input, Sparse Linear, Dense Output

// ── helpers ───────────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUE_TYPE>
inline bool connections_empty(const CSRSynapses<SIZE_TYPE, VALUE_TYPE>& c) {
    return !c.values[0] || c.values[0]->empty();
}

// Reserve all value/index vectors to max_weights if not already large enough.
// Called once at model init — after this, vectors never reallocate.
template <typename SIZE_TYPE, typename VALUE_TYPE>
void reserve_connections(
    CSRSynapses<SIZE_TYPE, VALUE_TYPE>& c,
    const SIZE_TYPE max_weights)
{
    if (!c.indices[0]) c.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>();
    if (!c.values[0])  c.values[0]  = std::make_shared<std::vector<VALUE_TYPE>>();
    if (!c.values[1])  c.values[1]  = std::make_shared<std::vector<VALUE_TYPE>>();
    if (!c.values[2])  c.values[2]  = std::make_shared<std::vector<VALUE_TYPE>>();

    if (c.indices[0]->capacity() < max_weights) {
        c.indices[0]->reserve(max_weights);
        c.values[0] ->reserve(max_weights);
        c.values[1] ->reserve(max_weights);
        c.values[2] ->reserve(max_weights);
    }
}

// ── outer_product ─────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUE_TYPE>
void outer_product(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& input_tensor,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& output_gradient_tensor,
    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>& gen)
{
    for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
        const SIZE_TYPE in_start  = (*input_tensor.ptrs[0])[batch];
        const SIZE_TYPE out_start = (*output_gradient_tensor.ptrs[0])[batch];
        const SIZE_TYPE in_len    = (*input_tensor.ptrs[0])[batch + 1]  - in_start;
        const SIZE_TYPE out_len   = (*output_gradient_tensor.ptrs[0])[batch + 1] - out_start;
        const SIZE_TYPE prod_start = in_start * out_start;

        #pragma omp parallel for schedule(static)
        for (SIZE_TYPE i = 0; i < in_len * out_len; ++i) {
            const SIZE_TYPE x         = i / out_len;
            const SIZE_TYPE y         = i % out_len;
            const SIZE_TYPE global_id = prod_start + x * out_len + y;

            (*gen.indices[0])[global_id] = (*input_tensor.indices[0])[in_start  + x];
            (*gen.indices[1])[global_id] = (*output_gradient_tensor.indices[0])[out_start + y];
            (*gen.values[0]) [global_id] = (*input_tensor.values[0])[in_start + x]
                                         * (*output_gradient_tensor.values[0])[out_start + y];
        }
    }
}

// ── generate_new_weights_coo ──────────────────────────────────────────────────
// Outer product of top-k inputs × top-k outputs → k² candidate connections.
// Caller tunes k knowing the cost is O(k²) — this is intentional.

template <typename SIZE_TYPE, typename VALUE_TYPE>
COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> generate_new_weights_coo(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& input_tensor,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& output_gradient_tensor,
    const int num_cpus = 4)
{
    SIZE_TYPE total_reserve = 0;
    for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
        const SIZE_TYPE in_len  = (*input_tensor.ptrs[0]) [batch + 1] - (*input_tensor.ptrs[0]) [batch];
        const SIZE_TYPE out_len = (*output_gradient_tensor.ptrs[0])[batch + 1] - (*output_gradient_tensor.ptrs[0])[batch];
        total_reserve += in_len * out_len;
    }

    if (total_reserve == 0)
        return COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>();

    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> gen_coo;
    gen_coo.ptrs       = total_reserve;
    gen_coo.indices[0] = std::make_shared<std::vector<SIZE_TYPE>> (total_reserve);
    gen_coo.indices[1] = std::make_shared<std::vector<SIZE_TYPE>> (total_reserve);
    gen_coo.values[0]  = std::make_shared<std::vector<VALUE_TYPE>>(total_reserve);
    gen_coo.cols       = output_gradient_tensor.cols;
    gen_coo.rows       = input_tensor.cols;

    outer_product(input_tensor, output_gradient_tensor, gen_coo);

    auto duplicates = merge_sort_coo(gen_coo.indices, gen_coo.values, gen_coo.nnz());
    gen_coo.ptrs -= duplicates;

    return gen_coo;
}

// ── forward ───────────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUE_TYPE>
void sisldo_forward(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& input_tensor,
    const SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    VALUE_TYPE* output,
    bool train,
    VALUE_TYPE solidify = 0.01,
    const int num_cpus = 4,
    VALUE_TYPE* original_contributions_output = nullptr)
{
    if (connections_empty(weights.connections)) return;

    const SIZE_TYPE out_cols    = weights.connections.cols;
    const SIZE_TYPE num_outputs = input_tensor.rows * out_cols;
    const SIZE_TYPE num_inputs  = input_tensor.cols;

    const auto& conn_ptrs    = *weights.connections.ptrs[0];
    const auto& conn_indices = *weights.connections.indices[0];
    const auto& conn_val0    = *weights.connections.values[0];
    auto&       conn_val2    = *weights.connections.values[2];

    // ── Phase 1 setup: one output buffer slice per thread ────────────────────
    // Threads write to disjoint slices — no synchronization needed during scatter.
    // Extra memory: num_cpus × num_outputs × sizeof(VALUE_TYPE).
    std::vector<VALUE_TYPE> all_outputs((size_t)num_cpus * num_outputs, VALUE_TYPE(0));
    std::vector<VALUE_TYPE> all_contributions(
        original_contributions_output ? (size_t)num_cpus * num_inputs : 0,
        VALUE_TYPE(0));

    std::vector<SIZE_TYPE> work_offsets;

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid      = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        VALUE_TYPE* thread_output = all_outputs.data() + (size_t)tid * num_outputs;
        VALUE_TYPE* thread_contributions = original_contributions_output
            ? all_contributions.data() + (size_t)tid * num_inputs
            : nullptr;

        // ── Phase 1: scatter — each thread owns its own output slice ─────────
        for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
            const SIZE_TYPE batch_start  = (*input_tensor.ptrs[0])[batch];
            const SIZE_TYPE batch_end    = (*input_tensor.ptrs[0])[batch + 1];
            const SIZE_TYPE batch_nnz    = batch_end - batch_start;
            const SIZE_TYPE batch_offset = batch * out_cols;

            #pragma omp single
            {
                work_offsets.resize(batch_nnz + 1);
                work_offsets[0] = 0;
                for (SIZE_TYPE i = 0; i < batch_nnz; ++i) {
                    const SIZE_TYPE in_idx = (*input_tensor.indices[0])[batch_start + i];
                    work_offsets[i + 1] = work_offsets[i]
                        + conn_ptrs[in_idx + 1] - conn_ptrs[in_idx];
                }
            }

            const SIZE_TYPE total_work = work_offsets[batch_nnz];
            const SIZE_TYPE chunk      = (total_work + nthreads - 1) / nthreads;
            const SIZE_TYPE w_start    = std::min((SIZE_TYPE)tid * chunk, total_work);
            const SIZE_TYPE w_end      = std::min(w_start + chunk, total_work);

            if (w_start < w_end) {
                SIZE_TYPE ip = static_cast<SIZE_TYPE>(
                    std::upper_bound(work_offsets.begin(), work_offsets.end(), w_start)
                    - work_offsets.begin()) - 1;

                for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                    while (ip + 1 < batch_nnz && work_offsets[ip + 1] <= w) ++ip;

                    const SIZE_TYPE  in_idx  = (*input_tensor.indices[0])[batch_start + ip];
                    const VALUE_TYPE in_val  = (*input_tensor.values[0]) [batch_start + ip];
                    const SIZE_TYPE  wptr    = conn_ptrs[in_idx] + (w - work_offsets[ip]);
                    const VALUE_TYPE wval    = conn_val0[wptr];
                    const SIZE_TYPE  out_idx = conn_indices[wptr];
                    const VALUE_TYPE contrib = wval * in_val;

                    if (train) conn_val2[wptr] += contrib * solidify;

                    thread_output[batch_offset + out_idx] += contrib;

                    if (thread_contributions)
                        thread_contributions[in_idx] += in_val * wval;
                }
            }
            #pragma omp barrier
        }

        // ── Phase 2: parallel tree reduction — no critical, no atomics ───────
        // Each pass halves active threads. Active pairs write to disjoint
        // destinations so no synchronization is needed within a pass.
        // The implicit OpenMP barrier between passes is the only sync required.
        for (int stride = 1; stride < nthreads; stride <<= 1) {
            #pragma omp barrier
            const int src = tid + stride;
            if (tid % (stride << 1) == 0 && src < nthreads) {
                const VALUE_TYPE* src_out = all_outputs.data() + (size_t)src * num_outputs;
                for (SIZE_TYPE i = 0; i < num_outputs; ++i)
                    thread_output[i] += src_out[i];

                if (thread_contributions) {
                    const VALUE_TYPE* src_con =
                        all_contributions.data() + (size_t)src * num_inputs;
                    for (SIZE_TYPE i = 0; i < num_inputs; ++i)
                        thread_contributions[i] += src_con[i];
                }
            }
        }
    }   // implicit barrier — thread 0's slice now holds the full reduction

    // Copy thread 0's result into the caller's output buffer.
    const VALUE_TYPE* result = all_outputs.data();
    for (SIZE_TYPE i = 0; i < num_outputs; ++i)
        output[i] += result[i];

    if (original_contributions_output) {
        const VALUE_TYPE* con_result = all_contributions.data();
        for (SIZE_TYPE i = 0; i < num_inputs; ++i)
            original_contributions_output[i] += con_result[i];
    }
}

// ── backward ──────────────────────────────────────────────────────────────────

template <class SIZE_TYPE, class VALUE_TYPE>
void sisldo_backward(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& in_tensor,
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& out_grad_sparse,
    VALUE_TYPE* input_gradients,
    VALUE_TYPE* output_gradients,
    VALUE_TYPE* neuron_input_accum,
    VALUE_TYPE* neuron_grad_accum,
    const int num_cpus)
{
    for(SIZE_TYPE i=0;i<in_tensor.rows;++i){
        for(SIZE_TYPE j=(*in_tensor.ptrs[0])[i];j<(*in_tensor.ptrs[0])[i+1];++j){
            neuron_input_accum[(*in_tensor.indices[0])[j]]+=std::abs((*in_tensor.values[0])[j]);
        }
    }
    for(SIZE_TYPE i=0;i<out_grad_sparse.rows;++i){
        for(SIZE_TYPE j=(*out_grad_sparse.ptrs[0])[i];j<(*out_grad_sparse.ptrs[0])[i+1];++j){
            neuron_grad_accum[(*out_grad_sparse.indices[0])[j]]+=std::abs((*out_grad_sparse.values[0])[j]);
        }
    }
    if (connections_empty(weights.connections)) {
        return;
    }

    const SIZE_TYPE batch_size  = in_tensor.rows;
    const SIZE_TYPE num_inputs  = weights.connections.rows;
    const SIZE_TYPE num_outputs = weights.connections.cols;

    const auto& conn_ptrs    = *weights.connections.ptrs[0];
    const auto& conn_indices = *weights.connections.indices[0];
    const auto& conn_val0    = *weights.connections.values[0];
    auto&       conn_val1    = *weights.connections.values[1];

    std::vector<SIZE_TYPE> weight_grad_offsets;
    std::vector<SIZE_TYPE> input_grad_offsets;

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid      = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        std::vector<VALUE_TYPE> local_input_accum(num_inputs,  VALUE_TYPE(0));
        std::vector<VALUE_TYPE> local_grad_accum (num_outputs, VALUE_TYPE(0));

        for (SIZE_TYPE batch = 0; batch < batch_size; ++batch) {
            const SIZE_TYPE batch_start = (*in_tensor.ptrs[0])[batch];
            const SIZE_TYPE batch_end   = (*in_tensor.ptrs[0])[batch + 1];
            const SIZE_TYPE batch_nnz   = batch_end - batch_start;
            const SIZE_TYPE og_start    = (*out_grad_sparse.ptrs[0])[batch];
            const SIZE_TYPE og_end      = (*out_grad_sparse.ptrs[0])[batch + 1];

            #pragma omp single
            {
                weight_grad_offsets.resize(batch_nnz + 1);
                weight_grad_offsets[0] = 0;
                for (SIZE_TYPE i = 0; i < batch_nnz; ++i) {
                    const SIZE_TYPE in_idx = (*in_tensor.indices[0])[batch_start + i];
                    weight_grad_offsets[i + 1] = weight_grad_offsets[i]
                        + conn_ptrs[in_idx + 1] - conn_ptrs[in_idx];
                }

                input_grad_offsets.resize(num_inputs + 1);
                input_grad_offsets[0] = 0;
                for (SIZE_TYPE i = 0; i < num_inputs; ++i)
                    input_grad_offsets[i + 1] = input_grad_offsets[i]
                        + conn_ptrs[i + 1] - conn_ptrs[i];
            }

            // ── weight gradients + input accumulation ─────────────────────────
            {
                const SIZE_TYPE total_work = weight_grad_offsets[batch_nnz];
                const SIZE_TYPE chunk      = (total_work + nthreads - 1) / nthreads;
                const SIZE_TYPE w_start    = std::min((SIZE_TYPE)tid * chunk, total_work);
                const SIZE_TYPE w_end      = std::min(w_start + chunk, total_work);

                if (w_start < w_end) {
                    SIZE_TYPE ip      = static_cast<SIZE_TYPE>(
                        std::upper_bound(weight_grad_offsets.begin(), weight_grad_offsets.end(), w_start)
                        - weight_grad_offsets.begin()) - 1;
                    SIZE_TYPE last_ip = std::numeric_limits<SIZE_TYPE>::max();

                    for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                        while (ip + 1 < batch_nnz && weight_grad_offsets[ip + 1] <= w) ++ip;

                        const SIZE_TYPE  in_idx  = (*in_tensor.indices[0])[batch_start + ip];
                        const VALUE_TYPE in_val  = (*in_tensor.values[0]) [batch_start + ip];
                        const SIZE_TYPE  wptr    = conn_ptrs[in_idx] + (w - weight_grad_offsets[ip]);
                        const SIZE_TYPE  out_idx = conn_indices[wptr];

                        conn_val1[wptr] += output_gradients[out_idx * batch_size + batch] * in_val;

                        if (ip != last_ip) {
                            local_input_accum[in_idx] += std::abs(in_val);
                            last_ip = ip;
                        }
                    }
                }
            }

            // ── input gradients + grad accumulation ───────────────────────────
            {
                const SIZE_TYPE total_work = input_grad_offsets[num_inputs];
                const SIZE_TYPE chunk      = (total_work + nthreads - 1) / nthreads;
                const SIZE_TYPE w_start    = std::min((SIZE_TYPE)tid * chunk, total_work);
                const SIZE_TYPE w_end      = std::min(w_start + chunk, total_work);

                if (w_start < w_end) {
                    SIZE_TYPE in_idx = static_cast<SIZE_TYPE>(
                        std::upper_bound(input_grad_offsets.begin(), input_grad_offsets.end(), w_start)
                        - input_grad_offsets.begin()) - 1;
                    SIZE_TYPE og_ptr = og_start;

                    for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                        while (in_idx + 1 < num_inputs
                               && input_grad_offsets[in_idx + 1] <= w) {
                            ++in_idx;
                            og_ptr = og_start;
                        }

                        const SIZE_TYPE wptr     = conn_ptrs[in_idx] + (w - input_grad_offsets[in_idx]);
                        const SIZE_TYPE out_idx  = conn_indices[wptr];

                        while (og_ptr < og_end && (*out_grad_sparse.indices[0])[og_ptr] < out_idx)
                            ++og_ptr;

                        if (og_ptr >= og_end || (*out_grad_sparse.indices[0])[og_ptr] != out_idx)
                            continue;

                        const VALUE_TYPE og_val = (*out_grad_sparse.values[0])[og_ptr];
                        input_gradients[in_idx] += conn_val0[wptr] * og_val;
                        local_grad_accum[out_idx] += std::abs(og_val);
                    }
                }
            }

            #pragma omp barrier
        }
    }
}

// ── optim_weights ─────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUE_TYPE>
void sisldo_optim_weights(
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const VALUE_TYPE learning_rate,
    const int num_cpus)
{
    if (connections_empty(weights.connections)) return;

    const SIZE_TYPE nnz = weights.connections.nnz();
    auto& val0 = *weights.connections.values[0];
    auto& val1 = *weights.connections.values[1];
    auto& val2 = *weights.connections.values[2];

    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE i = 0; i < nnz; ++i) {
        val0[i] += (val1[i] * -learning_rate) / (VALUE_TYPE(1) + std::abs(val2[i]));
        val1[i]  = VALUE_TYPE(0);
    }
}

// ── genesis_build_probes ──────────────────────────────────────────────────────
// Runs on genesis thread. Wraps dense accumulators as single-row CSR views,
// runs top_k + outer product to build probes for weights_b.

template <typename SIZE_TYPE, typename VALUE_TYPE>
void genesis_build_probes(
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights_b,
    const VALUE_TYPE* neuron_input_accum,
    const VALUE_TYPE* neuron_grad_accum,
    const SIZE_TYPE num_inputs,
    const SIZE_TYPE num_outputs,
    const SIZE_TYPE k,
    const int num_cpus)
{
    // Build single-row CSR views over the dense accumulators.
    // indices are identity [0..n), values point into caller's array via no-op deleter.
    auto make_view_csr = [](const VALUE_TYPE* accum, SIZE_TYPE n)
        -> CSRInput<SIZE_TYPE, VALUE_TYPE>
    {
        CSRInput<SIZE_TYPE, VALUE_TYPE> csr;
        csr.rows = 1;
        csr.cols = n;

        csr.ptrs[0] = std::make_shared<std::vector<SIZE_TYPE>>(2);
        (*csr.ptrs[0])[0] = 0;
        (*csr.ptrs[0])[1] = n;

        csr.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(n);
        std::iota(csr.indices[0]->begin(), csr.indices[0]->end(), SIZE_TYPE(0));

        // Simplest correct approach: just copy the accumulator into a vector.
        // It's num_inputs or num_outputs floats — negligible vs model size.
        csr.values[0] = std::make_shared<std::vector<VALUE_TYPE>>(accum, accum + n);

        return csr;
    };

    // ── Degree-weighted accumulator scores ───────────────────────────────────
    // Input degree is free from CSR ptrs. Output degree is cached on the
    // weights struct and maintained incrementally by sisldo_optim_synaptogenesis
    // — no O(nnz) recount here.
    std::vector<VALUE_TYPE> weighted_in(num_inputs);
    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE i = 0; i < num_inputs; ++i)
        weighted_in[i] = std::abs(neuron_input_accum[i]) / (VALUE_TYPE(1) + VALUE_TYPE(weights_b.in_degree(i)));

    std::vector<VALUE_TYPE> weighted_out(num_outputs);
    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE i = 0; i < num_outputs; ++i)
        weighted_out[i] = std::abs(neuron_grad_accum[i]) / (VALUE_TYPE(1) + VALUE_TYPE(weights_b.out_degree[i]));

    auto in_csr  = top_k_csr<SIZE_TYPE, VALUE_TYPE>(weighted_in.data(), 1, num_inputs, k,  num_cpus);
    auto out_csr = top_k_csr<SIZE_TYPE, VALUE_TYPE>(weighted_out.data(), 1, num_outputs, k,  num_cpus);

    weights_b.probes = generate_new_weights_coo(in_csr, out_csr, num_cpus);

    /*
    // ── Filter out connections that already exist ─────────────────────────────
    // Without this, the top-k accumulators keep proposing the same high-activity
    // neuron pairs, which are already connected. synap_mark_duplicates would
    // exclude them from being added, but they consume the entire probe budget,
    // leaving no room for genuinely novel connections.
    //
    // We filter here — before storing as probes — so the budget is spent only
    // on connections the network does not yet have.
    //
    // The existing weights are CSR over inputs: conn_ptrs[in_row] → out_cols.
    // A probe entry (in_row, out_col) is novel iff out_col does not appear in
    // connections[conn_ptrs[in_row] .. conn_ptrs[in_row+1]).
    if (!connections_empty(weights_b.connections) &&
        raw_probes.indices[0] && !raw_probes.indices[0]->empty())
    {
        const auto& cp  = *weights_b.connections.ptrs[0];
        const auto& ci  = *weights_b.connections.indices[0];
        const SIZE_TYPE probe_nnz = raw_probes.nnz();

        const SIZE_TYPE* row_ptr = raw_probes.indices[0]->data();  // in_row per probe
        const SIZE_TYPE* col_ptr = raw_probes.indices[1]->data();  // out_col per probe
        const VALUE_TYPE* val_ptr = raw_probes.values[0]->data();

        std::vector<SIZE_TYPE>  new_rows, new_cols;
        std::vector<VALUE_TYPE> new_vals;
        new_rows.reserve(probe_nnz);
        new_cols.reserve(probe_nnz);
        new_vals.reserve(probe_nnz);

        // ── Parallel stream compaction ────────────────────────────────────────
        // Phase 1: mark novel probes (parallel, read-only on connections)
        // Phase 2: exclusive prefix scan to get write positions
        // Phase 3: parallel scatter into output arrays
        //
        // Each probe lookup is an independent binary search — embarrassingly
        // parallel. The only sequential work is the O(probe_nnz) prefix scan,
        // which is negligible compared to the searches.

        std::vector<SIZE_TYPE> keep(probe_nnz, SIZE_TYPE(0));

        #pragma omp parallel for num_threads(num_cpus) schedule(static)
        for (SIZE_TYPE p = 0; p < probe_nnz; ++p) {
            const SIZE_TYPE r  = row_ptr[p];
            const SIZE_TYPE c  = col_ptr[p];
            const SIZE_TYPE w0 = cp[r];
            const SIZE_TYPE w1 = cp[r + 1];
            keep[p] = std::binary_search(ci.begin() + w0, ci.begin() + w1, c)
                      ? SIZE_TYPE(0) : SIZE_TYPE(1);
        }

        // Exclusive prefix scan — gives each kept probe its write position.
        std::vector<SIZE_TYPE> scan(probe_nnz + 1, SIZE_TYPE(0));
        for (SIZE_TYPE p = 0; p < probe_nnz; ++p)
            scan[p + 1] = scan[p] + keep[p];
        const SIZE_TYPE new_nnz = scan[probe_nnz];

        if (new_nnz == 0) {
            raw_probes.ptrs       = 0;
            raw_probes.indices[0] = nullptr;
            raw_probes.indices[1] = nullptr;
            raw_probes.values[0]  = nullptr;
        } else {
            std::vector<SIZE_TYPE>  new_rows(new_nnz);
            std::vector<SIZE_TYPE>  new_cols(new_nnz);
            std::vector<VALUE_TYPE> new_vals(new_nnz);

            #pragma omp parallel for num_threads(num_cpus) schedule(static)
            for (SIZE_TYPE p = 0; p < probe_nnz; ++p) {
                if (keep[p]) {
                    const SIZE_TYPE out = scan[p];
                    new_rows[out] = row_ptr[p];
                    new_cols[out] = col_ptr[p];
                    new_vals[out] = val_ptr[p];
                }
            }

            raw_probes.ptrs        = new_nnz;
            *raw_probes.indices[0] = std::move(new_rows);
            *raw_probes.indices[1] = std::move(new_cols);
            *raw_probes.values[0]  = std::move(new_vals);
        }
    }*/

    //weights_b.probes = std::move(raw_probes);
}

// ── copy_matching_weights ─────────────────────────────────────────────────────
// O(nnz) parallel copy of weight/importance values from src into dst
// for every connection that exists in both. New connections in dst stay zero.

template <typename SIZE_TYPE, typename VALUE_TYPE>
void copy_matching_weights(
    const CSRSynapses<SIZE_TYPE, VALUE_TYPE>& src,
    CSRSynapses<SIZE_TYPE, VALUE_TYPE>& dst,
    const int num_cpus)
{
    const SIZE_TYPE rows    = dst.rows;
    const SIZE_TYPE dst_nnz = dst.ptrs[0] ? (*dst.ptrs[0])[rows] : 0;
    if (dst_nnz == 0) return;

    const auto& src_ptrs    = *src.ptrs[0];
    const auto& src_indices = *src.indices[0];
    const auto& src_val0    = *src.values[0];
    const auto& src_val2    = *src.values[2];
    const auto& dst_ptrs    = *dst.ptrs[0];
    const auto& dst_indices = *dst.indices[0];
    auto&       dst_val0    = *dst.values[0];
    auto&       dst_val2    = *dst.values[2];

    #pragma omp parallel num_threads(num_cpus)
    {
        const int      tid      = omp_get_thread_num();
        const int      nthreads = omp_get_num_threads();
        const SIZE_TYPE chunk   = (dst_nnz + nthreads - 1) / nthreads;
        const SIZE_TYPE w_start = std::min((SIZE_TYPE)tid * chunk, dst_nnz);
        const SIZE_TYPE w_end   = std::min(w_start + chunk, dst_nnz);

        if (w_start < w_end) {
            SIZE_TYPE r = static_cast<SIZE_TYPE>(
                std::upper_bound(dst_ptrs.begin(), dst_ptrs.end(), w_start)
                - dst_ptrs.begin()) - 1;

            SIZE_TYPE s = src_ptrs[r];
            const SIZE_TYPE first_dst_col = dst_indices[w_start];
            while (s < src_ptrs[r + 1] && src_indices[s] < first_dst_col) ++s;

            for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                while (r + 1 < rows && dst_ptrs[r + 1] <= w) {
                    ++r;
                    s = src_ptrs[r];
                }

                const SIZE_TYPE dst_col = dst_indices[w];
                while (s < src_ptrs[r + 1] && src_indices[s] < dst_col) ++s;

                if (s < src_ptrs[r + 1] && src_indices[s] == dst_col) {
                    dst_val0[w] = src_val0[s];
                    dst_val2[w] = src_val2[s];
                    // values[1] (grad) intentionally left zero
                }
            }
        }
    }
}
// ── Importance decay ──────────────────────────────────────────────────────────
// Decays toward zero. Step size = rate / (1 + |importance|):
//   near 0   → step ≈ rate        (fast pruning of weak connections)
//   large    → step ≈ rate/|imp|  (stable, high-importance connections barely move)

template <typename SIZE_TYPE, typename VALUE_TYPE>
void sisldo_decay_importance(
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const VALUE_TYPE rate,
    const int num_cpus)
{
    if (connections_empty(weights.connections)) return;

    const SIZE_TYPE nnz = weights.connections.nnz();
    auto& imp = *weights.connections.values[2];

    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE i = 0; i < nnz; ++i) {
        const VALUE_TYPE v = imp[i];
        imp[i] = v - std::copysign(rate / (VALUE_TYPE(1) + std::abs(v)), v);
    }
}

// ── Synaptogenesis sub-functions ──────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUE_TYPE>
void synap_mark_duplicates(
    const SIZE_TYPE nnz_c,
    const SIZE_TYPE nnz_p,
    const std::vector<SIZE_TYPE>& cp,
    const std::vector<SIZE_TYPE>& ci,
    const std::vector<VALUE_TYPE>& cv2,
    const std::vector<SIZE_TYPE>& pp,
    const std::vector<SIZE_TYPE>& pi,
    const std::vector<VALUE_TYPE>& pv,
    const SIZE_TYPE rows,
    std::vector<VALUE_TYPE>& merged_imp_c,   // out: per-connection merged importance
    std::vector<int8_t>& is_dup_p,           // out: 1 if probe is duplicate of connection
    const int num_cpus)
{
    auto row_of_c = [&](SIZE_TYPE c) -> SIZE_TYPE {
        return static_cast<SIZE_TYPE>(
            std::upper_bound(cp.begin(), cp.end(), c) - cp.begin()) - 1;
    };

    auto find_probe = [&](SIZE_TYPE row, SIZE_TYPE col) -> SIZE_TYPE {
        const SIZE_TYPE p0 = pp[row], p1 = pp[row + 1];
        auto it = std::lower_bound(pi.begin() + p0, pi.begin() + p1, col);
        const SIZE_TYPE pos = static_cast<SIZE_TYPE>(it - pi.begin());
        return (pos < p1 && pi[pos] == col) ? pos : nnz_p;
    };

    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE c = 0; c < nnz_c; ++c) {
        const SIZE_TYPE r_c = row_of_c(c);
        const SIZE_TYPE pp_ = find_probe(r_c, ci[c]);
        if (pp_ != nnz_p) {
            merged_imp_c[c] = std::max(cv2[c], pv[pp_]);
            is_dup_p[pp_]   = 1;
        } else {
            merged_imp_c[c] = cv2[c];
        }
    }
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
VALUE_TYPE synap_compute_cutoff(
    const SIZE_TYPE nnz_c,
    const SIZE_TYPE nnz_p,
    const SIZE_TYPE max_weights,
    const std::vector<VALUE_TYPE>& merged_imp_c,
    const std::vector<VALUE_TYPE>& pv,
    const std::vector<int8_t>& is_dup_p)
{
    const SIZE_TYPE dup_count     = static_cast<SIZE_TYPE>(
        std::count(is_dup_p.begin(), is_dup_p.end(), int8_t(1)));
    const SIZE_TYPE total_merged  = nnz_c + nnz_p - dup_count;

    if (total_merged <= max_weights)
        return std::numeric_limits<VALUE_TYPE>::lowest();

    std::vector<VALUE_TYPE> all_imp;
    all_imp.reserve(total_merged);
    for (SIZE_TYPE c = 0; c < nnz_c; ++c)
        all_imp.push_back(merged_imp_c[c]);
    for (SIZE_TYPE p = 0; p < nnz_p; ++p)
        if (!is_dup_p[p]) all_imp.push_back(pv[p]);

    const SIZE_TYPE drop_count = total_merged - max_weights;
    std::nth_element(all_imp.begin(), all_imp.begin() + drop_count, all_imp.end());
    return all_imp[drop_count];
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void synap_build_scans(
    const SIZE_TYPE nnz_c,
    const SIZE_TYPE nnz_p,
    const VALUE_TYPE importance_cutoff,
    const std::vector<VALUE_TYPE>& merged_imp_c,
    const std::vector<VALUE_TYPE>& pv,
    const std::vector<int8_t>& is_dup_p,
    std::vector<SIZE_TYPE>& scan_c,   // out: size nnz_c + 1
    std::vector<SIZE_TYPE>& scan_p)   // out: size nnz_p + 1
{
    scan_c.resize(nnz_c + 1);
    scan_p.resize(nnz_p + 1);

    scan_c[0] = 0;
    for (SIZE_TYPE c = 0; c < nnz_c; ++c)
        scan_c[c + 1] = scan_c[c] + (merged_imp_c[c] >= importance_cutoff ? 1 : 0);

    scan_p[0] = 0;
    for (SIZE_TYPE p = 0; p < nnz_p; ++p)
        scan_p[p + 1] = scan_p[p]
            + (!is_dup_p[p] && pv[p] >= importance_cutoff ? 1 : 0);
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void synap_parallel_fill(
    const SIZE_TYPE nnz_c,
    const SIZE_TYPE nnz_p,
    const SIZE_TYPE rows,
    const SIZE_TYPE new_nnz,
    const VALUE_TYPE importance_cutoff,
    const std::vector<SIZE_TYPE>& cp,
    const std::vector<SIZE_TYPE>& ci,
    const std::vector<VALUE_TYPE>& cv0,
    const std::vector<VALUE_TYPE>& cv1,
    const std::vector<VALUE_TYPE>& cv2,
    const std::vector<SIZE_TYPE>& pp,
    const std::vector<SIZE_TYPE>& pi,
    const std::vector<VALUE_TYPE>& pv,
    const std::vector<VALUE_TYPE>& merged_imp_c,
    const std::vector<int8_t>&    is_dup_p,
    const std::vector<SIZE_TYPE>& scan_c,
    const std::vector<SIZE_TYPE>& scan_p,
    std::vector<SIZE_TYPE>&  out_i,
    std::vector<VALUE_TYPE>& out_v0,
    std::vector<VALUE_TYPE>& out_v1,
    std::vector<VALUE_TYPE>& out_v2,
    const int num_cpus)
{
    constexpr SIZE_TYPE SENTINEL = std::numeric_limits<SIZE_TYPE>::max();

    auto row_of = [&](const std::vector<SIZE_TYPE>& ptrs, SIZE_TYPE idx) -> SIZE_TYPE {
        return static_cast<SIZE_TYPE>(
            std::upper_bound(ptrs.begin(), ptrs.end(), idx) - ptrs.begin()) - 1;
    };

    // co_rank: given output position w_start, find (c, p) such that exactly
    // w_start kept entries from connections[0..c) and probes[0..p) precede it.
    auto co_rank = [&](SIZE_TYPE w_start) -> std::pair<SIZE_TYPE, SIZE_TYPE> {
        if (w_start == 0)       return {0, 0};
        if (w_start >= new_nnz) return {nnz_c, nnz_p};

        SIZE_TYPE lo = 0, hi = nnz_c;
        while (lo < hi) {
            const SIZE_TYPE mid   = lo + (hi - lo) / 2;
            const SIZE_TYPE r_mid = row_of(cp, mid);
            const SIZE_TYPE prk   = pp[r_mid] + static_cast<SIZE_TYPE>(
                std::lower_bound(pi.begin() + pp[r_mid], pi.begin() + pp[r_mid + 1], ci[mid])
                - (pi.begin() + pp[r_mid]));
            if (scan_c[mid] + scan_p[prk] < w_start) lo = mid + 1;
            else                                      hi = mid;
        }
        const SIZE_TYPE c_start = lo;
        const SIZE_TYPE k       = w_start - scan_c[c_start];
        const SIZE_TYPE p_start = static_cast<SIZE_TYPE>(
            std::lower_bound(scan_p.begin() + 1, scan_p.end(), k + 1)
            - scan_p.begin()) - 1;
        return {c_start, p_start};
    };

    #pragma omp parallel num_threads(num_cpus)
    {
        const int      tid      = omp_get_thread_num();
        const int      nthreads = omp_get_num_threads();
        const SIZE_TYPE chunk   = (new_nnz + nthreads - 1) / nthreads;
        const SIZE_TYPE w_start = std::min((SIZE_TYPE)tid * chunk, new_nnz);
        const SIZE_TYPE w_end   = std::min(w_start + chunk, new_nnz);
        const SIZE_TYPE w_count = w_end - w_start;

        std::vector<SIZE_TYPE>  local_i (w_count);
        std::vector<VALUE_TYPE> local_v0(w_count);
        std::vector<VALUE_TYPE> local_v1(w_count, VALUE_TYPE(0));
        std::vector<VALUE_TYPE> local_v2(w_count);

        if (w_count > 0) {
            auto [c, p] = co_rank(w_start);

            SIZE_TYPE r_c = (c < nnz_c) ? row_of(cp, c) : rows;
            SIZE_TYPE r_p = (p < nnz_p) ? row_of(pp, p) : rows;

            for (SIZE_TYPE w = 0; w < w_count; ++w) {
                while (c < nnz_c && merged_imp_c[c] < importance_cutoff) {
                    ++c;
                    while (r_c + 1 < rows && cp[r_c + 1] <= c) ++r_c;
                }
                while (p < nnz_p && (is_dup_p[p] || pv[p] < importance_cutoff)) {
                    ++p;
                    while (r_p + 1 < rows && pp[r_p + 1] <= p) ++r_p;
                }

                const SIZE_TYPE row_c_val = (c < nnz_c) ? r_c   : rows;
                const SIZE_TYPE col_c_val = (c < nnz_c) ? ci[c]  : SENTINEL;
                const SIZE_TYPE row_p_val = (p < nnz_p) ? r_p   : rows;
                const SIZE_TYPE col_p_val = (p < nnz_p) ? pi[p]  : SENTINEL;

                const bool c_first =
                    (row_c_val < row_p_val) ||
                    (row_c_val == row_p_val && col_c_val < col_p_val);

                if (c_first) {
                    local_i [w] = col_c_val;
                    local_v0[w] = cv0[c];
                    local_v1[w] = VALUE_TYPE(0);
                    local_v2[w] = merged_imp_c[c];
                    ++c;
                    while (r_c + 1 < rows && cp[r_c + 1] <= c) ++r_c;
                } else {
                    local_i [w] = col_p_val;
                    local_v0[w] = VALUE_TYPE(0);
                    local_v1[w] = VALUE_TYPE(0);
                    local_v2[w] = pv[p];
                    ++p;
                    while (r_p + 1 < rows && pp[r_p + 1] <= p) ++r_p;
                }
            }
        }

        // All reads done — resize in-place (no realloc; capacity guaranteed)
        #pragma omp single
        {
            out_i .resize(new_nnz);
            out_v0.resize(new_nnz);
            out_v1.resize(new_nnz, VALUE_TYPE(0));
            out_v2.resize(new_nnz);
        }

        for (SIZE_TYPE w = 0; w < w_count; ++w) {
            out_i [w_start + w] = local_i [w];
            out_v0[w_start + w] = local_v0[w];
            out_v1[w_start + w] = local_v1[w];
            out_v2[w_start + w] = local_v2[w];
        }
    }
}

template <typename SIZE_TYPE>
std::shared_ptr<std::vector<SIZE_TYPE>> synap_build_ptrs(
    const SIZE_TYPE rows,
    const std::vector<SIZE_TYPE>& cp,
    const std::vector<SIZE_TYPE>& pp,
    const std::vector<SIZE_TYPE>& scan_c,
    const std::vector<SIZE_TYPE>& scan_p,
    const int num_cpus)
{
    auto new_ptrs_vec = std::make_shared<std::vector<SIZE_TYPE>>(rows + 1);
    auto& new_ptrs = *new_ptrs_vec;
    new_ptrs[0] = 0;

    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE r = 0; r < rows; ++r)
        new_ptrs[r + 1] = (scan_c[cp[r + 1]] - scan_c[cp[r]])
                        + (scan_p[pp[r + 1]] - scan_p[pp[r]]);

    for (SIZE_TYPE r = 0; r < rows; ++r)
        new_ptrs[r + 1] += new_ptrs[r];

    return new_ptrs_vec;
}

// ── sisldo_optim_synaptogenesis ───────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUE_TYPE>
void sisldo_optim_synaptogenesis(
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    const VALUE_TYPE learning_rate,
    const VALUE_TYPE importance_beta,
    const SIZE_TYPE max_weights,
    const int num_cpus)
{
    if (!weights.probes.indices[0] || weights.probes.indices[0]->empty()) return;

    reserve_connections(weights.connections, max_weights);

    const SIZE_TYPE rows  = weights.connections.rows;
    const SIZE_TYPE nnz_c = weights.connections.nnz();

    {
        auto& pval = *weights.probes.values[0];
        const SIZE_TYPE pnnz = weights.probes.nnz();
        #pragma omp parallel for num_threads(num_cpus) schedule(static)
        for (SIZE_TYPE i = 0; i < pnnz; ++i)
            pval[i] = -(pval[i] / learning_rate) * importance_beta;
    }

    auto probes_csr    = to_csr(weights.probes, num_cpus);
    const SIZE_TYPE nnz_p = probes_csr.nnz();

    const auto& cp  = *weights.connections.ptrs[0];
    const auto& ci  = *weights.connections.indices[0];
    const auto& cv0 = *weights.connections.values[0];
    const auto& cv1 = *weights.connections.values[1];
    const auto& cv2 = *weights.connections.values[2];
    const auto& pp  = *probes_csr.ptrs[0];
    const auto& pi  = *probes_csr.indices[0];
    const auto& pv  = *probes_csr.values[0];

    std::vector<VALUE_TYPE> merged_imp_c(nnz_c);
    std::vector<int8_t>     is_dup_p(nnz_p, 0);
    synap_mark_duplicates(nnz_c, nnz_p, cp, ci, cv2, pp, pi, pv, rows,
                          merged_imp_c, is_dup_p, num_cpus);

    const VALUE_TYPE importance_cutoff = synap_compute_cutoff(
        nnz_c, nnz_p, max_weights, merged_imp_c, pv, is_dup_p);

    std::vector<SIZE_TYPE> scan_c, scan_p;
    synap_build_scans(nnz_c, nnz_p, importance_cutoff,
                      merged_imp_c, pv, is_dup_p, scan_c, scan_p);

    const SIZE_TYPE new_nnz = scan_c[nnz_c] + scan_p[nnz_p];

    if (new_nnz == 0) {
        weights.connections.indices[0]->clear();
        weights.connections.values[0] ->clear();
        weights.connections.values[1] ->clear();
        weights.connections.values[2] ->clear();
        std::fill(weights.connections.ptrs[0]->begin(),
                  weights.connections.ptrs[0]->end(), SIZE_TYPE(0));
        weights.probes.ptrs = 0;
        weights.probes.indices[0] = nullptr;
        weights.probes.indices[1] = nullptr;
        weights.probes.values[0]  = nullptr;
        return;
    }

    synap_parallel_fill(nnz_c, nnz_p, rows, new_nnz, importance_cutoff,
                        cp, ci, cv0, cv1, cv2, pp, pi, pv,
                        merged_imp_c, is_dup_p, scan_c, scan_p,
                        *weights.connections.indices[0],
                        *weights.connections.values[0],
                        *weights.connections.values[1],
                        *weights.connections.values[2],
                        num_cpus);

    for (SIZE_TYPE w = 0; w < new_nnz; ++w)
        assert((*weights.connections.indices[0])[w] < weights.connections.cols);

    /*
    note: synap_build_ptrs captures cp by const ref. That's still pointing into the old ptrs[0] vector, 
    which is fine because synap_build_ptrs only reads it and the new shared_ptr is assigned afterward. 
    If you reorder those two operations, cp becomes a dangling ref, so the ptrs[0] = assignment must stay last.
    */
    weights.connections.ptrs[0] = synap_build_ptrs(
        rows, cp, pp, scan_c, scan_p, num_cpus);

    weights.probes.ptrs       = 0;
    weights.probes.indices[0] = nullptr;
    weights.probes.indices[1] = nullptr;
    weights.probes.values[0]  = nullptr;

    // Rebuild cached out_degree from the new connection indices.
    // in_degree is free from CSR ptrs — no storage or rebuild needed.
    const SIZE_TYPE n_out = weights.connections.cols;
    weights.out_degree.assign(n_out, 0u);
    if (new_nnz > 0) {
        const auto& new_ci = *weights.connections.indices[0];
        #pragma omp parallel for num_threads(num_cpus) schedule(static)
        for (SIZE_TYPE w = 0; w < new_nnz; ++w) {
            #pragma omp atomic
            weights.out_degree[new_ci[w]]++;
        }
    }
}