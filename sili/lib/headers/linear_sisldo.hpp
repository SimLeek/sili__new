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

// ── generate_new_weights_csc ──────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUE_TYPE>
COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> generate_new_weights_csc(
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& input_tensor,
    const CSRInput<SIZE_TYPE, VALUE_TYPE>& output_gradient_tensor,
    const SIZE_TYPE k,
    const int num_cpus = 4)
{
    auto top_in  = top_k(input_tensor,           k, true, num_cpus, ComparatorGT<VALUE_TYPE, VALUE_TYPE>());
    auto top_out = top_k(output_gradient_tensor, k, true, num_cpus, ComparatorGT<VALUE_TYPE, VALUE_TYPE>());

    SIZE_TYPE total_reserve = 0;
    for (SIZE_TYPE batch = 0; batch < top_in.rows; ++batch) {
        const SIZE_TYPE in_len  = (*top_in.ptrs[0]) [batch + 1] - (*top_in.ptrs[0]) [batch];
        const SIZE_TYPE out_len = (*top_out.ptrs[0])[batch + 1] - (*top_out.ptrs[0])[batch];
        total_reserve += in_len * out_len;
    }

    if (total_reserve == 0)
        return COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>();

    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> gen_coo;
    gen_coo.ptrs     = total_reserve;
    gen_coo.indices[0] = std::make_shared<std::vector<SIZE_TYPE>> (total_reserve);
    gen_coo.indices[1] = std::make_shared<std::vector<SIZE_TYPE>> (total_reserve);
    gen_coo.values[0]  = std::make_shared<std::vector<VALUE_TYPE>>(total_reserve);
    gen_coo.cols = top_out.cols;
    gen_coo.rows = top_in.cols;

    outer_product(top_in, top_out, gen_coo);

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

    std::vector<SIZE_TYPE> work_offsets;

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid      = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        std::vector<VALUE_TYPE> thread_output(num_outputs, VALUE_TYPE(0));
        std::vector<VALUE_TYPE> thread_contributions;
        if (original_contributions_output != nullptr)
            thread_contributions.resize(num_inputs, VALUE_TYPE(0));

        for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
            const SIZE_TYPE batch_start = (*input_tensor.ptrs[0])[batch];
            const SIZE_TYPE batch_end   = (*input_tensor.ptrs[0])[batch + 1];
            const SIZE_TYPE batch_nnz   = batch_end - batch_start;
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

                    const SIZE_TYPE  in_idx   = (*input_tensor.indices[0])[batch_start + ip];
                    const VALUE_TYPE in_val   = (*input_tensor.values[0]) [batch_start + ip];
                    const SIZE_TYPE  wptr     = conn_ptrs[in_idx] + (w - work_offsets[ip]);
                    const VALUE_TYPE wval     = conn_val0[wptr];
                    const SIZE_TYPE  out_idx  = conn_indices[wptr];
                    const VALUE_TYPE contrib  = wval * in_val;

                    if (train) conn_val2[wptr] += contrib * solidify;

                    thread_output[batch_offset + out_idx] += contrib;

                    if (original_contributions_output != nullptr)
                        thread_contributions[in_idx] += in_val * wval;
                }
            }
            #pragma omp barrier
        }

        #pragma omp critical
        {
            for (SIZE_TYPE i = 0; i < num_outputs; ++i)
                output[i] += thread_output[i];

            if (original_contributions_output != nullptr) {
                for (SIZE_TYPE i = 0; i < num_inputs; ++i)
                    original_contributions_output[i] += thread_contributions[i];
            }
        }
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

        // Non-owning view — no-op deleter on the shared_ptr<vector>,
        // inner vector wraps the raw pointer without owning it
        csr.values[0] = std::shared_ptr<std::vector<VALUE_TYPE>>(
            new std::vector<VALUE_TYPE>(),
            [](std::vector<VALUE_TYPE>* p){ delete p; });
        // assign data pointer without taking ownership via a non-owning trick:
        // use the aliasing constructor to share lifetime with a dummy owner
        auto dummy = std::make_shared<int>(0);
        csr.values[0] = std::shared_ptr<std::vector<VALUE_TYPE>>(
            dummy,
            // aliasing: the vector is a non-owning wrapper around accum
            // We construct a vector that views the memory but doesn't own it
            // — safest to just copy for the accumulator size, it's 1 row
            [](std::vector<VALUE_TYPE>*){}
        );
        // Simplest correct approach: just copy the accumulator into a vector.
        // It's num_inputs or num_outputs floats — negligible vs model size.
        csr.values[0] = std::make_shared<std::vector<VALUE_TYPE>>(accum, accum + n);

        return csr;
    };

    auto in_csr  = make_view_csr(neuron_input_accum, num_inputs);
    auto out_csr = make_view_csr(neuron_grad_accum,  num_outputs);

    weights_b.probes = generate_new_weights_csc(in_csr, out_csr, k, num_cpus);
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

// ── optim_synaptogenesis ──────────────────────────────────────────────────────
// Three-way sorted merge: output = (connections \ to_remove) ∪ probes
// Enforces max_weights — reserves once, resizes down after merge.
// importance_beta: rate at which probe outer-product values become importance scores.

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
    constexpr SIZE_TYPE SENTINEL = std::numeric_limits<SIZE_TYPE>::max();

    // ── Convert probe importance scores ──────────────────────────────────────
    {
        auto& pval = *weights.probes.values[0];
        const SIZE_TYPE pnnz = weights.probes.nnz();
        #pragma omp parallel for num_threads(num_cpus) schedule(static)
        for (SIZE_TYPE i = 0; i < pnnz; ++i)
            pval[i] = -(pval[i] / learning_rate) * importance_beta;
    }

    auto probes_csr = to_csr(weights.probes, num_cpus);
    const SIZE_TYPE nnz_p = probes_csr.nnz();

    // Const refs to old data — valid until resize (which won't reallocate
    // because reserve_connections guarantees capacity >= max_weights >= new_nnz)
    const auto& cp  = *weights.connections.ptrs[0];
    const auto& ci  = *weights.connections.indices[0];
    const auto& cv0 = *weights.connections.values[0];
    const auto& cv1 = *weights.connections.values[1];
    const auto& cv2 = *weights.connections.values[2];
    const auto& pp  = *probes_csr.ptrs[0];
    const auto& pi  = *probes_csr.indices[0];
    const auto& pv  = *probes_csr.values[0];

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Row of flat connection index c — O(log rows)
    auto row_of_c = [&](SIZE_TYPE c) -> SIZE_TYPE {
        return static_cast<SIZE_TYPE>(
            std::upper_bound(cp.begin(), cp.end(), c) - cp.begin()) - 1;
    };

    // Row of flat probe index p — O(log rows)
    auto row_of_p = [&](SIZE_TYPE p) -> SIZE_TYPE {
        return static_cast<SIZE_TYPE>(
            std::upper_bound(pp.begin(), pp.end(), p) - pp.begin()) - 1;
    };

    // Flat probe index for (row, col). Returns nnz_p if absent.
    auto find_probe = [&](SIZE_TYPE row, SIZE_TYPE col) -> SIZE_TYPE {
        const SIZE_TYPE p0 = pp[row], p1 = pp[row + 1];
        auto it = std::lower_bound(pi.begin() + p0, pi.begin() + p1, col);
        const SIZE_TYPE pos = static_cast<SIZE_TYPE>(it - pi.begin());
        return (pos < p1 && pi[pos] == col) ? pos : nnz_p;
    };

    // # probes whose key (row, col) is strictly less than key of connection c.
    // = flat probe index at which a probe matching c's key would sit.
    auto p_rank_for_c = [&](SIZE_TYPE c, SIZE_TYPE r_c) -> SIZE_TYPE {
        return pp[r_c] + static_cast<SIZE_TYPE>(
            std::lower_bound(pi.begin() + pp[r_c], pi.begin() + pp[r_c + 1], ci[c])
            - (pi.begin() + pp[r_c]));
    };

    // ── Phase 1: Mark duplicates, compute merged importances (parallel) ───────
    std::vector<VALUE_TYPE> merged_imp_c(nnz_c);
    std::vector<int8_t>     is_dup_p(nnz_p, 0);

    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE c = 0; c < nnz_c; ++c) {
        const SIZE_TYPE r_c  = row_of_c(c);
        const SIZE_TYPE pp_  = find_probe(r_c, ci[c]);
        if (pp_ != nnz_p) {
            // Each (row,col) is unique in connections, so no two connections
            // map to the same probe — no write-write race on is_dup_p.
            merged_imp_c[c] = std::max(cv2[c], pv[pp_]);
            is_dup_p[pp_] = 1;
        } else {
            merged_imp_c[c] = cv2[c];
        }
    }

    // ── Phase 1b: Importance cutoff via nth_element if over budget ────────────
    SIZE_TYPE dup_count = 0;
    for (SIZE_TYPE p = 0; p < nnz_p; ++p) dup_count += is_dup_p[p];
    const SIZE_TYPE total_merged = nnz_c + nnz_p - dup_count;

    VALUE_TYPE importance_cutoff = std::numeric_limits<VALUE_TYPE>::lowest();

    if (total_merged > max_weights) {
        std::vector<VALUE_TYPE> all_imp;
        all_imp.reserve(total_merged);
        for (SIZE_TYPE c = 0; c < nnz_c; ++c)
            all_imp.push_back(merged_imp_c[c]);
        for (SIZE_TYPE p = 0; p < nnz_p; ++p)
            if (!is_dup_p[p]) all_imp.push_back(pv[p]);

        const SIZE_TYPE drop_count = total_merged - max_weights;
        std::nth_element(all_imp.begin(), all_imp.begin() + drop_count, all_imp.end());
        importance_cutoff = all_imp[drop_count];
    }

    // ── Phase 2: Exclusive prefix scans — O(nnz_c + nnz_p) serial ───────────
    // scan_c[c] = # kept connections in [0..c)
    // scan_p[p] = # kept probes     in [0..p)
    std::vector<SIZE_TYPE> scan_c(nnz_c + 1), scan_p(nnz_p + 1);
    scan_c[0] = 0;
    for (SIZE_TYPE c = 0; c < nnz_c; ++c)
        scan_c[c + 1] = scan_c[c] + (merged_imp_c[c] >= importance_cutoff ? 1 : 0);
    scan_p[0] = 0;
    for (SIZE_TYPE p = 0; p < nnz_p; ++p)
        scan_p[p + 1] = scan_p[p] + (!is_dup_p[p] && pv[p] >= importance_cutoff ? 1 : 0);

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

    // ── Co-rank ───────────────────────────────────────────────────────────────
    // Finds (c_start, p_start) such that exactly w_start kept elements from
    // connections[0..c_start) and probes[0..p_start) precede output[w_start].
    //
    // out_pos(c) = scan_c[c] + scan_p[p_rank_for_c(c)] is non-decreasing in c.
    // Binary search for first c where out_pos(c) >= w_start: O(log(nnz_c) * log(rows)).
    // Then p_start derived from: # kept probes before w_start = w_start - scan_c[c_start].
    auto co_rank = [&](SIZE_TYPE w_start) -> std::pair<SIZE_TYPE, SIZE_TYPE> {
        if (w_start == 0)       return {0, 0};
        if (w_start >= new_nnz) return {nnz_c, nnz_p};

        SIZE_TYPE lo = 0, hi = nnz_c;
        while (lo < hi) {
            const SIZE_TYPE mid   = lo + (hi - lo) / 2;
            const SIZE_TYPE r_mid = row_of_c(mid);
            const SIZE_TYPE prk   = p_rank_for_c(mid, r_mid);
            if (scan_c[mid] + scan_p[prk] < w_start) lo = mid + 1;
            else                                      hi = mid;
        }
        const SIZE_TYPE c_start = lo;

        // k-th kept probe (0-indexed k) has raw index:
        //   lower_bound(scan_p[1..end], k+1) - scan_p.begin() - 1
        const SIZE_TYPE k = w_start - scan_c[c_start];
        const SIZE_TYPE p_start = static_cast<SIZE_TYPE>(
            std::lower_bound(scan_p.begin() + 1, scan_p.end(), k + 1)
            - scan_p.begin()) - 1;

        return {c_start, p_start};
    };

    // ── Phase 3: Parallel read into local buffers, then resize, then write ────
    //
    // READ before RESIZE: reserve_connections guarantees ci.data() remains valid
    // after resize (no reallocation). But raw indices into ci[c] for c >= new_nnz
    // would be formally OOB after resize. omp single provides implicit barriers
    // before and after, separating all reads from the resize and all writes.
    #pragma omp parallel num_threads(num_cpus)
    {
        const int      tid      = omp_get_thread_num();
        const int      nthreads = omp_get_num_threads();
        const SIZE_TYPE chunk   = (new_nnz + nthreads - 1) / nthreads;
        const SIZE_TYPE w_start = std::min((SIZE_TYPE)tid * chunk, new_nnz);
        const SIZE_TYPE w_end   = std::min(w_start + chunk, new_nnz);
        const SIZE_TYPE w_count = w_end - w_start;

        // Local buffers bounded by new_nnz / num_cpus entries
        std::vector<SIZE_TYPE>  local_i (w_count);
        std::vector<VALUE_TYPE> local_v0(w_count);
        std::vector<VALUE_TYPE> local_v1(w_count, VALUE_TYPE(0));
        std::vector<VALUE_TYPE> local_v2(w_count);

        if (w_count > 0) {
            auto [c, p] = co_rank(w_start);

            // Maintain current rows incrementally — O(1) amortized per advance
            SIZE_TYPE r_c = (c < nnz_c) ? row_of_c(c) : rows;
            SIZE_TYPE r_p = (p < nnz_p) ? row_of_p(p) : rows;

            for (SIZE_TYPE w = 0; w < w_count; ++w) {
                // Skip non-kept connections
                while (c < nnz_c && merged_imp_c[c] < importance_cutoff) {
                    ++c;
                    while (r_c + 1 < rows && cp[r_c + 1] <= c) ++r_c;
                }
                // Skip duplicate or non-kept probes
                while (p < nnz_p && (is_dup_p[p] || pv[p] < importance_cutoff)) {
                    ++p;
                    while (r_p + 1 < rows && pp[r_p + 1] <= p) ++r_p;
                }

                const SIZE_TYPE row_c_val = (c < nnz_c) ? r_c  : rows;
                const SIZE_TYPE col_c_val = (c < nnz_c) ? ci[c] : SENTINEL;
                const SIZE_TYPE row_p_val = (p < nnz_p) ? r_p  : rows;
                const SIZE_TYPE col_p_val = (p < nnz_p) ? pi[p] : SENTINEL;

                // Keys are disjoint (duplicates removed), so strict < is correct
                const bool c_first =
                    (row_c_val < row_p_val) ||
                    (row_c_val == row_p_val && col_c_val < col_p_val);

                if (c_first) {
                    local_i [w] = col_c_val;
                    local_v0[w] = cv0[c];
                    local_v1[w] = VALUE_TYPE(0);  // grad reset on restructure
                    local_v2[w] = merged_imp_c[c];
                    ++c;
                    while (r_c + 1 < rows && cp[r_c + 1] <= c) ++r_c;
                } else {
                    local_i [w] = col_p_val;
                    local_v0[w] = VALUE_TYPE(0);  // new connection, no weight yet
                    local_v1[w] = VALUE_TYPE(0);
                    local_v2[w] = pv[p];
                    ++p;
                    while (r_p + 1 < rows && pp[r_p + 1] <= p) ++r_p;
                }
            }
        }
        // ── All reads complete. omp single resizes (implicit barrier before+after).
        #pragma omp single
        {
            weights.connections.indices[0]->resize(new_nnz);
            weights.connections.values[0] ->resize(new_nnz);
            weights.connections.values[1] ->resize(new_nnz, VALUE_TYPE(0));
            weights.connections.values[2] ->resize(new_nnz);
        }
        // ── Write local buffers to output ─────────────────────────────────────
        auto& out_i  = *weights.connections.indices[0];
        auto& out_v0 = *weights.connections.values[0];
        auto& out_v1 = *weights.connections.values[1];
        auto& out_v2 = *weights.connections.values[2];

        for (SIZE_TYPE w = 0; w < w_count; ++w) {
            out_i [w_start + w] = local_i [w];
            out_v0[w_start + w] = local_v0[w];
            out_v1[w_start + w] = local_v1[w];
            out_v2[w_start + w] = local_v2[w];
        }
    }

    // ── Phase 4: Update ptrs ──────────────────────────────────────────────────
    // scan_c and scan_p give per-row kept counts directly.
    // Note: cp is still valid here (we only resized the data vectors, not ptrs).
    auto new_ptrs_vec = std::make_shared<std::vector<SIZE_TYPE>>(rows + 1);
    auto& new_ptrs = *new_ptrs_vec;
    new_ptrs[0] = 0;

    #pragma omp parallel for num_threads(num_cpus) schedule(static)
    for (SIZE_TYPE r = 0; r < rows; ++r)
        new_ptrs[r + 1] = (scan_c[cp[r + 1]] - scan_c[cp[r]])
                        + (scan_p[pp[r + 1]] - scan_p[pp[r]]);

    // Serial prefix sum over counts — O(rows), negligible
    for (SIZE_TYPE r = 0; r < rows; ++r)
        new_ptrs[r + 1] += new_ptrs[r];

    weights.connections.ptrs[0] = new_ptrs_vec;

    // ── Clear probes ──────────────────────────────────────────────────────────
    weights.probes.ptrs       = 0;
    weights.probes.indices[0] = nullptr;
    weights.probes.indices[1] = nullptr;
    weights.probes.values[0]  = nullptr;
}