#include "csr.hpp"
#include "parallel.hpp"
#include <cstddef>
#include <vector>

// ── DISLDO: Dense Input, Sparse Linear, Dense Output ─────────────────────────

// ── forward ───────────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUE_TYPE>
void disldo_forward(
    const VALUE_TYPE* input,          // [batch, in_cols] row-major
    const SIZE_TYPE   batch,
    const SIZE_TYPE   in_cols,
    const SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    VALUE_TYPE*       output,         // [batch, out_cols] zeroed by caller
    bool              train,
    VALUE_TYPE        solidify = 0.01f,
    const int         num_cpus = 4)
{
    if (connections_empty(weights.connections)) return;

    const SIZE_TYPE out_cols     = weights.connections.cols;
    const auto&     conn_ptrs    = *weights.connections.ptrs[0];
    const auto&     conn_indices = *weights.connections.indices[0];
    const auto&     conn_val0    = *weights.connections.values[0];
    auto&           conn_val2    = *weights.connections.values[2];

    const SIZE_TYPE out_size = batch * out_cols;

    // ── Phase 1: privatized scatter ──────────────────────────────────────────
    // Each thread writes to its own slice — zero contention.
    std::vector<VALUE_TYPE> all_out((size_t)num_cpus * out_size, VALUE_TYPE(0));

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid = omp_get_thread_num();
        VALUE_TYPE* my_out = all_out.data() + (size_t)tid * out_size;

        #pragma omp for schedule(static)
        for (SIZE_TYPE ic = 0; ic < in_cols; ++ic) {
            const SIZE_TYPE wp_start = conn_ptrs[ic];
            const SIZE_TYPE wp_end   = conn_ptrs[ic + 1];
            if (wp_start == wp_end) continue;

            for (SIZE_TYPE b = 0; b < batch; ++b) {
                const VALUE_TYPE iv = input[b * in_cols + ic];
                for (SIZE_TYPE wp = wp_start; wp < wp_end; ++wp) {
                    const VALUE_TYPE w  = conn_val0[wp];
                    const SIZE_TYPE  oc = conn_indices[wp];
                    const VALUE_TYPE c  = w * iv;
                    my_out[b * out_cols + oc] += c;
                    if (train) conn_val2[wp] += c * solidify;
                }
            }
        }
    }   // implicit barrier — all scatter done

    // ── Phase 2: parallel tree reduction — no critical, no atomics ───────────
    // Active pairs are disjoint per pass so no synchronization within a pass.
    // The implicit barrier between omp parallel regions is the only sync.
    for (int stride = 1; stride < num_cpus; stride <<= 1) {
        #pragma omp parallel for num_threads(num_cpus) schedule(static)
        for (int tid = 0; tid < num_cpus; tid += stride << 1) {
            const int src = tid + stride;
            if (src >= num_cpus) continue;
            VALUE_TYPE*       dst = all_out.data() + (size_t)tid * out_size;
            const VALUE_TYPE* src_buf = all_out.data() + (size_t)src * out_size;
            for (SIZE_TYPE i = 0; i < out_size; ++i)
                dst[i] += src_buf[i];
        }
    }

    // Thread 0's slice holds the full result.
    const VALUE_TYPE* result = all_out.data();
    for (SIZE_TYPE i = 0; i < out_size; ++i)
        output[i] += result[i];
}

// ── backward ─────────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUE_TYPE>
void disldo_backward(
    const VALUE_TYPE* input,          // [batch, in_cols]
    const SIZE_TYPE   batch,
    const SIZE_TYPE   in_cols,
    const VALUE_TYPE* output_grad,    // [batch, out_cols]
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>& weights,
    VALUE_TYPE*       input_grad,     // [batch, in_cols] zeroed by caller
    VALUE_TYPE*       neuron_input_accum,   // [in_cols]
    VALUE_TYPE*       neuron_grad_accum,    // [out_cols]
    const int         num_cpus = 4)
{
    if (connections_empty(weights.connections)) return;

    const SIZE_TYPE out_cols     = weights.connections.cols;
    const auto&     conn_ptrs    = *weights.connections.ptrs[0];
    const auto&     conn_indices = *weights.connections.indices[0];
    const auto&     conn_val0    = *weights.connections.values[0];
    auto&           conn_val1    = *weights.connections.values[1];

    for (SIZE_TYPE b = 0; b < batch; ++b)
        for (SIZE_TYPE ic = 0; ic < in_cols; ++ic)
            neuron_input_accum[ic] += std::abs(input[b * in_cols + ic]);

    const SIZE_TYPE ig_size = batch * in_cols;
    std::vector<VALUE_TYPE> all_ig((size_t)num_cpus * ig_size,   VALUE_TYPE(0));
    std::vector<VALUE_TYPE> all_ga((size_t)num_cpus * out_cols,  VALUE_TYPE(0));

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid = omp_get_thread_num();
        VALUE_TYPE* my_ig = all_ig.data() + (size_t)tid * ig_size;
        VALUE_TYPE* my_ga = all_ga.data() + (size_t)tid * out_cols;

        #pragma omp for schedule(static)
        for (SIZE_TYPE ic = 0; ic < in_cols; ++ic) {
            const SIZE_TYPE wp_start = conn_ptrs[ic];
            const SIZE_TYPE wp_end   = conn_ptrs[ic + 1];
            if (wp_start == wp_end) continue;

            for (SIZE_TYPE b = 0; b < batch; ++b) {
                const VALUE_TYPE iv = input[b * in_cols + ic];
                for (SIZE_TYPE wp = wp_start; wp < wp_end; ++wp) {
                    const SIZE_TYPE  oc  = conn_indices[wp];
                    const VALUE_TYPE og  = output_grad[b * out_cols + oc];
                    conn_val1[wp]          += og * iv;
                    my_ig[b * in_cols + ic] += conn_val0[wp] * og;
                    my_ga[oc]               += std::abs(og);
                }
            }
        }
    }   // implicit barrier — all scatter done

    for (int stride = 1; stride < num_cpus; stride <<= 1) {
        #pragma omp parallel for num_threads(num_cpus) schedule(static)
        for (int tid = 0; tid < num_cpus; tid += stride << 1) {
            const int src = tid + stride;
            if (src >= num_cpus) continue;
            VALUE_TYPE*       dig = all_ig.data() + (size_t)tid * ig_size;
            const VALUE_TYPE* sig = all_ig.data() + (size_t)src * ig_size;
            for (SIZE_TYPE i = 0; i < ig_size; ++i) dig[i] += sig[i];

            VALUE_TYPE*       dga = all_ga.data() + (size_t)tid * out_cols;
            const VALUE_TYPE* sga = all_ga.data() + (size_t)src * out_cols;
            for (SIZE_TYPE i = 0; i < out_cols; ++i) dga[i] += sga[i];
        }
    }

    const VALUE_TYPE* rig = all_ig.data();
    for (SIZE_TYPE i = 0; i < ig_size; ++i)  input_grad[i]       += rig[i];
    const VALUE_TYPE* rga = all_ga.data();
    for (SIZE_TYPE i = 0; i < out_cols; ++i) neuron_grad_accum[i] += rga[i];
}