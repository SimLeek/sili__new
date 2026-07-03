/**
 * @file sparse_matrix.hpp
 * @brief Sparse matrix library with CSR and COO format support.
 */

#ifndef __DELTA_CSR_TYPES_HPP_
#define __DELTA_CSR_TYPES_HPP_

// Split out of sparse_struct.hpp to keep files under ~1k lines (see
// conversation). Core type definitions only: sparse_struct template,
// ValueAccessor<FP4BiPacked>/ValueAccessor<DeltaCSRBiValues<T>>,
// DeltaCSRLayout/DeltaCSRRowCursor/DeltaCSRWeights, SparseLinearWeightsDelta.
// Free functions operating on these types are in delta_csr_memory.hpp and
// delta_csr_ops.hpp. sparse_struct.hpp remains a valid, working include
// (umbrella of all three) for any existing code.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>
#include "fp4quant.hpp"

/**
 * @brief Type trait to check if a type is a std::array.
 * @tparam T The type to check.
 */
template <typename T>
struct is_std_array : std::false_type {};

/**
 * @brief Specialization of is_std_array for std::array types.
 * @tparam T The element type of the array.
 * @tparam N The size of the array.
 */
template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

/**
 * @brief Helper variable template to check if a type is a std::array.
 * @tparam T The type to check.
 */
template <typename T>
constexpr bool is_std_array_v = is_std_array<T>::value;

template <class SIZE_TYPE>
using CSRPointers = std::array<std::shared_ptr<std::vector<SIZE_TYPE>>, 1>;

template <class SIZE_TYPE>
using CSRIndices = std::array<std::shared_ptr<std::vector<SIZE_TYPE>>, 1>;

template <class SIZE_TYPE>
using COOPointers = SIZE_TYPE;  // just store nnz

template <class SIZE_TYPE>
using COOIndices = std::array<std::shared_ptr<std::vector<SIZE_TYPE>>, 2>;

template <class VALUE_TYPE>
using UnaryValues = std::array<std::shared_ptr<std::vector<VALUE_TYPE>>, 1>;

using BiValuesFP4 = FP4BiPacked;

template <class VALUE_TYPE>
using BiValues = std::array<std::shared_ptr<std::vector<VALUE_TYPE>>, 2>;

template <class VALUE_TYPE>
using TriValues = std::array<std::shared_ptr<std::vector<VALUE_TYPE>>, 3>;

template <class VALUE_TYPE>
using QuadValues = std::array<std::shared_ptr<std::vector<VALUE_TYPE>>, 4>;

template <class VALUE_TYPE>
using PentaValues = std::array<std::shared_ptr<std::vector<VALUE_TYPE>>, 5>;

template <typename INDEX_ARRAYS>
constexpr std::size_t num_indices = std::tuple_size<INDEX_ARRAYS>::value;

template <class SIZE_TYPE, class PTRS, class INDICES, class VALUES>
struct sparse_struct {
    PTRS ptrs;               // Pointers sub-template
    INDICES indices;         // Indices sub-template
    VALUES values;           // Values sub-template
    SIZE_TYPE rows;
    SIZE_TYPE cols;
    SIZE_TYPE _reserved_space = 0;

    using size_type = SIZE_TYPE;   // Exporting the type

    static constexpr std::size_t n_index_arrays = num_indices<INDICES>;
    static constexpr std::size_t n_value_arrays = num_indices<VALUES>;
    static constexpr std::size_t n_pointer_arrays = num_indices<PTRS>;

    /**
     * @brief Default constructor, initializes an empty sparse matrix.
     */
    sparse_struct()
        : rows(0), cols(0), _reserved_space(0) {}

    /**
     * @brief Constructor for pre-allocated arrays with reserved space.
     * @param p Pointers sub-template (moved into the structure).
     * @param ind Indices sub-template (moved into the structure).
     * @param val Values sub-template (moved into the structure).
     * @param num_p Number of rows.
     * @param max_idx Number of columns.
     * @param reserved Reserved space for future expansion.
     */
    sparse_struct(PTRS& p, INDICES& ind, VALUES& val, SIZE_TYPE num_p, SIZE_TYPE max_idx, SIZE_TYPE reserved)
        : ptrs(std::move(p)), indices(std::move(ind)), values(std::move(val)),
          rows(num_p), cols(max_idx), _reserved_space(reserved) {}

    /**
     * @brief Constructor for pre-allocated arrays without reserved space.
     * @param p Pointers sub-template (moved into the structure).
     * @param ind Indices sub-template (moved into the structure).
     * @param val Values sub-template (moved into the structure).
     * @param num_p Number of rows.
     * @param max_idx Number of columns.
     */
    sparse_struct(PTRS& p, INDICES& ind, VALUES& val, SIZE_TYPE num_p, SIZE_TYPE max_idx)
        : sparse_struct(std::move(p), std::move(ind), std::move(val), num_p, max_idx, 0) {}

    /**
     * @brief Get the number of non-zero elements in the sparse matrix.
     *
     * If PTRS is an array type (e.g., CSR), returns the last pointer value.
     * If PTRS is a single value (e.g., COO), returns that value directly.
     *
     * @return The number of non-zero elements.
     */
    SIZE_TYPE nnz() const {
        if constexpr (std::is_array_v<decltype(ptrs)> || is_std_array_v<decltype(ptrs)>) { // Check if ptrs is an array type
            return (ptrs[ptrs.size()-1] && !ptrs[ptrs.size()-1]->empty()) ? (*ptrs[ptrs.size()-1])[rows] : 0;
        } else { // ptrs is a single nnz value
            return ptrs;
        }
    }

    /**
     * @brief Clear all values in the sparse structure.
     */
    void clear() {
        if constexpr (is_std_array_v<VALUES>) {
            for (auto& v : values) {
                v.clear();
            }
        } else {
            values.clear();
        }
    }

};

// bi = weight multiplier, importance (for optim). Adagrad would use 2 for optim, using quad.
// Since all these have the same indices, it's much cheaper to store them in the same csr.
template <class SIZE_TYPE>
using CSRSynapses = sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, BiValuesFP4 >;
// easier to use in some algorithms
template <class SIZE_TYPE>
using COOSynapses = sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, BiValuesFP4 >;

template <class SIZE_TYPE, class VALUE_TYPE>
using CSRSynapsesV = sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, BiValues<VALUE_TYPE> >;
// easier to use in some algorithms
template <class SIZE_TYPE, class VALUE_TYPE>
using COOSynapsesV = sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, BiValues<VALUE_TYPE> >;

template <class SIZE_TYPE, class VALUE_TYPE>
using CSRInput = sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE> >;

//easier to use in some algorithms
template <class SIZE_TYPE, class VALUE_TYPE>
using COOSynaptogenesis = sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE> >;

template <class SYNAPSES, class SYNAPTOGENESIS>
struct sparse_weights{
    using size_type = typename SYNAPSES::size_type;

    SYNAPSES connections;
    SYNAPTOGENESIS probes;
    // out_degree[j] = #weights targeting output neuron j.
    // Cached because computing it requires O(nnz) — maintained incrementally by synaptogenesis.
    std::vector<size_type> out_degree;

    // in_degree is free from CSR ptrs — no storage needed.
    inline size_type in_degree(size_type i) const {
        return (*connections.ptrs[0])[i + 1] - (*connections.ptrs[0])[i];
    }
};

template <class SIZE_TYPE, class VALUE_TYPE>
using SparseLinearWeights = sparse_weights<CSRSynapses<SIZE_TYPE>, COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>>;
template <class SIZE_TYPE, class VALUE_TYPE>
using SparseLinearWeightsV = sparse_weights<CSRSynapsesV<SIZE_TYPE, VALUE_TYPE>, COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>>;

// Delta CSR section

/// Maximum bytes to encode an integer as ULEB128.
// fake ULEB128, but in practice we're not going to have more than 2^28 zeroes between items in a single row
template <typename T = uint32_t>
constexpr std::size_t uleb128_max_bytes() {
    return (sizeof(T) * 8 + 6) / 7;
}

/// Encode @p value into @p buf as ULEB128. Returns bytes written.
template <typename T = uint32_t>
inline std::size_t uleb128_encode(T value, uint8_t* buf) {
    std::size_t n = 0;
    do {
        uint8_t byte = static_cast<uint8_t>(value & 0x7Fu);
        value >>= 7;
        if (value) byte |= 0x80u;
        buf[n++] = byte;
    } while (value);
    return n;
}

/// Decode one ULEB128 value from @p buf at byte offset *pos. Advances *pos.
template <typename T = uint32_t>
inline T uleb128_decode(const uint8_t* buf, std::size_t& pos) {
    T result = 0;
    int shift = 0;
    uint8_t byte;
    do {
        byte = buf[pos++];
        result |= static_cast<T>(byte & 0x7Fu) << shift;
        shift += 7;
    } while (byte & 0x80u);
    return result;
}

template <typename V, typename = void>
struct ValueAccessor;

/// Trait to handle FP4BiPacked natively
template <>
struct ValueAccessor<FP4BiPacked> {
    using value_type = float;
    static value_type get_w(const FP4BiPacked& v, std::size_t i) { return v[0][i]; }
    static value_type get_imp(const FP4BiPacked& v, std::size_t i) { return v[1][i]; }
    static void set(FP4BiPacked& v, std::size_t i, value_type w, value_type imp) {
        v[0][i] = w;
        v[1][i] = imp;
    }
    static void reserve(FP4BiPacked& v, std::size_t n) { 
        v.reserve(n); 
    }
    static void resize(FP4BiPacked& v, std::size_t n, value_type val = 0.0f, value_type imp = 0.0f) { 
        v.resize(n, val, imp); 
    }
    static void move(FP4BiPacked& v, std::size_t dest, std::size_t src, std::size_t count) {
        if (count == 0 || !v._data) return;
        std::memmove(v._data->data() + dest, v._data->data() + src, count);
    }

    static std::size_t projected_byte_size(std::size_t n) {
        return n; 
    }
};

/// Fallback standard vector equivalent for floats (e.g. CSRSynapsesV uses)
template <typename T>
struct DeltaCSRBiValues {
    std::vector<T> weights;
    std::vector<T> importance;
};



template <typename T>
struct ValueAccessor<DeltaCSRBiValues<T>> {
    using value_type = T;
    static value_type get_w(const DeltaCSRBiValues<T>& v, std::size_t i) { return v.weights[i]; }
    static value_type get_imp(const DeltaCSRBiValues<T>& v, std::size_t i) { return v.importance[i]; }
    static void set(DeltaCSRBiValues<T>& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = w;
        v.importance[i] = imp;
    }
    static void resize(DeltaCSRBiValues<T>& v, std::size_t n, value_type val = value_type(0), value_type imp = value_type(0)) {
        v.weights.resize(n, val);
        v.importance.resize(n, imp);
    }
    static void move(DeltaCSRBiValues<T>& v, std::size_t dest, std::size_t src, std::size_t count) {
        if (count == 0) return;
        std::memmove(v.weights.data() + dest, v.weights.data() + src, count * sizeof(value_type));
        std::memmove(v.importance.data() + dest, v.importance.data() + src, count * sizeof(value_type));
    }
    static void reserve(DeltaCSRBiValues<T>& v, std::size_t n) {
        v.weights.reserve(n);
        v.importance.reserve(n);
    }

    static std::size_t projected_byte_size(std::size_t n) {
        return n * sizeof(T) * 2; 
    }
};

// ── Layout metadata ───────────────────────────────────────────────────────────

struct DeltaCSRLayout {
    std::size_t rows = 0;
    std::size_t cols = 0;

    std::vector<std::size_t> byte_start;   // size rows+1
    std::vector<std::size_t> byte_end;     // size rows

    std::vector<std::size_t> elem_start;   // size rows+1
    std::vector<std::size_t> elem_end;     // size rows

    std::size_t total_nnz = 0;

    std::size_t row_nnz        (std::size_t r) const { return elem_end[r] - elem_start[r]; }
    std::size_t row_byte_len   (std::size_t r) const { return byte_end[r] - byte_start[r]; }
    std::size_t row_alloc_bytes(std::size_t r) const { return byte_start[r+1] - byte_start[r]; }
    std::size_t row_alloc_elems(std::size_t r) const { return elem_start[r+1] - elem_start[r]; }
    std::size_t row_blank_bytes(std::size_t r) const { return byte_start[r+1] - byte_end[r]; }
    std::size_t row_blank_elems(std::size_t r) const { return elem_start[r+1] - elem_end[r]; }

    std::size_t total_alloc_bytes() const { return byte_start.empty() ? 0 : byte_start.back(); }
    std::size_t total_alloc_elems() const { return elem_start.empty() ? 0 : elem_start.back(); }

    std::size_t total_blank_bytes() const {
        std::size_t b = 0;
        for (std::size_t r = 0; r < rows; ++r) b += row_blank_bytes(r);
        return b;
    }
    std::size_t total_blank_elems() const {
        std::size_t b = 0;
        for (std::size_t r = 0; r < rows; ++r) b += row_blank_elems(r);
        return b;
    }

    std::size_t num_rows() const { return rows; }
};

// ── Forward-only row cursor ───────────────────────────────────────────────────

template <typename COL_TYPE = uint32_t>
struct DeltaCSRRowCursor {
    const uint8_t* buf      = nullptr;
    std::size_t    byte_pos = 0;
    std::size_t    byte_end = 0;
    COL_TYPE       cur_col  = 0;
    std::size_t    n_decoded = 0;

    DeltaCSRRowCursor() = default;

    DeltaCSRRowCursor(const uint8_t* indices_buf, const DeltaCSRLayout& L, std::size_t row)
        : buf(indices_buf)
        , byte_pos(L.byte_start[row])
        , byte_end(L.byte_end[row])
        , cur_col(0)
        , n_decoded(0)
    {}

    bool at_end() const { return byte_pos >= byte_end; }

    COL_TYPE advance() {
        cur_col += uleb128_decode<COL_TYPE>(buf, byte_pos);
        ++n_decoded;
        return cur_col;
    }

    void advance_to(std::size_t target) {
        while (n_decoded <= target) advance();
    }

    COL_TYPE col() const { return cur_col; }
};

// ── DeltaCSRWeights ──────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
struct DeltaCSRWeights {
    DeltaCSRLayout       layout;
    std::vector<uint8_t> indices_buf;
    VALUES_TYPE          values;

    std::size_t          max_indices_bytes = std::numeric_limits<std::size_t>::max();
    std::size_t          max_values_bytes  = std::numeric_limits<std::size_t>::max();

    using size_type  = SIZE_TYPE;
    using col_type   = COL_TYPE;
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;

    bool        empty()    const { return layout.total_nnz == 0; }
    std::size_t nnz()      const { return layout.total_nnz; }
    std::size_t num_rows() const { return layout.rows; }
    std::size_t total_blank_bytes() const { return layout.total_blank_bytes(); }

    DeltaCSRRowCursor<COL_TYPE> row_cursor(std::size_t row) const {
        return DeltaCSRRowCursor<COL_TYPE>(indices_buf.data(), layout, row);
    }

    void set_limits(std::size_t indices_limit_bytes, std::size_t values_limit_bytes) {
        max_indices_bytes = indices_limit_bytes;
        max_values_bytes  = values_limit_bytes;
    }

    void reserve_indices(std::size_t target_bytes) {
        if (target_bytes > max_indices_bytes) {
            throw std::bad_alloc(); 
        }
        indices_buf.reserve(target_bytes);
    }

    void reserve_values(std::size_t target_nnz) {
        std::size_t target_bytes = ValueAccessor<VALUES_TYPE>::projected_byte_size(target_nnz);
        if (target_bytes > max_values_bytes) {
            throw std::bad_alloc();
        }
        ValueAccessor<VALUES_TYPE>::reserve(values, target_nnz);
    }
};

// ── SparseLinearWeightsDelta ─────────────────────────────────────────────────

template <class SIZE_TYPE, class VALUES_TYPE = FP4BiPacked, class COL_TYPE = uint32_t>
struct SparseLinearWeightsDelta {
    using size_type  = SIZE_TYPE;
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;

    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> connections;
    COOSynaptogenesis<SIZE_TYPE, value_type>          probes;
    std::vector<SIZE_TYPE>                            out_degree;

    // Per-layer scale applied to STORED importance to get TRUE units before
    // any importance arithmetic (the Hebbian `1+|imp|` denominator, the
    // per-step decay). Motivation: FP4's smallest representable nonzero
    // magnitude is 0.5, but well-conditioned weight init scales as roughly
    // 1/sqrt(fan_in) -- for fan_in=1000 that's ~0.03, far below FP4's floor.
    // Without a scale, importance would either underflow to zero
    // immediately (losing all regularization signal) or need artificially
    // inflated raw values that don't correspond to anything meaningful.
    // Default 1.0 -- exact behavioral match to having no scale at all, so
    // this is fully backward compatible with every existing test/caller.
    // Read as: true_imp = stored_imp * importance_scale. Write as:
    // stored_imp = true_imp / importance_scale. See rescale_importance()
    // below for changing this mid-training without losing accumulated data.
    value_type importance_scale = value_type(1);

    // Running L1 / L2^2 / max|.| for STORED (quantized) importance and
    // weight values, maintained incrementally (O(1) per synapse touched,
    // not a full-layer rescan) by update_importance_stats()/
    // update_value_stats() below -- see conversation. These track the
    // STORED distribution specifically (not true units), since the
    // question they answer is "is the FP4 representable range being used
    // well," which is about the quantized values as they actually sit in
    // the buffer. double, not value_type(float) -- these are long-running
    // sums across potentially millions of training steps, and float32
    // accumulation drift is a real risk there even though individual
    // synapse values stay float. hoyer_importance()/hoyer_value() compute
    // Hoyer's measure from these in O(1); call recompute_stats() once
    // after constructing a layer via delta_csr_from_absolute() (or any
    // other path that writes values without going through
    // update_*_stats()), since these start at zero otherwise.
    //
    // LIMITATION, stated plainly: max_abs is a MONOTONIC upper bound, not
    // a live exact current max -- if the element currently holding the max
    // shrinks, max_abs cannot decrease without rescanning (unlike L1/L2^2,
    // which update exactly via O(1) arithmetic). Useful as "has this layer
    // ever touched the ceiling," not as "what is the max right now" --
    // call recompute_stats() to get an exact value if that distinction
    // matters for a particular decision.
    double     importance_l1      = 0.0;
    double     importance_l2_sq   = 0.0;
    value_type importance_max_abs = value_type(0);
    double     value_l1           = 0.0;
    double     value_l2_sq        = 0.0;
    value_type value_max_abs      = value_type(0);

    // Decay applied to max_abs on every update, BEFORE comparing against
    // the new value -- new_max = max(old_max * decay, |new_val|). Default
    // 1.0 (no decay, exact backward compat -- pure monotonic bound as
    // before). A decay slightly below 1.0 (e.g. 0.9999) lets max_abs drift
    // downward over time when the element that set it has since shrunk,
    // rather than staying stuck at a stale peak forever -- an approximate,
    // self-correcting live max rather than an exact one, which is judged
    // sufficient here (see conversation). Python-settable/viewable, same
    // spirit as importance_scale.
    value_type max_abs_decay = value_type(1);

    inline value_type hoyer_importance() const {
        return _hoyer_from_stats(importance_l1, importance_l2_sq);
    }
    inline value_type hoyer_value() const {
        return _hoyer_from_stats(value_l1, value_l2_sq);
    }

    // Call after any write to a synapse's stored importance -- old_val/
    // new_val must be the STORED (post-quantization) values, matching what
    // these stats track, not true units.
    //
    // THREAD SAFETY: these mutate shared state (importance_l1 etc.) with
    // no locking -- safe to call from single-threaded code, or serially
    // after a parallel region, but NOT safe to call concurrently from
    // multiple OpenMP threads (a real bug found and fixed here -- see
    // conversation: this was originally called directly inside
    // #pragma omp parallel loops in all four kernels, racing on these
    // exact fields, undetected because every test used num_cpus=1). For
    // parallel kernels, each thread should accumulate locally (sum of
    // |new|, sum of |old|, sum of new^2, sum of old^2, local max) and call
    // update_importance_stats_aggregate()/update_value_stats_aggregate()
    // ONCE per thread after the parallel region, not this method from
    // inside one.
    inline void update_importance_stats(value_type old_val, value_type new_val) {
        importance_l1      += std::abs(static_cast<double>(new_val)) - std::abs(static_cast<double>(old_val));
        importance_l2_sq   += static_cast<double>(new_val) * new_val - static_cast<double>(old_val) * old_val;
        importance_max_abs  = std::max(importance_max_abs * max_abs_decay, std::abs(new_val));
    }
    inline void update_value_stats(value_type old_val, value_type new_val) {
        value_l1      += std::abs(static_cast<double>(new_val)) - std::abs(static_cast<double>(old_val));
        value_l2_sq   += static_cast<double>(new_val) * new_val - static_cast<double>(old_val) * old_val;
        value_max_abs  = std::max(value_max_abs * max_abs_decay, std::abs(new_val));
    }

    // Thread-safe: apply ONE thread's worth of pre-summed partial totals.
    // Call once per thread after a parallel region (or serially, from
    // single-threaded code -- equivalent to calling update_*_stats() for
    // every synapse that thread touched, batched into 4 sums instead of
    // per-synapse calls). sum_abs_new/sum_abs_old = that thread's running
    // sum of |new_val|/|old_val| across every synapse it touched;
    // sum_sq_new/sum_sq_old = the same for squares; local_max_new = the
    // largest |new_val| that thread saw (NOT decayed -- decay is applied
    // once here, matching update_*_stats' per-call semantics as closely as
    // a batched call can).
    inline void update_importance_stats_aggregate(
        double sum_abs_new, double sum_abs_old,
        double sum_sq_new,  double sum_sq_old,
        value_type local_max_new)
    {
        importance_l1      += sum_abs_new - sum_abs_old;
        importance_l2_sq   += sum_sq_new  - sum_sq_old;
        importance_max_abs  = std::max(importance_max_abs * max_abs_decay, local_max_new);
    }
    inline void update_value_stats_aggregate(
        double sum_abs_new, double sum_abs_old,
        double sum_sq_new,  double sum_sq_old,
        value_type local_max_new)
    {
        value_l1      += sum_abs_new - sum_abs_old;
        value_l2_sq   += sum_sq_new  - sum_sq_old;
        value_max_abs  = std::max(value_max_abs * max_abs_decay, local_max_new);
    }

    // Recompute all six stats from scratch -- O(nnz), call once after
    // constructing a layer via delta_csr_from_absolute() or any path that
    // writes values without going through update_*_stats(), or whenever an
    // exact (not monotonic-bound) max_abs is needed.
    inline void recompute_stats() {
        importance_l1 = importance_l2_sq = 0.0; importance_max_abs = value_type(0);
        value_l1      = value_l2_sq      = 0.0; value_max_abs      = value_type(0);
        auto& L = connections.layout;
        for (std::size_t r = 0; r < L.rows; ++r) {
            const std::size_t n = L.row_nnz(r);
            for (std::size_t e = 0; e < n; ++e) {
                const std::size_t vb = L.elem_start[r] + e;
                update_value_stats(value_type(0),
                    ValueAccessor<VALUES_TYPE>::get_w(connections.values, vb));
                update_importance_stats(value_type(0),
                    ValueAccessor<VALUES_TYPE>::get_imp(connections.values, vb));
            }
        }
    }

    inline SIZE_TYPE in_degree(SIZE_TYPE i) const {
        return static_cast<SIZE_TYPE>(connections.layout.row_nnz(i));
    }

private:
    inline value_type _hoyer_from_stats(double l1, double l2_sq) const {
        const std::size_t n = connections.nnz();
        if (n <= 1) return value_type(0);
        const double l2 = std::sqrt(l2_sq);
        if (l2 <= 0.0) return value_type(1);   // all-zero -> maximally "sparse" by convention
        const double sqrt_n = std::sqrt(static_cast<double>(n));
        return static_cast<value_type>((sqrt_n - l1 / l2) / (sqrt_n - 1.0));
    }

public:
    // Change importance_scale mid-training without losing accumulated
    // importance: re-reads every stored synapse's importance at the OLD
    // scale into true units, re-encodes at the NEW scale, then updates
    // importance_scale itself. Without this, just assigning a new scale
    // directly would silently reinterpret all existing stored values as if
    // they'd always been at the new scale -- corrupting every synapse's
    // importance in one step, not just changing how future arithmetic
    // treats it.
    inline void rescale_importance(value_type new_scale) {
        if (new_scale == importance_scale) return;
        auto& dc = connections;
        auto& L  = dc.layout;
        for (std::size_t r = 0; r < L.rows; ++r) {
            const std::size_t n = L.row_nnz(r);
            for (std::size_t e = 0; e < n; ++e) {
                const std::size_t vb = L.elem_start[r] + e;
                const value_type w        = ValueAccessor<VALUES_TYPE>::get_w  (dc.values, vb);
                const value_type stored_i = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                const value_type true_i   = stored_i * importance_scale;
                ValueAccessor<VALUES_TYPE>::set(dc.values, vb, w, true_i / new_scale);
            }
        }
        importance_scale = new_scale;
    }

    inline void set_limits(std::size_t indices_limit_bytes, std::size_t values_limit_bytes) {
        connections.set_limits(indices_limit_bytes, values_limit_bytes);
    }

    inline void reserve_indices(std::size_t target_bytes) {
        connections.reserve_indices(target_bytes);
    }

    inline void reserve_values(std::size_t target_nnz) {
        connections.reserve_values(target_nnz);
    }
};

#endif
