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
// sisldo_ops.hpp. sparse_struct.hpp remains a valid, working include
// (umbrella of all three) for any existing code.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>
#include "fp4quant.hpp"
#include "fp8quant.hpp"
// block4.hpp is included further down (right before SparseLinearWeightsDelta,
// the only thing here that needs Block4Store's full definition) -- Block4Store
// itself now reuses DeltaCSRLayout/DeltaCSRRowCursor (defined below), so
// block4.hpp can't be included this early any more. See conversation
// (ULEB128 block4 tile indexing).

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
    /// Gradient-driven update only -- see fp4_quantize_stochastic()'s
    /// docstring (fp4quant.hpp) for why this is separate from set().
    static void set_stochastic(FP4BiPacked& v, std::size_t i, value_type w, value_type imp) {
        v.set_stochastic(i, w, imp);
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

/// Trait to handle FP8BiValues -- one full byte per value (weight,
/// importance separately), OCP MX E4M3 codec (fp8quant.hpp). Same shape
/// as ValueAccessor<DeltaCSRBiValues<T>> below, only value_type stays
/// the DECODED float (matching ValueAccessor<FP4BiPacked>'s convention,
/// not DeltaCSRBiValues<T>'s raw-passthrough one) since the byte array
/// itself holds an encoded code, not a usable float directly.
template <>
struct ValueAccessor<FP8BiValues> {
    using value_type = float;
    static value_type get_w(const FP8BiValues& v, std::size_t i) { return fp8_decode_bits(v.weights[i]); }
    static value_type get_imp(const FP8BiValues& v, std::size_t i) { return fp8_decode_bits(v.importance[i]); }
    static void set(FP8BiValues& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = fp8_quantize(w);
        v.importance[i] = fp8_quantize(imp);
    }
    /// Gradient-driven update only -- see fp8_quantize_stochastic()'s
    /// own docstring (fp8quant.hpp) for why this is separate from set().
    static void set_stochastic(FP8BiValues& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = fp8_quantize_stochastic(w);
        v.importance[i] = fp8_quantize_stochastic(imp);
    }
    static void resize(FP8BiValues& v, std::size_t n, value_type val = 0.0f, value_type imp = 0.0f) {
        v.weights.resize(n, fp8_quantize(val));
        v.importance.resize(n, fp8_quantize(imp));
    }
    static void move(FP8BiValues& v, std::size_t dest, std::size_t src, std::size_t count) {
        if (count == 0) return;
        std::memmove(v.weights.data() + dest, v.weights.data() + src, count);
        std::memmove(v.importance.data() + dest, v.importance.data() + src, count);
    }
    static void reserve(FP8BiValues& v, std::size_t n) {
        v.weights.reserve(n);
        v.importance.reserve(n);
    }

    static std::size_t projected_byte_size(std::size_t n) {
        return n * 2;  // 1 byte weight + 1 byte importance per element
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
    /// No quantization happens for this (float32 fallback) storage, so
    /// stochastic rounding is meaningless here -- passthrough to set()
    /// so callers that always use set_stochastic() for gradient-driven
    /// updates work identically against either VALUES_TYPE.
    static void set_stochastic(DeltaCSRBiValues<T>& v, std::size_t i, value_type w, value_type imp) {
        set(v, i, w, imp);
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

// ── Scale-update policies ─────────────────────────────────────────────────────
//
// Swappable in-place optimizer for value_scale/output_scale (disldo_backward's
// scattered path, linear_disldo.hpp). Template parameter, not a runtime flag --
// each policy is a stateless struct with one static `update()`, so choosing a
// policy costs nothing at runtime (inlined, no branch, no vtable) and callers
// select it by template argument, matching this codebase's existing VALUES_TYPE
// convention (SparseLinearLayer vs SparseLinearLayer8 vs DISLDOLayerV are
// already separate hand-instantiated callers of one shared templated
// implementation, not one class runtime-branching on a stored enum).
//
// Built to compare real update rules for value_scale/output_scale against a
// toy Python fake-quantize simulation's closed-form full-layer refit
// (sili_peridot's QuantizedDISLDOLayer32/rank1_fake_quantize) that reaches
// near-perfect accuracy on an out-of-context recall task while the real
// RMSprop-based scale update collapses -- see sili_peridot/JOURNAL.md's
// 2026-08-09 tile-recurrence entries for the full investigation.

template <typename VALUE_TYPE>
struct RMSpropScalePolicy {
    // Extracted verbatim from disldo_backward's existing inline formula --
    // must stay bit-identical to today's behavior on finite inputs (this
    // is the default, used by every existing caller) -- PLUS a NaN/Inf
    // guard. scale/scale_state update INLINE every step and are never
    // touched by any external gradient clip (unlike an optimizer-managed
    // parameter) -- a single overflowed g_agg (e.g. dense connectivity's
    // much larger fan-in summing far more per-column gradient terms than
    // sparse ever does, narrowed from a double accumulator to VALUE_TYPE
    // with no range check upstream) turns scale_state permanently Inf
    // (beta2*Inf never decays back down), then sqrt(Inf)+eps -> Inf,
    // g_agg/Inf -> Inf or Inf/Inf -> NaN, and once scale is NaN every
    // future beta2*NaN+... stays NaN forever. Confirmed via direct
    // diagnostic in sili_peridot (dense connectivity, JOURNAL.md
    // 2026-08-10): output_scale went NaN in lockstep with the whole
    // model while the raw stored weight code itself stayed correctly
    // bounded. Skip the update entirely on a non-finite input/result
    // rather than letting it corrupt scale/scale_state -- makes NaN
    // structurally unreachable through this path, not just unlikely.
    static void update(VALUE_TYPE& scale, VALUE_TYPE& scale_state,
                        VALUE_TYPE g_agg, VALUE_TYPE eff_lr,
                        VALUE_TYPE beta2, VALUE_TYPE eps) {
        if (!std::isfinite(g_agg)) return;
        const VALUE_TYPE new_state = beta2 * scale_state + (VALUE_TYPE(1) - beta2) * g_agg * g_agg;
        if (!std::isfinite(new_state)) return;
        const VALUE_TYPE new_scale = scale - eff_lr * g_agg / (std::sqrt(new_state) + eps);
        if (!std::isfinite(new_scale)) return;
        scale_state = new_state;
        scale = new_scale;
    }
};

template <typename VALUE_TYPE>
struct AdaMaxScalePolicy {
    // Matches AdaMax's own decayed running-max second-moment tracker
    // (Kingma & Ba 2015, Adam paper sec 7) applied to value_scale/
    // output_scale instead of a gradient: scale_state tracks
    // max(beta2*scale_state, |g_agg|) -- growth is INSTANT (never lets
    // scale_state fall below the current gradient magnitude, matching
    // the max-cover safety property a real fixed-point scale needs:
    // never let a stored value exceed what its levels can represent),
    // shrink is gradual (only the decay term reduces it, when nothing
    // larger has been seen recently). No sqrt needed -- scale_state is
    // already in the same units as |g_agg|, unlike RMSprop's g^2 EMA.
    // Same NaN/Inf guard as RMSpropScalePolicy::update above, and for the
    // same reason -- scale_state's std::max never lets a stray Inf decay
    // back down either, so an overflowed g_agg is just as permanently
    // corrupting here without this check.
    static void update(VALUE_TYPE& scale, VALUE_TYPE& scale_state,
                        VALUE_TYPE g_agg, VALUE_TYPE eff_lr,
                        VALUE_TYPE beta2, VALUE_TYPE eps) {
        if (!std::isfinite(g_agg)) return;
        const VALUE_TYPE new_state = std::max(beta2 * scale_state, std::abs(g_agg));
        if (!std::isfinite(new_state)) return;
        const VALUE_TYPE new_scale = scale - eff_lr * g_agg / (new_state + eps);
        if (!std::isfinite(new_scale)) return;
        scale_state = new_state;
        scale = new_scale;
    }
};

// No-op: scale/scale_state are never touched, so scale stays at whatever
// it was initialized/set to (value_type(1) by default -- see
// SparseLinearWeightsDelta's value_scale.resize(n, value_type(1)) --
// i.e. value_scale[row]*output_scale[col] is permanently the identity
// multiply, true_w == stored_w). Direct real-hardware test of the
// "zero trained scale" hypothesis (sili_peridot's fixed_digit_residual_
// quantize/TrueMultiDigitLayer work) instead of just fixing staleness --
// use with DeferredScaleWrite=true or false, doesn't matter here since
// there's nothing to defer (scale never changes either way).
template <typename VALUE_TYPE>
struct NoScalePolicy {
    static void update(VALUE_TYPE& /*scale*/, VALUE_TYPE& /*scale_state*/,
                        VALUE_TYPE /*g_agg*/, VALUE_TYPE /*eff_lr*/,
                        VALUE_TYPE /*beta2*/, VALUE_TYPE /*eps*/) {
        // Intentionally does nothing.
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

// Needs DeltaCSRLayout/DeltaCSRRowCursor just defined above -- Block4Store
// reuses both directly (both are already value-type-agnostic, no changes
// needed to either). See conversation (ULEB128 block4 tile indexing).
#include "block4.hpp"

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

// Selects which Block4Store variant SparseLinearWeightsDelta.block4 (below)
// uses, keyed on VALUES_TYPE -- default (Block4Store, the FP4-1-byte-per
// -slot dense-tile layout) matches every existing instantiation's exact
// prior behavior unchanged; only FP8BiValues gets Block4Store8 (2 bytes/
// slot, fp8quant.hpp's E4M3 codec -- see block4.hpp's own comment on why
// this needed a fully separate store, not a template-parameter retrofit
// of Block4Store itself). A pure additive trait, not a modification to
// either existing type.
template <typename VALUES_TYPE>
struct Block4StoreFor { using type = Block4Store; };
template <>
struct Block4StoreFor<FP8BiValues> { using type = Block4Store8; };

template <class SIZE_TYPE, class VALUES_TYPE = FP4BiPacked, class COL_TYPE = uint32_t>
struct SparseLinearWeightsDelta {
    using size_type  = SIZE_TYPE;
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    using block4_type = typename Block4StoreFor<VALUES_TYPE>::type;

    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> connections;
    COOSynaptogenesis<SIZE_TYPE, value_type>          probes;
    std::vector<SIZE_TYPE>                            out_degree;

    // Locally-dense companion to `connections` -- see block4.hpp. Shares
    // THIS struct's own value_scale/importance_scale below (not a separate
    // scale), so moving a synapse between `connections` and `block4` is a
    // lossless byte copy, not a requantization. Promotion/demotion logic
    // lives in delta_csr_memory.hpp (needs delta_csr_row_insert_col/
    // _remove_col, which would be a circular include from here).
    //
    // Type selected via Block4StoreFor<VALUES_TYPE> (above) -- Block4Store
    // for every existing VALUES_TYPE (FP4BiPacked, DeltaCSRBiValues<T>,
    // unchanged), Block4Store8 for FP8BiValues.
    block4_type block4;

    // Per-ROW scale applied to STORED importance to get TRUE units before
    // any importance arithmetic (the activity-correlation `1+|imp|` denominator, the
    // per-step decay). Motivation: FP4's smallest representable nonzero
    // magnitude is 0.5, but well-conditioned weight init scales as roughly
    // 1/sqrt(fan_in) -- for fan_in=1000 that's ~0.03, far below FP4's floor.
    // Without a scale, importance would either underflow to zero
    // immediately (losing all regularization signal) or need artificially
    // inflated raw values that don't correspond to anything meaningful.
    //
    // Per-row, not per-layer: different rows can have very different
    // natural importance-trace magnitude within the SAME layer (different
    // fan-in/connection counts, especially once synaptogenesis has been
    // running a while and row_nnz has diverged across rows) -- a single
    // layer-wide scale can't serve a sparsely-connected row and a
    // densely-connected row equally well at the same time.
    //
    // Stored as a lazily-sized vector, not pre-sized at construction (this
    // struct doesn't know L.rows until .connections is populated) --
    // get_importance_scale()/set_importance_scale() below handle sizing
    // safely, defaulting any not-yet-touched row to 1.0 (exact match to
    // having no scale at all, so this is fully backward compatible with
    // every existing test/caller that never touches per-row scale at all).
    //
    // Read as: true_imp = stored_imp * scale. Write as: stored_imp =
    // true_imp / scale. See rescale_importance()/rescale_importance_row()
    // below for changing this mid-training without losing accumulated data.
    std::vector<value_type> importance_scale;

    inline value_type get_importance_scale(std::size_t row) const {
        return row < importance_scale.size() ? importance_scale[row] : value_type(1);
    }
    inline void set_importance_scale_raw(std::size_t row, value_type v) {
        if (row >= importance_scale.size()) importance_scale.resize(row + 1, value_type(1));
        importance_scale[row] = v;
    }

    // Per-COLUMN counterpart to importance_scale, same relationship as
    // output_scale is to value_scale: true_imp = stored_imp *
    // importance_scale[row] * output_importance_scale[col]. A synapse's
    // stored importance and stored weight live at the same (row, col)
    // position, so if the weight's own representability scale needs a
    // column term (it does -- see output_scale), the importance's does
    // too, for the same reason (per-output activity magnitude can vary
    // as much as per-output weight magnitude). Default 1.0, same
    // lazy-sizing convention -- unused by from_descriptor today (no
    // caller sets it, so this is a pure no-op for every existing path).
    std::vector<value_type> output_importance_scale;
    inline value_type get_output_importance_scale(std::size_t col) const {
        return col < output_importance_scale.size() ? output_importance_scale[col] : value_type(1);
    }
    inline void set_output_importance_scale_raw(std::size_t col, value_type v) {
        if (col >= output_importance_scale.size()) output_importance_scale.resize(col + 1, value_type(1));
        output_importance_scale[col] = v;
    }

    // Compile-time cap on scale_rank, used by block4's SIMD backward path
    // (linear_disldo.hpp's process_tile) to size fixed stack arrays of
    // per-rank-component Block4Vec accumulators instead of heap-allocating
    // a std::vector on every tile-row visited (that loop runs extremely
    // often -- once per (block-row, block-col, li) triple touched by
    // backward -- so a heap alloc there would be a real, not hypothetical,
    // regression). 4 is generous headroom over the rank=2 this was built
    // for; raise it if a real use case needs more, but keep it small --
    // it directly sizes stack buffers in the hot path.
    static constexpr std::size_t SCALE_RANK_MAX = 4;

    // RANK of the value_scale/output_scale factorization. true_w =
    // quant * S[row,col], where S used to be a plain rank-1 outer product
    // value_scale[row]*output_scale[col] -- now generalized to
    // S[row,col] = sum_{k<scale_rank} value_scale_k(row,k) *
    // output_scale_k(col,k), a sum of `scale_rank` outer products.
    // Runtime-parameterized (not a C++ template per rank) so no new
    // class/pybind binding is needed per rank value tried -- rank=1 is
    // the default and reproduces the exact original behavior; storage
    // for k>=1 defaults to 0.0 (see get_value_scale_k), so an
    // unconfigured extra component contributes nothing until trained,
    // and every existing call site (single-component get_value_scale/
    // get_output_scale, meaning component 0) keeps working unmodified.
    //
    // WHY rank>1 at all: a single shared row-scalar (rank-1) can't serve
    // a row whose columns have genuinely conflicting persistent gradient
    // demand (column A wants the scale positive, column B wants it
    // negative) -- the row-aggregate gradient driving value_scale sums
    // g*quant across every column in the row, and opposite-signed real
    // signal cancels there exactly like noise would, even though neither
    // column's own signal is actually noisy. A second (u2,v2) component
    // gives a second, independently-signed channel to absorb the
    // opposite-signed column instead of cancelling against the first.
    //
    // SCOPE NOTE: disldo_forward, and disldo_backward's scattered-CSR path
    // AND block4's FP4 path (process_tile in linear_disldo.hpp) are all
    // rank-aware, using weights.get_scale(row,col). The SIMD fast path
    // keeps its 4-wide column vectorization regardless of rank by looping
    // the (small, SCALE_RANK_MAX-capped) rank dimension outside the lane
    // dimension with real Block4Vec accumulators per component, rather
    // than falling back to scalar for rank>1.
    //
    // Still rank-1-only (component 0 of a rank>1 layer): block4's FP8
    // branch (a separate, near-duplicate code path within the same
    // process_tile lambda -- the real toy/test model uses FP4, not FP8,
    // so this wasn't hit yet) and every DeferredScaleWrite class (e.g.
    // SparseLinearLayerResync) on both scattered and block4 paths (see
    // disldo_backward's own DeferredScaleWrite branch comment for why
    // rank>1 doesn't apply there without more work: eager multiply-by-
    // not-yet-finalized-scale would reintroduce the staleness
    // DeferredScaleWrite exists to avoid). If a rank>1 layer ever uses
    // FP8 block4 or a DeferredScaleWrite class, its effective scale
    // silently drops every component beyond 0 there -- tracked as a
    // follow-up, not fixed here.
    std::size_t scale_rank = 1;

    // Same per-row design, for STORED weight values instead of importance.
    // Same motivation, same lazy-sizing/default-1.0 pattern, same
    // read/write convention (true_w = stored_w * scale). Storage is now
    // row-major per-component: value_scale[row*scale_rank + k].
    std::vector<value_type> value_scale;

    // Per-component accessor. Component 0 default-value 1.0 matches the
    // original single-component convention (unset row => pass-through
    // scale of 1); component k>=1 defaults to 0.0 so an untrained extra
    // rank component is a pure no-op (S[row,col] reduces to exactly the
    // rank-1 formula until that component is actually trained).
    inline value_type get_value_scale_k(std::size_t row, std::size_t k) const {
        const std::size_t idx = row * scale_rank + k;
        if (idx < value_scale.size()) return value_scale[idx];
        return k == 0 ? value_type(1) : value_type(0);
    }
    inline void set_value_scale_raw_k(std::size_t row, std::size_t k, value_type v) {
        const std::size_t idx = row * scale_rank + k;
        if (idx >= value_scale.size())
            value_scale.resize(idx + 1, value_type(1));  // grows lazily; see backfill note below
        value_scale[idx] = v;
    }
    // Backward-compat single-component accessors -- component 0 only,
    // exact original meaning/behavior at scale_rank==1.
    inline value_type get_value_scale(std::size_t row) const {
        return get_value_scale_k(row, 0);
    }
    inline void set_value_scale_raw(std::size_t row, value_type v) {
        set_value_scale_raw_k(row, 0, v);
    }

    // value_scale/output_scale are themselves gradient-updated parameters
    // (disldo_backward), so -- like every per-synapse weight -- each gets
    // its own importance value damping its update step
    // (new = old - lr*grad / (1 + |importance|)). Default 0, same
    // convention as per-synapse importance. Same per-component,
    // row-major layout as value_scale.
    std::vector<value_type> value_scale_importance;
    inline value_type get_value_scale_importance_k(std::size_t row, std::size_t k) const {
        const std::size_t idx = row * scale_rank + k;
        return idx < value_scale_importance.size() ? value_scale_importance[idx] : value_type(0);
    }
    inline value_type get_value_scale_importance(std::size_t row) const {
        return get_value_scale_importance_k(row, 0);
    }

    // Adam-style FIRST moment (signed EMA of g_agg) for value_scale,
    // companion to value_scale_importance's SECOND moment (EMA of
    // g_agg^2, unsigned). Used only by the dead-row (nnz_row==0) path in
    // disldo_backward -- the existing live-synapse value_scale update
    // uses the instantaneous g_agg directly (RMSprop-style, unchanged).
    // A dead row has no per-synapse importance (no synapse exists to
    // hold one), so it needs its own signed accumulator to know which
    // direction to nudge value_scale; value_scale_importance doubles as
    // its second moment too (provably untouched by anything else while
    // nnz_row==0, since the live-synapse loop skips the row entirely).
    // Linear in g_agg (not g_agg^2) so E[update]=0 under zero-mean noise
    // regardless of variance -- squaring the pretend/reactive direction
    // (an earlier, rejected design) would have made an occasional large
    // -magnitude gradient dominate the accumulated direction even under
    // otherwise-cancelling noise. Same per-component, row-major layout.
    std::vector<value_type> value_scale_momentum;
    inline value_type get_value_scale_momentum_k(std::size_t row, std::size_t k) const {
        const std::size_t idx = row * scale_rank + k;
        return idx < value_scale_momentum.size() ? value_scale_momentum[idx] : value_type(0);
    }
    inline value_type get_value_scale_momentum(std::size_t row) const {
        return get_value_scale_momentum_k(row, 0);
    }

    // Per-COLUMN counterpart to value_scale: true_w = stored_w *
    // S[row,col], S = sum_k value_scale_k(row,k)*output_scale_k(col,k).
    // Same lazy-sizing/default convention (component 0 -> 1.0, k>=1 ->
    // 0.0), same row-major-per-component layout: output_scale[col*
    // scale_rank+k]. Gradient-updated by disldo_backward like value_scale
    // is, but ONLY once a caller has explicitly called
    // set_output_scale_raw{,_k} at least once -- output_scale_is_trainable
    // tracks that (not output_scale.empty(), which disldo_backward's own
    // internal resize would otherwise flip after the first call
    // regardless of intent).
    std::vector<value_type> output_scale;
    std::vector<value_type> output_scale_importance;
    bool output_scale_is_trainable = false;

    inline value_type get_output_scale_k(std::size_t col, std::size_t k) const {
        const std::size_t idx = col * scale_rank + k;
        if (idx < output_scale.size()) return output_scale[idx];
        return k == 0 ? value_type(1) : value_type(0);
    }
    inline value_type get_output_scale(std::size_t col) const {
        return get_output_scale_k(col, 0);
    }
    inline value_type get_output_scale_importance_k(std::size_t col, std::size_t k) const {
        const std::size_t idx = col * scale_rank + k;
        return idx < output_scale_importance.size() ? output_scale_importance[idx] : value_type(0);
    }
    inline value_type get_output_scale_importance(std::size_t col) const {
        return get_output_scale_importance_k(col, 0);
    }
    inline void set_output_scale_raw_k(std::size_t col, std::size_t k, value_type v) {
        const std::size_t idx = col * scale_rank + k;
        if (idx >= output_scale.size()) output_scale.resize(idx + 1, value_type(1));
        output_scale[idx] = v;
        output_scale_is_trainable = true;
    }
    inline void set_output_scale_raw(std::size_t col, value_type v) {
        set_output_scale_raw_k(col, 0, v);
    }

    // Combined rank-N scale: S[row,col] = sum_{k<scale_rank}
    // value_scale_k(row,k) * output_scale_k(col,k). THE quantity
    // Hadamard-multiplied against quant in both disldo_forward and
    // disldo_backward's quant-update -- replaces the old
    // get_value_scale(row)*get_output_scale(col) two-call pattern
    // wherever the caller wants full rank-N behavior (scattered-CSR
    // forward/backward; block4's paths still use the two-call rank-1
    // -only form, see scale_rank's own docstring).
    inline value_type get_scale(std::size_t row, std::size_t col) const {
        value_type s = value_type(0);
        for (std::size_t k = 0; k < scale_rank; ++k)
            s += get_value_scale_k(row, k) * get_output_scale_k(col, k);
        return s;
    }

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
    // Change ONE row's importance_scale mid-training without losing that
    // row's accumulated importance: re-reads its stored importance at
    // whatever scale it currently has (each row can have a DIFFERENT scale
    // -- see get_importance_scale()) into true units, re-encodes at the
    // new scale. Without this, just assigning a new scale directly would
    // silently reinterpret existing stored values as if they'd always
    // been at the new scale -- corrupting every synapse's importance in
    // one step, not just changing how future arithmetic treats it.
    inline void rescale_importance_row(std::size_t row, value_type new_scale) {
        const value_type old_scale = get_importance_scale(row);
        if (new_scale == old_scale) return;
        auto& dc = connections;
        auto& L  = dc.layout;
        if (row >= L.rows) return;
        const std::size_t n = L.row_nnz(row);
        for (std::size_t e = 0; e < n; ++e) {
            const std::size_t vb = L.elem_start[row] + e;
            const value_type w        = ValueAccessor<VALUES_TYPE>::get_w  (dc.values, vb);
            const value_type stored_i = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
            const value_type true_i   = stored_i * old_scale;
            ValueAccessor<VALUES_TYPE>::set(dc.values, vb, w, true_i / new_scale);
        }
        set_importance_scale_raw(row, new_scale);
    }

    // Bulk convenience: set EVERY row to the same new_scale. Backward-
    // compatible interface with the original per-layer-scalar design --
    // callers that never think about per-row scale at all can keep using
    // this exactly as before.
    inline void rescale_importance(value_type new_scale) {
        auto& L = connections.layout;
        for (std::size_t r = 0; r < L.rows; ++r) rescale_importance_row(r, new_scale);
    }

    // Same pattern, for STORED weight values instead of importance.
    inline void rescale_value_row(std::size_t row, value_type new_scale) {
        const value_type old_scale = get_value_scale(row);
        if (new_scale == old_scale) return;
        auto& dc = connections;
        auto& L  = dc.layout;
        if (row >= L.rows) return;
        const std::size_t n = L.row_nnz(row);
        for (std::size_t e = 0; e < n; ++e) {
            const std::size_t vb = L.elem_start[row] + e;
            const value_type stored_w = ValueAccessor<VALUES_TYPE>::get_w  (dc.values, vb);
            const value_type imp      = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
            const value_type true_w   = stored_w * old_scale;
            ValueAccessor<VALUES_TYPE>::set(dc.values, vb, true_w / new_scale, imp);
        }
        set_value_scale_raw(row, new_scale);
    }
    inline void rescale_value(value_type new_scale) {
        auto& L = connections.layout;
        for (std::size_t r = 0; r < L.rows; ++r) rescale_value_row(r, new_scale);
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
