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
    /// Same as set() but for a LIVE synapse -- see fp4_encode_bits_live's
    /// docstring (fp4quant.hpp) for the never-0 rationale. Applies to
    /// BOTH weight and importance -- see FP4BiPacked::set_live's own
    /// docstring for why importance gets the same treatment.
    static void set_live(FP4BiPacked& v, std::size_t i, value_type w, value_type imp) {
        v.set_live(i, w, imp);
    }
    /// Same as set_stochastic() but for a LIVE synapse (weight+importance).
    static void set_stochastic_live(FP4BiPacked& v, std::size_t i, value_type w, value_type imp) {
        v.set_stochastic_live(i, w, imp);
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
    /// Same as set() but for a LIVE synapse -- see fp8_encode_bits_live's
    /// docstring (fp8quant.hpp) for the never-0 (+0 AND -0) rationale.
    /// Applies to BOTH weight and importance: a live synapse's importance
    /// quantizing to the blank-slot sentinel is the same failure mode as
    /// weight doing so (nnz_row==0-style dead-row checks, and pruning
    /// decisions that read importance as the significance signal), not a
    /// separate concern -- see conversation.
    static void set_live(FP8BiValues& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = fp8_quantize_live(w);
        v.importance[i] = fp8_quantize_live(imp);
    }
    /// Same as set_stochastic() but for a LIVE synapse. Importance uses
    /// the _nonneg variant -- see fp4_quantize_stochastic_live_nonneg's
    /// docstring (fp4quant.hpp) for the full rationale (importance is
    /// always >= 0, fed into sqrt(ci); weight's cross-sign redirect would
    /// NaN it near zero).
    static void set_stochastic_live(FP8BiValues& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = fp8_quantize_stochastic_live(w);
        v.importance[i] = fp8_quantize_stochastic_live_nonneg(imp);
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
    /// No quantization happens for this (float32 fallback) storage, so
    /// the never-0 live-quantize invariant (fp4quant.hpp's
    /// fp4_encode_bits_live docstring) is meaningless here too -- an
    /// exact 0.0f float is a legitimate stored value, not a byte-0
    /// storage sentinel. Passthrough to set(), same rationale as
    /// set_stochastic() above.
    static void set_live(DeltaCSRBiValues<T>& v, std::size_t i, value_type w, value_type imp) {
        set(v, i, w, imp);
    }
    static void set_stochastic_live(DeltaCSRBiValues<T>& v, std::size_t i, value_type w, value_type imp) {
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
    // contrib_agg: row/column-aggregated forward-contribution signal
    // (Σ x*w, mirroring per-synapse ci's own contrib=x*w term -- see
    // linear_disldo.hpp's additive combination and
    // sili__new/lean_proofs/importance_signal_information_gain/
    // SiliImportanceProof/ImportanceSignalInformationGain.lean,
    // Joint.combined_signal_strictly_informative). value_scale_importance/
    // output_scale_importance are the SAME kind of RMSprop second-moment
    // accumulator as ci, just aggregated over a row/column instead of a
    // single synapse -- combined here the same way ci does: SQUARE first,
    // THEN sum (g_agg^2+contrib_agg^2), not (g_agg+contrib_agg)^2. This
    // value is the DIVISOR of the step below, so its job is safety, not
    // just importance-ranking: sum-then-square lets a large-magnitude
    // disagreement between g_agg and contrib_agg collapse the denominator
    // toward zero even though both signals are individually large,
    // exploding the step -- the same class of instability the bias-
    // correction fix below closes, just triggered by cancellation instead
    // of cold start (see conversation). Square-then-sum is bounded below
    // by max(g_agg,contrib_agg)^2 regardless of sign, so a large
    // disagreement still damps the step instead of amplifying it. The
    // actual STEP below still uses g_agg alone (unbiased: E[step]=0 under
    // zero-mean noise, only the magnitude ESTIMATE gets the extra
    // signal). Defaults to 0 for callers that don't have one (e.g.
    // NoScalePolicy siblings, or a caller not yet updated), reproducing
    // plain RMSprop exactly.
    //
    // step: Adam-style bias correction counter (Kingma & Ba 2015, sec 3).
    // scale_state starts at 0, so on step 1 it's (1-beta2)*(g_agg^2+
    // contrib_agg^2) -- badly SHRUNK toward zero, not the true magnitude
    // itself. Dividing by (1-beta2^step) undoes exactly that shrinkage:
    // on step 1, state_hat = new_state/(1-beta2) = g_agg^2+contrib_agg^2
    // exactly, so the step size becomes -eff_lr*g/(sqrt(g_agg^2+
    // contrib_agg^2)+eps) instead of -eff_lr*g/(sqrt(1-beta2)*sqrt(...)+
    // eps) -- ~1/sqrt(1-beta2) (~31.6x at the default beta2=0.999)
    // SMALLER, i.e. normal-sized instead of wildly inflated. This is
    // purely an optimizer-internal correction (nothing to do with a
    // model's own +b bias term) -- confirmed as the real root cause of a
    // genuine bug: value_scale swinging sign in a single first update
    // (1.0 -> -3.1, lr=0.5), corrupting every synapse sharing that row
    // (both scattered and block4-owned, since they share the same
    // value_scale[row]) -- see test_disldo_block4_backward.cpp's
    // regression test. Applied ONLY to value_scale_importance/
    // output_scale_importance (row/column-level, one uint32_t counter
    // each -- cheap), NOT to per-synapse `ci` (would need a counter the
    // same size as ci itself, doubling memory for an FP4/FP8 format
    // where every byte counts, for a self-limiting problem that doesn't
    // compound across a whole row the way value_scale's does).
    // log_space (default false): additive step assumes scale stays near
    // 1.0 -- a fixed-size eff_lr step is a huge RELATIVE change once scale
    // has shrunk far below 1 (which magnitude-scale reparametrization
    // deliberately does) and negligible once scale has grown large. This
    // mirrors update_cw's own scale_invariant fix, just applied to scale's
    // OWN update instead of the per-synapse weight update: d(loss)/
    // d(log(scale)) = d(loss)/d(scale)*scale = g_agg*scale (chain rule
    // through scale=exp(log_scale)), RMSprop-normalizing THAT keeps the
    // step a fixed RELATIVE (percentage) size regardless of scale's own
    // magnitude, and scale can never cross zero (exp()>0), unlike the
    // additive step. Ported verbatim from sili_peridot's torch prototype
    // (toy_tile_recurrence_rmt_torch.py's _scale_update,
    // scale_invariant_chain_rule branch) -- see that module for the
    // validated-in-torch derivation this is a direct port of.
    static void update(VALUE_TYPE& scale, VALUE_TYPE& scale_state,
                        VALUE_TYPE g_agg, VALUE_TYPE eff_lr,
                        VALUE_TYPE beta2, VALUE_TYPE eps,
                        VALUE_TYPE contrib_agg = VALUE_TYPE(0),
                        uint32_t* step = nullptr,
                        bool log_space = false) {
        if (!std::isfinite(g_agg) || !std::isfinite(contrib_agg)) return;
        if (log_space) {
            const VALUE_TYPE log_grad = g_agg * scale;
            const VALUE_TYPE log_contrib = contrib_agg * scale;
            const VALUE_TYPE new_state = beta2 * scale_state
                + (VALUE_TYPE(1) - beta2) * (log_grad * log_grad + log_contrib * log_contrib);
            if (!std::isfinite(new_state)) return;
            VALUE_TYPE state_hat = new_state;
            if (step != nullptr) {
                ++(*step);
                const VALUE_TYPE bias_correction = VALUE_TYPE(1) - std::pow(beta2, static_cast<VALUE_TYPE>(*step));
                if (bias_correction > VALUE_TYPE(0)) state_hat = new_state / bias_correction;
            }
            if (!std::isfinite(state_hat)) return;
            const VALUE_TYPE log_step = eff_lr * log_grad / (std::sqrt(state_hat) + eps);
            const VALUE_TYPE new_scale = scale * std::exp(-log_step);
            if (!std::isfinite(new_scale)) return;
            scale_state = new_state;
            scale = new_scale;
            return;
        }
        const VALUE_TYPE new_state = beta2 * scale_state
            + (VALUE_TYPE(1) - beta2) * (g_agg * g_agg + contrib_agg * contrib_agg);
        if (!std::isfinite(new_state)) return;
        VALUE_TYPE state_hat = new_state;
        if (step != nullptr) {
            ++(*step);
            const VALUE_TYPE bias_correction = VALUE_TYPE(1) - std::pow(beta2, static_cast<VALUE_TYPE>(*step));
            if (bias_correction > VALUE_TYPE(0)) state_hat = new_state / bias_correction;
        }
        if (!std::isfinite(state_hat)) return;
        const VALUE_TYPE new_scale = scale - eff_lr * g_agg / (std::sqrt(state_hat) + eps);
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
    // contrib_agg: same safety rationale as RMSpropScalePolicy's own
    // square-then-sum fix (see its docstring above) -- here the natural
    // analog is max(|g_agg|,|contrib_agg|), NOT |g_agg+contrib_agg|.
    // AdaMax's own state IS an L-infinity (max) norm tracker already
    // (that's what distinguishes it from Adam's L2/RMSprop), so combining
    // two signals via max is the same combine rule the policy already
    // uses for combining across TIME (max(beta2*scale_state, ...)) --
    // just applied across the two SIGNALS too. Summing before taking the
    // magnitude would have the identical cancellation hole as sum-then-
    // square did for RMSprop: two large, opposite-signed signals could
    // net near zero and fail to register as a large ceiling at all.
    // step: accepted only for call-site signature compatibility with
    // RMSpropScalePolicy::update -- unused here. Per Kingma & Ba 2015
    // sec 7, AdaMax's running-MAX state doesn't have RMSprop's EMA
    // cold-start shrinkage problem (max(0, combined_mag) on step 1 is
    // already the true value, not a shrunk fraction of it), so there's
    // nothing for bias correction to fix.
    static void update(VALUE_TYPE& scale, VALUE_TYPE& scale_state,
                        VALUE_TYPE g_agg, VALUE_TYPE eff_lr,
                        VALUE_TYPE beta2, VALUE_TYPE eps,
                        VALUE_TYPE contrib_agg = VALUE_TYPE(0),
                        uint32_t* step = nullptr,
                        bool log_space = false) {
        (void)step;
        (void)log_space;  // AdaMax's L-infinity tracker has no log-space variant (yet) -- accepted for call-site signature compatibility with RMSpropScalePolicy only.
        if (!std::isfinite(g_agg) || !std::isfinite(contrib_agg)) return;
        const VALUE_TYPE combined_mag = std::max(std::abs(g_agg), std::abs(contrib_agg));
        const VALUE_TYPE new_state = std::max(beta2 * scale_state, combined_mag);
        if (!std::isfinite(new_state)) return;
        const VALUE_TYPE new_scale = scale - eff_lr * g_agg / (new_state + eps);
        if (!std::isfinite(new_scale)) return;
        scale_state = new_state;
        scale = new_scale;
    }
};

// AQRS additive branch's default optimizer (task #277, see
// sili_peridot/AQRS_DESIGN.md) -- real Adam (Kingma & Ba 2015): RMSprop's
// second-moment tracking (same formula as RMSpropScalePolicy::update
// above) PLUS a genuine first-moment (momentum) EMA, used in the step's
// NUMERATOR instead of the raw gradient. Per direct instruction: default
// for the additive branch's own parameters because Adam trains faster and
// more stably than RMSprop for a small parameter count, while RMSprop
// stays available as an explicit alternative (this struct's own
// SIBLING, RMSpropScalePolicy, already exists and is unchanged) --
// callers select between them via the same template-parameter pattern
// already used everywhere else in this file, not a runtime branch.
//
// NOT a drop-in replacement for RMSpropScalePolicy::update's existing
// call sites (different signature -- needs a SEPARATE momentum_state
// reference alongside scale_state, since Adam genuinely needs two
// independent EMAs, not one) -- this is why it's a new sibling struct
// rather than an extra parameter bolted onto the existing function.
// Cannot simply call RMSpropScalePolicy::update internally either: Adam's
// second moment must be computed from the RAW gradient, but the step's
// numerator must use the momentum-SMOOTHED gradient -- RMSpropScalePolicy
// ::update uses the SAME g_agg value for both, so passing it either raw g
// or momentum-smoothed g would get one of the two uses wrong. The
// bias-correction PATTERN (not the full update) is intentionally similar
// to RMSpropScalePolicy::update's own inline version -- acknowledged as a
// small, deliberate duplication of that specific ~4-line snippet (not
// refactored into one shared helper, to avoid touching a working, tested
// function during this session) -- everything else here (the first
// -moment EMA and combining both moments in the final step) is genuinely
// new, not duplicated from anywhere.
template <typename VALUE_TYPE>
struct AdamScalePolicy {
    static void update(VALUE_TYPE& scale, VALUE_TYPE& scale_state,
                        VALUE_TYPE& momentum_state,
                        VALUE_TYPE g_agg, VALUE_TYPE eff_lr,
                        VALUE_TYPE beta1, VALUE_TYPE beta2, VALUE_TYPE eps,
                        uint32_t* step = nullptr) {
        if (!std::isfinite(g_agg)) return;

        // First moment (momentum): m = beta1*m + (1-beta1)*g.
        const VALUE_TYPE new_momentum = beta1 * momentum_state + (VALUE_TYPE(1) - beta1) * g_agg;
        if (!std::isfinite(new_momentum)) return;

        // Second moment (RMSprop-style, same formula as
        // RMSpropScalePolicy::update, computed from the RAW gradient --
        // NOT the momentum-smoothed one, per standard Adam).
        const VALUE_TYPE new_state = beta2 * scale_state + (VALUE_TYPE(1) - beta2) * (g_agg * g_agg);
        if (!std::isfinite(new_state)) return;

        VALUE_TYPE m_hat = new_momentum;
        VALUE_TYPE v_hat = new_state;
        if (step != nullptr) {
            ++(*step);
            const VALUE_TYPE t = static_cast<VALUE_TYPE>(*step);
            const VALUE_TYPE bc1 = VALUE_TYPE(1) - std::pow(beta1, t);
            const VALUE_TYPE bc2 = VALUE_TYPE(1) - std::pow(beta2, t);
            if (bc1 > VALUE_TYPE(0)) m_hat = new_momentum / bc1;
            if (bc2 > VALUE_TYPE(0)) v_hat = new_state / bc2;
        }
        if (!std::isfinite(m_hat) || !std::isfinite(v_hat)) return;

        const VALUE_TYPE new_scale = scale - eff_lr * m_hat / (std::sqrt(v_hat) + eps);
        if (!std::isfinite(new_scale)) return;
        momentum_state = new_momentum;
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
                        VALUE_TYPE /*beta2*/, VALUE_TYPE /*eps*/,
                        VALUE_TYPE /*contrib_agg*/ = VALUE_TYPE(0),
                        uint32_t* /*step*/ = nullptr,
                        bool /*log_space*/ = false) {
        // Intentionally does nothing.
    }
};

// ── Per-synapse ci-update policy (floor + clip) ───────────────────────────────
//
// Distinct from the ScalePolicy family above: those operate on value_scale/
// output_scale (one scalar per ROW/COLUMN). This operates on per-synapse `ci`
// (one scalar per SYNAPSE, linear_disldo.hpp's own RMSprop second-moment
// accumulator) -- historically hand-duplicated at ~8 call sites (6 scalar +
// 2 SIMD) rather than templated. Any future change to this formula should go
// through ONE template parameter, not shotgun surgery across every site again
// (this session already missed 2 SIMD sites once during an earlier revert,
// caught only by test_block4_scattered_divergence.cpp).
//
// Root cause this exists to fix (see conversation): a plain RMSprop `ci` EMA
// (beta2=0.999, no bias correction -- see RMSpropScalePolicy's own docstring
// for why bias correction isn't applied to per-synapse ci, unlike
// value_scale_importance/output_scale_importance) LAGS the true local
// gradient scale near a converged solution. That lag is what keeps the step
// naturally small during a long stable plateau (ci stays elevated relative to
// the now-tiny residual gradient). But ci eventually decays down to match the
// small residual too -- once it does, g/sqrt(ci) stops shrinking with the
// error and returns to ~full lr-sized steps regardless of how small the
// actual error is, causing overshoot, a larger resulting gradient, ci
// ratcheting back up, and the cycle repeating/compounding. Confirmed
// directly: a DISLDOLayer32 permutation-regression run stayed rock-stable at
// SSE=3.0000 for 300+ steps then diverged past SSE=180 by step ~405, at a
// CONSTANT lr, with `ci` visibly decaying from ~0.9 to ~0.66 right before
// the spike.
//
// An lr-decay schedule (any monotonic-to-zero form, including 1/(1+step))
// "fixes" this by eventually freezing the whole network -- incompatible with
// this project's lifelong-learning goal. Both policies below are STATIONARY
// (no dependency on step count / wall clock), so they stay compatible with
// an infinite training horizon.
//
// Exactly two entry points per policy, each owning its full formula
// end-to-end -- no math left inline at the call site, no third "combine"
// wrapper, per direct correction (an earlier draft split the RMSprop
// division/damp-branch out into the call site while only the floor and
// clip lived in the policy -- half-templated, still shotgun-surgery-prone
// for the actual optimizer math, the opposite of the point):
//   - update_ci: the complete per-synapse second-moment update. Bounded's
//     version additionally floors how fast ci is allowed to DECAY per step
//     (`max(ema, min_decay_frac * ci_old)`), closing the lag directly.
//     Explicitly NOT an AMSGrad-style permanent max: min_decay_frac<1 still
//     lets ci decay, just not fast enough to resonate, so a synapse can
//     still eventually "forget" a stale large-gradient event and re-adapt
//     -- AMSGrad's true max was considered and rejected for this reason
//     (would permanently suppress step size for any synapse that ever saw
//     one large gradient, the opposite of what a lifelong learner needs,
//     and would likely compound this project's own stuck-weights findings
//     rather than fix them).
//   - update_cw: the complete per-synapse weight delta -- the
//     damp_by_importance branch, the division, and (Bounded only) a hard
//     cap on |delta| independent of ci entirely, all in one place. The
//     clip is the standard safety net non-episodic/continual training
//     already uses everywhere else (e.g. PPO-style per-step update
//     clipping), bounding worst-case overshoot unconditionally regardless
//     of how update_ci behaves. `S` is the combined value_scale*out_scale
//     factor some call sites multiply into the delta (pass VALUE_TYPE(1)
//     for sites that don't scale) -- note `ci` itself never sees S, only
//     the delta does (matches every existing call site's own convention).
template <typename VALUE_TYPE>
struct PlainRMSpropSynapsePolicy {
    // Reproduces linear_disldo.hpp's original (pre-fix) inline formula
    // exactly, bit-for-bit, on finite inputs -- kept for explicit opt-in
    // and as the reference this whole fix was checked against, but NO
    // LONGER disldo_backward's own default (BoundedRMSpropSynapsePolicy,
    // below, is -- see the tuning sweep in
    // tests/unit/sweep_synapse_policy_min_decay_frac.cpp for why: with
    // min_decay_frac left at its own true-no-op default and max_abs_delta
    // tuned to 0.1, Bounded strictly dominates Plain -- same behavior in
    // the normal regime, real protection against the late-training
    // resonance Plain has no defense against at all).
    // No floor, no clip -- matches RMSpropScalePolicy's own "must stay
    // bit-identical to today's behavior" convention (on finite inputs),
    // PLUS the same NaN/Inf guard RMSpropScalePolicy::update already has
    // (delta_csr_types.hpp above): a non-finite g/contrib (e.g. from an
    // upstream weight that's already diverging) must not be allowed to
    // corrupt ci -- ci has no other bound in this policy, so a stray NaN
    // here is permanent (beta2*NaN+... stays NaN forever). Skip the
    // update (return the OLD ci unchanged) rather than writing NaN --
    // this is the fix for the coverage gap ba4af42 left: that commit
    // guarded value_scale/output_scale's own update but never extended
    // the same guard to the per-synapse ci/cw path (see conversation).
    static VALUE_TYPE update_ci(VALUE_TYPE ci, VALUE_TYPE g, VALUE_TYPE contrib,
                                 VALUE_TYPE beta2, VALUE_TYPE /*min_decay_frac*/,
                                 VALUE_TYPE /*max_ci*/) {
        if (!std::isfinite(g) || !std::isfinite(contrib)) return ci;
        const VALUE_TYPE new_ci = beta2 * ci + (VALUE_TYPE(1) - beta2) * (g * g + contrib * contrib);
        return std::isfinite(new_ci) ? new_ci : ci;
    }

    // Returns a DELTA (caller does `cw += update_cw(...)`), so "skip the
    // update" here means return 0 (a true no-op delta), matching
    // update_ci's own "keep old value" semantics -- same NaN/Inf guard
    // rationale as update_ci above and RMSpropScalePolicy::update.
    // scale_invariant: default false (bit-identical to every existing
    // result). ci above is calibrated to the RAW gradient g^2, unaffected
    // by S -- but the historical formula below folds S into the
    // numerator anyway, so raw isn't self-normalized w.r.t. S, and
    // Delta(true_weight) = S*Delta(cw) ends up scaling QUADRATICALLY
    // with S once S deviates from ~1.0 (found this session: fp32
    // accuracy 1.0->0.18 once a mechanism deliberately moved S away
    // from 1.0, despite true_weight = cw*S being algebraically
    // unchanged by that move). scale_invariant=true computes raw from
    // the RAW g (properly self-normalized by ci) and divides by S once
    // at the very end instead, giving Delta(true_weight) = eff_lr*raw,
    // independent of S. See BoundedRMSpropSynapsePolicy::update_cw's
    // own copy of this same fix for the max_abs_delta-clip interaction.
    static VALUE_TYPE update_cw(VALUE_TYPE g, VALUE_TYPE ci, VALUE_TYPE S,
                                 VALUE_TYPE eff_lr, VALUE_TYPE eps,
                                 bool damp_by_importance, VALUE_TYPE /*max_abs_delta*/,
                                 bool scale_invariant = false) {
        if (!std::isfinite(g) || !std::isfinite(ci) || !std::isfinite(S)) return VALUE_TYPE(0);
        VALUE_TYPE delta;
        if (scale_invariant) {
            const VALUE_TYPE raw = damp_by_importance ? (-g) / (std::sqrt(ci) + eps) : (-g);
            delta = std::isfinite(raw) ? (eff_lr * raw / S) : VALUE_TYPE(0);
        } else {
            delta = damp_by_importance
                ? (-eff_lr * g * S) / (std::sqrt(ci) + eps)
                : (-eff_lr * g * S);
        }
        return std::isfinite(delta) ? delta : VALUE_TYPE(0);
    }
};

template <typename VALUE_TYPE>
struct BoundedRMSpropSynapsePolicy {
    // Never let ci decay below `min_decay_frac * ci_old` in a single step,
    // regardless of how small the current (g,contrib) is.
    //
    // IMPORTANT for choosing a value: plain EMA's own WORST-CASE per-step
    // retention (g=contrib=0) is already exactly `beta2` -- `ci_new =
    // beta2*ci_old` in that case, nothing decays faster than that. So
    // min_decay_frac only has any effect at all when min_decay_frac > beta2
    // (retain MORE than the natural EMA floor would); min_decay_frac <=
    // beta2 is a silent no-op (the `max()` below always picks the ema
    // branch). min_decay_frac=1.0 would freeze ci forever (never decays at
    // all, AMSGrad-style, rejected -- see file-level docstring above);
    // min_decay_frac=0.0 (or any value <= beta2) reproduces plain unbounded
    // EMA decay (identical to PlainRMSpropSynapsePolicy). The intended use
    // is a value strictly between beta2 and 1 -- bounds the DECAY RATE
    // (slower than natural), not the decay itself.
    //
    // CHOSEN PRODUCTION DEFAULT (see tests/unit/sweep_synapse_policy_min_decay_frac.cpp
    // for the full sweep): min_decay_frac left at its own true-no-op value
    // (<=beta2), NOT a value in (beta2,1). Tested up to min_decay_frac=0.99995
    // with the delta clip disabled entirely and found ZERO measurable
    // protective effect against the late-training resonance this policy
    // exists to fix -- max_abs_delta's hard clip (below) is doing the ENTIRE
    // protective job on its own, UNDER DETERMINISTIC ROUNDING. Keeping the
    // ci floor "on" at some value in (beta2,1) anyway would add an unproven
    // mechanism and an extra, untested interaction surface with
    // max_abs_delta for no demonstrated benefit -- if a future task finds a
    // failure mode where the floor DOES help independently, revisit this
    // default then, with evidence.
    //
    // UPDATE (see tests/unit/sweep_synapse_policy_stochastic.cpp, and
    // conversation): under STOCHASTIC rounding (the actual production
    // rounding mode -- the deterministic sweep above turned out to get
    // permanently stuck at this exact config, an unrelated finding, see
    // that file), min_decay_frac is NOT provably inert -- it showed a real,
    // if single-seed/unconfirmed, benefit at the riskier end of the
    // max_abs_delta range (e.g. max_abs_delta=16: 0.9995 measurably beat
    // 0.999). BUT that benefit vanishes by lr~0.2 and is bit-identical
    // between tested values by lr=1.0 -- min_decay_frac does NOT extend
    // the safe lr range, so it doesn't change the production default
    // decision (still true-no-op, since production operates at lr<=0.05
    // where the clip alone is already deep-safe regardless). Revisit if a
    // real need for min_decay_frac at the risky max_abs_delta/lr combo
    // ever arises, with proper multi-seed confirmation first.
    //
    // max_ci: hard ceiling on ci itself. Confirmed directly (see
    // conversation): ci has NO ceiling anywhere else in this design --
    // min_decay_frac's floor only slows how fast ci can DECAY, it does
    // nothing to stop ci from GROWING. In the unsafe-pocket failure mode
    // (probe_unstable_pocket_growth.cpp), ci was directly measured
    // climbing continuously and unboundedly the entire 30000-step run
    // (0.0005 -> 163+, still setting a new max at literally every
    // checkpoint, never plateauing) -- ci chases g^2 upward with no cap
    // as long as the underlying divergence keeps the gradient growing.
    // Healthy production-default operation plateaus at ci~0.5, so a
    // ceiling with generous margin above that costs nothing in the normal
    // regime while giving ci's own growth an actual stopping point.
    // max_ci=1e30 (the function's own default) is a true no-op, matching
    // every other new parameter's own convention; the CHOSEN PRODUCTION
    // DEFAULT is max_ci=100.0 (set in cpu_backend.cpp's 3 real call
    // sites), verified directly in tests/unit/test_ci_ceiling.cpp: ci
    // never exceeds 100.0 across the full 30000-step unsafe-pocket run
    // (15.36M update_ci calls checked), and is a mathematically guaranteed
    // no-op at the healthy ~0.5 plateau.
    //
    // IMPORTANT CORRECTION vs an earlier hypothesis in this same
    // conversation: capping ci does NOT rescue an out-of-safe-zone
    // max_abs_delta/lr pocket's own SSE/weight-level divergence -- with
    // max_ci=100.0 applied, the SAME unsafe pocket's SSE still climbs
    // continuously (up to ~313544 by step 29950, matching the uncapped
    // run almost exactly) and STILL collapses to an all-zero output in the
    // final ~50 steps. So ci overflowing to non-finite was NOT the (sole)
    // root cause of that collapse-to-zero masking after all -- the
    // weight/cw accumulator has its own SEPARATE unbounded-growth
    // mechanism, untouched by this ceiling. max_ci is still worth
    // defaulting on as a genuine, free structural safety property for ci
    // itself, but it is NOT a substitute for staying inside the validated
    // safe max_abs_delta/lr range (sweep_synapse_policy_stochastic.cpp,
    // the lr-ceiling warning in cpu_backend.cpp).
    // Same NaN/Inf guard as PlainRMSpropSynapsePolicy::update_ci above
    // (and for the same reason -- see that struct's own docstring): a
    // non-finite g/contrib must not corrupt ci. Checked BEFORE the
    // floor/max_ci clamps since std::min/std::max's behavior on NaN is
    // comparison-order-dependent (not a reliable NaN-filter on its own).
    static VALUE_TYPE update_ci(VALUE_TYPE ci, VALUE_TYPE g, VALUE_TYPE contrib,
                                 VALUE_TYPE beta2, VALUE_TYPE min_decay_frac,
                                 VALUE_TYPE max_ci) {
        if (!std::isfinite(g) || !std::isfinite(contrib)) return ci;
        const VALUE_TYPE ema = beta2 * ci + (VALUE_TYPE(1) - beta2) * (g * g + contrib * contrib);
        if (!std::isfinite(ema)) return ci;
        const VALUE_TYPE floor = min_decay_frac * ci;
        return std::min(std::max(ema, floor), max_ci);
    }

    // Clips the LR-INDEPENDENT raw update, THEN multiplies by eff_lr --
    // NOT the other way around. Confirmed directly (see conversation): an
    // earlier version clipped the already-lr-scaled delta, making
    // max_abs_delta an ABSOLUTE cap regardless of lr -- fine at the lr=0.05
    // this policy was tuned against, but silently crushed every real step
    // for callers using a much larger lr (FoldedColumnLayer's lr=1.0
    // couldn't converge in its own test's step budget anymore, since any
    // step that would have exceeded the flat cap got clamped down to it no
    // matter how much bigger lr was set). Clipping BEFORE the lr multiply
    // (matching standard gradient-clipping convention, e.g. clip_grad_norm_
    // in this project's own Python code) makes the EFFECTIVE final-delta
    // clip naturally proportional to eff_lr (== eff_lr * max_abs_delta)
    // without needing a separate multiply -- lr keeps its intended meaning,
    // and the SAME max_abs_delta value now generalizes across callers using
    // different lr instead of only being valid at the one lr it was tuned
    // against. See tests/unit/sweep_synapse_policy_min_decay_frac.cpp's own
    // header comment for the raw-space value that reproduces the
    // already-validated lr=0.05 tuning exactly (0.1 final-space / 0.05 =
    // 2.0 raw-space).
    //
    // WARNING -- this does NOT make one fixed max_abs_delta safe for
    // UNLIMITED lr (see tests/unit/sweep_synapse_policy_stochastic.cpp
    // Round 2, and conversation): the effective final-space clip is
    // eff_lr*max_abs_delta, so a large enough lr eventually pushes it back
    // into the same large-step territory that's always been risky (the
    // underlying resonance risk is about the ABSOLUTE step size, not the
    // raw/lr split -- this redesign fixes the split, not the ceiling).
    // Measured at the production default (max_abs_delta=2.0): excellent
    // at lr<=0.05, visibly degraded by lr=0.2, diverging in absolute terms
    // by lr=0.5, genuinely unsafe by lr=1.0. cpu_backend.cpp's
    // backward_dense/backward wrappers print a one-time stderr warning if
    // called with learning_rate > 0.2 for exactly this reason -- a caller
    // that genuinely needs a much larger lr needs its own max_abs_delta
    // tuned for that regime, not this default.
    // Same "return 0 delta on non-finite input/result" guard as
    // PlainRMSpropSynapsePolicy::update_cw above, checked before the
    // max_abs_delta clip (clamping NaN against a finite bound is not a
    // reliable way to neutralize it).
    //
    // scale_invariant: default false (bit-identical to every existing
    // result) -- see PlainRMSpropSynapsePolicy::update_cw's own copy of
    // this docstring for the full derivation of why the historical
    // g*S-in-numerator formula makes Delta(true_weight) scale
    // quadratically with S. When true, the max_abs_delta clip is
    // applied to `raw` (computed from raw g, before the /S division) --
    // i.e. it bounds the properly S-normalized true-weight-space step,
    // the task-relevant quantity, not the internal w_stored-space one
    // (which is what's actually being deliberately resized by
    // magnitude-scale reparametrization).
    static VALUE_TYPE update_cw(VALUE_TYPE g, VALUE_TYPE ci, VALUE_TYPE S,
                                 VALUE_TYPE eff_lr, VALUE_TYPE eps,
                                 bool damp_by_importance, VALUE_TYPE max_abs_delta,
                                 bool scale_invariant = false) {
        if (!std::isfinite(g) || !std::isfinite(ci) || !std::isfinite(S)) return VALUE_TYPE(0);
        VALUE_TYPE raw = scale_invariant
            ? (damp_by_importance ? (-g) / (std::sqrt(ci) + eps) : (-g))
            : (damp_by_importance ? (-g * S) / (std::sqrt(ci) + eps) : (-g * S));
        if (!std::isfinite(raw)) return VALUE_TYPE(0);
        if (raw > max_abs_delta) raw = max_abs_delta;
        if (raw < -max_abs_delta) raw = -max_abs_delta;
        const VALUE_TYPE delta = scale_invariant ? (eff_lr * raw / S) : (eff_lr * raw);
        return std::isfinite(delta) ? delta : VALUE_TYPE(0);
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

// ── Per-synapse ci-update policy: Block4Vec (SIMD) specializations ────────────
//
// The primary templates above (PlainRMSpropSynapsePolicy<VALUE_TYPE>/
// BoundedRMSpropSynapsePolicy<VALUE_TYPE>) use std::max, which does not
// compile for Block4Vec (GCC vector-extension `<` produces a vector of
// comparison results, not a single bool -- same reason block4_vec_sqrt/
// block4_vec_max/block4_vec_clip_abs above are hand-written per-lane loops
// rather than calling std:: equivalents). Full explicit specializations,
// not just a VALUE_TYPE=Block4Vec instantiation of the generic template --
// mirrors this file's own scalar-vs-SIMD math duplication precedent
// (disldo_backward's scattered scalar sites vs its Block4Vec SIMD sites
// already hand-code the same formula twice; this policy abstraction
// unifies the 8 CALL SITES onto one template parameter, it doesn't
// eliminate the scalar/SIMD math split itself).
template <>
struct PlainRMSpropSynapsePolicy<Block4Vec> {
    // Same NaN/Inf guard as the scalar PlainRMSpropSynapsePolicy::update_ci
    // above (see its docstring) -- per-lane, via block4_vec_select_finite
    // (block4.hpp), since Block4Vec has no whole-vector isfinite/select.
    static Block4Vec update_ci(Block4Vec ci, Block4Vec g, Block4Vec contrib,
                                Block4Vec beta2, Block4Vec /*min_decay_frac*/,
                                Block4Vec /*max_ci*/) {
        const Block4Vec one = block4_vec_broadcast(1.0f);
        const Block4Vec new_ci = beta2 * ci + (one - beta2) * (g * g + contrib * contrib);
        return block4_vec_select_finite(new_ci, ci);
    }

    // Same "0 delta on non-finite" guard as the scalar
    // PlainRMSpropSynapsePolicy::update_cw above. scale_invariant: see
    // the scalar version's own docstring for the full derivation --
    // host-side bool (like damp_by_importance), selects which SIMD
    // formula runs, not a per-lane value.
    static Block4Vec update_cw(Block4Vec g, Block4Vec ci, Block4Vec S,
                                Block4Vec eff_lr, Block4Vec eps,
                                bool damp_by_importance, Block4Vec /*max_abs_delta*/,
                                bool scale_invariant = false) {
        Block4Vec delta;
        if (scale_invariant) {
            const Block4Vec neg_g = -g;
            const Block4Vec raw = damp_by_importance ? neg_g / (block4_vec_sqrt(ci) + eps) : neg_g;
            delta = (eff_lr * raw) / S;
        } else {
            const Block4Vec neg_lr_g_S = -(eff_lr * g * S);
            delta = damp_by_importance ? neg_lr_g_S / (block4_vec_sqrt(ci) + eps) : neg_lr_g_S;
        }
        return block4_vec_select_finite(delta, block4_vec_broadcast(0.0f));
    }
};

template <>
struct BoundedRMSpropSynapsePolicy<Block4Vec> {
    // See the scalar BoundedRMSpropSynapsePolicy::update_ci docstring above
    // for the full min_decay_frac semantics (must exceed beta2 to bind).
    // Same NaN/Inf guard as the scalar BoundedRMSpropSynapsePolicy::update_ci
    // above (checked before the floor/max_ci clamps, same rationale).
    static Block4Vec update_ci(Block4Vec ci, Block4Vec g, Block4Vec contrib,
                                Block4Vec beta2, Block4Vec min_decay_frac,
                                Block4Vec max_ci) {
        const Block4Vec one = block4_vec_broadcast(1.0f);
        const Block4Vec ema = beta2 * ci + (one - beta2) * (g * g + contrib * contrib);
        const Block4Vec ema_safe = block4_vec_select_finite(ema, ci);
        const Block4Vec floor = min_decay_frac * ci;
        return block4_vec_min(block4_vec_max(ema_safe, floor), max_ci);
    }

    // Clips the lr-independent raw update before the eff_lr multiply --
    // see the scalar BoundedRMSpropSynapsePolicy::update_cw docstring above
    // for why (must match it exactly, or SIMD full-tile vs scalar-boundary
    // results would diverge for the same synapse). Same NaN/Inf guard as
    // the scalar version, checked before the clip.
    // scale_invariant: see the scalar BoundedRMSpropSynapsePolicy::
    // update_cw docstring above for the full derivation -- host-side
    // bool (like damp_by_importance), selects which SIMD formula runs.
    static Block4Vec update_cw(Block4Vec g, Block4Vec ci, Block4Vec S,
                                Block4Vec eff_lr, Block4Vec eps,
                                bool damp_by_importance, Block4Vec max_abs_delta,
                                bool scale_invariant = false) {
        Block4Vec raw;
        if (scale_invariant) {
            const Block4Vec neg_g = -g;
            raw = damp_by_importance ? neg_g / (block4_vec_sqrt(ci) + eps) : neg_g;
        } else {
            const Block4Vec neg_g_S = -(g * S);
            raw = damp_by_importance ? neg_g_S / (block4_vec_sqrt(ci) + eps) : neg_g_S;
        }
        const Block4Vec raw_safe = block4_vec_select_finite(raw, block4_vec_broadcast(0.0f));
        const Block4Vec clipped = block4_vec_clip_abs(raw_safe, max_abs_delta);
        const Block4Vec delta = scale_invariant ? (eff_lr * clipped) / S : (eff_lr * clipped);
        return block4_vec_select_finite(delta, block4_vec_broadcast(0.0f));
    }
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

// Reshuffles `arr` (currently laid out as entity*old_rank+k) to
// entity*new_rank+k in place, preserving every existing (entity,k) pair
// with k<min(old_rank,new_rank) and filling any new slots with
// `default_for_k(k)`. No-op if arr is empty (nothing written yet -- the
// common case, since most layers never touch scale beyond component 0 of
// a couple of rows). Free function (not a SparseLinearWeightsDelta member)
// specifically so GammaEMATracker below can reuse it too, rather than
// hand-duplicating the same logic -- was a private static member here
// until AQRS's additive-branch gamma dynamic-rank-control work needed the
// identical reshuffle for a second, independent set of rank-length arrays.
template <typename T, typename DefaultFn>
static void reshuffle_rank_array(std::vector<T>& arr, std::size_t old_rank,
                                 std::size_t new_rank, DefaultFn default_for_k) {
    if (arr.empty() || old_rank == new_rank) return;
    const std::size_t n_entities = (arr.size() + old_rank - 1) / old_rank;
    std::vector<T> resized(n_entities * new_rank);
    for (std::size_t e = 0; e < n_entities; ++e) {
        for (std::size_t k = 0; k < new_rank; ++k) {
            const std::size_t old_idx = e * old_rank + k;
            resized[e * new_rank + k] =
                (k < old_rank && old_idx < arr.size()) ? arr[old_idx] : default_for_k(k);
        }
    }
    arr = std::move(resized);
}

// Shared EMA-tracking + Theorem 10 trigger-condition machinery for AQRS's
// per-rank-channel gamma, used by BOTH the multiplicative branch
// (scale_gamma) and the additive branch (additive_gamma) -- the math here
// (EMA update formula, apoptosis/neurogenesis trigger conditions, channel
// swap) is IDENTICAL between the two branches (see AQRS_DESIGN.md's
// Theorem 10, stated "per branch, per rank channel"); only the raw gamma
// VALUE's own storage/lazy-default semantics differ between branches
// (scale_gamma defaults transparently to 1.0, additive_gamma likewise
// defaults to 1.0 for backward compat with existing rank>0 additive
// callers that never touch gamma -- see get_additive_gamma_k's own
// docstring), so only that piece stays branch-specific, living directly
// on SparseLinearWeightsDelta rather than in here. Extracted specifically
// per direct instruction not to duplicate scale_gamma's already-proven
// EMA/trigger logic when adding the equivalent for additive_gamma.
template <typename value_type>
struct GammaEMATracker {
    std::vector<value_type> abs_ema;
    std::vector<value_type> share_ema;
    std::vector<value_type> grad_ema;

    inline value_type get_abs_ema_k(std::size_t k) const {
        return k < abs_ema.size() ? abs_ema[k] : value_type(0);
    }
    inline value_type get_share_ema_k(std::size_t k) const {
        return k < share_ema.size() ? share_ema[k] : value_type(0);
    }
    inline value_type get_grad_ema_k(std::size_t k) const {
        return k < grad_ema.size() ? grad_ema[k] : value_type(0);
    }
    // Called once per k, once per backward call, AFTER gamma's own value
    // update for every channel is finalized (share_k needs every
    // channel's current |gamma| to compute the group's L1 norm first).
    inline void update_k(std::size_t k, value_type abs_gamma_k, value_type share_k,
                         value_type abs_grad_k, value_type decay = value_type(0.98)) {
        if (abs_ema.size() <= k) abs_ema.resize(k + 1, value_type(0));
        if (share_ema.size() <= k) share_ema.resize(k + 1, value_type(0));
        if (grad_ema.size() <= k) grad_ema.resize(k + 1, value_type(0));
        abs_ema[k]   = decay * abs_ema[k]   + (value_type(1) - decay) * abs_gamma_k;
        share_ema[k] = decay * share_ema[k] + (value_type(1) - decay) * share_k;
        grad_ema[k]  = decay * grad_ema[k]  + (value_type(1) - decay) * abs_grad_k;
    }
    // Theorem 10's exact apoptosis trigger: A(gamma_i) = (|gamma_i| <
    // tau_death) AND (C_i < tau_death) -- evaluated against the EMA
    // values (the noise filter), not the raw instantaneous gamma.
    inline bool should_apoptose(std::size_t k, value_type tau_death) const {
        return get_abs_ema_k(k) < tau_death && get_share_ema_k(k) < tau_death;
    }
    // Theorem 10's exact neurogenesis trigger: N(gamma,grad) = (min_j
    // |gamma_j| > tau_active) AND (max_j grad_j > theta) -- takes the
    // CURRENT rank explicitly (not stored here) since "every existing
    // channel" means every k<rank, not every k the EMA arrays happen to
    // have grown to (a channel could have been apoptosed/shrunk away).
    inline bool should_neurogenesis(std::size_t rank, value_type tau_active, value_type theta) const {
        if (rank == 0) return false;
        value_type min_abs = get_abs_ema_k(0);
        value_type max_grad = get_grad_ema_k(0);
        for (std::size_t k = 1; k < rank; ++k) {
            min_abs = std::min(min_abs, get_abs_ema_k(k));
            max_grad = std::max(max_grad, get_grad_ema_k(k));
        }
        return min_abs > tau_active && max_grad > theta;
    }
    // EMA state travels WITH a relocated channel (see swap_scale_channels'
    // own docstring for why -- omitting this makes a relocated channel
    // look freshly-born to the trigger logic).
    inline void swap_k(std::size_t k1, std::size_t k2) {
        if (k1 == k2) return;
        auto swap_one = [&](std::vector<value_type>& arr) {
            const std::size_t need = std::max(k1, k2) + 1;
            if (arr.size() < need) arr.resize(need, value_type(0));
            std::swap(arr[k1], arr[k2]);
        };
        swap_one(abs_ema);
        swap_one(share_ema);
        swap_one(grad_ema);
    }
    inline void reshuffle(std::size_t old_rank, std::size_t new_rank) {
        auto zero_default = [](std::size_t) { return value_type(0); };
        reshuffle_rank_array(abs_ema, old_rank, new_rank, zero_default);
        reshuffle_rank_array(share_ema, old_rank, new_rank, zero_default);
        reshuffle_rank_array(grad_ema, old_rank, new_rank, zero_default);
    }
};

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

    // task #295 fix (real user request): block4's SIMD backward path
    // (linear_disldo.hpp) used to size its per-rank-component accumulators
    // as FIXED stack arrays capped at a compile-time SCALE_RANK_MAX=4 --
    // that constant is now GONE. Instead, `scale_rank_scratch` below is a
    // set of persistent, per-instance HEAP buffers that grow (never
    // shrink automatically) to fit whatever scale_rank the layer actually
    // uses, reused across every disldo_backward call rather than
    // reallocated per call or per tile -- the naive "heap-allocate a
    // std::vector fresh on every tile visited" approach WOULD be a real
    // regression (that loop runs once per (block-row, block-col, li)
    // triple, extremely often); this avoids that by allocating once and
    // reusing. Explicit shrinking (freeing unused capacity a layer grew
    // into early on) is available via reserve_scale_rank_scratch below,
    // separate from the automatic grow-only path disldo_backward uses.
    //
    // Deliberately holds ONLY value_type/double vectors, no Block4Vec --
    // this header doesn't include block4.hpp (Block4Vec's own home), so
    // linear_disldo.hpp's block4 SIMD code loads/stores Block4Vec values
    // from/to these plain buffers via block4_vec_load/block4_vec_store
    // (already its own established pattern, see out_scale_k4/mcol4_rank's
    // pre-existing use of that exact load/store convention) rather than
    // this struct storing SIMD-typed data directly.
    struct ScaleRankScratch {
        std::vector<value_type> value_scale_k;        // [thread][k]
        std::vector<value_type> out_scale_k;           // [thread][k][tile_width]
        std::vector<value_type> mcol_rank;              // [thread][k][tile_width]
        std::vector<double>     mrow_local_k;           // [thread][k]
        std::vector<value_type> mcol_rank_contrib;      // [thread][k][tile_width]
        std::vector<double>     mrow_local_k_contrib;   // [thread][k]
        std::vector<double>     mgamma_local_k;         // [thread][k]
        std::vector<double>     mgamma_local_k_contrib; // [thread][k]
        std::vector<value_type> mcol_acc_raw;           // [thread][k][tile_width] -- Block4Vec accumulator backing
        std::vector<value_type> mcol_acc_raw_contrib;   // [thread][k][tile_width]

        std::size_t cap_threads = 0, cap_rank = 0, cap_tile_width = 0;

        // Grow-only (never shrinks) -- called automatically at the top of
        // disldo_backward every call, a cheap no-op once already large
        // enough. tile_width is BLOCK4_TILE, passed in rather than
        // hardcoded (this header has no block4.hpp dependency).
        void ensure(std::size_t threads, std::size_t rank, std::size_t tile_width) {
            if (threads <= cap_threads && rank <= cap_rank && tile_width <= cap_tile_width) return;
            resize_to(std::max(cap_threads, threads), std::max(cap_rank, rank),
                       std::max(cap_tile_width, tile_width));
        }

        // Explicit, caller-driven resize -- unlike ensure(), this CAN
        // shrink (frees capacity a layer grew into early on and no longer
        // needs). Caller (set_scale_rank_scratch_capacity below) is
        // responsible for validating threads/rank/tile_width aren't
        // smaller than what's currently actually in use.
        void resize_to(std::size_t threads, std::size_t rank, std::size_t tile_width) {
            cap_threads = threads; cap_rank = rank; cap_tile_width = tile_width;
            const std::size_t flat = threads * rank;
            const std::size_t flat_tiled = flat * tile_width;
            value_scale_k.resize(flat);
            out_scale_k.resize(flat_tiled);
            mcol_rank.resize(flat_tiled);
            mrow_local_k.resize(flat);
            mcol_rank_contrib.resize(flat_tiled);
            mrow_local_k_contrib.resize(flat);
            mgamma_local_k.resize(flat);
            mgamma_local_k_contrib.resize(flat);
            mcol_acc_raw.resize(flat_tiled);
            mcol_acc_raw_contrib.resize(flat_tiled);
        }
    };
    ScaleRankScratch scale_rank_scratch;

    // Explicit, caller-driven scratch memory control (task #295, real
    // user request) -- separate from scale_rank_max/additive_rank_max
    // below (which govern the POLICY cap on how far rank may grow, not
    // memory). Allows shrinking (frees capacity a layer grew into
    // early on and no longer needs) as well as growing ahead of need
    // (preallocate once at a known max, avoid any reallocation during
    // training). threads/rank must not be smaller than what's currently
    // actually in use -- shrinking below that would corrupt the buffers
    // disldo_backward is actively reading/writing.
    inline void reserve_scale_rank_scratch(std::size_t threads, std::size_t rank, std::size_t tile_width) {
        if (rank < scale_rank)
            throw std::invalid_argument("reserve_scale_rank_scratch: rank below the layer's current scale_rank would corrupt live scratch data");
        if (threads < 1) throw std::invalid_argument("reserve_scale_rank_scratch: threads must be >= 1");
        scale_rank_scratch.resize_to(threads, rank, tile_width);
    }

    // additive_rank has no fixed-size-stack-array SIMD path (its own
    // forward/backward pass, linear_disldo.hpp, uses ordinary
    // std::vector throughout already, structure-agnostic w.r.t. block4)
    // -- so unlike scale_rank, it never needed a compile-time cap at all;
    // the one remaining fixed-size array (dgamma_by_k in disldo_backward's
    // additive gamma update block) is allocated once per call already, at
    // the same frequency as its own P/dP std::vectors, so a plain
    // std::vector<value_type>(r_o) there is a trivial, zero-risk swap
    // (task #295).

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
    // AND BOTH block4 paths -- FP4 and FP8 (process_tile in
    // linear_disldo.hpp) -- are all rank-aware, using
    // weights.get_scale(row,col). The SIMD fast path (both FP4 and FP8)
    // keeps its 4-wide column vectorization regardless of rank by looping
    // the (small, SCALE_RANK_MAX-capped) rank dimension outside the lane
    // dimension with real Block4Vec accumulators per component, rather
    // than falling back to scalar for rank>1. Scattered and block4 must
    // both be correct here, not just one: real training layers hold a
    // MIX of both storages simultaneously via synaptogenesis promotion/
    // demotion, sharing the same value_scale/output_scale arrays.
    //
    // Still rank-1-only (component 0 of a rank>1 layer): every
    // DeferredScaleWrite class (e.g. SparseLinearLayerResync) on both
    // scattered and block4 paths (see disldo_backward's own
    // DeferredScaleWrite branch comment for why rank>1 doesn't apply
    // there without more work: eager multiply-by-not-yet-finalized-scale
    // would reintroduce the staleness DeferredScaleWrite exists to
    // avoid). If a rank>1 layer ever uses a DeferredScaleWrite class, its
    // effective scale silently drops every component beyond 0 there --
    // tracked as a follow-up, not fixed here.
    std::size_t scale_rank = 1;

    // AQRS dynamic rank control (task #292 fix): per-branch cooldown
    // counter -- calls since the LAST rank mutation of EITHER kind
    // (apoptosis or neurogenesis), not just a per-channel apoptosis-only
    // age check. AQRS_DESIGN.md's own Theorem 10 text says the
    // tau_death/tau_active hysteresis gap is what "stops a channel from
    // immediately regrowing the instant it's pruned" -- but that only
    // holds if gamma's own per-step movement is small relative to the
    // gap; a real MQAR run showed gamma's raw gradient can be large
    // enough to jump the whole gap in one step, defeating it. This
    // counter is the belt-and-suspenders fix: BOTH apoptosis and
    // neurogenesis are additionally gated on grace_period_steps calls
    // having passed since the last mutation (symmetric, not just
    // apoptosis-only as before), same interim "age-gate" approach used
    // instead of a full energy/resource-cost-tied refractory period
    // (direct instruction: age-gate is fine for now, keep it a real
    // tunable parameter).
    uint32_t scale_rank_calls_since_mutation = UINT32_MAX;

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
        if (idx >= value_scale.size()) {
            // CORRECTED (real bug, see linear_disldo.hpp's disldo_backward
            // pre-size fix for the full trace): a uniform resize(...,1.0)
            // fill backfills every newly-appended slot with 1.0, including
            // k>=1 ones, contradicting this class's own documented default
            // (k==0 -> 1.0, k>=1 -> 0.0, see get_value_scale_k). Resize
            // neutral (0), then fix up only the k==0 slots in the new range.
            const std::size_t old_size = value_scale.size();
            value_scale.resize(idx + 1, value_type(0));
            for (std::size_t i = old_size; i < value_scale.size(); ++i)
                if (i % scale_rank == 0) value_scale[i] = value_type(1);
        }
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

    // Step counter for value_scale_importance's Adam-style bias correction
    // (RMSpropScalePolicy::update) -- see its own docstring for why this
    // is needed: on the FIRST update, an EMA started at 0 is badly
    // shrunk toward zero (state = (1-beta2)*g^2, not g^2), which makes
    // the very first RMSprop step ~1/sqrt(1-beta2) (~31.6x at the
    // default beta2=0.999) larger than intended -- confirmed as the real
    // cause of a genuine bug (value_scale swinging sign in one step,
    // corrupting every synapse sharing that row, both scattered and
    // block4-owned -- see test_disldo_block4_backward.cpp's regression
    // test). uint32_t, one per row*rank slot -- cheap (unlike a
    // per-synapse counter, which would double memory for an FP4/FP8
    // format where every byte counts; per-synapse `ci` does NOT get this
    // treatment for that reason, see linear_disldo.hpp's own note).
    std::vector<uint32_t> value_scale_step;
    inline uint32_t& get_value_scale_step_k(std::size_t row, std::size_t k) {
        const std::size_t idx = row * scale_rank + k;
        if (value_scale_step.size() <= idx) value_scale_step.resize(idx + 1, 0);
        return value_scale_step[idx];
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
    // Step counter for output_scale_importance's bias correction -- see
    // value_scale_step's own docstring for the full rationale (same
    // mechanism, one level over from row to column).
    std::vector<uint32_t> output_scale_step;
    inline uint32_t& get_output_scale_step_k(std::size_t col, std::size_t k) {
        const std::size_t idx = col * scale_rank + k;
        if (output_scale_step.size() <= idx) output_scale_step.resize(idx + 1, 0);
        return output_scale_step[idx];
    }
    inline value_type get_output_scale_importance(std::size_t col) const {
        return get_output_scale_importance_k(col, 0);
    }
    inline void set_output_scale_raw_k(std::size_t col, std::size_t k, value_type v) {
        const std::size_t idx = col * scale_rank + k;
        if (idx >= output_scale.size()) {
            // Same fix as set_value_scale_raw_k above -- see its comment.
            const std::size_t old_size = output_scale.size();
            output_scale.resize(idx + 1, value_type(0));
            for (std::size_t i = old_size; i < output_scale.size(); ++i)
                if (i % scale_rank == 0) output_scale[i] = value_type(1);
        }
        output_scale[idx] = v;
        output_scale_is_trainable = true;
    }
    inline void set_output_scale_raw(std::size_t col, value_type v) {
        set_output_scale_raw_k(col, 0, v);
    }

    // AQRS per-channel gamma (task #273/#282-283, see sili_peridot/
    // AQRS_DESIGN.md's gamma section): decouples channel MAGNITUDE from
    // channel DIRECTION. value_scale_k/output_scale_k above hold pure
    // direction; gamma_s_k (ONE scalar per channel k, not per row/col)
    // holds the magnitude: S[row,col] = sum_k gamma_s_k * value_scale_k
    // (row,k) * output_scale_k(col,k).
    //
    // CORRECTED (real backward-compat break, found via the full regression
    // suite after landing gamma -- see conversation): the LAZY default here
    // must be 1.0 for EVERY k, not just k==0. An earlier version defaulted
    // k>=1 to 0.0 to match Theorem 9's "new channel starts at zero
    // contribution" -- but that silently broke every EXISTING rank>1 layer
    // that sets value_scale_k/output_scale_k directly (the established,
    // still-valid construction pattern -- e.g. direct `w.scale_rank = N`
    // assignment, bypassing set_scale_rank entirely) without ever touching
    // gamma: their k>=1 components went from contributing normally to
    // silently zeroed, since scale_gamma stays empty and the lazy default
    // used to kick in unconditionally. Confirmed by a real regression:
    // test_scale_handling.cpp's magnitude_rescale_output rank-2 test
    // failed after the k>=1->0.0 default landed.
    //
    // The "new channel = zero contribution" property (still needed for
    // task #273's real dynamic growth) now lives ONLY in set_scale_rank's
    // reshuffle (below), which writes an explicit 0.0 into a genuinely new
    // k slot -- but ONLY fires when scale_gamma is already non-empty (i.e.
    // gamma is ALREADY in active use, matching #273's actual use case: a
    // layer under live dynamic rank control). For any layer that never
    // touches gamma (every pre-gamma caller, and every one-shot
    // construction-time rank sizing), reshuffle_rank_array no-ops on an
    // empty array and the lazy default below applies uniformly -- gamma
    // stays fully transparent (=1 everywhere), bit-identical to pre-gamma
    // behavior.
    //
    // scale_gamma_is_trainable: same opt-in gate as output_scale_is_
    // trainable above, same reason -- disldo_backward's gamma UPDATE
    // (not the accumulation, which is harmless and cheap even when
    // unused) must be skipped entirely unless a caller has explicitly
    // engaged gamma via set_scale_gamma_raw_k. Without this, EVERY
    // existing rank>=1 layer -- including every one that has never heard
    // of gamma -- would have its effective magnitude silently perturbed
    // by an unsolicited gradient-driven update to gamma_s_k(0) (which
    // starts at the transparent 1.0 default but is NOT const just
    // because it's untouched). Confirmed as a real regression: without
    // this gate, test_aqrs_additive_branch.cpp and test_aqrs_rank_growth_
    // shrink.cpp (neither of which ever touches gamma) both failed,
    // because gamma_s_k(0) was drifting away from 1.0 on every step.
    bool scale_gamma_is_trainable = false;
    std::vector<value_type> scale_gamma;
    std::vector<value_type> scale_gamma_state;   // RMSprop second moment, one per channel
    std::vector<uint32_t>   scale_gamma_step;    // bias-correction counter, one per channel
    inline value_type get_scale_gamma_k(std::size_t k) const {
        if (k < scale_gamma.size()) return scale_gamma[k];
        return value_type(1);
    }
    inline void set_scale_gamma_raw_k(std::size_t k, value_type v) {
        if (k >= scale_gamma.size()) scale_gamma.resize(k + 1, value_type(1));
        scale_gamma[k] = v;
        scale_gamma_is_trainable = true;
    }
    inline value_type& get_scale_gamma_state_k(std::size_t k) {
        if (scale_gamma_state.size() <= k) scale_gamma_state.resize(k + 1, value_type(0));
        return scale_gamma_state[k];
    }
    inline uint32_t& get_scale_gamma_step_k(std::size_t k) {
        if (scale_gamma_step.size() <= k) scale_gamma_step.resize(k + 1, 0);
        return scale_gamma_step[k];
    }

    // ══════════════════════════════════════════════════════════════════
    // AQRS dynamic rank control (task #273/#284, see sili_peridot/
    // AQRS_DESIGN.md's "corrected" noise-mitigation design): EMA-smoothed
    // per-channel signals, updated EVERY step (not every N steps -- the
    // design doc explicitly rejects periodic checking as a "luck filter",
    // not a real noise filter), feeding the exact Theorem 10 apoptosis/
    // neurogenesis trigger conditions below. Three signals per channel:
    //   |gamma_k|_ema  -- EMA of the channel's own magnitude
    //   C_k_ema        -- EMA of C_k = |gamma_k| / sum_j|gamma_j| (this
    //                     channel's share of the group's total L1 mass)
    //   grad_k_ema     -- EMA of |dL/d(gamma_k)| (gamma is a scalar per
    //                     channel, so this reduces to a plain magnitude,
    //                     not a Frobenius norm -- there's no row/col
    //                     structure at the gamma level to norm over)
    // decay=0.98 matches the same EMA pattern already used for loss_ema/
    // acc_ema in sili_peridot's MQAR curriculum (train_mqar_curriculum.py)
    // -- not a new convention, reused deliberately.
    //
    // EMA storage + Theorem 10 trigger logic itself now lives in the
    // shared GammaEMATracker (see its own docstring above) -- the
    // methods below are thin delegating wrappers, kept under their
    // original names/signatures so no existing caller (tests, disldo_
    // backward, apply_dynamic_rank_control) needed to change.
    GammaEMATracker<value_type> scale_gamma_ema;
    inline value_type get_scale_gamma_abs_ema_k(std::size_t k) const { return scale_gamma_ema.get_abs_ema_k(k); }
    inline value_type get_scale_gamma_share_ema_k(std::size_t k) const { return scale_gamma_ema.get_share_ema_k(k); }
    inline value_type get_scale_gamma_grad_ema_k(std::size_t k) const { return scale_gamma_ema.get_grad_ema_k(k); }
    // Called once per k, once per backward call, AFTER gamma's own value
    // update for every channel is finalized (C_k needs every channel's
    // current |gamma| to compute ||gamma||_1 first -- see disldo_backward's
    // own two-pass structure: update all gamma_k, THEN update all EMAs).
    inline void update_scale_gamma_ema_k(std::size_t k, value_type abs_gamma_k,
                                          value_type share_k, value_type abs_grad_k,
                                          value_type decay = value_type(0.98)) {
        scale_gamma_ema.update_k(k, abs_gamma_k, share_k, abs_grad_k, decay);
    }
    inline bool scale_gamma_should_apoptose(std::size_t k, value_type tau_death) const {
        return scale_gamma_ema.should_apoptose(k, tau_death);
    }
    inline bool scale_gamma_should_neurogenesis(std::size_t rank, value_type tau_active, value_type theta) const {
        return scale_gamma_ema.should_neurogenesis(rank, tau_active, theta);
    }

    // Combined rank-N scale: S[row,col] = sum_{k<scale_rank} gamma_s_k *
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
            s += get_scale_gamma_k(k) * get_value_scale_k(row, k) * get_output_scale_k(col, k);
        return s;
    }

    // ══════════════════════════════════════════════════════════════════════
    // AQRS additive branch: A[row,col] = sum_{k<additive_rank}
    // additive_u_k(row,k) * additive_v_k(col,k) -- ADDED (not Hadamard
    // -multiplied against quant like value_scale/output_scale above) to
    // the effective weight. See sili_peridot/AQRS_DESIGN.md for the full
    // derivation: proven structurally necessary (not just useful) because
    // the multiplicative branch's gradient is exactly zero at any (row,
    // col) where the quantized weight is the zero code, at any rank --
    // only an additive term can write a value there. Same two-plain
    // -vector convention as value_scale/output_scale (no separate
    // diag(gamma) scale term), confirmed by reading that implementation
    // first rather than inventing a different convention -- see
    // conversation. additive_rank default 0 means the branch is a pure
    // no-op (matches value_scale/output_scale's own "unconfigured
    // component contributes nothing" convention, just at rank 0 instead
    // of per-component). Optimizer state (importance/step/momentum for
    // an Adam-style update) deliberately NOT added here -- that's task
    // #277's scope, tied to the specific policy chosen; adding it now
    // without knowing that design would risk exactly the kind of
    // guessed-then-duplicated state this project's own "don't duplicate
    // code" instruction was warning against.
    std::size_t additive_rank = 0;
    // Same cooldown counter as scale_rank_calls_since_mutation above, own
    // copy since the two branches mutate independently.
    uint32_t additive_rank_calls_since_mutation = UINT32_MAX;
    std::vector<value_type> additive_u;  // row-major per-component: additive_u[row*additive_rank+k]
    std::vector<value_type> additive_v;  // row-major per-component: additive_v[col*additive_rank+k]

    inline value_type get_additive_u_k(std::size_t row, std::size_t k) const {
        const std::size_t idx = row * additive_rank + k;
        return idx < additive_u.size() ? additive_u[idx] : value_type(0);
    }
    inline void set_additive_u_raw_k(std::size_t row, std::size_t k, value_type v) {
        const std::size_t idx = row * additive_rank + k;
        if (idx >= additive_u.size()) additive_u.resize(idx + 1, value_type(0));
        additive_u[idx] = v;
    }
    inline value_type get_additive_v_k(std::size_t col, std::size_t k) const {
        const std::size_t idx = col * additive_rank + k;
        return idx < additive_v.size() ? additive_v[idx] : value_type(0);
    }
    inline void set_additive_v_raw_k(std::size_t col, std::size_t k, value_type v) {
        const std::size_t idx = col * additive_rank + k;
        if (idx >= additive_v.size()) additive_v.resize(idx + 1, value_type(0));
        additive_v[idx] = v;
    }

    // AQRS per-channel gamma for the additive branch (task #273/#282-283,
    // wired into forward/backward for real at task #289), same role as
    // scale_gamma above but for additive_u/additive_v: ONE scalar per
    // channel k, decoupling magnitude from direction --
    // A[row,col] = sum_k additive_gamma_k * additive_u_k(row,k) *
    // additive_v_k(col,k), matching AQRS_DESIGN.md's A(theta_o) =
    // sum_k gamma_o_k*u_k*v_k^T exactly.
    //
    // CORRECTED (real backward-compat break, same class of bug as
    // scale_gamma's own -- see get_scale_gamma_k's docstring above, and
    // caught here BEFORE landing rather than via a regression this time):
    // an earlier version of this code defaulted get_additive_gamma_k to
    // 0.0 on the reasoning "the additive branch has no legacy always-on
    // component to preserve." That's true of additive_rank itself
    // (0 = fully off) but NOT of gamma once it's actually multiplied into
    // the branch's forward/backward math -- every EXISTING caller that
    // sets additive_rank>0 and populates additive_u/additive_v directly
    // (task #278's pybind bindings, the fp8/fp4 MQAR curriculum runs
    // already on record) never touches gamma at all, so a 0.0 lazy
    // default would silently zero out their entire additive contribution
    // the moment gamma gets wired in. Lazy default is 1.0 (transparent),
    // exactly mirroring scale_gamma's own get_scale_gamma_k -- Theorem
    // 9's "new channel = zero contribution" property lives ONLY in
    // set_additive_rank's reshuffle below (zero_default), same split as
    // scale_gamma's.
    bool additive_gamma_is_trainable = false;
    std::vector<value_type> additive_gamma;
    // RMSprop-style state (matches scale_gamma's OWN update policy
    // exactly -- disldo_backward's additive_gamma update block uses the
    // function's generic `ScalePolicy` template param, same as scale_
    // gamma, NOT AdamScalePolicy, even though additive_u/additive_v use
    // AdamScalePolicy). Found via a real, direct test failure (see
    // conversation): an earlier version used AdamScalePolicy here to
    // "match additive_u/v's own optimizer choice" -- Adam's momentum
    // term overshoots a hard L1-created zero fixed point (Theorem 8),
    // since Adam keeps pushing in its accumulated momentum direction for
    // a step or two AFTER the raw gradient has already crossed zero,
    // driving gamma persistently negative instead of settling exactly at
    // 0 the way scale_gamma's own (momentum-free) L1 test does. gamma is
    // the SAME kind of shared magnitude-decoupling parameter in both
    // branches with the SAME Theorem 8 exact-zero-fixed-point
    // requirement, so it uses the SAME policy in both -- only the
    // direction vectors (value_scale/output_scale vs additive_u/v) get
    // to pick their own optimizer independently.
    std::vector<value_type> additive_gamma_state;
    std::vector<uint32_t>   additive_gamma_step;
    GammaEMATracker<value_type> additive_gamma_ema;
    inline value_type get_additive_gamma_k(std::size_t k) const {
        if (k < additive_gamma.size()) return additive_gamma[k];
        return value_type(1);
    }
    inline void set_additive_gamma_raw_k(std::size_t k, value_type v) {
        if (k >= additive_gamma.size()) additive_gamma.resize(k + 1, value_type(1));
        additive_gamma[k] = v;
        additive_gamma_is_trainable = true;
    }
    inline value_type& get_additive_gamma_state_k(std::size_t k) {
        if (additive_gamma_state.size() <= k) additive_gamma_state.resize(k + 1, value_type(0));
        return additive_gamma_state[k];
    }
    inline uint32_t& get_additive_gamma_step_k(std::size_t k) {
        if (additive_gamma_step.size() <= k) additive_gamma_step.resize(k + 1, 0);
        return additive_gamma_step[k];
    }
    inline value_type get_additive_gamma_abs_ema_k(std::size_t k) const { return additive_gamma_ema.get_abs_ema_k(k); }
    inline value_type get_additive_gamma_share_ema_k(std::size_t k) const { return additive_gamma_ema.get_share_ema_k(k); }
    inline value_type get_additive_gamma_grad_ema_k(std::size_t k) const { return additive_gamma_ema.get_grad_ema_k(k); }
    inline void update_additive_gamma_ema_k(std::size_t k, value_type abs_gamma_k,
                                            value_type share_k, value_type abs_grad_k,
                                            value_type decay = value_type(0.98)) {
        additive_gamma_ema.update_k(k, abs_gamma_k, share_k, abs_grad_k, decay);
    }
    inline bool additive_gamma_should_apoptose(std::size_t k, value_type tau_death) const {
        return additive_gamma_ema.should_apoptose(k, tau_death);
    }
    inline bool additive_gamma_should_neurogenesis(std::size_t rank, value_type tau_active, value_type theta) const {
        return additive_gamma_ema.should_neurogenesis(rank, tau_active, theta);
    }

    // AdamScalePolicy's own state for additive_u/additive_v (task #277) --
    // same lazy-growth, row-major-per-component convention as everything
    // else here. Two independent EMAs per Adam's own definition (first
    // moment = momentum, second moment = state), plus one step counter
    // for bias correction -- see AdamScalePolicy::update's own docstring.
    std::vector<value_type> additive_u_momentum, additive_u_state;
    std::vector<value_type> additive_v_momentum, additive_v_state;
    std::vector<uint32_t>   additive_u_step, additive_v_step;
    inline value_type& get_additive_u_momentum_k(std::size_t row, std::size_t k) {
        const std::size_t idx = row * additive_rank + k;
        if (additive_u_momentum.size() <= idx) additive_u_momentum.resize(idx + 1, value_type(0));
        return additive_u_momentum[idx];
    }
    inline value_type& get_additive_u_state_k(std::size_t row, std::size_t k) {
        const std::size_t idx = row * additive_rank + k;
        if (additive_u_state.size() <= idx) additive_u_state.resize(idx + 1, value_type(0));
        return additive_u_state[idx];
    }
    inline uint32_t& get_additive_u_step_k(std::size_t row, std::size_t k) {
        const std::size_t idx = row * additive_rank + k;
        if (additive_u_step.size() <= idx) additive_u_step.resize(idx + 1, 0);
        return additive_u_step[idx];
    }
    inline value_type& get_additive_v_momentum_k(std::size_t col, std::size_t k) {
        const std::size_t idx = col * additive_rank + k;
        if (additive_v_momentum.size() <= idx) additive_v_momentum.resize(idx + 1, value_type(0));
        return additive_v_momentum[idx];
    }
    inline value_type& get_additive_v_state_k(std::size_t col, std::size_t k) {
        const std::size_t idx = col * additive_rank + k;
        if (additive_v_state.size() <= idx) additive_v_state.resize(idx + 1, value_type(0));
        return additive_v_state[idx];
    }
    inline uint32_t& get_additive_v_step_k(std::size_t col, std::size_t k) {
        const std::size_t idx = col * additive_rank + k;
        if (additive_v_step.size() <= idx) additive_v_step.resize(idx + 1, 0);
        return additive_v_step[idx];
    }

    // A[row,col] = sum_{k<additive_rank} additive_u_k(row,k) *
    // additive_v_k(col,k) -- materializing a single entry directly, for
    // tests/small-scale callers. Real forward/backward paths (task #276/
    // #277) MUST use the fused Theorem-11 form (project X down to rank
    // additive_rank via additive_u, scale, project back up via
    // additive_v) instead of ever calling this per-entry across a whole
    // matrix -- that would defeat the entire point of the low-rank
    // representation.
    inline value_type get_additive(std::size_t row, std::size_t col) const {
        value_type a = value_type(0);
        for (std::size_t k = 0; k < additive_rank; ++k)
            a += get_additive_u_k(row, k) * get_additive_v_k(col, k);
        return a;
    }

    // ── Safe rank resize (multiplicative AND additive) ──────────────────────
    // CONFIRMED BUG this fixes (see conversation): naively assigning
    // scale_rank = new_rank does NOT reshuffle the existing flat
    // row*old_rank+k storage. If any row beyond row 0 already has live
    // data, changing scale_rank silently REINTERPRETS existing entries at
    // the wrong flat index (e.g. rank 1->2: what was row 1's rank-0 value
    // at flat index 1 becomes row 0's rank-1 value under the new
    // indexing). No existing caller ever hit this because scale_rank was
    // always set exactly once at construction, before any row was
    // touched -- but task #273's dynamic rank control needs to resize a
    // LIVE, already-trained layer, so this must be fixed for real, not
    // left as a footgun. Handles both the multiplicative arrays
    // (value_scale/value_scale_importance/value_scale_step/
    // value_scale_momentum, output_scale/output_scale_importance/
    // output_scale_step) and the new additive arrays (additive_u,
    // additive_v) with one shared reshuffle helper rather than
    // hand-duplicating the same logic eight times.
public:
    // Runtime-settable policy cap (task #295 -- was a compile-time
    // SCALE_RANK_MAX=4 constant forced by block4's SIMD backward path's
    // OWN fixed-size stack arrays; those are gone now, replaced by
    // scale_rank_scratch's heap buffers above, which grow to fit
    // whatever rank is actually requested with no hardcoded ceiling).
    // This field is a separate, independent concern: a POLICY limit on
    // how far scale_rank is allowed to grow (manually via set_scale_rank,
    // or automatically via apply_dynamic_rank_control's neurogenesis
    // trigger), not a memory-safety one. Default 4 matches the old
    // compile-time constant's value, preserving existing behavior for
    // any caller that never touches this.
    std::size_t scale_rank_max = 4;
    std::size_t additive_rank_max = 4;
    inline std::size_t get_scale_rank_max() const { return scale_rank_max; }
    // Lowering scale_rank_max below the CURRENT scale_rank is allowed --
    // it just means no further growth until scale_rank is manually
    // shrunk back under the new cap (apply_dynamic_rank_control's own
    // apoptosis path already handles shrinking independently of this).
    inline void set_scale_rank_max(std::size_t new_max) { scale_rank_max = new_max; }
    inline std::size_t get_additive_rank_max() const { return additive_rank_max; }
    inline void set_additive_rank_max(std::size_t new_max) { additive_rank_max = new_max; }

    inline void set_scale_rank(std::size_t new_rank) {
        if (new_rank == 0) throw std::invalid_argument("scale_rank must be >= 1");
        if (new_rank > scale_rank_max)
            throw std::invalid_argument("scale_rank exceeds scale_rank_max (the configured policy cap -- raise it via set_scale_rank_max first)");
        const std::size_t old_rank = scale_rank;
        auto scale_default  = [](std::size_t k) { return k == 0 ? value_type(1) : value_type(0); };
        auto zero_default   = [](std::size_t)   { return value_type(0); };
        auto step_default   = [](std::size_t)   { return uint32_t(0); };
        reshuffle_rank_array(value_scale, old_rank, new_rank, scale_default);
        reshuffle_rank_array(value_scale_importance, old_rank, new_rank, zero_default);
        reshuffle_rank_array(value_scale_step, old_rank, new_rank, step_default);
        reshuffle_rank_array(value_scale_momentum, old_rank, new_rank, zero_default);
        reshuffle_rank_array(output_scale, old_rank, new_rank, scale_default);
        reshuffle_rank_array(output_scale_importance, old_rank, new_rank, zero_default);
        reshuffle_rank_array(output_scale_step, old_rank, new_rank, step_default);
        // scale_gamma is a flat length-scale_rank array (one scalar per
        // channel, not per row/col) -- reshuffle_rank_array's own
        // n_entities computation reduces to exactly 1 "entity" here since
        // scale_gamma.size() never exceeds old_rank, so reusing it is
        // correct, not a hack. Uses zero_default, NOT scale_default (unlike
        // value_scale/output_scale above) -- a genuinely new gamma channel
        // (this branch only fires when scale_gamma is already non-empty,
        // i.e. gamma is already in active use -- see get_scale_gamma_k's
        // own docstring) should start at 0 regardless of k, matching
        // Theorem 9's "new channel = zero contribution" property. No
        // k==0-is-special case here: unlike value_scale/output_scale,
        // gamma's OWN "transparent by default" behavior comes entirely
        // from get_scale_gamma_k's lazy fallback (1.0), not from this
        // reshuffle, which only ever runs once gamma is already live.
        reshuffle_rank_array(scale_gamma, old_rank, new_rank, zero_default);
        reshuffle_rank_array(scale_gamma_state, old_rank, new_rank, zero_default);
        reshuffle_rank_array(scale_gamma_step, old_rank, new_rank, step_default);
        scale_gamma_ema.reshuffle(old_rank, new_rank);
        scale_rank = new_rank;
    }

    // AQRS dynamic rank control (task #273/#285): set_scale_rank's own
    // reshuffle can only SHRINK by truncating the highest-index channel
    // (k>=new_rank is simply dropped) -- but Theorem 10's apoptosis
    // trigger can fire on ANY channel, not just the last one. Swap the
    // dying channel to the end first, THEN call set_scale_rank(rank-1) to
    // truncate it -- this is the general "remove an arbitrary channel"
    // primitive, reused by whatever drives real apoptosis.
    //
    // Uses the existing get_*/set_*_raw_k accessors (not raw vector
    // indexing) specifically so lazy-unpopulated rows/cols are read via
    // their correct defaults and force-written, rather than silently
    // skipped -- a plain vector swap would corrupt any row/col that
    // hadn't been touched yet at one of the two indices.
    inline void swap_scale_channels(std::size_t k1, std::size_t k2, std::size_t n_rows, std::size_t n_cols) {
        if (k1 == k2) return;
        for (std::size_t r = 0; r < n_rows; ++r) {
            const value_type a = get_value_scale_k(r, k1);
            const value_type b = get_value_scale_k(r, k2);
            set_value_scale_raw_k(r, k1, b);
            set_value_scale_raw_k(r, k2, a);
        }
        for (std::size_t c = 0; c < n_cols; ++c) {
            const value_type a = get_output_scale_k(c, k1);
            const value_type b = get_output_scale_k(c, k2);
            set_output_scale_raw_k(c, k1, b);
            set_output_scale_raw_k(c, k2, a);
        }
        const value_type ga = get_scale_gamma_k(k1);
        const value_type gb = get_scale_gamma_k(k2);
        set_scale_gamma_raw_k(k1, gb);
        set_scale_gamma_raw_k(k2, ga);
        // EMA state travels WITH the channel -- swapping gamma's value
        // without its EMA history would make the (now-relocated) channel
        // look freshly-born to the trigger logic, defeating the whole
        // point of EMA smoothing being a real noise filter.
        scale_gamma_ema.swap_k(k1, k2);
        // scale_gamma_step doubles as the channel's AGE (see
        // apply_dynamic_rank_control's own grace-period comment) -- must
        // travel with the channel too, or a relocated channel keeps its
        // OLD position's age instead of its own, corrupting the grace
        // period exactly like a missed EMA swap would corrupt the trigger
        // signal itself.
        {
            const std::size_t need = std::max(k1, k2) + 1;
            if (scale_gamma_step.size() < need) scale_gamma_step.resize(need, 0);
            std::swap(scale_gamma_step[k1], scale_gamma_step[k2]);
        }
    }

    // Additive-branch counterpart to swap_scale_channels above (task
    // #289) -- same reasoning throughout, just additive_u/additive_v/
    // additive_gamma/additive_gamma_ema/additive_gamma_step instead of
    // value_scale/output_scale/scale_gamma/scale_gamma_ema/
    // scale_gamma_step. Needed because apply_additive_dynamic_rank_
    // control's apoptosis can target ANY channel, but set_additive_rank
    // can only truncate the LAST one.
    inline void swap_additive_channels(std::size_t k1, std::size_t k2, std::size_t n_rows, std::size_t n_cols) {
        if (k1 == k2) return;
        for (std::size_t r = 0; r < n_rows; ++r) {
            const value_type a = get_additive_u_k(r, k1);
            const value_type b = get_additive_u_k(r, k2);
            set_additive_u_raw_k(r, k1, b);
            set_additive_u_raw_k(r, k2, a);
        }
        for (std::size_t c = 0; c < n_cols; ++c) {
            const value_type a = get_additive_v_k(c, k1);
            const value_type b = get_additive_v_k(c, k2);
            set_additive_v_raw_k(c, k1, b);
            set_additive_v_raw_k(c, k2, a);
        }
        const value_type ga = get_additive_gamma_k(k1);
        const value_type gb = get_additive_gamma_k(k2);
        set_additive_gamma_raw_k(k1, gb);
        set_additive_gamma_raw_k(k2, ga);
        additive_gamma_ema.swap_k(k1, k2);
        {
            const std::size_t need = std::max(k1, k2) + 1;
            if (additive_gamma_step.size() < need) additive_gamma_step.resize(need, 0);
            std::swap(additive_gamma_step[k1], additive_gamma_step[k2]);
            if (additive_gamma_state.size() < need) additive_gamma_state.resize(need, value_type(0));
            std::swap(additive_gamma_state[k1], additive_gamma_state[k2]);
        }
    }

    // Shared control-flow for Theorem 10's apoptosis/neurogenesis dynamic
    // rank control (task #289), used by BOTH apply_dynamic_rank_control
    // (multiplicative branch) and apply_additive_dynamic_rank_control
    // (additive branch) below -- the DECISION logic (grace-period-gated
    // apoptose-else-neurogenesis, at-most-one-mutation-per-call) is
    // IDENTICAL between branches; only the actual mutation operations
    // differ (different swap/resize/seed calls per branch, and a
    // different min_rank floor -- scale_rank can never legally drop
    // below 1, additive_rank legitimately floors at 0, fully off), so
    // those are passed in as callbacks rather than duplicating this
    // control flow a second time. Extracted per direct instruction not
    // to copy apply_dynamic_rank_control's own logic when adding the
    // additive-branch equivalent.
    template <typename AgeFn, typename ApoptoseCheckFn, typename NeurogenesisCheckFn,
              typename DoApoptoseFn, typename DoNeurogenesisFn>
    // calls_since_mutation: symmetric branch-level cooldown (task #292
    // fix -- direct user instruction, "age-gate is good for now, expose
    // the option"). The per-channel age_of() gate below already protects
    // a freshly-grown channel from being immediately apoptosed (see this
    // function's callers' own docstrings for the original bug that
    // fixed), but nothing previously stopped the SYMMETRIC case: a real
    // 60k-step MQAR run showed gamma's own gradient can be large enough
    // to jump the tau_death/tau_active hysteresis gap in a single step,
    // so a channel could apoptose then immediately regrow (or vice
    // versa) every few calls -- 1464 mutations in 3000 steps, observed
    // directly. Gating BOTH apoptosis and neurogenesis behind "at least
    // grace_period_steps calls since the LAST mutation of either kind"
    // gives every mutation a real minimum window to matter before the
    // branch can change again, regardless of how large gamma's raw
    // gradient turns out to be. Passed by reference and owned by the
    // caller (scale_rank_calls_since_mutation/additive_rank_calls_since_
    // mutation) since scale and additive branches cool down
    // independently. Initial value UINT32_MAX (not 0) so a freshly
    // constructed layer's first-ever qualifying mutation isn't blocked
    // by a phantom cooldown; guarded against wraparound since a long
    // idle run would otherwise increment past UINT32_MAX.
    static bool apply_dynamic_rank_control_generic(std::size_t rank, std::size_t min_rank,
                                                     std::size_t max_rank, uint32_t grace_period_steps,
                                                     uint32_t& calls_since_mutation,
                                                     AgeFn age_of, ApoptoseCheckFn should_apoptose,
                                                     NeurogenesisCheckFn should_neurogenesis,
                                                     DoApoptoseFn do_apoptose, DoNeurogenesisFn do_neurogenesis) {
        if (calls_since_mutation < UINT32_MAX) ++calls_since_mutation;
        if (calls_since_mutation < grace_period_steps) return false;
        for (std::size_t k = 0; k < rank; ++k) {
            if (rank > min_rank && age_of(k) >= grace_period_steps && should_apoptose(k)) {
                do_apoptose(k);
                calls_since_mutation = 0;
                return true;
            }
        }
        if (rank < max_rank && should_neurogenesis()) {
            do_neurogenesis();
            calls_since_mutation = 0;
            return true;
        }
        return false;
    }

    // Evaluates Theorem 10's triggers against the CURRENT EMA state
    // (updated automatically every disldo_backward call, see task #284)
    // and performs at most ONE real mutation per call: apoptose the first
    // dying channel found (swap-to-end then shrink), or grow one new
    // channel if neurogenesis fires and no channel is currently dying.
    // ONE mutation per call, not "handle everything in one pass" --
    // apoptosis and neurogenesis firing on the SAME call would mean the
    // signal that triggered growth was measured against a rank about to
    // change anyway; simpler and safer to let the next call re-evaluate
    // against the post-mutation state.
    //
    // new_channel_seed(row) -- Theorem 9 says a new channel's direction
    // should align with the residual's top singular vector; AQRS_DESIGN.md
    // marks the practical proxy for this (neuron_grad_accum/importance)
    // as UNRESOLVED, not yet verified. This function deliberately does NOT
    // hardcode that unverified proxy -- it takes the new channel's
    // per-row direction as a caller-supplied callback instead, so a
    // caller can pass real residual-aligned values once Theorem 9's proxy
    // is validated, or (as every existing test in this codebase already
    // does for growth) a simple deterministic nonzero seed just to break
    // the symmetric zero-init deadlock in the meantime. output_scale's
    // side is seeded uniformly (1.0) -- no col-side residual signal is
    // available at this layer of the API either way.
    //
    // grace_period_steps: CORRECTED (real bug, found via a direct
    // integration test -- see conversation): the hysteresis gap
    // (tau_death < tau_active) only solves "a channel regrowing the
    // instant it's pruned" -- it does NOT solve the SYMMETRIC problem, "a
    // channel being pruned the instant it's grown." A freshly-grown
    // channel's gamma starts at exactly 0 (Theorem 9's own "zero
    // contribution" property), so its EMA also starts at ~0 -- which
    // trivially satisfies apoptosis's own (|gamma|_ema<tau_death AND
    // C_ema<tau_death) condition before the channel has had ANY chance to
    // train. Confirmed directly: without this gate, growth and apoptosis
    // fired on ALTERNATING steps forever, never letting a new channel
    // survive long enough to learn anything. Fix: a channel is only
    // ELIGIBLE for apoptosis once its own age (scale_gamma_step, which
    // already increments once per backward call as an Adam-style bias-
    // correction counter -- reused here as a free age signal, not a new
    // field) exceeds grace_period_steps. Default ~1/(1-0.98), matching
    // the EMA's own natural warm-up window at the default decay=0.98.
    template <typename SeedFn>
    inline bool apply_dynamic_rank_control(std::size_t n_rows, std::size_t n_cols,
                                            value_type tau_death, value_type tau_active,
                                            value_type theta, SeedFn new_channel_seed,
                                            uint32_t grace_period_steps = 50) {
        return apply_dynamic_rank_control_generic(
            scale_rank, /*min_rank=*/std::size_t(1), scale_rank_max, grace_period_steps,
            scale_rank_calls_since_mutation,
            [&](std::size_t k) { return k < scale_gamma_step.size() ? scale_gamma_step[k] : uint32_t(0); },
            [&](std::size_t k) { return scale_gamma_should_apoptose(k, tau_death); },
            [&]() { return scale_gamma_should_neurogenesis(scale_rank, tau_active, theta); },
            [&](std::size_t k) {
                swap_scale_channels(k, scale_rank - 1, n_rows, n_cols);
                set_scale_rank(scale_rank - 1);
            },
            [&]() {
                const std::size_t new_k = scale_rank;
                set_scale_rank(scale_rank + 1);
                for (std::size_t r = 0; r < n_rows; ++r) set_value_scale_raw_k(r, new_k, new_channel_seed(r));
                for (std::size_t c = 0; c < n_cols; ++c) set_output_scale_raw_k(c, new_k, value_type(1));
            });
    }

    // Additive-branch counterpart to apply_dynamic_rank_control above
    // (task #289/#292) -- same Theorem 10 trigger machinery (via
    // apply_dynamic_rank_control_generic), applied to additive_gamma/
    // additive_u/additive_v instead of scale_gamma/value_scale/
    // output_scale. Two seed callbacks (not one + a uniform 1.0 like the
    // multiplicative branch's own output_scale convention) because BOTH
    // additive_u and additive_v need a real nonzero direction for a new
    // channel to generate any gradient at all -- see _seed_additive_rank's
    // own docstring (sili/sparse_rnn.py) for the identical reasoning
    // applied to construction-time seeding. min_rank=0 (not 1): the
    // additive branch has no legacy always-on component to preserve
    // (additive_rank itself already defaults to 0, fully opt-in), so
    // apoptosis is free to shrink it all the way back off.
    template <typename SeedUFn, typename SeedVFn>
    inline bool apply_additive_dynamic_rank_control(std::size_t n_rows, std::size_t n_cols,
                                                     value_type tau_death, value_type tau_active,
                                                     value_type theta, SeedUFn new_channel_seed_u,
                                                     SeedVFn new_channel_seed_v,
                                                     uint32_t grace_period_steps = 50) {
        return apply_dynamic_rank_control_generic(
            additive_rank, /*min_rank=*/std::size_t(0), additive_rank_max, grace_period_steps,
            additive_rank_calls_since_mutation,
            [&](std::size_t k) { return k < additive_gamma_step.size() ? additive_gamma_step[k] : uint32_t(0); },
            [&](std::size_t k) { return additive_gamma_should_apoptose(k, tau_death); },
            [&]() { return additive_gamma_should_neurogenesis(additive_rank, tau_active, theta); },
            [&](std::size_t k) {
                swap_additive_channels(k, additive_rank - 1, n_rows, n_cols);
                set_additive_rank(additive_rank - 1);
            },
            [&]() {
                const std::size_t new_k = additive_rank;
                set_additive_rank(additive_rank + 1);
                for (std::size_t r = 0; r < n_rows; ++r) set_additive_u_raw_k(r, new_k, new_channel_seed_u(r));
                for (std::size_t c = 0; c < n_cols; ++c) set_additive_v_raw_k(c, new_k, new_channel_seed_v(c));
            });
    }

    inline void set_additive_rank(std::size_t new_rank) {
        // 0 is a valid, meaningful value here (branch fully disabled) --
        // unlike scale_rank, which must stay >= 1 since component 0 IS
        // the original rank-1 behavior every existing caller depends on.
        // additive_rank_max: runtime policy cap (task #295), same
        // reasoning as scale_rank_max's own guard in set_scale_rank
        // above -- no longer about a fixed-size stack array (dgamma_by_k
        // is a plain std::vector now, see disldo_backward), purely a
        // configurable growth ceiling.
        if (new_rank > additive_rank_max)
            throw std::invalid_argument("additive_rank exceeds additive_rank_max (the configured policy cap -- raise it via set_additive_rank_max first)");
        const std::size_t old_rank = additive_rank;
        auto zero_default = [](std::size_t) { return value_type(0); };
        auto step_default = [](std::size_t)  { return uint32_t(0); };
        reshuffle_rank_array(additive_u, old_rank, new_rank, zero_default);
        reshuffle_rank_array(additive_v, old_rank, new_rank, zero_default);
        reshuffle_rank_array(additive_u_momentum, old_rank, new_rank, zero_default);
        reshuffle_rank_array(additive_u_state, old_rank, new_rank, zero_default);
        reshuffle_rank_array(additive_u_step, old_rank, new_rank, step_default);
        reshuffle_rank_array(additive_v_momentum, old_rank, new_rank, zero_default);
        reshuffle_rank_array(additive_v_state, old_rank, new_rank, zero_default);
        reshuffle_rank_array(additive_v_step, old_rank, new_rank, step_default);
        reshuffle_rank_array(additive_gamma, old_rank, new_rank, zero_default);
        reshuffle_rank_array(additive_gamma_state, old_rank, new_rank, zero_default);
        reshuffle_rank_array(additive_gamma_step, old_rank, new_rank, step_default);
        additive_gamma_ema.reshuffle(old_rank, new_rank);
        additive_rank = new_rank;
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
            // Plain set(), not live -- this re-encodes whatever value the
            // row ALREADY had under a new scale (reparametrization, same
            // as block4_maybe_promote), not a training update. A row can
            // contain a freshly-grown, never-yet-trained synapse whose
            // weight/importance is deliberately 0 (insert_col's own
            // convention); redirecting that to a nonzero live code here
            // would be the same corruption class the block4_maybe_promote
            // regression already caught -- see its comment in
            // delta_csr_memory.hpp for the full incident.
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

    // Gradient-free reparametrization: true_weight = stored_w *
    // value_scale[row] * output_scale[col] is algebraically UNCHANGED by
    // this -- only WHERE the magnitude lives moves, from output_scale
    // into the stored per-synapse weight code. Drives each column's
    // stored-weight RMS (across all n_in rows, including rows with no
    // synapse in that column -- matches a dense parameter's zero-padded
    // mean, see below) toward `target` via a DAMPED (correction_rate)
    // multiplicative step per call rather than jumping there in one shot.
    // Ported from sili_peridot's torch-validated prototype
    // (toy_tile_recurrence_rmt_torch.py's _magnitude_rescale) -- see that
    // module for the full derivation and the empirical finding that
    // column-only (not also row/value_scale -- "both axes" was tested and
    // found to consistently HURT) is the winning configuration.
    //
    // scale_invariant: when true, per-synapse `ci` already tracks the RAW
    // gradient g (decoupled from S=value_scale*output_scale via
    // update_cw's own scale_invariant flag) so it does NOT need rescaling
    // here. When false, ci is calibrated to (g*S)^2 -- shrinking
    // output_scale by k without correspondingly rescaling ci silently
    // changes every touched synapse's effective RMSprop step size. See
    // update_cw's own docstring for the matching root cause on the
    // per-synapse weight update side.
    //
    // Column RMS is measured over n_in (the row COUNT), not nnz_in_col --
    // a column with zero active synapses is skipped entirely (k=1, no-op)
    // rather than treated as a real all-zero column, since "no synapse"
    // (sparse) and "synapse present but currently zero" (torch's dense
    // w_stored) are genuinely different things the sparse engine has no
    // reason to conflate; a torch all-zero-but-present column would
    // otherwise also degenerate toward the eps floor.
    //
    // Covers BOTH storages -- scattered CSR (`connections`) AND block4
    // (`block4`) -- not scattered-only. A real training layer promotes
    // synapses between the two continuously (synaptogenesis/pruning), so
    // a column's live weight can live in either storage, or split across
    // both, at any given moment; rescaling only one side would silently
    // leave the other side's synapses un-rescaled while still dividing
    // the SHARED output_scale[col] they both read, corrupting their true
    // weight. Both FP4 (Block4Store, nibble-packed weight|imp<<4) and
    // FP8 (Block4Store8, separate weight/importance byte planes) are
    // handled via the same `if constexpr` dispatch process_tile-style
    // code elsewhere in this codebase already uses -- see delta_csr_
    // memory.hpp's own scattered+block4 combined-export loop for the
    // read-side precedent this mirrors. Re-quantization here is
    // DETERMINISTIC (fp4_quantize/fp8_quantize), matching rescale_
    // value_row's own convention for this class of scale-bookkeeping
    // rewrite (not the gradient-driven stochastic set_stochastic()).
    inline void magnitude_rescale_output(value_type target, value_type correction_rate,
                                          bool scale_invariant, value_type eps = value_type(1e-8)) {
        auto& dc = connections;
        auto& L = dc.layout;
        const std::size_t n_out = L.cols;
        const std::size_t n_in  = L.rows;
        if (n_out == 0 || n_in == 0) return;

        std::vector<double> sum_sq(n_out, 0.0);
        std::vector<std::size_t> col_count(n_out, 0);
        for (std::size_t r = 0; r < n_in; ++r) {
            const std::size_t n = L.row_nnz(r);
            if (n == 0) continue;
            auto cursor = dc.row_cursor(r);
            for (std::size_t e = 0; e < n; ++e) {
                const COL_TYPE col = cursor.advance();
                const std::size_t vb = L.elem_start[r] + e;
                const value_type w = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                sum_sq[col] += static_cast<double>(w) * static_cast<double>(w);
                ++col_count[col];
            }
        }
        const auto& BL = block4.block_layout;
        for (std::size_t br = 0; br < BL.rows; ++br) {
            const std::size_t n_bc = BL.row_nnz(br);
            if (n_bc == 0) continue;
            auto bc_cursor = block4.row_cursor(uint32_t(br));
            for (std::size_t bk = 0; bk < n_bc; ++bk) {
                const uint32_t bc = bc_cursor.advance();
                const auto tile = block4.find(uint32_t(br), bc);
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                    if (col >= n_out) continue;
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = br * BLOCK4_TILE + li;
                        if (row >= n_in) continue;
                        value_type w;
                        if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                            const uint8_t w_byte = tile.at_weight(li, lj);
                            const uint8_t i_byte = tile.at_importance(li, lj);
                            if (w_byte == 0 && i_byte == 0) continue;  // empty slot
                            w = fp8_decode_bits(w_byte);
                        } else {
                            const uint8_t byte = tile.at(li, lj);
                            if (byte == 0) continue;  // empty slot
                            w = FP4_TABLE[byte & 0xFu];
                        }
                        sum_sq[col] += static_cast<double>(w) * static_cast<double>(w);
                        ++col_count[col];
                    }
                }
            }
        }

        std::vector<value_type> k(n_out, value_type(1));
        for (std::size_t c = 0; c < n_out; ++c) {
            if (col_count[c] == 0) continue;  // nothing to rescale here
            const double mean_sq = sum_sq[c] / static_cast<double>(n_in);
            const value_type col_rms = static_cast<value_type>(std::sqrt(mean_sq + static_cast<double>(eps)));
            if (!std::isfinite(col_rms) || col_rms <= value_type(0)) continue;
            value_type kc = target / col_rms;
            if (kc < value_type(1e-6)) kc = value_type(1e-6);
            kc = std::pow(kc, correction_rate);
            if (!std::isfinite(kc) || kc <= value_type(0)) continue;
            k[c] = kc;
        }

        for (std::size_t r = 0; r < n_in; ++r) {
            const std::size_t n = L.row_nnz(r);
            if (n == 0) continue;
            auto cursor = dc.row_cursor(r);
            for (std::size_t e = 0; e < n; ++e) {
                const COL_TYPE col = cursor.advance();
                if (k[col] == value_type(1)) continue;
                const std::size_t vb = L.elem_start[r] + e;
                const value_type w   = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                const value_type imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                const value_type new_w   = w * k[col];
                const value_type new_imp = scale_invariant ? imp : imp * k[col] * k[col];
                if (!std::isfinite(new_w) || !std::isfinite(new_imp)) continue;
                ValueAccessor<VALUES_TYPE>::set_live(dc.values, vb, new_w, new_imp);
            }
        }
        for (std::size_t br = 0; br < BL.rows; ++br) {
            const std::size_t n_bc = BL.row_nnz(br);
            if (n_bc == 0) continue;
            auto bc_cursor = block4.row_cursor(uint32_t(br));
            for (std::size_t bk = 0; bk < n_bc; ++bk) {
                const uint32_t bc = bc_cursor.advance();
                // Any column in this tile need rescaling? Skip the whole
                // tile (no mutable handle, no dirty/re-pack cost) if not.
                bool any_col_touched = false;
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                    if (col < n_out && k[col] != value_type(1)) { any_col_touched = true; break; }
                }
                if (!any_col_touched) continue;
                auto tile = block4.find(uint32_t(br), bc);
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                    if (col >= n_out || k[col] == value_type(1)) continue;
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = br * BLOCK4_TILE + li;
                        if (row >= n_in) continue;
                        if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                            const uint8_t w_byte = tile.at_weight(li, lj);
                            const uint8_t i_byte = tile.at_importance(li, lj);
                            if (w_byte == 0 && i_byte == 0) continue;
                            const value_type w   = fp8_decode_bits(w_byte);
                            const value_type imp = fp8_decode_bits(i_byte);
                            const value_type new_w   = w * k[col];
                            const value_type new_imp = scale_invariant ? imp : imp * k[col] * k[col];
                            if (!std::isfinite(new_w) || !std::isfinite(new_imp)) continue;
                            tile.at_weight(li, lj)     = fp8_quantize_live(new_w);
                            tile.at_importance(li, lj) = fp8_quantize_live(new_imp);
                        } else {
                            const uint8_t byte = tile.at(li, lj);
                            if (byte == 0) continue;
                            const value_type w   = FP4_TABLE[byte & 0xFu];
                            const value_type imp = FP4_TABLE[(byte >> 4) & 0xFu];
                            const value_type new_w   = w * k[col];
                            const value_type new_imp = scale_invariant ? imp : imp * k[col] * k[col];
                            if (!std::isfinite(new_w) || !std::isfinite(new_imp)) continue;
                            tile.at(li, lj) = uint8_t(fp4_quantize_live(new_w) | (fp4_quantize_live(new_imp) << 4));
                        }
                    }
                }
            }
        }

        // true_weight = stored_w * S[row,col], where S[row,col] =
        // sum_{ki<scale_rank} value_scale_k(row,ki)*output_scale_k(col,ki)
        // (get_scale's own formula). Dividing EVERY rank component's
        // output_scale_k(col,ki) by the SAME column-level k[c] divides
        // the whole sum by k[c] exactly (S/k[c] = sum_ki(vs_ki*(os_ki/
        // k[c])) = (sum_ki vs_ki*os_ki)/k[c]), so this generalizes
        // cleanly to any scale_rank -- at scale_rank==1 it's identical
        // to the original single-component form.
        for (std::size_t c = 0; c < n_out; ++c) {
            if (k[c] == value_type(1)) continue;
            for (std::size_t ki = 0; ki < scale_rank; ++ki) {
                const value_type new_os = get_output_scale_k(c, ki) / k[c];
                if (!std::isfinite(new_os)) continue;
                set_output_scale_raw_k(c, ki, new_os);
            }
        }
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
