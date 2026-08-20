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
                ValueAccessor<VALUES_TYPE>::set(dc.values, vb, new_w, new_imp);
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
                            tile.at_weight(li, lj)     = fp8_quantize(new_w);
                            tile.at_importance(li, lj) = fp8_quantize(new_imp);
                        } else {
                            const uint8_t byte = tile.at(li, lj);
                            if (byte == 0) continue;
                            const value_type w   = FP4_TABLE[byte & 0xFu];
                            const value_type imp = FP4_TABLE[(byte >> 4) & 0xFu];
                            const value_type new_w   = w * k[col];
                            const value_type new_imp = scale_invariant ? imp : imp * k[col] * k[col];
                            if (!std::isfinite(new_w) || !std::isfinite(new_imp)) continue;
                            tile.at(li, lj) = uint8_t(fp4_quantize(new_w) | (fp4_quantize(new_imp) << 4));
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
