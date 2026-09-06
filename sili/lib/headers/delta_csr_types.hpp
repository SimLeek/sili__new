/**
 * @file sparse_matrix.hpp
 * @brief Sparse matrix library with CSR and COO format support.
 */

#ifndef __DELTA_CSR_TYPES_HPP_
#define __DELTA_CSR_TYPES_HPP_

// Split out of sparse_struct.hpp to keep files under ~1k lines. Core type
// definitions only; free functions operating on these types are in
// delta_csr_memory.hpp and sisldo_ops.hpp -- see
// docs/research/delta_csr_types.rst (top of file) for the full layout
// rationale.

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
// block4.hpp is included further down, right before SparseLinearWeightsDelta
// -- see docs/research/delta_csr_types.rst (top of file) for why.

/**
 * @brief Type trait to check if a type is a std::array.
 * @tparam T The type to check.
 */
template <typename T> struct is_std_array : std::false_type {};

/**
 * @brief Specialization of is_std_array for std::array types.
 * @tparam T The element type of the array.
 * @tparam N The size of the array.
 */
template <typename T, std::size_t N> struct is_std_array<std::array<T, N>> : std::true_type {};

/**
 * @brief Helper variable template to check if a type is a std::array.
 * @tparam T The type to check.
 */
template <typename T> constexpr bool is_std_array_v = is_std_array<T>::value;

template <class SIZE_TYPE>
using CSRPointers = std::array<std::shared_ptr<std::vector<SIZE_TYPE>>, 1>;

template <class SIZE_TYPE>
using CSRIndices = std::array<std::shared_ptr<std::vector<SIZE_TYPE>>, 1>;

template <class SIZE_TYPE> using COOPointers = SIZE_TYPE; // just store nnz

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

template <class SIZE_TYPE, class PTRS, class INDICES, class VALUES> struct sparse_struct {
    PTRS ptrs;       // Pointers sub-template
    INDICES indices; // Indices sub-template
    VALUES values;   // Values sub-template
    SIZE_TYPE rows;
    SIZE_TYPE cols;
    SIZE_TYPE _reserved_space = 0;

    using size_type = SIZE_TYPE; // Exporting the type

    static constexpr std::size_t n_index_arrays = num_indices<INDICES>;
    static constexpr std::size_t n_value_arrays = num_indices<VALUES>;
    static constexpr std::size_t n_pointer_arrays = num_indices<PTRS>;

    /**
     * @brief Default constructor, initializes an empty sparse matrix.
     */
    // (false positive: ptrs/indices/values are class types -- e.g.
    // std::array<std::shared_ptr<...>> -- with their own default ctors, so
    // they're safely default-constructed even though this ctor's own
    // init-list only lists the POD SIZE_TYPE members.)
    // cppcheck-suppress uninitMemberVar
    sparse_struct() : rows(0), cols(0), _reserved_space(0) {}

    /**
     * @brief Constructor for pre-allocated arrays with reserved space.
     * @param p Pointers sub-template (moved into the structure).
     * @param ind Indices sub-template (moved into the structure).
     * @param val Values sub-template (moved into the structure).
     * @param num_p Number of rows.
     * @param max_idx Number of columns.
     * @param reserved Reserved space for future expansion.
     */
    sparse_struct(PTRS& p, INDICES& ind, VALUES& val, SIZE_TYPE num_p, SIZE_TYPE max_idx,
                  SIZE_TYPE reserved)
        : ptrs(std::move(p)), indices(std::move(ind)), values(std::move(val)), rows(num_p),
          cols(max_idx), _reserved_space(reserved) {}

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
        if constexpr (std::is_array_v<decltype(ptrs)> ||
                      is_std_array_v<decltype(ptrs)>) { // Check if ptrs is an array type
            return (ptrs[ptrs.size() - 1] && !ptrs[ptrs.size() - 1]->empty())
                       ? (*ptrs[ptrs.size() - 1])[rows]
                       : 0;
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
using CSRSynapses =
    sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, BiValuesFP4>;
// easier to use in some algorithms
template <class SIZE_TYPE>
using COOSynapses =
    sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, BiValuesFP4>;

template <class SIZE_TYPE, class VALUE_TYPE>
using CSRSynapsesV =
    sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, BiValues<VALUE_TYPE>>;
// easier to use in some algorithms
template <class SIZE_TYPE, class VALUE_TYPE>
using COOSynapsesV =
    sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, BiValues<VALUE_TYPE>>;

template <class SIZE_TYPE, class VALUE_TYPE>
using CSRInput = sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>,
                               UnaryValues<VALUE_TYPE>>;

// easier to use in some algorithms
template <class SIZE_TYPE, class VALUE_TYPE>
using COOSynaptogenesis = sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>,
                                        UnaryValues<VALUE_TYPE>>;

template <class SYNAPSES, class SYNAPTOGENESIS> struct sparse_weights {
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
using SparseLinearWeights =
    sparse_weights<CSRSynapses<SIZE_TYPE>, COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>>;
template <class SIZE_TYPE, class VALUE_TYPE>
using SparseLinearWeightsV =
    sparse_weights<CSRSynapsesV<SIZE_TYPE, VALUE_TYPE>, COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>>;

// Delta CSR section

/// Maximum bytes to encode an integer as ULEB128.
// fake ULEB128, but in practice we're not going to have more than 2^28 zeroes between items in a
// single row
template <typename T = uint32_t> constexpr std::size_t uleb128_max_bytes() {
    return (sizeof(T) * 8 + 6) / 7;
}

/// Encode @p value into @p buf as ULEB128. Returns bytes written.
template <typename T = uint32_t> inline std::size_t uleb128_encode(T value, uint8_t* buf) {
    std::size_t n = 0;
    do {
        uint8_t byte = static_cast<uint8_t>(value & 0x7Fu);
        value >>= 7;
        if (value)
            byte |= 0x80u;
        buf[n++] = byte;
    } while (value);
    return n;
}

/// Decode one ULEB128 value from @p buf at byte offset *pos. Advances *pos.
template <typename T = uint32_t> inline T uleb128_decode(const uint8_t* buf, std::size_t& pos) {
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

template <typename V, typename = void> struct ValueAccessor;

/// Trait to handle FP4BiPacked natively
template <> struct ValueAccessor<FP4BiPacked> {
    using value_type = float;
    static value_type get_w(const FP4BiPacked& v, std::size_t i) { return v[0][i]; }
    static value_type get_imp(const FP4BiPacked& v, std::size_t i) { return v[1][i]; }
    static void set(FP4BiPacked& v, std::size_t i, value_type w, value_type imp) {
        v[0][i] = w;
        v[1][i] = imp;
    }
    /// Gradient-driven update only. See
    /// docs/research/delta_csr_types.rst:value_accessor.live_variants.
    static void set_stochastic(FP4BiPacked& v, std::size_t i, value_type w, value_type imp) {
        v.set_stochastic(i, w, imp);
    }
    /// Same as set() but for a LIVE synapse (never-0 code). See
    /// docs/research/delta_csr_types.rst:value_accessor.live_variants.
    static void set_live(FP4BiPacked& v, std::size_t i, value_type w, value_type imp) {
        v.set_live(i, w, imp);
    }
    /// Same as set_stochastic() but for a LIVE synapse (weight+importance).
    static void set_stochastic_live(FP4BiPacked& v, std::size_t i, value_type w, value_type imp) {
        v.set_stochastic_live(i, w, imp);
    }
    static void reserve(FP4BiPacked& v, std::size_t n) { v.reserve(n); }
    static void resize(FP4BiPacked& v, std::size_t n, value_type val = 0.0f,
                       value_type imp = 0.0f) {
        v.resize(n, val, imp);
    }
    static void move(FP4BiPacked& v, std::size_t dest, std::size_t src, std::size_t count) {
        if (count == 0 || !v._data)
            return;
        std::memmove(v._data->data() + dest, v._data->data() + src, count);
    }

    static std::size_t projected_byte_size(std::size_t n) { return n; }

    static std::size_t size(const FP4BiPacked& v) { return v[0].size(); }
};

/// Trait to handle FP8BiValues -- one full byte per value (weight,
/// importance separately), OCP MX E4M3 codec (fp8quant.hpp). value_type
/// stays the DECODED float (matching ValueAccessor<FP4BiPacked>, not
/// DeltaCSRBiValues<T>'s raw-passthrough) since the byte array holds an
/// encoded code, not a usable float directly.
template <> struct ValueAccessor<FP8BiValues> {
    using value_type = float;
    static value_type get_w(const FP8BiValues& v, std::size_t i) {
        return fp8_decode_bits(v.weights[i]);
    }
    static value_type get_imp(const FP8BiValues& v, std::size_t i) {
        return fp8_decode_bits(v.importance[i]);
    }
    static void set(FP8BiValues& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = fp8_quantize(w);
        v.importance[i] = fp8_quantize(imp);
    }
    /// Gradient-driven update only. See
    /// docs/research/delta_csr_types.rst:value_accessor.live_variants.
    static void set_stochastic(FP8BiValues& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = fp8_quantize_stochastic(w);
        v.importance[i] = fp8_quantize_stochastic(imp);
    }
    /// Same as set() but for a LIVE synapse (never-0, both weight and
    /// importance). See
    /// docs/research/delta_csr_types.rst:value_accessor.live_variants.
    static void set_live(FP8BiValues& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = fp8_quantize_live(w);
        v.importance[i] = fp8_quantize_live(imp);
    }
    /// Same as set_stochastic() but for a LIVE synapse. Importance uses
    /// the _nonneg variant (always >= 0, feeds sqrt(ci)) -- see
    /// docs/research/delta_csr_types.rst:value_accessor.live_variants.
    static void set_stochastic_live(FP8BiValues& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = fp8_quantize_stochastic_live(w);
        v.importance[i] = fp8_quantize_stochastic_live_nonneg(imp);
    }
    static void resize(FP8BiValues& v, std::size_t n, value_type val = 0.0f,
                       value_type imp = 0.0f) {
        v.weights.resize(n, fp8_quantize(val));
        v.importance.resize(n, fp8_quantize(imp));
    }
    static void move(FP8BiValues& v, std::size_t dest, std::size_t src, std::size_t count) {
        if (count == 0)
            return;
        std::memmove(v.weights.data() + dest, v.weights.data() + src, count);
        std::memmove(v.importance.data() + dest, v.importance.data() + src, count);
    }
    static void reserve(FP8BiValues& v, std::size_t n) {
        v.weights.reserve(n);
        v.importance.reserve(n);
    }

    static std::size_t projected_byte_size(std::size_t n) {
        return n * 2; // 1 byte weight + 1 byte importance per element
    }

    static std::size_t size(const FP8BiValues& v) { return v.weights.size(); }
};

/// Fallback standard vector equivalent for floats (e.g. CSRSynapsesV uses)
template <typename T> struct DeltaCSRBiValues {
    std::vector<T> weights;
    std::vector<T> importance;
};

template <typename T> struct ValueAccessor<DeltaCSRBiValues<T>> {
    using value_type = T;
    static value_type get_w(const DeltaCSRBiValues<T>& v, std::size_t i) { return v.weights[i]; }
    static value_type get_imp(const DeltaCSRBiValues<T>& v, std::size_t i) {
        return v.importance[i];
    }
    static void set(DeltaCSRBiValues<T>& v, std::size_t i, value_type w, value_type imp) {
        v.weights[i] = w;
        v.importance[i] = imp;
    }
    /// No quantization for this float32-fallback storage -- passthrough
    /// to set(). See
    /// docs/research/delta_csr_types.rst:value_accessor.live_variants.
    static void set_stochastic(DeltaCSRBiValues<T>& v, std::size_t i, value_type w,
                               value_type imp) {
        set(v, i, w, imp);
    }
    /// No quantization for this float32-fallback storage -- passthrough
    /// to set(). See
    /// docs/research/delta_csr_types.rst:value_accessor.live_variants.
    static void set_live(DeltaCSRBiValues<T>& v, std::size_t i, value_type w, value_type imp) {
        set(v, i, w, imp);
    }
    static void set_stochastic_live(DeltaCSRBiValues<T>& v, std::size_t i, value_type w,
                                    value_type imp) {
        set(v, i, w, imp);
    }
    static void resize(DeltaCSRBiValues<T>& v, std::size_t n, value_type val = value_type(0),
                       value_type imp = value_type(0)) {
        v.weights.resize(n, val);
        v.importance.resize(n, imp);
    }
    static void move(DeltaCSRBiValues<T>& v, std::size_t dest, std::size_t src, std::size_t count) {
        if (count == 0)
            return;
        std::memmove(v.weights.data() + dest, v.weights.data() + src, count * sizeof(value_type));
        std::memmove(v.importance.data() + dest, v.importance.data() + src,
                     count * sizeof(value_type));
    }
    static void reserve(DeltaCSRBiValues<T>& v, std::size_t n) {
        v.weights.reserve(n);
        v.importance.reserve(n);
    }

    static std::size_t projected_byte_size(std::size_t n) { return n * sizeof(T) * 2; }

    static std::size_t size(const DeltaCSRBiValues<T>& v) { return v.weights.size(); }
};

// ── Amortized decoupled decay + running stats ───────────────────────────────
// Generic over VALUES_TYPE via ValueAccessor, touches `chunk_size` synapses
// per call via a persistent rolling cursor, decays WEIGHT only. See
// docs/research/delta_csr_types.rst:amortized_decay.chunked_cursor for the
// decay_factor derivation, the FP4/FP8-vs-fp32 rationale, and why
// cycle_complete gates whether the stats fields are meaningful.
struct AmortizedDecayStats {
    double mean_abs = 0.0;
    double rms = 0.0;
    double max_abs = 0.0;
    std::size_t n = 0;
    bool cycle_complete = false;
};

template <typename VALUES_TYPE, typename V>
AmortizedDecayStats apply_amortized_decay_stats(VALUES_TYPE& values, std::size_t& cursor,
                                                double& sum_abs, double& sum_sq, double& max_abs,
                                                std::size_t& n, std::size_t chunk_size,
                                                V decay_factor) {
    using VA = ValueAccessor<VALUES_TYPE>;
    const std::size_t total = VA::size(values);
    bool cycle_complete = false;
    if (total > 0) {
        for (std::size_t i = 0; i < chunk_size; ++i) {
            if (cursor >= total)
                cursor = 0;
            const V w = static_cast<V>(VA::get_w(values, cursor));
            const V imp = static_cast<V>(VA::get_imp(values, cursor));
            const V new_w = static_cast<V>(w * decay_factor);
            VA::set_live(values, cursor, new_w, imp);
            const double aw = std::abs(static_cast<double>(new_w));
            sum_abs += aw;
            sum_sq += aw * aw;
            if (aw > max_abs)
                max_abs = aw;
            ++n;
            ++cursor;
            if (cursor >= total) {
                cycle_complete = true;
                cursor = 0;
            }
        }
    } else {
        cycle_complete = true;
    }
    AmortizedDecayStats out;
    out.cycle_complete = cycle_complete;
    if (cycle_complete && n > 0) {
        out.mean_abs = sum_abs / static_cast<double>(n);
        out.rms = std::sqrt(sum_sq / static_cast<double>(n));
        out.max_abs = max_abs;
        out.n = n;
        sum_abs = 0.0;
        sum_sq = 0.0;
        max_abs = 0.0;
        n = 0;
    }
    return out;
}

// ── Scale-update policies ─────────────────────────────────────────────────────
// Swappable in-place optimizer for value_scale/output_scale (disldo_backward's
// scattered path, linear_disldo.hpp). Template parameter, not a runtime flag.
// See docs/research/delta_csr_types.rst:scale_policy.nan_inf_guard for the
// motivating comparison against a torch fake-quantize prototype.

template <typename VALUE_TYPE> struct RMSpropScalePolicy {
    // Extracted verbatim from disldo_backward's original inline formula
    // (bit-identical on finite inputs, the default policy) PLUS a NaN/Inf
    // guard -- see
    // docs/research/delta_csr_types.rst:scale_policy.nan_inf_guard for the
    // real dense-connectivity NaN-corruption bug this fixes.
    // contrib_agg: row/column-aggregated forward-contribution signal,
    // combined with g_agg via square-then-sum (not sum-then-square) --
    // see docs/research/delta_csr_types.rst:scale_policy.contrib_agg_combination.
    // step: Adam-style bias correction (Kingma & Ba 2015 sec 3) -- fixes a
    // real ~31.6x-oversized first step that caused a sign-flip regression.
    // See docs/research/delta_csr_types.rst:scale_policy.adam_bias_correction.
    // log_space: scale-invariant update mirroring update_cw's own
    // scale_invariant fix. See
    // docs/research/delta_csr_types.rst:scale_policy.log_space_variant.
    static void update(VALUE_TYPE& scale, VALUE_TYPE& scale_state, VALUE_TYPE g_agg,
                       VALUE_TYPE eff_lr, VALUE_TYPE beta2, VALUE_TYPE eps,
                       VALUE_TYPE contrib_agg = VALUE_TYPE(0), uint32_t* step = nullptr,
                       bool log_space = false) {
        if (!std::isfinite(g_agg) || !std::isfinite(contrib_agg))
            return;
        if (log_space) {
            const VALUE_TYPE log_grad = g_agg * scale;
            const VALUE_TYPE log_contrib = contrib_agg * scale;
            const VALUE_TYPE new_state =
                beta2 * scale_state +
                (VALUE_TYPE(1) - beta2) * (log_grad * log_grad + log_contrib * log_contrib);
            if (!std::isfinite(new_state))
                return;
            VALUE_TYPE state_hat = new_state;
            if (step != nullptr) {
                ++(*step);
                const VALUE_TYPE bias_correction =
                    VALUE_TYPE(1) - std::pow(beta2, static_cast<VALUE_TYPE>(*step));
                if (bias_correction > VALUE_TYPE(0))
                    state_hat = new_state / bias_correction;
            }
            if (!std::isfinite(state_hat))
                return;
            const VALUE_TYPE log_step = eff_lr * log_grad / (std::sqrt(state_hat) + eps);
            const VALUE_TYPE new_scale = scale * std::exp(-log_step);
            if (!std::isfinite(new_scale))
                return;
            scale_state = new_state;
            scale = new_scale;
            return;
        }
        const VALUE_TYPE new_state =
            beta2 * scale_state +
            (VALUE_TYPE(1) - beta2) * (g_agg * g_agg + contrib_agg * contrib_agg);
        if (!std::isfinite(new_state))
            return;
        VALUE_TYPE state_hat = new_state;
        if (step != nullptr) {
            ++(*step);
            const VALUE_TYPE bias_correction =
                VALUE_TYPE(1) - std::pow(beta2, static_cast<VALUE_TYPE>(*step));
            if (bias_correction > VALUE_TYPE(0))
                state_hat = new_state / bias_correction;
        }
        if (!std::isfinite(state_hat))
            return;
        const VALUE_TYPE new_scale = scale - eff_lr * g_agg / (std::sqrt(state_hat) + eps);
        if (!std::isfinite(new_scale))
            return;
        scale_state = new_state;
        scale = new_scale;
    }
};

template <typename VALUE_TYPE> struct AdaMaxScalePolicy {
    // AdaMax's own decayed running-max second moment (Kingma & Ba 2015,
    // sec 7): scale_state tracks max(beta2*scale_state, |g_agg|), no sqrt
    // needed. Same NaN/Inf guard as RMSpropScalePolicy::update, same
    // reason. contrib_agg combines via max(|g_agg|,|contrib_agg|), the
    // max-tracker's own analog of RMSprop's square-then-sum. step is
    // accepted for signature compatibility only (unused -- AdaMax's
    // running-max has no cold-start shrinkage to correct). See
    // docs/research/delta_csr_types.rst:scale_policy.adamax_design.
    static void update(VALUE_TYPE& scale, VALUE_TYPE& scale_state, VALUE_TYPE g_agg,
                       VALUE_TYPE eff_lr, VALUE_TYPE beta2, VALUE_TYPE eps,
                       VALUE_TYPE contrib_agg = VALUE_TYPE(0), uint32_t* step = nullptr,
                       bool log_space = false) {
        (void)step;
        (void)log_space; // AdaMax's L-infinity tracker has no log-space variant (yet) -- accepted
                         // for call-site signature compatibility with RMSpropScalePolicy only.
        if (!std::isfinite(g_agg) || !std::isfinite(contrib_agg))
            return;
        const VALUE_TYPE combined_mag = std::max(std::abs(g_agg), std::abs(contrib_agg));
        const VALUE_TYPE new_state = std::max(beta2 * scale_state, combined_mag);
        if (!std::isfinite(new_state))
            return;
        const VALUE_TYPE new_scale = scale - eff_lr * g_agg / (new_state + eps);
        if (!std::isfinite(new_scale))
            return;
        scale_state = new_state;
        scale = new_scale;
    }
};

// AQRS additive branch's default optimizer (task #277, see
// sili_peridot/AQRS_DESIGN.md) -- real Adam (Kingma & Ba 2015): RMSprop's
// second-moment tracking PLUS a genuine first-moment (momentum) EMA, used
// in the step's numerator instead of the raw gradient. NOT a drop-in
// replacement for RMSpropScalePolicy::update (needs its own
// momentum_state, and the second moment must use the RAW gradient while
// the numerator uses the momentum-smoothed one -- can't share one
// g_agg). See
// docs/research/delta_csr_types.rst:scale_policy.adam_additive_branch.
template <typename VALUE_TYPE> struct AdamScalePolicy {
    static void update(VALUE_TYPE& scale, VALUE_TYPE& scale_state, VALUE_TYPE& momentum_state,
                       VALUE_TYPE g_agg, VALUE_TYPE eff_lr, VALUE_TYPE beta1, VALUE_TYPE beta2,
                       VALUE_TYPE eps, uint32_t* step = nullptr) {
        if (!std::isfinite(g_agg))
            return;

        // First moment (momentum): m = beta1*m + (1-beta1)*g.
        const VALUE_TYPE new_momentum = beta1 * momentum_state + (VALUE_TYPE(1) - beta1) * g_agg;
        if (!std::isfinite(new_momentum))
            return;

        // Second moment (RMSprop-style), computed from the RAW gradient,
        // not the momentum-smoothed one, per standard Adam.
        const VALUE_TYPE new_state =
            beta2 * scale_state + (VALUE_TYPE(1) - beta2) * (g_agg * g_agg);
        if (!std::isfinite(new_state))
            return;

        VALUE_TYPE m_hat = new_momentum;
        VALUE_TYPE v_hat = new_state;
        if (step != nullptr) {
            ++(*step);
            const VALUE_TYPE t = static_cast<VALUE_TYPE>(*step);
            const VALUE_TYPE bc1 = VALUE_TYPE(1) - std::pow(beta1, t);
            const VALUE_TYPE bc2 = VALUE_TYPE(1) - std::pow(beta2, t);
            if (bc1 > VALUE_TYPE(0))
                m_hat = new_momentum / bc1;
            if (bc2 > VALUE_TYPE(0))
                v_hat = new_state / bc2;
        }
        if (!std::isfinite(m_hat) || !std::isfinite(v_hat))
            return;

        const VALUE_TYPE new_scale = scale - eff_lr * m_hat / (std::sqrt(v_hat) + eps);
        if (!std::isfinite(new_scale))
            return;
        momentum_state = new_momentum;
        scale_state = new_state;
        scale = new_scale;
    }
};

// No-op: scale/scale_state are never touched (true_w == stored_w always).
// A real ablation, not a stub -- see
// docs/research/delta_csr_types.rst:scale_policy.no_scale_policy.
template <typename VALUE_TYPE> struct NoScalePolicy {
    static void update(VALUE_TYPE& /*scale*/, VALUE_TYPE& /*scale_state*/, VALUE_TYPE /*g_agg*/,
                       VALUE_TYPE /*eff_lr*/, VALUE_TYPE /*beta2*/, VALUE_TYPE /*eps*/,
                       VALUE_TYPE /*contrib_agg*/ = VALUE_TYPE(0), uint32_t* /*step*/ = nullptr,
                       bool /*log_space*/ = false) {
        // Intentionally does nothing.
    }
};

// ── Per-synapse ci-update policy (floor + clip) ───────────────────────────────
// Distinct from the ScalePolicy family above: this operates on per-synapse
// `ci` (one scalar per SYNAPSE), historically hand-duplicated at ~8 call
// sites rather than templated. See
// docs/research/delta_csr_types.rst:synapse_policy.overview for the root
// cause this whole family exists to fix (a plain RMSprop ci EMA lags the
// true gradient scale near convergence, causing late-training resonance)
// and why an lr-decay schedule was rejected as incompatible with lifelong
// learning.
template <typename VALUE_TYPE> struct PlainRMSpropSynapsePolicy {
    // Bit-identical to the pre-fix inline formula on finite inputs -- kept
    // as the reference BoundedRMSpropSynapsePolicy (below, now the real
    // default) was checked against. NaN/Inf guard closes a coverage gap
    // an earlier commit (ba4af42) left. See
    // docs/research/delta_csr_types.rst:synapse_policy.plain_reference.
    static VALUE_TYPE update_ci(VALUE_TYPE ci, VALUE_TYPE g, VALUE_TYPE contrib, VALUE_TYPE beta2,
                                VALUE_TYPE /*min_decay_frac*/, VALUE_TYPE /*max_ci*/) {
        if (!std::isfinite(g) || !std::isfinite(contrib))
            return ci;
        const VALUE_TYPE new_ci =
            beta2 * ci + (VALUE_TYPE(1) - beta2) * (g * g + contrib * contrib);
        return std::isfinite(new_ci) ? new_ci : ci;
    }

    // Returns a DELTA (caller does `cw += update_cw(...)`). scale_invariant
    // fixes a real bug: the historical S-in-numerator formula makes
    // Delta(true_weight) scale QUADRATICALLY with S (fp32 accuracy
    // 1.0->0.18 measured once S drifted from 1.0). See
    // docs/research/delta_csr_types.rst:synapse_policy.scale_invariant_quadratic_bug.
    static VALUE_TYPE update_cw(VALUE_TYPE g, VALUE_TYPE ci, VALUE_TYPE S, VALUE_TYPE eff_lr,
                                VALUE_TYPE eps, bool damp_by_importance,
                                VALUE_TYPE /*max_abs_delta*/, bool scale_invariant = false) {
        if (!std::isfinite(g) || !std::isfinite(ci) || !std::isfinite(S))
            return VALUE_TYPE(0);
        VALUE_TYPE delta;
        if (scale_invariant) {
            const VALUE_TYPE raw = damp_by_importance ? (-g) / (std::sqrt(ci) + eps) : (-g);
            delta = std::isfinite(raw) ? (eff_lr * raw / S) : VALUE_TYPE(0);
        } else {
            delta =
                damp_by_importance ? (-eff_lr * g * S) / (std::sqrt(ci) + eps) : (-eff_lr * g * S);
        }
        return std::isfinite(delta) ? delta : VALUE_TYPE(0);
    }
};

template <typename VALUE_TYPE> struct BoundedRMSpropSynapsePolicy {
    // min_decay_frac: floors how fast ci can DECAY per step (only binds
    // above beta2; explicitly not an AMSGrad-style permanent max).
    // max_ci: hard ceiling on ci's GROWTH (no cap otherwise -- measured
    // climbing unboundedly, 0.0005->163+, in an unsafe-pocket run).
    // CHOSEN PRODUCTION DEFAULTS and the tuning sweeps behind them (why
    // min_decay_frac stays a no-op, why max_ci=100.0, and the important
    // correction that capping ci does NOT rescue an out-of-range
    // max_abs_delta/lr pocket's own divergence) are in
    // docs/research/delta_csr_types.rst:synapse_policy.bounded_beats_plain.
    // Same NaN/Inf guard convention as PlainRMSpropSynapsePolicy::update_ci,
    // checked BEFORE the floor/max_ci clamps (std::min/max's NaN behavior
    // is comparison-order-dependent, not a reliable filter on its own).
    static VALUE_TYPE update_ci(VALUE_TYPE ci, VALUE_TYPE g, VALUE_TYPE contrib, VALUE_TYPE beta2,
                                VALUE_TYPE min_decay_frac, VALUE_TYPE max_ci) {
        if (!std::isfinite(g) || !std::isfinite(contrib))
            return ci;
        const VALUE_TYPE ema = beta2 * ci + (VALUE_TYPE(1) - beta2) * (g * g + contrib * contrib);
        if (!std::isfinite(ema))
            return ci;
        const VALUE_TYPE floor = min_decay_frac * ci;
        return std::min(std::max(ema, floor), max_ci);
    }

    // Clips the LR-INDEPENDENT raw update, THEN multiplies by eff_lr --
    // a real bug (clipping the already-lr-scaled delta made max_abs_delta
    // an absolute cap regardless of lr, silently crushing every step for
    // callers using lr >> 0.05). See
    // docs/research/delta_csr_types.rst:synapse_policy.clip_order_and_lr_ceiling
    // for the fix, the FoldedColumnLayer regression it caused, and why
    // this does NOT make one max_abs_delta safe for unlimited lr.
    // scale_invariant: same quadratic-in-S fix as
    // PlainRMSpropSynapsePolicy::update_cw -- here the clip applies to
    // `raw` (before the /S division), bounding the S-normalized
    // true-weight-space step rather than the internal w_stored-space one.
    static VALUE_TYPE update_cw(VALUE_TYPE g, VALUE_TYPE ci, VALUE_TYPE S, VALUE_TYPE eff_lr,
                                VALUE_TYPE eps, bool damp_by_importance, VALUE_TYPE max_abs_delta,
                                bool scale_invariant = false) {
        if (!std::isfinite(g) || !std::isfinite(ci) || !std::isfinite(S))
            return VALUE_TYPE(0);
        VALUE_TYPE raw = scale_invariant
                             ? (damp_by_importance ? (-g) / (std::sqrt(ci) + eps) : (-g))
                             : (damp_by_importance ? (-g * S) / (std::sqrt(ci) + eps) : (-g * S));
        if (!std::isfinite(raw))
            return VALUE_TYPE(0);
        if (raw > max_abs_delta)
            raw = max_abs_delta;
        if (raw < -max_abs_delta)
            raw = -max_abs_delta;
        const VALUE_TYPE delta = scale_invariant ? (eff_lr * raw / S) : (eff_lr * raw);
        return std::isfinite(delta) ? delta : VALUE_TYPE(0);
    }
};

// ── Layout metadata ───────────────────────────────────────────────────────────

struct DeltaCSRLayout {
    std::size_t rows = 0;
    std::size_t cols = 0;

    std::vector<std::size_t> byte_start; // size rows+1
    std::vector<std::size_t> byte_end;   // size rows

    std::vector<std::size_t> elem_start; // size rows+1
    std::vector<std::size_t> elem_end;   // size rows

    std::size_t total_nnz = 0;

    std::size_t row_nnz(std::size_t r) const { return elem_end[r] - elem_start[r]; }
    std::size_t row_byte_len(std::size_t r) const { return byte_end[r] - byte_start[r]; }
    std::size_t row_alloc_bytes(std::size_t r) const { return byte_start[r + 1] - byte_start[r]; }
    std::size_t row_alloc_elems(std::size_t r) const { return elem_start[r + 1] - elem_start[r]; }
    std::size_t row_blank_bytes(std::size_t r) const { return byte_start[r + 1] - byte_end[r]; }
    std::size_t row_blank_elems(std::size_t r) const { return elem_start[r + 1] - elem_end[r]; }

    std::size_t total_alloc_bytes() const { return byte_start.empty() ? 0 : byte_start.back(); }
    std::size_t total_alloc_elems() const { return elem_start.empty() ? 0 : elem_start.back(); }

    std::size_t total_blank_bytes() const {
        std::size_t b = 0;
        for (std::size_t r = 0; r < rows; ++r)
            b += row_blank_bytes(r);
        return b;
    }
    std::size_t total_blank_elems() const {
        std::size_t b = 0;
        for (std::size_t r = 0; r < rows; ++r)
            b += row_blank_elems(r);
        return b;
    }

    std::size_t num_rows() const { return rows; }
};

// ── Forward-only row cursor ───────────────────────────────────────────────────

template <typename COL_TYPE = uint32_t> struct DeltaCSRRowCursor {
    const uint8_t* buf = nullptr;
    std::size_t byte_pos = 0;
    std::size_t byte_end = 0;
    COL_TYPE cur_col = 0;
    std::size_t n_decoded = 0;

    DeltaCSRRowCursor() = default;

    DeltaCSRRowCursor(const uint8_t* indices_buf, const DeltaCSRLayout& L, std::size_t row)
        : buf(indices_buf), byte_pos(L.byte_start[row]), byte_end(L.byte_end[row]), cur_col(0),
          n_decoded(0) {}

    bool at_end() const { return byte_pos >= byte_end; }

    COL_TYPE advance() {
        cur_col += uleb128_decode<COL_TYPE>(buf, byte_pos);
        ++n_decoded;
        return cur_col;
    }

    void advance_to(std::size_t target) {
        while (n_decoded <= target)
            advance();
    }

    COL_TYPE col() const { return cur_col; }
};

// Needs DeltaCSRLayout/DeltaCSRRowCursor just defined above -- Block4Store
// reuses both directly (both are already value-type-agnostic, no changes
// needed to either). See conversation (ULEB128 block4 tile indexing).
#include "block4.hpp"

// ── Per-synapse ci-update policy: Block4Vec (SIMD) specializations ────────────
// Full explicit specializations (not a VALUE_TYPE=Block4Vec instantiation)
// because the generic template uses std::max, which does not compile for
// Block4Vec. See
// docs/research/delta_csr_types.rst:synapse_policy.block4vec_specializations.
template <> struct PlainRMSpropSynapsePolicy<Block4Vec> {
    // Same NaN/Inf guard as the scalar update_ci, per-lane via
    // block4_vec_select_finite (block4.hpp has no whole-vector isfinite).
    static Block4Vec update_ci(Block4Vec ci, Block4Vec g, Block4Vec contrib, Block4Vec beta2,
                               Block4Vec /*min_decay_frac*/, Block4Vec /*max_ci*/) {
        const Block4Vec one = block4_vec_broadcast(1.0f);
        const Block4Vec new_ci = beta2 * ci + (one - beta2) * (g * g + contrib * contrib);
        return block4_vec_select_finite(new_ci, ci);
    }

    // Same "0 delta on non-finite" guard and scale_invariant fix as the
    // scalar version -- host-side bool, selects which SIMD formula runs.
    static Block4Vec update_cw(Block4Vec g, Block4Vec ci, Block4Vec S, Block4Vec eff_lr,
                               Block4Vec eps, bool damp_by_importance, Block4Vec /*max_abs_delta*/,
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

template <> struct BoundedRMSpropSynapsePolicy<Block4Vec> {
    // Same min_decay_frac semantics and NaN/Inf guard as the scalar
    // update_ci (checked before the floor/max_ci clamps).
    static Block4Vec update_ci(Block4Vec ci, Block4Vec g, Block4Vec contrib, Block4Vec beta2,
                               Block4Vec min_decay_frac, Block4Vec max_ci) {
        const Block4Vec one = block4_vec_broadcast(1.0f);
        const Block4Vec ema = beta2 * ci + (one - beta2) * (g * g + contrib * contrib);
        const Block4Vec ema_safe = block4_vec_select_finite(ema, ci);
        const Block4Vec floor = min_decay_frac * ci;
        return block4_vec_min(block4_vec_max(ema_safe, floor), max_ci);
    }

    // Clips the lr-independent raw update before the eff_lr multiply --
    // must match the scalar version exactly, or SIMD full-tile vs
    // scalar-boundary results would diverge for the same synapse. Same
    // NaN/Inf guard and scale_invariant fix as the scalar version.
    static Block4Vec update_cw(Block4Vec g, Block4Vec ci, Block4Vec S, Block4Vec eff_lr,
                               Block4Vec eps, bool damp_by_importance, Block4Vec max_abs_delta,
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
    DeltaCSRLayout layout;
    std::vector<uint8_t> indices_buf;
    VALUES_TYPE values;

    std::size_t max_indices_bytes = std::numeric_limits<std::size_t>::max();
    std::size_t max_values_bytes = std::numeric_limits<std::size_t>::max();

    using size_type = SIZE_TYPE;
    using col_type = COL_TYPE;
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;

    bool empty() const { return layout.total_nnz == 0; }
    std::size_t nnz() const { return layout.total_nnz; }
    std::size_t num_rows() const { return layout.rows; }
    std::size_t total_blank_bytes() const { return layout.total_blank_bytes(); }

    DeltaCSRRowCursor<COL_TYPE> row_cursor(std::size_t row) const {
        return DeltaCSRRowCursor<COL_TYPE>(indices_buf.data(), layout, row);
    }

    void set_limits(std::size_t indices_limit_bytes, std::size_t values_limit_bytes) {
        max_indices_bytes = indices_limit_bytes;
        max_values_bytes = values_limit_bytes;
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
template <typename VALUES_TYPE> struct Block4StoreFor {
    using type = Block4Store;
};
template <> struct Block4StoreFor<FP8BiValues> {
    using type = Block4Store8;
};
// DISLDOLayerV's plain-float32 storage (VALUES_TYPE=DeltaCSRBiValues<float>)
// gets Block4Store32 (block4.hpp) -- float32 needs no bit-packing at all, so
// this is the simplest of the three Block4Store variants (see that file's
// own comment on why its SIMD path needs no decode/encode). Every other
// DeltaCSRBiValues<T> instantiation (e.g. <double>, if one is ever added)
// keeps the default Block4Store above, unchanged.
template <> struct Block4StoreFor<DeltaCSRBiValues<float>> {
    using type = Block4Store32;
};

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
static void reshuffle_rank_array(std::vector<T>& arr, std::size_t old_rank, std::size_t new_rank,
                                 DefaultFn default_for_k) {
    if (arr.empty() || old_rank == new_rank)
        return;
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
// per-rank-channel gamma, used by BOTH the multiplicative (scale_gamma)
// and additive (additive_gamma) branches -- the math is IDENTICAL between
// them; only the raw gamma value's own lazy-default semantics stay
// branch-specific. See
// docs/research/delta_csr_types.rst:sparse_linear_weights_delta.gamma_ema_tracker_shared.
template <typename value_type> struct GammaEMATracker {
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
        if (abs_ema.size() <= k)
            abs_ema.resize(k + 1, value_type(0));
        if (share_ema.size() <= k)
            share_ema.resize(k + 1, value_type(0));
        if (grad_ema.size() <= k)
            grad_ema.resize(k + 1, value_type(0));
        abs_ema[k] = decay * abs_ema[k] + (value_type(1) - decay) * abs_gamma_k;
        share_ema[k] = decay * share_ema[k] + (value_type(1) - decay) * share_k;
        grad_ema[k] = decay * grad_ema[k] + (value_type(1) - decay) * abs_grad_k;
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
    inline bool should_neurogenesis(std::size_t rank, value_type tau_active,
                                    value_type theta) const {
        if (rank == 0)
            return false;
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
        if (k1 == k2)
            return;
        auto swap_one = [&](std::vector<value_type>& arr) {
            const std::size_t need = std::max(k1, k2) + 1;
            if (arr.size() < need)
                arr.resize(need, value_type(0));
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
    using size_type = SIZE_TYPE;
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    using block4_type = typename Block4StoreFor<VALUES_TYPE>::type;

    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> connections;
    COOSynaptogenesis<SIZE_TYPE, value_type> probes;
    std::vector<SIZE_TYPE> out_degree;

    // Locally-dense companion to `connections` -- see block4.hpp. Shares
    // THIS struct's own value_scale/importance_scale (not a separate
    // scale), so moving a synapse between `connections` and `block4` is a
    // lossless byte copy, not a requantization. Type selected via
    // Block4StoreFor<VALUES_TYPE> above. Promotion/demotion logic lives in
    // delta_csr_memory.hpp (would be a circular include from here).
    block4_type block4;

    // Per-ROW scale applied to STORED importance to get TRUE units before
    // any importance arithmetic. Read as: true_imp = stored_imp * scale.
    // Lazily-sized, defaulting any untouched row to 1.0 (fully backward
    // compatible). See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.importance_scale_rationale
    // for why FP4's 0.5 floor makes this necessary and why it's per-row
    // rather than per-layer.
    std::vector<value_type> importance_scale;

    inline value_type get_importance_scale(std::size_t row) const {
        return row < importance_scale.size() ? importance_scale[row] : value_type(1);
    }
    inline void set_importance_scale_raw(std::size_t row, value_type v) {
        if (row >= importance_scale.size())
            importance_scale.resize(row + 1, value_type(1));
        importance_scale[row] = v;
    }

    // Per-COLUMN counterpart to importance_scale (true_imp = stored_imp *
    // importance_scale[row] * output_importance_scale[col]), same
    // rationale as output_scale is to value_scale. Default 1.0, same
    // lazy-sizing convention; unused by from_descriptor today (a pure
    // no-op until a caller opts in).
    std::vector<value_type> output_importance_scale;
    inline value_type get_output_importance_scale(std::size_t col) const {
        return col < output_importance_scale.size() ? output_importance_scale[col] : value_type(1);
    }
    inline void set_output_importance_scale_raw(std::size_t col, value_type v) {
        if (col >= output_importance_scale.size())
            output_importance_scale.resize(col + 1, value_type(1));
        output_importance_scale[col] = v;
    }

    // task #295 fix: replaces a compile-time SCALE_RANK_MAX=4 stack-array
    // cap in block4's SIMD backward path with persistent, per-instance
    // HEAP buffers that grow to fit whatever scale_rank is actually used,
    // reused across calls rather than reallocated per tile. Holds only
    // value_type/double (no Block4Vec -- this header excludes block4.hpp),
    // loaded/stored via block4_vec_load/store from linear_disldo.hpp. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.scale_rank_scratch_task295.
    struct ScaleRankScratch {
        std::vector<value_type> value_scale_k;      // [thread][k]
        std::vector<value_type> out_scale_k;        // [thread][k][tile_width]
        std::vector<value_type> mcol_rank;          // [thread][k][tile_width]
        std::vector<double> mrow_local_k;           // [thread][k]
        std::vector<value_type> mcol_rank_contrib;  // [thread][k][tile_width]
        std::vector<double> mrow_local_k_contrib;   // [thread][k]
        std::vector<double> mgamma_local_k;         // [thread][k]
        std::vector<double> mgamma_local_k_contrib; // [thread][k]
        std::vector<value_type>
            mcol_acc_raw; // [thread][k][tile_width] -- Block4Vec accumulator backing
        std::vector<value_type> mcol_acc_raw_contrib; // [thread][k][tile_width]

        std::size_t cap_threads = 0, cap_rank = 0, cap_tile_width = 0;

        // Grow-only (never shrinks) -- called automatically at the top of
        // every disldo_backward call, a cheap no-op once large enough.
        void ensure(std::size_t threads, std::size_t rank, std::size_t tile_width) {
            if (threads <= cap_threads && rank <= cap_rank && tile_width <= cap_tile_width)
                return;
            resize_to(std::max(cap_threads, threads), std::max(cap_rank, rank),
                      std::max(cap_tile_width, tile_width));
        }

        // Explicit, caller-driven resize -- unlike ensure(), CAN shrink.
        // Caller must not pass below what's currently in use.
        void resize_to(std::size_t threads, std::size_t rank, std::size_t tile_width) {
            cap_threads = threads;
            cap_rank = rank;
            cap_tile_width = tile_width;
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

    // Explicit, caller-driven scratch memory control (task #295) --
    // separate from scale_rank_max/additive_rank_max below (POLICY cap on
    // rank growth, not memory). Can shrink or preallocate ahead of need;
    // must not go below what's currently in use, or it corrupts buffers
    // disldo_backward is actively reading/writing.
    inline void reserve_scale_rank_scratch(std::size_t threads, std::size_t rank,
                                           std::size_t tile_width) {
        if (rank < scale_rank)
            throw std::invalid_argument("reserve_scale_rank_scratch: rank below the layer's "
                                        "current scale_rank would corrupt live scratch data");
        if (threads < 1)
            throw std::invalid_argument("reserve_scale_rank_scratch: threads must be >= 1");
        scale_rank_scratch.resize_to(threads, rank, tile_width);
    }

    // additive_rank never needed a compile-time cap (its own forward/
    // backward pass uses ordinary std::vector throughout already) --
    // task #295's ScaleRankScratch above is scale_rank-specific.

    // RANK of the value_scale/output_scale factorization: true_w = quant *
    // S[row,col], S[row,col] = sum_{k<scale_rank} value_scale_k(row,k) *
    // output_scale_k(col,k) (generalizes the old rank-1 outer product).
    // Runtime-parameterized so no extra class/pybind binding is needed per
    // rank tried; k>=1 defaults to 0.0 so an unconfigured extra component
    // is a pure no-op. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.scale_rank_rationale
    // for why rank>1 is needed at all (conflicting per-column gradient
    // demand within one row), the scattered/block4 scope note, and the
    // known DeferredScaleWrite rank-1-only limitation.
    std::size_t scale_rank = 1;

    // AQRS dynamic rank control (task #292 fix): calls since the LAST
    // rank mutation of EITHER kind on this branch -- a real 60k-step MQAR
    // run showed gamma's gradient can jump the tau_death/tau_active
    // hysteresis gap in one step, causing rapid oscillation without this
    // gate. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.rank_mutation_cooldown.
    uint32_t scale_rank_calls_since_mutation = UINT32_MAX;

    // Same per-row design as importance_scale, for STORED weight values.
    // Row-major per-component: value_scale[row*scale_rank + k].
    std::vector<value_type> value_scale;

    // Per-component accessor. Component 0 defaults to 1.0 (original
    // single-component convention); k>=1 defaults to 0.0 (untrained extra
    // rank component is a pure no-op).
    inline value_type get_value_scale_k(std::size_t row, std::size_t k) const {
        const std::size_t idx = row * scale_rank + k;
        if (idx < value_scale.size())
            return value_scale[idx];
        return k == 0 ? value_type(1) : value_type(0);
    }
    inline void set_value_scale_raw_k(std::size_t row, std::size_t k, value_type v) {
        const std::size_t idx = row * scale_rank + k;
        if (idx >= value_scale.size()) {
            // Real bug fix: a uniform resize(...,1.0) fill backfills EVERY
            // appended slot with 1.0, including k>=1 ones. Resize neutral
            // (0), fix up only k==0 in the appended range -- see
            // docs/research/linear_disldo.rst:disldo_backward.setup_and_presizing_bugs.
            const std::size_t old_size = value_scale.size();
            value_scale.resize(idx + 1, value_type(0));
            for (std::size_t i = old_size; i < value_scale.size(); ++i)
                if (i % scale_rank == 0)
                    value_scale[i] = value_type(1);
        }
        value_scale[idx] = v;
    }
    // Backward-compat single-component accessors -- component 0 only,
    // exact original meaning/behavior at scale_rank==1.
    inline value_type get_value_scale(std::size_t row) const { return get_value_scale_k(row, 0); }
    inline void set_value_scale_raw(std::size_t row, value_type v) {
        set_value_scale_raw_k(row, 0, v);
    }

    // value_scale/output_scale are themselves params trained via gradient
    // descent, so each gets its own importance damping its update step.
    // Default 0, same per-component row-major layout as value_scale.
    std::vector<value_type> value_scale_importance;
    inline value_type get_value_scale_importance_k(std::size_t row, std::size_t k) const {
        const std::size_t idx = row * scale_rank + k;
        return idx < value_scale_importance.size() ? value_scale_importance[idx] : value_type(0);
    }
    inline value_type get_value_scale_importance(std::size_t row) const {
        return get_value_scale_importance_k(row, 0);
    }

    // Step counter for value_scale_importance's Adam-style bias
    // correction -- fixes a real ~31.6x-oversized first step / sign-flip
    // regression. uint32_t, one per row*rank slot (cheap; per-synapse
    // `ci` does NOT get this treatment, would double memory). See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.value_scale_step_bias_correction.
    std::vector<uint32_t> value_scale_step;
    inline uint32_t& get_value_scale_step_k(std::size_t row, std::size_t k) {
        const std::size_t idx = row * scale_rank + k;
        if (value_scale_step.size() <= idx)
            value_scale_step.resize(idx + 1, 0);
        return value_scale_step[idx];
    }

    // Adam-style FIRST moment (signed EMA of g_agg) for value_scale, used
    // only by the dead-row (nnz_row==0) bootstrap path in
    // disldo_backward. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.value_scale_momentum_dead_row
    // for why a dead row needs its own signed accumulator and why it's
    // linear in g_agg, not squared.
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
    // Same lazy-sizing/default convention, row-major per-component layout.
    // Trained via gradient descent ONLY once a caller opts in via
    // set_output_scale_raw{,_k} -- see
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.output_scale_trainable_gate
    // for why output_scale_is_trainable exists instead of checking
    // output_scale.empty().
    std::vector<value_type> output_scale;
    std::vector<value_type> output_scale_importance;
    bool output_scale_is_trainable = false;

    inline value_type get_output_scale_k(std::size_t col, std::size_t k) const {
        const std::size_t idx = col * scale_rank + k;
        if (idx < output_scale.size())
            return output_scale[idx];
        return k == 0 ? value_type(1) : value_type(0);
    }
    inline value_type get_output_scale(std::size_t col) const { return get_output_scale_k(col, 0); }
    inline value_type get_output_scale_importance_k(std::size_t col, std::size_t k) const {
        const std::size_t idx = col * scale_rank + k;
        return idx < output_scale_importance.size() ? output_scale_importance[idx] : value_type(0);
    }
    // Step counter for output_scale_importance's bias correction -- same
    // mechanism as value_scale_step, one level over from row to column.
    std::vector<uint32_t> output_scale_step;
    inline uint32_t& get_output_scale_step_k(std::size_t col, std::size_t k) {
        const std::size_t idx = col * scale_rank + k;
        if (output_scale_step.size() <= idx)
            output_scale_step.resize(idx + 1, 0);
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
                if (i % scale_rank == 0)
                    output_scale[i] = value_type(1);
        }
        output_scale[idx] = v;
        output_scale_is_trainable = true;
    }
    inline void set_output_scale_raw(std::size_t col, value_type v) {
        set_output_scale_raw_k(col, 0, v);
    }

    // Bulk raw-vector accessors (task #295 follow-up): expose AQRS scale
    // channels as "virtual neurons" for a one-pass bulk correction
    // instead of n*scale_rank individual get/set_*_k calls. Size-
    // preserving (correction pass only, not resize -- that's
    // set_scale_rank's job). See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.bulk_raw_vector_accessors.
    inline const std::vector<value_type>& get_value_scale_raw_vector() const { return value_scale; }
    inline void set_value_scale_raw_vector(const std::vector<value_type>& v) {
        if (v.size() != value_scale.size())
            throw std::invalid_argument(
                "set_value_scale_raw_vector: size mismatch (correction pass only, not resize)");
        value_scale = v;
    }
    inline const std::vector<value_type>& get_output_scale_raw_vector() const {
        return output_scale;
    }
    inline void set_output_scale_raw_vector(const std::vector<value_type>& v) {
        if (v.size() != output_scale.size())
            throw std::invalid_argument(
                "set_output_scale_raw_vector: size mismatch (correction pass only, not resize)");
        output_scale = v;
    }

    // AQRS per-channel gamma (task #273/#282-283, see sili_peridot/
    // AQRS_DESIGN.md's gamma section): decouples channel MAGNITUDE from
    // DIRECTION. gamma_s_k (one scalar per channel k) holds the
    // magnitude: S[row,col] = sum_k gamma_s_k * value_scale_k(row,k) *
    // output_scale_k(col,k). Lazy default is 1.0 for EVERY k (a real
    // backward-compat break otherwise -- see
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.scale_gamma_backward_compat_bug
    // for the k>=1->0.0 regression this fixes and why the "freshly-grown
    // channel = zero contribution" property instead lives only in
    // set_scale_rank's reshuffle). scale_gamma_is_trainable gates disldo_backward's gamma
    // UPDATE the same way output_scale_is_trainable does -- without it,
    // every rank>=1 layer that's never heard of gamma gets an unsolicited
    // perturbation to gamma_s_k(0) (confirmed as a real regression).
    bool scale_gamma_is_trainable = false;
    std::vector<value_type> scale_gamma;
    std::vector<value_type> scale_gamma_state; // RMSprop second moment, one per channel
    std::vector<uint32_t> scale_gamma_step;    // bias-correction counter, one per channel
    inline value_type get_scale_gamma_k(std::size_t k) const {
        if (k < scale_gamma.size())
            return scale_gamma[k];
        return value_type(1);
    }
    inline void set_scale_gamma_raw_k(std::size_t k, value_type v) {
        if (k >= scale_gamma.size())
            scale_gamma.resize(k + 1, value_type(1));
        scale_gamma[k] = v;
        scale_gamma_is_trainable = true;
    }
    inline value_type& get_scale_gamma_state_k(std::size_t k) {
        if (scale_gamma_state.size() <= k)
            scale_gamma_state.resize(k + 1, value_type(0));
        return scale_gamma_state[k];
    }
    inline uint32_t& get_scale_gamma_step_k(std::size_t k) {
        if (scale_gamma_step.size() <= k)
            scale_gamma_step.resize(k + 1, 0);
        return scale_gamma_step[k];
    }

    // AQRS dynamic rank control (task #273/#284): EMA-smoothed per-channel
    // signals (|gamma_k|_ema, C_k_ema = channel's share of L1 mass,
    // grad_k_ema), refreshed EVERY step, feeding Theorem 10's apoptosis/
    // neurogenesis triggers. EMA storage + trigger logic live in the
    // shared GammaEMATracker; these are thin delegating wrappers kept
    // under their original names so no existing caller needed to change.
    // See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.gamma_ema_tracker_shared.
    GammaEMATracker<value_type> scale_gamma_ema;
    inline value_type get_scale_gamma_abs_ema_k(std::size_t k) const {
        return scale_gamma_ema.get_abs_ema_k(k);
    }
    inline value_type get_scale_gamma_share_ema_k(std::size_t k) const {
        return scale_gamma_ema.get_share_ema_k(k);
    }
    inline value_type get_scale_gamma_grad_ema_k(std::size_t k) const {
        return scale_gamma_ema.get_grad_ema_k(k);
    }
    // Called once per k, AFTER gamma's own value update for every channel
    // is finalized (C_k needs every channel's current |gamma| first).
    inline void update_scale_gamma_ema_k(std::size_t k, value_type abs_gamma_k, value_type share_k,
                                         value_type abs_grad_k,
                                         value_type decay = value_type(0.98)) {
        scale_gamma_ema.update_k(k, abs_gamma_k, share_k, abs_grad_k, decay);
    }
    inline bool scale_gamma_should_apoptose(std::size_t k, value_type tau_death) const {
        return scale_gamma_ema.should_apoptose(k, tau_death);
    }
    inline bool scale_gamma_should_neurogenesis(std::size_t rank, value_type tau_active,
                                                value_type theta) const {
        return scale_gamma_ema.should_neurogenesis(rank, tau_active, theta);
    }

    // Combined rank-N scale: S[row,col] = sum_{k<scale_rank} gamma_s_k *
    // value_scale_k(row,k) * output_scale_k(col,k) -- THE quantity
    // Hadamard-multiplied against quant in disldo_forward/backward. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.get_scale_formula.
    inline value_type get_scale(std::size_t row, std::size_t col) const {
        value_type s = value_type(0);
        for (std::size_t k = 0; k < scale_rank; ++k)
            s += get_scale_gamma_k(k) * get_value_scale_k(row, k) * get_output_scale_k(col, k);
        return s;
    }

    // AQRS additive branch: A[row,col] = sum_{k<additive_rank}
    // additive_u_k(row,k) * additive_v_k(col,k) -- SUMMED into (not
    // Hadamard-multiplied like value_scale/output_scale above) the
    // effective weight. Structurally necessary, not just useful: the
    // multiplicative branch's gradient is exactly zero wherever the
    // quantized weight is the zero code, at any rank -- only an additive
    // term can write a value there. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.additive_branch_overview.
    std::size_t additive_rank = 0;
    // Same cooldown counter as scale_rank_calls_since_mutation, own copy
    // since the two branches mutate independently.
    uint32_t additive_rank_calls_since_mutation = UINT32_MAX;
    std::vector<value_type> additive_u; // row-major per-component: additive_u[row*additive_rank+k]
    std::vector<value_type> additive_v; // row-major per-component: additive_v[col*additive_rank+k]

    inline value_type get_additive_u_k(std::size_t row, std::size_t k) const {
        const std::size_t idx = row * additive_rank + k;
        return idx < additive_u.size() ? additive_u[idx] : value_type(0);
    }
    inline void set_additive_u_raw_k(std::size_t row, std::size_t k, value_type v) {
        const std::size_t idx = row * additive_rank + k;
        if (idx >= additive_u.size())
            additive_u.resize(idx + 1, value_type(0));
        additive_u[idx] = v;
    }
    inline value_type get_additive_v_k(std::size_t col, std::size_t k) const {
        const std::size_t idx = col * additive_rank + k;
        return idx < additive_v.size() ? additive_v[idx] : value_type(0);
    }
    inline void set_additive_v_raw_k(std::size_t col, std::size_t k, value_type v) {
        const std::size_t idx = col * additive_rank + k;
        if (idx >= additive_v.size())
            additive_v.resize(idx + 1, value_type(0));
        additive_v[idx] = v;
    }

    // Bulk raw-vector accessors -- same rationale as
    // get/set_value_scale_raw_vector above.
    inline const std::vector<value_type>& get_additive_u_raw_vector() const { return additive_u; }
    inline void set_additive_u_raw_vector(const std::vector<value_type>& v) {
        if (v.size() != additive_u.size())
            throw std::invalid_argument(
                "set_additive_u_raw_vector: size mismatch (correction pass only, not resize)");
        additive_u = v;
    }
    inline const std::vector<value_type>& get_additive_v_raw_vector() const { return additive_v; }
    inline void set_additive_v_raw_vector(const std::vector<value_type>& v) {
        if (v.size() != additive_v.size())
            throw std::invalid_argument(
                "set_additive_v_raw_vector: size mismatch (correction pass only, not resize)");
        additive_v = v;
    }

    // AQRS per-channel gamma for the additive branch (task #273/#282-283,
    // wired in at task #289), same role as scale_gamma but for
    // additive_u/additive_v. Lazy default is 1.0 (transparent), NOT 0.0 --
    // caught before landing this time. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.additive_gamma_backward_compat_bug.
    bool additive_gamma_is_trainable = false;
    std::vector<value_type> additive_gamma;
    // RMSprop-style state (matches scale_gamma's OWN update policy, NOT
    // AdamScalePolicy, even though additive_u/v use Adam) -- Adam's
    // momentum overshoots the L1-created zero fixed point. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.additive_gamma_adam_overshoot.
    std::vector<value_type> additive_gamma_state;
    std::vector<uint32_t> additive_gamma_step;
    GammaEMATracker<value_type> additive_gamma_ema;
    inline value_type get_additive_gamma_k(std::size_t k) const {
        if (k < additive_gamma.size())
            return additive_gamma[k];
        return value_type(1);
    }
    inline void set_additive_gamma_raw_k(std::size_t k, value_type v) {
        if (k >= additive_gamma.size())
            additive_gamma.resize(k + 1, value_type(1));
        additive_gamma[k] = v;
        additive_gamma_is_trainable = true;
    }
    inline value_type& get_additive_gamma_state_k(std::size_t k) {
        if (additive_gamma_state.size() <= k)
            additive_gamma_state.resize(k + 1, value_type(0));
        return additive_gamma_state[k];
    }
    inline uint32_t& get_additive_gamma_step_k(std::size_t k) {
        if (additive_gamma_step.size() <= k)
            additive_gamma_step.resize(k + 1, 0);
        return additive_gamma_step[k];
    }
    inline value_type get_additive_gamma_abs_ema_k(std::size_t k) const {
        return additive_gamma_ema.get_abs_ema_k(k);
    }
    inline value_type get_additive_gamma_share_ema_k(std::size_t k) const {
        return additive_gamma_ema.get_share_ema_k(k);
    }
    inline value_type get_additive_gamma_grad_ema_k(std::size_t k) const {
        return additive_gamma_ema.get_grad_ema_k(k);
    }
    inline void update_additive_gamma_ema_k(std::size_t k, value_type abs_gamma_k,
                                            value_type share_k, value_type abs_grad_k,
                                            value_type decay = value_type(0.98)) {
        additive_gamma_ema.update_k(k, abs_gamma_k, share_k, abs_grad_k, decay);
    }
    inline bool additive_gamma_should_apoptose(std::size_t k, value_type tau_death) const {
        return additive_gamma_ema.should_apoptose(k, tau_death);
    }
    inline bool additive_gamma_should_neurogenesis(std::size_t rank, value_type tau_active,
                                                   value_type theta) const {
        return additive_gamma_ema.should_neurogenesis(rank, tau_active, theta);
    }

    // AdamScalePolicy's own state for additive_u/additive_v (task #277) --
    // same lazy-growth, row-major-per-component convention as everything
    // else here. Two independent EMAs per Adam's own definition (first
    // moment = momentum, second moment = state), plus one step counter
    // for bias correction -- see AdamScalePolicy::update's own docstring.
    std::vector<value_type> additive_u_momentum, additive_u_state;
    std::vector<value_type> additive_v_momentum, additive_v_state;
    std::vector<uint32_t> additive_u_step, additive_v_step;
    inline value_type& get_additive_u_momentum_k(std::size_t row, std::size_t k) {
        const std::size_t idx = row * additive_rank + k;
        if (additive_u_momentum.size() <= idx)
            additive_u_momentum.resize(idx + 1, value_type(0));
        return additive_u_momentum[idx];
    }
    inline value_type& get_additive_u_state_k(std::size_t row, std::size_t k) {
        const std::size_t idx = row * additive_rank + k;
        if (additive_u_state.size() <= idx)
            additive_u_state.resize(idx + 1, value_type(0));
        return additive_u_state[idx];
    }
    inline uint32_t& get_additive_u_step_k(std::size_t row, std::size_t k) {
        const std::size_t idx = row * additive_rank + k;
        if (additive_u_step.size() <= idx)
            additive_u_step.resize(idx + 1, 0);
        return additive_u_step[idx];
    }
    inline value_type& get_additive_v_momentum_k(std::size_t col, std::size_t k) {
        const std::size_t idx = col * additive_rank + k;
        if (additive_v_momentum.size() <= idx)
            additive_v_momentum.resize(idx + 1, value_type(0));
        return additive_v_momentum[idx];
    }
    inline value_type& get_additive_v_state_k(std::size_t col, std::size_t k) {
        const std::size_t idx = col * additive_rank + k;
        if (additive_v_state.size() <= idx)
            additive_v_state.resize(idx + 1, value_type(0));
        return additive_v_state[idx];
    }
    inline uint32_t& get_additive_v_step_k(std::size_t col, std::size_t k) {
        const std::size_t idx = col * additive_rank + k;
        if (additive_v_step.size() <= idx)
            additive_v_step.resize(idx + 1, 0);
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
    // Runtime-settable POLICY cap (task #295 -- was a compile-time
    // SCALE_RANK_MAX=4 forced by block4's now-gone fixed-size stack
    // arrays). Independent of scale_rank_scratch's memory sizing; default
    // 4 matches the old compile-time constant.
    std::size_t scale_rank_max = 4;
    std::size_t additive_rank_max = 4;
    inline std::size_t get_scale_rank_max() const { return scale_rank_max; }
    // Lowering below the CURRENT scale_rank is allowed -- just blocks
    // further growth until scale_rank is manually shrunk back under it.
    inline void set_scale_rank_max(std::size_t new_max) { scale_rank_max = new_max; }
    inline std::size_t get_additive_rank_max() const { return additive_rank_max; }
    inline void set_additive_rank_max(std::size_t new_max) { additive_rank_max = new_max; }

    inline void set_scale_rank(std::size_t new_rank) {
        if (new_rank == 0)
            throw std::invalid_argument("scale_rank must be >= 1");
        if (new_rank > scale_rank_max)
            throw std::invalid_argument("scale_rank exceeds scale_rank_max (the configured policy "
                                        "cap -- raise it via set_scale_rank_max first)");
        const std::size_t old_rank = scale_rank;
        auto scale_default = [](std::size_t k) { return k == 0 ? value_type(1) : value_type(0); };
        auto zero_default = [](std::size_t) { return value_type(0); };
        auto step_default = [](std::size_t) { return uint32_t(0); };
        reshuffle_rank_array(value_scale, old_rank, new_rank, scale_default);
        reshuffle_rank_array(value_scale_importance, old_rank, new_rank, zero_default);
        reshuffle_rank_array(value_scale_step, old_rank, new_rank, step_default);
        reshuffle_rank_array(value_scale_momentum, old_rank, new_rank, zero_default);
        reshuffle_rank_array(output_scale, old_rank, new_rank, scale_default);
        reshuffle_rank_array(output_scale_importance, old_rank, new_rank, zero_default);
        reshuffle_rank_array(output_scale_step, old_rank, new_rank, step_default);
        // scale_gamma uses zero_default, NOT scale_default (unlike
        // value_scale/output_scale) -- a freshly-grown gamma channel
        // should start at 0 (Theorem 9), since this reshuffle only ever
        // fires once gamma is already live. See
        // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.rank_swap_and_resize.
        reshuffle_rank_array(scale_gamma, old_rank, new_rank, zero_default);
        reshuffle_rank_array(scale_gamma_state, old_rank, new_rank, zero_default);
        reshuffle_rank_array(scale_gamma_step, old_rank, new_rank, step_default);
        scale_gamma_ema.reshuffle(old_rank, new_rank);
        scale_rank = new_rank;
    }

    // AQRS dynamic rank control (task #273/#285): set_scale_rank can only
    // SHRINK by truncating the highest-index channel, but Theorem 10's
    // apoptosis can fire on ANY channel -- swap the dying channel to the
    // end first, then truncate. Uses get_*/set_*_raw_k accessors (not raw
    // indexing) so lazy-unpopulated rows/cols read correct defaults. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.rank_swap_and_resize.
    inline void swap_scale_channels(std::size_t k1, std::size_t k2, std::size_t n_rows,
                                    std::size_t n_cols) {
        if (k1 == k2)
            return;
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
        // EMA state and age (scale_gamma_step) travel WITH the channel --
        // omitting either makes a relocated channel look freshly-born,
        // corrupting the trigger/grace-period logic.
        scale_gamma_ema.swap_k(k1, k2);
        {
            const std::size_t need = std::max(k1, k2) + 1;
            if (scale_gamma_step.size() < need)
                scale_gamma_step.resize(need, 0);
            std::swap(scale_gamma_step[k1], scale_gamma_step[k2]);
        }
    }

    // Additive-branch counterpart to swap_scale_channels (task #289) --
    // same reasoning, additive_u/v/gamma instead of value/output_scale.
    inline void swap_additive_channels(std::size_t k1, std::size_t k2, std::size_t n_rows,
                                       std::size_t n_cols) {
        if (k1 == k2)
            return;
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
            if (additive_gamma_step.size() < need)
                additive_gamma_step.resize(need, 0);
            std::swap(additive_gamma_step[k1], additive_gamma_step[k2]);
            if (additive_gamma_state.size() < need)
                additive_gamma_state.resize(need, value_type(0));
            std::swap(additive_gamma_state[k1], additive_gamma_state[k2]);
        }
    }

    // Shared control-flow for Theorem 10's apoptosis/neurogenesis dynamic
    // rank control (task #289), used by BOTH apply_dynamic_rank_control
    // (multiplicative) and apply_additive_dynamic_rank_control (additive)
    // below -- the decision logic is IDENTICAL between branches; only the
    // mutation ops and min_rank floor differ, passed in as callbacks. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.dynamic_rank_control_generic.
    template <typename AgeFn, typename ApoptoseCheckFn, typename NeurogenesisCheckFn,
              typename DoApoptoseFn, typename DoNeurogenesisFn>
    // calls_since_mutation: symmetric branch-level cooldown (task #292) --
    // a real 60k-step MQAR run showed 1464 mutations in 3000 steps without
    // it (gamma's gradient can jump the hysteresis gap in one step).
    // grow_grace_period_steps/shrink_grace_period_steps: kept independent
    // per call, but each branch defaults them equal to each other (the
    // real evidenced asymmetry is CROSS-branch, not within one). See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.rank_mutation_cooldown
    // and
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.grow_shrink_asymmetry_biology.
    static bool apply_dynamic_rank_control_generic(
        std::size_t rank, std::size_t min_rank, std::size_t max_rank,
        uint32_t grow_grace_period_steps, uint32_t shrink_grace_period_steps,
        uint32_t& calls_since_mutation, AgeFn age_of, ApoptoseCheckFn should_apoptose,
        NeurogenesisCheckFn should_neurogenesis, DoApoptoseFn do_apoptose,
        DoNeurogenesisFn do_neurogenesis) {
        if (calls_since_mutation < UINT32_MAX)
            ++calls_since_mutation;
        const uint32_t min_grace = std::min(grow_grace_period_steps, shrink_grace_period_steps);
        if (calls_since_mutation < min_grace)
            return false;
        if (calls_since_mutation >= shrink_grace_period_steps) {
            for (std::size_t k = 0; k < rank; ++k) {
                if (rank > min_rank && age_of(k) >= shrink_grace_period_steps &&
                    should_apoptose(k)) {
                    do_apoptose(k);
                    calls_since_mutation = 0;
                    return true;
                }
            }
        }
        if (calls_since_mutation >= grow_grace_period_steps && rank < max_rank &&
            should_neurogenesis()) {
            do_neurogenesis();
            calls_since_mutation = 0;
            return true;
        }
        return false;
    }

    // Evaluates Theorem 10's triggers against the CURRENT EMA state and
    // performs at most ONE real mutation per call. new_channel_seed(row)
    // deliberately does NOT hardcode Theorem 9's unverified
    // residual-alignment proxy -- it's a caller-supplied callback instead.
    // grace_period_steps closes the symmetric "pruned the instant it's
    // grown" bug (a freshly-grown channel's gamma/EMA starts near 0,
    // trivially satisfying apoptosis). Default 50/50, matching dendritic
    // spine formation/elimination timescales. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.dynamic_rank_control_generic
    // and
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.grow_shrink_asymmetry_biology.
    template <typename SeedFn>
    inline bool apply_dynamic_rank_control(std::size_t n_rows, std::size_t n_cols,
                                           value_type tau_death, value_type tau_active,
                                           value_type theta, SeedFn new_channel_seed,
                                           uint32_t grow_grace_period_steps = 50,
                                           uint32_t shrink_grace_period_steps = 50) {
        return apply_dynamic_rank_control_generic(
            scale_rank, /*min_rank=*/std::size_t(1), scale_rank_max, grow_grace_period_steps,
            shrink_grace_period_steps, scale_rank_calls_since_mutation,
            [&](std::size_t k) {
                return k < scale_gamma_step.size() ? scale_gamma_step[k] : uint32_t(0);
            },
            [&](std::size_t k) { return scale_gamma_should_apoptose(k, tau_death); },
            [&]() { return scale_gamma_should_neurogenesis(scale_rank, tau_active, theta); },
            [&](std::size_t k) {
                swap_scale_channels(k, scale_rank - 1, n_rows, n_cols);
                set_scale_rank(scale_rank - 1);
            },
            [&]() {
                const std::size_t new_k = scale_rank;
                set_scale_rank(scale_rank + 1);
                for (std::size_t r = 0; r < n_rows; ++r)
                    set_value_scale_raw_k(r, new_k, new_channel_seed(r));
                for (std::size_t c = 0; c < n_cols; ++c)
                    set_output_scale_raw_k(c, new_k, value_type(1));
            });
    }

    // Additive-branch counterpart to apply_dynamic_rank_control (task
    // #289/#292). Two seed callbacks (not one + uniform 1.0) since BOTH
    // additive_u and additive_v need a real nonzero direction for a new
    // channel's gradient. min_rank=0 (not 1): the additive branch has no
    // legacy always-on component. Default 5000/5000, ~100x the scale
    // branch's 50/50 -- matches the Hebbian-vs-homeostatic-plasticity
    // timescale separation. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.grow_shrink_asymmetry_biology.
    template <typename SeedUFn, typename SeedVFn>
    inline bool apply_additive_dynamic_rank_control(std::size_t n_rows, std::size_t n_cols,
                                                    value_type tau_death, value_type tau_active,
                                                    value_type theta, SeedUFn new_channel_seed_u,
                                                    SeedVFn new_channel_seed_v,
                                                    uint32_t grow_grace_period_steps = 5000,
                                                    uint32_t shrink_grace_period_steps = 5000) {
        return apply_dynamic_rank_control_generic(
            additive_rank, /*min_rank=*/std::size_t(0), additive_rank_max, grow_grace_period_steps,
            shrink_grace_period_steps, additive_rank_calls_since_mutation,
            [&](std::size_t k) {
                return k < additive_gamma_step.size() ? additive_gamma_step[k] : uint32_t(0);
            },
            [&](std::size_t k) { return additive_gamma_should_apoptose(k, tau_death); },
            [&]() { return additive_gamma_should_neurogenesis(additive_rank, tau_active, theta); },
            [&](std::size_t k) {
                swap_additive_channels(k, additive_rank - 1, n_rows, n_cols);
                set_additive_rank(additive_rank - 1);
            },
            [&]() {
                const std::size_t new_k = additive_rank;
                set_additive_rank(additive_rank + 1);
                for (std::size_t r = 0; r < n_rows; ++r)
                    set_additive_u_raw_k(r, new_k, new_channel_seed_u(r));
                for (std::size_t c = 0; c < n_cols; ++c)
                    set_additive_v_raw_k(c, new_k, new_channel_seed_v(c));
            });
    }

    inline void set_additive_rank(std::size_t new_rank) {
        // 0 is a valid, meaningful value here (branch fully disabled) --
        // unlike scale_rank, which must stay >= 1. additive_rank_max: same
        // runtime policy cap as scale_rank_max.
        if (new_rank > additive_rank_max)
            throw std::invalid_argument("additive_rank exceeds additive_rank_max (the configured "
                                        "policy cap -- raise it via set_additive_rank_max first)");
        const std::size_t old_rank = additive_rank;
        auto zero_default = [](std::size_t) { return value_type(0); };
        auto step_default = [](std::size_t) { return uint32_t(0); };
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

    // Running L1 / L2^2 / max|.| for STORED (quantized) importance/weight
    // values, maintained incrementally (O(1) per synapse touched). double,
    // not value_type -- long-running sums across millions of steps.
    // max_abs is a MONOTONIC upper bound, not a live exact max. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.importance_value_stats_thread_safety.
    double importance_l1 = 0.0;
    double importance_l2_sq = 0.0;
    value_type importance_max_abs = value_type(0);
    double value_l1 = 0.0;
    double value_l2_sq = 0.0;
    value_type value_max_abs = value_type(0);

    // Decay applied to max_abs BEFORE comparing against the incoming
    // value (default 1.0 = no decay, exact backward compat). See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.importance_value_stats_thread_safety.
    value_type max_abs_decay = value_type(1);

    inline value_type hoyer_importance() const {
        return _hoyer_from_stats(importance_l1, importance_l2_sq);
    }
    inline value_type hoyer_value() const { return _hoyer_from_stats(value_l1, value_l2_sq); }

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
        importance_l1 +=
            std::abs(static_cast<double>(new_val)) - std::abs(static_cast<double>(old_val));
        importance_l2_sq +=
            static_cast<double>(new_val) * new_val - static_cast<double>(old_val) * old_val;
        importance_max_abs = std::max(importance_max_abs * max_abs_decay, std::abs(new_val));
    }
    inline void update_value_stats(value_type old_val, value_type new_val) {
        value_l1 += std::abs(static_cast<double>(new_val)) - std::abs(static_cast<double>(old_val));
        value_l2_sq +=
            static_cast<double>(new_val) * new_val - static_cast<double>(old_val) * old_val;
        value_max_abs = std::max(value_max_abs * max_abs_decay, std::abs(new_val));
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
    inline void update_importance_stats_aggregate(double sum_abs_new, double sum_abs_old,
                                                  double sum_sq_new, double sum_sq_old,
                                                  value_type local_max_new) {
        importance_l1 += sum_abs_new - sum_abs_old;
        importance_l2_sq += sum_sq_new - sum_sq_old;
        importance_max_abs = std::max(importance_max_abs * max_abs_decay, local_max_new);
    }
    inline void update_value_stats_aggregate(double sum_abs_new, double sum_abs_old,
                                             double sum_sq_new, double sum_sq_old,
                                             value_type local_max_new) {
        value_l1 += sum_abs_new - sum_abs_old;
        value_l2_sq += sum_sq_new - sum_sq_old;
        value_max_abs = std::max(value_max_abs * max_abs_decay, local_max_new);
    }

    // Recompute all six stats from scratch -- O(nnz), call once after
    // constructing a layer via delta_csr_from_absolute() or any path that
    // writes values without going through update_*_stats(), or whenever an
    // exact (not monotonic-bound) max_abs is needed.
    inline void recompute_stats() {
        importance_l1 = importance_l2_sq = 0.0;
        importance_max_abs = value_type(0);
        value_l1 = value_l2_sq = 0.0;
        value_max_abs = value_type(0);
        const auto& L = connections.layout;
        for (std::size_t r = 0; r < L.rows; ++r) {
            const std::size_t n = L.row_nnz(r);
            for (std::size_t e = 0; e < n; ++e) {
                const std::size_t vb = L.elem_start[r] + e;
                update_value_stats(value_type(0),
                                   ValueAccessor<VALUES_TYPE>::get_w(connections.values, vb));
                update_importance_stats(
                    value_type(0), ValueAccessor<VALUES_TYPE>::get_imp(connections.values, vb));
            }
        }
    }

    inline SIZE_TYPE in_degree(SIZE_TYPE i) const {
        return static_cast<SIZE_TYPE>(connections.layout.row_nnz(i));
    }

  private:
    inline value_type _hoyer_from_stats(double l1, double l2_sq) const {
        const std::size_t n = connections.nnz();
        if (n <= 1)
            return value_type(0);
        const double l2 = std::sqrt(l2_sq);
        if (l2 <= 0.0)
            return value_type(1); // all-zero -> maximally "sparse" by convention
        const double sqrt_n = std::sqrt(static_cast<double>(n));
        return static_cast<value_type>((sqrt_n - l1 / l2) / (sqrt_n - 1.0));
    }

  public:
    // Change ONE row's importance_scale mid-training without losing
    // accumulated importance: re-reads stored importance into true units
    // at the OLD scale, re-encodes at the target scale. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.rescale_row.
    inline void rescale_importance_row(std::size_t row, value_type new_scale) {
        const value_type old_scale = get_importance_scale(row);
        if (new_scale == old_scale)
            return;
        auto& dc = connections;
        const auto& L = dc.layout;
        if (row >= L.rows)
            return;
        const std::size_t n = L.row_nnz(row);
        for (std::size_t e = 0; e < n; ++e) {
            const std::size_t vb = L.elem_start[row] + e;
            const value_type w = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
            const value_type stored_i = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
            const value_type true_i = stored_i * old_scale;
            // Plain set(), not live -- reparametrization, not a training
            // update (see rescale_row's own RST anchor above).
            ValueAccessor<VALUES_TYPE>::set(dc.values, vb, w, true_i / new_scale);
        }
        set_importance_scale_raw(row, new_scale);
    }

    // Bulk convenience: set EVERY row to the same new_scale.
    inline void rescale_importance(value_type new_scale) {
        const auto& L = connections.layout;
        for (std::size_t r = 0; r < L.rows; ++r)
            rescale_importance_row(r, new_scale);
    }

    // Same pattern, for STORED weight values instead of importance.
    inline void rescale_value_row(std::size_t row, value_type new_scale) {
        const value_type old_scale = get_value_scale(row);
        if (new_scale == old_scale)
            return;
        auto& dc = connections;
        const auto& L = dc.layout;
        if (row >= L.rows)
            return;
        const std::size_t n = L.row_nnz(row);
        for (std::size_t e = 0; e < n; ++e) {
            const std::size_t vb = L.elem_start[row] + e;
            const value_type stored_w = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
            const value_type imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
            const value_type true_w = stored_w * old_scale;
            ValueAccessor<VALUES_TYPE>::set(dc.values, vb, true_w / new_scale, imp);
        }
        set_value_scale_raw(row, new_scale);
    }
    inline void rescale_value(value_type new_scale) {
        const auto& L = connections.layout;
        for (std::size_t r = 0; r < L.rows; ++r)
            rescale_value_row(r, new_scale);
    }

    // Gradient-free reparametrization: true_weight = stored_w *
    // value_scale[row] * output_scale[col] is algebraically UNCHANGED --
    // only WHERE the magnitude lives moves, into the stored per-synapse
    // code. Drives each column's stored-weight RMS toward `target` via a
    // DAMPED (correction_rate) step. Covers BOTH storages (scattered CSR
    // and block4) -- rescaling only one side would corrupt the other,
    // since both read the SAME output_scale[col]. Re-quantization is
    // DETERMINISTIC, not gradient-driven stochastic. See
    // docs/research/delta_csr_types.rst:sparse_linear_weights_delta.magnitude_rescale_output
    // for the scale_invariant/ci interaction and the column-RMS-over-n_in
    // convention.
    inline void magnitude_rescale_output(value_type target, value_type correction_rate,
                                         bool scale_invariant, value_type eps = value_type(1e-8)) {
        auto& dc = connections;
        const auto& L = dc.layout;
        const std::size_t n_out = L.cols;
        const std::size_t n_in = L.rows;
        if (n_out == 0 || n_in == 0)
            return;

        std::vector<double> sum_sq(n_out, 0.0);
        std::vector<std::size_t> col_count(n_out, 0);
        for (std::size_t r = 0; r < n_in; ++r) {
            const std::size_t n = L.row_nnz(r);
            if (n == 0)
                continue;
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
            if (n_bc == 0)
                continue;
            auto bc_cursor = block4.row_cursor(uint32_t(br));
            for (std::size_t bk = 0; bk < n_bc; ++bk) {
                const uint32_t bc = bc_cursor.advance();
                const auto tile = block4.find(uint32_t(br), bc);
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                    if (col >= n_out)
                        continue;
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = br * BLOCK4_TILE + li;
                        if (row >= n_in)
                            continue;
                        value_type w;
                        if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                            const uint8_t w_byte = tile.at_weight(li, lj);
                            const uint8_t i_byte = tile.at_importance(li, lj);
                            if (w_byte == 0 && i_byte == 0)
                                continue; // empty slot
                            w = fp8_decode_bits(w_byte);
                        } else {
                            const uint8_t byte = tile.at(li, lj);
                            if (byte == 0)
                                continue; // empty slot
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
            if (col_count[c] == 0)
                continue; // nothing to rescale here
            const double mean_sq = sum_sq[c] / static_cast<double>(n_in);
            const value_type col_rms =
                static_cast<value_type>(std::sqrt(mean_sq + static_cast<double>(eps)));
            if (!std::isfinite(col_rms) || col_rms <= value_type(0))
                continue;
            value_type kc = target / col_rms;
            if (kc < value_type(1e-6))
                kc = value_type(1e-6);
            kc = std::pow(kc, correction_rate);
            if (!std::isfinite(kc) || kc <= value_type(0))
                continue;
            k[c] = kc;
        }

        for (std::size_t r = 0; r < n_in; ++r) {
            const std::size_t n = L.row_nnz(r);
            if (n == 0)
                continue;
            auto cursor = dc.row_cursor(r);
            for (std::size_t e = 0; e < n; ++e) {
                const COL_TYPE col = cursor.advance();
                if (k[col] == value_type(1))
                    continue;
                const std::size_t vb = L.elem_start[r] + e;
                const value_type w = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                const value_type imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                const value_type new_w = w * k[col];
                const value_type new_imp = scale_invariant ? imp : imp * k[col] * k[col];
                if (!std::isfinite(new_w) || !std::isfinite(new_imp))
                    continue;
                ValueAccessor<VALUES_TYPE>::set_live(dc.values, vb, new_w, new_imp);
            }
        }
        for (std::size_t br = 0; br < BL.rows; ++br) {
            const std::size_t n_bc = BL.row_nnz(br);
            if (n_bc == 0)
                continue;
            auto bc_cursor = block4.row_cursor(uint32_t(br));
            for (std::size_t bk = 0; bk < n_bc; ++bk) {
                const uint32_t bc = bc_cursor.advance();
                // Any column in this tile need rescaling? Skip the whole
                // tile (no mutable handle, no dirty/re-pack cost) if not.
                bool any_col_touched = false;
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                    if (col < n_out && k[col] != value_type(1)) {
                        any_col_touched = true;
                        break;
                    }
                }
                if (!any_col_touched)
                    continue;
                auto tile = block4.find(uint32_t(br), bc);
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                    if (col >= n_out || k[col] == value_type(1))
                        continue;
                    for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                        const std::size_t row = br * BLOCK4_TILE + li;
                        if (row >= n_in)
                            continue;
                        if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                            const uint8_t w_byte = tile.at_weight(li, lj);
                            const uint8_t i_byte = tile.at_importance(li, lj);
                            if (w_byte == 0 && i_byte == 0)
                                continue;
                            const value_type w = fp8_decode_bits(w_byte);
                            const value_type imp = fp8_decode_bits(i_byte);
                            const value_type new_w = w * k[col];
                            const value_type new_imp =
                                scale_invariant ? imp : imp * k[col] * k[col];
                            if (!std::isfinite(new_w) || !std::isfinite(new_imp))
                                continue;
                            tile.at_weight(li, lj) = fp8_quantize_live(new_w);
                            tile.at_importance(li, lj) = fp8_quantize_live(new_imp);
                        } else {
                            const uint8_t byte = tile.at(li, lj);
                            if (byte == 0)
                                continue;
                            const value_type w = FP4_TABLE[byte & 0xFu];
                            const value_type imp = FP4_TABLE[(byte >> 4) & 0xFu];
                            const value_type new_w = w * k[col];
                            const value_type new_imp =
                                scale_invariant ? imp : imp * k[col] * k[col];
                            if (!std::isfinite(new_w) || !std::isfinite(new_imp))
                                continue;
                            tile.at(li, lj) = uint8_t(fp4_quantize_live(new_w) |
                                                      (fp4_quantize_live(new_imp) << 4));
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
            if (k[c] == value_type(1))
                continue;
            for (std::size_t ki = 0; ki < scale_rank; ++ki) {
                const value_type new_os = get_output_scale_k(c, ki) / k[c];
                if (!std::isfinite(new_os))
                    continue;
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

    inline void reserve_values(std::size_t target_nnz) { connections.reserve_values(target_nnz); }
};

#endif
