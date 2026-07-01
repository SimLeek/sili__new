/**
 * @file sparse_matrix.hpp
 * @brief Sparse matrix library with CSR and COO format support.
 */

#ifndef __SPARSE_STRUCT_HPP_
#define __SPARSE_STRUCT_HPP_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <omp.h>
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

    inline SIZE_TYPE in_degree(SIZE_TYPE i) const {
        return static_cast<SIZE_TYPE>(connections.layout.row_nnz(i));
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

// ── Build from / convert to absolute CSR ─────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> delta_csr_from_absolute(
    const std::vector<SIZE_TYPE>&   csr_ptrs,
    const std::vector<SIZE_TYPE>&   csr_indices,
    const std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& csr_weights,
    const std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& csr_importance,
    std::size_t rows, std::size_t cols,
    std::size_t index_bytes, std::size_t values_bytes)
{
    // Let the shifting and gradual modification handle blank space fragmenting
    // We don't know row sizes or byte lengths for each index, so if we try to evenly assign spaces, we could run out of memory
    // Todo: We can guarantee we won't by assuming uleb128_max_bytes indices each time if needed, but this is a rarely used function.
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> dc;
    dc.reserve_values(values_bytes);
    dc.reserve_indices(index_bytes);

    auto& L = dc.layout;
    L.rows = rows;
    L.cols = cols;
    L.byte_start.resize(rows + 1);
    L.byte_end  .resize(rows);
    L.elem_start.resize(rows + 1);
    L.elem_end  .resize(rows);

    uint8_t tmp[uleb128_max_bytes<COL_TYPE>()];

    std::vector<std::size_t> row_bytes(rows, 0);
    for (std::size_t r = 0; r < rows; ++r) {
        COL_TYPE prev = 0;
        for (SIZE_TYPE i = csr_ptrs[r]; i < csr_ptrs[r + 1]; ++i) {
            COL_TYPE col = static_cast<COL_TYPE>(csr_indices[i]);
            row_bytes[r] += uleb128_encode<COL_TYPE>(col - prev, tmp);
            prev = col;
        }
    }


    std::size_t blank_fraction = 
    L.byte_start[0] = 0;
    L.elem_start[0] = 0;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t n = csr_ptrs[r + 1] - csr_ptrs[r];
        L.byte_end[r]       = L.byte_start[r] + row_bytes[r];
        const std::size_t byte_blank = static_cast<std::size_t>(row_bytes[r] * blank_fraction)
                                       + uleb128_max_bytes<COL_TYPE>(); 
        L.byte_start[r + 1] = L.byte_end[r] + byte_blank;
        
        L.elem_end[r]       = L.elem_start[r] + n;
        const std::size_t elem_blank = std::max(std::size_t(1),
                                                static_cast<std::size_t>(n * blank_fraction));
        L.elem_start[r + 1] = L.elem_end[r] + elem_blank;
    }
    L.total_nnz = static_cast<std::size_t>(csr_ptrs[rows]);

    dc.indices_buf.assign(L.byte_start[rows], uint8_t(0));
    ValueAccessor<VALUES_TYPE>::resize(dc.values, L.elem_start[rows], value_type(0));

    const std::size_t byte_headroom = static_cast<std::size_t>(L.byte_start[rows] * (1.0f + blank_fraction));
    const std::size_t elem_headroom = static_cast<std::size_t>(L.elem_start[rows] * (1.0f + blank_fraction));
    dc.indices_buf.reserve(byte_headroom);
    ValueAccessor<VALUES_TYPE>::reserve(dc.values, elem_headroom);

    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t bpos = L.byte_start[r];
        COL_TYPE prev = 0;
        std::size_t epos = L.elem_start[r];
        for (SIZE_TYPE i = csr_ptrs[r]; i < csr_ptrs[r + 1]; ++i) {
            COL_TYPE col = static_cast<COL_TYPE>(csr_indices[i]);
            bpos += uleb128_encode<COL_TYPE>(col - prev, dc.indices_buf.data() + bpos);
            prev = col;
            
            ValueAccessor<VALUES_TYPE>::set(dc.values, epos, csr_weights[i], csr_importance[i]);
            ++epos;
        }
    }

    return dc;
}

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_to_absolute(
    const DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc,
    std::vector<SIZE_TYPE>&  out_ptrs,
    std::vector<SIZE_TYPE>&  out_indices,
    std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& out_weights,
    std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& out_importance)
{
    const auto& L   = dc.layout;
    const std::size_t nnz = L.total_nnz;

    out_ptrs     .resize(L.rows + 1);
    out_indices  .resize(nnz);
    out_weights  .resize(nnz);
    out_importance.resize(nnz);

    out_ptrs[0] = 0;
    std::size_t flat = 0;
    for (std::size_t r = 0; r < L.rows; ++r) {
        auto cursor = dc.row_cursor(r);
        const std::size_t n = L.row_nnz(r);
        for (std::size_t k = 0; k < n; ++k, ++flat) {
            out_indices[flat]    = static_cast<SIZE_TYPE>(cursor.advance());
            out_weights[flat]    = ValueAccessor<VALUES_TYPE>::get_w(dc.values, L.elem_start[r] + k);
            out_importance[flat] = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, L.elem_start[r] + k);
        }
        out_ptrs[r + 1] = static_cast<SIZE_TYPE>(flat);
    }
}

// ── Blank-space management ────────────────────────────────────────────────────

inline std::size_t delta_csr_target_alloc_bytes(const DeltaCSRLayout& L) {
    return L.rows > 0 ? (L.total_alloc_bytes() + L.rows - 1) / L.rows : 0;
}
inline std::size_t delta_csr_target_alloc_elems(const DeltaCSRLayout& L) {
    return L.rows > 0 ? (L.total_alloc_elems() + L.rows - 1) / L.rows : 0;
}

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_shift_row(
    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc,
    std::size_t row,
    std::size_t target_byte_alloc,
    std::size_t target_elem_alloc)
{
    auto& L    = dc.layout;
    auto& ibuf = dc.indices_buf;

    // ── byte side ────────────────────────────────────────────────────────────
    const std::size_t cur_byte_alloc = L.row_alloc_bytes(row);
    if (cur_byte_alloc != target_byte_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = L.byte_start[row + 1];
        const std::size_t move_len  = L.byte_start[L.rows] - move_src;
        const std::size_t new_start = L.byte_start[row] + target_byte_alloc;

        if (target_byte_alloc > cur_byte_alloc) {
            ibuf.resize(ibuf.size() + (target_byte_alloc - cur_byte_alloc));
        }
        if (move_len > 0)
            std::memmove(ibuf.data() + new_start, ibuf.data() + move_src, move_len);
        if (target_byte_alloc < cur_byte_alloc)
            ibuf.resize(ibuf.size() - (cur_byte_alloc - target_byte_alloc));

        const std::ptrdiff_t byte_delta =
            static_cast<std::ptrdiff_t>(target_byte_alloc) -
            static_cast<std::ptrdiff_t>(cur_byte_alloc);
        for (std::size_t r = row + 1; r <= L.rows; ++r)
            L.byte_start[r] = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(L.byte_start[r]) + byte_delta);
    }

    // ── element side ─────────────────────────────────────────────────────────
    const std::size_t cur_elem_alloc = L.row_alloc_elems(row);
    if (cur_elem_alloc != target_elem_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = L.elem_start[row + 1];
        const std::size_t move_len  = L.elem_start[L.rows] - move_src;
        const std::size_t new_start = L.elem_start[row] + target_elem_alloc;
        const std::size_t current_total = L.total_alloc_elems();

        if (target_elem_alloc > cur_elem_alloc) {
            ValueAccessor<VALUES_TYPE>::resize(dc.values, current_total + (target_elem_alloc - cur_elem_alloc));
        }
        
        ValueAccessor<VALUES_TYPE>::move(dc.values, new_start, move_src, move_len);
        
        if (target_elem_alloc < cur_elem_alloc) {
            ValueAccessor<VALUES_TYPE>::resize(dc.values, current_total - (cur_elem_alloc - target_elem_alloc));
        }

        const std::ptrdiff_t elem_delta =
            static_cast<std::ptrdiff_t>(target_elem_alloc) -
            static_cast<std::ptrdiff_t>(cur_elem_alloc);
        for (std::size_t r = row + 1; r <= L.rows; ++r)
            L.elem_start[r] = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(L.elem_start[r]) + elem_delta);
    }
}

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_equalize_step(
    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc,
    std::size_t& current_row)
{
    if (dc.layout.rows == 0) return;
    const std::size_t row        = current_row % dc.layout.rows;
    const std::size_t target_b   = delta_csr_target_alloc_bytes(dc.layout);
    const std::size_t target_e   = delta_csr_target_alloc_elems(dc.layout);
    delta_csr_shift_row(dc, row, target_b, target_e);
    current_row = (current_row + 1) % dc.layout.rows;
}

// ── Row-level insert / remove (for incremental synaptogenesis) ────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
COL_TYPE delta_csr_row_last_col(
    const DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc, std::size_t row)
{
    if (dc.layout.row_nnz(row) == 0) return 0;
    auto cursor = dc.row_cursor(row);
    const std::size_t n = dc.layout.row_nnz(row);
    for (std::size_t k = 0; k < n; ++k) cursor.advance();
    return cursor.col();
}

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
bool delta_csr_row_rebuild(
    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc, std::size_t row,
    const std::vector<COL_TYPE>& cols,
    const std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& weights,
    const std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& importance)
{
    auto& L = dc.layout;
    const std::size_t n = cols.size();

    uint8_t tmp[uleb128_max_bytes<COL_TYPE>()];
    std::size_t needed_bytes = 0;
    COL_TYPE prev = 0;
    for (std::size_t k = 0; k < n; ++k) {
        needed_bytes += uleb128_encode<COL_TYPE>(cols[k] - prev, tmp);
        prev = cols[k];
    }
    if (needed_bytes > L.row_alloc_bytes(row)) return false;
    if (n > L.row_alloc_elems(row))            return false;

    std::size_t bpos = L.byte_start[row];
    prev = 0;
    for (std::size_t k = 0; k < n; ++k) {
        bpos += uleb128_encode<COL_TYPE>(cols[k] - prev, dc.indices_buf.data() + bpos);
        prev = cols[k];
    }
    L.byte_end[row] = bpos;

    std::size_t epos = L.elem_start[row];
    for (std::size_t k = 0; k < n; ++k) {
        ValueAccessor<VALUES_TYPE>::set(dc.values, epos + k, weights[k], importance[k]);
    }
    L.elem_end[row] = L.elem_start[row] + n;

    const std::ptrdiff_t nnz_delta =
        static_cast<std::ptrdiff_t>(n) -
        static_cast<std::ptrdiff_t>(dc.layout.row_nnz(row));
    L.total_nnz = static_cast<std::size_t>(
        static_cast<std::ptrdiff_t>(L.total_nnz) + nnz_delta);

    return true;
}

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
bool delta_csr_row_append(
    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc, std::size_t row,
    COL_TYPE col, 
    typename ValueAccessor<VALUES_TYPE>::value_type weight, 
    typename ValueAccessor<VALUES_TYPE>::value_type imp)
{
    auto& L = dc.layout;
    const COL_TYPE prev_col = delta_csr_row_last_col(dc, row);
    assert(col >= prev_col && "delta_csr_row_append: column not in sorted order");

    uint8_t tmp[uleb128_max_bytes<COL_TYPE>()];
    const std::size_t nbytes = uleb128_encode<COL_TYPE>(col - prev_col, tmp);

    if (nbytes > L.row_blank_bytes(row)) return false;
    if (L.row_blank_elems(row) == 0)    return false;

    std::memcpy(dc.indices_buf.data() + L.byte_end[row], tmp, nbytes);
    L.byte_end[row] += nbytes;

    const std::size_t epos = L.elem_end[row];
    ValueAccessor<VALUES_TYPE>::set(dc.values, epos, weight, imp);
    L.elem_end[row]++;

    L.total_nnz++;
    return true;
}

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_row_remove(
    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc, std::size_t row,
    std::size_t elem_within_row)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& L = dc.layout;
    const std::size_t n = L.row_nnz(row);
    assert(elem_within_row < n);

    std::vector<COL_TYPE>   cols(n);
    std::vector<value_type> weights(n), importance(n);
    auto cursor = dc.row_cursor(row);
    for (std::size_t k = 0; k < n; ++k) {
        cols[k]       = cursor.advance();
        weights[k]    = ValueAccessor<VALUES_TYPE>::get_w(dc.values, L.elem_start[row] + k);
        importance[k] = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, L.elem_start[row] + k);
    }

    cols      .erase(cols      .begin() + static_cast<std::ptrdiff_t>(elem_within_row));
    weights   .erase(weights   .begin() + static_cast<std::ptrdiff_t>(elem_within_row));
    importance.erase(importance.begin() + static_cast<std::ptrdiff_t>(elem_within_row));

    delta_csr_row_rebuild(dc, row, cols, weights, importance);
}

// ── Incremental synaptogenesis step ──────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
bool delta_csr_synap_row_step(
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    std::size_t& current_row,
    typename ValueAccessor<VALUES_TYPE>::value_type importance_cutoff,
    SIZE_TYPE max_row_weights)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    if (!weights.probes.indices[0] || weights.probes.indices[0]->empty()) return false;

    auto& dc = weights.connections;
    auto& L  = dc.layout;
    if (L.rows == 0) return false;

    const std::size_t row   = current_row % L.rows;
    current_row = (current_row + 1) % L.rows;

    const std::size_t n_exist = L.row_nnz(row);
    std::vector<COL_TYPE>   exist_cols(n_exist);
    std::vector<value_type> exist_w(n_exist), exist_imp(n_exist);
    {
        auto cursor = dc.row_cursor(row);
        for (std::size_t k = 0; k < n_exist; ++k) {
            exist_cols[k] = cursor.advance();
            exist_w[k]    = ValueAccessor<VALUES_TYPE>::get_w(dc.values, L.elem_start[row] + k);
            exist_imp[k]  = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, L.elem_start[row] + k);
        }
    }

    std::vector<COL_TYPE>   probe_cols;
    std::vector<value_type> probe_imp;
    {
        const auto& prow = *weights.probes.indices[0];
        const auto& pcol = *weights.probes.indices[1];
        const auto& pval = *weights.probes.values[0];
        const SIZE_TYPE pnnz = weights.probes.nnz();
        for (SIZE_TYPE p = 0; p < pnnz; ++p) {
            if (static_cast<std::size_t>(prow[p]) == row) {
                probe_cols.push_back(static_cast<COL_TYPE>(pcol[p]));
                probe_imp .push_back(pval[p]);
            }
        }
    }

    if (probe_cols.empty() && n_exist == 0) return false;

    std::vector<COL_TYPE>   merged_cols;
    std::vector<value_type> merged_w, merged_imp;
    merged_cols.reserve(n_exist + probe_cols.size());
    merged_w   .reserve(n_exist + probe_cols.size());
    merged_imp .reserve(n_exist + probe_cols.size());

    std::size_t ei = 0, pi = 0;
    while (ei < n_exist || pi < probe_cols.size()) {
        const COL_TYPE ec = (ei < n_exist)           ? exist_cols[ei]  : std::numeric_limits<COL_TYPE>::max();
        const COL_TYPE pc = (pi < probe_cols.size()) ? probe_cols[pi]  : std::numeric_limits<COL_TYPE>::max();
        
        if (ec < pc) {
            merged_cols.push_back(ec);
            merged_w   .push_back(exist_w[ei]);
            merged_imp .push_back(exist_imp[ei]);
            ++ei;
        } else if (pc < ec) {
            merged_cols.push_back(pc);
            merged_w   .push_back(value_type(0));
            merged_imp .push_back(probe_imp[pi]);
            ++pi;
        } else {
            merged_cols.push_back(ec);
            merged_w   .push_back(exist_w[ei]);
            merged_imp .push_back(std::max(exist_imp[ei], probe_imp[pi]));
            ++ei; ++pi;
        }
    }

    {
        std::vector<std::size_t> keep_idx;
        keep_idx.reserve(merged_cols.size());
        for (std::size_t k = 0; k < merged_cols.size(); ++k)
            if (merged_imp[k] >= importance_cutoff)
                keep_idx.push_back(k);

        if (keep_idx.size() > static_cast<std::size_t>(max_row_weights)) {
            std::partial_sort(keep_idx.begin(),
                              keep_idx.begin() + max_row_weights,
                              keep_idx.end(),
                              [&](std::size_t a, std::size_t b){
                                  return merged_imp[a] > merged_imp[b];
                              });
            keep_idx.resize(max_row_weights);
            std::sort(keep_idx.begin(), keep_idx.end());
        }

        std::vector<COL_TYPE>   final_cols;
        std::vector<value_type> final_w, final_imp;
        final_cols.reserve(keep_idx.size());
        final_w   .reserve(keep_idx.size());
        final_imp .reserve(keep_idx.size());
        for (std::size_t k : keep_idx) {
            final_cols.push_back(merged_cols[k]);
            final_w   .push_back(merged_w   [k]);
            final_imp .push_back(merged_imp [k]);
        }
        merged_cols = std::move(final_cols);
        merged_w    = std::move(final_w);
        merged_imp  = std::move(final_imp);
    }

    delta_csr_row_rebuild(dc, row, merged_cols, merged_w, merged_imp);

    if (!weights.out_degree.empty()) {
        for (std::size_t k = 0; k < n_exist; ++k)
            if (weights.out_degree[exist_cols[k]] > 0)
                --weights.out_degree[exist_cols[k]];
        for (COL_TYPE col : merged_cols)
            ++weights.out_degree[col];
    }

    return true;
}

// ── Forward pass ─────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_forward(
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& input_tensor,
    const SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    typename ValueAccessor<VALUES_TYPE>::value_type* output,
    typename ValueAccessor<VALUES_TYPE>::value_type   learning_rate = 0.01f,
    const int    num_cpus = 4,
    typename ValueAccessor<VALUES_TYPE>::value_type* original_contributions_output = nullptr)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    const auto& dc = weights.connections;
    if (dc.empty()) return;

    const auto& L           = dc.layout;
    const std::size_t out_cols    = L.cols;
    const std::size_t num_outputs = static_cast<std::size_t>(input_tensor.rows) * out_cols;
    const std::size_t num_inputs  = L.rows;

    std::vector<value_type> all_outputs(static_cast<std::size_t>(num_cpus) * num_outputs,
                                        value_type(0));
    std::vector<value_type> all_contributions(
        original_contributions_output
            ? static_cast<std::size_t>(num_cpus) * num_inputs : 0,
        value_type(0));

    std::vector<SIZE_TYPE> work_offsets;

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid      = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        value_type* thread_output = all_outputs.data() + static_cast<std::size_t>(tid) * num_outputs;
        value_type* thread_contrib = original_contributions_output
            ? all_contributions.data() + static_cast<std::size_t>(tid) * num_inputs
            : nullptr;

        for (SIZE_TYPE batch = 0; batch < input_tensor.rows; ++batch) {
            const SIZE_TYPE batch_start  = (*input_tensor.ptrs[0])[batch];
            const SIZE_TYPE batch_end    = (*input_tensor.ptrs[0])[batch + 1];
            const SIZE_TYPE batch_nnz    = batch_end - batch_start;
            const SIZE_TYPE batch_offset = batch * static_cast<SIZE_TYPE>(out_cols);

            #pragma omp single
            {
                work_offsets.resize(batch_nnz + 1);
                work_offsets[0] = 0;
                for (SIZE_TYPE i = 0; i < batch_nnz; ++i) {
                    const SIZE_TYPE in_idx = (*input_tensor.indices[0])[batch_start + i];
                    work_offsets[i + 1] = work_offsets[i]
                        + static_cast<SIZE_TYPE>(L.row_nnz(in_idx));
                }
            }

            const SIZE_TYPE total_work = work_offsets[batch_nnz];
            const SIZE_TYPE chunk      = (total_work + nthreads - 1) / nthreads;
            const SIZE_TYPE w_start    = std::min(static_cast<SIZE_TYPE>(tid) * chunk, total_work);
            const SIZE_TYPE w_end      = std::min(w_start + chunk, total_work);

            if (w_start < w_end) {
                SIZE_TYPE ip = static_cast<SIZE_TYPE>(
                    std::upper_bound(work_offsets.begin(), work_offsets.end(), w_start)
                    - work_offsets.begin()) - 1;

                SIZE_TYPE last_ip = std::numeric_limits<SIZE_TYPE>::max();
                DeltaCSRRowCursor<COL_TYPE> cursor;

                for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                    while (ip + 1 < batch_nnz && work_offsets[ip + 1] <= w) ++ip;

                    const SIZE_TYPE  in_idx      = (*input_tensor.indices[0])[batch_start + ip];
                    const value_type in_val      = (*input_tensor.values[0]) [batch_start + ip];
                    const SIZE_TYPE  elem_offset = w - work_offsets[ip];

                    if (ip != last_ip) {
                        cursor  = DeltaCSRRowCursor<COL_TYPE>(dc.indices_buf.data(), L, in_idx);
                        cursor.advance_to(elem_offset);
                        last_ip = ip;
                    } else {
                        cursor.advance();
                    }

                    const SIZE_TYPE   out_idx = static_cast<SIZE_TYPE>(cursor.col());
                    const std::size_t wptr    = L.elem_start[in_idx] + elem_offset;
                    const value_type  wval    = ValueAccessor<VALUES_TYPE>::get_w(dc.values, wptr);
                    const value_type  contrib = wval * in_val;

                    if (learning_rate != 0) {
                        value_type cur_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, wptr);
                        cur_imp += contrib * learning_rate / (value_type(1) + std::abs(cur_imp));
                        ValueAccessor<VALUES_TYPE>::set(dc.values, wptr, wval, cur_imp);
                    }

                    thread_output[batch_offset + out_idx] += contrib;
                    if (thread_contrib)
                        thread_contrib[in_idx] += in_val * wval;
                }
            }
            #pragma omp barrier
        }

        for (int stride = 1; stride < nthreads; stride <<= 1) {
            #pragma omp barrier
            const int src = tid + stride;
            if (tid % (stride << 1) == 0 && src < nthreads) {
                const value_type* src_out = all_outputs.data() +
                                            static_cast<std::size_t>(src) * num_outputs;
                for (std::size_t i = 0; i < num_outputs; ++i)
                    thread_output[i] += src_out[i];
                if (thread_contrib) {
                    const value_type* src_con =
                        all_contributions.data() + static_cast<std::size_t>(src) * num_inputs;
                    for (std::size_t i = 0; i < num_inputs; ++i)
                        thread_contrib[i] += src_con[i];
                }
            }
        }
    }

    for (std::size_t i = 0; i < num_outputs; ++i)
        output[i] += all_outputs[i];
    if (original_contributions_output)
        for (std::size_t i = 0; i < num_inputs; ++i)
            original_contributions_output[i] += all_contributions[i];
}

// ── Backward pass ─────────────────────────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_backward(
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& in_tensor,
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    const CSRInput<SIZE_TYPE, typename ValueAccessor<VALUES_TYPE>::value_type>& out_grad_sparse,
    typename ValueAccessor<VALUES_TYPE>::value_type* input_gradients,
    typename ValueAccessor<VALUES_TYPE>::value_type* output_gradients,
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_input_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type* neuron_grad_accum,
    typename ValueAccessor<VALUES_TYPE>::value_type   learning_rate = 0.01f,
    const int    num_cpus = 4)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    
    for (SIZE_TYPE i = 0; i < in_tensor.rows; ++i)
        for (SIZE_TYPE j = (*in_tensor.ptrs[0])[i]; j < (*in_tensor.ptrs[0])[i+1]; ++j)
            neuron_input_accum[(*in_tensor.indices[0])[j]] +=
                std::abs((*in_tensor.values[0])[j]);

    for (SIZE_TYPE i = 0; i < out_grad_sparse.rows; ++i)
        for (SIZE_TYPE j = (*out_grad_sparse.ptrs[0])[i]; j < (*out_grad_sparse.ptrs[0])[i+1]; ++j)
            neuron_grad_accum[(*out_grad_sparse.indices[0])[j]] +=
                std::abs((*out_grad_sparse.values[0])[j]);

    auto& dc = weights.connections;
    if (dc.empty()) return;

    const auto& L           = dc.layout;
    const SIZE_TYPE batch_size  = in_tensor.rows;
    const std::size_t num_inputs  = L.rows;

    std::vector<SIZE_TYPE> weight_grad_offsets;
    std::vector<SIZE_TYPE> input_grad_offsets;

    #pragma omp parallel num_threads(num_cpus)
    {
        const int tid      = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

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
                        + static_cast<SIZE_TYPE>(L.row_nnz(in_idx));
                }
                
                input_grad_offsets.resize(num_inputs + 1);
                input_grad_offsets[0] = 0;
                for (std::size_t i = 0; i < num_inputs; ++i)
                    input_grad_offsets[i + 1] = input_grad_offsets[i]
                        + static_cast<SIZE_TYPE>(L.row_nnz(i));
            }

            // ── weight update ─────────────────────────────────────────────────
            {
                const SIZE_TYPE total_work = weight_grad_offsets[batch_nnz];
                const SIZE_TYPE chunk      = (total_work + nthreads - 1) / nthreads;
                const SIZE_TYPE w_start    = std::min(static_cast<SIZE_TYPE>(tid) * chunk, total_work);
                const SIZE_TYPE w_end      = std::min(w_start + chunk, total_work);

                if (w_start < w_end) {
                    SIZE_TYPE ip = static_cast<SIZE_TYPE>(
                        std::upper_bound(weight_grad_offsets.begin(),
                                         weight_grad_offsets.end(), w_start)
                        - weight_grad_offsets.begin()) - 1;

                    SIZE_TYPE last_ip = std::numeric_limits<SIZE_TYPE>::max();
                    DeltaCSRRowCursor<COL_TYPE> cursor;

                    for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                        while (ip + 1 < batch_nnz && weight_grad_offsets[ip + 1] <= w) ++ip;

                        const SIZE_TYPE  in_idx      = (*in_tensor.indices[0])[batch_start + ip];
                        const value_type in_val      = (*in_tensor.values[0]) [batch_start + ip];
                        const SIZE_TYPE  elem_offset = w - weight_grad_offsets[ip];

                        if (ip != last_ip) {
                            cursor  = DeltaCSRRowCursor<COL_TYPE>(dc.indices_buf.data(), L, in_idx);
                            cursor.advance_to(elem_offset);
                            last_ip = ip;
                        } else {
                            cursor.advance();
                        }

                        const SIZE_TYPE   out_idx = static_cast<SIZE_TYPE>(cursor.col());
                        const std::size_t wptr    = L.elem_start[in_idx] + elem_offset;
                        const value_type  grad    = output_gradients[out_idx * batch_size + batch] * in_val;

                        value_type cur_w   = ValueAccessor<VALUES_TYPE>::get_w(dc.values, wptr);
                        value_type cur_imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, wptr);

                        cur_imp -= grad * learning_rate;
                        cur_w   += (-learning_rate * grad) / (value_type(1) + std::abs(cur_imp));

                        ValueAccessor<VALUES_TYPE>::set(dc.values, wptr, cur_w, cur_imp);
                    }
                }
            }

            // ── input gradients ───────────────────────────────────────────────
            {
                const SIZE_TYPE total_work = input_grad_offsets[num_inputs];
                const SIZE_TYPE chunk      = (total_work + nthreads - 1) / nthreads;
                const SIZE_TYPE w_start    = std::min(static_cast<SIZE_TYPE>(tid) * chunk, total_work);
                const SIZE_TYPE w_end      = std::min(w_start + chunk, total_work);

                if (w_start < w_end) {
                    SIZE_TYPE in_idx = static_cast<SIZE_TYPE>(
                        std::upper_bound(input_grad_offsets.begin(),
                                         input_grad_offsets.end(), w_start)
                        - input_grad_offsets.begin()) - 1;

                    SIZE_TYPE last_in_idx = std::numeric_limits<SIZE_TYPE>::max();
                    DeltaCSRRowCursor<COL_TYPE> cursor;
                    SIZE_TYPE og_ptr = og_start;

                    for (SIZE_TYPE w = w_start; w < w_end; ++w) {
                        while (in_idx + 1 < static_cast<SIZE_TYPE>(num_inputs)
                               && input_grad_offsets[in_idx + 1] <= w) {
                            ++in_idx;
                            og_ptr = og_start;
                        }

                        const SIZE_TYPE elem_offset = w - input_grad_offsets[in_idx];

                        if (in_idx != last_in_idx) {
                            cursor = DeltaCSRRowCursor<COL_TYPE>(dc.indices_buf.data(), L, in_idx);
                            cursor.advance_to(elem_offset);
                            last_in_idx = in_idx;
                        } else {
                            cursor.advance();
                        }

                        const SIZE_TYPE  out_idx  = static_cast<SIZE_TYPE>(cursor.col());
                        const std::size_t wptr    = L.elem_start[in_idx] + elem_offset;

                        while (og_ptr < og_end &&
                               (*out_grad_sparse.indices[0])[og_ptr] < out_idx)
                            ++og_ptr;
                        if (og_ptr >= og_end ||
                            (*out_grad_sparse.indices[0])[og_ptr] != out_idx)
                            continue;

                        const value_type og_val = (*out_grad_sparse.values[0])[og_ptr];
                        input_gradients[in_idx] += ValueAccessor<VALUES_TYPE>::get_w(dc.values, wptr) * og_val;
                    }
                }
            }

            #pragma omp barrier
        }
    }
}

#endif