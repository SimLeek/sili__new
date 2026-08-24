#ifndef __DELTA_CSR_MEMORY_HPP_
#define __DELTA_CSR_MEMORY_HPP_

// Row-level memory operations on DeltaCSRWeights: 
//  build from / convert to absolute CSR, blank-space (headroom) management, 
//  row insert/remove/rebuild, synaptogenesis application (synap_row_step) 
//  and candidate generation (build_probes) 
#include "delta_csr_types.hpp"
#include <unordered_set>
#include <numeric>
#include <algorithm>
#include <type_traits>

// ── Build from / convert to absolute CSR ─────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> delta_csr_from_absolute(
    const std::vector<SIZE_TYPE>&   csr_ptrs,
    const std::vector<SIZE_TYPE>&   csr_indices,
    const std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& csr_weights,
    const std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& csr_importance,
    std::size_t rows, std::size_t cols,
    std::size_t index_bytes, std::size_t values_bytes,
    float blank_fraction = 0.2f,
    std::size_t hard_index_limit_bytes = std::numeric_limits<std::size_t>::max(),
    std::size_t hard_values_limit_bytes = std::numeric_limits<std::size_t>::max())
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> dc;
    dc.set_limits(hard_index_limit_bytes, hard_values_limit_bytes);
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
                                                static_cast<std::size_t>(n * blank_fraction) + 1);
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

// Like delta_csr_to_absolute() above, but also merges in any block4-resident synapses
// todo: move else functionality into delta_block4_to_absolute to improve code quality
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_combined_to_absolute(
    const SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    std::vector<SIZE_TYPE>&  out_ptrs,
    std::vector<SIZE_TYPE>&  out_indices,
    std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& out_weights,
    std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& out_importance)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    const auto& dc = weights.connections;
    const auto& L  = dc.layout;

    if constexpr (!std::is_same_v<VALUES_TYPE, FP4BiPacked> && !std::is_same_v<VALUES_TYPE, FP8BiValues>) {
        // block4 is FP4/FP8-specific (see block4.hpp) -- always empty otherwise.
        delta_csr_to_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
            dc, out_ptrs, out_indices, out_weights, out_importance);
        return;
    } else if (weights.block4.n_tiles() == 0) {
        delta_csr_to_absolute<SIZE_TYPE, VALUES_TYPE, COL_TYPE>(
            dc, out_ptrs, out_indices, out_weights, out_importance);
        return;
    } else {
        out_ptrs.assign(L.rows + 1, SIZE_TYPE(0));
        out_indices.clear();
        out_weights.clear();
        out_importance.clear();

        struct Entry { COL_TYPE col; value_type w, imp; };
        std::vector<Entry> row_entries;

        for (std::size_t r = 0; r < L.rows; ++r) {
            row_entries.clear();
            auto cursor = dc.row_cursor(r);
            const std::size_t n = L.row_nnz(r);
            for (std::size_t k = 0; k < n; ++k) {
                row_entries.push_back({
                    cursor.advance(),
                    ValueAccessor<VALUES_TYPE>::get_w(dc.values, L.elem_start[r] + k),
                    ValueAccessor<VALUES_TYPE>::get_imp(dc.values, L.elem_start[r] + k)});
            }

            const uint32_t br = uint32_t(r / BLOCK4_TILE);
            const uint32_t li = uint32_t(r % BLOCK4_TILE);
            if (br < weights.block4.block_layout.rows) {
                const auto& BL = weights.block4.block_layout;
                const std::size_t n_bc = BL.row_nnz(br);
                auto bc_cursor = weights.block4.row_cursor(br);
                for (std::size_t bk = 0; bk < n_bc; ++bk) {
                    const uint32_t bc = bc_cursor.advance();
                    // const: read-only export path
                    const auto tile = weights.block4.find(br, bc);
                    if constexpr (std::is_same_v<VALUES_TYPE, FP8BiValues>) {
                        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                            const uint8_t w_byte   = tile.at_weight(li, lj);
                            const uint8_t imp_byte = tile.at_importance(li, lj);
                            // we consider weight==0 & importance==0 as empty
                            if (w_byte == 0 && imp_byte == 0) continue;
                            const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                            if (col >= L.cols) continue;
                            row_entries.push_back({
                                COL_TYPE(col), fp8_decode_bits(w_byte), fp8_decode_bits(imp_byte)});
                        }
                    } else {
                        for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                            const uint8_t byte = tile.at(li, lj);
                            // we consider weight==0 & importance==0 as empty
                            if (byte == 0) continue;
                            const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                            if (col >= L.cols) continue;
                            row_entries.push_back({
                                COL_TYPE(col), FP4_TABLE[byte & 0xFu], FP4_TABLE[(byte >> 4) & 0xFu]});
                        }
                    }
                }
            }

            std::sort(row_entries.begin(), row_entries.end(),
                      [](const Entry& a, const Entry& b) { return a.col < b.col; });
            for (const auto& e : row_entries) {
                out_indices.push_back(e.col);
                out_weights.push_back(e.w);
                out_importance.push_back(e.imp);
            }
            out_ptrs[r + 1] = static_cast<SIZE_TYPE>(out_indices.size());
        }
    }
}

// Blank-space management

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

    // byte side
    const std::size_t cur_byte_alloc = L.row_alloc_bytes(row);
    if (cur_byte_alloc != target_byte_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = L.byte_start[row + 1];
        const std::size_t move_len  = L.byte_start[L.rows] - move_src;
        const std::size_t new_start = L.byte_start[row] + target_byte_alloc;

        if (target_byte_alloc > cur_byte_alloc) {
            const std::size_t new_total = ibuf.size() + (target_byte_alloc - cur_byte_alloc);
            if (new_total > dc.max_indices_bytes) throw std::bad_alloc();
            ibuf.resize(new_total);
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
        for (std::size_t r = row + 1; r < L.rows; ++r)
            L.byte_end[r] = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(L.byte_end[r]) + byte_delta);
    }

    // element side
    const std::size_t cur_elem_alloc = L.row_alloc_elems(row);
    if (cur_elem_alloc != target_elem_alloc && row + 1 < L.rows) {
        const std::size_t move_src  = L.elem_start[row + 1];
        const std::size_t move_len  = L.elem_start[L.rows] - move_src;
        const std::size_t new_start = L.elem_start[row] + target_elem_alloc;
        const std::size_t current_total = L.total_alloc_elems();

        if (target_elem_alloc > cur_elem_alloc) {
            const std::size_t new_total_elems = current_total + (target_elem_alloc - cur_elem_alloc);
            if (ValueAccessor<VALUES_TYPE>::projected_byte_size(new_total_elems) > dc.max_values_bytes)
                throw std::bad_alloc();
            ValueAccessor<VALUES_TYPE>::resize(dc.values, new_total_elems);
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
        for (std::size_t r = row + 1; r < L.rows; ++r)
            L.elem_end[r] = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(L.elem_end[r]) + elem_delta);
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

// Row-level insert / remove (for incremental synaptogenesis)

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

    const std::size_t old_row_nnz = L.row_nnz(row);

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
        static_cast<std::ptrdiff_t>(old_row_nnz);
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


// In-place insert/remove for delta-encoded rows
//
// Remove the connection at column `target_col` from row `row` in-place.
// Returns false if target_col is not found (no-op).
//
// Calling this before adding requires less headroom.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
bool delta_csr_row_remove_col(
    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc,
    std::size_t row,
    COL_TYPE target_col)
{
    auto& L   = dc.layout;
    auto& buf = dc.indices_buf;
    const std::size_t n = L.row_nnz(row);
    if (n == 0) return false;

    std::size_t byte_pos  = L.byte_start[row];
    std::size_t elem_pos  = L.elem_start[row];
    COL_TYPE    prev_col  = 0;

    for (std::size_t e = 0; e < n; ++e) {
        std::size_t delta_len = 0;
        const COL_TYPE delta  = uleb128_decode<COL_TYPE>(buf.data() + byte_pos, delta_len);
        const COL_TYPE col    = prev_col + delta;

        if (col == target_col) {
            const std::size_t next_byte_pos = byte_pos + delta_len;

            if (e + 1 < n) {
                // Merge this delta with the next one: next_col - prev_col
                std::size_t next_delta_len = 0;
                const COL_TYPE next_delta  =
                    uleb128_decode<COL_TYPE>(buf.data() + next_byte_pos, next_delta_len);
                const COL_TYPE merged_delta = delta + next_delta;

                uint8_t merged_buf[uleb128_max_bytes<COL_TYPE>()];
                const std::size_t merged_len = uleb128_encode<COL_TYPE>(merged_delta, merged_buf);

                // Write merged delta at byte_pos
                std::memcpy(buf.data() + byte_pos, merged_buf, merged_len);

                // Shift the remainder of the row left to fill the freed gap
                const std::size_t shift_from = next_byte_pos + next_delta_len;
                const std::size_t shift_len  = L.byte_end[row] - shift_from;
                const std::size_t freed      = delta_len + next_delta_len - merged_len;
                if (shift_len > 0)
                    std::memmove(buf.data() + byte_pos + merged_len,
                                 buf.data() + shift_from, shift_len);
                L.byte_end[row] -= freed;
            } else {
                // Last connection: just remove its delta bytes
                L.byte_end[row] -= delta_len;
            }

            // Shift value elements left to fill the removed slot
            const std::size_t row_end = L.elem_end[row];
            if (elem_pos + 1 < row_end)
                ValueAccessor<VALUES_TYPE>::move(dc.values,
                    elem_pos, elem_pos + 1, row_end - elem_pos - 1);
            L.elem_end[row]--;
            L.total_nnz--;
            return true;
        }

        prev_col  = col;
        byte_pos += delta_len;
        elem_pos++;
    }
    return false; // not found
}

// Insert a new connection at `new_col` in row `row` in sorted order, in-place.
// Returns true on success, false if the row has insufficient blank space.
// On false: the row's blank space is exhausted. Call equalizer_step() to
// redistribute space from adjacent rows, then retry.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
bool delta_csr_row_insert_col(
    DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& dc,
    std::size_t row,
    COL_TYPE    new_col,
    typename ValueAccessor<VALUES_TYPE>::value_type weight,
    typename ValueAccessor<VALUES_TYPE>::value_type importance)
{
    auto& L   = dc.layout;
    auto& buf = dc.indices_buf;
    const std::size_t n = L.row_nnz(row);

    // Walk to find insertion point (first existing column > new_col)
    std::size_t byte_pos      = L.byte_start[row];
    std::size_t elem_pos      = L.elem_start[row];
    COL_TYPE    prev_col      = 0;
    std::size_t ins_byte_pos  = L.byte_end[row]; // default: append after last
    std::size_t ins_elem_pos  = L.elem_end[row];
    bool        has_next      = false;
    COL_TYPE    next_col      = 0;
    std::size_t next_dlen     = 0;

    for (std::size_t e = 0; e < n; ++e) {
        std::size_t dlen = 0;
        const COL_TYPE delta = uleb128_decode<COL_TYPE>(buf.data() + byte_pos, dlen);
        const COL_TYPE col   = prev_col + delta;
        if (col == new_col) return false; // duplicate, skip
        if (col > new_col) {
            ins_byte_pos = byte_pos;
            ins_elem_pos = elem_pos;
            has_next     = true;
            next_col     = col;
            next_dlen    = dlen;
            break;
        }
        prev_col  = col;
        byte_pos += dlen;
        elem_pos++;
    }

    // Bytes for the new delta and (if inserting before an existing) the updated
    // next delta. Net byte change for the index buffer.
    uint8_t new_d_buf[uleb128_max_bytes<COL_TYPE>()];
    const std::size_t new_d_len = uleb128_encode<COL_TYPE>(new_col - prev_col, new_d_buf);

    uint8_t upd_d_buf[uleb128_max_bytes<COL_TYPE>()];
    std::size_t upd_d_len = 0;
    if (has_next)
        upd_d_len = uleb128_encode<COL_TYPE>(next_col - new_col, upd_d_buf);

    const std::ptrdiff_t idx_delta =
        static_cast<std::ptrdiff_t>(new_d_len + upd_d_len) -
        static_cast<std::ptrdiff_t>(next_dlen);

    // Check headroom (byte and element)
    const std::size_t used_bytes = L.byte_end[row] - L.byte_start[row];
    if (idx_delta > 0 &&
        static_cast<std::size_t>(idx_delta) > L.row_alloc_bytes(row) - used_bytes)
        return false; // not enough index byte headroom
    if (L.row_nnz(row) >= L.row_alloc_elems(row))
        return false; // not enough element headroom

    // Shift index bytes to make room (or shrink if idx_delta < 0)
    if (idx_delta != 0) {
        const std::size_t shift_from = ins_byte_pos;
        const std::size_t shift_len  = L.byte_end[row] - shift_from;
        if (shift_len > 0)
            std::memmove(buf.data() + shift_from + idx_delta,
                         buf.data() + shift_from, shift_len);
        L.byte_end[row] = static_cast<std::size_t>(
            static_cast<std::ptrdiff_t>(L.byte_end[row]) + idx_delta);
    }

    // Write new delta and (if applicable) the updated next delta
    std::memcpy(buf.data() + ins_byte_pos, new_d_buf, new_d_len);
    if (has_next)
        std::memcpy(buf.data() + ins_byte_pos + new_d_len, upd_d_buf, upd_d_len);

    // Shift value elements right and write the new one
    if (ins_elem_pos < L.elem_end[row])
        ValueAccessor<VALUES_TYPE>::move(dc.values,
            ins_elem_pos + 1, ins_elem_pos, L.elem_end[row] - ins_elem_pos);
    ValueAccessor<VALUES_TYPE>::set(dc.values, ins_elem_pos, weight, importance);
    L.elem_end[row]++;
    L.total_nnz++;
    return true;
}

// block4 promotion / demotion
//
// growth (synaptogenesis) can only promote scattered -> block4; pruning can
// only DEMOTE block4 -> scattered.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
void block4_maybe_promote(
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    std::size_t row, COL_TYPE col)
{
    constexpr bool is_fp4 = std::is_same_v<VALUES_TYPE, FP4BiPacked>;
    constexpr bool is_fp8 = std::is_same_v<VALUES_TYPE, FP8BiValues>;
    if constexpr (!is_fp4 && !is_fp8) {
        (void)weights; (void)row; (void)col; // block4 is FP4/FP8-specific.
    } else {
        using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
        auto& dc = weights.connections;
        auto& L  = dc.layout;
        // Lazy self-init. Tile-count budget identical for both formats
        // (max_values_bytes synapses at max fill of BLOCK4_TILE_SLOTS=16/
        // tile); byte budget differs (BLOCK4_TILE_SLOTS8_BYTES=32/tile for
        // FP8 vs BLOCK4_TILE_SLOTS=16 for FP4, matching
        // SparseLinearLayer8's own identical constructor-time sizing).
	if (weights.block4.block_layout.rows == 0 && L.rows > 0) {
            weights.block4.init(L.rows, L.cols);
            // If the caller never set a real CSR-side budget either
            // (dc.max_values_bytes still at its own SIZE_MAX-ish
            // default), the derived byte budget below is still
            // enormous after the /BLOCK4_TILE_SLOTS*BLOCK4_TILE_SLOTS
            // rounding -- confirmed directly: this used to silently
            // fail inside set_limits()'s own reserve() call (a real
            // std::length_error, "vector::reserve", exceeding
            // std::vector<uint8_t>::max_size()), previously masked by
            // a try/catch that's since been removed (see set_limits'
            // own docstring for why silently swallowing that isn't
          // safe to do anymore). Pass the block4 sentinel through
            // unchanged instead of computing a derived-but-nonsensical
            // near-max value -- "no CSR budget set" should mean "no
            // block4 budget set" too, not "compute something that
            // happens to overflow vector::reserve()."
            constexpr std::size_t kNoLimit = std::numeric_limits<std::size_t>::max();
            const std::size_t tile_budget = (dc.max_values_bytes == kNoLimit) ? kNoLimit :
                (std::max<std::size_t>(4, dc.max_values_bytes / BLOCK4_TILE_SLOTS) *
                 (is_fp8 ? BLOCK4_TILE_SLOTS8_BYTES : BLOCK4_TILE_SLOTS));
            weights.block4.set_limits(dc.max_indices_bytes, tile_budget);
        }
        const uint32_t br = uint32_t(row / BLOCK4_TILE);
        const uint32_t bc = uint32_t(col / BLOCK4_TILE);
        const uint32_t li = uint32_t(row % BLOCK4_TILE);
        const uint32_t lj = uint32_t(col % BLOCK4_TILE);

        if (weights.block4.find(br, bc)) {
            const std::size_t n = L.row_nnz(row);
            auto cursor = dc.row_cursor(row);
            for (std::size_t k = 0; k < n; ++k) {
                const COL_TYPE c = cursor.advance();
                if (c == col) {
                    const std::size_t vb = L.elem_start[row] + k;
                    const value_type w   = ValueAccessor<VALUES_TYPE>::get_w(dc.values, vb);
                    const value_type imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, vb);
                    {
                        auto tile = weights.block4.find(br, bc);
                        // NOT the live variant, deliberately: this re-encodes
                        // whatever value the scattered side ALREADY had,
                        // including a genuinely fresh, never-yet-trained
                        // synapse's deliberate weight=0.0 (see synaptogenesis's
                        // own insert_col convention) -- format migration, not
                        // a training update. Confirmed directly: using the
                        // live variant here broke test_disldo_block4_promotion's
                        // own documented round-trip check (a fresh synapse's
                        // 0.0 must survive promotion/demotion exactly). The
                        // actual never-0 protection belongs at the POST-
                        // promotion training sites (disldo_backward's block4
                        // branches, magnitude_rescale_output), not here.
                        if constexpr (is_fp8) {
                            tile.at_weight(li, lj)     = fp8_quantize(w);
                            tile.at_importance(li, lj) = fp8_quantize(imp);
                        } else {
                            tile.at(li, lj) = uint8_t(fp4_quantize(w) | (fp4_quantize(imp) << 4));
                        }
                    }
		    weights.block4.maybe_compress(br, bc);
                    delta_csr_row_remove_col(dc, row, col);
                    return;
                }
            }
            return; // not found -- shouldn't happen, caller just inserted it
        }

        const std::size_t row_lo = std::size_t(br) * BLOCK4_TILE;
        const std::size_t row_hi = std::min(row_lo + BLOCK4_TILE, L.rows);
        const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
        const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, L.cols);

        struct Found { std::size_t row; COL_TYPE col; std::size_t elem_idx; };
        std::vector<Found> found;
        for (std::size_t r = row_lo; r < row_hi; ++r) {
            const std::size_t n = L.row_nnz(r);
            if (n == 0) continue;
            auto cursor = dc.row_cursor(r);
            for (std::size_t k = 0; k < n; ++k) {
                const COL_TYPE c = cursor.advance();
                if (std::size_t(c) >= col_lo && std::size_t(c) < col_hi)
                    found.push_back({r, c, L.elem_start[r] + k});
            }
        }
        if (found.size() < BLOCK4_PROMOTE_MIN_LIVE) return;

        {
            auto tile = weights.block4.get_or_create(br, bc);
            for (const auto& f : found) {
                const value_type w   = ValueAccessor<VALUES_TYPE>::get_w(dc.values, f.elem_idx);
                const value_type imp = ValueAccessor<VALUES_TYPE>::get_imp(dc.values, f.elem_idx);
                const uint32_t fli = uint32_t(f.row - row_lo);
                const uint32_t flj = uint32_t(std::size_t(f.col) - col_lo);
                // Not the live variant -- see the single-synapse promote
                // branch's identical comment above.
                if constexpr (is_fp8) {
                    tile.at_weight(fli, flj)     = fp8_quantize(w);
                    tile.at_importance(fli, flj) = fp8_quantize(imp);
                } else {
                    tile.at(fli, flj) = uint8_t(fp4_quantize(w) | (fp4_quantize(imp) << 4));
                }
            }
        }
	weights.block4.maybe_compress(br, bc);
        for (const auto& f : found)
            delta_csr_row_remove_col(dc, f.row, f.col);
    }
}

// Bulk LOADING (not quantization) of already-quantized dense weight/
// importance codes directly into block4 storage, bypassing the scattered
// -CSR side and the importance-gated growth-insertion path entirely.
// `weight_codes`/`importance_codes` are raw FP4 (0-15) or FP8 (E4M3 byte)
// codes, row-major n_in x n_out -- the CALLER decides how those codes were
// produced (round-to-nearest via fp4_quantize_array/fp8_quantize_array,
// a rank-1-scaled fit, a residual scheme, real gradient-derived importance
// for a converted checkpoint, etc.); this function does no quantization
// and no scale division of its own, matching Block4Store's own native
// tile-write interface (`tile.at(li,lj)`/`at_weight`/`at_importance` are
// raw uint8_t&, zero quantization logic) rather than ValueAccessor::set's
// float-in-quantize-on-write convention used by load_weights()/
// block4_maybe_promote. If a row/col value_scale is wanted, set it
// afterward via set_value_scale_raw()/set_output_scale_raw() -- same
// two-step pattern sili_peridot's own FoldedLayer.from_descriptor already
// uses for the scattered-CSR path (sparse_rnn.py).
//
// weights.connections (scattered side) is left untouched (zero nnz),
// mirroring load_weights()'s own "only touch the side being loaded"
// precedent -- the combined .nnz property (connections.nnz() +
// block4.live_synapses(), both live-computed, not cached) then correctly
// reports n_in*n_out once this returns.
// Redistributes each block-row's TILE-BYTE headroom (tile_byte_start/
// tile_byte_end/tile_data) to blank_fraction slack above its current
// content, mirroring sisldo_ops.hpp's expand_headroom() for the scattered
// -CSR side. Deliberately does NOT touch block_layout (which (br,bc) tiles
// exist stays exactly the same -- this is not block4_load_dense's "place a
// tile at every combination" behavior, just a byte-range redistribution)
// or any tile's packed content/sparsity choice.
//
// WHY this exists (see conversation): merge_row_workspace (block4.hpp)
// clamps a row's committed content to tile_byte_start[br+1]-
// tile_byte_start[br] and EVICTS synapses to fit if it doesn't -- correct
// behavior when a row is genuinely at its budget, but a real bug when a
// row was simply never given any slack in the first place. block4_load_dense
// (below) used to size each row's headroom to EXACTLY its initial packed
// content (0 extra bytes) -- for a slot whose weight AND importance are both
// 0 at load time, that content packs to a 1-byte empty-sparse tile, so the
// very first weight that escapes 0 (needing the tile to go dense, 16 bytes)
// got evicted right back out by merge_row_workspace on the SAME backward
// call, silently undoing every gradient step forever.
//
// Deliberately NOT "reserve full dense-tile headroom per row" (rejected --
// real memory blowup for large sparse layers, many of which would
// otherwise fit) -- blank_fraction-proportional slack, distributed per row
// exactly like the scattered path already does, is the same tradeoff this
// codebase already made once and validated (delta_csr_from_absolute/
// expand_headroom), not a new policy invented here.
// FP4-only for now (block4_stored_tile_len is FP4's tile-length formula;
// FP8's Block4Store8 needs block4_stored_tile_len8 and its own pass --
// same scoping as this session's other block4 rank-N/chain-rule work,
// since the real toy/test model uses FP4, not FP8. A no-op for FP8 rather
// than a silent miscompile.
// min_slack_bytes: a floor added to EVERY row's blank space regardless of
// blank_fraction -- default BLOCK4_TILE_SLOTS (one full FP4 dense tile) so
// a row whose entire current content is a single empty sparse tile (1
// byte) still gets enough room for that ONE tile to go fully dense
// without immediately re-triggering eviction (blank_fraction alone rounds
// to 0 extra bytes at that scale: 1 byte * 0.2 truncates to 0). Pass 0
// (alongside blank_fraction=0) for a true tight compact -- see
// block4_compact below, the opposite operation of this function, mirroring
// compact()/expand_headroom()'s existing pairing on the scattered-CSR
// side.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
void block4_expand_headroom(
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    float blank_fraction = 0.2f,
    std::size_t min_slack_bytes = BLOCK4_TILE_SLOTS)
{
    if constexpr (!std::is_same_v<VALUES_TYPE, FP4BiPacked>) {
        (void)weights; (void)blank_fraction; (void)min_slack_bytes;
    } else {
    auto& store = weights.block4;
    const std::size_t rows = store.block_layout.rows;
    if (rows == 0) return;

    // Each row's CURRENT content length -- same per-tile length accessor
    // merge_row_workspace itself walks (block4_stored_tile_len), so this
    // is exactly what's live today, no assumption about dense-vs-sparse.
    std::vector<std::size_t> row_bytes(rows, 0);
    for (std::size_t br = 0; br < rows; ++br) {
        std::size_t pos = store.tile_byte_start[br];
        const std::size_t n_bc = store.block_layout.row_nnz(br);
        std::size_t elem_pos = store.block_layout.elem_start[br];
        for (std::size_t k = 0; k < n_bc; ++k, ++elem_pos) {
            pos += block4_stored_tile_len(store.tile_is_sparse[elem_pos], &store.tile_data[pos]);
        }
        row_bytes[br] = pos - store.tile_byte_start[br];
    }

    // Lay out new tile_byte_start with blank_fraction slack -- same
    // byte_blank formula as delta_csr_from_absolute, plus min_slack_bytes.
    std::vector<std::size_t> new_start(rows + 1, 0);
    for (std::size_t br = 0; br < rows; ++br) {
        const std::size_t blank = static_cast<std::size_t>(row_bytes[br] * blank_fraction)
                                   + min_slack_bytes;
        new_start[br + 1] = new_start[br] + row_bytes[br] + blank;
    }

    if (new_start[rows] > store.max_tile_bytes) throw std::bad_alloc();

    std::vector<uint8_t> new_data(new_start[rows], uint8_t(0));
    for (std::size_t br = 0; br < rows; ++br) {
        std::memcpy(new_data.data() + new_start[br],
                    store.tile_data.data() + store.tile_byte_start[br],
                    row_bytes[br]);
    }

    store.tile_data = std::move(new_data);
    store.tile_byte_end.resize(rows);
    for (std::size_t br = 0; br < rows; ++br) {
        store.tile_byte_start[br] = new_start[br];
        store.tile_byte_end[br]   = new_start[br] + row_bytes[br];
    }
    store.tile_byte_start[rows] = new_start[rows];
    // tile_is_sparse/block_layout untouched -- per-tile content, sparsity
    // choice, and which (br,bc) tiles exist are all unchanged.
    }
}

// Opposite of block4_expand_headroom(): shrinks every row's tile-byte
// headroom back down to EXACTLY its current live content, zero slack --
// same "compact() normalizes to exactly 0%, expand() normalizes to
// exactly blank_fraction" pairing sisldo_ops.hpp's compact()/
// expand_headroom() already establish for the scattered-CSR side. Use
// before a row is done growing (e.g. post-pruning, or a layer that's
// plateaued) to reclaim the blank_fraction slack block4_load_dense/
// block4_expand_headroom reserved. A subsequent call needing to grow
// again should call block4_expand_headroom() again afterward, same as
// the scattered-CSR compact()->expand_headroom() cycle.
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
void block4_compact(SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights) {
    block4_expand_headroom(weights, 0.0f, std::size_t(0));
}

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
void block4_load_dense(
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    const uint8_t* weight_codes, const uint8_t* importance_codes,
    std::size_t n_in, std::size_t n_out,
    float blank_fraction = 0.2f)
{
    constexpr bool is_fp4 = std::is_same_v<VALUES_TYPE, FP4BiPacked>;
    constexpr bool is_fp8 = std::is_same_v<VALUES_TYPE, FP8BiValues>;
    if constexpr (!is_fp4 && !is_fp8) {
        (void)weights; (void)weight_codes; (void)importance_codes; (void)n_in; (void)n_out; (void)blank_fraction;
    } else {
        const uint32_t block_rows = uint32_t((n_in  + BLOCK4_TILE - 1) / BLOCK4_TILE);
        const uint32_t block_cols = uint32_t((n_out + BLOCK4_TILE - 1) / BLOCK4_TILE);

        weights.block4.init(n_in, n_out);
        // Indices budget: EVERY row starts with zero pre-allocated index
        // bytes (Block4Store::init() zero-fills tile_byte_start/
        // byte_start), so inserting the first tile into any row goes
        // through block4_ensure_row_headroom's growth path, which
        // requests up to uleb128_max_bytes<uint32_t>() (5) bytes per
        // entry -- found directly (bad_alloc thrown at a too-small
        // budget) that this can be requested more than once per tile as
        // block4_row_shift cascades a growth request through earlier
        // rows to make room in a later one. 16 bytes/tile is a generous
        // multiple of that 5-byte worst case to absorb the cascade
        // without hand-deriving its exact worst case; indices bytes are
        // cheap relative to tile bytes so over-provisioning here costs
        // little.
        //
        // Tile-byte budget (the hard cap set_limits enforces, NOT the
        // same thing as the per-row headroom block4_expand_headroom lays
        // out below -- this is just the ceiling that layout is allowed
        // to grow up to): sized for full density PLUS blank_fraction, not
        // exactly full density with zero slack -- block4_expand_headroom
        // below deliberately adds real per-row growth room on top of
        // whatever's actually used, and a cap with zero slack to begin
        // with makes that immediately throw bad_alloc regardless of how
        // little is actually live. Note: expand_headroom_to()
        // (sisldo_ops.hpp) only touches weights.connections (scattered
        // CSR) -- it does NOT extend to block4 at all, despite an
        // earlier version of this comment implying it did.
        // block4_expand_headroom() (below, called automatically at the
        // end of this function) is the real block4-side equivalent.
        const std::size_t idx_budget = std::size_t(block_rows) * block_cols * 16;
        const std::size_t dense_tile_bytes = std::size_t(block_rows) * block_cols
            * (is_fp8 ? BLOCK4_TILE_SLOTS8_BYTES : BLOCK4_TILE_SLOTS);
        // + block_rows*BLOCK4_TILE_SLOTS, not a single flat margin --
        // block4_expand_headroom adds its minimum-slack term
        // (BLOCK4_TILE_SLOTS) PER ROW, so the cap needs that same
        // per-row margin summed across every row or a fully/near-fully
        // dense load throws bad_alloc the moment expand_headroom runs
        // (confirmed directly: real regression at frac_live=1.0 in
        // testing, only a single BLOCK4_TILE_SLOTS margin here).
        const std::size_t tile_budget =
            static_cast<std::size_t>(dense_tile_bytes * (1.0 + blank_fraction))
            + std::size_t(block_rows) * BLOCK4_TILE_SLOTS;
        weights.block4.set_limits(idx_budget, tile_budget);

        for (uint32_t br = 0; br < block_rows; ++br) {
            const std::size_t row_lo = std::size_t(br) * BLOCK4_TILE;
            const std::size_t row_hi = std::min(row_lo + BLOCK4_TILE, n_in);
            for (uint32_t bc = 0; bc < block_cols; ++bc) {
                const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
                const std::size_t col_hi = std::min(col_lo + BLOCK4_TILE, n_out);
                auto tile = weights.block4.get_or_create(br, bc);
                for (std::size_t row = row_lo; row < row_hi; ++row) {
                    const uint32_t li = uint32_t(row - row_lo);
                    for (std::size_t col = col_lo; col < col_hi; ++col) {
                        const uint32_t lj = uint32_t(col - col_lo);
                        const std::size_t idx = row * n_out + col;
                        if constexpr (is_fp8) {
                            tile.at_weight(li, lj)     = weight_codes[idx];
                            tile.at_importance(li, lj) = importance_codes[idx];
                        } else {
                            tile.at(li, lj) = uint8_t(weight_codes[idx] | (importance_codes[idx] << 4));
                        }
                    }
                }
                // tile's destructor (scope end) commits scratch_ back to
                // the store, choosing dense vs sparse-packed encoding
                // based on live count vs switch_point -- picks dense
                // whenever at least one slot's byte is nonzero (matches
                // block4_count_live's own "live iff weight OR importance
                // nonzero" convention), sparse-empty (1 byte) for a tile
                // whose weight AND importance codes are ALL zero (real
                // for a zero-init training layer, not just a hypothetical
                // -- see block4_expand_headroom's docstring for why that
                // specifically needs headroom reserved regardless). No
                // separate maybe_compress() call needed here (that's for
                // re-checking an EXISTING tile's encoding, not relevant
                // to a from-scratch bulk load of brand-new tiles).
            }
        }
        // Reserve real per-row growth slack for the tile-byte storage --
        // see block4_expand_headroom's own docstring for exactly why this
        // is needed even for a "static" load (a slot that's all-zero at
        // load time, e.g. a zero-init training layer, packs to a 1-byte
        // empty-sparse tile with otherwise zero headroom to grow back out
        // of once training escapes it from 0). FP4-only for now, matching
        // block4_expand_headroom's own scope.
        if constexpr (is_fp4) block4_expand_headroom(weights, blank_fraction);
    }
}

// Pruning-only hook.
// Throws if a target row has run out of blank space 
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
void block4_demote_tile(
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    uint32_t br, uint32_t bc)
{
    constexpr bool is_fp8 = std::is_same_v<VALUES_TYPE, FP8BiValues>;
    if constexpr (!std::is_same_v<VALUES_TYPE, FP4BiPacked> && !is_fp8) {
        (void)weights; (void)br; (void)bc;
    } else {
        using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
        auto& dc = weights.connections;
        auto& L  = dc.layout;
        const std::size_t row_lo = std::size_t(br) * BLOCK4_TILE;
        const std::size_t col_lo = std::size_t(bc) * BLOCK4_TILE;
        {
            auto tile = weights.block4.find(br, bc);
            if (!tile) return;
            for (uint32_t li = 0; li < BLOCK4_TILE; ++li) {
                const std::size_t row = row_lo + li;
                if (row >= L.rows) continue;
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    value_type w, imp;
                    if constexpr (is_fp8) {
                        const uint8_t w_byte   = tile.at_weight(li, lj);
                        const uint8_t imp_byte = tile.at_importance(li, lj);
                        if (w_byte == 0 && imp_byte == 0) continue;
                        w   = fp8_decode_bits(w_byte);
                        imp = fp8_decode_bits(imp_byte);
                    } else {
                        const uint8_t byte = tile.at(li, lj);
                        if (byte == 0) continue;
                        w   = FP4_TABLE[byte & 0xFu];
                        imp = FP4_TABLE[(byte >> 4) & 0xFu];
                    }
                    const std::size_t col = col_lo + lj;
                    if (col >= L.cols) continue;
                    if (!delta_csr_row_insert_col(dc, row, COL_TYPE(col), w, imp)) {
                        throw std::runtime_error(
                            "block4_demote_tile: row " + std::to_string(row) +
                            " ran out of blank space while demoting tile (" +
                            std::to_string(br) + "," + std::to_string(bc) + ")."
                            " Call equalizer_step() to redistribute space from"
                            " adjacent rows before retrying.");
                    }
                }
            }
        }
	weights.block4.erase(br, bc);
    }
}

// Incremental synaptogenesis step
//
// in-place per-connection insert and remove. 
//
// Algorithm per row:
//   1. Walk row once: collect (col, weight, importance) for all connections.
//   2. Collect probes for this row from weights.probes.
//   3. Determine removes: connections below importance_cutoff OR lowest-
//      importance connections when n_exist > max_row_weights.
//   4. Determine adds: top probes not already present, up to the slots freed by
//      removes plus any remaining capacity below max_row_weights.
//   5. Apply removes (O(n) each, but K << n so O(K*n) total).
//      Removes always shrink the row -- freed bytes are immediately available.
//   6. Apply adds using delta_csr_row_insert_col. Throws std::runtime_error
//      on first insertion failure -- no silent skipping. Caller must handle
//      the error by calling equalizer_step() to redistribute blank space from
//      adjacent rows, then retry. If the total pool is exhausted, the error
//      message explains what to do (prune more / lower max_row_weights).
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
bool delta_csr_synap_row_step(
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    std::size_t& current_row,
    typename ValueAccessor<VALUES_TYPE>::value_type importance_cutoff,
    SIZE_TYPE max_row_weights,
    // Caps how many connections this ONE row-step call may remove --
    // per direct request: without a cap, a single call can prune an
    // entire row at once (e.g. importance_cutoff raised mid-training,
    // or many probes tying at the eps floor above), which is exactly
    // the kind of large abrupt connectivity loss synaptogenesis is
    // meant to avoid (it's throttled to O(1)-ish per call by design --
    // see synap_step's own docstring). Default is generous (rarely
    // binds in normal operation) rather than tiny, since it's a safety
    // ceiling, not a throttle on ordinary capacity-driven trimming.
    SIZE_TYPE max_prune_per_step = SIZE_TYPE(8),
    // A "ghost" floor: used ONLY for the importance_cutoff comparison
    // below, NEVER written to storage anywhere. A synapse whose real,
    // stored importance has decayed to exactly the FP4 zero code (a
    // real, discrete quantization bucket many independently-decaying
    // synapses can land on simultaneously -- FP4's smallest nonzero
    // magnitude is 0.5, so 0 is a wide, common landing bucket) isn't
    // automatically "below cutoff" the instant it gets there, without
    // ever inflating what's actually persisted. Does NOT affect
    // by_imp's sort order (lowest-real-importance-first removal
    // priority is unchanged -- a floored synapse can still be removed
    // via the keep>max_rw capacity criterion, just not solely because
    // its stored value happens to be exactly 0). This also naturally
    // protects a freshly-grown synapse (which starts with whatever its
    // REAL probe score is, per delta_csr_build_probes -- often exactly
    // 0 for a row with no activity yet, and stored as such): on its
    // first subsequent synap_row_step visit it won't be evicted purely
    // for reading as 0, giving real backprop time to move it for real.
    typename ValueAccessor<VALUES_TYPE>::value_type importance_eps = typename ValueAccessor<VALUES_TYPE>::value_type(1e-3))
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    constexpr bool is_fp4 = std::is_same_v<VALUES_TYPE, FP4BiPacked>;
    constexpr bool is_fp8 = std::is_same_v<VALUES_TYPE, FP8BiValues>;
    constexpr bool has_block4 = is_fp4 || is_fp8;
    auto& dc = weights.connections;
    auto& L  = dc.layout;
    if (L.rows == 0) return false;

    const std::size_t row = current_row % L.rows;
    current_row = (current_row + 1) % L.rows;

    const std::size_t n_scattered = L.row_nnz(row);
    const bool has_probes = weights.probes.indices[0] &&
                            !weights.probes.indices[0]->empty();
    const uint32_t br = uint32_t(row / BLOCK4_TILE);
    const uint32_t li = uint32_t(row % BLOCK4_TILE);
    std::vector<uint32_t> b4_bc, b4_lj;
    if constexpr (has_block4) {
        if (br < weights.block4.block_layout.rows) {
            const auto& BL = weights.block4.block_layout;
            const std::size_t n_bc = BL.row_nnz(br);
            auto bc_cursor = weights.block4.row_cursor(br);
            for (std::size_t bk = 0; bk < n_bc; ++bk) {
                const uint32_t bc = bc_cursor.advance();
                // const: read-only discovery scan
		const auto tile = weights.block4.find(br, bc);
                for (uint32_t lj = 0; lj < BLOCK4_TILE; ++lj) {
                    if constexpr (is_fp8) {
                        if (tile.at_weight(li, lj) == 0 && tile.at_importance(li, lj) == 0) continue;
                    } else {
                        if (tile.at(li, lj) == 0) continue;
                    }
                    const std::size_t col = std::size_t(bc) * BLOCK4_TILE + lj;
                    if (col >= L.cols) continue;
                    b4_bc.push_back(bc);
                    b4_lj.push_back(lj);
                }
            }
        }
    }
    const std::size_t n_exist = n_scattered + b4_bc.size();
    if (n_exist == 0 && !has_probes) return false;

    // Step 1: Read existing connections (scattered, then block4)
    std::vector<COL_TYPE>   exist_cols(n_exist);
    std::vector<value_type> exist_w(n_exist), exist_imp(n_exist);
    std::vector<bool>       exist_is_b4(n_exist, false);
    {
        auto cursor = dc.row_cursor(row);
        for (std::size_t k = 0; k < n_scattered; ++k) {
            exist_cols[k] = cursor.advance();
            exist_w[k]    = ValueAccessor<VALUES_TYPE>::get_w(
                dc.values, L.elem_start[row] + k);
            exist_imp[k]  = ValueAccessor<VALUES_TYPE>::get_imp(
                dc.values, L.elem_start[row] + k);
        }
    }
    if constexpr (has_block4) {
        for (std::size_t j = 0; j < b4_bc.size(); ++j) {
            const std::size_t k = n_scattered + j;
            const uint32_t bc = b4_bc[j], lj = b4_lj[j];
            auto tile = weights.block4.find(br, bc);
            exist_cols[k]  = COL_TYPE(std::size_t(bc) * BLOCK4_TILE + lj);
            if constexpr (is_fp8) {
                exist_w[k]   = fp8_decode_bits(tile.at_weight(li, lj));
                exist_imp[k] = fp8_decode_bits(tile.at_importance(li, lj));
            } else {
                const uint8_t byte = tile.at(li, lj);
                exist_w[k]   = FP4_TABLE[byte & 0xFu];
                exist_imp[k] = FP4_TABLE[(byte >> 4) & 0xFu];
            }
            exist_is_b4[k] = true;
        }
    }

    // Step 2: Collect probes for this row
    std::vector<COL_TYPE>   probe_cols;
    std::vector<value_type> probe_scores;
    if (has_probes) {
        const auto& prow = *weights.probes.indices[0];
        const auto& pcol = *weights.probes.indices[1];
        const auto& pval = *weights.probes.values[0];
        const SIZE_TYPE pnnz = weights.probes.nnz();
        for (SIZE_TYPE p = 0; p < pnnz; ++p) {
            if (static_cast<std::size_t>(prow[p]) == row) {
                probe_cols.push_back(static_cast<COL_TYPE>(pcol[p]));
                probe_scores.push_back(pval[p]);
            }
        }
    }

    // Step 3: Determine which connections to remove
    std::vector<std::size_t> by_imp(n_exist);
    std::iota(by_imp.begin(), by_imp.end(), 0);
    std::sort(by_imp.begin(), by_imp.end(),
              [&](std::size_t a, std::size_t b) {
                  return exist_imp[a] < exist_imp[b];
              });

    struct RemoveEntry { COL_TYPE col; bool is_b4; };
    std::vector<RemoveEntry> to_remove;
    const std::size_t max_rw = static_cast<std::size_t>(max_row_weights);
    const std::size_t max_prune = static_cast<std::size_t>(max_prune_per_step);
    for (std::size_t rank = 0; rank < n_exist && to_remove.size() < max_prune; ++rank) {
        const std::size_t k    = by_imp[rank];
        const std::size_t keep = n_exist - to_remove.size();
        if (std::max(exist_imp[k], importance_eps) < importance_cutoff || keep > max_rw)
            to_remove.push_back({exist_cols[k], exist_is_b4[k]});
    }
    // Sort descending so scattered removes happen high col first -- keeps
    // byte positions of lower-col elements stable while we walk and remove
    std::sort(to_remove.begin(), to_remove.end(),
              [](const RemoveEntry& a, const RemoveEntry& b) { return a.col > b.col; });

    // Step 4: Determine which probes to add
    // Filter out probes that already have a connection.
    {
        std::unordered_set<COL_TYPE> exist_set(exist_cols.begin(), exist_cols.end());
        // (Will also filter against to_remove to avoid immediately re-adding
        // a just-removed connection. Not strictly necessary but clean.)
        std::unordered_set<COL_TYPE> remove_set;
        remove_set.reserve(to_remove.size());
        for (const auto& re : to_remove) remove_set.insert(re.col);

        std::vector<std::size_t> pidx(probe_cols.size());
        std::iota(pidx.begin(), pidx.end(), 0);
        // Sort probes by score descending (highest score = best candidate)
        std::sort(pidx.begin(), pidx.end(),
                  [&](std::size_t a, std::size_t b) {
                      return probe_scores[a] > probe_scores[b];
                  });

        std::vector<COL_TYPE>   add_cols;
        std::vector<value_type> add_scores;
        const std::size_t slots = max_rw - (n_exist - to_remove.size());
        for (std::size_t p : pidx) {
            if (add_cols.size() >= slots) break;
            const COL_TYPE c = probe_cols[p];
            if (!exist_set.count(c) || remove_set.count(c))
                if (!exist_set.count(c)) { // truly not present
                    add_cols.push_back(c);
                    add_scores.push_back(probe_scores[p]);
                }
        }
        probe_cols   = std::move(add_cols);
        probe_scores = std::move(add_scores);
    }

    // Step 5: Apply removes (in-place, high-col first)
    for (const auto& re : to_remove) {
        if (re.is_b4) {
            if constexpr (has_block4) {
                const uint32_t bc = uint32_t(re.col) / BLOCK4_TILE;
                const uint32_t lj = uint32_t(re.col) % BLOCK4_TILE;
                bool should_demote = false;
                {
                    auto tile = weights.block4.find(br, bc);
                    if constexpr (is_fp8) {
                        if (tile && (tile.at_weight(li, lj) != 0 || tile.at_importance(li, lj) != 0)) {
                            tile.at_weight(li, lj)     = 0;
                            tile.at_importance(li, lj) = 0;
                            should_demote = tile.count_live() < BLOCK4_PROMOTE_MIN_LIVE;
                        }
                    } else {
                        if (tile && tile.at(li, lj) != 0) {
                            tile.at(li, lj) = 0;
                            should_demote = tile.count_live() < BLOCK4_PROMOTE_MIN_LIVE;
                        }
                    }
                }
		if (should_demote)
                    block4_demote_tile(weights, br, bc);
            }
        } else {
            delta_csr_row_remove_col(dc, row, re.col);
        }
        if (!weights.out_degree.empty() && weights.out_degree[re.col] > 0)
            --weights.out_degree[re.col];
    }

    // Step 6: Apply adds in-place
    for (std::size_t i = 0; i < probe_cols.size(); ++i) {
        const COL_TYPE col = probe_cols[i];
        if (!delta_csr_row_insert_col(dc, row, col, value_type(0), probe_scores[i])) {
            const std::size_t used  = dc.layout.byte_end[row] - dc.layout.byte_start[row];
            const std::size_t alloc = dc.layout.row_alloc_bytes(row);
            // Include the probe index so callers can infer the exact split
            throw std::runtime_error(
                "delta_csr_synap_row_step: row " + std::to_string(row) +
                " ran out of blank space at probe_index=" + std::to_string(i) +
                " col=" + std::to_string(col) +
                " (used " + std::to_string(used) + " / " + std::to_string(alloc) +
                " bytes, " + std::to_string(probe_cols.size() - i) + " insertions skipped)."
                " Call equalizer_step() to redistribute space from adjacent rows"
                " before retrying, or reduce max_row_weights / raise importance_cutoff.");
        }
        if (!weights.out_degree.empty())
            ++weights.out_degree[col];
        if constexpr (has_block4)
            block4_maybe_promote(weights, row, col);
    }

    return true;
}

// Probe generation (outer product, top-k)

/**
 * @brief Build COO probe candidates for synaptogenesis via outer product.
 *
 * Selects the top-@p k input neurons by neuron_input_accum and top-@p k output
 * neurons by neuron_grad_accum, then forms the outer product of those two
 * sets. Pairs that already have a connection are skipped. Each novel pair
 * gets importance = accum_in * accum_out. Existing probes are cleared and
 * replaced.
 *
 * Generic over VALUES_TYPE via ValueAccessor -- works identically for
 * FP4BiPacked (default, 4-bit) and DeltaCSRBiValues<float> (32-bit) with no
 * separate implementation, matching delta_csr_synap_row_step (which
 * applies these probes) and sisldo_forward/backward.
 *
 * Existing-connection check uses DeltaCSRRowCursor directly (no full row
 * materialization) -- O(row_nnz) per candidate input row, same complexity
 * class as scanning the row any other way.
 *
 * Complexity: O(n_inputs + n_outputs) for the top-k selection, O(k^2 *
 * avg_row_nnz) for the existing-connection filter -- fine for small k
 * (typically 2-4).
 *
 * @param weights             Layer state -- probes are replaced.
 * @param neuron_input_accum  [n_inputs] accumulated |x| across recent passes.
 * @param neuron_grad_accum   [n_outputs] accumulated |grad| across recent passes.
 * @param k                   Top-k candidates per side.
 * @param per_row             false (default): ONE shared top-k output set
 *                            computed globally, outer-producted against the
 *                            top-k input rows, THEN filtered for existing
 *                            connections -- cheap (one O(n_out log k) sort
 *                            total), but a row can lose probe slots to
 *                            duplicates it already has, so it may end up
 *                            with fewer than k genuinely-new candidates.
 *                            true: EACH candidate row gets its OWN top-k
 *                            search over ALL n_out outputs with existing
 *                            connections excluded DURING selection (not
 *                            after) -- guarantees up to k genuinely-new
 *                            candidates per row, at the cost of one
 *                            O(n_out) scan PER candidate row instead of
 *                            one shared O(n_out log k) sort. Matters more
 *                            for large layers, where a shared top-k set is
 *                            a much smaller fraction of all outputs and the
 *                            wasted-slots effect is more pronounced -- not
 *                            worth the extra cost for small layers.
 *
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_build_probes(
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    const typename ValueAccessor<VALUES_TYPE>::value_type* neuron_input_accum,
    const typename ValueAccessor<VALUES_TYPE>::value_type* neuron_grad_accum,
    SIZE_TYPE k,
    bool per_row = false)
{
    // Probe scores are the REAL input_accum*grad_accum product, written
    // verbatim as a newly-inserted synapse's stored importance (Step 6
    // of delta_csr_synap_row_step) -- no eps floor here. A cold row/
    // column genuinely has 0 signal and should be stored as 0; giving
    // it a fake nonzero importance would write a value into FP4 storage
    // that never actually happened. Protection against a freshly-grown
    // (real importance=0) synapse being immediately re-pruned belongs
    // entirely in delta_csr_synap_row_step's importance_cutoff
    // comparison (a read-time-only "ghost" floor, never persisted) --
    // see that function's own importance_eps parameter.
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;
    auto& L  = dc.layout;
    const std::size_t n_in  = L.rows;
    const std::size_t n_out = L.cols;
    if (n_in == 0 || n_out == 0 || k <= 0) {
        weights.probes.ptrs = 0;
        return;
    }

    const std::size_t kk_in  = std::min(static_cast<std::size_t>(k), n_in);
    const std::size_t kk     = static_cast<std::size_t>(k);

    // Top-k inputs by accumulated activity (shared by both modes)
    std::vector<std::size_t> in_idx(n_in);
    std::iota(in_idx.begin(), in_idx.end(), 0);
    std::partial_sort(in_idx.begin(), in_idx.begin() + kk_in, in_idx.end(),
        [&](std::size_t a, std::size_t b) {
            return neuron_input_accum[a] > neuron_input_accum[b];
        });
    in_idx.resize(kk_in);

    std::vector<SIZE_TYPE>   prow, pcol;
    std::vector<value_type>  pval;

    if (!per_row) {
        // Global mode: one shared top-k output set, outer product, filter after
        const std::size_t kk_out = std::min(kk, n_out);
        std::vector<std::size_t> out_idx(n_out);
        std::iota(out_idx.begin(), out_idx.end(), 0);
        std::partial_sort(out_idx.begin(), out_idx.begin() + kk_out, out_idx.end(),
            [&](std::size_t a, std::size_t b) {
                return neuron_grad_accum[a] > neuron_grad_accum[b];
            });
        out_idx.resize(kk_out);

        prow.reserve(kk_in * kk_out);
        pcol.reserve(kk_in * kk_out);
        pval.reserve(kk_in * kk_out);

        for (std::size_t r : in_idx) {
            const std::size_t n_exist = L.row_nnz(r);
            std::vector<COL_TYPE> exist_cols(n_exist);
            {
                auto cur = dc.row_cursor(r);
                for (std::size_t i = 0; i < n_exist; ++i) exist_cols[i] = cur.advance();
            }
            for (std::size_t c : out_idx) {
                if (std::binary_search(exist_cols.begin(), exist_cols.end(),
                                       static_cast<COL_TYPE>(c))) continue;
                prow.push_back(static_cast<SIZE_TYPE>(r));
                pcol.push_back(static_cast<SIZE_TYPE>(c));
                pval.push_back(neuron_input_accum[r] * neuron_grad_accum[c]);
            }
        }
    } else {
        // Per-row mode: independent top-k per row, existing conns excluded
        //    DURING selection -- guarantees up to k genuinely-new candidates
        //    per row instead of losing slots to duplicates found afterward.
        prow.reserve(kk_in * kk);
        pcol.reserve(kk_in * kk);
        pval.reserve(kk_in * kk);

        for (std::size_t r : in_idx) {
            const std::size_t n_exist = L.row_nnz(r);
            std::vector<COL_TYPE> exist_cols(n_exist);
            {
                auto cur = dc.row_cursor(r);
                for (std::size_t i = 0; i < n_exist; ++i) exist_cols[i] = cur.advance();
            }

            // Candidates = all outputs not already connected to this row.
            std::vector<std::size_t> cand;
            cand.reserve(n_out);
            for (std::size_t c = 0; c < n_out; ++c)
                if (!std::binary_search(exist_cols.begin(), exist_cols.end(),
                                        static_cast<COL_TYPE>(c)))
                    cand.push_back(c);

            const std::size_t kk_row = std::min(kk, cand.size());
            if (kk_row == 0) continue;
            std::partial_sort(cand.begin(), cand.begin() + kk_row, cand.end(),
                [&](std::size_t a, std::size_t b) {
                    return neuron_grad_accum[a] > neuron_grad_accum[b];
                });

            for (std::size_t i = 0; i < kk_row; ++i) {
                const std::size_t c = cand[i];
                prow.push_back(static_cast<SIZE_TYPE>(r));
                pcol.push_back(static_cast<SIZE_TYPE>(c));
                pval.push_back(neuron_input_accum[r] * neuron_grad_accum[c]);
            }
        }
    }

    weights.probes.rows       = static_cast<SIZE_TYPE>(n_in);
    weights.probes.cols       = static_cast<SIZE_TYPE>(n_out);
    weights.probes.ptrs       = static_cast<SIZE_TYPE>(prow.size());
    weights.probes.indices[0] = std::make_shared<std::vector<SIZE_TYPE>>(std::move(prow));
    weights.probes.indices[1] = std::make_shared<std::vector<SIZE_TYPE>>(std::move(pcol));
    weights.probes.values[0]  = std::make_shared<std::vector<value_type>>(std::move(pval));
}

#endif
