#ifndef __DELTA_CSR_MEMORY_HPP_
#define __DELTA_CSR_MEMORY_HPP_

// Split out of sparse_struct.hpp (see conversation). Row-level memory
// operations on DeltaCSRWeights: build from / convert to absolute CSR,
// blank-space (headroom) management, row insert/remove/rebuild,
// synaptogenesis application (synap_row_step) and candidate generation
// (build_probes) -- these two are natural pairs (generate candidates, then
// apply them), kept together here. Whole-structure memory ops (compact,
// expand_headroom) and the actual forward/backward computation are in
// delta_csr_ops.hpp instead.
//
// NOTE: build_probes' docstring was previously separated from its own
// function by ~130 unrelated lines (compact/expand_headroom got inserted
// between them by an earlier edit) -- fixed here, docstring now sits
// directly above the function it documents.

#include "delta_csr_types.hpp"
#include <unordered_set>
#include <numeric>
#include <algorithm>

// ── Build from / convert to absolute CSR ─────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
DeltaCSRWeights<SIZE_TYPE, VALUES_TYPE, COL_TYPE> delta_csr_from_absolute(
    const std::vector<SIZE_TYPE>&   csr_ptrs,
    const std::vector<SIZE_TYPE>&   csr_indices,
    const std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& csr_weights,
    const std::vector<typename ValueAccessor<VALUES_TYPE>::value_type>& csr_importance,
    std::size_t rows, std::size_t cols,
    std::size_t index_bytes, std::size_t values_bytes,
    float blank_fraction = 0.2f)
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


    // BUG FIX (see conversation): this was `std::size_t blank_fraction = `
    // with nothing after the `=` before the next statement -- valid C++ that
    // silently parsed as the chained assignment `blank_fraction = (L.byte_start[0] = 0)`,
    // zeroing the intended growth-headroom fraction. Confirmed via direct
    // testing: delta_csr_row_rebuild() failed silently on every row of a
    // freshly-imported layer (row_alloc_bytes/row_alloc_elems left almost no
    // slack), so synaptogenesis could never actually grow a connection --
    // exactly matching this function's own TODO comment above ("we could run
    // out of memory... can guarantee we won't by assuming uleb128_max_bytes").
    // 0.2 (20% headroom, elementwise and bytewise) mirrors this session's own
    // delta_csr_from_absolute (1.2x multiplier) -- proven sufficient for
    // ordinary synaptogenesis rates in testing throughout this project.
    // blank_fraction is now a parameter (default 0.2, matching the original
    // fixed value -- see conversation for the earlier bug where this was
    // accidentally hardcoded via an incomplete assignment). Exposed so
    // expand() can offer real caller control over how much headroom to
    // restore, rather than reusing a fixed amount unconditionally.
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
        // byte_end[r] for r > row must shift by the same delta: the memmove
        // physically relocated the data for those rows, so byte_end[r] is now
        // pointing to the wrong (old) position. Without this update,
        // row_nnz(r) reads as 0 or underflows for any r > row during a bulk
        // equalization pass (safe in the one-per-cycle path only because
        // synap_step rebuilds byte_end via row_rebuild before the equalizer
        // touches each row).
        for (std::size_t r = row + 1; r < L.rows; ++r)
            L.byte_end[r] = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(L.byte_end[r]) + byte_delta);
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
        // Same fix as byte side: elem_end for shifted rows must track the
        // physical move. row_nnz(r) = elem_end[r] - elem_start[r] underflows
        // if elem_start is updated but elem_end is not.
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
    // BUG FIX (see conversation): must capture the OLD row_nnz BEFORE
    // L.elem_end[row] is overwritten below -- row_nnz(row) reads
    // elem_end[row]-elem_start[row], so calling it AFTER the write (as the
    // original code did, a few lines down) always returns the NEW count,
    // making nnz_delta = n - n = 0 unconditionally. Confirmed via direct
    // testing: delta_csr_row_rebuild reported success and correctly wrote
    // new synapse data, but dc.nnz()/total_nnz never changed -- so every
    // caller relying on nnz() to observe growth (including synaptogenesis
    // callers checking "did this actually grow") silently saw no change.
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


// ── In-place insert/remove for delta-encoded rows ────────────────────────────
//
// These replace the old "read all, rebuild from scratch" pattern in
// delta_csr_synap_row_step. The key property: removing a connection shrinks
// the row's byte count by (delta_A_bytes + delta_B_bytes - delta_AB_bytes),
// which is at least 0 and usually positive. Inserting a connection expands by
// delta_new_bytes + delta_updated_next_bytes - delta_old_next_bytes, which for
// typical small-delta layers (column indices 0..n_out where n_out <= 16384) is
// 1-2 bytes. Doing removals first then insertions means the freed bytes from
// removals are immediately available for insertions -- synaptogenesis with
// equal add/remove counts works with near-zero headroom.

// Remove the connection at column `target_col` from row `row` in-place.
// Returns false if target_col is not found (no-op).
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
// redistribute space from adjacent rows, then retry. Callers must not
// silently skip on false -- check the return value and handle it.
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

// ── Incremental synaptogenesis step ──────────────────────────────────────────
//
// Replaces the old "read all, merge, rebuild from scratch" approach with
// in-place per-connection insert and remove. This eliminates the need to
// pre-allocate uleb128_max (5) bytes per potential connection: for typical
// layers (n_out <= 16384), deltas encode in 1-2 bytes, so the blank space per
// row only needs to cover the NET GROWTH per step (additions - removals).
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
//
// out_degree update mirrors the old implementation: decrement for removed
// columns, increment for added columns.

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked,
          typename COL_TYPE = uint32_t>
bool delta_csr_synap_row_step(
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    std::size_t& current_row,
    typename ValueAccessor<VALUES_TYPE>::value_type importance_cutoff,
    SIZE_TYPE max_row_weights)
{
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    auto& dc = weights.connections;
    auto& L  = dc.layout;
    if (L.rows == 0) return false;

    const std::size_t row = current_row % L.rows;
    current_row = (current_row + 1) % L.rows;

    const std::size_t n_exist = L.row_nnz(row);
    const bool has_probes = weights.probes.indices[0] &&
                            !weights.probes.indices[0]->empty();
    if (n_exist == 0 && !has_probes) return false;

    // ── Step 1: Read existing connections ──────────────────────────────────
    std::vector<COL_TYPE>   exist_cols(n_exist);
    std::vector<value_type> exist_w(n_exist), exist_imp(n_exist);
    {
        auto cursor = dc.row_cursor(row);
        for (std::size_t k = 0; k < n_exist; ++k) {
            exist_cols[k] = cursor.advance();
            exist_w[k]    = ValueAccessor<VALUES_TYPE>::get_w(
                dc.values, L.elem_start[row] + k);
            exist_imp[k]  = ValueAccessor<VALUES_TYPE>::get_imp(
                dc.values, L.elem_start[row] + k);
        }
    }

    // ── Step 2: Collect probes for this row ────────────────────────────────
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

    // ── Step 3: Determine which connections to remove ──────────────────────
    // Collect indices into exist_* sorted by importance ascending.
    std::vector<std::size_t> by_imp(n_exist);
    std::iota(by_imp.begin(), by_imp.end(), 0);
    std::sort(by_imp.begin(), by_imp.end(),
              [&](std::size_t a, std::size_t b) {
                  return exist_imp[a] < exist_imp[b];
              });

    std::vector<COL_TYPE> to_remove;
    const std::size_t max_rw = static_cast<std::size_t>(max_row_weights);
    for (std::size_t rank = 0; rank < n_exist; ++rank) {
        const std::size_t k    = by_imp[rank];
        const std::size_t keep = n_exist - to_remove.size();
        if (exist_imp[k] < importance_cutoff || keep > max_rw)
            to_remove.push_back(exist_cols[k]);
    }
    // Sort descending so we remove from high col first -- keeps byte positions
    // of lower-col elements stable while we walk and remove.
    std::sort(to_remove.rbegin(), to_remove.rend());

    // ── Step 4: Determine which probes to add ─────────────────────────────
    // Filter out probes that already have a connection.
    {
        std::unordered_set<COL_TYPE> exist_set(exist_cols.begin(), exist_cols.end());
        // (Will also filter against to_remove to avoid immediately re-adding
        // a just-removed connection. Not strictly necessary but clean.)
        std::unordered_set<COL_TYPE> remove_set(to_remove.begin(), to_remove.end());

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

    // ── Step 5: Apply removes (in-place, high-col first) ──────────────────
    for (COL_TYPE col : to_remove) {
        delta_csr_row_remove_col(dc, row, col);
        if (!weights.out_degree.empty() && weights.out_degree[col] > 0)
            --weights.out_degree[col];
    }

    // ── Step 6: Apply adds in-place ──────────────────────────────────────
    // Throws on first insertion failure (no blank space in this row).
    // Caller should call equalizer_step() to redistribute blank from adjacent
    // rows before retrying. If the total pool is exhausted, prune more
    // aggressively (lower max_row_weights or raise importance_cutoff).
    for (std::size_t i = 0; i < probe_cols.size(); ++i) {
        const COL_TYPE col = probe_cols[i];
        if (!delta_csr_row_insert_col(dc, row, col, value_type(0), probe_scores[i])) {
            const std::size_t used  = dc.layout.byte_end[row] - dc.layout.byte_start[row];
            const std::size_t alloc = dc.layout.row_alloc_bytes(row);
            throw std::runtime_error(
                "delta_csr_synap_row_step: row " + std::to_string(row) +
                " ran out of blank space during insertion (used " +
                std::to_string(used) + " / " + std::to_string(alloc) +
                " bytes). Call equalizer_step() to redistribute space "
                "from adjacent rows before retrying, or reduce "
                "max_row_weights / raise importance_cutoff to prune.");
        }
        if (!weights.out_degree.empty())
            ++weights.out_degree[col];
    }

    return true;
}

// ── Probe generation (outer product, top-k) ───────────────────────────────────

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
 * applies these probes) and delta_csr_forward/backward.
 *
 * Existing-connection check uses DeltaCSRRowCursor directly (no full row
 * materialization) -- O(row_nnz) per candidate input row, same complexity
 * class as scanning the row any other way.
 *
 * Complexity: O(n_inputs + n_outputs) for the top-k selection, O(k^2 *
 * avg_row_nnz) for the existing-connection filter -- fine for small k
 * (typically 64-256, matching genesis_build_probes' historical usage).
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
 * NOTE (test): with k >= n_inputs and k >= n_outputs, every input/output is
 * a candidate, so probe count == (n_inputs*n_outputs - existing_nnz) minus
 * any duplicate outer-product collisions (none possible here since inputs
 * and outputs are each selected without repetition) -- a useful upper-bound
 * regression check.
 * NOTE (test): a pair that already exists in connections must never appear
 * in probes, regardless of how high its accum-derived importance would be.
 * NOTE (test): per_row=true must yield >= as many probes as per_row=false
 * for the same (weights, accum, k) -- guaranteed since per_row never wastes
 * a candidate slot on an already-connected pair. Construct a case where a
 * row's shared-top-k output set is dominated by its own existing
 * connections (global mode starves it) to see the difference concretely.
 */
template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
void delta_csr_build_probes(
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights,
    const typename ValueAccessor<VALUES_TYPE>::value_type* neuron_input_accum,
    const typename ValueAccessor<VALUES_TYPE>::value_type* neuron_grad_accum,
    SIZE_TYPE k,
    bool per_row = false)
{
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

    // ── Top-k inputs by accumulated activity (shared by both modes) ──────────
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
        // ── Global mode: one shared top-k output set, outer product, filter after ──
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
        // ── Per-row mode: independent top-k per row, existing conns excluded
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

            // Candidates = all outputs NOT already connected to this row.
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
