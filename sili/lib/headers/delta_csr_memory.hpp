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

// ── Incremental synaptogenesis step ──────────────────────────────────────────

template <typename SIZE_TYPE, typename VALUES_TYPE = FP4BiPacked, typename COL_TYPE = uint32_t>
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

    const std::size_t row   = current_row % L.rows;
    current_row = (current_row + 1) % L.rows;

    const std::size_t n_exist = L.row_nnz(row);

    // Skip when there is genuinely nothing to do: no existing connections AND
    // no probe candidates. Do NOT skip when n_exist > 0 just because probes
    // are empty -- that would block the prune-only path (e.g. a fully-dense
    // layer where n_exist > max_row_weights but all (row,col) pairs already
    // exist so build_probes returns empty). Pruning must work regardless of
    // whether growth candidates are available.
    const bool has_probes = weights.probes.indices[0] &&
                            !weights.probes.indices[0]->empty();
    if (n_exist == 0 && !has_probes) return false;
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
    if (has_probes) {
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

    // BUG FIX (see conversation): row_rebuild's return value was discarded --
    // on failure (insufficient reserved headroom, e.g. right after compact(),
    // which deliberately leaves zero headroom on both axes) this function
    // reported did_work=true and updated out_degree as if the merge had been
    // written, while nnz silently never changed. A silent failure here is
    // worse than a crash: training just stops improving with no signal why.
    // Now throws (pybind converts std::runtime_error to a catchable Python
    // exception) BEFORE touching out_degree, so bookkeeping can't be
    // corrupted by a rebuild that didn't actually happen. Only the genuine
    // "growth was attempted and failed" case throws -- the "nothing to do"
    // no-op above (line ~811) is a legitimate, quiet false, not an error.
    // See compact()/expand() in this file: call expand() to restore growth
    // headroom before resuming synaptogenesis on a compacted layer.
    const bool rebuilt = delta_csr_row_rebuild(dc, row, merged_cols, merged_w, merged_imp);
    if (!rebuilt) {
        uint8_t tmp[uleb128_max_bytes<COL_TYPE>()];
        std::size_t needed_bytes = 0;
        COL_TYPE prev = 0;
        for (std::size_t k = 0; k < merged_cols.size(); ++k) {
            needed_bytes += uleb128_encode<COL_TYPE>(merged_cols[k] - prev, tmp);
            prev = merged_cols[k];
        }
        throw std::runtime_error(
            "delta_csr_synap_row_step: row " + std::to_string(row) +
            " needs " + std::to_string(needed_bytes) + " index bytes / " +
            std::to_string(merged_cols.size()) + " elements, but only has " +
            std::to_string(L.row_alloc_bytes(row)) + " bytes / " +
            std::to_string(L.row_alloc_elems(row)) + " elements of reserved "
            "headroom. Call expand_headroom() on this layer's weights.connections "
            "before resuming synaptogenesis -- growth headroom was likely "
            "removed by a prior compact() call.");
    }

    if (!weights.out_degree.empty()) {
        for (std::size_t k = 0; k < n_exist; ++k)
            if (weights.out_degree[exist_cols[k]] > 0)
                --weights.out_degree[exist_cols[k]];
        for (COL_TYPE col : merged_cols)
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
