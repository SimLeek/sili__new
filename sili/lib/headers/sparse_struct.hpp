/**
 * @file sparse_struct.hpp
 * @brief Sparse matrix library with CSR and COO format support.
 *
 * Umbrella include, split (see conversation) to keep individual files under
 * ~1k lines: delta_csr_types.hpp (struct/class definitions), delta_csr_memory.hpp
 * (row-level memory operations, synaptogenesis), delta_csr_ops.hpp
 * (compact/expand_headroom, forward/backward computation). Any existing
 * `#include "sparse_struct.hpp"` continues to work unchanged -- this file
 * still provides everything it always did, just via three files instead of
 * one. New code that only needs part of this can include the specific
 * split file(s) directly instead.
 */
#ifndef __SPARSE_STRUCT_HPP_
#define __SPARSE_STRUCT_HPP_

#include "delta_csr_types.hpp"
#include "delta_csr_memory.hpp"
#include "delta_csr_ops.hpp"

#endif
