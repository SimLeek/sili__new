#ifndef __BLOCK4_CODEC_HPP_
#define __BLOCK4_CODEC_HPP_

// Block4Codec<VALUES_TYPE>: a thin per-precision wrapper trait, NOT a
// reimplementation of fp4quant.hpp/fp8quant.hpp's bit-quantization logic.
// linear_disldo.hpp/sisldo_ops.hpp currently branch on
// `if constexpr (std::is_same_v<VALUES_TYPE, ...>)` three times per kernel
// to do near-identical math, differing only in which decode/encode call
// gets made -- this trait lets that math be written ONCE as a generic
// template function parameterized on `Block4Codec<VALUES_TYPE>`, instead
// of being copy-pasted per precision. See
// docs/research/linear_disldo.rst:block4_codec_refactor for the full
// design rationale and the phased rollout this is Phase 1 of.
#include "delta_csr_types.hpp" // pulls in block4.hpp (Block4Tile/8/32), which
                               // itself pulls in fp4quant.hpp/fp8quant.hpp

// Primary template intentionally undefined -- only the three specializations
// below are valid VALUES_TYPE instantiations, matching the set linear_disldo.hpp
// and sisldo_ops.hpp already dispatch on.
template <typename VALUES_TYPE> struct Block4Codec;

// ── FP4 ──────────────────────────────────────────────────────────────────
// Packed single-byte-per-slot storage: low nibble = weight code, high
// nibble = importance code (Block4Tile::at, block4.hpp). has_zero_escape
// is true here ONLY -- see quant_floor's own comment.
template <> struct Block4Codec<FP4BiPacked> {
    using tile_type = Block4Tile;
    static constexpr std::size_t scratch_bytes = BLOCK4_TILE_SLOTS;
    static constexpr bool has_zero_escape = true;
    static constexpr bool has_was_live_gate = true;

    static float decode_weight(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        return FP4_TABLE[tdata[Block4Tile::slot_index(li, lj)] & 0xFu];
    }
    static float decode_importance(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        return FP4_TABLE[(tdata[Block4Tile::slot_index(li, lj)] >> 4) & 0xFu];
    }
    // SIMD-width (li=0..3) weight decode for a fixed column lj -- used by
    // disldo_forward's cross-tile/column-pairing loops, which need all 4
    // rows of a column at once, not one (li,lj) cell at a time.
    static Block4Vec decode_weight_column4(const uint8_t* tdata, uint32_t lj) {
        const Block4VecU codes = {uint32_t(tdata[Block4Tile::slot_index(0, lj)] & 0xFu),
                                  uint32_t(tdata[Block4Tile::slot_index(1, lj)] & 0xFu),
                                  uint32_t(tdata[Block4Tile::slot_index(2, lj)] & 0xFu),
                                  uint32_t(tdata[Block4Tile::slot_index(3, lj)] & 0xFu)};
        return block4_vec_decode_fp4(codes);
    }
    // FP4-specific: the AQRS gradient-accumulation terms (mrow/mgamma/mcol)
    // substitute a small epsilon for an exactly-zero quantized weight, so a
    // permanently-dead synapse can still escape zero via its importance
    // gradient -- see disldo_backward.deferred_vs_direct_quant in
    // docs/research/linear_disldo.rst. FP8/FP32 have no such floor (their
    // specializations below return `quant` unchanged).
    static float quant_floor(float quant, float zero_escape_eps) {
        return quant == 0.0f ? zero_escape_eps : quant;
    }
    template <bool StochasticRounding>
    static void encode(uint8_t* tdata, uint32_t li, uint32_t lj, float w, float imp,
                       bool was_live) {
        const uint32_t slot = Block4Tile::slot_index(li, lj);
        uint8_t new_w, new_imp;
        if constexpr (StochasticRounding) {
            if (was_live) {
                new_w = fp4_quantize_stochastic_live(w);
                new_imp = fp4_quantize_stochastic_live_nonneg(imp);
            } else {
                new_w = fp4_quantize_stochastic(w);
                new_imp = fp4_quantize_stochastic(imp);
            }
        } else {
            if (was_live) {
                new_w = fp4_quantize_live(w);
                new_imp = fp4_quantize_live(imp);
            } else {
                new_w = fp4_quantize(w);
                new_imp = fp4_quantize(imp);
            }
        }
        tdata[slot] = uint8_t((new_imp << 4) | new_w);
    }
    static std::size_t stored_tile_len(bool is_sparse, const uint8_t* tile_bytes) {
        return block4_stored_tile_len(is_sparse, tile_bytes);
    }
};

// ── FP8 (E4M3) ───────────────────────────────────────────────────────────
// Two separate byte ranges: weight byte at slot_index, importance byte at
// BLOCK4_TILE_SLOTS + slot_index (Block4Tile8, block4.hpp).
template <> struct Block4Codec<FP8BiValues> {
    using tile_type = Block4Tile8;
    static constexpr std::size_t scratch_bytes = BLOCK4_TILE_SLOTS8_BYTES;
    static constexpr bool has_zero_escape = false;
    static constexpr bool has_was_live_gate = true;

    static float decode_weight(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        return fp8_decode_bits(tdata[Block4Tile8::slot_index(li, lj)]);
    }
    static float decode_importance(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        return fp8_decode_bits(tdata[BLOCK4_TILE_SLOTS + Block4Tile8::slot_index(li, lj)]);
    }
    static Block4Vec decode_weight_column4(const uint8_t* tdata, uint32_t lj) {
        const Block4VecU codes = {uint32_t(tdata[Block4Tile8::slot_index(0, lj)]),
                                  uint32_t(tdata[Block4Tile8::slot_index(1, lj)]),
                                  uint32_t(tdata[Block4Tile8::slot_index(2, lj)]),
                                  uint32_t(tdata[Block4Tile8::slot_index(3, lj)])};
        return block4_vec_decode_fp8(codes);
    }
    static float quant_floor(float quant, float /*zero_escape_eps*/) { return quant; }
    template <bool StochasticRounding>
    static void encode(uint8_t* tdata, uint32_t li, uint32_t lj, float w, float imp,
                       bool was_live) {
        const uint32_t slot = Block4Tile8::slot_index(li, lj);
        uint8_t new_w, new_imp;
        if constexpr (StochasticRounding) {
            if (was_live) {
                new_w = fp8_quantize_stochastic_live(w);
                new_imp = fp8_quantize_stochastic_live_nonneg(imp);
            } else {
                new_w = fp8_quantize_stochastic(w);
                new_imp = fp8_quantize_stochastic(imp);
            }
        } else {
            if (was_live) {
                new_w = fp8_quantize_live(w);
                new_imp = fp8_quantize_live(imp);
            } else {
                new_w = fp8_quantize(w);
                new_imp = fp8_quantize(imp);
            }
        }
        tdata[slot] = new_w;
        tdata[BLOCK4_TILE_SLOTS + slot] = new_imp;
    }
    static std::size_t stored_tile_len(bool is_sparse, const uint8_t* tile_bytes) {
        return block4_stored_tile_len8(is_sparse, tile_bytes);
    }
};

// ── FP32 ─────────────────────────────────────────────────────────────────
// Raw float storage, two separate ranges (Block4Tile32, block4.hpp). No
// quantization at all -- no was_live gate, no zero-escape floor, no
// stochastic/deterministic distinction (StochasticRounding is accepted
// but unused, matching disldo_backward's existing FP32 write-back, which
// is a plain memcpy either way).
template <> struct Block4Codec<DeltaCSRBiValues<float>> {
    using tile_type = Block4Tile32;
    static constexpr std::size_t scratch_bytes = BLOCK4_TILE_SLOTS32_BYTES;
    static constexpr bool has_zero_escape = false;
    static constexpr bool has_was_live_gate = false;

    static float decode_weight(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        float v;
        std::memcpy(&v, tdata + sizeof(float) * Block4Tile32::slot_index(li, lj), sizeof(v));
        return v;
    }
    static float decode_importance(const uint8_t* tdata, uint32_t li, uint32_t lj) {
        float v;
        std::memcpy(&v,
                    tdata + sizeof(float) * (BLOCK4_TILE_SLOTS + Block4Tile32::slot_index(li, lj)),
                    sizeof(v));
        return v;
    }
    static Block4Vec decode_weight_column4(const uint8_t* tdata, uint32_t lj) {
        Block4Vec w;
        std::memcpy(&w, tdata + sizeof(float) * Block4Tile32::slot_index(0, lj), sizeof(w));
        return w;
    }
    static float quant_floor(float quant, float /*zero_escape_eps*/) { return quant; }
    template <bool StochasticRounding>
    static void encode(uint8_t* tdata, uint32_t li, uint32_t lj, float w, float imp,
                       bool /*was_live*/) {
        const uint32_t slot = Block4Tile32::slot_index(li, lj);
        std::memcpy(tdata + sizeof(float) * slot, &w, sizeof(w));
        std::memcpy(tdata + sizeof(float) * (BLOCK4_TILE_SLOTS + slot), &imp, sizeof(imp));
    }
    static std::size_t stored_tile_len(bool is_sparse, const uint8_t* tile_bytes) {
        return block4_stored_tile_len32(is_sparse, tile_bytes);
    }
};

// Thin 2D view over a flat scratch buffer -- was a local struct defined inside
// disldo_backward itself (task #295); hoisted here (templated on value_type
// explicitly, since it no longer has a using-alias value_type in scope) so
// the free functions extracted below can use it too.
template <typename value_type> struct Flat2DView {
    value_type* base;
    std::size_t stride;
    inline value_type* operator[](std::size_t k) const { return base + k * stride; }
};

// disldo_backward.block4_extract_function_refactor: parameter-object structs for
// pulling disldo_backward's nested process_tile/process_row_pair/process_tile_pair
// closures out into real, independently-measured (by lizard/CCN tools) free
// functions instead of [&]-capturing lambdas defined inline inside disldo_backward.
// See docs/research/linear_disldo.rst:block4_backward_extract_function.
//
// GPU-portability note (why this shape, not std::function/virtual dispatch): this
// kernel is a future Kompute/Vulkan/GLSL port target, and GLSL has no lambdas, no
// closures, no runtime polymorphism -- only plain functions taking explicit
// parameters/structs. Converting these closures into real functions taking a small
// parameter-object struct is a move TOWARD that model, not away from it. The two
// structs below are split to mirror a future GPU binding layout: Block4BackwardParams
// is broadcast/read-only state (constant for the whole disldo_backward call --
// eventually a UBO/push-constant), Block4BackwardAccumulators is the per-thread
// mutable output state (eventually per-workitem SSBO writes, reduced differently on
// GPU via atomics/workgroup-reduce, but the same field grouping). Per-call
// coordinates (br, bc, li, row, tdata pointers) are deliberately NOT folded into
// either struct -- they stay explicit function parameters, since on GPU they'd be
// derived from gl_GlobalInvocationID (genuinely per-invocation, not uniform state).
template <typename SIZE_TYPE, typename VALUES_TYPE, typename COL_TYPE> struct Block4BackwardParams {
    using value_type = typename ValueAccessor<VALUES_TYPE>::value_type;
    SparseLinearWeightsDelta<SIZE_TYPE, VALUES_TYPE, COL_TYPE>& weights;
    const value_type* input;
    const value_type* output_grad;
    SIZE_TYPE batch;
    SIZE_TYPE in_cols;
    std::size_t n_in;
    std::size_t n_out;
    std::size_t rank;
    value_type learning_rate;
    value_type beta2;
    value_type eps;
    value_type min_decay_frac;
    value_type max_abs_delta;
    value_type max_ci;
    value_type zero_escape_eps;
    bool damp_by_importance;
    bool scale_invariant;
    bool lr_per_row_nnz;
    const value_type* gamma_k_arr;  // [rank]
    const uint32_t* row_live_count; // [n_in]
};

// Per-thread mutable output accumulators -- replaces the mcol_at/mrow_at/
// mcol_at_contrib/mrow_at_contrib/mgamma_at/mgamma_at_contrib lambdas (which
// captured raw base pointers by reference) with real named methods on a real
// struct; call-site syntax is unchanged (accum.mcol_at(col, k)). tid is this
// thread's index, needed to address weights.scale_rank_scratch's own
// per-thread-strided scratch buffers. Constructed once per thread, inside the
// parallel region, before the per-row-block loop.
template <typename value_type> struct Block4BackwardAccumulators {
    int tid;
    value_type* mdx; // this thread's dx accumulator slice
    value_type* mcol_base;
    value_type* mcol_contrib_base;
    double* mrow_base;
    double* mrow_contrib_base;
    value_type* mgamma_base;
    value_type* mgamma_contrib_base;
    std::size_t rank;

    value_type& mcol_at(std::size_t col, std::size_t k) { return mcol_base[col * rank + k]; }
    value_type& mcol_at_contrib(std::size_t col, std::size_t k) {
        return mcol_contrib_base[col * rank + k];
    }
    double& mrow_at(std::size_t row, std::size_t k) { return mrow_base[row * rank + k]; }
    double& mrow_at_contrib(std::size_t row, std::size_t k) {
        return mrow_contrib_base[row * rank + k];
    }
    value_type& mgamma_at(std::size_t k) { return mgamma_base[k]; }
    value_type& mgamma_at_contrib(std::size_t k) { return mgamma_contrib_base[k]; }
};

// Replaces process_tile_pair's std::pair<bool,bool> return -- std::pair has no GLSL
// analogue, a plain struct does.
struct Block4TileDirtyPair {
    bool a;
    bool b;
};

#endif
