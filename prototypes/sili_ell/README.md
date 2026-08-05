# Dual-Codec Sparse Trainable Weights

Storage and kernels for sparse weight matrices that are trained online at
4 bits per weight, addressed in both orientations without atomics, and
structurally rewired (grown, pruned, expanded, contracted) in O(1) per
synapse. Two interchangeable index codecs sit behind one kernel skeleton,
and a controller switches a layer between them at runtime based on
measured statistics.

Target properties and the mechanism that delivers each:

- FAST: closed-form slot addressing (banked codec), gather-mode kernels
  with one writer per output (no atomics on any training pass), inner
  loops verified auto-vectorized by gcc (SSE + AVX2 paths emitted).
- MEMORY EFFICIENT: 32 bits per stored slot in the banked codec
  (12b + 12b index quotients, 4b fp4 weight, 4b importance); the packed
  codec is measured per layer and wins when index spans are clustered.
- SPARSE: activity bitmaps gate gathers; silence is in-band (weight
  code 0), so capacity sleeps at zero marginal cost and pruning is a
  one-byte write.
- REACHES GOOD GRAPHS: probe/commit growth with explicit block reasons,
  O(1) transposition trades, provably demotion-free capacity expansion,
  and importance-adjudicated contraction. Policy (importance scores,
  outer-product candidates) is strictly separated from mechanism.

See THEORY.md for the full lecture-notes treatment: every invariant,
proof, codec formula, and measured prediction, written for someone
entering the field rather than someone already in it.

## Files

Current implementation:

| file                        | role                                          |
|-----------------------------|-----------------------------------------------|
| dual_codec_ell.comp.glsl    | GPU kernels (Vulkan GLSL). One file, stages selected by -DSTAGE_*, codec by -DCODEC_BANKED / -DCODEC_PACKED. Atomic-free FORWARD / DX / WUPDATE; TRADE commits transpositions. |
| dual_codec_ell_cpu.hpp      | CPU translation of the same kernels: workgroups -> OpenMP, warp lanes -> flat branchless omp simd loops, byte-aligned branchless bitfield fetch. Byte-exact update semantics shared with the shader. |
| sparse_format_controller.hpp| Interchange form (Syn), packed codec encoder/decoder, multiplier audition, packed<->banked converters (lossless down, importance-contested up, regrow queue out), per-slot cost metrics, hysteresis controller. |
| synapto.hpp                 | Structural mechanism: probe_grow / commit_grow / prune, collision-asserting assemble, demotion-free expand_banked. |
| test_dual_codec.cpp         | Harness: converter invariants, both codecs vs scalar references, byte-exact update checks, trade audits, stride-structure audition demo, large-config benchmark, controller demo. |
| test_synapto.cpp            | Harness: mechanism exactness (blocked-candidate eviction paths, expansion preservation) plus the quality-under-churn experiment (starve -> expand -> improve -> contract gracefully). |
| succinct_inverse.hpp        | Optional 32 -> ~23 bits/slot: drops the stored inverse; pi^{-1} answered by O(t) cycle walks with shortcut marks (MRRR 2003). Only backward-dx and the growth probe pay; lazy per-plane rebuild after trades with a correct fallback on dirty planes. |
| test_succinct.cpp           | Harness: exhaustive inverse equality, dx parity, dirty/fallback/rebuild lifecycle, and the t-sweep table (bits per slot, hops, dx throughput). |
| test_stochastic_importance.cpp | Harness: dithered-counter unbiasedness on the real kernel, teacher/junk importance separation, prune-by-importance vs random/magnitude damage, and the exact 32.000 bits/param + O(M+N) memory ledger. |
| test_synapto_packed.cpp     | The identical churn experiment on the PACKED mechanism (in-place prune, batched re-encode growth, lossless rebuild both directions): same policy code, codec-correct constants asserted (hard capacity wall, 100% unblock on expansion). |

Reference / lineage (kept for context, superseded for production):

| file                     | role                                             |
|--------------------------|--------------------------------------------------|
| fixed_margin_ell.cuh     | First iteration: fp32 doubly-regular ELL in CUDA with explicit r2c/c2r permutations and a sync cursor. The 192-bit baseline the rest of the work compresses. |
| banked_ell.comp.glsl     | First GLSL iteration: event-driven push kernels with atomics. Superseded by the gather-mode dual-codec file; kept as the push-mode reference. |

## Build and test (CPU)

    g++ -O3 -march=native -fopenmp -std=c++17 test_dual_codec.cpp -o test_dual
    ./test_dual
    g++ -O3 -march=native -fopenmp -std=c++17 test_synapto.cpp -o test_syn
    ./test_syn
    g++ -O3 -march=native -fopenmp -std=c++17 test_succinct.cpp -o test_succ
    ./test_succ
    g++ -O3 -march=native -fopenmp -std=c++17 test_synapto_packed.cpp -o test_sp
    ./test_sp
    g++ -O3 -march=native -fopenmp -std=c++17 test_stochastic_importance.cpp -o test_imp
    ./test_imp

Both should end with "TOTAL: PASS (0 failures)". Add
-fopt-info-vec-optimized=vec.log to confirm the four hot inner loops
vectorize in your toolchain.

## Build (GPU)

One SPIR-V pipeline per (stage, codec), e.g.:

    glslangValidator -V --target-env vulkan1.1 \
        -DSTAGE_FORWARD -DCODEC_BANKED -o fwd_banked.spv dual_codec_ell.comp.glsl

Stages: STAGE_FORWARD, STAGE_DX, STAGE_WUPDATE, STAGE_TRADE (banked only).
Spec constants: LOG_S, C_LOG, R_LOG, workgroup size (id 4), W_SCALE.
Push constants: count, lr, seed, aC, aCinv, aR, aRinv, nReal.
Bindings are documented in the shader header. Kompute is a reasonable
host framework (build C++ from a pinned commit; the tagged release and
PyPI bindings lag head); the shaders do not depend on it.

## Verified claims (from the shipped harnesses)

- pi/inv are mutual inverses over every slot after conversion, updates,
  and trades (full audits).
- Forward and dx match independent scalar references for both codecs;
  weight updates match byte-exactly against independently addressed
  references (shared stochastic quantizer, independent slot math).
- Expansion (R_LOG + 1) preserves the live set and the function, with a
  runtime assert re-proving zero bank collisions; on a capacity-starved
  teacher, 4691 blocked teacher positions were measured with exactly 50%
  freed by the bank split (theory predicts one half).
- Structured-graph audition: 3x3 RGB receptive fields on the 210x160x3
  raster demote 86.3% under identity banking and 0.0% under the searched
  multiplier. Blind-hash worst case (uniform random graph) measured
  28.7%, matching the balls-in-bins formula; the controller's demotion
  gate correctly refuses that conversion.
- Succinct inverse: exhaustive equality with the stored inverse
  (average hops = t + 1), 24.3 bits per slot at t = 4 versus 32, with
  the slowdown confined to backward-dx; answers remain correct on
  dirty planes via the fallback walk. See THEORY.md 5.5 for the full
  menu between 32 bits and the ~14.6-bit family floor.
- The fused update is the whole optimizer: 32.000 bits/param persistent
  (asserted), O(M+N) transients, no optimizer state. Importance lives in
  its 4 bits as an unbiased dithered counter (imp_lr parameter on the
  update kernels; lrImp push constant appended to the GLSL push block --
  update host structs accordingly): counter means match clamp-free
  predictions, teacher/junk separation 9.12 vs 4.50, and 40%-pruning by
  importance does 24x less damage than random, at parity with magnitude.
  Zombies self-silence under the rule (contribution 0, pressure > 0).
- Structural plasticity runs on BOTH codecs with the same policy code:
  the packed churn experiment reproduces the full arc with codec-correct
  constants (hard capacity wall; 100% of row blocks freed by expansion
  vs banked's exact 50%; lower floor on random targets, per the
  occupancy coverage bound in THEORY.md 7.7).
- Quality under churn: MSE 0.0531 -> 0.0275 while capacity-starved,
  -> 0.0209 after expansion; contraction removed 29% of memory while MSE
  improved to 0.0200, with graceful monotone degradation past the knee.

## Integration seams for sili

- Interchange: everything converts through sfc::Syn {i, j, imp<<4|w4}.
  A CPU-resident layer may keep its own encoding (e.g. the existing
  SIMD uleb128 CSR); pack_view is the transcode at GPU upload.
- Policy plugs into probe_grow / commit_grow / prune. Importance decides
  evictions and contraction contests; the top-k outer product proposes
  candidates. Accumulate the outer product over a window (traces), not a
  single step; see THEORY.md lecture 8 for why, learned the hard way.
- Controller cadence: run estimate_packed_row_bits and the audition on
  the overnight harness; Controller::decide per layer; convert at a
  barrier. Expansion trigger: sustained high block rate with high grow
  demand. Contraction trigger: high zombie rate with low block rate.
- Plasticity pressure is a controller input alongside memory pressure:
  sustained banked block-rate under grow demand means either expand
  banks (demotion-free) or convert the layer to packed, decided by the
  measured bits of each. Conversions carry the weight+importance bytes,
  so plasticity state survives format switches.
- Reproducibility: bit-deterministic within a codec; across a format
  switch the dither streams differ (slot-keyed hash). Key the hash on
  (i, j) instead if cross-representation determinism matters.
- The fp4 stochastic rounding is deliberate noise; expect the branching
  ratio tracker to read it as temperature and widen its band for fp4
  layers rather than let density control fight it.

## Known gaps

- GPU shaders are logic-verified through the CPU mirror but not yet
  hardware-verified; run each stage against the CPU outputs on a small
  layer (the harness small config is the fixture).
- Kompute host scaffold not included.
- Packed in-place insertion is delegated to the existing slack-space
  uleb encoder; the PackedView here rebuilds instead.
- Power-of-two R, C, S assumed; R, C <= 128; total nnz < 2^31.
