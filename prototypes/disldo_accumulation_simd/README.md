# disldo_forward accumulation loop -- SIMD attempts (all negative on this hardware)

Four correctness-verified prototypes targeting `linear_disldo.hpp`'s
`disldo_forward` inner accumulation loop (`mo[...] += contrib`, around
lines 104-135 as of this writing -- see that file directly, line numbers
drift). Confirmed via real-checkpoint py-spy profiling (see
sili_peridot's JOURNAL.md) that this loop is the actual dominant cost in
the dense forward path: ~55% of eval-phase samples, vs. ~1% for column-
index decode (see the `for_delta_encoding/` prototype next to this one --
decode was never the bottleneck, this loop is). All four attempts here
are real losses on THIS machine (Ryzen 7 3750H, Zen+, AVX2 + BMI1/BMI2,
no AVX-512) -- kept as working, tested reference in case a different CPU
architecture (particularly one with real hardware scatter, i.e. AVX-512,
or a faster AVX2 gather implementation than AMD Zen/Zen+ has) makes any
of these approaches pay off. **Do not merge any of this into the real
library without re-testing on the target hardware first** -- these are
confirmed *negative* results here, not confirmed positive results
elsewhere.

## Constraint that shaped all four attempts: no batch parallelism

sili's real, primary use case is online/recurrent learning -- batch=1
almost always in practice. `sili_peridot`'s B6/B7 one-shot conversion
pipeline happens to batch many tokens together per fold step (batch=
20-50 in the real-checkpoint benchmark this was profiled against), but
that's an atypical use of the library, not representative of what the
core library should be optimized around. Attempt 1 (below) initially
looked promising specifically because it vectorized across that batch
dimension -- a real result, but for the wrong workload; not adopted.

## The four attempts

1. **`attempt1_gather_batch_axis.cpp`** -- vectorize across 8 batch
   elements for a fixed synapse (AVX2 gather for the read, SIMD add,
   8 individual scalar stores back since AVX2 has no scatter). Real,
   correctness-verified **1.3-1.84x speedup** at batch=20-50 -- but
   this is the batch axis, which doesn't exist at batch=1. Superseded
   by attempt 4, which applies the same technique to the correct axis.

2. **`attempt2_windowed_local_buffer.cpp`** -- exploit that CSR
   columns within a row are sorted (both the old ULEB128 encoding and
   the new FOR encoding preserve this) and, at real checkpoint density
   (67-90%), a decoded group's columns cluster into a small window
   rather than spanning the whole output width. Scatter the group's
   values into a small per-window local buffer, then do one contiguous
   vectorized load-add-store against `mo[]` over that window. **2x
   SLOWER** than plain scalar -- the local-buffer scatter is exactly
   the same O(n) scalar work as the original loop, just redirected to
   a temporary buffer, PLUS a redundant memset and a second merge
   pass. Net addition of work, not a reduction.

3. **`attempt3_windowed_two_pointer_merge.cpp`** -- fixes attempt 2's
   redundant buffer: no per-window scratch allocation, direct
   two-pointer merge between window positions and the sorted synapse
   list, filling one small (8-float, register-resident) chunk at a
   time and adding straight into `mo[]`. Also skips chunks with no
   active synapse at all (tracked for free from the merge, no separate
   SIMD compare needed). Still **2-3x SLOWER** -- the two-pointer
   merge's own bookkeeping/branching overhead exceeds what's saved by
   having fewer, wider vector ops at this problem scale (groups of 32,
   thousands of synapses per row).

4. **`attempt4_gather_synapse_axis.cpp`** -- attempt 1's proven
   technique (gather + SIMD add + scalar store-back), reapplied to the
   correct axis: 8 synapses at once, batch=1, no merge logic needed
   (just gather 8 arbitrary `mo[]` positions, multiply w[]*input_r,
   add, store back individually). Still **1.3-1.9x SLOWER**. Best
   explanation: AVX2 `vgatherdps` throughput on AMD Zen/Zen+ is
   documented to be substantially worse than scalar loads for
   genuinely random access patterns (unlike attempt 1's batch-axis
   case, where the 8 gather addresses were regularly strided by
   `n_out`, not arbitrary -- a much easier case for the gather unit).
   This is a real, hardware-specific limitation, not a flaw in the
   reasoning -- Intel chips and/or AVX-512's dedicated scatter may
   perform very differently.

## What to check on different hardware

- Re-run all four benchmarks as-is (each is standalone, `g++ -O3
  -march=haswell -mavx2 -mbmi2 -std=c++17 <file>.cpp -o bench`, adjust
  `-march`/add `-mavx512f` etc. for the target CPU) before assuming any
  of this applies.
- Attempt 4 (gather on the synapse axis) is the most likely to flip
  positive on hardware with genuinely fast AVX2 gather (some Intel
  parts) or real scatter (AVX-512) -- worth trying `_mm512_i32scatter_ps`
  directly if targeting AVX-512, rather than the gather+scalar-store
  emulation here.
- If pursuing a hardware accelerator (FPGA or otherwise) for this,
  the actual bottleneck operation is a sparse-vector accumulate with
  known-sorted, known-dense (67-90%) column indices -- a custom
  scatter-add or "segmented dense accumulate" primitive matching
  attempt 3's intent (but without the CPU-side merge overhead) is
  probably the right target, not a general gather/scatter unit.

See `../for_delta_encoding/` for the related column-index decode work
(4-12x faster in isolation, but confirmed via profiling to be ~1% of
total time -- not the bottleneck this folder targets).
