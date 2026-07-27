# Frame-of-reference delta encoding -- SIMD decode prototype

Reference/design-validation code for the group-relative (frame-of-reference,
FOR) delta encoding being integrated into `DeltaCSRWeights`'s column-index
storage (see conversation). Not part of the active CMake build -- this is
the prototype that motivated and validated the real implementation, kept
here as a working reference rather than lost in a scratch directory.

## Background

The current column-index storage (`delta_csr_types.hpp`'s
`DeltaCSRRowCursor`) encodes each column index as a ULEB128-varint DELTA
from the PREVIOUS column index. Reconstructing absolute column indices
requires a running cumulative sum, which is an inherently sequential
dependency chain -- confirmed via `gcc -fopt-info-vec-missed` that this
specific loop cannot be auto-vectorized (see sili_peridot's JOURNAL.md).

Seven SIMD decode strategies were tried against the EXISTING (delta-from-
previous) encoding before this one; all six that touch the raw ULEB128
byte stream converged on the same ~1.6-2.5x ceiling, because every one of
them was still fighting the same cumulative-sum dependency, just with
progressively cleverer SIMD around it (see sili_peridot's JOURNAL.md for
the full trail: branchy fallback, correct-boundary scalar fallback,
group-varint re-encoding, two-pass decoupled prefix sum, wider uint16
lanes, warp/SIMT-style decode across rows).

## The actual fix: encode offset-from-group-start, not delta-from-previous

Column indices are monotonically increasing (all deltas are positive).
Grouping values into blocks of G and encoding each value as an OFFSET
FROM THAT GROUP'S OWN STARTING VALUE (rather than a delta from the
immediately preceding value) makes every value in a group independently
computable as `group_start + offset[i]` -- no dependency on any other
value in the group. Decode becomes: widen the group's fixed-width offsets,
ONE broadcast-add of `group_start`, done. No shift-add prefix-sum tree at
all (unlike the group-varint attempts, which still needed one). The
group's own last (already-computed) output value doubles as the next
group's `group_start` for free -- the cross-group dependency that a
two-pass design needs a whole separate pass for costs nothing extra here.

Format: groups of G values, each group prefixed by a 1-byte width-tier
descriptor (0/1/2 = 1/2/4 bytes per offset, chosen by that group's own max
offset), followed by G offsets at that fixed width.

## Results (this file, `for_encoding_bench.cpp`)

Correctness verified bit-exact against a scalar ULEB128 reference decoder
in every configuration tested (G in {8, 16, 32, 64}, multi-byte-delta rate
in {0%, 0.1%, 1%}, N in {100..8000} matching real per-row nnz scale).

| G | speedup (0% multi-byte) | size overhead (0% mb) | size overhead (1% mb) |
|---|---|---|---|
| 8 | 1.7-2.6x | ~12-19% | ~17-20% |
| 16 | 2.8-3.9x | ~6-19% | ~17-22% |
| 32 | 4.0-5.3x | ~3-32% | ~27-53% |
| 64 | 4.3-12.0x (mostly 5-7x) | ~2-30% | ~52-53% |

Real MiniCPM5 checkpoint data (measured separately, see sili_peridot's
JOURNAL.md) is close to the 0% multi-byte column for the suffixes
sampled -- median delta 1, ~100% single-byte in the current per-delta
ULEB128 encoding. Bigger groups amortize the per-group tier descriptor
better at low multi-byte rates, but pay more when a single large delta
forces the WHOLE group to a wider tier -- a real tradeoff, not
hand-waved, and part of why the real implementation makes group size
configurable rather than hardcoding one value: sparsity/locality patterns
(e.g. synapses clustering near the diagonal, if that turns out to hold)
could shift which G is actually best for a given weight tensor.

## Build

```
g++ -O3 -march=haswell -mavx2 -mbmi2 -std=c++17 for_encoding_bench.cpp -o bench
./bench
```

Requires AVX2 + BMI2 (matches this project's existing `-march=haswell`
CMake flag). No dependency on the rest of sili__new -- fully standalone.
