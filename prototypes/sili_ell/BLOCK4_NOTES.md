# Block4 design notes: hardware- and structure-dependent choices

`dense_block4.hpp`'s tile size (4x4) and the measured ~10% local-fill
breakeven are **specific to this machine** (AMD Ryzen 7 3750H, Zen+) and
to the uniform-random test data used to validate them. Both need
re-checking, not blindly reusing, on different hardware or different
real sparsity patterns.

## Tile size is a hardware-specific choice, not a universal constant

Confirmed via `-fopt-info-vec` and hand-written `_mm256` intrinsics
(see conversation, `bench_block_kernel.cpp`): on this CPU, true AVX2
(256-bit, 8-wide float) throughput is **not actually available** --
hand AVX2 intrinsics beat GCC's own SSE-width (128-bit, 4-wide)
auto-vectorization by only ~4-6%, not the ~2x a real 8-wide unit would
give. This matches Zen/Zen+'s well-documented internal AVX2
implementation as two 128-bit micro-ops (effectively double-pumped) --
pre-Zen2 AMD chips don't have a real 256-bit datapath. 4x4 was sized to
this measured ~4x real ceiling deliberately, not an assumed 8x.

**On different hardware** (a real full-rate-AVX2 chip -- Intel
Skylake or later, or AMD Zen2+; or an AVX-512 machine with a genuine
16-wide float unit), **8x8 or 16x16 tiles are the matched choice
instead**, and would need their own from-scratch validation pass: the
`-fopt-info-vec` check, the hand-intrinsics comparison, and especially
the local-fill breakeven sweep (bench_block4_dual.cpp's methodology) --
none of these numbers transfer across CPU generations. This is also
architecturally a **different network** each time, not just a
recompile: the tile size determines what "locally dense enough to
promote" means, so a model converted/trained assuming 4x4 breakeven
would carry different structural decisions (which regions ended up in
the block path vs. the scattered disldo path) than the same model
targeting 8x8 or 16x16 on different hardware.

**Checked directly (not assumed) whether going SMALLER helps: it
doesn't, and 2x2 is a real regression, not just a non-improvement.**
Extended `bench_block_kernel.cpp` with BS=2: `#pragma omp simd`
vectorization attempted on a 2-lane accumulator came out **0.66x
scalar (34% SLOWER)** -- the setup/teardown cost of vectorizing
something that thin exceeds any benefit, and 2 lanes can't fill even a
128-bit SSE register's 4-float width. BS=4 (this file's actual choice)
gets a real but modest 1.23-1.29x; the benefit climbs with tile size
on this hardware's 4-wide ceiling (BS=8: ~2.2-2.3x, BS=16: ~2.5-2.6x,
diminishing returns past the point where a tile fills more than one
native register's worth of work per fixed per-tile overhead). The
general principle, likely true on other hardware too, not just this
chip: **the tile needs to be at least as wide as the machine's native
SIMD register (in float32 lanes) to see any real benefit at all** --
below that, you pay vectorization overhead for zero (or negative)
return, regardless of architecture. 4x4 is this CPU's floor as well as
its ceiling (its native width IS effectively 4, given AVX2's
double-pumped behavior here); a real AVX2 or AVX-512 machine's floor
would likely sit at 8x8 or 16x16 respectively, not lower -- 2x2 or even
4x4 tiles would likely under-fill their wider native registers the
same way 2x2 under-fills this one.

## Future idea: compile-time architecture detection, multiple tile sizes, runtime promotion/demotion between them

Noted for later, not scoped or built: GCC can detect the target
architecture at build time (`-march=native`, or explicit
`__builtin_cpu_supports("avx512f")`-style runtime dispatch), so a real
production version of this design could compile SEVERAL tile-size
variants (4x4, 8x8, 16x16 -- whichever the detected hardware's native
SIMD width actually supports well, per the finding above) into the
same binary, with a **runtime promotion/demotion policy choosing
between them per region**, not just choosing once between "block4 vs
disldo" per tile the way this prototype does. This is a direct
generalization of Fable's own `sparse_format_controller.hpp`
`Controller` (measured bits/param + hysteresis + dwell count, deciding
packed vs. banked per layer) -- the same idea, extended to a THIRD axis
(tile size) rather than just two codecs. Not attempted here: the
breakeven-fill-rate sweep would need to be re-run per candidate tile
size on the SAME machine to know when promotion between sizes is
actually worth it, and the memory/bookkeeping cost of maintaining
multiple live tile-size regions in one layer is a real, unexplored
design question of its own.

## Structure matters at least as much as tile size

The real breakeven check on uniform-random synthetic data (fill rate
needed to beat disldo) is a *worst-case-ish* baseline, not
representative of every real sparsity pattern. Checked directly against
the real, already-pruned MiniCPM5 checkpoint (`model.embed_tokens.weight`,
20.16% density): local block fill tracks overall density closely (no
strong natural clustering at 8x8/16x16, a small real signal only at
4x4) -- consistent with uniform-random, not adversarial, but also not
favorably structured.

**Some real sparse network patterns cluster far more favorably than
uniform-random**, and this design (or an 8x8/16x16 variant on other
hardware) would benefit substantially more from them:

- **Diagonal-banded sparsity** -- some architectures (local/windowed
  attention being the most direct example already in this codebase's
  own `banded_attention`, but also some learned sparse patterns in
  general) concentrate connections near the diagonal of the weight
  matrix. A diagonal band maps onto a narrow strip of DENSE 4x4/8x8/
  16x16 tiles running along the matrix's diagonal, with everything off
  the band empty -- close to the best case for this design (most active
  tiles near-fully dense, most of the matrix skipped entirely).
- Fable's own `sili_ell` README documents an analogous, directly
  relevant finding for its banked codec: structured data (3x3 receptive
  fields on a raster) demoted 86.3% under a naive/identity scheme but
  0.0% under a searched multiplier -- i.e. exploitable structure can be
  the difference between "barely works" and "works great" for a format
  whose whole premise depends on locality/regularity, same as this one.

**Not yet checked**: whether real importance-driven synaptogenesis on
this project's fold-recurrent architecture produces diagonal-like or
otherwise favorably-clustered structure, or the scattered/long-distance
pattern the fold recurrence's own architecture argues for (see
conversation -- the recurrent accumulated state gives every position a
legitimate reason to want a long-distance connection, which argues
against clustering). This can only be checked by actually running
synaptogenesis at scale and inspecting the resulting sparsity pattern's
local fill statistics the same way `BLOCK4_NOTES` checked the static
pruned weights above -- not yet done.
