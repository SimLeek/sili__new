``linear_disldo.hpp`` research notes
=====================================

Background for the dense-input DISLDO (Dense Input, Sparse Linear, Dense
Output) forward/backward kernels. This document holds the investigation
narrative, measured numbers, and design-tradeoff reasoning that used to
live inline as comments in ``sili/lib/headers/linear_disldo.hpp``; the
source file keeps only terse, load-bearing comments and points here for
the "why."

Generic over ``VALUES_TYPE`` via ``ValueAccessor`` -- the same code
compiles for ``FP4BiPacked`` (4-bit) and ``DeltaCSRBiValues<float>``
(32-bit fallback), matching ``sisldo_forward`` (the sparse-input
equivalent in ``sisldo_ops.hpp``) and ``delta_csr_synap_row_step`` /
``delta_csr_build_probes``, which already used this pattern. This file
supersedes an earlier float32/absolute-CSR ``disldo_forward``/
``disldo_backward`` that never used ``DeltaCSRLayout``/``FP4BiPacked``
at all. The dense-input walk is embarrassingly parallel by input row,
unlike the sparse-input SISLDO path, which needs a work-offset table to
balance threads across a variable-density CSR batch.

.. _disldo_forward:

``disldo_forward``
-------------------

*ID:* ``disldo_forward``

.. _disldo_forward.pure_computation:

Forward is pure computation now -- no importance side effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.pure_computation``

Forward used to run its own gradient-free ADSP-style (Activity-Dependent
Structural Plasticity) importance update whenever a nonzero
``learning_rate`` was passed, independently of whether a matching
``backward_dense()`` call would ever follow. This was not Hebbian
learning (no VALUE changes, only the importance/wiring-strength signal --
closer to activity-driven synaptic sprouting/pruning than a weight
update), and it was confirmed as a real footgun: traced directly, it
fired on every forward call including ones with no corresponding
gradient (e.g. every non-query tick of an online RNN), measurably
corrupting training at low learning rates independent of any real task
signal.

Importance is now updated ONLY by ``disldo_backward()``, coupled to a
real gradient -- the same principle weight updates have always followed
(backward-only). The mechanism was REMOVED, not just disabled: a caller
that still wants an unconditional activity-correlation signal should
build that explicitly rather than getting it silently bundled into every
forward pass.

Test invariant: output must equal the dense matmul ``input @ W_dense``
where ``W_dense[r,c]`` is the weight of synapse ``r->c``. Same reference
check used for ``sisldo_forward`` and the standalone ``disldo_ops.hpp``
path -- both pass it.

.. _disldo_forward.dc_empty_check:

``!dc.empty()`` no longer means "nothing to do"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.dc_empty_check``

block4 may hold live synapses even when the scattered CSR is empty (e.g.
everything in a small/dense layer has been promoted). ``L.rows``/
``L.cols`` stay valid either way (set at construction, independent of
nnz), so skipping just the scattered block when ``dc.empty()`` is safe.

.. _disldo_forward.rank_n_scale:

Rank-N scale
~~~~~~~~~~~~

*ID:* ``disldo_forward.rank_n_scale``

``w = w_stored * weights.get_scale(r, col)`` -- see ``scale_rank``'s own
docstring (``delta_csr_types.hpp``). Reduces to the exact original
``val_scale*out_scale`` at ``scale_rank==1``.

.. _disldo_forward.tile_coord_collection:

block4 tile-coordinate collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.tile_coord_collection``

Row-major cursor walk isn't parallel-for-friendly directly (same reason
the old hash-map iteration wasn't), so this collects ``(br, bc,
elem_pos, byte_pos)`` tuples once per call before the parallel region,
rather than pre-collecting ``Block4Tile`` pointers/handles. A handle
can't be pre-collected across the parallel region since it's move-only
RAII, and per-tile compress/decompress decisions must happen within one
thread's ownership of one tile at a time -- each thread constructs its
own handle fresh, inside the loop body, from these coordinates.

``elem_pos`` is the tile's index into block4's own address space (this
walk already knows it for free, as ``block_layout.elem_start[br]+bk``).
``byte_pos`` is its position in block4's flat variable-length
``tile_data`` buffer -- a tile's byte length varies with its live count
when sparse, so unlike ``elem_pos`` this can't be derived by simple
arithmetic; it's the running sum of every preceding tile's real length,
which this collection loop is already computing. Both are passed to
``Block4Store::at_index()`` so the hot loop below doesn't redo an
O(row_nnz) coordinate re-scan per tile via ``find()``. That redundant
second scan (discover a tile here, then re-discover it again via
``find()``'s own ``raw_find()``) measured as the dominant real cost of
this loop at batch=1 -- batch=1 has too little per-tile compute (16
FLOPs) to amortize even one such scan, let alone two.

Persistent scratch (``Block4Store::scratch_tile_br/bc/elem/byte``), not
a fresh vector every call: batch=1 real-time calls can't amortize
repeated heap allocation the way a large training batch could.

``resize()`` + direct indexing, not ``reserve()`` + ``push_back()``:
``push_back``'s per-call capacity check (branch + increment) is real,
measured exclusive cost at this scale -- roughly 49k ``push_back`` calls
across the 3 vectors on a fully block4-resident 512x512 layer, confirmed
via callgrind. Switching to scratch buffers alone barely moved this cost
since reused capacity still pays the per-``push_back`` check every call
regardless of allocation. ``resize()`` is a single capacity check for
the whole vector; the fill loop then writes through plain indexed
stores.

.. _disldo_forward.per_thread_output_buffers:

block4 forward compute: per-thread output buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.per_thread_output_buffers``

Same pattern as the scattered path's ``t_out`` above, and necessary, not
optional: two tiles that share a block-COLUMN (different block-rows,
i.e. different input rows feeding the same output columns) write to the
same output positions, so parallelizing freely over tiles without this
would race exactly the way the scattered path's own scatter-write would
without ``t_out``.

.. _disldo_forward.hoisted_tile_count:

Hoisting ``tile_br.size()`` -- measured instruction-count win, no
wall-clock effect
~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.hoisted_tile_count``

``n_tiles_local`` is hoisted out of the loop condition rather than
re-reading ``tile_br.size()`` every iteration. Measured, not assumed:
confirmed via callgrind that 8.78% of this function's total instruction
count was spent purely inside ``std::vector::size()`` on a fully
block4-resident 512x512 layer -- the compiler apparently couldn't prove
``tile_br``'s size is loop-invariant across the omp-outlined function
boundary, so it re-read ``_M_finish - _M_start`` every iteration instead
of hoisting it. A plain local variable is trivially provably invariant.

NOTE: correct and a real instruction-count win, but measured (isolated-
process methodology) as NOT moving wall-clock time noticeably --
apparently absorbed by the CPU's own execution resources (same pattern
already seen once for the vector-allocation-churn fix above). Kept
anyway: real, harmless, zero-risk, and instruction-count reductions
aren't guaranteed irrelevant on every CPU/compiler this code will ever
run on.

.. _disldo_forward.const_handle_at_index:

Per-tile handle: ``const`` and ``at_index()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.const_handle_at_index``

The handle is fetched ``const`` so ``.at()`` routes through the const
overload, which does NOT mark it dirty -- forward is read-only, so a
sparse tile's destructor should do nothing here (no wasted re-pack of
unchanged content). ``at_index()`` uses the coordinates the collection
loop already resolved, skipping ``find()``'s redundant O(row_nnz)
re-scan.

``tile.raw_data()`` is resolved once per tile instead of once per
``.at()`` call (16 calls/tile otherwise, each re-branching on whether
the tile is sparse-packed -- a property that can't change mid-tile). See
``Block4TileHandle::raw_data()``.

.. _disldo_forward.decode_bitshift_vs_table:

Decode via bit-shift, not table lookup -- and why the scalar-table
result didn't transfer to backward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.decode_bitshift_vs_table``

This column's whole 4-wide weight vector is decoded via
``block4_vec_decode_fp4`` (``fp4quant.hpp``'s bit-shift formula), not
``FP4_TABLE[code]``'s 4 separate gathers -- see ``fp4quant.hpp``'s
header comment. The remaining per-row scale multiply/clamp stays scalar
(``get_value_scale(row)`` itself isn't a SIMD operation).

Measured, not assumed: an isolated microbenchmark suggested scalar
``FP4_TABLE`` decode should win here too (same as backward's decode,
elsewhere in this file) -- but swapping it in for THIS specific loop
measurably regressed the real ``disldo_forward`` benchmark at batch=1
(~1.71x -> ~1.55x speedup vs scattered CSR at 100% density, reproduced
consistently across repeats), unlike backward where the same swap was a
real, consistent win. Reverted here; kept for backward. Compiler codegen
interactions with the surrounding code apparently differ enough between
the two functions that the isolated test's result didn't transfer --
trust the real benchmark over the isolated one. See
``TODO_DUAL_BLOCK4.md``'s Part C.

Templated ``LJ`` (compile-time constant), not a runtime loop:
``-fopt-info-vec`` confirmed GCC could not vectorize the runtime version
at all -- "loop nest containing two or more consecutive inner loops
cannot be vectorized" (the li-decode loop followed by the b-batch loop,
both nested inside the lj loop). A per-LJ templated lambda gives the
compiler 4 SEPARATE, independent instantiations instead of one loop nest
it has to reason about jointly, each with a compile-time-known column
offset, matching the pattern already used for the decode step's own
4-way unroll. See ``TODO_DUAL_BLOCK4.md``'s Part C for the measured
effect.

.. _disldo_forward.fp8_dispatch:

FP8 dispatch
~~~~~~~~~~~~

*ID:* ``disldo_forward.fp8_dispatch``

``Block4Tile8``'s layout is a full byte/slot (no nibble mask) and
decodes via E4M3 (``fp8quant.hpp``/``block4_vec_decode_fp8``), not FP4's
table-driven bit-shift codec. Originally the only thing that differed
from the (then-shared) FP4 branch, with everything past decode (row-scale
multiply, batch accumulation) identical float32 math regardless of
storage width -- **partially superseded** by
``disldo_forward.fp8_block4_avx2_column_pairing`` below: FP8 now has its
own dedicated ``process_pair8`` branch (column-paired, 8-wide) rather than
sharing FP4's per-column ``process_col``, though the decode call itself
(``block4_vec_decode_fp8``) is unchanged. FP4's branch remains the
original per-column code, untouched.

.. _disldo_forward.fp32_block4_avx2_column_pairing:

FP32 block4: AVX2 column pairing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.fp32_block4_avx2_column_pairing``

Float32 has no decode step at all (``Block4Tile32``'s weight half already
stores 4 raw, contiguous floats per column), so unlike the FP4/FP8
branches, there's nothing for a wider decode to buy here -- the real
SIMD-arithmetic opportunity is the scale-multiply and the per-batch
dot-product accumulate, both previously plain 4-wide scalar loops.

Two adjacent columns of the SAME tile (``LJ0``, ``LJ0+1``) read the exact
same 4 input rows (``br*4..br*4+4`` -- fixed per tile) and are laid out
back-to-back in ``tdata`` (``slot_index(li,lj) = lj*4+li``, so column
``LJ0``'s 4 floats end exactly where ``LJ0+1``'s begin). That means ONE
32-byte memcpy loads both columns' weights as a single 8-wide
``Block8Vec``, the row-scale gather can build one 8-wide scale vector
(zeroing the upper half when the second column is out of bounds -- true
whenever ``col0 < col1`` and only ``col0`` survives an ``n_out`` boundary,
which is the only partial case since ``col1 = col0+1``), and the per-batch
inner loop gathers the tile's 4 input values ONCE, duplicates them into
both halves of an 8-wide vector, and does one 256-bit multiply followed
by a split horizontal sum (lanes 0-3 -> column 0's output, lanes 4-7 ->
column 1's). ``BLOCK4_TILE == 4`` makes this exact -- two pairs, no
remainder -- so ``process_pair`` is called with ``LJ0 in {0, 2}``, mirroring
the existing 4-way ``process_col`` unroll it replaces for this value type
only (FP4/FP8 keep the original ``process_col`` path, untouched, in the
``else`` branch).

TDD methodology (see ``tests/unit/test_disldo_block4_fp32_wide_simd.cpp``):
wrote the test FIRST against the unmodified kernel, correctness checked
against an independent dense-matmul reference kept permanently in the
test file (not a kernel toggle -- a ``SILI_BLOCK4_FORCE_SCALAR_*``-style
compile-time toggle was explicitly rejected for this work, since it's
unnecessary code that wouldn't be reused), recorded that baseline's timing,
then modified the kernel in place and re-measured with the same test.

Measured on ``arch-sandbox`` (full-rate Zen2 AVX2, num_cpus=4,
n_in=256/n_out=256/batch=8, 4096 tiles): AVX2 codegen confirmed via
objdump (``vmulps ymm`` in the compiled block4 OpenMP-outlined function,
not just SSE). Timing -- 15 before + 15 after binary invocations, randomly
interleaved to cancel thermal/scheduling drift, each invocation's own
median-of-200-calls as one sample -- gave before median-of-medians 173990
ns/call (mean-of-medians 161149) vs after 160920 ns/call (mean-of-medians
151265): a real but modest ~6-8% speedup, well inside the ~25-30%
single-run noise band at this size. Not unexpected given how much else is
already going on in this loop (tile lookup via ``at_index()``,
``get_scale()``'s rank-N loop, thread-buffer reduction) -- the multiply-
accumulate was never the whole cost.

.. _disldo_forward.fp8_block4_avx2_column_pairing:

FP8 block4: AVX2 column pairing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.fp8_block4_avx2_column_pairing``

Unlike FP32 (no decode step at all -- see
``disldo_forward.fp32_block4_avx2_column_pairing`` above), FP8 DOES have a
real 4-wide SIMD decode (``block4_vec_decode_fp8``). Deliberately kept
UNCHANGED here rather than rewritten 8-wide: it has a hybrid design (a
whole-vector bit-shift fast path for normal-range codes, plus a per-lane
scalar correction loop for rare subnormal/NaN-slot codes), and the batch
loop that follows the decode (running ``batch`` times per tile, vs
decode's once) is the dominant per-call cost anyway -- same reasoning
FP32's widening used for its own arithmetic-only case. The two paired
columns' weight codes are decoded separately via the existing
``block4_vec_decode_fp8`` (called twice, once per column), then their
DECODED ``Block4Vec`` results are combined into one ``Block8Vec`` via
``block8_vec_from_lo_hi`` and fed through the exact same 8-wide
scale-multiply/batch-accumulate/split-hsum pattern as FP32's
``process_pair``. New branch ``process_pair8``, split out of what used to
be a shared FP8/FP4 ``process_col`` (see
``disldo_forward.fp8_dispatch`` above) -- FP4 keeps that original
per-column code unchanged in its own ``else`` branch.

TDD methodology matches FP32's exactly (see
``tests/unit/test_disldo_block4_fp8_wide_simd.cpp``): written first
against the unmodified kernel, correctness checked with tolerance against
an all-scattered layer holding the IDENTICAL already-quantized weights
(scattered's values are ``fp8_decode_bits(code)``, not the pre-
quantization float, so both arms represent the exact same quantized value
with no double-quantization mismatch -- matches
``test_fp8_block4_scattered_divergence.cpp``'s convention), baseline
timing recorded, then the kernel modified and re-measured with the same
test.

Measured on ``arch-sandbox`` (full-rate Zen2 AVX2, num_cpus=4,
n_in=256/n_out=256/batch=8, 4096 tiles): AVX2 codegen confirmed via
objdump (``vmulps ymm`` in the compiled block4 OpenMP-outlined function).
Timing -- 15 before + 15 after binary invocations, randomly interleaved --
gave before median-of-medians 176200 ns/call (mean-of-medians 168429) vs
after 166960 ns/call (mean-of-medians 156001): a real but modest
**~5.5-8% speedup**, the same magnitude as FP32's forward result (not
FP8 backward's larger one, expected below) since the decode step was
deliberately left untouched here -- the win is entirely from widening the
same batch-loop arithmetic FP32's forward widening targeted.

**Also tried and REJECTED**: a genuinely 8-wide ``block8_vec_decode_fp8``
(``Block8VecU`` + a direct 8-lane port of the bit-shift fast path and
rare-code scalar-correction loop above), gathering both paired columns'
8 weight bytes in one contiguous load (they're adjacent in ``tdata``,
same layout the column pairing already exploits) and decoding them
together instead of calling the 4-wide decode twice. MEASURED WORSE on
``arch-sandbox`` (15 before/after interleaved): median-of-medians 180730
ns/call (mean-of-medians 177699) -- slower than even the pre-widening
baseline (176200), let alone this section's twice-4-wide-decode design
(133110). Root cause, confirmed by diffing the two versions' compiled
disassembly: doubling every decode temporary (``s``/``e``/``m``/
``bits_normal``/the ``codes_arr``/``result_arr`` scratch buffers) from
128-bit to 256-bit pushed the combined YMM working set -- decode's own
temporaries plus the pairing logic's own ``s8``/``w8``/``in8``/``prod``
-- past the 16-register file, forcing spill/reload that outweighed the
saved call overhead: 55% more stack-spill instructions (280 vs 181) and
11% more total instructions (1962 vs 1768) in the compiled block4
function. Reverted; ``block8_vec_decode_fp8``/``Block8VecU`` removed from
``block4.hpp`` rather than left as unused dead code. Lesson: "pull two
blocks together" doesn't uniformly help once the combined per-op register
footprint exceeds what the target actually has -- worth checking via
disassembly, not just codegen presence, before trusting a "wider must be
faster" intuition.

.. _disldo_forward.fp4_block4_avx2_column_pairing:

FP4 block4: AVX2 column pairing, twice-4-wide decode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.fp4_block4_avx2_column_pairing``

FP4's decode (``block4_vec_decode_fp4``) is structurally different from
FP8's: it is FULLY BRANCHLESS -- pure bitwise mask-and-select via
comparison-producing vector masks, no scalar correction loop, no
``codes_arr``/``result_arr`` scratch buffers at all. This directly raised
the question ``disldo_forward.fp8_block4_avx2_column_pairing`` above left
open: does FP4 avoid the register-pressure penalty that made FP8's
genuinely-8-wide decode attempt measure worse? Two designs were built and
compared, not assumed from the structural difference alone:

1. **Twice-4-wide-decode-call** (the same "safe default" shape adopted for
   FP8/FP32): pair two adjacent columns, call the existing 4-wide
   ``block4_vec_decode_fp4`` once per column, combine the two DECODED
   ``Block4Vec`` results into one ``Block8Vec`` via ``block8_vec_from_lo_hi``.
2. **Genuinely 8-wide gather-then-decode**: a new ``block8_vec_decode_fp4``
   (direct 8-lane port of ``block4_vec_decode_fp4``'s branchless
   mask-and-select logic, using new ``Block8VecU``/``block8_vecu_broadcast``
   helpers in ``block4.hpp``) decodes both paired columns' 8 masked weight
   codes in one call, gathered into one ``Block8VecU`` literal instead of
   two separate 4-wide literals.

**RESULT: design 1 (twice-4-wide) was ADOPTED, after an initial wrong
call.** A local laptop measurement (AMD Ryzen 7 3750H, real AVX2 via
``-march=native``) first suggested design 2 was a clear ~8% win (267362 vs
291513 ns/call, 8-run median-of-medians), corroborated by disassembly
showing FP4 avoids most of FP8's register-pressure penalty when widened
this way (only 207->231 stack-spill instructions, +11.6%, going from
design 1 to design 2, vs FP8's +55% for the analogous comparison) -- so
design 2 was initially adopted on that basis.

That did not hold up on ``arch-sandbox`` (AMD Ryzen 7 3800XT, the
authoritative AVX2 machine used for every other before/after comparison in
this document). An initial n=15 interleaved round was ambiguous (design 1
median 154520 vs design 2's 156430), so a larger n=35 interleaved round
was run: design 1 median 154600 ns/call (mean 143139) vs design 2 median
151200 (mean 144067) -- **statistically indistinguishable**. Individual
runs for BOTH designs swung between ~124000 and ~178000 ns/call (a
bimodal, likely frequency-scaling-driven pattern affecting both binaries
equally), a spread far larger than the ~2% gap between their medians. The
local laptop's clear-looking win simply did not replicate on the machine
that matters.

Given a genuine tie on ``arch-sandbox``, design 1 (twice-4-wide) was kept:
it reuses the already-proven 4-wide ``block4_vec_decode_fp4`` rather than
maintaining a second, wider decode function purely for a difference that
turned out to be noise. ``block8_vec_decode_fp4``/``Block8VecU``/
``block8_vecu_broadcast`` were removed from ``block4.hpp`` rather than left
as unused dead code (same convention as FP8's rejected
``block8_vec_decode_fp8``).

**Lesson** (the actually load-bearing one here, worth remembering for
future AVX2-widening work on this codebase): a clear-looking local result,
even one that disassembly seems to corroborate, is not a substitute for
measuring on the authoritative machine -- the laptop and arch-sandbox
disagreed not just in magnitude but in DIRECTION. Every prior AVX2-widening
decision in this document (FP32 forward/backward, FP8 forward/backward)
was measured on ``arch-sandbox`` from the start; this is the one case that
wasn't, and it produced a wrong initial conclusion as a direct result. Do
not adopt a kernel change based on local-only timing again, regardless of
how large or well-explained the local margin looks.

.. _disldo_forward.aqrs_additive_branch:

AQRS additive branch
~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.aqrs_additive_branch``

(task #276, gamma wired in at task #289, see
``sili_peridot/AQRS_DESIGN.md``) ``A[row,col] = sum_k gamma_k *
additive_u_k(row,k) * additive_v_k(col,k)``, ADDED (not Hadamard-
multiplied against quant like the scale_rank branch above) to the
effective weight. Genuinely independent of the sparse/block4 structure
above -- it's a dense low-rank correction that touches every output
regardless of which synapses happen to be live, so it's computed as its
own small pass rather than woven into either per-synapse loop.

Fused per Theorem 11 (never materializes the ``n_in x n_out`` A matrix):
project the input down to ``additive_rank`` dimensions via
``additive_u``, then back up to ``n_out`` via ``additive_v`` --
``O(batch * additive_rank * (n_in + n_out))``, cheap as long as
``additive_rank << min(n_in, n_out)``.

No-op (skipped entirely) at the default ``additive_rank==0``, matching
``value_scale``/``output_scale``'s own "unconfigured component
contributes nothing" convention. ``gamma_k`` itself defaults to 1.0 (see
``get_additive_gamma_k``'s own docstring) so this multiply is
transparent for every caller that's never touched gamma.

.. _disldo_backward:

``disldo_backward``
--------------------

*ID:* ``disldo_backward``

Weight/importance update is parallelised over ROWS, not synapses, since
``DeltaCSRRowCursor`` decodes sequentially within a row -- each row is an
independent, unique ``elem_start`` range, so no races.

.. _disldo_backward.rmsprop_floor_rationale:

Why RMSprop-with-a-floor, not plain RMSprop or an lr-decay schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.rmsprop_floor_rationale``

Root-caused via direct trace on ``DISLDOLayer32``: plain RMSprop
diverges late in training once ``ci`` decays down to match a shrinking
residual gradient. An lr-decay schedule "fixes" it by eventually
freezing the whole network, which is incompatible with this project's
lifelong-learning goal -- hence a stationary per-synapse floor+clip
instead (``BoundedRMSpropSynapsePolicy``, the default; see its own
docstring in ``delta_csr_types.hpp`` for the tuned production defaults).
``PlainRMSpropSynapsePolicy`` remains available as explicit opt-in and
as the bit-identical reference the fix was checked against --
``min_decay_frac``/``max_abs_delta``/``max_ci`` are inert under Plain.

``SynapsePolicyT`` is a template-TEMPLATE parameter (not a fully
instantiated ``typename SynapsePolicy = Policy<value_type>`` the way
``ScalePolicy`` is), because this function's SIMD sites need the SAME
chosen policy instantiated at both ``value_type`` and ``Block4Vec`` --
one caller-visible choice, two internal instantiations
(``SynapsePolicy``/``SynapsePolicyVec``). It's placed LAST in the
template parameter list, not next to ``ScalePolicy``: existing callers
across the test suite specify template args positionally through
``DeferredScaleWrite``/``StochasticRounding``, and inserting a new
parameter earlier silently shifted every positional arg onto the wrong
parameter (confirmed directly -- broke the build with "expected a class
template, got 'false'").

``l1_coef`` (AQRS gamma's L1 penalty coefficient, Theorem 8) is appended
LAST, after ``scale_invariant``, for the same positional-argument
reason. Default 0 disables it, matching every existing call site
unchanged.

.. _disldo_backward.setup_and_presizing_bugs:

Setup: deferred-write buffering, rank-N pre-sizing, two real bugs fixed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.setup_and_presizing_bugs``

``DeferredScaleWriteEntry`` buffers a touched scattered entry's
true-units ``(cw, ci)`` instead of storing immediately, written out via
``set_stochastic`` only once ``value_scale[row]`` AND ``output_scale[col]``
are both finalized for the whole call (block4 is untouched by
``DeferredScaleWrite``). Stats use the pre-store true-units ``ci`` as a
stand-in for the post-quantization readback since the real stored code
doesn't exist yet when the write is deferred -- a documented
approximation, harmless since these stats are purely observational.

There used to be an early return when both storages (scattered + block4)
were empty. Removed: the dead-row ``value_scale`` bootstrap pass (see
below) needs to run precisely in that case -- a genuinely fresh
zero-synapse-init layer. Everything else is already independently
guarded (``if (!dc.empty())``, ``if (n_tiles() > 0)``) and degrades to
safe no-ops when both are empty, so removing the early return only costs
a little harmless extra work, not correctness.

``value_scale``/``output_scale`` are now ``n_in*rank``/``n_out*rank``
(see ``scale_rank``'s own docstring) and get pre-sized before the
parallel region so direct indexed writes inside it are safe (a
per-thread ``resize`` would race).

**Real bug #1** (found via ``test_aqrs_rank_growth_shrink.cpp``): a
uniform ``resize(..., value_type(1))`` fill backfills EVERY newly
appended slot with 1.0, not just the ``k==0`` ones -- but the documented
default is ``k==0 -> 1.0, k>=1 -> 0.0`` (an untrained extra rank
component must be a pure no-op). Confirmed via direct probe: growing
``scale_rank`` from 1 to 2 mid-training made the new ``k=1`` channel
start IDENTICAL to ``k=0``, so it received identical gradients every
step by symmetry and stayed in permanent lockstep -- effectively a
scaled rank-1, not real rank-2 capacity. Fix: resize with a neutral 0
fill, then explicitly set only the ``k==0`` slots in the newly added
range to 1.0 (matches ``reshuffle_rank_array``'s own ``scale_default``
lambda in ``set_scale_rank``, ``delta_csr_types.hpp``).

.. code-block:: cpp

   // as of PR #45, linear_disldo.hpp:
   if (weights.value_scale.size() < n_in * rank) {
       const std::size_t old_size = weights.value_scale.size();
       weights.value_scale.resize(n_in * rank, value_type(0));
       for (std::size_t idx = old_size; idx < weights.value_scale.size(); ++idx)
           if (idx % rank == 0)
               weights.value_scale[idx] = value_type(1);
   }
   // WRONG (the bug): weights.value_scale.resize(n_in * rank, value_type(1));

**Real bug #2** (found via AddressSanitizer): ``value_scale_step`` was
missing from the pre-sizing list. ``get_value_scale_step_k``'s own lazy
``.resize()`` runs completely unguarded, called directly from every row
inside the ``#pragma omp parallel`` region -- two threads touching a
not-yet-grown index at the same time race on the same vector's
resize/reallocation. Confirmed as a genuine heap-use-after-free (ASan:
one thread reading ``value_scale_step[idx]`` while another thread's
``resize()`` had already freed the old buffer). Pre-sizing here (same
shape/reason as ``value_scale_importance``) makes the lazy-resize branch
dead in the normal case.

.. _disldo_backward.gamma_fetched_once:

AQRS gamma fetched once per call, not baked into the scale caches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.gamma_fetched_once``

``gamma_k`` (task #273/#283) is layer-wide -- doesn't vary by row/col/
tile -- so it's fetched once here rather than per-row like
``value_scale_k``/``output_scale_k``, and shared by the scattered loop
and every block4 sub-path. Deliberately NOT baked into the direction
caches: gamma's own gradient needs the PURE (un-multiplied)
``value_direction_k * output_direction_k`` product, and baking gamma
into either side would make recovering that require dividing by
``gamma_k`` -- fragile exactly at ``gamma_k=0`` (the common case, since
gamma defaults to 0 for any channel beyond the always-on ``k=0``).
Applied as an explicit extra factor at each gradient accumulation site
instead.

.. _disldo_backward.scattered_lr_and_signal:

Scattered path: per-row learning rate and the additive importance signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.scattered_lr_and_signal``

``lr_per_row_nnz``: a row with more synapses gets more simultaneous
per-synapse nudges each backward pass, so the aggregate shift in that
row's behavior scales roughly with ``nnz_this_row`` for a fixed
``learning_rate`` -- dividing by ``nnz_this_row`` keeps the aggregate
update comparable across rows regardless of connection count (matters
because synaptogenesis makes ``nnz_this_row`` genuinely vary within one
layer). The layer-wide equivalent needs no kernel support -- a caller
can pre-divide ``learning_rate`` by the layer's total nnz themselves,
since that quantity doesn't vary within a call.

``value_scale``'s own gradient ALWAYS divides by ``nnz_this_row``,
independent of ``lr_per_row_nnz``: its accumulator sums
``nnz_this_row*batch`` contributions, so dividing by ``nnz_this_row``
normalizes back to the average, matching the semantics of a gradient on
a single scalar parameter (not a vector of n weights). The gradient sums
first across ALL (synapse, batch) pairs for a row, THEN applies lr once
-- applying lr per-contribution inside the innermost loop risks each
increment falling below ``ULP(value_scale)`` in float32 and
disappearing.

**Real bug, fixed this session**: ``cw``/``ci`` used to mutate INSIDE
the batch loop, once per batch row, so row ``b``'s own contrib/dx used
row ``b-1``'s already-applied update instead of a consistent parameter
snapshot -- ``b`` sequential mini-steps instead of one real step, and
even ``dx`` (the gradient flowing to the previous layer) was computed
against a moving target. Fixed: ``cw_start``/``quant_start`` (and the
block4 SIMD equivalents) are now FIXED snapshots for the whole batch
loop, with ``g_agg``/``contrib_agg`` scalar accumulators summed across
the batch (mirroring the torch reference's own ``x_sum = x.sum(dim=0)``
aggregation by linearity), and exactly ONE optimizer step applied after
-- the standard mini-batch-optimizer contract. This pattern repeats at
every one of this function's per-synapse update sites (scattered,
block4 FP4 SIMD/scalar, block4 FP8 SIMD/scalar) -- each has its own
``*_start`` snapshot for the same reason.

Additive combination of the backward sensitivity signal (``g = dy*x``)
with the forward contribution signal (``contrib = x*w``) is proved to
strictly improve the importance estimate whenever the two are not
conditionally independent given it -- see
``lean_proofs/importance_signal_information_gain/SiliImportanceProof/
ImportanceSignalInformationGain.lean``, theorem
``Joint.combined_signal_strictly_informative`` (built on
``Joint.entropy_le_condEntropy``, H(Θ|X,Y) ≤ H(Θ|X)). Additive, not
multiplicative, so ``ci`` still updates from ``contrib`` alone even when
``g=0``, instead of reintroducing the old quant=0-forever deadlock a
multiplicative gate would cause. This combination appears at every
importance-accumulation site in this function (per-synapse ``ci``,
``value_scale``/``output_scale``'s own importance, gamma's own
importance) -- each mirrors the same rationale rather than repeating it.

**Square first, THEN sum** (``g^2+contrib^2``, not ``(g+contrib)^2``):
``ci`` is the divisor of the weight update's step size, so its job is
safety -- never let the denominator collapse toward zero while the
numerator stays large. Sum-then-square fails exactly that: when ``g``
and ``contrib`` are large and near-opposite in sign (a real, common case
-- a synapse whose current value and the task's error signal are
pulling against each other), ``(g+contrib)^2`` can collapse toward zero
even though both signals are individually large, making the step
explode -- the same class of instability the value_scale bias-correction
fix elsewhere in this file closes, just triggered by cancellation
instead of cold start. Square-then-sum is bounded below by
``max(g,contrib)^2`` regardless of sign, so a large-magnitude
disagreement damps the step instead of amplifying it. This still
captures the additive-signal information-gain property the Lean proof
establishes -- that proof is about the two signals being jointly more
informative than either alone in the abstract, it does not prescribe
sum-before-square as the numeric encoding.

.. _disldo_backward.deferred_vs_direct_quant:

DeferredScaleWrite branch vs. the direct-quant-update formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.deferred_vs_direct_quant``

Two branches exist for the per-synapse update, selected by
``if constexpr (DeferredScaleWrite)``:

- **DeferredScaleWrite**: the old true-units round-trip formula
  (unchanged) -- ``value_scale``/``output_scale`` finalize AFTER this
  loop, so eagerly multiplying the code's own step by a not-yet-finalized
  combined scale would reintroduce exactly the staleness
  ``DeferredScaleWrite`` exists to avoid. This branch does NOT yet get
  deterministic-rounding zero-escape; only the non-deferred branch does.

- **Non-deferred (the "NEW formula")**: ``quant`` (the stored CODE) is
  the primary optimized quantity, updated DIRECTLY via
  ``dL/d(quant) = g * S(row,col)`` -- proper chain rule on
  ``true_w = quant * S`` -- instead of the old true-units round-trip
  (``cw += ...; new_code = cw / combined_scale``), which DIVIDED by the
  scale, backwards: a LARGER scale SHRUNK the per-call code step instead
  of growing it. This version is unconditionally nonzero given
  ``S != 0``, so ``quant`` can escape exactly 0 without needing
  stochastic rounding, no matter how small ``learning_rate`` is, given
  enough persistent-signal steps.

**Real bug, corrected**: the zero-escape floor on ``value_scale``/
``output_scale``'s own gradient (``quant_floor``) must be GATED on
``quant==0``, not applied unconditionally as
``zero_escape_eps + |quant|``. The unconditional version silently
discarded ``quant``'s SIGN on every synapse, not just stuck-at-zero
ones -- for any nonzero ``quant`` (the overwhelming majority once
training gets going), it fed an always-positive floor into a formula
whose correct gradient is signed. Not a small epsilon bias: wholesale
directional corruption of every trained synapse's contribution to
``value_scale``/``output_scale``'s gradient. Confirmed as the root cause
of a real, reproducible failure: zero-init models produced predictions
and ``eval_acc`` BIT-IDENTICAL to a completely untrained model after
15000 real training steps (energy and rank-N included -- see
``sili_peridot/scripts/zeroinit_minimal_repro.py``). Only ``quant==0``
(the genuinely stuck case, where the correct gradient truly is zero and
sign is legitimately free) substitutes the small positive epsilon; every
other value uses itself directly, exact and signed. ``quant_floor`` is
computed once, fixed, from the pre-call ``cw_orig``/``quant4`` snapshot
-- same batch-aggregation-bug fix as ``cw_start`` above.

.. code-block:: cpp

   // as of PR #45, linear_disldo.hpp -- scattered non-deferred branch:
   const value_type quant_floor =
       (cw_orig == value_type(0)) ? zero_escape_eps : cw_orig;
   // WRONG (the bug): value_type quant_floor = zero_escape_eps + std::abs(cw_orig);

AQRS gamma's own per-component gradient factor (dS/d(value_direction_k)
``= gamma_k * output_direction_k``, symmetrically for output_direction_k)
is applied explicitly at each rank component's accumulation site, since
``value_scale_k``/``output_scale_k`` are pure DIRECTION and never
gamma-baked (only ``get_scale()`` combines direction*gamma for the
weight-update math). gamma's OWN gradient
(``dS/d(gamma_k) = value_direction_k * output_direction_k``) is a
separate, layer-wide (not per-row/col) accumulator.

.. _disldo_backward.block4_overview_races:

block4 backward: overview, races, and a documented simplification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.block4_overview_races``

Mirrors the scattered loop above but keyed by tile instead of CSR row,
sharing the same ``value_scale``/``output_scale`` (moving a synapse
between representations stays a lossless byte copy, so its gradient math
must stay consistent too).

Race note: the scattered loop parallelizes over ROWS, so each row's
``value_scale`` update is owned by exactly one thread. Here we
parallelize over TILES, and two different tiles can share the same
block-ROW (different block-columns) -- so a row's ``value_scale``
gradient can now be touched by more than one thread concurrently. Fixed
the same way ``output_scale``'s gradient already is: per-thread-private
accumulator buffers (``t_row_grad``, indexed like ``t_col_grad``),
reduced serially once after the parallel region, instead of applying the
update inline.

**Known, documented simplification (not a bug)**: if a row has BOTH
scattered and block4 synapses, its ``value_scale`` gets two sequential
gradient steps (the scattered loop's step, then block4's step) rather
than one combined step over the true total nnz. Mathematically this is
just two successive descent steps, not an incorrect one -- acceptable
for a first working version.

.. _disldo_backward.row_ti_start_workspace:

``row_ti_start``/``row_live_count`` and the row-workspace rewrite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.row_ti_start_workspace``

``row_ti_start`` is a cumulative tile count per block-row, needed to
split the row-partitioned parallel loop (each thread's ``#pragma omp
for`` chunk is a contiguous block-row range, but tile counts per row
vary). It is explicitly NOT storage offsets -- since the row-workspace
rewrite, tile byte/elem positions live entirely within each row's own
``RowWorkspace``, snapshotted fresh per row, not precomputed globally (a
global precompute was the root of the byte_pos-staleness class of bugs
this rewrite closes -- see ``Block4Store::RowWorkspace``'s own comment).

``row_live_count`` is the per-row slot count across ALL block4 tiles
touching that row, needed for both ``lr_per_row_nnz`` and the
unconditional ``scale_eff_lr`` normalization -- every tile contributes
exactly ``BLOCK4_TILE`` slots per row it covers (dense; a live tile's
slots are all real synapses, weight=0.0 included).

.. _disldo_backward.schedule_static_measured:

``schedule(static)`` measured to beat dynamic/guided
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.schedule_static_measured``

Row widths can vary a lot, which in principle makes ``static``'s flat
contiguous split load-balance worse than ``schedule(dynamic)``/
``schedule(guided)`` once ``num_cpus > 1``. Tried both, measured worse
in practice: real per-chunk dispatch overhead showed up as a real
backward slowdown at high tile density even at ``num_cpus=1``
(``scripts/bench_block4_vs_dense_fp4.cpp``: dynamic/guided both measured
~1.69x speedup over the dense floor at 100% fill, vs ~1.97x for
static -- worse than even the pre-fix documented baseline of ~1.88x),
where there's no actual load-balancing benefit to buy that overhead
with. Partitioning is also BY BLOCK-ROW, not flat tile index: each
thread exclusively owns every tile in the rows it's assigned, so no two
threads ever touch the same row's ``tile_data`` concurrently -- this is
what makes it safe for a tile's handle to resize (real sparse<->dense
transitions) inside the parallel region at all.

.. _disldo_backward.row_workspace_snapshot_fix:

Row-workspace snapshot: fixing a cross-row memmove hazard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.row_workspace_snapshot_fix``

**Real bug, fixed**: the old version of this loop read and wrote each
tile directly through the shared store (via ``at_index()``'s fast path),
relying only on row-exclusive thread ownership for safety. That protects
against two threads touching the SAME row, but NOT against a DIFFERENT
row's growth: growing row X past its own current headroom shifts
``tbyte_start``/``tbyte_end`` -- and physically MEMMOVES the tile bytes
-- for every row after X, including whatever row this thread owns.
Confirmed via ASan as a real, reproducible heap-use-after-free /
negative-size memmove, even with ``tile_data``'s capacity pre-reserved
(only fixes buffer reallocation, a separate hazard) and even with a lock
guarding concurrent growers (only fixes two growers racing each other,
not a grower racing this row's reader). ``disldo_backward_sparse_grad``
in ``sisldo_ops.hpp`` has the identical fix.

Fix: snapshot this row into a thread-private workspace ONCE, do all
reads/writes against that private copy, then merge back in one
row-exclusive step at the end (evicting lowest-importance synapses only
if the row genuinely grew past its own existing headroom -- see
``Block4Store::merge_row_workspace``'s comment). This makes the
cross-row-shift hazard structurally unreachable rather than merely less
likely. Used UNCONDITIONALLY, even when ``learning_rate == 0`` (which
never writes anything back), rather than branching read-only vs writing:
``dx`` doesn't depend on whether writes happen, and this keeps one
tested code path instead of two. The read-only (``learning_rate == 0``)
case skips the snapshot+merge-back entirely instead -- process_tile
structurally never writes in that case (every write inside it is gated
by the same condition), so the plain shared-store ``at_index()`` path
has no concurrency hazard, and copying the row's bytes in and back out
unchanged would be measured real overhead for zero benefit.

.. _disldo_backward.fp8_simd_measured:

FP8 block4 backward: measured SIMD results (batch=1 loses, batch=32 wins)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.fp8_simd_measured``

Measured (``scripts/bench_block4_fp8_simd.cpp``, 512x512 100%-dense
block4, best-of-200, ``-O3 -ffast-math -march=native``), not assumed:

- batch=1: full-SIMD 0.0060s vs plain-scalar 0.0048s -- SIMD LOST
  (~20% slower).
- batch=1: scalar decode+encode + SIMD accumulate-only ~= plain-scalar
  (no measurable win or loss -- batch=1 means the SIMD accumulate loop's
  own inner ``for (b<batch)`` runs once, never amortizing its setup cost
  against reused decoded state).
- batch=32: scalar decode+encode + SIMD accumulate 0.0227s vs
  plain-scalar 0.0298s -- SIMD WON (~24% faster), confirmed via objdump
  that this is real 128-bit packed SIMD (vmulps/vrsqrtps/vaddps on xmm
  registers), not GCC auto-vectorizing scalar code.

Conclusion: ``block4_vec_decode_fp8``/``block4_vec_quantize_stochastic_fp8``
(the SIMD decode/encode built alongside ``disldo_forward``'s own block4
section) measurably LOSE here -- E4M3's 256-code space makes their
subnormal/NaN-lane scalar-correction fallback (``block4.hpp``) real,
non-negligible overhead that FP4's simpler E2M1 (16 codes) never pays.
So: scalar ``fp8_decode_bits``/``fp8_quantize_stochastic`` for
decode/encode (matching FP4's own empirical finding for backward, now
independently confirmed for FP8 too), SIMD (``Block4Vec``) kept ONLY for
the batch-loop accumulation math -- the one piece that measurably earns
its complexity at realistic (>1) batch sizes.

FP8's ``cw4_8`` local is kept in CODE-SPACE (matching FP4's ``quant4``
convention), not true-weight units. Real bug caught before landing:
keeping it in true units while ``update_cw``'s RMSprop-normalized delta
is ~S-independent (magnitude ~``eff_lr`` regardless of S) meant the
later ``/ combined_scale4_8`` at write time amplified every step by
``1/S`` instead of damping it by S the way FP4's convention does,
blowing up explosively for any layer with small ``output_scale`` (e.g.
a wide dense layer's fan-in-corrected scale).

.. _disldo_backward.fp4_table_decode:

FP4 block4 backward: FP4_TABLE beats the bit-shift decode (opposite of forward)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.fp4_table_decode``

NOT a "SIMD loses to scalar" finding -- that was checked properly and is
FALSE: ``fp4_decode_bits()`` (the true scalar equivalent of the SIMD
bit-shift formula) is measurably the WORST of the three options here
(~1.6x slower than either alternative in the full backward benchmark),
confirming SIMD genuinely beats scalar bit-shift decode, consistent with
this codebase's earlier documented finding (``TODO_DUAL_BLOCK4.md``) and
not contradicted by anything here. What DOES win, measured in the real
``disldo_backward`` benchmark (not just an isolated microbenchmark,
which misleadingly suggested a bigger and differently-shaped effect --
see ``TODO_DUAL_BLOCK4.md``'s Part C) is ``FP4_TABLE[code]``
specifically -- a different decode ALGORITHM (branchless array lookup vs
bit-field reconstruction), not a SIMD-vs-scalar swap. Real, reproducible
~6% win over the SIMD bit-shift version for backward specifically (3
repeats each, clean non-overlapping ranges); forward showed the OPPOSITE
(SIMD bit-shift wins there, kept as-is in ``disldo_forward``) -- the two
functions' surrounding code apparently interacts with this choice
differently enough that the same swap doesn't transfer between them.
GCC's own SLP vectorizer proof for the batch/lj loop (``-fopt-info-vec``)
rejects a gather-looking ``output_grad[...col4[lj]]`` index as
"unprofitable" but accepts a plain affine ``output_grad[...col_base+lj]``
index as contiguous -- hence the ``full_tile_cols`` split (whole
tile-column in bounds) checked once per tile, not per batch element.

.. _disldo_backward.fp32_block4_avx2_row_pairing:

FP32 block4 backward: AVX2 row pairing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.fp32_block4_avx2_row_pairing``

Backward's counterpart to
``disldo_forward.fp32_block4_avx2_column_pairing`` (above), on the OTHER axis.
Forward's existing SIMD width already spans a fixed column's 4 rows, so it
paired two columns; backward's existing SIMD width spans a fixed row's 4
columns, so this pairs two adjacent rows (``li``, ``li+1``) of the same
tile into one 8-wide op, called via a new ``process_row_pair_fp32``
lambda inside ``process_tile``. Scoped to the common case only -- both
rows must be live (``row < n_in``, ``row_live_count[row] > 0``) and the
tile's column range must be fully in-bounds (mirrors ``full_tile_cols32``)
-- a partial-column tile at the ``n_out`` edge or a dead partner row falls
back unchanged to the existing single-row path. The outer ``for (li...)``
loop detects a pairable pair, calls the new lambda, then does an extra
``++li`` before its own increment/``continue`` to skip the consumed
``li+1`` -- net advance of 2, not 1.

Per-quantity handling, since backward's math mixes quantities that depend
on the row, the column, or both:

- ``dy``/``output_scale_k`` depend only on the column -- shared/duplicated
  across both paired rows' halves of the 8-wide vector
  (``block8_vec_dup4``), not recomputed per row.
- The gathered input value, ``value_scale_k``, and ``effective_lr`` depend
  only on the row -- broadcast per-half (``block8_vec_broadcast_pair``),
  differing between the two rows.
- The decoded weight/importance and combined scale genuinely depend on
  BOTH row and column -- 8 distinct values, no sharing possible. No single
  contiguous memcpy across rows either (``Block4Tile32::slot_index(li,lj)
  = lj*4+li`` means fixed-li/varying-lj is already a stride-4 gather, same
  total loads as two single-row passes) -- the win is the 8-wide
  ARITHMETIC that follows the loads, not the loads themselves.
- ``mrow``/``mdx`` are per-ROW reductions -- split via ``block8_vec_lo4``/
  ``block8_vec_hi4`` (lower 4 lanes -> row0, upper 4 -> row1).
- ``mcol`` is a per-COLUMN reduction, and a column's identity doesn't
  depend on which of the two paired rows contributed -- folded via
  ``block8_vec_fold4`` (elementwise lower+upper) into the SAME 4-wide
  scratch a single-row pass would have used, no new scratch shape needed.
- ``mgamma`` is a single layer-wide scalar accumulator -- summed across
  ALL 8 lanes at once (``block8_vec_hsum``), combining both rows'
  contributions directly.
- The RMSprop-style ``ci``/``cw`` update is genuinely per-(row,col) cell
  (no shared/duplicated lanes to exploit), and no ``Block8Vec``
  specialization of ``SynapsePolicy`` exists (``PlainRMSpropSynapsePolicy``/
  ``BoundedRMSpropSynapsePolicy`` are hand-specialized for ``Block4Vec``
  specifically, using per-lane-loop helpers like ``block4_vec_sqrt`` that
  don't generalize for free). Rather than write a new 8-wide specialization,
  the accumulated 8-wide state is split back into two ``Block4Vec`` halves
  and the EXISTING ``SynapsePolicyVec::update_ci``/``update_cw`` calls run
  twice. This is a once-per-row-pair cost, not a per-batch one, so it
  doesn't undo the batch loop's win.

New ``Block8Vec`` helpers added to ``block4.hpp`` for this:
``block8_vec_load``/``store``/``broadcast``, ``block8_vec_broadcast_pair``
(different scalar per half), ``block8_vec_dup4`` (same 4-wide vector in
both halves), ``block8_vec_lo4``/``hi4`` (extract a half),
``block8_vec_from_lo_hi`` (recombine), ``block8_vec_fold4`` (elementwise
lower+upper sum), ``block8_vec_hsum`` (all-8 sum) -- all memcpy/literal-
based, mirroring ``Block4Vec``'s own helpers' style exactly.

TDD methodology (see
``tests/unit/test_disldo_block4_fp32_backward_wide_simd.cpp``): written
FIRST against the unmodified kernel, correctness checked (with tolerance,
not bit-exact -- unlike the tiny hand-placed-tile test elsewhere, this
one's 32-row-dense scale means block4's parallel-thread reduction and
scattered's sequential walk legitimately sum in different orders) against
an all-scattered cross-check plus a post-backward forward probe, baseline
timing recorded, then the kernel modified and re-measured with the same
test -- no compile-time toggle added (same "don't add unnecessary code
that would not be used again" rejection as forward's widening).

.. _disldo_backward.fp8_block4_avx2_row_pairing:

FP8 block4 backward: AVX2 row pairing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.fp8_block4_avx2_row_pairing``

FP8's existing block4 backward SIMD structure is the exact same shape as
FP32's (see ``disldo_backward.fp32_block4_avx2_row_pairing`` above), so
this ports the same row-pairing design (``process_row_pair_fp8``), with
three real differences from the FP32 port: (1) decode is already SCALAR
in the existing single-row code (``fp8_decode_bits``, not
``block4_vec_decode_fp8``) -- measured faster for backward specifically
(see ``disldo_backward.fp8_simd_measured`` above) -- kept scalar, just 8
calls instead of 4, no widening attempted (unlike forward's rejected
8-wide-decode experiment, this path never used a SIMD decode to begin
with, so there was nothing to "un-widen"); (2) FP8's ``cw``/``ci`` are
quantized CODE-SPACE values needing ``was_live``-gated encode-on-write
(FP32 has no quantization floor to escape, so no such gating exists
there); (3) encode is the existing ``StochasticRounding`` x ``was_live``
4-way scalar dispatch (``fp8_quantize_live``/``_stochastic_live``/
``_stochastic``/plain), called once per (row,col) cell -- 8 cells now
instead of 4, no SIMD encode to widen either.

TDD methodology matches FP32's exactly (see
``tests/unit/test_disldo_block4_fp8_backward_wide_simd.cpp``): written
FIRST against the unmodified kernel, correctness checked with tolerance
against an all-scattered layer holding the identical already-quantized
weights (same ``fp8_decode_bits(code)`` convention as the forward test)
plus a post-backward forward probe, baseline timing recorded, then the
kernel modified and re-measured with the same test.

Measured on ``arch-sandbox`` (full-rate Zen2 AVX2, num_cpus=4,
n_in=256/n_out=256/batch=8, 4096 tiles): AVX2 codegen confirmed via
objdump (``vaddps ymm`` in the compiled block4 OpenMP-outlined function).
Timing -- 12 before + 12 after binary invocations, randomly interleaved,
each invocation's own median-of-200-calls as one sample -- gave before
median-of-medians 9673260 ns/call (mean-of-medians 11653644) vs after
3643525 ns/call (mean-of-medians 4001787): a real **~2.65-2.9x speedup**,
matching FP32 backward's ~2.7-3.2x (same underlying reason: real per-cell
RMSprop update plus rank-N AQRS bookkeeping to widen, unlike forward's
modest ~5.5-8% where decode dominates and was deliberately left
unwidened -- and where an attempt to widen it anyway was tried and
measured worse, see ``disldo_forward.fp8_block4_avx2_column_pairing``
above).

Measured on ``arch-sandbox`` (full-rate Zen2 AVX2, num_cpus=4,
n_in=256/n_out=256/batch=8, 4096 tiles): AVX2 codegen confirmed via
objdump (``vmulps ymm`` in the compiled block4 OpenMP-outlined function).
Timing -- 15 before + 15 after binary invocations, randomly interleaved to
cancel thermal/scheduling drift, each invocation's own median-of-200-calls
as one sample -- gave before median-of-medians 10967140 ns/call
(mean-of-medians 13120815) vs after 3400970 ns/call (mean-of-medians
4785922): a real **~2.7-3.2x speedup**, much larger than forward's modest
~6-8%. Consistent with expectations going in: forward's FP32 "decode" was
a no-op memcpy with almost nothing to widen, while backward's per-cell
RMSprop update plus rank-N AQRS bookkeeping (mrow/mcol/mgamma, gamma
weighting, contrib terms) is exactly the kind of "lot more stuff going on
in the SIMD part" that a wider op actually pays off on.

.. _disldo_backward.fp4_block4_avx2_row_pairing:

FP4 block4 backward: AVX2 row pairing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.fp4_block4_avx2_row_pairing``

FP4's existing block4 backward decodes via scalar ``FP4_TABLE`` lookups
(``disldo_backward.fp4_table_decode`` above), not SIMD bit-shift, same
deliberate choice FP8's backward makes for its own decode -- so this ports
the same row-pairing design (``process_row_pair_fp4``) as
``disldo_backward.fp8_block4_avx2_row_pairing`` above, with two real
differences specific to FP4BiPacked: (1) weight+importance are packed as
TWO NIBBLES in ONE byte per slot (not FP8's two separate byte arrays), so
decode is one byte read + two ``FP4_TABLE`` lookups per (li,lj), and
write-back is a single bit-OR pack
(``uint8_t((new_imp<<4)|new_w)``) instead of two separate ``tdata``
writes; (2) FP4's gradient-accumulation terms (mrow/mcol/mgamma) use a
"quant_floor" -- the stored quantized weight with a ``zero_escape_eps``
substitution at exactly ``quant==0``, giving a currently-zero-weight cell
a real synaptogenesis growth signal (see
``disldo_backward.deferred_vs_direct_quant`` above) -- where FP8 has no
such floor and uses ``cw_orig`` directly. Both differences are read
straight off FP4's existing single-row code, not new design; only the
4-wide-vs-8-wide packing itself is new.

TDD methodology matches FP8's exactly (see
``tests/unit/test_disldo_block4_fp4_backward_wide_simd.cpp``): written
FIRST against the unmodified kernel, correctness checked with tolerance
against an all-scattered layer holding the identical already-quantized
weights plus a post-backward forward probe, baseline timing recorded, then
the kernel modified and re-measured with the same test. Correctness
additionally verified against the broader existing regression suite
(``test_block4_scattered_divergence``, ``test_aqrs_gamma``,
``test_aqrs_additive_branch``, ``test_aqrs_rank_growth_shrink``,
``test_block4_memory_cap_and_compression``, ``test_ci_ceiling``'s
30000-step long run, ``sweep_synapse_policy_stochastic`` for
``StochasticRounding=true`` coverage) -- all passed unmodified, since
``process_row_pair_fp4`` reuses the exact same ``SynapsePolicyVec``/AQRS
rank-N/gamma machinery as the single-row path, just at 8-wide.

Measured on ``arch-sandbox`` (full-rate Zen2 AVX2, num_cpus=4,
n_in=256/n_out=256/batch=8, 4096 tiles): timing -- 15 before + 15 after
binary invocations, randomly interleaved, each invocation's own
median-of-200-calls as one sample -- gave before median-of-medians
9311840 ns/call (mean-of-medians 11626223) vs after 3545899 ns/call
(mean-of-medians 5461988): a real **~2.63x speedup**, matching FP8
backward's ~2.65-2.9x almost exactly (same underlying reason: real
per-cell RMSprop update plus rank-N AQRS bookkeeping to widen, unlike
forward's much smaller and ultimately noise-confounded gain -- see
``disldo_forward.fp4_block4_avx2_column_pairing`` above for that
contrasting result). Both arms showed real bimodal run-to-run variance (a
roughly 2x high/low split within each arm, most likely thermal/
frequency-scaling, affecting before and after similarly) -- the
median-of-15 is what actually separates the real signal from that noise;
a single non-interleaved before/after pair measured only ~14% at one
point during this work, purely from landing in each arm's opposite mode,
before the full interleaved run resolved it. Directly following
``disldo_forward.fp4_block4_avx2_column_pairing``'s own lesson, this
result WAS verified on ``arch-sandbox`` before being adopted (a local
laptop sanity check also showed a consistent, same-direction ~3.36x, but
was not treated as the real number).

.. _disldo_backward.was_live_gating:

``was_live`` gating: a real bug from block4's dense-tile semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.was_live_gating``

block4 tiles are DENSE 4x4 blocks touched at every in-bounds ``(li,lj)``,
including cells that were never a real synapse -- block4's own "no
synapse here" IS ``weight==importance==0``, unlike scattered CSR's
structural absence (matches ``block4_count_live``/``block4_sparse_pack``'s
identical ``dense[i]!=0`` criterion). The never-zero live quantizer must
ONLY protect an ALREADY-established synapse from rounding back to the
dead code; applying it unconditionally to every valid-column cell would
force every touched-but-never-connected cell permanently "live",
corrupting block4's own sparse/dense repacking. Confirmed directly: this
caused a whole tile in ``test_disldo_block4_backward`` to zero out via a
corrupted live-count/repack. Fix: ``was_live[lj]`` (true only if the cell
held a genuine synapse -- weight OR importance byte nonzero -- BEFORE
this call, checked against the PRE-update bytes) gates which quantizer
each cell uses.

.. code-block:: cpp

   // as of PR #45, linear_disldo.hpp -- pattern repeats per FP4/FP8 x
   // SIMD/scalar branch, this is the FP4 scalar-fallback instance:
   was_live4[lj] = (w_decoded_arr[lj] != value_type(0)) ||
                   (imp_decoded_arr[lj] != value_type(0));
   // ... later, at encode time:
   const uint8_t new_w = was_live4[lj] ? fp4_quantize_live(quant4[lj])
                                        : fp4_quantize(quant4[lj]);

.. _disldo_backward.nondeterminism_bug:

Real non-determinism bug in the "Deterministic" layer variant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.nondeterminism_bug``

Before this fix, block4 (dense/promoted) synapses ALWAYS used
``fp4_quantize_stochastic()`` regardless of the ``StochasticRounding``
template parameter, silently making "Deterministic" layers
non-deterministic whenever they touched block4 storage. Confirmed via a
standalone C++ repro: back-to-back runs of the exact same binary gave
different final nnz purely from this unseeded/uncontrolled stochastic
rounding, with no memory corruption or uninitialized reads involved
(valgrind memcheck came back clean). Fixed: the ``if constexpr
(!StochasticRounding)`` branch always uses the scalar ``fp4_quantize()``
codec -- no deterministic SIMD kernel exists yet
(``block4_vec_quantize_fp4``), only the stochastic one does, so this
branch is never SIMD-gated. A SIMD deterministic variant would be a
reasonable follow-up, not needed for correctness.

.. code-block:: cpp

   // as of PR #45, linear_disldo.hpp:
   if constexpr (!StochasticRounding) {
       // always fp4_quantize() (deterministic) here, regardless of block4
       const uint8_t new_w = was_live4[lj] ? fp4_quantize_live(quant4[lj])
                                            : fp4_quantize(quant4[lj]);
   }
   // WRONG (the bug): block4's own path called fp4_quantize_stochastic()
   // unconditionally, ignoring the StochasticRounding template parameter.

.. _disldo_backward.value_scale_reduction_dead_row:

block4's hand-inlined value_scale reduction and the dead-row bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.value_scale_reduction_dead_row``

block4's own ``value_scale`` reduction (after the parallel region) is
hand-inlined rather than routed through ``ScalePolicy`` (this path
predates that abstraction), but carries the same NaN/Inf guard and for
the same reason: the reduction sum (double, accumulated across possibly
many live synapses per row under dense connectivity) can overflow to Inf
when narrowed to ``value_type``, and the importance EMA never decays a
stray Inf back down, so a once-off overflow becomes permanent
corruption (see ``RMSpropScalePolicy::update``'s docstring,
``delta_csr_types.hpp``, for the full trace --
``sili_peridot/JOURNAL.md`` 2026-08-10). Same square-then-sum combination
and Adam-style bias correction as ``RMSpropScalePolicy::update`` for the
same cold-start-cancellation reason described above.

**Dead-row value_scale bootstrap**: a row with ZERO live synapses in
EITHER representation (scattered or block4) gets no gradient through
either path above, so ``value_scale``/``value_scale_importance`` would
stay frozen forever -- blocking any hope of bootstrapping predictions
from a genuine zero-synapse init (not the same as ``all_zero_init``,
which keeps full block4 connectivity with weight=0; a truly
zero-synapse row has no tile/synapse allocated at all). Checked after
both the scattered and block4 sections, using COMBINED liveness -- a row
live in ONE representation is already correctly handled by its own path
and must not be double-touched here. Deliberately NOT nested inside the
block4-tiles-exist check: a layer with zero block4 tiles (pure scattered
CSR, or genuinely empty) is exactly the fresh zero-synapse-init case this
exists for.

``g_agg = input[row] * S[b]``, where ``S[b] = sum_col(output_scale[col]*
output_grad[b,col])`` -- exact (distributive law), not an approximation,
and row-independent, so it costs ``O(batch*n_out)`` ONCE regardless of
how many rows are dead (only paid if at least one row is dead), not an
``O(n_in_dead*n_out)`` dense rescan per row -- real MiniCPM5-scale layers
can't afford the latter even while mostly zero-synapse. Drives
``value_scale`` via a standard two-moment Adam step (``value_scale_momentum``
as first moment, ``value_scale_importance`` reused as second moment --
safe since it's provably untouched by both paths above whenever a row is
dead in both) instead of single-moment RMSprop -- linear in ``g_agg``
(not ``g_agg^2``), so ``E[update]=0`` under zero-mean noise regardless of
variance, letting a genuinely inconsistent signal cancel out and stay at
zero (preserving sparsity) while a persistent bias still accumulates.

.. _disldo_backward.output_scale_gamma_rank_control:

output_scale reduction, AQRS gamma update, and dynamic rank control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.output_scale_gamma_rank_control``

``output_scale``'s gradient is reduced across threads then applied once
per column -- same "sum first, apply lr once" reasoning as ``value_scale``'s
own update, normalized by ``out_degree[c]`` (the column-axis equivalent
of ``nnz_this_row``).

AQRS gamma's own update (task #273/#283, Theorem 8) is gated on
``scale_gamma_is_trainable`` (same opt-in pattern as
``output_scale_is_trainable``) -- without it, every existing rank>=1
layer, including ones that have never heard of gamma, would get an
unsolicited gradient-driven perturbation to ``gamma_s_k(0)`` every step.
Uses the same ``ScalePolicy`` convention (RMSprop default) for the
gradient step, THEN a proximal L1 soft-threshold shrinkage on top --
that second step creates a genuine attracting fixed point at exactly
``gamma=0`` (soft-thresholding zeroes anything within
``l1_coef*learning_rate`` of zero after the gradient step; plain L2-style
decay only asymptotically approaches zero, never reaches it exactly).
L1 only applies to ``k>=1`` -- channel 0 is the always-on baseline
(``set_scale_rank`` rejects ``rank==0``, so channel 0 can never actually
be pruned), so penalizing it anyway just fights the fit with no possible
payoff.

**AQRS dynamic rank control** (task #273/#284): EMA-smoothed
``|gamma_k|``/``C_k``/``|grad_k|`` tracking, updated EVERY step -- see
``AQRS_DESIGN.md``'s corrected noise-mitigation design (EMA every step is
the actual noise filter; periodic N-step checking was rejected as a
"luck filter"). This is a second pass, after every channel's gamma value
is finalized, since ``C_k = |gamma_k| / sum_j|gamma_j|`` needs every
channel's current value first. The raw per-synapse-accumulated gradient
sum scales with layer WIDTH, not "how much does this channel matter" --
normalized by ``n_in*n_out`` here (trigger tracking only, task #294 fix;
gamma's own ``ScalePolicy`` step above stays on the raw gradient, since
that update already self-normalizes via its own second-moment estimate).

.. _disldo_backward.aqrs_additive_backward:

AQRS additive branch backward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.aqrs_additive_backward``

Differentiates ``disldo_forward``'s additive-branch block (task #277).
Genuinely independent of the sparse/block4 structure, computed as its
own self-contained pass. No-op at the default ``additive_rank==0``.

Forward recap: ``P[b,k] = sum_r U[r,k]*X[b,r]``; ``Y[b,c] +=
sum_k V[c,k]*P[b,k]``. Standard backprop:

.. code-block:: text

   dV[c,k]  = sum_b dY[b,c]*P[b,k]
   dP[b,k]  = sum_c dY[b,c]*V[c,k]
   dU[r,k]  = sum_b dP[b,k]*X[b,r]
   dX[b,r] += sum_k dP[b,k]*U[r,k]

``P`` is NOT cached from forward (this function has no access to
forward's locals, and this codebase's convention elsewhere is to
recompute from ``input`` rather than carry hidden state across the
forward/backward call boundary) -- recomputed here, cheap given
``additive_rank`` is small.

``dP``/``P`` above are the RAW (un-gamma'd) projections -- the real
``dL/dP_k = gamma_k * dP_raw[b,k]`` (since ``Y_k[b,c] =
gamma_k*V[c,k]*P_k[b]``), so:

.. code-block:: text

   dX          = sum_k U[r,k] * gamma_k * dP_raw[b,k]
   dU[r,k]     = gamma_k * sum_b dP_raw[b,k]*X[b,r]
   dV[c,k]     = gamma_k * sum_b dY[b,c]*P[b,k]
   dgamma_k    = sum_b P[b,k] * dP_raw[b,k]   (no gamma factor -- gamma_k
                 IS the thing being differentiated)

``additive_u``/``additive_v`` use ``AdamScalePolicy`` (per component,
applied AFTER the whole-batch gradient is aggregated, matching every
other per-scale update in this function: one update per call).

``additive_gamma``'s own update (task #289) mirrors ``scale_gamma``'s
update block exactly (same ScalePolicy-then-L1-then-EMA structure), but
uses the function's generic ``ScalePolicy`` (RMSprop-style,
momentum-free) rather than ``AdamScalePolicy``, despite
``additive_u``/``v`` using Adam. Found via a real test failure: Adam's
momentum overshoots the L1-created zero fixed point (Theorem 8), driving
gamma persistently negative instead of settling exactly at 0. Both
gamma variants (``scale_gamma`` and ``additive_gamma``) need the same
exact-zero-fixed-point property, so both use the same policy; only the
direction vectors get their own independent optimizer choice.
``log_space=false`` unconditionally -- that flag's meaning is
specifically about the multiplicative branch's coupling with the
quantized weight, which the additive branch doesn't have.

UNLIKE ``scale_gamma``'s k>0 L1 exemption: ``additive_rank`` has no
legacy always-on channel to protect (``min_rank=0`` in
``apply_additive_dynamic_rank_control`` -- the branch can legitimately
shrink itself back to fully off), so L1 applies to every ``k`` here,
including ``k==0``.

.. _disldo_backward.ccx_aware_reduction:

group-aware (CCX) reduction of ``t_dx``/``t_col_grad``, and why it needs pinning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_backward.ccx_aware_reduction``

**The problem this closes.** On a real dual-CCX desktop part (AMD Ryzen
7 3800XT, confirmed via
``/sys/devices/system/cpu/cpuN/cache/index3/shared_cpu_list``: cpu0-3
+8-11 share one L3 slice, cpu4-7+12-15 share a separate one),
``disldo_backward``'s final serial reduction loops -- one thread reading
every ``t_dx``/``t_col_grad`` slot in flat ``tid`` order after the
parallel region closes -- pay full cross-CCX Infinity Fabric latency for
any slot written by a thread on the OTHER cache domain. Measured on an
isolated single-size backward benchmark (width=288, 82944 elements,
fresh process per ``num_cpus``, no pinning): ``num_cpus`` 1-4 (fits one
CCX) ran 1020-1289us; 5-9 (spans both CCXes) jumped to 1882-2366us; 10-16
gradually diluted the fixed tax back down to 1162-1725us. This ISN'T a
software bug being fixed elsewhere in this file (critical-section
removal, persistent scratch) -- it survived process isolation and
thread-affinity experiments unrelated to those fixes.

**GPU-porting analogy** (this investigation doubled as a GPU-porting
exercise -- see ``project_sili_optimal_hardware_vision`` /
``sili_beyond_llm_benchmarks`` planning): a CCX here plays the same role
an SM (streaming multiprocessor) plays on a GPU -- a group of cores that
share one fast local cache/scratch resource, with a much slower path to
any OTHER group's cache. The fix below (each cache-domain's own "leader"
thread pre-sums its own group locally, then a small cross-group merge)
is exactly the two-level reduction pattern used for GPU cross-block
reductions: reduce WITHIN a block/SM using its fast shared memory first,
then combine the small number of per-block partials across the slow
global-memory path. The "persistent thread pool instead of forking a new
OpenMP team every call" idea discussed alongside this (not implemented)
is the CPU analog of a GPU persistent/mega-kernel: keep workers resident
and dispatch via a lock-free queue instead of paying a fresh
launch/rendezvous cost per kernel invocation.

**The fix.** ``cpu_topology.hpp`` parses the sysfs cache-domain table
once per process (cached), with a same-file API (``current_thread_group()``)
that queries a thread's CURRENT physical CPU via ``sched_getcpu()`` (a
cheap vDSO call) and maps it to a cache-group id. This is captured for
every thread once, inside the existing zero-fill parallel region (before
the scattered/block4 accumulate regions run), and used to build
``group_leader_tid``/``tid_to_g``. After BOTH accumulate regions close,
``t_dx``/``t_col_grad``/``t_col_grad_contrib`` are reduced in two levels:
if every thread landed in one group (``num_groups<=1`` -- true on any
single-CCX machine, and true whenever ``num_cpus`` fits inside one CCX
even on a multi-CCX one), a plain flat serial accumulate runs, matching
the old cost plus one negligible extra copy pass. Otherwise, a NEW small
``#pragma omp parallel`` region lets each group's leader thread sum ONLY
its own group's slots (cache-local reads, and every leader runs
concurrently), leaving only ``num_groups`` (typically 1-2) values for a
tiny final serial merge -- collapsing what used to be O(num_cpus)
cross-domain reads down to O(num_groups). A stale/wrong group assignment
can only cost performance, never correctness: the reduction still visits
every ``tid`` in ``[0, num_cpus)`` exactly once regardless of grouping.
Gamma's own reduction (tiny, ``rank`` is small) and block4's
``t_row_grad`` reduction (a separate, pre-existing, deliberately
out-of-scope buffer) are left flat.

**Why this needs thread pinning to actually pay off.** The whole design
assumes a thread's cache-group membership, captured once near the start
of the call, stays valid for the REST of that call. Without
``OMP_PROC_BIND``/``OMP_PLACES`` set, nothing guarantees that -- the OS
is free to migrate a thread across CCXes mid-call. Measured directly:
unpinned, this fix alone was a wash-to-regression (``num_cpus=4``, the
previous best option, went from 1020us to 1478us -- worse, because at
low thread counts the OS sometimes DOES scatter threads across both
CCXes even for 3-4 threads, so the code correctly detects
``num_groups=2`` and pays for an ENTIRE EXTRA parallel region's
cross-CCX fork/join rendezvous just to reduce a handful of values,
adding cost instead of avoiding it). Pinned
(``OMP_PROC_BIND=true OMP_PLACES=cores``, set in ``sili/__init__.py``
before the compiled extension is ever imported -- see that file's own
comment for why Python-level ``os.environ`` is the only place early
enough to influence libgomp's own env parsing), the same code hit a new
session-best 812us at ``num_cpus=16`` (previous best-ever across every
configuration measured all session: 890.8us at ``num_cpus=2``, from
BEFORE any of this session's fixes) and meaningfully improved
``num_cpus`` 2/8/9/10 too -- but ``num_cpus`` 3-6 got WORSE than the
pre-fix baseline even pinned (``num_cpus=4``: 1605us). **Practical
upshot**: this fix is only a net win when paired with pinning AND a
``num_cpus`` choice of 2 or >=8 on this machine's topology -- staying at
the old "safe default" of ``num_cpus=4`` under the new pinned regime is
now the single worst non-plateau option measured.

.. _disldo_forward.fp32_block4_cross_tile_pairing:

FP32 block4: cross-tile (non-adjacent) pairing PoC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*ID:* ``disldo_forward.fp32_block4_cross_tile_pairing`` /
``disldo_backward.fp32_block4_cross_tile_pairing``

The column/row-pairing widenings above (``disldo_forward.
fp32_block4_avx2_column_pairing``, ``disldo_backward.
fp32_block4_avx2_row_pairing``) pair two halves of the SAME tile. This is
a different axis: pairing TWO DIFFERENT tiles that share a block-row
(``br``) but sit at different block-columns (``bc``) -- the case that
matters once a layer has real gaps (checkered/sparse block4 occupancy),
not just the fully-dense case those widenings targeted.

**Forward's cross-tile axis**: for a fixed column-within-tile ``lj``,
combine that SAME ``lj`` position from two DIFFERENT tiles sharing ``br``
(different ``bc``) into one 8-wide op -- both tiles still read the same 4
input rows, so this is cheap: forward's tile-collection loop already
walks a row's occupied tiles in ``bc``-ascending order, so pairing
consecutive entries needs no new indexing.

**Backward's cross-tile axis** (the mirror image, per the design
decision behind pairing on ``br`` rather than ``bc``: "same br different
bc for backward as well -- sum the input grad, the weight grads are
separate anyway"): for a fixed row ``li`` (same actual row), combine that
row's computation against TWO DIFFERENT tiles (``bc0``, ``bc1``) sharing
``br``. Row-level quantities (``imp_scale``, ``effective_lr``) become
SHARED; column-level quantities (``col``, ``get_output_importance_scale``,
``dy``) become PER-HALF (genuinely different columns from the two tiles);
``dx`` accumulation FOLDS both tiles' contributions into one row entry
(analogous to ``mdx``'s per-batch accumulation, NOT a single accumulator
hoisted outside the batch loop -- ``mdx`` is shaped ``(batch, in_cols)``,
each batch sample needs its own slot); weight/importance updates write to
TWO SEPARATE tiles. This stays within one thread's row-ownership (no new
column index, no cross-thread race), since a whole row's tiles are
already collected together -- confirmed cheap, unlike an earlier
(wrong) assumption that it would need new column-indexed infrastructure.

Standalone PoC files (not yet ported into the real kernel; see below):
``tests/unit/test_disldo_block4_fp32_crosstile_forward.cpp`` and
``test_disldo_block4_fp32_crosstile_backward.cpp``. Both hand-build a
genuinely gapped ("checkered", every odd ``bc`` skipped) block4 fp32
layer via ``get_or_create``/``set_weight``/``set_importance`` (mirroring
``block4_load_dense_fp32``'s own construction loop minus the odd-``bc``
skip), use the REAL ``disldo_forward``/``disldo_backward`` as both
correctness oracle and timing baseline, and measure only the one new
hand-written cross-tile function against it.

**Backward-specific correctness pitfall, found and fixed**: two
``Block4TileHandle32`` handles into the SAME row held open
simultaneously while mutating one is a real use-after-free hazard (a
write-triggered sparse<->dense repack on one handle can memmove the
row's shared byte buffer out from under the other's still-open
``byte_pos``) -- exactly the hazard ``disldo_backward.
row_workspace_snapshot_fix`` documents and the real kernel avoids via
``snapshot_row``/``unpack_workspace_tile``. The PoC's simpler
(valid only because tiles never actually resize in this fixed-checkered
test scenario) fix: read BOTH tiles fully into local arrays first, each
handle opened and closed sequentially and never two alive at once,
compute using only local data, then write both tiles back the same way.
A second, purely mathematical bug: the PoC's ``ci`` (combined importance)
formula initially used ``imp_scale(row) * value_scale(row)`` -- both
row-level -- copying the row-pairing kernel's shared/per-half framing
too literally. The real formula (``process_row_pair_fp32``,
``combined_imp_scale8[lj] = imp_scale0 * out_imp_scale4[lj]``) uses
``imp_scale(row) * get_output_importance_scale(col)`` -- the SECOND
factor is column-level, not row-level, so it must be per-half (from each
tile's own columns), not shared. Both bugs were undetectable by a naive
forward-probe correctness check post-backward, because ``disldo_backward``
ALSO runs its own unconditional per-row ``value_scale`` RMSprop update
after the per-cell loop (``linear_disldo.hpp:3568-3648``, block4's
dead-row-independent value_scale pass) which the PoC deliberately doesn't
replicate (out of scope -- that machinery is orthogonal to the cross-tile
per-cell math under test, and the real kernel port reuses it unmodified)
-- so a forward probe conflated expected ``S(row)`` drift with actual
correctness. Fixed by comparing raw stored ``cw``/``ci`` directly instead
of probing through a forward pass.

**Measured** (arch-sandbox, 15 interleaved-equivalent runs, ``n_in=n_out=
256, batch=8``, oracle vs hand-written function, median ns/call):

- Forward: ``adjacent_only`` (existing column-pairing) vs
  ``cross_tile``: a real, consistent **~17-20% speedup**. Root cause:
  the tile-handle fetch cost (``at_index``/``raw_data``) is halved (once
  per pair instead of once per tile), not really about memory adjacency
  itself.
- Backward: ``adjacent_only`` (existing row-pairing) vs ``cross_tile``:
  median 1,594,630 ns/call vs 589,950 ns/call across the 15 runs -- a
  much larger **~2.7x speedup**, and far more STABLE than the adjacent
  baseline (which showed large run-to-run variance, 1.5M-4.2M ns/call,
  vs cross_tile's tight 589k-650k ns/call band across the same 15 runs)
  -- plausibly because cross-tile pairing halves the number of per-tile
  RMSprop ``update_ci``/``update_cw`` calls (once per row-pair across
  TWO tiles' worth of columns, instead of once per tile), which is a
  larger fixed per-tile cost on the backward path than on forward's.
  (Note: as expected, this is dense-checkered-layout-neutral in principle
  -- the real payoff is enabling cheaper work under genuine sparsity, not
  a dense speedup per se, though the measured dense case is faster too.)

Both PoCs were correctness-validated (weight/importance/dx all
effectively exact, not just tolerance-close) before the port into the
real kernels described in the commit(s) alongside this doc entry.

**Forward production port, measured against the real kernel (not the
PoC harness).** ``disldo_forward``'s existing tile-collection loop
already walks each block-row's tiles in ``bc``-ascending order before
the parallel region; a new per-tile ``uint8_t`` scratch flag
(``scratch_tile_is_follower``) marks every SECOND tile of a same-``br``
run as "already consumed by its leader," computed for free (one
``bk % 2`` check) during that same walk. The parallel loop then pairs a
leader with its follower into one 8-wide op (mirroring the PoC's
``forward_cross_tile``), or falls back to the existing within-tile
column-pairing (``process_pair``, unchanged) for a genuine solo tile.
FP8/FP4 are structurally untouched (their whole per-tile branch moved
into an ``else`` of a top-level ``if constexpr`` on ``VALUES_TYPE``, so
none of this new code is even compiled for those instantiations).

The PoC's own ~17-20% speedup (measured as a hand-rolled, single-
threaded, no-OMP-overhead harness) does NOT reproduce when measured on
the real ``disldo_forward`` call (interleaved remote A/B, 20 runs each,
``n_in=n_out=256, batch=8``, checkered layout): at ``num_cpus=4``, mean
before ~77.2us vs after ~79.0us (statistically indistinguishable); at
``num_cpus=1`` (matching the PoC's single-threaded shape), medians were
essentially identical (~174.3-181.5us both arms). The real kernel's
fixed per-call overhead (rebuilding the ``tile_br``/``tile_bc``/
``tile_elem``/``tile_byte``/``tile_is_follower`` scratch vectors, the
per-thread output-buffer allocation and final reduction pass, OMP
fork/join) is large enough relative to this problem size that halving
the per-pair ``at_index()``/``raw_data()`` calls doesn't show up above
the noise floor -- unlike the PoC's minimal harness, where that halving
was a much larger fraction of total cost.

**This is treated as a real, useful result, not a null result**: cross-
tile pairing reaches PARITY with the already-optimized fully-dense
adjacent-pairing baseline while now correctly handling a genuinely
GAPPED (checkered) sparse tile pattern -- the two cases are "nearly
indistinguishable" in measured cost. That parity is the actual target:
this pairing axis exists to let sparse block4 occupancy patterns (which
won't generally have convenient adjacent-column pairs to combine) still
reach full 8-wide SIMD width, which matters more as sparsity increases
past this 50%-density test case, and is the same underlying pattern
intended to later fill GPU warps (task #408) -- a CPU-side wash on a
50%-dense pattern is a good sign for both of those, not evidence against
landing it.

**Backward production port.** Unlike forward (read-only, no aliasing
risk), backward's real block4 write loop already snapshots each block-
row into a ``RowWorkspace`` and unpacks ONE tile at a time into a local
``scratch_buf`` before calling ``process_tile``/committing it back --
this means the cross-tile port needed no PoC-style read-all-then-write-
all workaround at all: two DIFFERENT tiles' scratch buffers can safely
be unpacked, processed together, and committed independently, since
neither is ever a live handle into the shared store. A new
``process_tile_pair_fp32`` lambda (mirroring ``process_row_pair_fp32``
with the shared/per-half roles swapped, exactly as documented in the
PoC's own header comment) handles the paired case; the existing
``process_tile`` is unchanged and still used for solos, FP8, FP4, and
(deliberately) any ``scale_rank > 1`` FP32 layer -- gated on
``rank == 1`` so AQRS/rank-N layers get zero behavior change, matching
the validated PoC's own scope. **Sequencing within a pair is the one
place this differs from a naive port**: tile B's byte position depends
on tile A's stored length, and committing A can change that length by
resizing A's sparse/dense encoding in the row's shared byte buffer,
shifting where B's bytes actually live -- so A is unpacked, then B is
unpacked (both using A's PRE-write length to locate B), the row's math
is computed from both, A is committed, A's length is RE-READ (post-
commit), and only THEN is B committed at its now-correct position. This
mirrors, applied twice per pair, exactly how the pre-existing single-
tile loop already re-reads its own tile's post-commit length before
advancing to the next tile -- not a new pattern, just applied to a pair
instead of a singleton.

Correctness gate: a NEW standalone test,
``test_disldo_block4_fp32_crosstile_scattered_divergence.cpp`` (not
just re-running the PoC's own oracle checks), builds two arms with
IDENTICAL weights -- one entirely scattered-CSR (completely untouched by
this change), one entirely block4 with checkered occupancy (``n_out=24``
gives 3 block-columns, so both a genuine PAIR and a genuine SOLO occur
in the same row, exercising both new code paths at once) -- and confirms
forward output, ``dx``, and post-backward weight updates all match to
float precision (errors ~1e-6, pure summation-order noise). Passes
locally and on arch-sandbox. Full local C++ suite (160/160) still
passes unmodified.

Measured against the real kernel (interleaved remote A/B, checkered,
``n_in=n_out=256, batch=8``): at ``num_cpus=4`` (20 runs), before/after
medians overlap heavily (~406us vs ~407us median-of-medians, though the
FIRST ~11 of 20 runs showed a real ~20-23% win before an apparent
machine-wide regime shift mid-run affected both arms roughly equally);
at ``num_cpus=1`` (15 runs, matching the PoC's own single-threaded
shape), before/after track each other closely through the same kind of
shift (~1247us vs ~1197us median-of-medians, within the shift's own
noise band). Net result: the SAME parity finding as forward, for the
SAME underlying reason (the real kernel's fixed per-call overhead --
``RowWorkspace`` snapshot/merge, per-thread reductions, OMP fork/join --
dilutes the PoC-measured 2.7x win, which came from a minimal harness
that paid almost nothing else). Landed as a real result for the same
reason as forward: parity on a 50%-dense checkered pattern while now
correctly handling genuinely gapped occupancy is the actual target here,
not a further win on the already-dense case.

**FP8/FP4 forward production timing.** Same interleaved-A/B methodology
as fp32 (15 remote runs, striped, ``n_in=n_out=256, batch=8,
num_cpus=4``, before=commit ``c13d1b7`` vs after=commit ``6b89ce8``):
both precisions' before/after medians overlap heavily inside the same
bimodal noise band already characterized elsewhere in this doc (values
cluster in two clusters, ~57-65us and ~78-93us, in BOTH arms, tracking
each other run-to-run) -- fp8 before/after median-of-medians ~85.2us/
~83.8us, fp4 ~73.6us/~79.8us. Same parity conclusion as fp32 and the
same root cause (real kernel overhead dilutes the isolated pairing
win); not treated as a regression signal given the shared noise
structure across both arms.

**FP8 backward production port.** Same structural approach as fp32's
port (``process_tile_pair_fp8`` mirroring ``process_tile_pair_fp32``
with the shared/per-half role swap; RowWorkspace already provides
independent per-tile scratch buffers, so no aliasing hazard; scoped to
``scale_rank == 1``; falls back to the existing ``process_tile`` for
solos and rank>1), but with a genuine fp8 encode/decode step per cell
(``fp8_decode_bits``/``fp8_quantize_live``/``fp8_quantize_stochastic_
live``, including the ``was_live8`` zero-escape gate) mirroring
``process_row_pair_fp8``'s own decode/encode exactly, instead of FP32's
raw float memcpy.

Correctness gate: ``test_disldo_block4_fp8_crosstile_backward_
divergence.cpp``, same striped-scattered-vs-block4 pattern as fp32's
(both a real pair and a real solo per row) -- forward/dx/post-backward
weights all match to within fp8 quantization-noise tolerance. (One real
test bug found and fixed during this: the scattered arm's importance
was initialized to 0 while the block4 arm's was nonzero, an asymmetry
between the two arms' STARTING state, not a kernel bug -- caught
because it produced a large post-backward divergence despite matching
forward/dx, which was the same shape of false alarm as fp32's earlier
value_scale confound, but this time the actual cause was a test setup
mismatch rather than an incomplete PoC scope.) Full local C++ suite
(160/160) and existing dense fp8 backward regression test both pass
unmodified (the dense test now exercises cross-tile pairing on every
tile, since a fully-dense row always has a same-br partner).

Measured against the real kernel (interleaved remote A/B, striped,
``n_in=n_out=256, batch=8, num_cpus=4``, 15 runs, before=commit
``2344b84``): unlike every other cross-tile port measured so far
(fp32 forward/backward, fp8/fp4 forward -- all landed at parity), FP8
backward shows a real, consistently separated speedup: median-of-
medians 876.4us before vs 621.5us after (~29% faster), with ``after``
faster than ``before`` in 13 of 15 interleaved samples -- not just
overlapping noise bands like the other precisions. Plausible reason
this one differs: FP8's per-tile fixed overhead (scale-rank scratch
setup, ``was_live8`` bookkeeping, the scalar ``fp8_quantize_live``/
``fp8_quantize_stochastic_live`` encode calls) is heavier than FP32's
bare float write, so halving the number of PER-TILE setup/lookup
operations (not just per-cell math) has a larger relative payoff here
than it did for the lighter-weight FP32 kernel.

**FP8/FP4 backward: rank-N generalization.** The FP8/FP4 backward ports
below were initially scoped to ``scale_rank == 1``, matching FP32's own
scope. Direct feedback caught a real gap this left open: ``scale_rank``
is not fixed at construction for FP8/FP4 layers -- AQRS dynamic rank
control (task #273/#285) can grow a layer's rank live during training --
so a rank==1 gate meant any layer AQRS grew past rank 1 would silently
stop getting the cross-tile speedup for as long as it stayed there, with
no error, just quietly slower. FP32 was deliberately left unchanged
(unlike FP8/FP4, it has no AQRS low-rank use case, so the gap doesn't
apply there). Root-caused first that FORWARD's cross-tile pairing (all
three precisions) already needed NO change at all: it reads the
multiplicative scale via ``weights.get_scale(row, col)``, which already
sums over every ``scale_rank`` component internally -- there was never a
rank gate in forward to begin with. Only BACKWARD's
``process_tile_pair_fp8``/``process_tile_pair_fp4`` needed the
generalization, since updating each rank-``k`` component's own
value_scale/output_scale via gradient descent requires the full
per-``k`` accumulator machinery that the scalar rank==1 version didn't
have. Both lambdas were rewritten to loop over ``k`` explicitly (plain
scalar loops, not the SIMD/``scale_rank_scratch`` machinery
``process_row_pair_fp8``/``process_row_pair_fp4`` use for their own
rank-N support -- lower-risk given this is a correctness generalization,
not a further speed push), re-mapping the row0/row1 role split those
lambdas use to this lambda's tileA/tileB role split (``value_scale_k``
becomes SHARED across both halves since they're the same row;
``out_scale_k`` becomes PER-HALF since A and B are different columns;
``mrow``/``mgamma`` fold across all 8 lanes into ONE accumulator per
``k`` instead of two, since one row feeds both halves). FP4 additionally
keeps its ``quant_floor`` zero-escape substitution in the ``prod_g``/
``prod_contrib`` terms exactly as before, now inside the ``k`` loop. The
``rank == 1`` gate was removed from both write-loop branches (FP32's
stays -- it's still the deliberately-scoped case).

Correctness gates: two NEW standalone tests,
``test_disldo_block4_fp{8,4}_crosstile_backward_rank2_divergence.cpp``,
mirroring the existing rank==1 striped tests but with ``scale_rank=2``
and a real, nonzero, IDENTICAL-across-both-arms second rank channel
(``set_value_scale_raw_k(row,1,...)``/``set_output_scale_raw_k(col,1,
...)``) -- not just a default-zero rank-2 that would pass trivially.
Assert forward, ``dx``, post-backward weights, AND the per-component
``k=1`` ``value_scale``/``output_scale`` themselves (not just the summed
``S``) all match to float precision. Both pass (err ~3e-6 float, exact
0.0 on the per-component check). Full local suite (160/160) and both
existing rank==1 gates still pass unmodified, on both this machine and
arch-sandbox.

Measured cost of the generalization (interleaved remote A/B, striped,
``n_in=n_out=256, batch=8, num_cpus=4``, 10 runs, rank==1 case only --
this measures whether generality itself has a price, not a new
precision): the rank-N version still beats the pre-cross-tile baseline
by a real, fully consistent ~2.25x (median-of-medians 1882.0us before
vs 837.6us after, 10/10 samples faster), but costs ~30% versus the
earlier rank==1-SPECIALIZED version's 643.9us (the ``std::vector``
scratch allocated per row for the ``k``-loop -- ``value_scale_k_row``,
``out_scale_kA``/``out_scale_kB``, the ``mrow``/``mgamma``/``mcol``
accumulators -- isn't free, even when ``rank==1`` makes each vector
length 1). Judged worth it: the alternative is a real, un-flagged
performance cliff the instant AQRS grows a layer's rank, which is a
documented, active mechanism in this codebase, not a hypothetical.

**FP4 backward production port.** Same structural approach as fp32/fp8's
ports (``process_tile_pair_fp4`` mirroring ``process_tile_pair_fp8``'s
shared/per-half role swap; scoped to ``scale_rank == 1``; falls back to
the existing ``process_tile`` for solos and rank>1), with FP4's packed
single-byte nibble decode/encode (``Block4Tile::slot_index``, weight in
the low nibble / importance in the high, ``tdata[slot] = uint8_t((new_imp
<< 4) | new_w)``) instead of FP8's two separate byte arrays. One
FP4-specific detail carried over exactly from ``process_row_pair_fp4``:
the gradient-accumulation terms feeding ``mrow``/``mgamma``/``mcol``
(``prod_g``/``prod_contrib``) use ``quant_floor`` -- the raw quantized
weight with a ``zero_escape_eps`` substitution at exactly zero, so a
permanently-dead synapse still receives an importance gradient -- while
``contrib``/``g_agg``/``mdx`` use the raw (unfloored) quantized value,
matching FP32/FP8's equivalent terms. FP32/FP8 have no such split (no
zero-escape substitution in either).

Correctness gate: ``test_disldo_block4_fp4_crosstile_backward_
divergence.cpp``, same striped-scattered-vs-block4 pattern as fp32/fp8's
(both arms' importance round-tripped through the fp4 grid identically,
learning the fp8 test's lesson about symmetric starting state) --
forward/dx/post-backward weights all match to within fp4 quantization-
noise tolerance (~2e-6). Full local C++ suite (160/160) and the existing
dense fp4 backward regression test (``test_disldo_block4_fp4_backward_
wide_simd.cpp``) both pass unmodified.

Measured against the real kernel (interleaved remote A/B, striped,
``n_in=n_out=256, batch=8, num_cpus=4``, 15 runs, before=commit
``f2f68f8``): a real, large, and extremely consistent speedup --
median-of-medians 1837.2us before vs 643.9us after (~2.85x faster),
``after`` faster than ``before`` in all 15 of 15 interleaved samples
(not overlapping noise bands like fp32/fp8 forward or fp32 backward).
Root cause, unlike fp8 backward's more modest ~29%: FP4's EXISTING
within-tile adjacent-row pairing path (``process_row_pair_fp4``, still
used for every solo tile) is the rank-N-generic implementation --
allocating several ``std::vector``\\ s (``value_scale_k_row0/1``,
``mrow_local0/1_k`` and their ``_contrib`` counterparts) on the HEAP
per call, twice per tile (once per adjacent-row pair) -- while the new
``process_tile_pair_fp4`` is the same scalar, rank==1-specialized,
entirely-stack-array design already used for FP32/FP8's cross-tile
lambdas. So for FP4 specifically, cross-tile pairing doesn't just halve
the number of per-tile setup/lookup operations (the fp8 backward
finding) -- on the striped occupancy pattern tested here, it replaces
the ONLY previously-available multi-row-fused path (the heap-allocating
rank-N one) with a heap-free one, which is a substantially larger win.

**Cross-precision dequantization equality tests.** Before the planned
cyclomatic-complexity reduction pass (splitting the ``if constexpr``
precision branches in ``disldo_forward``/``disldo_backward`` into
separate per-precision template instantiations/files), a regression
gate was needed to prove the three precisions' underlying MATH is
identical when their storage happens to agree, so the refactor has
something concrete to preserve. FP4's representable grid (``{0,
+-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}``) is a strict subset of FP8
E4M3's, which is itself a strict subset of float32's -- so a weight/
importance value drawn from the coarser grid decodes to the EXACT SAME
float in either storage format. Two new standalone tests,
``test_dequant_equality_fp4_vs_fp8.cpp`` and
``test_dequant_equality_fp8_vs_fp32.cpp`` (scattered CSR only, not
block4 -- this tests the precisions themselves, independent of storage
format, which block4-vs-scattered parity tests already cover
elsewhere), build two arms with IDENTICAL dequantized weights/
importance (fp4 arm drawn from ``FP4_TABLE`` directly; fp8-vs-fp32 arm
drawn from arbitrary floats round-tripped once through fp8's own codec
so both land on an fp8 fixed point) and assert forward output and
backward's ``dx``/``neuron_input_accum``/``neuron_grad_accum`` all
match -- using a REAL nonzero ``learning_rate`` (not just ``lr=0``),
since ``dx`` is computed from the pre-update stored weight and is
provably independent of how each precision's own grid will round the
post-update value. Post-update WEIGHTS are deliberately never compared
between arms: once a real update happens, fp4's coarser grid and fp8's
finer one round the same float delta to different codes, and that
divergence is expected ("outside of learning... predictable"), not a
correctness bug. Both tests passed BIT-EXACT (all four measured
quantities: ``0.00000000``) on the first run, not just within
tolerance -- confirming the scattered forward/backward math performs
literally the same floating-point operations in the same order across
all three precisions, with quantization as the only actual difference.
Registered in ``CMakeLists.txt``'s ``SILI_STANDALONE_TESTS`` (162/162
total now) so they gate the upcoming refactor going forward.

.. _block4_codec_refactor:

**block4_codec_refactor.** Ran the refactor the dequant-equality tests
above were built to gate: ``disldo_forward``/``disldo_backward``'s
``if constexpr (std::is_same_v<VALUES_TYPE, ...>)`` triplication
across fp32/fp8/fp4 -- three near-identical copies of the same kernel
math, differing only in which decode/encode/quant_floor call gets
made -- collapsed onto a new shared trait, ``Block4Codec<VALUES_TYPE>``
(``sili/lib/headers/block4_codec.hpp``): a thin wrapper around the
ALREADY-separate ``fp4quant.hpp``/``fp8quant.hpp`` bit-quantization
functions (not a reimplementation), exposing ``decode_weight``/
``decode_importance``/``decode_weight_column4``/``quant_floor``/
``encode<StochasticRounding>``/``stored_tile_len`` plus the
``has_zero_escape``/``has_was_live_gate``/``scratch_bytes`` compile-time
constants. Every real kernel branch that used to be written three times
per precision is now written once, generic over ``Block4Codec``.

Genuinely-non-trivial per-precision differences (not just decode/encode
swaps) were preserved exactly via the trait rather than papered over:
FP4 alone substitutes ``zero_escape_eps`` for an exactly-zero quantized
weight in the AQRS gradient-accumulation terms (``mrow``/``mgamma``/
``mcol``), never in ``contrib``/``g_agg``/``mdx`` -- ``Block4Codec::
quant_floor`` is identity for fp8/fp32, epsilon-substituting for fp4,
called at exactly the same sites the original per-precision code did.
FP8 and FP4 both gate their write-back through a ``was_live`` check
(a cell that was never a real synapse must stay allowed to round to
zero); FP32 has none (no quantization floor to escape) --
``Block4Codec<DeltaCSRBiValues<float>>::encode`` accepts and ignores
the parameter, matching FP32's existing unconditional-memcpy write-back
exactly. ``process_tile_pair`` (cross-tile pairing) additionally
generalized FP32's historically rank==1-hardcoded implementation onto
the same rank-N-generic loop FP8/FP4 already used -- verified safe
because ``scale_rank`` is 1 for every fp32 block4 backward configuration
actually exercised anywhere in this codebase, so the collapse is
behavior-identical for every real caller (the rank==1 gate at the
pairing call site was nonetheless kept exactly as before, so this
refactor does not silently widen fp32's reachable code path to an
untested rank>1 configuration).

``sisldo_ops.hpp``'s ``disldo_backward_sparse_grad`` had already
centralized its own per-precision dispatch into four local lambdas
(``decode_weight``/``decode_importance``/``encode_and_store``/
``tile_len_of``) doing the identical three-way ``if constexpr``
this file used to have inline -- their bodies now call into the same
shared ``Block4Codec``, eliminating that second, independent copy of
the bit-quantization logic.

``linear_disldo.hpp`` was then split by concern, mirroring the existing
``delta_csr_types.hpp``/``delta_csr_memory.hpp`` precedent (the only
prior multi-file split in ``sili/lib/headers/``): ``linear_disldo.hpp``
is now a thin entry point including ``linear_disldo_forward.hpp`` and
``linear_disldo_backward.hpp``, with no external ``#include
"linear_disldo.hpp"`` site needing any change.

Measured via ``lizard`` (NLOC / CCN, before -> after):

.. list-table::
   :header-rows: 1

   * - function
     - before
     - after
   * - ``disldo_forward``
     - 479 / 97
     - 216 / 48
   * - ``disldo_backward``
     - 3640 / 524
     - 1561 / 257
   * - ``disldo_backward_sparse_grad``
     - 729 / 140
     - 663 / 120
   * - ``sisldo_forward``
     - 290 / 59
     - 290 / 59 (unchanged -- its own inline decode-with-early-zero-
       skip and merge-eviction callback were left as-is, each only 3
       lines per branch, not worth a new trait member)

Even after the collapse, ``disldo_backward`` at CCN 257 is still far
above ``lizard``'s own default warning threshold (CCN>15) -- getting
under that would need genuinely separate functions per code path
(training vs not, full-tile vs boundary, stochastic vs deterministic,
``DeferredScaleWrite``), not just per precision, which is out of scope
here: this pass's goal was removing duplication and shrinking file
size (motivated in large part by Claude Code itself hitting context
compression reading the pre-refactor ~4600-line monolith), not hitting
a specific CCN target. ``tools/lint_all.sh``'s ``lizard`` check is
advisory-only and remains so.

Every phase of this refactor was gated on the two dequant-equality
tests above staying BIT-EXACT (``0.00000000``) throughout, the full
local ``ctest`` suite, and re-verification on the arch-sandbox remote
(including direct recompilation+rerun of the cross-tile backward
divergence/timing tests specifically, since those are this session's
own earlier ~2.85x-fp4/~29%-fp8 speedup work and needed confirmation
the refactor didn't quietly regress it, not just that correctness
held).

.. _block4_backward_extract_function_refactor:

**block4_backward_extract_function_refactor.** Even after
``block4_codec_refactor`` above, ``disldo_backward`` still measured
CCN 257 -- flagged as still too high for normal engineering practice
(most teams would reject this in review), not because the precision
triplication was still there (it wasn't), but because the function
body contained three deeply nested closures (``process_tile``, and
inside it ``process_row_pair``; plus a sibling ``process_tile_pair``)
plus a large inlined single-row update block, all defined as ``auto x
= [&](...){...}`` lambdas capturing ~25 variables by reference.
``lizard`` (like most CCN tools) folds every branch inside an
inline-defined lambda body into the ENCLOSING function's own
complexity count, so none of that internal branching (SIMD-vs-scalar,
full-tile-vs-boundary, training-vs-not, stochastic-vs-deterministic)
was visible to the tool as belonging to a separate unit -- it was all
attributed to ``disldo_backward`` itself.

The fix was textbook Extract Function (Fowler): pull each closure out
into a real, independently-named, top-level template function taking
explicit parameters (or a small parameter-object struct) instead of
capturing the enclosing scope by reference. Each extracted function
then gets measured separately by ``lizard``, and ``disldo_backward``
itself shrinks down to its actual orchestration logic (the row-block
loop, the pairing-vs-single-tile dispatch, byte-position bookkeeping,
the final reduction).

**GPU-portability constraint.** This kernel's eventual target is
Kompute/Vulkan/GLSL (with NVIDIA extensions). GLSL has no lambdas, no
closures, no C++ templates, no ``std::function``/virtual dispatch, no
``std::pair``/STL containers as first-class types -- only plain
functions with explicit ``in``, ``out``, and combined-direction
parameter qualifiers, and plain structs.
The planned path for "templates" on the GPU side is Boost.Preprocessor-
style macro expansion or an external Python/C++ codegen step that
mechanically expands compile-time-dispatched C++ (template parameters,
``if constexpr`` -- exactly what ``Block4Codec`` already does) into
concrete GLSL variants, not runtime polymorphism, which has no GLSL
analogue at all. This shaped the extraction:

- Converting the nested lambdas into free functions is a move *toward*
  GLSL's function model, not away from it -- good alignment, not a
  tradeoff.
- Two new parameter-object structs (``sili/lib/headers/
  block4_codec.hpp``) group state the way a future GPU binding layout
  would: ``Block4BackwardParams<SIZE_TYPE, VALUES_TYPE, COL_TYPE>``
  holds broadcast/read-only call-level state (constructed once before
  the ``#pragma omp parallel`` region -- eventually a UBO/push-
  constant), ``Block4BackwardAccumulators<value_type>`` holds
  per-thread mutable output state with named accessor methods
  (``mcol_at``/``mrow_at``/``mgamma_at`` and their ``_contrib``
  counterparts, replacing the old per-thread lambda closures 1:1 at
  the call-site syntax level) -- eventually an SSBO, reduced
  differently via atomics/workgroup-reduce on GPU, but the same
  CPU-side field grouping conceptually. Named ``...Params``/
  ``...Accumulators``, not ``...ThreadState``, since "thread" is a
  CPU-specific concept this struct shouldn't presuppose.
- Per-call coordinates (``br``, ``bc``, ``li``, ``row``, ``nnz``,
  ``tdata`` pointers) stayed explicit scalar/pointer function
  arguments, not folded into either struct -- on GPU these derive
  from ``gl_GlobalInvocationID``, genuinely per-invocation rather than
  uniform/broadcast state, so keeping them as plain params is the
  correct analogue.
- No ``std::function``, virtual dispatch, or other runtime-polymorphic
  indirection was introduced to shrink CCN -- that would make the
  eventual GPU port harder, not easier, since none of it exists in
  GLSL. All dispatch stayed at compile time (template parameters /
  ``if constexpr``), matching ``Block4Codec``'s existing pattern: every
  extracted function takes the same 7 template parameters
  ``disldo_backward`` itself has (``SIZE_TYPE, VALUES_TYPE, COL_TYPE,
  ScalePolicy, DeferredScaleWrite, StochasticRounding,
  SynapsePolicyT``), specified explicitly at every call site since 4
  of them don't appear in any function parameter type and thus aren't
  deducible.
- ``process_tile_pair``'s ``std::pair<bool, bool>`` return became a
  plain named struct, ``Block4TileDirtyPair{bool a; bool b;}`` --
  ``std::pair`` has no GLSL analogue, a plain struct does.

**Explicitly out of scope for this pass** (flagged, not fixed, to
avoid conflating two different efforts): ``process_row_pair``/
``process_tile_pair``'s existing ``std::vector<value_type>``
allocations for per-rank scratch (``value_scale_k_row``,
``out_scale_kA``, etc.) are a real future GPU-portability concern --
dynamic-rank heap allocation has no GPU equivalent, and would need a
fixed max-rank array or an SSBO-backed scratch buffer instead -- but
touching that now would conflate this CCN-reduction pass with a
genuinely separate allocation-strategy redesign.

Four functions were extracted from ``disldo_backward``, in this order,
each verified independently (full local + remote ``ctest``, plus a
direct bit-exact correctness check via the relevant standalone timing
test) before moving to the next:

.. list-table::
   :header-rows: 1

   * - function
     - NLOC
     - CCN
     - replaces
   * - ``block4_backward_process_single_row``
     - 255
     - 33
     - the inlined generic single-row update block
   * - ``block4_backward_process_row_pair``
     - 201
     - 22
     - the ``process_row_pair`` lambda (AVX2 within-tile adjacent-row
       pairing)
   * - ``block4_backward_process_single_tile``
     - 37
     - 8
     - the ``process_tile`` lambda (per-tile row-loop + pairing
       dispatch)
   * - ``block4_backward_process_tile_pair``
     - 208
     - 38
     - the ``process_tile_pair`` lambda (FP32-only, rank==1 cross-tile
       pairing)

``disldo_backward`` itself, measured via ``lizard`` before/after each
extraction:

.. list-table::
   :header-rows: 1

   * - phase
     - NLOC
     - CCN
   * - before (post-``block4_codec_refactor``)
     - 1561
     - 257
   * - after struct wiring, lambdas untouched (7a)
     - 1561
     - 257
   * - after extracting ``process_single_row`` (7b)
     - --
     - 225
   * - after extracting ``process_row_pair`` (7c)
     - --
     - 204
   * - after extracting ``process_single_tile`` (7d)
     - 1072
     - 197
   * - after extracting ``process_tile_pair`` (7e, final)
     - 861
     - 160

``disldo_backward`` went from CCN 257 to CCN 160 (a 38% further
reduction on top of ``block4_codec_refactor``'s own 524 -> 257) without
changing any runtime behavior -- every phase's local + remote
regression pass matched exactly, and the two dequant-equality tests
plus the fp32 crosstile backward test's own dx/weight/importance
correctness check stayed bit-exact (``0.00000000``) throughout. CCN
160 is still above ``lizard``'s CCN>15 warning threshold -- the
remaining complexity is genuine per-call orchestration (byte-position
bookkeeping across a variable-length compressed tile stream, the
read-only-vs-writing branch, the pairing-vs-single dispatch) that does
not decompose further without either the ``std::vector`` scratch
redesign flagged above or changing the on-disk tile format itself,
both explicitly out of scope here. ``disldo_backward_sparse_grad``
(``sisldo_ops.hpp``, CCN 120) and ``disldo_forward`` (CCN 48) were
deliberately not touched in this pass -- offered as options and
declined in favor of shipping the one function actually flagged,
faster; revisit as a follow-up if wanted.

.. _disldo_backward.batch_stride_transpose:

``disldo_backward``: batch-major stride in the block4 backward inner loop
---------------------------------------------------------------------------

*ID:* ``disldo_backward.batch_stride_transpose``

sili-vs-torch benchmarking (fp32 dense 288x288, batch swept 1/4/16/64/256)
found sili's backward+update cost growing super-linearly with batch
(16x batch -> 35x slower) while torch's stayed nearly flat. Root cause:
``block4_backward_process_{single_row,row_pair,tile_pair}`` read
``input``/``output_grad`` (stored ``[batch, features]``, batch-major)
with a stride of ``in_cols``/``n_out`` floats inside the per-row/tile
batch-aggregation loop -- one 64-byte cache line fetched per sample, 4
bytes used, repeated once per (row, tile) visited (~72x redundancy at
288-wide). Fix: ``Block4BackwardParams::input_T``/``output_grad_T``
(see block4_codec.hpp), built once per ``disldo_backward`` call.
``output_grad_T`` is a BLOCK transpose to
``[ceil(n_out/BLOCK4_TILE), batch, BLOCK4_TILE]`` rather than a flat
transpose, so the existing 4-wide SIMD load per sample stays contiguous
*and* the batch loop becomes contiguous (stride 4 floats instead of
n_out) -- no tradeoff between the two access patterns. Zero-padded past
``n_out`` so boundary tiles read 0 without an extra bounds check.
Shared template code across ``VALUES_TYPE``, so the fix (and the two
new scratch fields added to all three ``Block4Store``/``Store8``/
``Store32`` structs) applies to fp8/fp4 too, not just fp32.

Verified via the full local ``ctest`` suite (163/163, including both
dequant-equality bit-exact gates) and the Python suite (215/215) --
numerically correct.

**Did not close the measured gap**: batch=256 backward+update cost was
unchanged (53.4ms before -> 54.9ms after). ``Block4BackwardAccumulators::
mdx`` (the dx accumulator write-back) has the identical batch-major
strided pattern at the same call frequency and was deliberately left
untouched in this pass -- leading hypothesis for why a numerically
verified fix produced no measured speedup is that it's now the
dominant remaining cost. Not yet confirmed; re-profiling or applying
the same block-transpose treatment to the dx accumulator is the
natural next step.
