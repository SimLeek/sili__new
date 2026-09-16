# TODO — batch-blocked accumulation + per-thread private buffer rollout

Tracking doc for a multi-kernel perf rollout, written to disk specifically
so it survives context compaction (in-chat checklists don't). Update the
checkboxes as each phase lands. See `docs/research/linear_disldo.rst` for
the eventual permanent writeup once a phase ships; this file is the
working queue, not the final record.

## Background (why this exists)

Profiled disldo_forward's block4 wide path at batch=1024 (callgrind,
num_cpus=1): the per-sample horizontal-reduce-and-scalar-store
(`mo[...] += prod[0]+prod[1]+prod[2]+prod[3]`) was ~74% of real compute,
vs ~4% for the actual useful SIMD multiply. Validated a fix via a
standalone PoC (not yet in the real kernel):

1. **Transpose input once per call** (shared, read-only across threads)
   so a block of batch samples for one row is a genuine contiguous wide
   load instead of a scalar gather.
2. **Register-blocked accumulate** (BLOCK_B=8 samples at a time), zero
   horizontal reduction until a whole block is done.
3. **Per-thread PRIVATE column-major accumulator buffer** (not a shared
   transposed output) -- avoids cross-thread contention on the real
   output array during the hot loop. Confirmed via a REAL multi-threaded
   (OpenMP, num_cpus=8) PoC that this matters a lot: single-threaded
   isolation only showed ~1.1-1.4x; real concurrent threads showed
   2.2-4.4x, with the private-buffer variant consistently ~10-20% ahead
   of a shared-transposed-buffer variant at every batch size.
4. **Batch-threshold switch**: below batch~8, the transpose overhead
   isn't amortized and the new path is SLOWER than today's code (0.1-0.6x
   at batch=1). Real crossover measured around batch=8; keep the existing
   scalar path below threshold.

PoC numbers (arch-sandbox, CCX-pinned, num_cpus=8, n=288):

| batch | current (shared) | blocked+transposed, shared | blocked+transposed, PRIVATE |
|---|---|---|---|
| 1    | baseline | 0.45x | 0.64x |
| 8    | baseline | 2.22x | 2.64x |
| 32   | baseline | 3.67x | 4.39x |
| 256  | baseline | 3.80x | 4.10x |
| 1024 | baseline | 3.80x | 4.14x |

PoC source: `/home/simleek/.claude/jobs/4c378ed3/tmp/poc_mt_contention.cpp`
(and the earlier single-threaded `poc_batch_blocked3.cpp`) -- not part of
the repo, scratch only.

## Phases

Each phase: implement, verify bit-exact correctness (C++ + Python suites,
local AND arch-sandbox), THEN run a real kernel benchmark (not the
isolated PoC) confirming the speedup actually lands across the batch
range -- do not consider a phase done on PoC evidence alone.

- [x] **Phase 1 -- disldo_forward, wide (column-partitioned) path, ALL
      precisions (fp32/fp8/fp4 -- see note below, this landed for free).**
      Transpose input (reuses `scratch_input_T`, already existed for
      backward), per-thread private column-major accumulator (fresh
      per-call, NOT yet persistent scratch -- see follow-up below),
      register-blocked accumulate via `block4_batch_accumulate`
      (`block4.hpp`), one flush pass per thread into real `output`,
      batch-threshold switch at `BLOCK4_BATCH_BLOCK_THRESHOLD=8`. Landed
      2026-09-15, `sili/lib/headers/block4.hpp` +
      `linear_disldo_forward.hpp`, new test
      `test_disldo_block4_fp32_batch_blocked.cpp`.

      **Free precision generality**: `process_block4_item` was already
      ONE generic implementation shared by fp32/fp8/fp4 (only the weight
      *decode* differs, via `Block4Codec<VALUES_TYPE>` -- decode always
      produces a plain float `Block4Vec`/`Block8Vec` regardless of storage
      precision). Phase 1's change only touches the FINAL accumulate step
      (after decode), so it applies to all three precisions automatically
      -- Phase 3 (originally planned separately) is therefore already
      done as a side effect, not separately implemented. Not
      independently re-benchmarked per-precision yet though -- see
      follow-up.

      **Real bug caught before shipping**: first implementation copied
      the isolated PoC's `blocked_pattern_TT` verbatim, including a
      per-lane broadcast-fill loop (`for(k) wv[k]=w4[li]`) INSIDE the
      per-batch-chunk hot loop -- this is the exact anti-pattern
      `block4_vec_broadcast`'s own docstring already warns about
      elsewhere in this file (defeats GCC's ability to keep a
      loop-invariant broadcast in a register; was 7.6% of a whole
      disldo_backward profile once before). First real-kernel A/B showed
      ZERO speedup (even regressions at large batch) until this was
      caught and fixed by hoisting the broadcast out of the loop. Lesson:
      a standalone PoC can validate the *algorithmic* idea while still
      containing a real perf bug of its own -- don't trust a PoC's
      exact code, only its shape, and always re-measure the REAL shipped
      kernel, not just re-run the PoC.

      **Measured result (real kernel, not the PoC) -- arch-sandbox,
      n_in=288/n_out=288, num_cpus=4 (`taskset -c 0-3`, 4 real physical
      cores, no SMT sharing), reproduced twice, <2% run-to-run variance**:

      | batch | before (ns/call) | after (ns/call) | speedup |
      |---|---|---|---|
      | 1    | ~105,000 | ~100,000 | ~1.05x |
      | 4    | ~110,500 | ~111,500 | ~1.00x (below threshold, unchanged code path) |
      | 8    | ~154,300 | ~103,100 | **~1.50x** |
      | 16   | ~158,300 | ~104,900 | **~1.51x** |
      | 32   | ~165,100 | ~111,600 | **~1.48x** |
      | 64   | ~178,200 | ~122,300 | **~1.46x** |
      | 256  | ~294,700 | ~275,100 | ~1.07x |
      | 1024 | ~945,000 | ~927,500 | ~1.02x |

      This is a REAL, reproducible, worthwhile win (~1.5x) in the
      batch=8-64 sweet spot, but well short of the isolated PoC's
      2.2-4.4x claim, and tapers to near-parity at batch>=256. Two
      follow-ups, one now RESOLVED:

      1. **RESOLVED 2026-09-16** (follow-up #1 from Phase 2's own
         buffer-allocation discovery, applied back here): `thread_buf`
         was exactly the same "fresh std::vector every call" issue as
         Phase 2's `b4_out`/`b4_out_T`, except WORSE -- it was allocated
         PER-THREAD, INSIDE the parallel region, meaning `num_cpus`
         threads were all hitting malloc concurrently on every call
         (allocator lock contention on top of the per-call cost itself).
         Fixed the same way: one shared persistent buffer
         (`scratch_thread_buf`, sized `n_out*batch`, not
         `num_cpus*n_out*batch` -- threads own disjoint contiguous
         column ranges that exactly tile `[0,n_out)`, so no per-thread
         copy is needed, just an offset into one buffer), zeroed
         per-thread inside the parallel region. This was THE actual
         explanation for "why does the gain shrink at batch>=256" --
         it wasn't shrinking due to any fundamental limit, it was being
         masked by this same allocation tax the whole time. Real-kernel
         A/B (arch-sandbox, num_cpus=4, n_in=n_out=288, 3 interleaved
         pairs, before = genuinely pre-Phase1/2 code via a pinned
         worktree, after = blocked-accumulate + this fix combined,
         <1% variance across repeats):

         | batch | before (ns/call) | after (ns/call) | speedup |
         |---|---|---|---|
         | 1    | 112,320 | 99,610  | 1.13x |
         | 4    | 115,610 | 110,590 | 1.05x |
         | 8    | 136,310-136,710 | 102,140-103,050 | **1.32-1.33x** |
         | 16   | 171,920 | 107,470 | **1.60x** |
         | 32   | 261,650-264,549 | 106,580-107,845 | **2.44-2.46x** |
         | 64   | 376,100 | 126,830 | **2.97x** |
         | 256  | 1,251,181-1,264,639 | 277,170-278,510 | **4.51-4.54x** |
         | 1024 | 4,718,426-4,738,903 | 929,330-945,903 | **5.01-5.10x** |

         The speedup now GROWS with batch instead of tapering off --
         exactly the shape you'd expect once the fixed per-call
         allocation tax stops eating a growing share of an otherwise-
         shrinking-per-sample cost. This is comfortably past the
         original isolated PoC's 2.2-4.4x claim at the largest batch
         sizes, not short of it.
      2. At num_cpus=8 with a wider layer (n_out=576, so the wide path
         still triggers), measured NO speedup at all (before==after
         within noise), confirmed with TWO different pinnings: first
         `taskset -c 0-3,8-11` (later found via `lscpu -e` to be an SMT
         sibling pair of the SAME 4 physical cores on this box, not 8
         independent ones -- logical CPU N and N+8 share a physical
         core here), then re-tested with `taskset -c 0-7` (8 genuinely
         distinct physical cores, confirmed via `lscpu -e`'s CORE
         column) -- SAME null result both times. Best explanation:
         `0-7` spans this box's two L3/CCX domains (cores 0-3 = L3
         domain 0, cores 4-7 = L3 domain 1 per `lscpu -e`), and
         num_cpus=8 falls squarely in the "worst" cross-CCX-tax bracket
         already documented in [[project_sili_arch_sandbox_ccx_topology_thread_wall]]
         (num_cpus 5-9 = worst on this exact box, for disldo_backward --
         apparently also true here). If that's right, the cross-CCX tax
         is likely dominating total wall time at num_cpus=8 regardless
         of which forward-path algorithm runs, making Phase 1's
         algorithmic change genuinely invisible at that specific thread
         count on this box -- NOT evidence the private-buffer idea stops
         working at higher thread counts. Re-test at num_cpus=10-16 (the
         documented "recovers" bracket) before drawing any conclusion
         about scaling beyond 4 threads; not done yet.

         **Re-tested 2026-09-16 with the persistent-buffer fix**: no
         longer null -- before=3.27-3.43M ns/call, after=418-432K
         ns/call at batch=256 (before=11.4-11.7M, after=1.25-1.30M at
         batch=1024), reproduced twice. BUT the "before" absolute
         number here is wildly inconsistent with an EARLIER same-shape
         measurement taken previously in this same session
         (516,560-516,940 ns/call at batch=256 for the un-fixed code,
         6-7x faster than this run's 3.27-3.43M) -- `uptime` showed
         `load average: 3.36, 2.07, 1.32` at measurement time with no
         obvious live CPU hog in `ps aux`, suggesting recent transient
         system load (possibly leftover from this session's own earlier
         heavy valgrind/compile activity) rather than a clean
         measurement. The "after" number, by contrast, was stable and
         consistent with expectations both times. Plausible (not
         confirmed) explanation: the OLD code's per-thread CONCURRENT
         malloc is a lock-contention hotspot that gets disproportionately
         worse under any scheduling noise (a thread preempted while
         holding/waiting on the allocator lock creates a convoy effect),
         while the fixed code has no such hotspot and stays more robust
         to background system noise -- if true, that's a real point in
         the fix's favor beyond the raw speedup, but NOT confirmed via a
         controlled clean-machine re-run. Directionally the fix helps
         here too either way (never measured worse); the num_cpus=5-9
         CCX-tax bracket theory from above is not ruled out either --
         could be compounding, not competing, explanations. Re-test on
         an idle machine (and at num_cpus=10-16) before trusting the
         precise ratio here.
- [x] **Phase 2 -- disldo_forward, narrow (tree-reduction) path, FP32
      (+fp8/fp4 for free, same reasoning as Phase 1/3).** Landed
      2026-09-16. Reused the SAME `process_block4_item`/
      `block4_batch_accumulate` machinery: each thread's private buffer
      (already existed, `b4_out`) becomes column-major [n_out, batch]
      instead of row-major [batch, n_out] at/above threshold, the
      existing tree reduction is reused UNCHANGED (elementwise add is
      layout-agnostic), only the final flush into `output` gains a
      transpose instead of a straight copy. Correctness verified same
      way as Phase 1 (below/at/above threshold, boundary dims).

      **Real bug caught while writing the correctness test**: the
      original Phase 1 test file (`test_disldo_block4_fp32_batch_blocked.cpp`)
      claimed in its own comments to exercise the WIDE path, but the
      arithmetic (`cols_per_thread = ceil(n_out/4)/num_cpus`) actually
      worked out to the NARROW path in every one of its cases -- so
      Phase 1's dedicated test never actually tested Phase 1's own code;
      it was accidentally covered only by a pre-existing, differently-
      purposed test (`test_disldo_block4_fp32_wide_simd.cpp`). Fixed
      while implementing Phase 2 by checking the real formula and adding
      genuinely-wide cases (`n_out=200/256`, `num_cpus=4` ->
      `cols_per_thread=12/16`) alongside genuinely-narrow ones, both
      spanning the threshold boundary. Lesson: a comment asserting which
      code path a test exercises is a claim, not a fact -- verify it
      against the actual dispatch condition, don't just eyeball the
      shape.

      **Measured result: NO measurable speedup**, at every combination
      tried (arch-sandbox, reproduced twice each, <2% variance):
      - n_in=n_out=288, num_cpus=8 (narrow, cols_per_thread=9): flat
        within noise at batch=8-256; a modest ~1.05x only at batch=1024
        (1,709,257 -> 1,628,829 ns). Also num_cpus=8 = this box's known
        cross-CCX "worst zone" (see Phase 1's follow-up #2), so this
        number is suspect anyway.
      - n_in=n_out=128, num_cpus=4 (narrow, cols_per_thread=8 -- the
        SAME core count where Phase 1's wide path got a clean 1.5x):
        before/after within ~1% at every batch from 1 to 1024. Genuinely
        null, not a CCX artifact this time.

      **UPDATE 2026-09-16, three follow-up rounds, third one landed a
      real win:**

      1. *Barrier-count reduction* (user's suggestion: `#pragma omp for`
         has an implicit barrier by default, but it isn't semantically
         required by this reduction -- a tree reduction combines buffer
         copies at a FIXED index, never across indices, so if one
         thread owns a static column range for the whole
         reduction+flush (computed once via `omp_get_thread_num()`,
         GPU-`global_id`-style, not redispatched via a fresh
         `#pragma omp for` every round), it needs zero barriers between
         rounds -- only ONE barrier is genuinely load-bearing (between
         item-processing and the start of reduction, since a thread's
         owned columns may hold data written by ANY other thread's
         items). Implemented, cut 4 barrier crossings/call down to 1.
         Introduced and then caught+fixed a real bug of its own along
         the way: the first version nested column/batch OUTSIDE the
         stride/round loop, meaning every single (col,b) element jumped
         between up to `num_cpus` buffer copies `ost` elements (hundreds
         of KB) apart on EVERY iteration instead of doing one
         sequential sweep per round -- measured 2-13% SLOWER than
         no-fix-at-all until the loop nesting was corrected back to
         stride-outermost. Once fixed: STILL no measurable win, before/
         after within noise (4 interleaved pairs, arch-sandbox,
         num_cpus=4/n_out=128/batch=1024: ~314-333k both before and
         after, no consistent direction). Real lesson from this round:
         callgrind's instruction-count share (52% "barrier-wait") does
         NOT mean 52% of real wall-clock time -- a tight spin-wait loop
         is instruction-DENSE but cheap-per-instruction (no memory
         stalls), so a large % of Ir doesn't translate proportionally
         to a large % of real time. Kept anyway (correctness-verified,
         objectively fewer synchronization points, never worse) but
         did NOT explain the null result on its own.
      2. *Buffer-allocation hypothesis* (also user-prompted): `b4_out`/
         `b4_out_T` were a FRESH `std::vector<value_type>(size, 0)`
         constructed on EVERY call -- unlike `input_T`
         (`scratch_input_T`, persistent scratch, no realloc once warm)
         these paid a real malloc plus a SINGLE-THREADED zero-fill of a
         multi-MB region (`num_cpus * batch * n_out` floats -- 2MB at
         batch=1024/n_out=128/num_cpus=4) BEFORE the parallel region
         even opened. Fixed: made `scratch_b4_out` a persistent member
         (`Block4Store`/`Store8`/`Store32`, resized not reallocated) and
         moved the zero-fill to per-thread, INSIDE the parallel region
         (each thread zeros only its own slice, in parallel, instead of
         one thread zeroing everything serially up front).
      3. **This one worked.** Real-kernel A/B (arch-sandbox, num_cpus=4,
         n_in=n_out=128, 4 interleaved before/after pairs, all three
         fixes combined -- blocked-accumulate + barrier-reduction +
         persistent-buffer):

         | batch | before (ns/call) | after (ns/call) | speedup |
         |---|---|---|---|
         | 1    | ~20,430 | ~19,740 | ~1.03x |
         | 8    | ~24,020 | ~23,130 | ~1.04x |
         | 16   | ~24,720 | ~25,470 | ~0.97x (noise) |
         | 32   | ~26,830 | ~26,020 | ~1.03x |
         | 64   | ~35,190 | ~35,610 | ~0.99x (noise) |
         | 256  | ~84,640-101,990 | ~76,130-81,170 | **~1.10-1.33x** |
         | 1024 | ~317,220-381,670 | ~287,410-304,070 | **~1.09-1.33x** |

         Consistent across all 4 repeats, biggest win exactly where the
         buffer is largest (256/1024) -- matches the "allocation+zero
         cost scales with buffer size" story cleanly. Confirms the
         buffer-allocation hypothesis was the real (or at least the
         dominant remaining) bottleneck, not accumulate compute (round
         1) and not barrier count (round 2) alone.

      **Resolution**: the original "why did Phase 1 win but Phase 2
      not?" question (both use the same blocked-accumulate primitive,
      both already avoided cross-thread contention on their private
      buffer) is answered by the above -- it was never about the
      accumulate primitive or contention at all. The wide path's
      thread_buf happened to be small (owned column range only) AND
      Phase 1 never profiled/fixed its own allocation cost, so that
      confound never got exposed there. The narrow path's buffer covers
      the FULL n_out range (much bigger), which is exactly what made its
      allocation+zero cost large enough to swamp the real accumulate-step
      win until fixed. Concretely: Phase 1's wide-path `thread_buf` has
      this SAME "fresh std::vector every call" issue (flagged as an open
      follow-up in Phase 1's own writeup above) -- worth applying the
      identical persistent-scratch treatment there too, now that it's
      confirmed to matter this much. Not yet done.
- [x] **Phase 3 -- disldo_forward FP8 + FP4 (precision parity).** Landed
      for free as part of Phase 1 -- see Phase 1's "Free precision
      generality" note above. Still not independently re-benchmarked
      per-precision (only fp32 has a real-kernel A/B number so far); a
      quick fp8/fp4 real-kernel bench is a cheap thing to add whenever
      convenient, not urgent since the code path is provably identical
      past weight decode.
- [x] **Phase 4 -- disldo_forward's scattered (non-block4, `!dc.empty()`)
      path.** Landed 2026-09-16. Applied the two ALREADY-PROVEN fixes
      (persistent scratch buffer instead of fresh-per-call
      `std::vector(size,0)`; static-per-thread-range parallel tree
      reduction instead of the fully SERIAL one-thread-sums-everything
      loop this path still had -- exactly the pattern
      disldo_forward.per_thread_output_buffers already diagnosed and
      fixed for block4, just never ported here). New scratch member
      `Block4Store::scratch_scattered_out`.

      Real-kernel A/B (arch-sandbox, num_cpus=4, n_in=n_out=288,
      10%-density pure-scattered layer via
      `scripts/bench_disldo_forward_scattered.cpp`, 2 interleaved
      pairs, full batch range): **before == after within noise at every
      batch size, 1 through 1024.** A real, clean negative result, not a
      bug -- correctness fully verified (same 5 pre-existing unrelated
      ctest failures, nothing new), the fix is real and harmless, it
      just doesn't matter here. Best explanation: this path's compute is
      already the dominant cost and it's expensive for an unrelated
      reason -- confirmed by comparing totals directly: at batch=1024
      this scattered layer (nnz=8064, so 8.25M total multiply-adds) took
      ~2.25ms, more than DOUBLE the fully-dense block4 case's ~0.93ms
      (82944 synapses, 84.9M multiply-adds -- 10x more actual work,
      less than half the time). Matches this project's own established
      understanding (see
      [[project_sili_optimal_hardware_vision]]: "scattered CSR is
      gather/scatter-bound not SIMD-bound") -- the scalar, strided
      per-(row,col,batch) access pattern here is the real bottleneck,
      not allocation, and allocation-tax fixes (which worked great for
      block4's cheap SIMD-efficient compute) are noise against it. A
      genuine fix for this path would need something more like block4's
      own transpose+blocked-accumulate treatment, adapted for arbitrary
      (non-4-wide-tile) CSR column patterns -- a bigger, more speculative
      redesign, not attempted here; flagged as a possible future
      direction, not started.
- [x] **Phase 5 -- sisldo_forward.** Landed 2026-09-16. Investigated
      first (per the note this replaces): sisldo's structure is entirely
      different from disldo's (row-major gather with a per-batch cursor,
      not a fixed block4 tile SIMD shape), so the transpose+blocked-
      accumulate idea doesn't apply here at all. But `sisldo_ops.hpp`'s
      `sisldo_forward` turned out to have the SAME two allocation/
      reduction issues as disldo's scattered path did before Phase 4,
      in BOTH of its two branches:
      - Scattered branch: `all_outputs`/`all_contributions` were fresh
        `std::vector`s every call, but the final reduction was ALREADY
        a parallel tree (raw `#pragma omp barrier` + half-the-threads-
        each-round, not `#pragma omp for`) -- only the buffer-allocation
        half of the fix applied here.
      - Block4 branch: `all_b4_outputs` was ALSO fresh every call, AND
        the final reduction was a fully SERIAL one-thread-sums-
        everything loop (same as disldo's old scattered path) -- both
        halves of the fix applied. New scratch members
        `Block4Store::scratch_sisldo_out/_contrib/_b4_out`.

      Real-kernel A/B (arch-sandbox, num_cpus=4, n_in=n_out=288, 2
      interleaved pairs each):
      - **Block4 branch** (10% input density, fully-dense block4
        weights, `scripts/bench_sisldo_forward_block4.cpp`): real,
        consistent win -- **~1.1-1.2x at batch=1-8**, growing to
        **~1.5-1.8x at batch=32-256**, **~1.35-1.5x at batch=1024**.
        Same shape of win as disldo_forward's block4 paths, for the same
        reason (this branch decodes real block4 tiles, dense compute,
        allocation was a real visible cost once the reduction stopped
        being the bottleneck).
      - **Scattered branch** (10%/10% weight/input density,
        `scripts/bench_sisldo_forward_scattered.cpp`, `block4_tiles=0`
        confirmed): before == after within noise at every batch size,
        1 through 1024 -- expected and confirmed, not just assumed: the
        reduction here was already parallel, so only the (apparently
        negligible on its own) buffer-allocation half of the fix
        applied, and this branch is likely compute-bound the same way
        disldo's scattered path is (gather-based, scalar).

      Correctness: full local+remote ctest (including existing sisldo/
      disldo parity and block4-sparse-input-forward tests), same 5
      pre-existing unrelated failures as baseline, nothing new.
- [x] **Phase 6 -- disldo_backward.** Landed 2026-09-16. Audited first
      (this function is CCN 257, huge, and already the subject of a
      SEPARATE active CCN-reduction refactoring plan -- did not touch
      its control flow, only buffer lifecycle):
      - **Main per-thread buffers already fixed, long before this
        session.** `t_dx`/`t_col_grad`/`t_col_grad_contrib`/
        `t_gamma_grad`/`t_gamma_grad_contrib` already live in a
        dedicated persistent `weights.disldo_backward_scratch`
        (`DisldoBackwardScratch`, `delta_csr_types.hpp`) with a
        grow-only `ensure()`/`resize_to()` and per-thread parallel
        zero-fill via `std::fill_n` -- exactly this rollout's pattern,
        already done.
      - **The dx horizontal-reduce fix was explicitly TRIED and
        REJECTED already, too.** `disldo_backward.batch_stride_transpose`
        in `docs/research/linear_disldo.rst` documents that literally
        Phase 1's core idea (transposing the `mdx` write-back to avoid a
        per-sample `block4_vec_hsum`) was tried here first and made
        things WORSE (54.9ms -> 63.2ms at batch=256) -- rejected as
        "pure overhead with no offsetting benefit." The REAL fix that
        actually worked (`disldo_backward.batch_hsum_deferral`) was a
        different, more powerful technique: an ALGEBRAIC reassociation
        (`sum_b(hsum(C*x_b)) == hsum(C*sum_b(x_b))`) that defers the
        weight-gradient hsum to ONCE per tile instead of once per
        sample -- something dx itself can't use (dx needs a genuinely
        distinct value per sample, not an aggregate), but which beat a
        literal transpose+block-accumulate attempt for everything that
        COULD use it. Already landed this gap from ~25-30x slower than
        torch down to 3.1-3.5x. Given this rollout's core idea was
        already tried here and lost to a better technique, re-attempting
        it now would just rediscover the same rejected result -- did
        not re-attempt.
      - **One real gap found**: `group_dx`/`group_col_grad`/
        `group_col_grad_contrib` (the CCX-group-aware reduction's own
        buffers, `disldo_backward.ccx_aware_reduction`) were STILL a
        fresh `std::vector` every call, unlike the main t_dx/t_col_grad
        buffers right next to them. Fixed the same way: added to
        `DisldoBackwardScratch` (new `group_dx`/`group_col_grad`/
        `group_col_grad_contrib` fields + `cap_groups`, `ensure()`/
        `resize_to()` extended to take a `groups` param), addressed via
        the same `cap_dst`/`cap_out_rank` strides the thread buffers
        already use (not this call's own possibly-smaller `dst`/
        `n_out*rank`, for the same historical-max-aliasing-safety
        reason), zero-filled per-group inside the parallel region.

        **Correctness verified with extra care** (this is a
        stride-indexing change in gradient-accumulation code, and the
        `num_groups>1` branch only triggers on real dual-CCX hardware
        at high thread counts -- easy to ship an indexing bug that
        never gets exercised by the local/default test suite): wrote a
        dedicated probe confirming `num_groups` is ACTUALLY 2 at
        num_cpus=8/16 on arch-sandbox (not just plausible-by-topology --
        directly measured via `sili_topology::current_thread_group()`
        inside a real parallel region), then a separate correctness
        verifier (`scripts/verify_disldo_backward_group_reduction.cpp`)
        comparing `sum(|input_grad|)` across num_cpus=1/4/8/16 with an
        IDENTICAL weight seed -- bit-exact match at every thread count,
        confirming the `num_groups=2` branch specifically produces the
        same answer as the trivial `num_groups=1` path. Full local+
        remote ctest also clean (same 5 pre-existing unrelated
        failures).

        **Real-kernel A/B** (`scripts/bench_disldo_backward_group_reduction.cpp`,
        num_cpus=8 with `OMP_PROC_BIND=true OMP_PLACES=cores` so
        `num_groups=2` actually triggers, 2 repeats): small, modest,
        somewhat noisy win as expected for a small batch-INdependent
        buffer -- roughly **~3-16%** depending on the run (batch=256:
        1.03x-1.13x; batch=1024: 1.03x-1.16x), nowhere near the
        batch-scaling buffers' 2-5x wins elsewhere in this rollout, but
        real and never measured worse.
- [x] **Phase 7 -- sisldo backward (`disldo_backward_sparse_grad`,
      sisldo_ops.hpp).** Landed 2026-09-16. Unlike disldo_backward
      (Phase 6), this function's t_col_grad/t_col_grad_contrib/
      t_gamma_grad/t_gamma_grad_contrib had NOT already received the
      persistent-buffer treatment -- still a fresh `std::vector` every
      call. Same shape as Phase 6's group buffers (small,
      batch-INdependent, out_cols/rank-scaled) so the SAME expectation
      applied: modest win, not the 2-5x batch-scaling wins elsewhere.

      One real subtlety this function has that disldo_backward's
      group buffers didn't: TWO separate `#pragma omp parallel for`
      regions (scattered accumulate, then block4 accumulate) both
      accumulate (`+=`, via `mcol_at`/`mgamma_at` lambdas) into the
      SAME `t_col_grad`/`t_gamma_grad` buffers -- so the zero-fill has
      to happen exactly ONCE, before EITHER region runs, not per-thread
      inside one region (which would wipe the other region's
      contribution). Used `.assign()` (serial, but the buffer is small
      enough that this is cheap) rather than the per-thread-parallel
      fill pattern used for the large batch-scaling buffers elsewhere.
      New scratch members `Block4Store::scratch_sisldo_bwd_col_grad(
      _contrib)/scratch_sisldo_bwd_gamma_grad(_contrib)` -- kept
      separate from disldo_backward's own scratch struct deliberately,
      to avoid coupling the two functions' scratch lifetimes given this
      file's recent heap-corruption history.

      **Correctness verified with the SAME extra care as Phase 6**,
      given the heap-corruption bug fixed earlier this session was in
      this exact function: full local+remote ctest (same 5 pre-existing
      unrelated failures, nothing new), the dedicated
      `test_sisldo_sparse_load_backward_stability` regression test
      (500 repeated calls, the test that originally caught the
      heap-corruption bug) passing on both, AND re-run under
      AddressSanitizer (`LD_PRELOAD=libasan.so`) locally -- clean, no
      memory errors.

      **Real-kernel A/B** (`scripts/bench_sisldo_backward_sparse_grad.cpp`,
      num_cpus=4, dense dy, 2 interleaved pairs): modest win at small/
      medium batch (**batch=1: ~1.02-1.10x; batch=8: ~1.02-1.14x;
      batch=64: ~1.06-1.08x**), essentially flat/negligible by
      batch=256-1024 (~1.00x both times) -- at that point the call
      takes 10-39ms and is dominated by real per-synapse/per-batch
      compute, so the small fixed buffer-allocation saving becomes
      negligible in comparison. Consistent with and slightly more
      pronounced than Phase 6's finding for the same class of buffer.
- [ ] **Phase 7.5 -- scattered-path transpose+blocked-accumulate.**
      Flagged by the user 2026-09-16, right after Phase 4's negative
      result: Phase 4's persistent-buffer fix didn't move the scattered
      (non-block4) path's needle because allocation was never the real
      bottleneck there -- this path is scalar, strided,
      gather/scatter-bound (see the scattered-path note at the end of
      disldo_forward.persistent_scratch_buffers in
      docs/research/linear_disldo.rst). A genuine fix would need
      something like block4's own transpose+register-blocked-accumulate
      treatment (disldo_forward.batch_blocked_threshold), adapted for
      ARBITRARY (non-4-wide-tile) CSR column patterns -- e.g. transpose
      input to `[row, batch]` same as block4 does, but the accumulate
      side can't use a fixed BLOCK4_TILE=4 stride since a CSR row's
      columns aren't grouped into tiles; would need per-row batch-
      blocking into a column-indexed (not tile-indexed) accumulator, or
      some other structure -- not designed yet, just flagged. Explicitly
      placed HERE (after Phase 7, right before Phase 1.5's Python check)
      per the user's own instruction: "write that down for a later check
      to come back to right before the python side checking."
- [ ] **Phase 1.5 -- check the PYTHON side, not just the C++ real-kernel
      bench.** Flagged 2026-09-15: all of Phase 1's verification so far
      (unit tests + `scripts/bench_disldo_forward_kernel.cpp`) is pure
      C++, but the actual perf-comparison-vs-torch numbers this whole
      investigation reports to the user live in
      `scripts/bench_sili_vs_torch_matrix.py`, which goes through the
      pybind11 bindings (`sili/cpu_backend.cpp` -> `DISLDOLayerV`), not
      `disldo_forward` directly. Needs checking, not assumed:
      1. Does `DISLDOLayerV`'s forward binding call `disldo_forward` with
         the SAME `num_cpus` value the C++ bench used (4, the measured
         sweet spot on arch-sandbox), or a different default that could
         land in the num_cpus=5-9 dead zone or below the batch=8
         threshold for some of the bench matrix's rows?
      2. Re-run `scripts/bench_sili_vs_torch_matrix.py` on arch-sandbox
         (before vs after this whole rollout, same worktree-diff
         technique as the C++ A/Bs) across its existing batch sweep
         (1/32/64/128/256/512/1024) and confirm the C++-bench wins
         actually show up end-to-end through Python, not just in the
         isolated kernel call.
      3. If it DOESN'T show up proportionally, that's a real finding
         (Python/pybind overhead dominating, or GIL/dispatch overhead
         swamping the C++-side win at these batch sizes) worth its own
         investigation before claiming this rollout "done" end-to-end --
         the C++ kernel wins are necessary but not sufficient for the
         thing the user actually cares about (real torch-comparison
         numbers).

## Also queued (unrelated to this rollout, don't lose these either)

- Sweep input density more finely (0.5, 0.4, 0.3, 0.2, 0.1 between the
  already-tested 0.05 and 0.5) to find the actual sisldo/disldo crossover
  density for automatic engine switching, rather than just bracketing it
  between 0.05 (sisldo wins) and 0.5 (disldo wins).
- User raised "DIDLDO" (dense input, dense linear/weights, dense output --
  this is just standard dense matmul/BLAS, what torch already does) and
  "SIDLDO" (sparse input, DENSE linear/weights, dense output -- doesn't
  exist yet) as two more points on the same 2x2 grid as DISLDO/SISLDO.
  Check whether SIDLDO would meaningfully beat SISLDO when weights happen
  to be dense but input is sparse -- SISLDO currently pays block4/CSR
  weight-lookup overhead designed for SPARSE weights even when the
  weights are actually fully dense, which may be a real, avoidable chunk
  of why sisldo underperforms at high input density in the current
  benchmark matrix.
