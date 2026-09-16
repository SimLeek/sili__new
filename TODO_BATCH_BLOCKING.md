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
- [x] **Phase 7.5 -- scattered-path transpose+blocked-accumulate.**
      Landed 2026-09-16. Flagged by the user right after Phase 4's
      negative result, with an explicit request to profile FIRST before
      implementing (rather than trust the Phase 4 writeup's causal
      story). Good call: that story turned out to be WRONG.

      **Profiled first, and the earlier "gather/scatter-bound" claim
      didn't hold up.** callgrind on the real scattered path (num_cpus=1,
      batch=1024, 10% density, so no GOMP artifact) showed the
      per-batch-sample accumulate loop itself -- `for(b) mo[...]+=w*iv`
      -- was **~92% of the function's real instructions** (57.8% loop
      overhead + 34.6% the strided store), with CSR
      traversal/decode (`row_cursor`, `cursor.advance()`, weight/scale
      lookup) a negligible ~0.02%. Phase 4's buffer-allocation fix
      genuinely didn't help (that was never the bottleneck), but the
      explanation attached to that finding was incomplete -- the real
      story is this loop is STRIDED on both sides (`input[b*in_cols+r]`
      and `mo[b*n_out+col]`), structurally identical to block4's
      ORIGINAL pre-Phase-1 problem, just with one weight instead of
      four. Lesson: a plausible-sounding explanation for a negative
      result still needs its own verification before being trusted for
      the next decision -- it wasn't.

      **Implementation**: added `scalar_batch_accumulate` (block4.hpp,
      single-weight sibling of `block4_batch_accumulate`) and gated the
      scattered path on the SAME `BLOCK4_BATCH_BLOCK_THRESHOLD` (hoisted
      to the top of `disldo_forward`, shared with the block4 dispatch
      below it). At/above threshold: transpose input to `[row, batch]`
      (reuses `scratch_input_T`, same buffer the block4 paths use --
      accepted minor redundancy if BOTH scattered and block4 content
      exist on the same layer/call, since this section runs first and
      unconditionally rebuilds it; not a big deal at O(n_in*batch), not
      restructured further to avoid touching the block4 section's
      control flow), and accumulate into `scratch_scattered_out`
      reinterpreted as column-major `[n_out, batch]` per thread (dual
      row-major/column-major reuse of one field, same pattern as Phase
      2's `scratch_b4_out`/`scratch_b4_out_T`). Reused the existing
      static-per-thread-range reduction unchanged (layout-agnostic); the
      final flush maps flat index -> (col, b) via plain div/mod rather
      than forcing column-aligned thread ranges, since alignment isn't
      needed for correctness and the added complexity wasn't worth it
      for a flush that's a small fraction of total work.

      **Correctness**: new `test_disldo_scattered_batch_blocked.cpp` (8
      cases, below/at/above threshold, remainder-tail batches, very
      sparse/near-dense rows, non-dividing num_cpus, verified
      `block4_tiles==0` so it only ever exercises this phase's code)
      against an independent dense-matmul reference. Full local+remote
      ctest clean (same 5 pre-existing unrelated failures).

      **Real-kernel A/B** (`scripts/bench_disldo_forward_scattered.cpp`,
      num_cpus=4, 10% density, 2 interleaved pairs, arch-sandbox): clean
      and highly consistent across both repeats --

      | batch | speedup |
      |---|---|
      | 8    | ~1.08x |
      | 16   | ~1.17-1.27x |
      | 32   | ~1.70-1.71x |
      | 64   | ~1.99x |
      | 256  | ~2.71-2.72x |
      | 1024 | **~3.22x** |

      Same growing-with-batch shape as the wide path's own Phase 1
      persistent-buffer fix, for the same reason (a fixed-shape cost
      that used to dominate shrinks as a share of an otherwise-scaling
      workload once removed). One of the largest wins in this whole
      rollout, on a path that looked like a dead end two phases ago.
- [x] **Phase 1.5 -- check the PYTHON side, not just the C++ real-kernel
      bench.** Landed 2026-09-16. Re-ran the full matrix
      (`scripts/bench_sili_vs_torch_matrix.py`, no CPU pinning, torch's
      own default thread count -- 8 on arch-sandbox) across 336 cells
      (2 layer widths x 2 synapse regimes x 4 input densities x 7
      batches x 3 ops), published to the "Block4 Bench" artifact.

      User then asked the right follow-up directly: is this Python-level
      timing (pybind11-wrapped calls) actually trustworthy, or could
      wrapping overhead be confounding the disldo/sisldo/torch
      comparison, especially at small batch where fixed overhead matters
      most? Checked directly rather than assumed: built a matching pure-
      C++ real-kernel bench at the SAME num_cpus=8, same shape
      (n=288, dense), and compared ns/call to the Python-measured ms/call
      for both disldo and sisldo forward across the full batch range:

      | batch | disldo python/C++ | sisldo python/C++ |
      |---|---|---|
      | 1    | 0.88x | 0.90x |
      | 32   | 1.02x | 0.90x |
      | 64   | 1.12x | 0.91x |
      | 256  | 1.15x | 1.05x |
      | 1024 | 1.14x | 1.09x |

      Tight band around 1.0x across 3 orders of magnitude of batch, for
      BOTH engines -- the signature of a large FIXED per-call overhead
      would be a big ratio at batch=1 shrinking toward 1.0x as batch
      grows; that's not what happens here. Conclusion: pybind11
      wrapping overhead is small (<=15%) and roughly proportional, not a
      meaningful confound on the disldo-vs-sisldo-vs-torch ratios
      reported by the bench matrix. Checked one representative slice
      (dense, 288-wide, full batch sweep) for both engines, not all 336
      cells, but a ratio staying flat across that whole batch range is a
      real pattern, not a coincidence specific to one point.

## Also queued (unrelated to this rollout, don't lose these either)

- Sweep input density more finely (0.5, 0.4, 0.3, 0.2, 0.1 between the
  already-tested 0.05 and 0.5) to find the actual sisldo/disldo crossover
  density for automatic engine switching, rather than just bracketing it
  between 0.05 (sisldo wins) and 0.5 (disldo wins).

- **DIDLDO / SIDLDO -- a second 2x2 grid, dense-weight-storage siblings
  of DISLDO/SISLDO.** Raised 2026-09-15/16, refined into a concrete
  implementation sketch after looking at the fresh Block4 Bench data
  (336-cell matrix, see `disldo_forward.python_side_verified` above).
  Naming follows the existing convention (`<input><linear><output>`,
  D=dense/S=sparse):

  - **DIDLDO** (dense input, dense linear/weights, dense output) --
    doesn't need new sili code at all, it's just a standard BLAS/MKL
    matmul + backprop + optimizer step. This is architecturally what
    torch's `nn.Linear` already does, so DIDLDO's "implementation" is
    really just "call BLAS directly instead of going through sili's
    CSR/block4 machinery for a layer that's fully dense anyway."
  - **SIDLDO** (sparse input, DENSE linear/weights, dense output) --
    doesn't exist yet. Forward (and contribution) would load only the
    ROWS of the dense weight matrix that correspond to nonzero positions
    in the input CSR; backward (and importance updates, and in-place
    optimizer step) would load only the COLUMNS corresponding to nonzero
    output-gradient positions. Because sparsity is on the ACTIVATIONS
    (input/grad) rather than the weights, most of the weight matrix's
    RAM genuinely never gets touched on a given call, proportional to
    input/grad density -- this is a different sparsity axis than
    DISLDO/SISLDO exploit (weight sparsity via CSR/block4) and should
    compose with it, not replace it.

  **Two-group interchangeability architecture** (user's key structural
  insight, worth preserving verbatim): DISLDO and SISLDO already share
  the SAME underlying weight storage (`weights.connections` / CSR +
  `weights.block4`), which is exactly why they're interchangeable
  per-call today -- you can run DISLDO forward then SISLDO backward (or
  vice versa) against the identical weight object with no conversion.
  DIDLDO and SIDLDO would likewise share a single DENSE weight storage
  representation, and would be interchangeable with EACH OTHER the same
  way. But the two groups are NOT interchangeable with each other
  without an actual weight-storage relayout (CSR/block4 <-> dense),
  which is real work and its own separate, longer-term speedup question
  (worth asking: could a layer hold BOTH representations simultaneously
  during a transition/relayout window, the way some sparse formats keep
  a COO staging buffer? Not decided, just flagging the question exists).
  So near-term engine switching is two independent binary choices, not
  one 4-way choice:
  - Group A (sparse/block4 weight storage): DISLDO <-> SISLDO, switch
    freely per forward/backward call based on input density that call.
  - Group B (dense weight storage, not yet built): DIDLDO <-> SIDLDO,
    same idea once SIDLDO exists.
  - Crossing groups (A->B or B->A) means physically relaying out the
    weights -- expensive, and only worth doing when a layer's weight
    density has genuinely shifted enough (e.g. after synaptogenesis/
    pruning) to justify it, not on a per-call basis.

  **Why this looks promising**: the fresh 336-cell bench matrix shows
  sili's best-case wins (up to ~5x vs torch) cluster at high batch AND
  high synapse density -- exactly the regime where DISLDO/SISLDO are
  paying CSR/block4 bookkeeping overhead for weights that are barely
  sparse at all. A dense-weight-storage engine (DIDLDO/SIDLDO) sidesteps
  that overhead entirely and should dominate in that corner of the
  parameter space; SISLDO's row/column-selective loading (SIDLDO) should
  still beat plain BLAS (DIDLDO) whenever the ACTIVATIONS are sparse
  enough, independent of weight density.

  **DIDLDO proof-of-concept: done, sanity check passed (2026-09-16).**
  Built `scripts/bench_didldo_kernel.cpp` -- plain forward GEMM via
  `cblas_sgemm`, calling numpy's own bundled OpenBLAS
  (`scipy.libs/libscipy_openblas-*.so`, real production BLAS, no new
  system dependency needed) directly from C++, and a single
  `didldo_backward(x, dy, w, square_avg, dx, dw_scratch, ..., lr)` call
  that always computes dx+dw via `cblas_sgemm` and gates an in-place,
  OpenMP-parallel RMSprop update on `lr != 0` -- matching the RMSprop
  torch itself uses in the comparison bench (alpha=0.99, eps=1e-8) and,
  more importantly, matching disldo_backward's own API shape: lr=0.0 IS
  the "grad-only" case, lr!=0 IS "grad+update", ONE function, no
  separate optimizer/"update" call ever exposed (first draft used a
  standalone `rmsprop_update_inplace` helper and plain SGD instead of
  RMSprop -- both corrected after review before trusting any number from
  this).

  Ran on arch-sandbox (same machine as the Block4 Bench data) at
  num_cpus=8, same n=288/1024 shapes and batch sweep, compared directly
  against the torch_ms already recorded in that dataset (torch's own
  `nn.Linear` is density-agnostic, so those numbers are valid to compare
  against regardless of synapse/input density). didldo/torch ratio,
  bwd0 = fwd+dx+dw (no update), bwdX = fwd+dx+dw+RMSprop step:

  | width | op | batch=1 | batch=32 | batch=256 | batch=1024 |
  |---|---|---|---|---|---|
  | 288  | fwd  | 3.01x | 1.41x | 1.80x | 1.43x |
  | 288  | bwd0 | 0.79x | 0.71x | 1.31x | 1.42x |
  | 288  | bwdX | 0.92x | 0.59x | 0.97x | 1.44x |
  | 1024 | fwd  | 4.34x | 2.48x | 1.58x | 1.36x |
  | 1024 | bwd0 | 1.24x | 1.36x | 1.06x | 0.99x |
  | 1024 | bwdX | 1.39x | 1.32x | 1.06x | 0.95x |

  **Root-caused the batch=1 outliers instead of hand-waving them (user
  pushed back: "why is didldo fwd slow, it should just be a BLAS call
  right?").** Two SEPARATE, real mechanisms, not one:

  1. **`sgemm` vs `sgemv` dispatch.** `didldo_forward`/`didldo_backward`'s
     dx step called `cblas_sgemm` unconditionally, even at batch=1 where
     M=1 -- a pure GEMV shape. Direct A/B
     (`scripts/probe_didldo_sgemv.cpp`): `sgemm` pays real fixed blocked/
     packed-GEMM setup cost a dedicated `sgemv` call skips -- 5.6x slower
     at n=288 (0.0209ms vs 0.0037ms), 2.4x at n=1024 (0.0996ms vs
     0.0417ms). Fixed by dispatching to `sgemv` at batch==1 for both
     forward and dx (dw's batch=1 case is a K=1 outer product, an `sger`
     shape -- noted, not applied, since it needs an explicit zero-fill
     first and wasn't the reported problem). This alone explains most of
     n=288's improvement and confirms numpy's own `x @ w` almost
     certainly makes this same shape-dependent dispatch internally
     (a follow-up probe showed numpy(openblas) BEATING torch(mkl) at
     batch=1 despite using "the slower" BLAS library -- shape dispatch
     mattered far more than which BLAS).
  2. **Cross-thread-pool contention between the RMSprop OpenMP region and
     OpenBLAS's own internal threading -- bigger, and still open.**
     Even after the sgemv fix, n=1024 batch=1 forward stayed stuck around
     0.23-0.27ms (vs an isolated sgemv's 0.035-0.04ms) no matter what else
     was tried (removing a global `omp_set_num_threads` call, testing
     `-fopenmp` linkage alone, testing one prior `sgemm`/`sger`-shaped
     call alone -- none reproduced it standalone). Root-caused by direct
     per-rep instrumentation: it's specifically the ALTERNATION -- this
     bench's timed loop does forward, then backward+RMSprop-update, then
     forward again, every iteration (which is also exactly what a real
     training loop does). A minimal repro
     (`fwd()` timed immediately after `#pragma omp parallel for` RMSprop
     region, repeated) reproduced it cleanly and consistently: ~0.29ms
     every single time, vs ~0.035ms with no RMSprop region in between.
     `OMP_WAIT_POLICY=PASSIVE` cut it to ~0.17-0.22ms but did not close
     it. Root cause, confirmed (not just theorized) via `nm -D`/`readelf`
     on the actual `.so`: this specific OpenBLAS build (numpy's bundled
     `scipy.libs/libscipy_openblas-*.so`) uses OpenBLAS's own PTHREADS
     threading backend -- `pthread_create`, `blas_server_avail`,
     `blas_thread_init`, zero `GOMP_*`/`omp_get_*` symbols, no libgomp
     dependency in `readelf -d` at all. So this is two COMPLETELY
     SEPARATE, uncoordinated thread-pool runtimes (my libgomp team, its
     own persistent pthread worker pool) contending for the same cores
     -- the textbook "don't mix threading runtimes" HPC anti-pattern,
     not a subtler libgomp-internal effect.

     User asked directly: is there no BLAS that uses OpenMP (or
     something like it), so the two pools could coexist? Yes -- OpenBLAS
     itself supports an OpenMP build (`USE_OPENMP=1`, just not the
     specific wheel-bundled copy here), and MKL (what torch actually
     uses) explicitly supports a selectable threading layer, including a
     GNU/libgomp one (`libmkl_gnu_thread.so`, present locally via
     oneAPI). Tested directly, not assumed: copied the needed MKL `.so`s
     (~200MB: `libmkl_rt`, `libmkl_gnu_thread`, `libmkl_core`,
     `libmkl_def`, `libmkl_avx2`, `libmkl_intel_lp64`) to arch-sandbox
     and reran the SAME alternating-call probe with
     `MKL_THREADING_LAYER=GNU` forcing MKL's GEMV calls through libgomp,
     the same pool my own `#pragma omp` region uses:

     | config | isolated sgemv | after-RMSprop-region sgemv | penalty |
     |---|---|---|---|
     | OpenBLAS (pthreads) | ~0.035-0.04ms | ~0.29ms | ~7.3-8.3x |
     | MKL (GNU/libgomp layer) | ~0.08ms | ~0.17ms | ~2.1x |

     Confirms the theory: sharing ONE thread-pool runtime cuts the
     alternation penalty by ~3.5x (relative to each config's own
     isolated baseline), even though MKL's GNU-layer isolated throughput
     for this tiny GEMV happens to be worse than OpenBLAS's pthreads
     path in absolute terms (GNU layer isn't MKL's fastest threading
     choice -- Intel's own iomp5 layer or the sequential layer would
     likely win outright, but neither shares libgomp with custom code).
     STILL NOT fully closed at that point (2.1x real, not noise) -- user
     directed the next step: route the RMSprop update through MKL too
     (VML: `vsMul`/`vsSqrt`/`vsDiv`/`vsLinearFrac`, plus `cblas_sscal`/
     `cblas_saxpy` for the scale/accumulate steps), eliminating the
     hand-rolled `#pragma omp` region entirely -- ALL backward work
     (dx/dw GEMM+GEMV and the RMSprop update) now goes through MKL,
     zero custom OpenMP anywhere. Tested directly: **no change** --
     still ~0.17ms, same 2.1x penalty as the custom-omp version. This
     disproved the "custom omp region vs BLAS's own region" framing --
     the penalty survives with literally zero custom omp code involved.

     Real cause, found by varying call COUNT, not library: the 7-call
     VML RMSprop sequence (`vsMul`, `sscal`, `saxpy`, `vsSqrt`,
     `vsLinearFrac`, `vsDiv`, `saxpy`) makes 7 separate parallel
     dispatches before the next `sgemv`. Replaced it with a single
     `vsSqrt` call (same element count, doing nothing useful, just
     testing dispatch count) between `fwd()` calls: **penalty
     vanished** -- 0.083ms, matching the isolated 0.0803ms baseline
     almost exactly. So the driver is DISPATCH COUNT (how many separate
     parallel regions get entered/torn down before the next call), not
     which library or runtime issues them. Confirmed progression:

     | config | penalty vs isolated |
     |---|---|
     | OpenBLAS(pthreads) + custom omp (cross-runtime) | ~7.3-8.3x |
     | MKL(GNU) + custom omp (same-runtime, ~2 dispatches) | ~2.1x |
     | MKL(GNU), all-MKL RMSprop (same-runtime, 7 dispatches) | ~2.1x |
     | MKL(GNU), single fused call (same-runtime, 1 dispatch) | ~1.0x |

     **Actionable conclusion for DIDLDO/SIDLDO's real implementation**:
     (1) use a BLAS built with the SAME OpenMP/libgomp backend sili's
     own kernels already use (MKL with `MKL_THREADING_LAYER=GNU`, not
     the numpy-wheel-bundled pthreads OpenBLAS) -- this alone kills the
     worst-case cross-runtime tax (~7-8x down to ~2x); AND (2) fuse the
     backward/update math into as FEW separate parallel-dispatching
     calls as possible, ideally one, not just "any BLAS call however
     many" -- a real RMSprop-via-VML implementation should look for the
     most fused VML entry points available (or, if MKL doesn't have a
     single call that does the whole RMSprop step, consider whether a
     hand-fused single-pass kernel beats a multi-call VML pipeline,
     re-measuring rather than assuming). Neither point alone was
     sufficient; both together are. Not yet implemented as a real
     kernel -- this was root-causing via probes, not landing production
     code.

     **Mixed-engine (Group A sparse-native + Group B dense-BLAS) hand-off,
     per the user's specific question**: since the driver is dispatch
     COUNT rather than which library, a Group-A-to-Group-B layer
     transition in one model pays a MUCH smaller relative tax than
     DIDLDO's own internal RMSprop fragmentation did -- a layer boundary
     is one hand-off per layer call, not per-element multiplied many
     times over. Sharing the same OpenMP/libgomp runtime across both
     groups (sili's native kernels are already libgomp-based; picking a
     libgomp-backed BLAS for Group B matches that) avoids the worse
     cross-runtime case (~7-8x) at every such boundary; the residual
     same-runtime handoff cost (~2x scale, from the isolated-single-call
     row above) would apply once per layer transition, not per
     internal op -- real but far smaller than staying naive on BOTH
     axes (cross-runtime AND unfused) would be.

     User called this "quite important," not secondary -- validated
     directly with REAL production code
     (`scripts/probe_mixed_engine_handoff.cpp`), not another synthetic
     proxy: the actual `disldo_forward` kernel (`linear_disldo.hpp`,
     Group A, sili's own `#pragma omp` regions) called immediately
     before a real MKL `cblas_sgemv` (Group B), both on libgomp
     (`MKL_THREADING_LAYER=GNU`; confirmed sili's own build already
     targets libgomp too -- `setup.py` passes `-fopenmp`,
     `tests/unit/CMakeLists.txt` does `find_package(OpenMP REQUIRED)` --
     so this match isn't hypothetical, it's what the build already
     does). Result: MKL fwd right after a real `disldo_forward` call,
     0.0713-0.0774ms across 15 reps, vs 0.0709-0.0751ms fully isolated
     -- no measurable penalty. Confirms the dispatch-count theory holds
     at a real layer boundary (one kernel call = one dispatch, same as
     the synthetic single-fused-call row), not just in the synthetic
     probe. Mixing Group A and Group B layers in one model is safe from
     this specific tax as long as both sides share the runtime -- which
     they already would, with no extra work needed on sili's existing
     kernels' side.

  **This is NOT just a DIDLDO benchmark quirk -- it's a real preemptive
  warning for SIDLDO's high-batch design below**, which explicitly
  proposes mixing custom nnz-balanced `#pragma omp` work with per-row
  BLAS calls (`sger`/small `sgemm`) in the same hot path. If alternating
  a custom OpenMP region with an external-BLAS call pays this same
  cross-team contention tax, SIDLDO's high-batch kernel could inherit it
  directly unless the threading strategy is unified up front -- flagged
  in that section below too, not just here.

  Conclusion: dense-weight-storage parity with torch confirmed for
  isolated calls (mechanism 1, fixed); a real, unresolved thread-pool-
  coexistence cost exists for realistic forward/backward-update
  alternation (mechanism 2, open) that any BLAS+custom-OpenMP engine in
  this codebase (DIDLDO now, SIDLDO later) will need a real answer for.
  Not blocking SIDLDO's scoping -- the gather-strategy design (rows/
  columns to load) is orthogonal to the threading-coexistence question --
  but the threading question needs its own resolution before either
  engine is trusted in a real training loop, not just isolated benchmark
  calls. SIDLDO's row/column-selective loader still sits on top of the
  same `cblas_sgemm`/`sgemv` calls (SIDLDO forward = gather the nonzero-
  input-position rows of W into a compact buffer, then one smaller
  `sgemm`/`sgemv`; backward same idea over columns -- still "just BLAS",
  per the user's framing, just on a pre-gathered slice of the weight
  matrix rather than the
  whole thing).

  **Aside, worth folding into the eventual selection surface**: even at
  full density and batch=1, SISLDO already beats DISLDO on forward
  (n=288: sisldo_ratio=3.48 vs disldo_ratio=7.94x torch) -- so the
  crossover isn't purely a density threshold even within the existing
  DISLDO/SISLDO pair; batch=1 specifically favors sisldo regardless of
  density. One more data point for why this needs a real multi-axis
  surface, not a hand-tuned single-density cutoff.

  **SIDLDO gather strategy -- two regimes, not one (2026-09-16).** A
  batch's sparse-input positions vary per sample, so "load only the rows
  where input is nonzero" needs a real answer to "nonzero for WHICH
  sample." Settled on two strategies for two batch regimes, not a single
  approach:

  - **Low batch: union-of-batch gather.** Take the OR of every sample's
    nonzero input positions in the batch, gather that shared row set
    from W once, run ONE `sgemm` over the compacted rows for the whole
    batch. The union-reduction itself is parallelizable to `log2(batch)`
    passes (pairwise OR-merge of per-sample bitmasks/index sets, same
    tree-reduction shape as this rollout's barrier-elimination work).
    Degrades as batch grows -- at density=0.05, the union already covers
    ~81% of rows by batch=32 (`1-(1-0.05)^32`) -- so this only pays off
    while the union stays meaningfully smaller than the full row count.
  - **High batch: feature-major CSR + nnz-balanced threading.**
    Transpose the batch's sparsity into an input-row-major structure:
    for each input feature (row of W), the "columns" are which batch
    samples have that feature active -- i.e. a CSR with n_in rows and
    batch columns, the transpose of the usual per-sample-row layout.
    Partition work across threads by TOTAL NNZ in this structure (not by
    row or by batch), so every thread gets an equal amount of real
    compute regardless of how unevenly features are activated across
    the batch -- each thread computes its own start/end row + col
    position via the nnz-partitioned row-ptr (same load-balancing shape
    `DeltaCSRLayout`/`row_nnz`-based partitioning already uses elsewhere
    in this codebase). Then each thread walks its row range and, for
    each input row it owns, loads that ONE dense weight row once (n_out
    wide) and applies it to every batch sample active for that
    feature while the row is still hot in cache/registers -- rank-1-
    update shaped (one weight row broadcast against several samples'
    scalar input values, accumulating into each of those samples'
    output rows), so per the user's framing this can likely still route
    through BLAS (`sger`/small `sgemm` on a thread's compacted slice)
    rather than needing a fully custom scalar accumulate kernel -- to be
    confirmed once this is actually built and profiled, not assumed.
    This is the regime the union approach breaks down in, and not
    coincidentally the regime (high batch) where the bench matrix shows
    the biggest sili-vs-torch wins clustering. **Real risk, found while
    root-causing DIDLDO's own batch=1 numbers (see above)**: mixing a
    custom nnz-balanced `#pragma omp` partitioning with per-thread BLAS
    calls (`sger`/small `sgemm`) is exactly the "alternate custom OpenMP
    with external BLAS calls" pattern that measurably contends for
    threads in DIDLDO's own bench (a lone forward call cost ~7-8x more
    immediately after an unrelated `#pragma omp` region than in
    isolation, and `OMP_WAIT_POLICY=PASSIVE` only partially closed it).
    This high-batch design does that alternation once per thread's row
    range rather than once per training step, which could be worse, not
    better -- the threading strategy needs to be unified (one thread
    team doing the nnz-partitioned work AND issuing the BLAS calls,
    not two separate teams handing off) before trusting this design's
    numbers, not just its gather-shape correctness.

  Not started. This is a bigger design than DIDLDO's POC -- new
  feature-major CSR construction, new nnz-based thread partitioning,
  and an open question about how much of the high-batch path can stay
  BLAS-shaped vs needs a custom accumulate kernel, NOW COMPOUNDED by the
  open thread-pool-coexistence question above. Given the size,
  treat as its own multi-phase effort (like this rollout was) rather
  than a quick follow-on, once scoped in more detail.

  **Multi-dimensional engine-selection surface, deferred until engines
  exist**: the same bench matrix shows the DISLDO/SISLDO crossover isn't
  a single density threshold -- batch count shifts it too (see the
  backward-ratio-vs-batch pattern discussed in chat 2026-09-16: forward
  ratios IMPROVE with batch after this rollout, but backward ratios
  WORSEN with batch, e.g. disldo bwd0 at n=288 dense goes from beating
  torch at batch=1 to ~6x worse at batch=1024 -- a real, monotonic,
  batch-dependent effect, not noise). Once DIDLDO/SIDLDO exist, engine
  auto-selection genuinely needs a function of (batch, input density,
  weight/synapse density, layer width) rather than a hand-tuned
  threshold on one axis -- either a fitted decision surface or a
  clustering approach over benchmarked (params -> best engine) samples.
  Not started; explicitly gated on SIDLDO existing so there's a second
  real engine to switch to.

  Not yet implemented. Original, thinner note (density-threshold-only
  framing) superseded by the above.

  **DIDLDO landed for real (2026-09-16).** User direction: "stick to the
  same omp backend everywhere," route backward through another library
  so there's no handoff, and package MKL's `.so` files the same way
  torch packages its CUDA `.so` files (a separate, optional, pip-managed
  dependency -- never vendored into this repo).

  - `setup.py`: `_find_mkl()` detects an installed `mkl`/`mkl-include`
    pip package via `sys.prefix` (confirmed by real install + inspection:
    `mkl` -> `<prefix>/lib/libmkl_*.so.N`, `mkl-include` ->
    `<prefix>/include/mkl*.h`). When found, direct-links
    `libmkl_intel_lp64.so.3`+`libmkl_gnu_thread.so.3`+`libmkl_core.so.3`
    (the GNU/libgomp threading layer, NOT the `libmkl_rt.so` runtime
    dispatcher + `MKL_THREADING_LAYER=GNU` env var -- baking the
    threading choice into the binary itself is more robust than relying
    on correct env-var configuration at deploy time; confirmed via
    `readelf -d`: `NEEDED libgomp.so.1`, no env var required) and defines
    `-DSILI_HAVE_MKL=1`. Absent MKL, the extension builds identically to
    before -- graceful compile-out, matching torch's CPU-only fallback
    when CUDA isn't present. New extra: `pip install sili[mkl]`.
  - `tests/unit/CMakeLists.txt`: mirrors the same detection (via
    `find_package(Python3 COMPONENTS Interpreter)` + `sys.prefix` query
    -- found and fixed a real bug here: the existing
    `find_package(Python3 COMPONENTS Development REQUIRED)` does NOT
    populate `Python3_EXECUTABLE`, only `Interpreter` does; the first
    attempt silently queried an empty prefix). `test_didldo_kernel` is
    kept out of the uniform `SILI_STANDALONE_TESTS` foreach (needs MKL
    include/link flags none of the other tests do) and only built when
    `SILI_MKL_FOUND`.
  - `sili/lib/headers/linear_didldo.hpp`: `DenseLinearWeights` (plain
    `w`/`square_avg`/`dw_scratch`, persistent, `.resize()`-once -- same
    lesson as every other engine in this rollout, no fresh
    per-call allocation) + `didldo_forward`/`didldo_backward`. forward
    and backward's dx both dispatch to `cblas_sgemv` at batch==1,
    `cblas_sgemm` otherwise (the root-caused fix from this investigation).
    `didldo_backward(x, dy, weights, dx, batch, num_cpus, lr)` always
    computes dx+dw, gates an RMSprop update on `lr != 0` -- same shape as
    `disldo_backward`'s own API, no separate optimizer call anywhere. The
    RMSprop step routes entirely through MKL (`vsMul`/`cblas_sscal`/
    `cblas_saxpy`/`vsSqrt`/`vsLinearFrac`/`vsDiv`/`cblas_saxpy`) -- zero
    hand-rolled `#pragma omp` regions in this file at all, per the user's
    direction. fp32 only for now (BLAS is inherently single-precision;
    fp8/fp4 would need a dense fp32 staging buffer, separate future work).
  - `tests/unit/test_didldo_kernel.cpp`: forward, bwd0 (dx correct, lr=0
    leaves weights bit-identical), and bwdX (dx correct AND weights match
    an independent reference RMSprop step applied to an independently
    computed reference dw) checked against hand-written dense-matmul/
    RMSprop references, across batch=1 (sgemv path) and batch>1 (sgemm
    path), including a non-square shape (where a transpose-direction bug
    would hide on a square matrix). PASSED locally AND on arch-sandbox
    (same real kernel file, same test file, both machines).
  - Full local regression: 174 tests via `ctest`, 169 passed, the only 5
    failures are pre-existing/already-documented
    (`pre_existing_failure`-tagged, unrelated to this change) --
    `test_didldo_kernel` itself passed clean, nothing else regressed from
    the shared `CMakeLists.txt` edit.
  - **Packaging reality check, worth knowing before anyone runs
    `pip install sili[mkl]` fresh**: `mkl`'s pip dependency tree is
    deep -- `mkl` alone pulls in `onemkl-license`, `intel-openmp`, `tbb`,
    and `intel-openmp` itself further pulls `intel-cmplr-lib-ur` (and
    likely more beyond that, not fully walked). On a machine WITH
    internet this resolves transparently (confirmed on the local
    machine: clean `pip install mkl mkl-include`, no manual
    intervention). arch-sandbox has no internet at all, so validating
    there used the already-scp'd subset of `.so`/`.h` files from this
    investigation's earlier probes (`~/claude_code/mkl_probe_libs`)
    rather than a full pip install -- sufficient to confirm the actual
    kernel/build/test are correct, but NOT a demonstration that
    `pip install sili[mkl]` itself works offline. Anyone deploying to an
    offline/restricted environment needs the full wheel set staged
    ahead of time (same category of problem as `pip install torch` with
    CUDA support on an offline machine -- not unique to this).

  **SIDLDO forward landed for real (2026-09-16), low-batch (union-gather)
  path only.** `sili/lib/headers/linear_sidldo.hpp`: `SidldoForwardScratch`
  (persistent, grow-only `active_row`/`inverse_map`/`w_compact`/
  `x_compact`) + `sidldo_forward`, sharing `DenseLinearWeights` with
  DIDLDO (same Group-B storage, per the two-group architecture). Union of
  the batch's nonzero input positions computed SERIALLY, deliberately not
  parallelized -- the DIDLDO threading investigation directly informed
  this: SIDLDO targets low batch, where a custom `#pragma omp` region for
  the gather step would pay the same handoff tax found for DIDLDO's
  RMSprop step, for a step cheap enough at this scale to not need
  parallelizing at all. Only the final `cblas_sgemv`/`cblas_sgemm` call
  (MKL's own threading) does real parallel work -- one dispatch per call,
  matching the ~1.0x/no-handoff result from that investigation.

  `tests/unit/test_sidldo_kernel.cpp`: forward vs an independent
  dense-matmul reference across batch=1/batch>1, near-dense and fully-
  empty inputs, plus a scratch-reuse-across-different-active-set-sizes
  case (catches stale-state bugs a fixed-shape test wouldn't). Passed
  locally and on arch-sandbox. `tests/unit/CMakeLists.txt`'s MKL-gated
  test block generalized from a single `if(SILI_MKL_FOUND)` for
  `test_didldo_kernel` into a `SILI_MKL_TESTS` foreach covering both.

  `scripts/bench_sidldo_forward.cpp`: timed at n=288 across the low-batch
  range (1-64) and a density sweep (0.005-0.5), compared directly against
  sisldo_ms/torch_ms already recorded in the Block4 Bench dataset at
  matching (density, batch) points (same dense-weight, syn=1.0 rows).
  SIDLDO beats SISLDO (the existing sparse-weight engine, paying CSR/
  block4 overhead for weights that are actually fully dense here) at
  EVERY density/batch combination tested, often by 5-10x (e.g. density=
  0.5, batch=64: sisldo 0.6207ms vs sidldo 0.1040ms). At low batch
  specifically, SIDLDO also beats torch's own dense GEMM outright (e.g.
  density=0.5, batch=1: torch 0.0099ms vs sidldo 0.0064ms) -- torch has
  no way to exploit input sparsity, SIDLDO does. Loses to torch at higher
  batch + higher density (e.g. density=0.5, batch=64: torch 0.0316ms vs
  sidldo 0.1040ms) -- the union has filled in by then, exactly the
  degradation this design predicted and accepted going in.

  **High-batch feature-major-CSR path landed too (2026-09-16), same
  day.** User asked directly whether the high-batch design had been
  built -- it hadn't yet; built it. `sidldo_forward_high_batch`:
  transposes the batch's per-sample CSR into a feature-major CSR
  (counting-sort, serial, O(nnz+n_in)), partitions total nnz evenly
  across threads, each thread accumulates into its own buffer, `y` gets
  a final combine. `sidldo_forward` now dispatches on
  `SIDLDO_HIGH_BATCH_THRESHOLD` (64) between this and the low-batch
  union-gather path. First direct A/B against the flagged threading risk
  above (`probe_nested_mkl.cpp`): calling MKL's `sger` from inside a
  custom `#pragma omp` region, once per row per thread, measured ~3x
  SLOWER than a plain hand-rolled scalar loop doing the same math -- so
  this function calls MKL nowhere at all, confirming the risk was real
  and resolving it by avoiding BLAS entirely in this path, not by
  unifying threading further.

  First version used a full `[num_threads x batch x n_out]` private
  buffer per thread (same shape disldo_forward/backward's own reductions
  use) -- correctness-verified, but benching against the union-gather
  path directly (`probe_crossover2.cpp`/`probe_crossover3.cpp`, not just
  against the dispatcher) found it losing badly in some cells, most
  starkly at low density + high batch (n=288, density=0.005, batch=1024:
  union 0.577ms vs high-batch 1.33ms -- LOST despite being "the sparse
  path"). Root-caused, not guessed: the reduction step costs
  `O(threads*batch*n_out)`, independent of density -- at low density
  there's little real work to justify that fixed cost.

  **Fixed the real cause, not the symptom (user's explicit direction)**:
  replaced the full private buffer with a compact per-thread
  touched-sample buffer -- the SAME union/inverse-map trick the
  low-batch path already uses for weight ROWS, applied to output SAMPLES
  instead. `sample_local_map` is grow-only and reset only at touched
  entries (never a full clear), matching the low-batch path's own
  `inverse_map` discipline. The combine step is now bounded by actual
  touched work (nnz-bounded), not `threads*batch*n_out`. Re-ran the same
  A/B after the fix:

  | n_in | density | batch | union_ms | highbatch_ms (before fix) | highbatch_ms (after fix) |
  |---|---|---|---|---|---|
  | 288 | 0.005 | 1024 | 0.577 | 1.33 (LOST) | **0.133** (4.3x win) |
  | 2048 | 0.005 | 1024 | 18.61 | 12.32 | **8.84** (bigger win) |
  | 2048 | 0.02 | 1024 | 18.74 | 22.58 (LOST) | 20.32 (still lost, closer) |
  | 2048 | 0.05 | 1024 | 17.17 | 44.74 (LOST) | 41.98 (still lost) |

  The fix eliminated the worst class of loss (low density) outright and
  meaningfully narrowed the rest, but did NOT fully close the
  moderate-to-high-density + very-high-batch cells. Root cause there is
  DIFFERENT and not a memory-overhead problem: once nnz is large enough
  that each thread's chunk already touches most of the batch anyway, the
  compact-buffer trick has nothing left to save, and the real bottleneck
  becomes raw accumulate-loop throughput -- MKL's cache-blocked dense
  GEMM genuinely computes faster than this hand-rolled loop at that
  scale, the same "BLAS wins at genuinely dense work" fact this whole
  investigation kept re-confirming. **This means `SIDLDO_HIGH_BATCH_
  THRESHOLD`'s flat batch-only dispatch is STILL known-imperfect** at
  high density + very high batch -- the correct fix is density-aware
  dispatch (or just always trying both and keeping the winner, at some
  extra cost), not implemented, folded into the multi-dimensional
  engine-selection-surface work already scoped above rather than solved
  ad hoc here.

  Correctness: `tests/unit/test_sidldo_kernel.cpp` extended with the
  threshold boundary (batch=63 vs 64) and high-batch cases at varying,
  non-power-of-two `num_cpus` (1, 3, 7) to stress the nnz-balanced
  partition and binary-search row lookup. Passes locally and on
  arch-sandbox, full local regression clean (same 5 pre-existing
  failures, nothing new).

  **SIDLDO backward landed (2026-09-16), union-gather only (no high-batch
  variant yet).** First had to resolve a real ambiguity before writing
  any code: SIDLDO's own name only commits FORWARD's input to being
  sparse -- what's sparse for backward wasn't obvious, and guessing
  wrong would mean scrapping the implementation. Asked directly; user
  confirmed it matches the codebase's EXISTING convention for the
  sisldo family (`disldo_backward_sparse_grad`, sisldo_ops.hpp): DENSE
  input `x`, SPARSE output gradient `dy` (CSR) -- not a reuse of
  forward's sparse `x`. `dy`'s sparsity comes from whatever downstream
  sparsifying op produced it, independent of whether this layer's own
  forward input was sparse.

  `sidldo_backward(x, dy_ptrs, dy_idx, dy_val, batch, weights, scratch,
  dx, lr)`: mirrors forward's union-gather exactly, transposed -- union
  of `dy`'s active output COLUMNS (not input rows), gather those
  COLUMNS of `W` into a compact `[num_active x n_in]` buffer (a strided
  read from row-major `W`, unavoidable given `W`'s layout is fixed by
  forward/DIDLDO's needs, but a contiguous write, and the buffer this
  produces is exactly what both GEMMs below need with no further
  transpose), densify `dy` into a compact `[batch x num_active]` buffer.
  `dx = dy_compact @ w_col_compact` (sgemv at batch=1, sgemm otherwise,
  same M=1 dispatch as everywhere else in this arc) and
  `dw_compact = x^T @ dy_compact` (one sgemm). RMSprop update (lr != 0
  gate, same one-function shape as DIDLDO/disldo_backward, no separate
  optimizer call) applied via a hand-rolled loop directly at the real
  strided `(row, active_col)` positions in `weights.w`/`square_avg` --
  not through VML, same reasoning as the high-batch forward path:
  gathering/scattering a compact `square_avg` buffer just to use VML's
  contiguous-array ops would cost more than this plain loop saves, and
  `dw_compact` is already contiguous from the GEMM immediately above it.

  `tests/unit/test_sidldo_kernel.cpp` extended with `run_backward_case`:
  dx checked against an independent dense reference, bwd0 (lr=0) leaves
  weights bit-identical, bwdX (lr!=0) matches an independently computed
  reference RMSprop step applied to an independently computed reference
  dw -- across batch=1/batch>1, near-dense, fully-empty dy, and a
  scratch-reuse-across-different-active-column-count case (same
  stale-state class of bug forward's own scratch-reuse test already
  catches). Passes locally and on arch-sandbox. Full local regression
  clean (same 5 pre-existing failures, nothing new).

  Not done: a high-batch (feature-major-CSR-style) SIDLDO backward
  variant -- forward's high-batch path exists because forward's
  union-gather measurably degrades at high batch; backward's
  union-gather almost certainly has the same degradation shape (dy's
  active-column union fills in the same way x's active-row union does),
  but this hasn't been benched yet, so building a high-batch backward
  now would be speculative rather than measured. Natural next step if/
  when asked, same as forward's own high-batch path was a separate,
  later request rather than built preemptively.

  **Not done**: density-aware dispatch (see above); python bindings for
  either DIDLDO or SIDLDO (both are C++-kernel-plus-tests only so far,
  same phased approach the rest of this rollout used) -- user's explicit
  direction: when bindings land, fp8/fp4 should raise not-implemented
  errors rather than silently doing the wrong thing (both engines are
  fp32-only; this is a binding-layer concern, not resolved until
  bindings themselves are built).
