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
      follow-ups, not yet done:
      1. Investigate why the gain shrinks at batch>=256 -- candidates:
         `input_T`/per-thread `thread_buf` being sized/touched
         proportional to batch every call (memory-bandwidth-bound at
         that size?), or the flush pass itself becoming a bigger share
         of the work. `thread_buf` is NOT yet a persistent scratch
         buffer (unlike `input_T`) -- allocates+zeroes fresh every call,
         inside the parallel region; making it a reused
         `weights.block4.scratch_*` member (matching `input_T`'s own
         pattern) is an obvious next thing to try.
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
- [ ] **Phase 2 -- disldo_forward, narrow (tree-reduction) path, FP32.**
      Same horizontal-reduce problem exists here too. Buffer story
      differs -- narrow path already uses num_cpus private FULL buffers
      pre-reduction, not column-owned ranges -- needs its own design pass,
      not a copy-paste of Phase 1.
- [ ] **Phase 3 -- disldo_forward FP8 + FP4 (precision parity).** Same
      loop shape, only weight decode differs -- see
      `feedback_kernel_perf_precision_parity_and_bench_isolation` memory:
      fp32 speedups must port to fp8/fp4, not stay fp32-only.
- [ ] **Phase 4 -- disldo_forward's scattered (non-block4, `!dc.empty()`)
      path.** Different shape already (scalar multiply-add, not
      horizontal-reduce) -- needs its own check for whether the
      cross-thread-contention angle still applies before assuming yes.
- [ ] **Phase 5 -- sisldo_forward.** Different structure (gather-based,
      work-offset table for variable-density CSR batches) -- needs its
      own investigation of whether/how this pattern applies at all before
      implementing anything.
- [ ] **Phase 6 -- disldo_backward.** Already has SOME transpose infra
      (`batch_stride_transpose`, input_T/output_grad_T -- see
      `docs/research/linear_disldo.rst`). Audit whether the missing piece
      is just the private-per-thread-buffer/contention angle, or more.
- [ ] **Phase 7 -- sisldo backward (`disldo_backward_sparse_grad`,
      sisldo_ops.hpp).** Same audit as Phase 6, plus keep in mind the
      heap-corruption bug just fixed there (stale tile position tracking)
      -- any restructuring must not reintroduce that class of bug.

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
