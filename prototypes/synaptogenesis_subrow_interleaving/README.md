# Sub-row interleaved free space for synap_step (design sketch, NOT implemented)

Not code, not benchmarked -- a design note capturing a direction raised
while diagnosing why online training's per-token synaptogenesis was
unexpectedly slow on real (large, already-fairly-dense) MiniCPM5 MLP
layers, kept here rather than lost in conversation.

## The measured problem

Profiling `sili_peridot`'s online training probe (`model/train_online.py`)
against the real checkpoint found `synap_step` costing 25-188ms per call
on the last fold step's MLP layers (`gate_proj`/`up_proj`: ~1536x4608,
~3,500 real entries/row; `down_proj`: ~4608x1536, ~1,390 entries/row) --
`build_probes` and `equalizer_step` were both negligible (<0.06ms) in
the same calls. Cost scaled with each row's ABSOLUTE existing content
size, not its density as a fraction of the row: `down_proj` rows are
*more* dense (90.5%) than `gate_proj`/`up_proj` rows (74-78%) but have
fewer absolute entries (1,390 vs ~3,500) and cost ~7x less per call
(26ms vs 170-188ms).

(The training probe skips synaptogenesis entirely rather than working
around this cost -- growing/pruning connections needs real per-synapse
importance signal to be a meaningful decision, and this probe hasn't
run long enough to accumulate any; that's the actual reason, this
finding was secondary. Kept here since it's real and worth fixing on
its own terms.)

## Why row size (not row density) drives the cost

Each row in the delta-CSR layout is a single sequential byte buffer
(FOR-encoded groups of G values -- see `../for_delta_encoding/`).
Inserting a new synapse into a row that already has thousands of real
entries means finding the right sorted position and shifting
everything after it, whether the row is "sparse" in the sense of
having empty columns elsewhere or not -- the cost is a function of how
much byte content already exists in that row's buffer, not how many
possible (row, col) pairs remain unused. A row with 3,500 real entries
costs far more to insert into than a row with 1,390, independent of
either row's nominal density.

## The idea: interleave blank space at the GROUP level, not once per row

Currently, a row's spare/blank capacity (from `equalize_to_capacity`/
`expand_headroom`) is reserved as one contiguous region relative to the
row's content -- inserting in the middle of a large row still means
shifting everything between the insertion point and wherever that
blank region sits. The FOR encoding already breaks each row into fixed
-size groups of G values (each independently decodable, with its own
width-tier byte header -- see `../for_delta_encoding/README.md`).
That's a natural, already-existing sub-row unit: if each GROUP carried
its own small slice of blank space (rather than the row carrying one
big slice), an insertion that lands within group `g` would only need
to shift bytes within that group (and, if it overflows the group's own
slack, cascade to at most the next group or two) -- not the whole rest
of the row. For a row with 3,500 entries in ~110 groups of G=32, that
turns an O(row content) insertion into something closer to O(group
size) = O(32) in the common case.

Quick lookup into which group a given column index falls into would
need the group boundaries' own starting columns indexed (e.g. a small
per-row array of each group's `group_start` value, binary-searchable)
-- this is the piece the "work pointers" idea gestured at, though that
exact term doesn't correspond to anything in the current codebase
(grepped: no `parallel_ptr`/`ParallelPtr`/"work pointer" anywhere in
`sili/`, only in `tests/unit/python/test_sili.py`'s already-known-stale,
pre-existing-broken tests -- likely a vestigial concept from an earlier
API generation, not something to build on directly). The GROUP-start
index is the concrete, already-real structure to hang this on instead.

## Status

Not started. No prototype, no benchmark, no correctness check -- purely
a design note. Next step if picked up: a standalone CPU prototype (same
pattern as `../for_delta_encoding/` and `../disldo_accumulation_simd/`)
measuring insertion cost with per-group vs per-row blank space on
synthetic rows at realistic sizes (thousands of entries, matching the
measured MiniCPM5 MLP layers) before touching the real library. Real,
buildable work, comparable in scope to `../gpu_dual_csr_csc_training/`
-- both are "likely method of improvement," not scheduled.
