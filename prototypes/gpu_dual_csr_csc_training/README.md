# Dual CSR+CSC sparse weight storage for GPU-parallel training (design sketch, NOT implemented)

Not code, not benchmarked -- this is a design note capturing a direction
raised while scoping sili_peridot's post-conversion training work, kept
here (rather than lost in conversation) since it's a real future option
for making sparse training viable on GPU, and connects directly to the
FOR delta encoding work in `../for_delta_encoding/` and
`feature/delta-csr-for-encoding` (PR #17, unmerged).

## The problem this targets

The current single-CSR weight representation is built for a CPU,
single-format, sequential-decode access pattern. Forward needs
row-wise (CSR) access; the backward pass (gradient w.r.t. input) needs
column-wise (CSR-transposed, i.e. CSC) access to the same weights.
Doing that transpose on the fly, or doing backward as a scatter into a
row-major structure, is exactly the kind of random-access pattern that
starves GPU throughput (and, per `../disldo_accumulation_simd/`,
starves this CPU's AVX2 gather too).

## The idea

Store each sparse weight matrix TWICE: once as CSR, once as CSC --
duplicated values, not two index structures pointing at one shared
value array. Forward becomes a CSR (activations) x CSC (weights)
product; backward (gradient w.r.t. input) becomes the same CSR x CSC
shape against the transposed problem, using the CSC copy directly
instead of transposing at runtime. Both directions become sorted-merge
operations against two already-sorted-order representations, with no
runtime transpose and no scatter in the hot forward/backward path --
fully parallel, GPU-friendly.

### The one place a transpose is still needed: importance-averaging + optimizer merge

The optimizer step (importance-weighted averaging of new values into
existing weights) naturally accumulates into the CSR copy, since
backprop's per-synapse updates arrive in CSR's row order. Getting
those updates into the CSC copy needs a transpose -- but:

- This does NOT need to happen in real time / every step. It can be
  queued and applied step-wise/batched, decoupled from the
  forward/backward hot path, same as other deferred maintenance work
  in this library (e.g. `compact()`, `expand_headroom()`).
- With sparse forward+backward (not dense), only the synapses that
  were actually touched by backprop on a given step need to be
  transposed and merged -- not the whole matrix. This is a small,
  bounded set per step, not an O(nnz) full-matrix transpose.

### Cost

Storing values twice roughly doubles per-parameter storage: current
single-CSR format is ~16-24 bits/parameter (FP4-ish value + delta-
encoded column index); dual CSR+CSC would land around 32-48 bits/
parameter. Judged worthwhile: 32 bits/parameter for genuinely sparse
GPU-parallel training at decent speed is still a very good number, and
doubling memory is a reasonable price for making training (not just
inference) viable on GPU, and for potentially helping the CPU path
too (same sorted-merge property that helps GPU parallelism should also
help vectorize the CPU forward/backward, unlike the gather/scatter
attempts in `../disldo_accumulation_simd/` which lost to genuinely
random access).

### Hard constraint: real-time single-step causal operation is not negotiable

sili's core use case (see project memory: non-causal is the default,
causal is opt-in but implies per-step KV-cache-style online operation)
requires that a single forward+backward step at batch=1 stay fast and
low-latency -- this is a requirement, not a nice-to-have, for whatever
comes out of this design. Any GPU-parallel batched formulation needs
to keep a real single-step path that doesn't regress to needing a full
batch to be efficient.

### Relationship to FOR delta encoding (PR #17)

The FOR (frame-of-reference) column-index encoding in
`../for_delta_encoding/` and the full migration on
`feature/delta-csr-for-encoding` were shelved as not worth merging on
their own (decode was never the CPU bottleneck here, see that folder's
README). But a dual CSR+CSC format doubles the number of decode passes
per step (one per format), so the FOR encoding's 4-12x decode speedup
may become relevant again here even without a hardware change -- worth
revisiting PR #17 if/when this design gets built, rather than treating
it as dead work.

## Status

Not started. No prototype, no benchmark, no correctness check. Next
step if picked up: a standalone CPU prototype (same pattern as
`../for_delta_encoding/` and `../disldo_accumulation_simd/`) proving
out the CSR x CSC sorted-merge forward/backward on synthetic data
before touching GPU code or the real library.
