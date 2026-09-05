``hoyer_sparsify.hpp`` research notes
========================================

Companion doc to ``sili/lib/headers/hoyer_sparsify.hpp``. NOT wired into an
automatic dense/sparse dispatch (see ``TODO.md``) -- this is the standalone
operation, exposed so its actual behavior on real data can be
explored/tested from Python before deciding on dispatch thresholds ("not
obvious" -- per conversation).

.. _hoyer_sparsify.math_derivation:

Hoyer's sparsity measure: L1/L2 ratio as an effective-k estimator
------------------------------------------------------------------------

*ID:* ``hoyer_sparsify.math_derivation``

Rather than an arbitrary epsilon-threshold count of "near-zero" elements,
use the ratio of L1 to L2 norm as a principled estimate of the EFFECTIVE
number of significant elements in a vector:

.. code-block:: text

    hoyer(x) = (sqrt(n) - ||x||_1/||x||_2) / (sqrt(n) - 1)     in [0, 1]

For a vector with exactly k nonzero entries of EQUAL magnitude (rest
exactly zero), ``||x||_1/||x||_2 = sqrt(k)`` exactly -- verified in the
implementation, not just asserted. For a realistic vector (mixed
magnitudes, not exactly k-sparse), the same ratio still gives a smooth
estimate of the "effective" significant-element count:

.. code-block:: text

    k_estimate = (||x||_1 / ||x||_2)^2

which becomes the target k for a top-k sparsification pass -- keep only
the k_estimate largest-magnitude elements, treat the rest as noise (not
"whatever happens to be exactly zero").

.. _hoyer_sparsify_per_batch.row_terminology_warning:

``hoyer_sparsify_per_batch``: "row" means batch sample, not weight row
------------------------------------------------------------------------

*ID:* ``hoyer_sparsify_per_batch.row_terminology_warning``

One ``HoyerSparsifyRow`` per BATCH SAMPLE (not weight-matrix row) -- "row"
here means one row of the ``[batch, cols]`` activation array, i.e. one
sample; ``DeltaCSRWeights`` elsewhere in this codebase uses "row" for
input NEURON, a different axis entirely -- same word, easy to conflate,
worth being careful about. ``k_estimate`` is computed independently per
sample, not one shared k across the batch (unlike ``top_k_csr``) --
different samples can have genuinely different effective sparsity.

.. _hoyer_sparsify_per_batch.not_for_routing_decision:

Per-sample construction vs. the routing decision -- not the same question
--------------------------------------------------------------------------------

*ID:* ``hoyer_sparsify_per_batch.not_for_routing_decision``

``hoyer_sparsify_per_batch`` is for CONSTRUCTING an accurate sparse
representation once you've already decided to route a batch through the
sparse path -- it is NOT the right thing to base that routing decision on.
``forward_dense``/``forward_sparse`` are each called ONCE for the whole
batch together; a per-sample "is this one sparse enough" answer isn't
actionable at that granularity, since you can't send some samples through
one kernel and some through the other in a single call. For the actual
dense-vs-sparse ROUTING decision, use ``hoyer_score()`` instead, which
aggregates over the whole batch to produce the one number that question
actually needs.

.. _hoyer_score.batch_aggregate_rationale:

``hoyer_score``: the batch-level aggregate a routing decision should use
--------------------------------------------------------------------------------

*ID:* ``hoyer_score.batch_aggregate_rationale``

Batch-level aggregate Hoyer's measure -- the actual quantity a
dense-vs-sparse ROUTING decision should be based on, computed over the
WHOLE flattened batch (all rows*cols elements together) rather than per
sample, since ``forward_dense``/``forward_sparse`` are each invoked once
for the entire batch in a single call, not once per sample.

Returns the raw ``hoyer_score`` (l1/l2-derived, in [0,1]) plus l1/l2 and a
batch-wide ``k_estimate`` -- what a threshold comparison should actually
use to decide which of ``forward_dense``/``forward_sparse`` to call for
this batch. Per-sample construction of the actual CSR data, once that
decision is made, should still use ``hoyer_sparsify_per_batch()`` (or a
fixed/shared k if uniform treatment across the batch is preferred) --
this function only answers "which kernel", not "which elements to keep
per row".
