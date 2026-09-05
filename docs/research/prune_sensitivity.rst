``prune_sensitivity.py`` research notes
==========================================

Companion doc to ``sili/conversion/prune_sensitivity.py``. Not part of the
sili runtime path -- a conversion-time diagnostic/calibration tool, like
``sparse_prune.py``'s calibration functions.

.. _prune_sensitivity.motivating_finding:

Why per-role thresholds, not one global sparsity threshold
--------------------------------------------------------------

*ID:* ``prune_sensitivity.motivating_finding``

Per-tensor-role pruning sensitivity: does a model tolerate the SAME
sparsity threshold on every 2-D matrix, or do different weight roles
(attention Q vs V, MLP gate vs down, embeddings, ...) need very different
thresholds to avoid destroying the model?

Motivating finding (see sili_peridot's own conversion notes): on
MiniCPM5-1B-Base, ``embed_tokens`` tolerated 90% magnitude-pruned sparsity
with only a mild quality drop, while ``v_proj`` (architecturally similar to
``k_proj``, which tolerated 70% fine) collapsed already at 30%. A single
global ``calibrate_min_abs_param()`` threshold cannot serve both of those
tensors well -- one group's safe ceiling is another group's already-
catastrophic floor. This module makes that kind of per-role analysis a
reusable procedure instead of one-off manual sweeps, for whichever
downstream task (next-token perplexity, classification accuracy, a custom
metric) the caller cares about -- ``eval_fn`` is the only model/task-specific
piece; everything else in the module is generic.

.. _prune_sensitivity.role_grouping_rule:

How tensors get grouped into "roles"
-----------------------------------------

*ID:* ``prune_sensitivity.role_grouping_rule``

``group_tensor_names_by_role`` groups 2-D tensor names by "role": tensors
that recur once per repeated block (e.g. every layer's own
``self_attn.q_proj.weight``) are grouped by their shared (prefix, suffix) --
matching ``rnn_fold.detect_repeated_block_groups``'s own notion of a
repeated block, since these are the tensors a single fold-time CSR
threshold will eventually apply to uniformly. Tensors with no numeric
block index (``embed_tokens``, ``lm_head``, a final norm, ...) each get
their own singleton group under their full name -- there's nothing to
group them WITH.

Only 2-D tensors are included (matches ``_keep_dense_reason``/
``calibrate_min_abs_param``'s own eligibility rule -- vectors and higher-
rank tensors are never threshold-pruned, so grouping them for a
sensitivity sweep would be meaningless).

.. _prune_sensitivity.iterative_search_greedy:

``iterative_threshold_search``'s greedy algorithm and why it's not optimal
--------------------------------------------------------------------------------

*ID:* ``prune_sensitivity.iterative_search_greedy``

Greedy search: run ``stepwise_cumulative_eval`` with the CURRENT
thresholds, find whichever step caused the single worst marginal score
drop, shrink JUST the group(s) newly added in that step by
``shrink_factor`` (floored at ``min_threshold``), and repeat. Stops as
soon as the final (all-groups-combined) score reaches ``target_score``, or
after ``max_iterations`` rounds, or once nothing is left with room to
shrink -- whichever comes first.

This encodes the "look at the stepwise trace, reduce whatever caused the
biggest single jump, re-check the combined result" procedure used to
calibrate MiniCPM5-1B-Base's own per-group thresholds (see sili_peridot's
conversion notes) -- greedy and NOT guaranteed globally optimal (shrinking
one group changes how much a group added AFTER it costs, since costs
compound along ``step_order``; only the single worst offender per round
gets touched, never a joint optimization across all groups at once), but
it's exactly the manual procedure that worked there, made repeatable
instead of ad hoc.

``stepwise_cumulative_eval`` itself always evaluates against the original
``state_dict`` as the baseline reference at every step (not against the
previous step's result) -- the point is to see whether combining per-group
thresholds compounds into a worse-than-expected quality loss, which
isolated single-group sweeps (``sweep_group_sensitivity``) can't reveal on
their own.
