``rnn_fold.py`` research notes
==================================

Companion doc to ``sili/conversion/rnn_fold.py``. Source comments point back
here by anchor ID (``*ID:* `` marker under each heading below). See
``docs/research/linear_disldo.rst`` for the pattern this follows (semantic
dotted anchor IDs, visible ID markers, frozen code snippets on real-bug /
non-obvious-derivation sections only), and
``docs/research/prune_sensitivity.rst`` for the closest sibling module
(also ``sili/conversion/*.py``, also keys blocks by ``(prefix, index)`` via
``_parse_block_key``).

.. _rnn_fold.module_overview:

What folding does: N identical blocks -> one recurrent block, N steps
---------------------------------------------------------------------------

*ID:* ``rnn_fold.module_overview``

Detects consecutive, structurally identical transformer blocks in a
(sparse-pruned) model and folds them into a single recurrent block whose
weights are one stacked sparse CSR matrix::

    Original:  in --[Block_0]--[Block_1]--...--[Block_N-1]--> out
               N identical-structure blocks, different trained weights each.

    Folded:    state = 0
               for i in 0..N-1:
                   state += Block_folded(x + state)
               # One block, N steps, state accumulates all fold outputs.

**Weight layout.** Each block's weight is ``[out, in]``; the stacked matrix
is ``[N*out, in]`` with nonzeros forming horizontal bands -- rows
``[i*out:(i+1)*out]`` are block ``i``'s original weights. At fold step ``i``
the executing code slices that row range from the CSR matrix and uses it as
the current weight, faithfully reproducing the original per-block
computation. See ``rnn_fold.package_qualified_import_bug`` below for the
``_cpu`` extension import, and ``make_banded_attention_mask.locality_rationale``
for how attention is kept local across fold steps.

**Integration with sparse_prune.py**::

    from sparse_prune import load_state_dict
    from rnn_fold import fold_sparse_payload

    payload = torch.load("model_sparse.pt")   # from sparse_prune.py
    folded  = fold_sparse_payload(payload)
    torch.save(folded, "model_folded.pt")

    # Or as part of the sparse_prune pipeline:
    python sparse_prune.py model.pt --rnn-fold

**CLI**::

    python rnn_fold.py model_sparse.pt                  # auto-detect & fold
    python rnn_fold.py model_sparse.pt -o model_folded.pt
    python rnn_fold.py model_sparse.pt --show-groups     # detect only, no fold
    python rnn_fold.py model.pt        --also-prune      # prune then fold

.. _rnn_fold.package_qualified_import_bug:

Why the ``_cpu`` extension import must be package-qualified only
---------------------------------------------------------------------

*ID:* ``rnn_fold.package_qualified_import_bug``

A previous version of this import block did ``sys.path.insert(0,
.../sili)`` then a bare ``import _cpu``, which makes the compiled extension
importable under TWO DIFFERENT ``sys.modules`` keys (``'_cpu'`` and
``'sili._cpu'``) depending on which import path some other module in the
process happens to use first. Each key triggers a SEPARATE execution of the
extension's module-init code, so pybind11's static type registrations run
twice, raising ``"generic_type ... already registered"`` the moment both
paths get exercised in one process -- exactly what happened repeatedly when
this file was transitively imported alongside ``sili/sparse_rnn.py``'s
``import sili._cpu as _cpu``.

A single, consistent, package-qualified import path everywhere makes
``sys.modules`` naturally deduplicate the extension regardless of import
order -- also required for tools that walk the whole ``sili`` package tree
(pdoc, Sphinx autodoc) without controlling import order.

.. _detect_repeated_block_groups.prefix_index_keying_bug:

Why block groups are keyed by ``(prefix, index)``, never bare index
------------------------------------------------------------------------

*ID:* ``detect_repeated_block_groups.prefix_index_keying_bug``

Keying on ``(prefix, index)`` rather than bare index is load-bearing, not
cosmetic. Bare-index keying merges unrelated block families that happen to
share index ranges -- caught by the toy VLM generator
(``tests/unit/python/gen_toy_mistral_vlm.py``): 12 language layers + 6
vision layers were returned as two groups of 6 instead of one of 12 and one
of 6, because indices 0-5 silently merged both families' suffixes into one
shape-dict, then diverged from indices 6-11 (language-only) once the vision
family ran out of blocks.

Every downstream match on ``_parse_block_key`` output must check the prefix
too, not just the index. This bug had three latent instances, all fixed
together: ``detect_repeated_block_groups`` itself, ``report_block_groups``'s
suffix collection for its summary printout, and ``fold_block_group``'s own
suffix collection (matching prefix AND index when gathering a sample
block's parameter suffixes) plus ``fold_sparse_payload``'s removal step
(matching prefix AND index, not index alone, when deciding which param
names a fold group consumed).

.. _make_banded_attention_mask.locality_rationale:

Why attention needs a band mask after folding
---------------------------------------------------

*ID:* ``make_banded_attention_mask.locality_rationale``

In the original N-block transformer each block computes full attention over
the sequence. After folding into one RNN block that runs N times, the state
carried between steps mixes context from previous fold steps. To prevent
step ``i`` from attending to "stale" context more than one fold step away --
which would not have been possible in the original sequential layout -- the
attention window is capped at ``band_half_width``:

.. code-block:: python

    mask[q, k] = 0     if |q - k| <= band_half_width   # allowed
               = -inf   otherwise                        # blocked

Setting ``band_half_width = seq_len`` (the default in
``fold_sparse_payload``, inferred by ``infer_seq_len_from_attn_weight`` from
the Q/K/V weight's row dimension) means every query sees the full current
sequence plus exactly one step of prior state, matching the per-block
locality of the original model. For pure same-sequence self-attention (no
cross-step state in the keys) this band covers the entire matrix and the
mask is all-0 -- it only becomes non-trivial once key/value context is
extended with state from previous fold steps. Architectures using rotary or
ALiBi position embeddings should override ``band_half_width`` explicitly in
``fold_sparse_payload()`` rather than trust the inferred value.

.. _RNNFoldedBlock.average_vs_sum_rationale:

Why ``skip_connection_outputs`` averages per-step outputs, not sums
------------------------------------------------------------------------

*ID:* ``RNNFoldedBlock.average_vs_sum_rationale``

Two output modes, controlled by ``descriptor.skip_connection_outputs``:

.. code-block:: python

    # Standard (False): final accumulated state
    state = 0
    for i in range(n_folds):
        out = block(x + state); state += out
    return state

    # Skip-connection (True): mean of all per-step outputs
    state, outputs = 0, []
    for i in range(n_folds):
        out = block(x + state); state += out; outputs.append(out)
    return mean(outputs)

Each fold step computes at a different temporal scale: early steps see only
local context (narrow banded attention over ``x``), later steps see richer
integrated context (``x`` + accumulated state from prior steps). Averaging
lets every scale contribute equally to the final output regardless of fold
count -- analogous to pyramidal neurons in cortex receiving fast/local
signals on basal dendrites and slow/long-range signals on apical dendrites,
with the soma averaging across both.

Mathematically, with windowed attention of half-width W, fold step ``i`` has
an effective receptive field of ``(i+1)*W`` tokens. Averaging the outputs
produces a weighted sum over all receptive-field sizes ``1*W .. N*W`` with
equal weight ``1/N`` -- an ensemble over scales. A plain sum would make the
last step dominate (it has seen the most context); averaging removes that
bias.

.. _RNNFoldedBlock.no_gating_rationale:

Why no LSTM/GRU-style gating
----------------------------------

*ID:* ``RNNFoldedBlock.no_gating_rationale``

Gates add parameters and complexity. The original transformer already has
residual connections, layer norms, and attention -- these collectively
perform the role of gating. A plain additive state lets the folded block
leverage those existing mechanisms without duplicating them. Attention
banding (``make_banded_attention_mask.locality_rationale``) enforces
locality between fold steps, the main stability mechanism an LSTM gate
would otherwise provide.

.. _SiliBlock.single_call_stacked_layer_design:

Why one ``SparseLinearLayer`` per suffix, loaded with the FULL stack
-------------------------------------------------------------------------

*ID:* ``SiliBlock.single_call_stacked_layer_design``

``SiliBlock`` builds ONE ``SparseLinearLayer`` per parameter suffix, loaded
with the full stacked weight matrix spanning all N fold steps -- not N
separate layers, and not individual per-fold-step slices loaded on demand.
This means ``forward_sili`` calls ``_forward_one_suffix`` EXACTLY ONCE per
suffix, not N times.

At initialization the result is equivalent to running the N original
transformer blocks sequentially. After synaptogenesis, redundant
connections across fold steps get pruned -- only what is genuinely needed
survives. A dense stacked matrix would just be an N-wide layer with no
benefit; sparsity is what makes the single-call design viable, not a
micro-optimization.

Weight orientation: ``SparseLinearLayer`` is ``[n_inputs x n_outputs]``, but
``stacked_weights[suffix]`` is ``[n_folds*out_dim x in_dim]`` (standard
weight-matrix orientation) -- transpose before loading so the layer has
``n_inputs=in_dim``, ``n_outputs=n_folds*out_dim``.

Transposing must go through ``csr.t().to_sparse_csr()``, never
``.to_dense()``: ``.t()`` is metadata-only (CSC relabelling, no densify, no
nnz-proportional copy) and ``.to_sparse_csr()`` does the real reorganization
into row-major order, but it's still nnz-proportional, never dense. ``csr``
here is a whole folded/stacked layer's matrix, which can be too large to
safely materialize -- ``sili/sparse_rnn.py``'s ``FoldedLayer.from_descriptor``
had the identical bug (see ``docs/research/sparse_rnn.rst``).

.. _SiliBlock.per_row_fp4_scaling:

Per-row value scaling before FP4 quantization
----------------------------------------------------

*ID:* ``SiliBlock.per_row_fp4_scaling``

The stacked matrix spans rows from N different original layers with
potentially very different weight magnitudes. FP4's table is
``{0, +-0.5 ... +-6.0}``, so a row with max-abs ~0.1 would map almost
entirely to 0 or +-0.5 under a single global scale. Per-row scaling maps
each row's max-abs to 6.0 before quantization, then records the inverse
scale in ``value_scale[r]`` so the forward kernel recovers
``true_w = stored_fp4 * value_scale``.

Workflow: pre-scale -> ``load_weights`` (quantizes accurately) ->
``set_value_scale_raw`` (metadata only, no re-encoding). Do NOT call
``rescale_value_row()`` after a pre-scaled load -- it would re-encode the
already-scaled values.

.. _fold_block_group.dense_raw_entry_bug:

Bug: dense-stored 2-D suffixes silently deleted, not just left unfolded
------------------------------------------------------------------------------

*ID:* ``fold_block_group.dense_raw_entry_bug``

Found converting a real checkpoint. ``sparse_prune.py``'s ``"raw"`` key is
NOT used only for true scalars/vectors -- ``_keep_dense_reason`` falls back
to ``"raw"`` for ANY 2-D matrix that stayed dense (low sparsity, or CSR
overhead not worth it), which is the common case for a gently-pruned model.

The old code checked ``raw.get("csr") is None`` and treated that as
"scalar, skip":

.. code-block:: python

    # BUG: silently skipped every dense-stored 2-D suffix instead of
    # stacking it like the plain-dense-tensor branch already did.
    if raw.get("csr") is None:
        continue  # wrong: "raw" dict entries can be dense 2-D matrices too

Worse, ``fold_sparse_payload``'s removal step deletes a block's per-layer
keys by ``(prefix, index)`` alone, not by whether the suffix was actually
captured in ``fold_block_group`` -- a suffix skipped this way had its
weights silently DELETED from the payload with no trace, not just left
unfolded. Fixed by giving a ``"raw"`` dict entry identical treatment to the
plain-dense-tensor branch (convert to CSR, no pruning, just layout), and by
having ``fold_sparse_payload`` only remove names whose suffix is actually in
``desc.stacked_weights``.

.. _fold_sparse_payload.nnz_accounting_bug:

Bug: "lossless stacking" false-negative from dense-layout nnz undercount
-------------------------------------------------------------------------------

*ID:* ``fold_sparse_payload.nnz_accounting_bug``

``original_nnz`` must count ACTUAL nonzero content regardless of whether a
given tensor started as CSR or dense/strided layout. The old filter:

.. code-block:: python

    # BUG: silently excluded dense-format entries (LayerNorm weights, and
    # anything that fell back to dense via min_sparsity/max_sparse_ratio)
    # from the "before" count entirely.
    original_nnz = sum(
        t.values().numel() for t in ... if t.layout == torch.sparse_csr
    )

...while ``folded_nnz`` correctly counts EVERYTHING post-stacking
(``fold_block_group`` converts dense entries to CSR too). That asymmetry
made ``"lossless stacking"`` read ``False`` whenever any dense-format layer
participated in a fold group -- not because stacking actually lost or
duplicated data, but because the "before" count was silently smaller than
the true content it was supposed to represent. Fixed with a ``_real_nnz``
helper that counts ``values().numel()`` for CSR and ``(t != 0).sum()`` for
dense/strided alike.

.. _rnn_all.zero_init_transform_overview:

RNN-all: per-layer recurrent extension, zero-init, no new synapses
------------------------------------------------------------------------

*ID:* ``rnn_all.zero_init_transform_overview``

A separate, opt-in transform (``--rnn-all``) from block-folding: for every
linear weight ``W`` of shape ``[out, in]`` (or conv weight
``[out, in_c, *k]``), extend the input dimension so the layer accepts
``cat(x, state)``:

.. code-block:: text

    W_extended : [out, in + out]        # linear
    W_extended : [out, in_c + out, *k]  # conv (out channels appended to in_c)

    cols 0..in-1        : original feedforward weights (unchanged)
    cols in..in+out-1   : recurrent block (zero-sparse at conversion time)

    y     = W_ff @ x + W_rec @ state    # W_ff = cols[:in], W_rec = cols[in:]
    state = y                            # update for next call

**Zero init.** ``state`` starts all-zeros, and the recurrent block starts
with no nonzero CSR entries -- so the first forward pass is bit-identical to
the original model. Synaptogenesis (a separate process) grows nonzeros into
the recurrent block over time, using the zero-init trajectory as a
supervised signal. Because no new synapses exist at conversion time, file
size does not increase (new CSR columns are free -- they just extend the
shape metadata).

**Conv handling.** For a conv kernel ``[out_c, in_c, kH, kW]`` the recurrent
state is a feature map ``[out_c, H, W]``. The kernel extends to
``[out_c, in_c+out_c, kH, kW]``; at call time the state is spatially aligned
with ``x`` and concatenated along the channel dimension before convolution.

**Skipped layers.** Embeddings, biases, layer-norm parameters, and 1-D
weights are not extended -- they are not linear projections in the matmul
sense (see ``_is_rnn_all_eligible``'s skip-pattern regexes).

.. _main.rnn_fold_flag_bug:

CLI bug: ``--rnn-fold`` was never registered as a flag
-------------------------------------------------------------

*ID:* ``main.rnn_fold_flag_bug``

Folding is the default operation; ``--rnn-all`` is the opt-in per-layer
extension. ``--rnn-fold`` was never added to the ``argparse`` parser
(pre-existing bug), so the original ``if args.rnn_fold or not
args.rnn_all:`` would crash with ``AttributeError`` the moment it ran. The
corrected intent: fold unless ``--rnn-all`` was the ONLY flag passed
(meaning the user explicitly opted out of folding) -- ``if not
args.rnn_all:``.
