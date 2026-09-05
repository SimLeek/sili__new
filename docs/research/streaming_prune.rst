``streaming_prune.py`` research notes
========================================

Companion doc to ``sili/conversion/streaming_prune.py``: layer-by-layer
model conversion for RAM-limited boxes.

.. _streaming_prune.two_phase_architecture:

Why two streaming phases, and the peak-memory bound each one holds
------------------------------------------------------------------------

*ID:* ``streaming_prune.two_phase_architecture``

TOP PRIORITY (see ``docs/requirements_vlm_streaming_rtac.md`` section 3):
the conversion must run on the same 32-96 GB machine that runs the model.
A 24B bf16 checkpoint is ~48 GB; whole-state-dict loading is marginal at
96 GB and impossible at 32. safetensors' ``safe_open`` loads ONE tensor
at a time, which makes two-phase streaming trivial:

- **Phase 1** ``streaming_sparsify``: for each tensor (across shards),
  load -> prune -> CSR -> save to ``out_dir/tensors/<name>.pt`` -> free.
  Peak memory = one dense tensor + its CSR (largest 24B tensor:
  ``down_proj``, 671 MB fp32). Resumable via ``manifest.json``.
- **Phase 2** ``streaming_fold_suffix``: load ONE parameter suffix across
  all layers sequentially and stack. Peak = that suffix's total CSR nnz.
  ``down_proj`` x40 at density d is ~54 GB * d -- so d=0.10 fits 32 GB but
  d=0.50 does not. When the manifest-predicted nnz exceeds
  ``mem_budget_gb``, falls back to per-layer descriptors (``n_folds=1``
  each, no stacking) instead of OOMing.

Follow-up edge cases (requirements doc section 7): gated-repo auth,
tied-weight dedup (resolved as a non-issue, see requirements doc section
1.2), MoE expert-merge (later TODO).

.. _streaming_prune.resume_fsck_rationale:

Resume is a real fsck, not a bare existence check
-------------------------------------------------------

*ID:* ``streaming_prune.resume_fsck_rationale``

``_tensor_file_ok()`` calls ``torch.load()`` on the existing file before
trusting it, so a truncated file from a process killed mid-write (crash,
OOM-kill) gets regenerated instead of silently accepted -- a plain
``os.path.exists()`` check would pass for such a file and corrupt the
pipeline much later, at ``streaming_fold_suffix`` time, with a far less
obvious error.

.. code-block:: python

    def _tensor_file_ok(path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            torch.load(path, weights_only=False)  # actually parse it
            return True
        except Exception:
            return False

.. _streaming_prune.conv_kernel_dense_rationale:

Why conv-kernel tensors (ndim > 2) are kept dense, not pruned
--------------------------------------------------------------------

*ID:* ``streaming_prune.conv_kernel_dense_rationale``

Conv-kernel tensors (ndim > 2, e.g. Pixtral's ``patch_conv.weight``,
shape (1024,3,14,14)) are kept DENSE, not pruned -- same treatment as
ndim <= 1 tensors (norms, biases), for two independent reasons:

(a) small in every dimension -- prior sparsification attempts on
    vision-tower-scale tensors found they don't sparsify well (density
    stays high relative to the text stack's 5120x5120 matrices), so the
    CSR overhead isn't worth it even where it's technically possible;
(b) CSR is 2-D only, so ndim>2 would need a reshape to (out, -1) first,
    adding a reconstruction step for no proven benefit at this size
    (~600K params vs the ~21M-scale text matrices where sparsity
    actually pays off).

Deferred to later-todo alongside MoE expert-merge; see
``docs/requirements_vlm_streaming_rtac.md`` section 7.
