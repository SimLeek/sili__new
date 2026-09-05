``rl_utils.py`` research notes
===============================

Companion doc to ``sili/rl_utils.py``. Source comments point back here by
anchor ID (``*ID:* `` marker under each heading below); this doc links back
to source by file/symbol name. See ``docs/research/linear_disldo.rst`` for
the pattern this follows (semantic dotted anchor IDs, visible ID markers,
frozen code snippets on real-bug/non-obvious-derivation sections).

.. _popart.purpose:

Why PopArt: normalize the target, not just clip the loss
----------------------------------------------------------

*ID:* ``popart.purpose``

PopArt (van Hasselt et al. 2016; used in rtrl/rtac.py via ``rtrl.nn.PopArt``)
learns a running mean/std of the value target via EMA and trains the critic
against the *normalized* target, so gradient magnitudes stay well-behaved
regardless of the raw reward scale. That much is a standard running
normalizer -- the "Pop" half is the part that's easy to get wrong and the
reason this class exists as a dedicated unit rather than an inline EMA.

.. _popart.pop_invariance:

The Pop half: renormalization must not perturb existing predictions
-----------------------------------------------------------------------

*ID:* ``popart.pop_invariance``

Whenever mean/std update, the output layer's weights must be rescaled so
predictions *in the original (unnormalized) space* are unchanged by the
renormalization event itself -- only by actual learning. Without this,
every EMA update of mean/std silently perturbs every existing prediction,
which fights the critic's own gradient updates: the critic would be forced
to spend gradient steps re-correcting for a shift that carried zero
information about the environment.

.. _popart.derivation:

Derivation for a per-action bias row (not a single scalar bias)
---------------------------------------------------------------------

*ID:* ``popart.derivation``

For a linear value head ``v(h,a) = h @ W + b(a)``, where ``b(a)`` is itself
a per-action row of a "bias" matrix (since ``sili``'s value head is
conditioned on the one-hot previous action rather than a single additive
scalar bias -- see ``test_mandelbrot_rl.py``'s ``Wv_h`` / ``Wv_a``):

.. code-block:: text

    normalized_pred = h @ W + b
    original_pred    = normalized_pred * std_old + mean_old
    We want original_pred to be reproduced by the NEW normalization:
        original_pred = new_normalized_pred * std_new + mean_new
    Solving for new_normalized_pred:
        new_normalized_pred = (normalized_pred*std_old + mean_old - mean_new) / std_new
                             = h @ (W * std_old/std_new)
                               + (b*std_old + mean_old - mean_new) / std_new
    So: W_new = W_old * scale            (weight-like arrays: SCALE ONLY)
        b_new = b_old * scale + shift     (bias-like arrays: SCALE + SHIFT)
    where scale = std_old/std_new, shift = (mean_old - mean_new)/std_new.

.. _popart.generalization:

Generalizes to any additive term, not just a single scalar bias
---------------------------------------------------------------------

*ID:* ``popart.generalization``

This generalizes beyond a single scalar bias to any array that acts as an
additive term in the pre-normalization prediction -- e.g. every row of a
one-hot-selected bias matrix gets the same shift, since exactly one row is
ever active per prediction. That's why ``update_and_rescale`` takes
``weight_arrays``/``bias_arrays`` as lists rather than a single ``(W, b)``
pair: any number of arrays can be classified into "multiplies the input"
(scale only) vs. "adds independent of the input" (scale + shift), and each
gets rescaled in place accordingly.

.. _popart.start_pop_guard:

``start_pop`` guard: don't rescale from noisy early statistics
---------------------------------------------------------------------

*ID:* ``popart.start_pop_guard``

Below ``start_pop`` samples, mean/std still update every call (so
statistics warm up from the first sample), but the weight rescale is
skipped -- scale/shift computed from noisy early statistics would perturb
the weights more than they should move. Mirrors ``rtrl.nn.PopArt``'s own
``start_pop`` guard.
