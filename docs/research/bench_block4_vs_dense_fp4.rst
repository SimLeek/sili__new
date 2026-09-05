``bench_block4_vs_dense_fp4.cpp`` research notes
====================================================

Companion doc to ``scripts/bench_block4_vs_dense_fp4.cpp``. Fair
block4-vs-dense-FP4-floor benchmark, batch=1 (sili's real-time target --
see ``TODO_DUAL_BLOCK4.md``'s Part C). NOT wired into ctest -- a timing
report, not a pass/fail correctness check; matches
``scripts/bench_block4_layer.py``'s convention (benchmark, not test).

.. _bench_block4_vs_dense_fp4.methodology_bugs:

Two real methodology bugs found and fixed, and the corrected result
------------------------------------------------------------------------

*ID:* ``bench_block4_vs_dense_fp4.methodology_bugs``

1. An earlier "dense floor" used plain float32 weights with no FP4 cost
   at all, making block4 look ~20x off. FP4 encode/decode is required
   regardless of representation -- not a cost to blame on block4. The
   floor here decodes/encodes through the SAME ``FP4_TABLE``/
   ``fp4_quantize_stochastic`` block4 itself uses.

2. A later "dense floor" got the FP4 codec right but still only did a
   bare weight update (no value_scale/output_scale composition, no
   per-synapse importance tracking/damping, deterministic not stochastic
   re-quantization) -- while the block4 benchmark it was compared against
   used ``learning_rate=0``, which skips ALL of that same work on
   block4's side too, but for a DIFFERENT (accidental) reason. Neither
   comparison was apples-to-apples. This benchmark runs BOTH sides
   through the real block4 ``disldo_forward``/``disldo_backward`` feature
   set (value_scale/output_scale, ``damp_by_importance``, stochastic
   re-quantize of both weight and importance nibbles) at a real, nonzero
   ``learning_rate``, matching what real online learning actually pays.

Result once corrected: block4 backward beats the fair dense floor at
EVERY density tested, including 100% (the earlier "~3x slower" framing
was measuring an unfair baseline, not a real block4 shortfall). Forward
remains genuinely slower (~5x) since forward at ``learning_rate=0`` --
the case this library's block4 population is normally read through --
never touches the importance/stochastic machinery that backward's SIMD
wins on.

.. _bench_block4_vs_dense_fp4.fresh_process_per_density:

Run once per density point, in a fresh process
----------------------------------------------------

*ID:* ``bench_block4_vs_dense_fp4.fresh_process_per_density``

Usage: run once per density point, in a FRESH process each time -- see
``TODO_DUAL_BLOCK4.md``'s Part C for why a bare sequential loop over
multiple densities in one process is unreliable (a real thermal/
frequency-scaling confound hit twice in this file's history).
