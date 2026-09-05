``bench_block4_memory_and_compression.cpp`` research notes
==============================================================

Companion doc to ``scripts/bench_block4_memory_and_compression.cpp``.
Speed cost of real block4 compression's pack/unpack/resize machinery. The
correctness side of this question (memory never exceeds budget, compression
genuinely shrinks real bytes) is a permanent, hard-CHECK ctest test -- see
``tests/unit/test_block4_memory_cap_and_compression.cpp``. This script is
the informational timing companion, NOT wired into ctest (a timing report,
not a pass/fail correctness check, matching ``bench_block4_vs_dense_fp4.cpp``'s
own convention).

.. _bench_block4_memory_and_compression.online_training_no_compression:

Real online training never exercises compression at all (measured)
------------------------------------------------------------------------

*ID:* ``bench_block4_memory_and_compression.online_training_no_compression``

A full online-training loop (forward+backward+synaptogenesis every step)
turned out NOT to exercise compression at all: ``disldo_backward`` writes
every one of a touched tile's 16 slots on every call (see ``block4.hpp``),
so an actively-trained tile is forced back to full occupancy before
compression has any lasting effect. Confirmed by an earlier version of
this script, which measured 1.00x compression (i.e. none) under that
workload.

.. _bench_block4_memory_and_compression.isolated_toggle_design:

Why this benchmark drives pack/unpack directly instead of training
------------------------------------------------------------------------

*ID:* ``bench_block4_memory_and_compression.isolated_toggle_design``

Real compression's value is for tiles NOT under active per-step training
(a settled connection, an inference-heavy phase, a partial-batch
schedule) -- so this benchmark instead directly drives the pack/unpack/
resize machinery: repeatedly nudge each tile's live count up across
``switch_point`` (forcing a real sparse->dense resize in the handle
destructor) then back down (forcing a real dense->sparse resize via
``maybe_compress``), and times that against the identical toggle sequence
with compression disabled (``switch_point=0``, so no tile is ever packed --
pure in-place dense writes, zero resize cost) to isolate what real
compression costs, not what the whole online-learning pipeline costs.
