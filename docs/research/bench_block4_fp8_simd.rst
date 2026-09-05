``bench_block4_fp8_simd.cpp`` research notes
===============================================

Companion doc to ``scripts/bench_block4_fp8_simd.cpp``. Same benchmarking
convention as ``scripts/bench_block4_vs_dense_fp4.cpp`` -- build once per
variant via ``SILI_BLOCK4_FORCE_SCALAR_BACKWARD``, run each variant in a
fresh process (avoids the thermal/frequency-scaling confound that file's
own comment documents). NOT wired into ctest -- a timing report, not a
pass/fail correctness check.

.. _bench_block4_fp8_simd.measured_results:

Measured: SIMD accumulate wins at batch>1, ties at batch=1
--------------------------------------------------------------

*ID:* ``bench_block4_fp8_simd.measured_results``

512x512 100%-dense block4, batch arg, best-of-200, ``-O3 -ffast-math
-march=native``:

- ``batch=1``: SIMD 0.0048s vs scalar 0.0048s -- tied, no measurable win or
  loss (the accumulate loop's own inner batch-loop only runs once, so it
  never amortizes its own setup cost).
- ``batch=32``: SIMD 0.0227s vs scalar 0.0300s -- SIMD ~24% faster, a real
  win. Confirmed via ``objdump`` that this is genuine 128-bit packed SIMD
  (``vmulps``/``vrsqrtps``/``vaddps`` on xmm registers) inside
  ``disldo_backward<..., FP8BiValues, ...>``'s ``process_tile`` lambda, not
  GCC auto-vectorizing scalar code.

.. _bench_block4_fp8_simd.scalar_decode_simd_accumulate:

Why FP8's decode/encode stays scalar but accumulation goes SIMD
---------------------------------------------------------------------

*ID:* ``bench_block4_fp8_simd.scalar_decode_simd_accumulate``

This measurement is WHY ``linear_disldo.hpp``'s FP8 branch decodes/encodes
via scalar ``fp8_decode_bits``/``fp8_quantize_stochastic`` (not
``block4_vec_decode_fp8``/``block4_vec_quantize_stochastic_fp8``, which
measurably LOSE here) but keeps the batch-loop accumulation math (RMSprop
update, ``dx``) as SIMD (``Block4Vec``) -- identical to FP4's own
accumulate loop once weight/importance are decoded to float, and the one
piece that measurably earns its complexity at realistic (batch>1) sizes.

E4M3's 256-code space makes the vectorized decode/encode path's
subnormal/NaN-lane scalar-correction fallback real overhead that FP4's
simpler 16-code E2M1 never pays -- FP4 has no such fallback cost, so its
vectorized decode/encode wins outright, while FP8's doesn't.
