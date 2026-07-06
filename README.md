# sili -- Sparse Intelligence Library

A research codebase for homeostatic, energy-driven sparse neural networks:
FP4-quantized delta-CSR weight storage, synaptogenesis (runtime growth and
pruning of connections), and integrate-and-fire-style energy dynamics that
produce curiosity and predictive coding as emergent effects of homeostatic
pressure rather than explicit objectives. A C++/pybind11 core handles the
sparse compute; a thin Python layer provides autograd, RNN/transformer
building blocks, and a model-conversion pipeline for folding real
transformer checkpoints into sparse runtime layers.

This is an active, private research repository, not a published package.

## Repository layout

```
sili/                      Python package
  tensor.py                 Minimal autograd Tensor (matmul, add, tanh, relu,
                             power, reduce_sum, abs, combine_losses)
  energy.py                 EnergyDynamics: homeostatic integrate-and-fire
                             gating, KL sparsity, curiosity/aux-loss signal
  sparse_rnn.py              FoldedLayer (sparse FP4 delta-CSR layer wired
                             into the autograd graph), SparseRNNCell
  module.py                 Module base class, dense Linear/RMSNorm
  cpu.py / backend.py       Backend registration for the compiled extension
  rl_utils.py               PopArt output normalization
  cpu_backend.cpp           pybind11 bindings -> sili._cpu
  lib/headers/*.hpp         C++ core: CSR/COO/delta-CSR sparse formats,
                             FP4 quantization, SISLDO/DISLDO linear layers
                             with synaptogenesis, attention, Hoyer sparsity,
                             parallel utilities
  conversion/               Model conversion pipeline:
    sparse_prune.py           dense weights -> pruned CSR
    rnn_fold.py                repeated transformer blocks -> one
                                FoldedBlockDescriptor (block detection,
                                vertical CSR stacking, attention-band masks)
    streaming_prune.py        layer-by-layer conversion for models larger
                                than available RAM (two-phase: per-tensor
                                sparsify, then per-suffix fold with an
                                over-budget --no-stack fallback)
    model_reconstruct.py, sparse_runtime.py, trace_model.py

test/                       C++ unit tests (Catch2) + legacy Python scripts.
                             Fast; run on every commit.
  run_tests.sh                Runs both run_cpp_tests.sh and run_py_tests.sh

tests/integration/          Python integration tests: slower, validate that
                             components work together end-to-end (energy +
                             autograd RNNs, sparse FoldedLayer training,
                             transformer attention vs. a PyTorch baseline,
                             the toy-Mistral conversion pipeline, streaming
                             conversion against a real on-disk safetensors
                             checkpoint, curiosity RL). Should pass before
                             merging a branch; not required on every commit.

docs/                       Hand-written design and requirements docs.
                             docs/doxygen/ and docs/pdoc/ are generated API
                             documentation output (gitignored, not source).

backburner/, examples/      Parked ideas and worked examples referenced from
                             the docs below.
```

## Active planning documents

- `refactoring_todo.md` -- the active priority queue: what's in progress,
  what's blocked, and what's explicitly deferred to later (MoE expert-merge,
  conv-kernel sparsification, RTAC's critic-through-trunk and replay buffer,
  per-patch vision + spatial merge).
- `refactoring_done.md` -- a log of completed refactoring milestones.
- `docs/requirements_vlm_streaming_rtac.md` -- the current major workstream:
  a vision-language model pipeline (verified against the real
  Mistral-Small-3.1-24B-Base-2503 checkpoint schema), layer-by-layer
  streaming conversion for RAM-limited machines, and an RTAC-based curiosity
  agent. Includes a follow-up checklist tracking what's done vs. open.
- `TODO.md` -- older backlog items not yet triaged into the above.
- `energy-params.md`, `energy-personality.md`, `energy-proofs.md` --
  design notes on the energy dynamics model, personality-trait parameter
  mappings, and supporting derivations.

## Building

```bash
pip install -e .
```

This compiles `sili/cpu_backend.cpp` into `sili._cpu` via pybind11
(`setup.py`; requires a C++20 compiler and OpenMP). Python dependencies are
pinned in `requirements.txt`.

**Import note:** always import the compiled extension via `from sili import
_cpu` (or transitively through `import sili`), never as a bare top-level
`import _cpu`. The two resolve to different `sys.modules` keys for the same
`.so` file, and pybind11 will raise `generic_type: ... already registered`
if both paths ever get exercised in one process.

## Testing

```bash
test/run_tests.sh              # C++ unit tests + legacy Python scripts (fast)
python -m tests.integration.<name>   # any individual integration test
```

Each integration test is runnable standalone and takes `--quiet` for
pass/fail-only output, e.g.:

```bash
python -m tests.integration.test_energy_rnn --task rare --steps 2000
python -m tests.integration.test_mandelbrot_rl --compare --core zero --steps 150000 --timeout 3600 --display
python -m tests.integration.test_toy_mistral
```

## Contributing (internal)

`main` is protected -- changes land via branches merged through pull
requests, not direct commits. Current branch sequence:

1. `docs/api-docs-and-readme` (this branch) -- Doxygen for the C++ headers,
   a Python API doc generator, this README, and TODO/requirements cleanup.
2. Continuous integration -- run the full test suite (including
   `tests/integration/`) and generate documentation artifacts on every PR.
3. Continue the original consolidation plan: merge remaining old `sili`
   repositories in, remove superseded files, per `refactoring_todo.md`.
