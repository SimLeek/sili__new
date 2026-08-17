# Testing sili__new

Run these from the repo root (`sili__new/`) using the project's venv
(`/home/simleek/claude_code/.venv/bin/python` / `pip`).

## 1. Check that the C++ extension passes unit tests

```bash
cd tests/unit/
./run_cpp_tests.sh
```

After that builds and runs the tests, you can also launch specific tests via:

```bash
./build_tests/test_name
```

## 2. Rebuild the C++ Python extension

Any change under `sili/lib/headers/*.hpp` or `sili/cpu_backend.cpp`
requires a rebuild before Python sees it:

```bash
/home/simleek/claude_code/.venv/bin/pip install -e . --no-build-isolation
```

A clean rebuild takes 1-3 minutes. Watch for `static assertion failed:
The number of argument annotations does not match the number of
function arguments` -- this means a pybind `.def(...)` call's
`py::arg(...)` list doesn't match the actual C++ function signature.
Several classes (`SparseLinearLayer`, `SparseLinearLayerResync`,
`SparseLinearLayerNoScale`, `SparseLinearLayerDeterministic`,
`SparseLinearLayerResyncDeterministic`,
`SparseLinearLayerNoScaleDeterministic`) all instantiate the SAME
`SparseLinearLayerImpl` template -- changing a shared method's
signature (e.g. `build_probes`/`synap_row_step`/`synap_step`) requires
updating the pybind bindings for ALL SIX classes, not just the one you
were testing with.
todo: that is an extremely obvious code duplication and shotgun surgery issue. Fix it.

## 2. Python unit tests

Most of the model and training code is in python, so python verifies that 
AIs can learn while C++ tests verify the math and basic functionality is correct.

```bash
/home/simleek/claude_code/.venv/bin/python -m pytest tests/unit/python/ -q
```

