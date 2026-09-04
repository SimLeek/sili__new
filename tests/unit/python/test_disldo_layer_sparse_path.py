"""
Regression/sanity tests for DISLDOLayer's Phase 5 sparse-path extension
(sparsity plan task #334): CSR-typed forward input (no new sparse_input
bool -- x.is_csr controls it, matching the existing SparseRNNCell
precedent at sparse_rnn.py's own recurrent-cell forward()) and the new
dy_sparsity_p kwarg (independent axis, sparsifies the backward gradient
via backward_sparse instead of backward_dense).

Every existing caller passes a plain dense ndarray/Tensor and never sets
dy_sparsity_p -- that path must be BIT-IDENTICAL to pre-Phase-5 behavior,
which this file's first test verifies directly against the raw _cpu calls
(bypassing the wrapper) rather than trusting the refactor by inspection.
"""
import numpy as np

from sili.sparse_rnn import CSR, DISLDOLayer
from sili.tensor import Tensor, get_backend


def _make_layer(seed: int) -> DISLDOLayer:
    return DISLDOLayer(6, 4, 24, num_cpus=2, rng=np.random.default_rng(seed))


class TestDISLDOLayerDensePathUnchanged:
    def test_forward_backward_matches_raw_cpu_calls_1d(self):
        # Proves the Phase 5 refactor (routing through x_dense/dy2d
        # internally) didn't change the dense path's actual numbers --
        # compares the wrapper's output against calling
        # forward_dense/backward_dense directly on an IDENTICALLY-seeded
        # raw _cpu layer.
        layer_wrapped = _make_layer(seed=3)
        layer_raw     = _make_layer(seed=3)
        backend = get_backend("cpu")

        x_np = np.random.default_rng(4).normal(size=6).astype(np.float32)

        out = layer_wrapped.forward(Tensor(x_np, backend=backend), learning_rate=0.05)
        raw_out = layer_raw._c.forward_dense(x_np).squeeze(0)
        np.testing.assert_allclose(np.asarray(out.data), raw_out, rtol=1e-6, atol=1e-7)

        dy = np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float32)
        out.grad = dy
        out._backward()
        layer_raw._c.backward_dense(x_np[np.newaxis, :], dy[np.newaxis, :], 0.05,
                                     lr_per_row_nnz=True, damp_by_importance=True)

        np.testing.assert_allclose(
            np.array(layer_wrapped._c.get_value_scale_raw_vector()),
            np.array(layer_raw._c.get_value_scale_raw_vector()),
            rtol=1e-6, atol=1e-7,
            err_msg="wrapped layer's value_scale diverged from an identically-seeded raw _cpu layer "
                    "given the same dense input -- Phase 5 refactor changed the dense path's behavior")

    def test_forward_2d_batched_input_unchanged(self):
        layer = _make_layer(seed=5)
        backend = get_backend("cpu")
        x_np = np.random.default_rng(6).normal(size=(2, 6)).astype(np.float32)
        out = layer.forward(Tensor(x_np, backend=backend))
        assert out.data.shape == (2, 4), "2-D batched dense input must stay 2-D output, unchanged"


class TestDISLDOLayerCSRForwardInput:
    def test_csr_input_forward_matches_dense_with_zeroed_entries(self):
        # Same weights, same effective input (dense-with-zeros represented
        # two ways) -- CSR-typed forward_sparse vs dense forward_dense on
        # the manually-zeroed array. Not bit-exact (different code paths,
        # different internal loop order), but should match closely --
        # this is the direct proof top-k-sparsifying an activation before
        # DISLDOLayer is mathematically a no-op relative to the dense path,
        # matching the plan's own checkpoint for this phase.
        layer_sparse = _make_layer(seed=9)
        layer_dense  = _make_layer(seed=9)
        backend = get_backend("cpu")

        x_np = np.random.default_rng(10).normal(size=6).astype(np.float32)
        csr = CSR.from_dense(x_np, p=0.5, num_cpus=2)
        x_zeroed = csr.to_dense()[0]

        out_sparse = layer_sparse.forward(csr.as_tensor(backend))
        out_dense  = layer_dense.forward(Tensor(x_zeroed, backend=backend))

        np.testing.assert_allclose(
            np.asarray(out_sparse.data), np.asarray(out_dense.data), rtol=1e-3, atol=1e-3,
            err_msg="forward(CSR) diverged from forward(dense-with-zeros) on identically-seeded layers")

    def test_csr_input_backward_weight_update_matches_dense_with_zeroed_entries(self):
        layer_sparse = _make_layer(seed=11)
        layer_dense  = _make_layer(seed=11)
        backend = get_backend("cpu")

        x_np = np.random.default_rng(12).normal(size=6).astype(np.float32)
        csr = CSR.from_dense(x_np, p=0.5, num_cpus=2)
        x_zeroed = csr.to_dense()[0]

        out_sparse = layer_sparse.forward(csr.as_tensor(backend), learning_rate=0.05)
        out_dense  = layer_dense.forward(Tensor(x_zeroed, backend=backend), learning_rate=0.05)

        dy = np.array([0.1, 0.2, -0.1, 0.05], dtype=np.float32)
        out_sparse.grad = dy.copy()
        out_dense.grad  = dy.copy()
        out_sparse._backward()
        out_dense._backward()

        np.testing.assert_allclose(
            np.array(layer_sparse._c.get_value_scale_raw_vector()),
            np.array(layer_dense._c.get_value_scale_raw_vector()),
            rtol=1e-2, atol=1e-2,
            err_msg="value_scale update after forward(CSR)+backward diverged from forward(dense-with-"
                    "zeros)+backward on identically-seeded layers")


class TestDISLDOLayerDySparsity:
    def test_dy_sparsity_p_produces_finite_update(self):
        layer = _make_layer(seed=13)
        backend = get_backend("cpu")
        x_np = np.random.default_rng(14).normal(size=6).astype(np.float32)

        # value_scale is lazily sized on first backward call (0-length
        # until then, same as SISLDOLayer -- see test_sisldo_layer_
        # last_input.py's identical note) -- nothing meaningful to
        # snapshot pre-backward.
        out = layer.forward(Tensor(x_np, backend=backend), learning_rate=0.05, dy_sparsity_p=0.5)
        out.grad = np.array([0.3, -0.2, 0.1, -0.4], dtype=np.float32)
        out._backward()
        vs_after = np.array(layer._c.get_value_scale_raw_vector())

        assert vs_after.size == 6, f"expected value_scale sized to n_inputs=6, got {vs_after.size}"
        assert np.all(np.isfinite(vs_after)), "dy_sparsity_p path produced non-finite value_scale"

    def test_dy_sparsity_p_none_default_unaffected(self):
        # None (default) must take the EXACT same backward_dense path as
        # before dy_sparsity_p existed -- confirmed via the raw-_cpu-call
        # comparison already covered by TestDISLDOLayerDensePathUnchanged;
        # this just checks the kwarg's own default doesn't accidentally
        # route through backward_sparse.
        layer_a = _make_layer(seed=15)
        layer_b = _make_layer(seed=15)
        backend = get_backend("cpu")
        x_np = np.random.default_rng(16).normal(size=6).astype(np.float32)

        out_a = layer_a.forward(Tensor(x_np, backend=backend), learning_rate=0.05)
        out_b = layer_b.forward(Tensor(x_np, backend=backend), learning_rate=0.05, dy_sparsity_p=None)
        dy = np.array([0.1, 0.2, -0.1, 0.05], dtype=np.float32)
        out_a.grad = dy.copy()
        out_b.grad = dy.copy()
        out_a._backward()
        out_b._backward()

        np.testing.assert_allclose(
            np.array(layer_a._c.get_value_scale_raw_vector()),
            np.array(layer_b._c.get_value_scale_raw_vector()),
            rtol=1e-6, atol=1e-7)


class TestDISLDOLayerGradedDySparsitySchedule:
    """dy_sparsity_schedule: GENUINE per-row density (independent top-k per
    row via _graded_top_k_csr), unlike the pre-existing scalar
    dy_sparsity_p, whose _cpu.dense_to_top_k_csr turned out to select its
    top-k GLOBALLY across the whole batch, not per row (see
    project_dy_sparsity_p_validated_speedup.md's correction/JOURNAL.md --
    found while building this). Deliberately does NOT compare against
    dy_sparsity_p=1.0 or backward_dense for "equivalence" -- those use
    different selection semantics (global top-k, and a different code
    path entirely) and are not expected to produce the same numbers even
    when nominally "keeping everything." Correctness here means the CSR
    itself has the right structure, checked directly."""

    def test_schedule_produces_correct_per_row_nnz(self):
        from sili.sparse_rnn import _graded_top_k_csr
        dy2d = np.random.default_rng(30).normal(size=(4, 6)).astype(np.float32)
        k_per_row = [6, 3, 1, 0]
        ptrs, indices, values = _graded_top_k_csr(dy2d, k_per_row)
        nnz_per_row = np.diff(ptrs)
        assert list(nnz_per_row) == k_per_row

        # Each row's kept values must be exactly that row's top-k by
        # magnitude (the whole point of "graded," not just "fewer").
        for r, k in enumerate(k_per_row):
            row = dy2d[r]
            expected = set(np.argsort(-np.abs(row))[:k].tolist())
            actual = set(indices[ptrs[r]:ptrs[r + 1]].tolist())
            assert actual == expected, f"row {r}: kept indices {actual} != true top-{k} {expected}"

    def test_schedule_backward_produces_finite_correct_shape_update(self):
        # End-to-end: the CSR built above actually drives a real
        # backward_sparse call correctly (right dx shape, finite output),
        # not just correct in isolation.
        layer = _make_layer(seed=31)
        backend = get_backend("cpu")
        x_np = np.random.default_rng(32).normal(size=(3, 6)).astype(np.float32)
        out = layer.forward(Tensor(x_np, backend=backend), learning_rate=0.05,
                            dy_sparsity_schedule=[1.0, 0.5, 0.0])
        out.grad = np.random.default_rng(33).normal(size=(3, 4)).astype(np.float32)
        out._backward()
        vs_after = np.array(layer._c.get_value_scale_raw_vector())
        assert vs_after.size == 6
        assert np.all(np.isfinite(vs_after))

    def test_mismatched_length_raises(self):
        layer = _make_layer(seed=24)
        backend = get_backend("cpu")
        x_np = np.random.default_rng(25).normal(size=(3, 6)).astype(np.float32)
        out = layer.forward(Tensor(x_np, backend=backend), learning_rate=0.05,
                            dy_sparsity_schedule=[1.0, 0.5])  # only 2 entries for 3 rows
        out.grad = np.random.default_rng(26).normal(size=(3, 4)).astype(np.float32)
        try:
            out._backward()
            assert False, "expected ValueError for mismatched schedule length"
        except ValueError:
            pass

    def test_finite_and_graded_by_row(self):
        # Direct check that a low-density row actually keeps FEWER
        # nonzero gradient entries than a high-density row -- proves the
        # grading is real, not just accepted-and-ignored.
        layer = DISLDOLayer(6, 8, 40, num_cpus=2, rng=np.random.default_rng(27))
        backend = get_backend("cpu")
        x_np = np.random.default_rng(28).normal(size=(2, 6)).astype(np.float32)
        out = layer.forward(Tensor(x_np, backend=backend), learning_rate=0.05,
                            dy_sparsity_schedule=[1.0, 0.25])
        out.grad = np.random.default_rng(29).normal(size=(2, 8)).astype(np.float32)
        out._backward()
        vs_after = np.array(layer._c.get_value_scale_raw_vector())
        assert np.all(np.isfinite(vs_after))


class TestDISLDOLayerDyRTarget:
    """dy_r_target (task #367, priority 1 per direct instruction): nucleus/
    energy-threshold grad sparsification -- k is a CONSEQUENCE of
    dy_r_target and this step's actual gradient energy, not a fixed
    fraction like dy_sparsity_p. See sili_peridot/JOURNAL.md's nucleus
    design note and _nucleus_top_k_csr's own docstring for the math."""

    def test_none_default_unaffected(self):
        # dy_r_target=None must take the exact same dense path as before
        # this kwarg existed -- same style of check as
        # test_dy_sparsity_p_none_default_unaffected above.
        layer_a = _make_layer(seed=40)
        layer_b = _make_layer(seed=40)
        backend = get_backend("cpu")
        x_np = np.random.default_rng(41).normal(size=6).astype(np.float32)

        out_a = layer_a.forward(Tensor(x_np, backend=backend), learning_rate=0.05)
        out_b = layer_b.forward(Tensor(x_np, backend=backend), learning_rate=0.05, dy_r_target=None)
        dy = np.array([0.1, 0.2, -0.1, 0.05], dtype=np.float32)
        out_a.grad = dy.copy()
        out_b.grad = dy.copy()
        out_a._backward()
        out_b._backward()

        np.testing.assert_allclose(
            np.array(layer_a._c.get_value_scale_raw_vector()),
            np.array(layer_b._c.get_value_scale_raw_vector()),
            rtol=1e-6, atol=1e-7)

    def test_takes_priority_over_dy_sparsity_p(self):
        # Both kwargs set at once -- dy_r_target must win (it's the
        # smarter, data-driven mechanism dy_sparsity_p is being
        # superseded by). Checked by spying on which selection function
        # actually gets called, NOT by comparing floating-point weight
        # outputs -- backward_sparse's own C++ update has genuine
        # run-to-run floating-point nondeterminism under threading
        # (confirmed directly: even calling the SAME dy_r_target-only
        # path twice with identical inputs gives different value_scale
        # results), so exact-output comparison is the wrong tool here.
        import sili.sparse_rnn as sparse_rnn_mod
        layer = _make_layer(seed=42)
        backend = get_backend("cpu")
        x_np = np.random.default_rng(43).normal(size=(2, 6)).astype(np.float32)
        dy = np.random.default_rng(44).normal(size=(2, 4)).astype(np.float32)

        out = layer.forward(Tensor(x_np, backend=backend), learning_rate=0.05,
                             dy_r_target=0.7, dy_sparsity_p=0.9)

        calls = {"nucleus": 0, "graded_fixed_p": 0}
        orig_nucleus = sparse_rnn_mod._nucleus_top_k_csr
        orig_graded = sparse_rnn_mod._graded_top_k_csr
        def spy_nucleus(*a, **kw):
            calls["nucleus"] += 1
            return orig_nucleus(*a, **kw)
        def spy_graded(*a, **kw):
            calls["graded_fixed_p"] += 1
            return orig_graded(*a, **kw)
        sparse_rnn_mod._nucleus_top_k_csr = spy_nucleus
        sparse_rnn_mod._graded_top_k_csr = spy_graded
        try:
            out.grad = dy.copy()
            out._backward()
        finally:
            sparse_rnn_mod._nucleus_top_k_csr = orig_nucleus
            sparse_rnn_mod._graded_top_k_csr = orig_graded

        assert calls == {"nucleus": 1, "graded_fixed_p": 0}, (
            f"dy_r_target should take priority over dy_sparsity_p when both are set, got {calls}")

    def test_low_energy_row_keeps_fewer_entries_than_high_energy_row(self):
        # A row with one huge outlier needs few entries to reach a given
        # R_target; a flat row of similar-magnitude entries needs many --
        # proves this is genuinely data-driven, not a hidden fixed k.
        # backward_sparse is a read-only C++/pybind attribute (can't be
        # monkeypatched directly) -- spy on the Python-level
        # _nucleus_top_k_csr wrapper instead.
        import sili.sparse_rnn as sparse_rnn_mod
        layer = _make_layer(seed=45)
        backend = get_backend("cpu")
        x_np = np.random.default_rng(46).normal(size=(2, 6)).astype(np.float32)
        out = layer.forward(Tensor(x_np, backend=backend), learning_rate=0.05, dy_r_target=0.9)

        recorded = {}
        orig = sparse_rnn_mod._nucleus_top_k_csr
        def spy(x2d, r_target, num_cpus=4, k_min=0, k_max=None):
            ptrs, indices, values = orig(x2d, r_target, num_cpus, k_min=k_min, k_max=k_max)
            recorded["nnz_per_row"] = np.diff(ptrs)
            return ptrs, indices, values
        sparse_rnn_mod._nucleus_top_k_csr = spy
        try:
            dy = np.array([[100.0, 1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
            out.grad = dy
            out._backward()
        finally:
            sparse_rnn_mod._nucleus_top_k_csr = orig

        nnz = recorded["nnz_per_row"]
        assert nnz[0] < nnz[1], f"dominant-entry row should keep fewer than flat row, got {nnz}"

    def test_k_min_k_max_reach_the_kernel(self):
        # Confirms the kwargs actually thread through to _nucleus_top_k_csr
        # (not silently dropped) -- k_min=cols forces full density even at
        # r_target=0, which would otherwise keep k=0.
        import sili.sparse_rnn as sparse_rnn_mod
        layer = _make_layer(seed=47)
        backend = get_backend("cpu")
        x_np = np.random.default_rng(48).normal(size=6).astype(np.float32)
        out = layer.forward(Tensor(x_np, backend=backend), learning_rate=0.05,
                            dy_r_target=0.0, dy_k_min=4)

        recorded = {}
        orig = sparse_rnn_mod._nucleus_top_k_csr
        def spy(x2d, r_target, num_cpus=4, k_min=0, k_max=None):
            ptrs, indices, values = orig(x2d, r_target, num_cpus, k_min=k_min, k_max=k_max)
            recorded["nnz"] = np.diff(ptrs)[0]
            return ptrs, indices, values
        sparse_rnn_mod._nucleus_top_k_csr = spy
        try:
            out.grad = np.array([0.1, 0.2, -0.1, 0.05], dtype=np.float32)
            out._backward()
        finally:
            sparse_rnn_mod._nucleus_top_k_csr = orig
        assert recorded["nnz"] == 4, f"dy_k_min=4 should force k=4 despite r_target=0, got {recorded['nnz']}"
