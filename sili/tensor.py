from __future__ import annotations

import numpy as np

from sili.backend import Backend, get_backend

# ══════════════════════════════════════════════════════════════════════════════
#  Tensor
# ══════════════════════════════════════════════════════════════════════════════


class Tensor:
    def __init__(self, data, _children: tuple = (), _op: str = "", backend: Backend = None):
        self.data = data
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.backend = backend or get_backend("cpu")

    # ── device ────────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self.backend.name

    def to(self, device: str) -> Tensor:
        target = get_backend(device)
        if target == self.backend:
            return self
        return Tensor(target.move(self.data, self.backend), backend=target)

    def detach(self) -> Tensor:
        return Tensor(self.data, backend=self.backend)

    # ── sparse (CSR) data delegation ─────────────────────────────────────────
    #
    # Deliberately a separate concept from any flag marking the WEIGHT-side
    # novel delta-CSR format (that lives in the C++-backed sparse layers,
    # e.g. FoldedLayer/SparseLinearLayer) -- this is purely about whether
    # this Tensor's ACTIVATION data happens to be a sili.sparse_rnn.CSR
    # namedtuple instead of a dense ndarray.

    @property
    def is_csr(self) -> bool:
        from sili.sparse_rnn import CSR

        return isinstance(self.data, CSR)

    @property
    def is_dense(self) -> bool:
        return not self.is_csr

    def __getattr__(self, name):
        # Only invoked when normal attribute lookup fails (self.data itself
        # is always found normally -- set in __init__ -- so this never
        # recurses on that). Delegates to self.data when it's a CSR, so
        # tensor.nnz / tensor.rows / tensor.cols work directly without
        # unwrapping .data by hand.
        from sili.sparse_rnn import CSR

        data = self.data
        if isinstance(data, CSR) and hasattr(data, name):
            return getattr(data, name)
        raise AttributeError(f"'Tensor' object has no attribute {name!r} (data is {type(data).__name__}, not a CSR)")

    # ── coerce scalar to same-device Tensor ───────────────────────────────────

    def _coerce(self, other) -> Tensor:
        if isinstance(other, Tensor):
            if other.backend != self.backend:
                raise RuntimeError(
                    f"Operands on different devices: "
                    f"{self.backend.name!r} vs {other.backend.name!r}. "
                    "Use .to(device) to move one first."
                )
            return other
        return Tensor(self.backend.scalar(other), backend=self.backend)

    # ── arithmetic — delegate to op functions below ───────────────────────────

    def __add__(self, other):
        return add(self, self._coerce(other))

    def __mul__(self, other):
        return mul(self, self._coerce(other))

    def __matmul__(self, other):
        return matmul(self, self._coerce(other))

    def __pow__(self, exp):
        return power(self, exp)

    def __neg__(self):
        return neg(self)

    def relu(self):
        return relu(self)

    def tanh(self):
        return tanh(self)

    def sum(self, axis=None):
        return reduce_sum(self, axis)

    def reshape(self, shape):
        return reshape(self, shape)

    def transpose(self, axes=None):
        return transpose(self, axes)

    def bounded_gate(self, n=2.0):
        return bounded_gate(self, n)

    def __abs__(self):
        return tensor_abs(self)

    def __radd__(self, o):
        return add(self._coerce(o), self)

    def __sub__(self, o):
        return add(self, neg(self._coerce(o)))

    def __rsub__(self, o):
        return add(self._coerce(o), neg(self))

    def __rmul__(self, o):
        return mul(self._coerce(o), self)

    def __truediv__(self, o):
        return mul(self, power(self._coerce(o), -1))

    def __rtruediv__(self, o):
        return mul(self._coerce(o), power(self, -1))

    def __rmatmul__(self, o):
        return matmul(self._coerce(o), self)

    # ── backward ──────────────────────────────────────────────────────────────

    def backward(self) -> None:
        if self.grad is None:
            self.grad = self.backend.ones_like(self.data)
        for node in reversed(_topo_sort(self)):
            node._backward()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def shape(self):
        return self.backend.shape(self.data)

    @property
    def dtype(self):
        return self.backend.dtype(self.data)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Tensor(shape={self.shape}, device={self.device!r}, op={self._op!r})"


# ══════════════════════════════════════════════════════════════════════════════
#  Op library — each function owns its Tensor node + backward
# ══════════════════════════════════════════════════════════════════════════════


def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.backend.add(a.data, b.data), (a, b), "add", a.backend)

    def _bwd():
        g = np.asarray(out.grad)
        _acc(a, _unbroadcast_to(g, np.asarray(a.data).shape))
        _acc(b, _unbroadcast_to(g, np.asarray(b.data).shape))

    out._backward = _bwd
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.backend.mul(a.data, b.data), (a, b), "mul", a.backend)

    def _bwd():
        da = a.backend.mul(b.data, out.grad)
        db = b.backend.mul(a.data, out.grad)
        _acc(a, _unbroadcast_to(da, np.asarray(a.data).shape))
        _acc(b, _unbroadcast_to(db, np.asarray(b.data).shape))

    out._backward = _bwd
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.backend.matmul(a.data, b.data), (a, b), "matmul", a.backend)

    def _bwd():
        da, db = a.backend.matmul_backward(a.data, b.data, out.grad)
        _acc(a, da)
        _acc(b, db)

    out._backward = _bwd
    return out


def power(a: Tensor, exp: float) -> Tensor:
    assert isinstance(exp, int | float)
    out = Tensor(a.backend.pow(a.data, exp), (a,), f"pow{exp}", a.backend)

    def _bwd():
        _acc(a, a.backend.pow_backward(a.data, exp, out.grad))

    out._backward = _bwd
    return out


def neg(a: Tensor) -> Tensor:
    out = Tensor(a.backend.neg(a.data), (a,), "neg", a.backend)

    def _bwd():
        _acc(a, a.backend.neg(out.grad))

    out._backward = _bwd
    return out


def relu(a: Tensor) -> Tensor:
    out = Tensor(a.backend.relu(a.data), (a,), "relu", a.backend)

    def _bwd():
        _acc(a, a.backend.relu_backward(a.data, out.grad))

    out._backward = _bwd
    return out


def tanh(a: Tensor) -> Tensor:
    t = np.tanh(np.asarray(a.data, dtype=np.float32))
    out = Tensor(t, (a,), "tanh", a.backend)

    def _bwd():
        _acc(a, (1.0 - t * t) * np.asarray(out.grad, dtype=np.float32))

    out._backward = _bwd
    return out


def exp(a: Tensor) -> Tensor:
    """d(exp(x))/dx = exp(x) -- standard use is the log-sigma -> sigma
    positivity trick (train an unconstrained log_sigma parameter,
    exponentiate before using it as a strictly-positive scale/spread),
    so nothing downstream needs to special-case sign/zero."""
    e = np.exp(np.asarray(a.data, dtype=np.float32))
    out = Tensor(e, (a,), "exp", a.backend)

    def _bwd():
        _acc(a, e * np.asarray(out.grad, dtype=np.float32))

    out._backward = _bwd
    return out


def log(a: Tensor) -> Tensor:
    """d(log(x))/dx = 1/x. Mirrors exp()'s pattern exactly -- standard
    use here is a softmax-cross-entropy loss built from exp/reduce_sum/
    gather rather than a dedicated cross-entropy op."""
    l = np.log(np.asarray(a.data, dtype=np.float32))
    out = Tensor(l, (a,), "log", a.backend)

    def _bwd():
        _acc(a, np.asarray(out.grad, dtype=np.float32) / np.asarray(a.data, dtype=np.float32))

    out._backward = _bwd
    return out


def reduce_sum(a: Tensor, axis=None) -> Tensor:
    out = Tensor(a.backend.sum(a.data, axis), (a,), "sum", a.backend)

    def _bwd():
        g = np.asarray(out.grad, dtype=np.float32)
        if axis is not None:
            # backend.sum drops `axis` entirely (no keepdims) -- for any
            # axis other than 0, the reduced dim isn't trailing, so
            # broadcast_to's plain np.broadcast_to (right-aligned) would
            # try to match the wrong dimensions. Re-insert `axis` as a
            # size-1 dim first so broadcasting expands the RIGHT one.
            # (axis=0 previously "worked" only by coincidence -- the
            # remaining shape happens to already be right-aligned.)
            g = np.expand_dims(g, axis)
        _acc(a, a.backend.broadcast_to(g, a.data))

    out._backward = _bwd
    return out


def tensor_abs(a: Tensor) -> Tensor:
    out = Tensor(a.backend.tensor_abs(a.data), (a,), "abs", a.backend)

    def _bwd():
        _acc(a, a.backend.abs_backward(a.data, out.grad))

    out._backward = _bwd
    return out


def silu(a: Tensor) -> Tensor:
    out = Tensor(a.backend.silu(a.data), (a,), "silu", a.backend)

    def _bwd():
        _acc(a, a.backend.silu_backward(a.data, out.grad))

    out._backward = _bwd
    return out


def reshape(a: Tensor, shape: tuple) -> Tensor:
    """Gradient is just reshaping the incoming grad back to a's original
    shape -- reshape doesn't change values, only their logical arrangement."""
    orig_shape = a.data.shape
    out = Tensor(np.asarray(a.data).reshape(shape), (a,), "reshape", a.backend)

    def _bwd():
        _acc(a, np.asarray(out.grad).reshape(orig_shape))

    out._backward = _bwd
    return out


def transpose(a: Tensor, axes=None) -> Tensor:
    """Gradient is the incoming grad transposed by the INVERSE permutation.
    np.argsort(axes) gives that inverse for any axes tuple; axes=None
    (full reverse) is its own inverse."""
    out = Tensor(np.transpose(a.data, axes), (a,), "transpose", a.backend)
    inv_axes = None if axes is None else tuple(np.argsort(axes))

    def _bwd():
        _acc(a, np.transpose(np.asarray(out.grad), inv_axes))

    out._backward = _bwd
    return out


def concat(tensors, axis: int = 0) -> Tensor:
    """Concatenate 2+ Tensors along `axis` (default 0, the leading/token
    dimension for the usual [T, hidden] convention), differentiably --
    backward slices the incoming gradient back into each operand's own
    extent along that axis (a plain np.split on the concat point offsets),
    matching each operand's original shape exactly. All operands must
    share every OTHER axis's size (numpy's own concatenate enforces this,
    raising ValueError otherwise -- not re-checked here)."""
    tensors = list(tensors)
    arrays = [np.asarray(t.data, dtype=np.float32) for t in tensors]
    out_np = np.concatenate(arrays, axis=axis)
    out = Tensor(out_np, tuple(tensors), "concat", tensors[0].backend)
    sizes = [a.shape[axis] for a in arrays]
    split_points = np.cumsum(sizes)[:-1]

    def _bwd():
        g = np.asarray(out.grad, dtype=np.float32)
        for t, part in zip(tensors, np.split(g, split_points, axis=axis), strict=False):
            _acc(t, part)

    out._backward = _bwd
    return out


def gather(a: Tensor, indices) -> Tensor:
    """out[i] = a.flat[indices[i]], differentiably. Gradient scatters back
    to a's original shape; repeated indices correctly accumulate
    contributions from each occurrence (np.add.at, not plain assignment)."""
    idx = np.asarray(indices, dtype=np.int64)
    a_flat = np.asarray(a.data, dtype=np.float32).ravel()
    out = Tensor(a_flat[idx], (a,), "gather", a.backend)
    orig_shape = np.asarray(a.data).shape

    def _bwd():
        grad_flat = np.zeros(a_flat.shape, dtype=np.float32)
        np.add.at(grad_flat, idx, np.asarray(out.grad, dtype=np.float32))
        _acc(a, grad_flat.reshape(orig_shape))

    out._backward = _bwd
    return out


def bounded_gate(a: Tensor, n: float = 2.0) -> Tensor:
    """
    Saturating output gate: 1 - 1/(x^n + 1), mapping [0, inf) -> [0, 1).

    f(0) = 0: no activation -> no downstream effect.
    f(x) -> 1 as x -> inf: bounded, never overshoots regardless of how large
        the activation gets.
    n controls suppression strength near zero (the "how much do small
    activations get suppressed" knob, not a hard threshold): n=1 passes
    small x through nearly linearly (f'(0)=1); n=2 (default) suppresses
    small x quadratically -- f(x) ~= x^2 for small x, from the Taylor
    expansion of 1/(1+x^2) -- so f'(0)=0, meaning a near-zero (settled,
    quiet) activation produces near-zero effect, and effect only grows
    meaningfully once activation moves clearly away from zero. Higher n
    pushes that "clearly away from zero" threshold further out.

    Intended for converting an energy-gated activation into a magnitude for
    some downstream effect (e.g. how much to move, how much to act) where a
    settled/quiet neuron should correspond to near-zero effect and only a
    genuinely active one should produce a substantial one -- rather than
    forcing a single discrete choice via argmax/softmax-sampling ("old
    actor methods"), every input dimension gets its own continuous,
    energy-earned magnitude, and multiple can act at once. Assumes a >= 0
    (the non-negative, ReLU/energy-gated activation range this is meant
    for); not defined for non-integer n at negative a.
    """
    xn = power(a, n)
    denom = xn + 1.0
    return neg(power(denom, -1.0)) + 1.0


def broadcast_add(a: Tensor, bias: Tensor) -> Tensor:
    """Add a bias vector to a batched tensor (handles broadcasting)."""
    out = Tensor(a.backend.broadcast_add(a.data, bias.data), (a, bias), "broadcast_add", a.backend)

    def _bwd():
        _acc(a, out.grad)
        _acc(bias, a.backend.sum_axis0(out.grad))

    out._backward = _bwd
    return out


def sparse_mm(weight, x: Tensor) -> Tensor:
    """
    Sparse matmul: weight is a pybind11 SparseTensor, x is a dense Tensor.
    Backward is wired through the C++ object's backward_x / backward_vals.
    """
    out = Tensor(weight.mm(x.data), (x,), "sparse_mm", x.backend)

    def _bwd():
        _acc(x, weight.backward_x(x.data, out.grad))
        _acc_sparse(weight, weight.backward_vals(x.data, out.grad))

    out._backward = _bwd
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Attention ops — Tensor-graph wrappers over the C++ forward/backward kernels
# ══════════════════════════════════════════════════════════════════════════════
#
# The C++ side (sili/lib/headers/attention.hpp, bound in cpu_backend.cpp) has
# always had real forward *and* backward kernels for all three variants below;
# what was missing was any Tensor-graph wrapper connecting them, so nothing
# using attention could be trained end-to-end -- only frozen-weight inference.
# These wrappers are that missing piece: each stashes Q/K/V (the backward
# kernels recompute the softmax internally rather than needing it cached) and
# calls the matching `*_backward` binding from its `_backward` closure, the
# same pattern `sparse_mm` above uses for the sparse-linear C++ object.
#
# Q/K/V are 2-D [T, d] / [K, d] float32 arrays, no batch axis -- matches every
# other op in this library today (see EnergyDynamics's own single-vector
# assumption); batching all of these together is a separate, larger project
# tracked elsewhere, not attempted here.


def _attn_f32(t: Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.data, dtype=np.float32)


def sparse_attention(
    q: Tensor, k: Tensor, v: Tensor, top_k: int = 0, num_cpus: int = 4, causal: bool = False
) -> Tensor:
    """Global top-k sparse attention, differentiable w.r.t. q/k/v.
    causal=True masks any selected (query,key) pair where the key's
    sequence position is after the query's -- see attention.hpp."""
    from sili import _cpu

    qd, kd, vd = _attn_f32(q), _attn_f32(k), _attn_f32(v)
    out_np = _cpu.sparse_attention(qd, kd, vd, top_k, num_cpus, causal)
    out = Tensor(out_np, (q, k, v), "sparse_attention", q.backend)

    def _bwd():
        dQ, dK, dV = _cpu.sparse_attention_backward(
            qd, kd, vd, np.ascontiguousarray(out.grad, dtype=np.float32), top_k, num_cpus, causal
        )
        _acc(q, dQ)
        _acc(k, dK)
        _acc(v, dV)

    out._backward = _bwd
    return out


def banded_attention(
    q: Tensor, k: Tensor, v: Tensor, half_bandwidth: int, num_cpus: int = 4, causal: bool = False
) -> Tensor:
    """Dense banded (geometric-diagonal) attention, differentiable w.r.t.
    q/k/v. causal=True requires q/k to have the same length (self-attention)
    and clamps each query's band so it never sees a key past its own
    position -- see attention.hpp."""
    from sili import _cpu

    qd, kd, vd = _attn_f32(q), _attn_f32(k), _attn_f32(v)
    out_np = _cpu.banded_attention(qd, kd, vd, half_bandwidth, num_cpus, causal)
    out = Tensor(out_np, (q, k, v), "banded_attention", q.backend)

    def _bwd():
        dQ, dK, dV = _cpu.banded_attention_backward(
            qd, kd, vd, np.ascontiguousarray(out.grad, dtype=np.float32), half_bandwidth, num_cpus, causal
        )
        _acc(q, dQ)
        _acc(k, dK)
        _acc(v, dV)

    out._backward = _bwd
    return out


def gaussian_attention(
    q: Tensor, k: Tensor, v: Tensor, centers: Tensor, sigmas: Tensor, num_cpus: int = 4, causal: bool = False
) -> Tensor:
    """Full (every query x every key) attention with a learnable per-query
    Gaussian log-bias on top of the usual Q.K score -- differentiable
    w.r.t. q/k/v AND centers/sigmas (unlike banded_attention's fixed,
    non-learnable geometric-diagonal band). See attention.hpp for the
    exact math.

    centers/sigmas are [T] Tensors (one pair per query row) -- pass them
    as real Tensor objects (not raw arrays) so Module.parameters() picks
    them up as trainable leaves and gradients flow into centers.grad/
    sigmas.grad like any other parameter.

    sigmas must stay strictly positive through training. This function
    does not enforce that itself -- callers should store an unconstrained
    log_sigma parameter and pass `exp(log_sigma)` in here (see `exp()`
    above), not a raw trainable sigma directly, so the existing autograd
    chain rule keeps it positive automatically."""
    from sili import _cpu

    qd, kd, vd = _attn_f32(q), _attn_f32(k), _attn_f32(v)
    cd = np.ascontiguousarray(centers.data, dtype=np.float32)
    sd = np.ascontiguousarray(sigmas.data, dtype=np.float32)
    out_np = _cpu.gaussian_attention(qd, kd, vd, cd, sd, num_cpus, causal)
    out = Tensor(out_np, (q, k, v, centers, sigmas), "gaussian_attention", q.backend)

    def _bwd():
        dQ, dK, dV, dC, dS = _cpu.gaussian_attention_backward(
            qd, kd, vd, np.ascontiguousarray(out.grad, dtype=np.float32), cd, sd, num_cpus, causal
        )
        _acc(q, dQ)
        _acc(k, dK)
        _acc(v, dV)
        _acc(centers, dC)
        _acc(sigmas, dS)

    out._backward = _bwd
    return out


def sparse_banded_attention(
    q: Tensor, k: Tensor, v: Tensor, half_bandwidth: int, inner_k: int = 0, num_cpus: int = 4, causal: bool = False
) -> Tensor:
    """Banded attention with an inner top-k within each band, differentiable
    w.r.t. q/k/v. causal=True -- see banded_attention."""
    from sili import _cpu

    qd, kd, vd = _attn_f32(q), _attn_f32(k), _attn_f32(v)
    out_np = _cpu.sparse_banded_attention(qd, kd, vd, half_bandwidth, inner_k, num_cpus, causal)
    out = Tensor(out_np, (q, k, v), "sparse_banded_attention", q.backend)

    def _bwd():
        dQ, dK, dV = _cpu.sparse_banded_attention_backward(
            qd, kd, vd, np.ascontiguousarray(out.grad, dtype=np.float32), half_bandwidth, inner_k, num_cpus, causal
        )
        _acc(q, dQ)
        _acc(k, dK)
        _acc(v, dV)

    out._backward = _bwd
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Gradient helpers
# ══════════════════════════════════════════════════════════════════════════════


def _unbroadcast_to(grad, target_shape: tuple):
    """Sum `grad` down to `target_shape`, undoing whatever numpy
    broadcasting happened in a forward op -- numpy right-aligns shapes,
    so: drop any EXTRA leading axes grad has beyond target_shape's own
    rank (sum them away entirely), then for each remaining axis where
    target_shape is 1 but grad's is larger, sum over that axis
    (keepdims, so the axis count still matches)."""
    grad = np.asarray(grad)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, t_dim in enumerate(target_shape):
        if t_dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _acc(node: Tensor, grad) -> None:
    if node.grad is None:
        node.grad = node.backend.zeros_like(node.data)
    node.grad = node.backend.add(node.grad, grad)


def _acc_sparse(sparse_node, dvals) -> None:
    b = sparse_node.backend
    if sparse_node.grad is None:
        sparse_node.grad = b.zeros_like(sparse_node.vals)
    sparse_node.grad = b.add(sparse_node.grad, dvals)


def _topo_sort(root) -> list:
    topo, visited = [], set()

    def _visit(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                _visit(child)
            topo.append(v)

    _visit(root)
    return topo


def combine_losses(*terms):
    """
    Build ONE scalar loss from heterogeneous terms so a single backward()
    traverses the shared graph exactly once. Calling backward() from multiple
    roots double-counts shared subgraphs in sili (each traversal re-pushes the
    accumulated grad of shared nodes), so multi-loss training must combine
    first. This is the standard pattern for energy aux + task loss + RL terms.

    Term forms:
      (tensor, grad_array)  -- inject d(total)/d(tensor) = grad_array,
                               via (tensor * grad_array).sum()
      (tensor, weight)      -- weighted scalar-loss term: (tensor*w).sum()
      tensor                -- bare scalar-loss term, weight 1.0

    Returns a scalar Tensor; caller does total.backward() exactly once.
    """
    total = None
    for term in terms:
        if isinstance(term, tuple):
            t, g = term
            if isinstance(g, int | float):
                contrib = (t * float(g)).sum()
            else:
                contrib = (t * np.asarray(g, dtype=np.float32)).sum()
        else:
            contrib = term.sum()
        total = contrib if total is None else total + contrib
    return total
