from __future__ import annotations
import numpy as np
from sili.backend import Backend, get_backend


# ══════════════════════════════════════════════════════════════════════════════
#  Tensor
# ══════════════════════════════════════════════════════════════════════════════

class Tensor:

    def __init__(self, data, _children: tuple = (), _op: str = "",
                 backend: Backend = None):
        self.data      = data
        self.grad      = None
        self._backward = lambda: None
        self._prev     = set(_children)
        self._op       = _op
        self.backend   = backend or get_backend("cpu")

    # ── device ────────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self.backend.name

    def to(self, device: str) -> "Tensor":
        target = get_backend(device)
        if target == self.backend: return self
        return Tensor(target.move(self.data, self.backend), backend=target)

    def detach(self) -> "Tensor":
        return Tensor(self.data, backend=self.backend)

    # ── coerce scalar to same-device Tensor ───────────────────────────────────

    def _coerce(self, other) -> "Tensor":
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

    def __add__(self, other):      return add(self, self._coerce(other))
    def __mul__(self, other):      return mul(self, self._coerce(other))
    def __matmul__(self, other):   return matmul(self, self._coerce(other))
    def __pow__(self, exp):        return power(self, exp)
    def __neg__(self):             return neg(self)
    def relu(self):                return relu(self)
    def tanh(self):                return tanh(self)
    def sum(self, axis=None):      return reduce_sum(self, axis)
    def reshape(self, shape):      return reshape(self, shape)
    def transpose(self, axes=None): return transpose(self, axes)
    def bounded_gate(self, n=2.0): return bounded_gate(self, n)
    def __abs__(self):             return tensor_abs(self)

    def __radd__(self, o): return add(self._coerce(o), self)
    def __sub__(self, o):  return add(self, neg(self._coerce(o)))
    def __rsub__(self, o): return add(self._coerce(o), neg(self))
    def __rmul__(self, o): return mul(self._coerce(o), self)
    def __truediv__(self, o):  return mul(self, power(self._coerce(o), -1))
    def __rtruediv__(self, o): return mul(self._coerce(o), power(self, -1))
    def __rmatmul__(self, o):  return matmul(self._coerce(o), self)

    # ── backward ──────────────────────────────────────────────────────────────

    def backward(self) -> None:
        if self.grad is None:
            self.grad = self.backend.ones_like(self.data)
        for node in reversed(_topo_sort(self)):
            node._backward()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def shape(self): return self.backend.shape(self.data)
    @property
    def dtype(self): return self.backend.dtype(self.data)

    def zero_grad(self): self.grad = None

    def __repr__(self):
        return f"Tensor(shape={self.shape}, device={self.device!r}, op={self._op!r})"


# ══════════════════════════════════════════════════════════════════════════════
#  Op library — each function owns its Tensor node + backward
# ══════════════════════════════════════════════════════════════════════════════

def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.backend.add(a.data, b.data), (a, b), "add", a.backend)
    def _bwd(): _acc(a, out.grad); _acc(b, out.grad)
    out._backward = _bwd
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.backend.mul(a.data, b.data), (a, b), "mul", a.backend)
    def _bwd():
        _acc(a, a.backend.mul(b.data, out.grad))
        _acc(b, b.backend.mul(a.data, out.grad))
    out._backward = _bwd
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.backend.matmul(a.data, b.data), (a, b), "matmul", a.backend)
    def _bwd():
        da, db = a.backend.matmul_backward(a.data, b.data, out.grad)
        _acc(a, da); _acc(b, db)
    out._backward = _bwd
    return out


def power(a: Tensor, exp: float) -> Tensor:
    assert isinstance(exp, (int, float))
    out = Tensor(a.backend.pow(a.data, exp), (a,), f"pow{exp}", a.backend)
    def _bwd(): _acc(a, a.backend.pow_backward(a.data, exp, out.grad))
    out._backward = _bwd
    return out


def neg(a: Tensor) -> Tensor:
    out = Tensor(a.backend.neg(a.data), (a,), "neg", a.backend)
    def _bwd(): _acc(a, a.backend.neg(out.grad))
    out._backward = _bwd
    return out


def relu(a: Tensor) -> Tensor:
    out = Tensor(a.backend.relu(a.data), (a,), "relu", a.backend)
    def _bwd(): _acc(a, a.backend.relu_backward(a.data, out.grad))
    out._backward = _bwd
    return out


def tanh(a: Tensor) -> Tensor:
    t   = np.tanh(np.asarray(a.data, dtype=np.float32))
    out = Tensor(t, (a,), "tanh", a.backend)
    def _bwd(): _acc(a, (1.0 - t * t) * np.asarray(out.grad, dtype=np.float32))
    out._backward = _bwd
    return out


def reduce_sum(a: Tensor, axis=None) -> Tensor:
    out = Tensor(a.backend.sum(a.data, axis), (a,), "sum", a.backend)
    def _bwd(): _acc(a, a.backend.broadcast_to(out.grad, a.data))
    out._backward = _bwd
    return out

def tensor_abs(a: Tensor) -> Tensor:
    out = Tensor(a.backend.tensor_abs(a.data), (a,), "abs", a.backend)
    def _bwd(): _acc(a, a.backend.abs_backward(a.data, out.grad))
    out._backward = _bwd
    return out

def silu(a: Tensor) -> Tensor:
    out = Tensor(a.backend.silu(a.data), (a,), "silu", a.backend)
    def _bwd(): _acc(a, a.backend.silu_backward(a.data, out.grad))
    out._backward = _bwd
    return out


def reshape(a: Tensor, shape: tuple) -> Tensor:
    """Gradient is just reshaping the incoming grad back to a's original
    shape -- reshape doesn't change values, only their logical arrangement."""
    orig_shape = a.data.shape
    out = Tensor(np.asarray(a.data).reshape(shape), (a,), "reshape", a.backend)
    def _bwd(): _acc(a, np.asarray(out.grad).reshape(orig_shape))
    out._backward = _bwd
    return out


def transpose(a: Tensor, axes=None) -> Tensor:
    """Gradient is the incoming grad transposed by the INVERSE permutation.
    np.argsort(axes) gives that inverse for any axes tuple; axes=None
    (full reverse) is its own inverse."""
    out = Tensor(np.transpose(a.data, axes), (a,), "transpose", a.backend)
    inv_axes = None if axes is None else tuple(np.argsort(axes))
    def _bwd(): _acc(a, np.transpose(np.asarray(out.grad), inv_axes))
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
    out = Tensor(a.backend.broadcast_add(a.data, bias.data),
                 (a, bias), "broadcast_add", a.backend)
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
        _acc(x,           weight.backward_x(x.data, out.grad))
        _acc_sparse(weight, weight.backward_vals(x.data, out.grad))
    out._backward = _bwd
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Gradient helpers
# ══════════════════════════════════════════════════════════════════════════════

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
            for child in v._prev: _visit(child)
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
            if isinstance(g, (int, float)):
                contrib = (t * float(g)).sum()
            else:
                contrib = (t * np.asarray(g, dtype=np.float32)).sum()
        else:
            contrib = term.sum()
        total = contrib if total is None else total + contrib
    return total
