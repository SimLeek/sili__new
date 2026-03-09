from __future__ import annotations
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
    def sum(self, axis=None):      return reduce_sum(self, axis)

    def __radd__(self, o): return add(self._coerce(o), self)
    def __sub__(self, o):  return add(self, neg(self._coerce(o)))
    def __rsub__(self, o): return add(self._coerce(o), neg(self))
    def __rmul__(self, o): return mul(self._coerce(o), self)
    def __truediv__(self, o):  return mul(self, power(self._coerce(o), -1))
    def __rtruediv__(self, o): return mul(self._coerce(o), power(self, -1))
    def __rmatmul__(self, o):  return matmul(self._coerce(o), self)

    # ── backward ──────────────────────────────────────────────────────────────

    def backward(self) -> None:
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


def reduce_sum(a: Tensor, axis=None) -> Tensor:
    out = Tensor(a.backend.sum(a.data, axis), (a,), "sum", a.backend)
    def _bwd(): _acc(a, a.backend.broadcast_to(out.grad, a.data))
    out._backward = _bwd
    return out


def silu(a: Tensor) -> Tensor:
    out = Tensor(a.backend.silu(a.data), (a,), "silu", a.backend)
    def _bwd(): _acc(a, a.backend.silu_backward(a.data, out.grad))
    out._backward = _bwd
    return out


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