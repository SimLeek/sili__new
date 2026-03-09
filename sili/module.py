"""
sili/nn.py
──────────
Neural network modules.

Kernel calls go through the tensor's backend directly: a.backend.matmul(...)
Op functions from engine.py handle Tensor wrapping — no Tensor(...) boilerplate here.
"""

from __future__ import annotations
from typing import List, Optional

from sili.backend import get_backend
from sili.tensor  import (Tensor, _acc,
                           matmul, broadcast_add, relu, silu, sparse_mm)

# ══════════════════════════════════════════════════════════════════════════════
#  Module base
# ══════════════════════════════════════════════════════════════════════════════

class Module:

    def parameters(self) -> list:
        seen, out = set(), []
        for _, p in self._iter_leaves():
            if id(p) not in seen:
                seen.add(id(p)); out.append(p)
        return out

    def _iter_leaves(self, prefix=""):
        for attr, val in self.__dict__.items():
            full = f"{prefix}.{attr}" if prefix else attr
            if isinstance(val, Tensor):
                yield full, val
            elif isinstance(val, Module):
                yield from val._iter_leaves(full)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    tag = f"{full}[{i}]"
                    if isinstance(item, Tensor):
                        yield tag, item
                    elif isinstance(item, Module):
                        yield from item._iter_leaves(tag)

    def to(self, device: str) -> "Module":
        for attr, val in list(self.__dict__.items()):
            if isinstance(val, Tensor):
                setattr(self, attr, val.to(device))
            elif isinstance(val, Module):
                val.to(device)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, Tensor):       val[i] = item.to(device)
                    elif isinstance(item, Module):     item.to(device)
        return self

    def zero_grad(self):
        for p in self.parameters(): p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
#  Layers
# ══════════════════════════════════════════════════════════════════════════════

class Linear(Module):
    """y = x @ W + b"""

    def __init__(self, nin: int, nout: int,
                 bias: bool = True, nonlin: bool = False, device: str = "cpu"):
        b      = get_backend(device)
        self.w = Tensor(b.uniform(-1.0, 1.0, (nin, nout)), backend=b)
        self.b = Tensor(b.zeros((nout,)), backend=b) if bias else None
        self.nonlin = nonlin

    def forward(self, x: Tensor) -> Tensor:
        out = matmul(x, self.w)
        if self.b is not None:
            out = broadcast_add(out, self.b)
        return relu(out) if self.nonlin else out

    def parameters(self): return [self.w] + ([self.b] if self.b is not None else [])
    def __repr__(self):
        nin, nout = self.w.shape
        return f"Linear(in={nin}, out={nout}, bias={self.b is not None})"

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: return relu(x)
    def parameters(self): return []
    def __repr__(self): return "ReLU()"


class SwiGLU(Module):
    """silu(gate) * up — no parameters."""
    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        return silu(gate) * up
    def parameters(self): return []
    def __repr__(self): return "SwiGLU()"


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6, device: str = "cpu"):
        b           = get_backend(device)
        self.weight = Tensor(b.ones((dim,)), backend=b)
        self.eps    = eps

    def forward(self, x: Tensor) -> Tensor:
        b   = x.backend
        out = Tensor(b.rmsnorm(x.data, self.weight.data, self.eps),
                     (x, self.weight), "rmsnorm", b)
        w_ref = self.weight
        def _bwd():
            dx, dw = b.rmsnorm_backward(x.data, w_ref.data, self.eps, out.grad)
            _acc(x, dx); _acc(w_ref, dw)
        out._backward = _bwd
        return out

    def parameters(self): return [self.weight]
    def __repr__(self): return f"RMSNorm(dim={self.weight.shape[0]}, eps={self.eps})"