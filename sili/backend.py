from __future__ import annotations
from typing import Dict


class Backend:
    name: str = ""

    @property
    def mod(self):
        """Return the kernel module for this device."""
        raise NotImplementedError

    def move(self, buf, src: "Backend"):
        """Copy buf (on src) to this device and return the new buffer."""
        raise NotImplementedError

    def __getattr__(self, name):
        """Forward any op not defined on the class to the kernel module."""
        try:
            return getattr(self.mod, name)
        except AttributeError:
            raise AttributeError(
                f"Backend {self.name!r} (mod={type(self.mod).__name__}) "
                f"has no op {name!r}."
            )

    def __eq__(self, other):  return isinstance(other, Backend) and self.name == other.name
    def __hash__(self):       return hash(self.name)
    def __repr__(self):       return f"Backend({self.name!r})"


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    if not backend.name:
        raise ValueError("Backend.name must be non-empty.")
    _REGISTRY[backend.name] = backend


def get_backend(device: str) -> Backend:
    if device in _REGISTRY:
        return _REGISTRY[device]
    base = device.split(":")[0]          # "cuda:1" → "cuda"
    if base in _REGISTRY:
        return _REGISTRY[base]
    raise KeyError(
        f"No backend registered for {device!r}.  "
        f"Registered: {list(_REGISTRY)}.  "
        "Call register_backend(MyBackend()) to add one."
    )