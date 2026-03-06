from __future__ import annotations
from sili.backend import Backend, register_backend


class CpuBackend(Backend):
    name = "cpu"

    @property
    def mod(self):
        from sili import _cpu
        return _cpu

    def move(self, buf, src: Backend):
        if src == self:
            return buf
        return self.mod.from_host(src.mod.to_host(buf))


register_backend(CpuBackend())