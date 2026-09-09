# Pin OpenMP threads before _cpu.so loads; see linear_disldo.rst's ccx_aware_reduction section.
import os

os.environ.setdefault("OMP_PROC_BIND", "true")
os.environ.setdefault("OMP_PLACES", "cores")

import sili.cpu  # noqa: F401 -- side-effect import, triggers register_backend()
# import sili.vulkan  # not implemented yet
