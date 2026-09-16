import os
import sys

from setuptools import Extension, find_packages, setup

reqs = ["numpy", "pybind11"]
with open("README.md") as _readme_file:
    readme = _readme_file.read()


def _find_mkl():
    # Detect an installed `mkl`/`mkl-include` pip package (`pip install
    # sili[mkl]`), the same precedence torch uses for its CUDA .so
    # packages: pip-managed, venv-local, never vendored into this repo.
    # Both packages install under sys.prefix (confirmed by direct
    # install): mkl -> <prefix>/lib/libmkl_*.so.N, mkl-include ->
    # <prefix>/include/mkl*.h.
    lib_dir = os.path.join(sys.prefix, "lib")
    include_dir = os.path.join(sys.prefix, "include")
    if os.path.exists(os.path.join(lib_dir, "libmkl_gnu_thread.so.3")) and os.path.exists(
        os.path.join(include_dir, "mkl_cblas.h")
    ):
        return lib_dir, include_dir
    return None, None


class get_pybind_include:
    # Lazily evaluated pybind11 include path (pybind11's own official
    # example pattern): setuptools calls str() on this only during
    # build_ext, after pyproject.toml's build-system deps are installed,
    # so setup.py itself doesn't need pybind11 importable just to load --
    # a second layer of defense beyond build-system.requires for anyone
    # bypassing build isolation.
    def __str__(self):
        import pybind11

        return pybind11.get_include()


_mkl_lib_dir, _mkl_include_dir = _find_mkl()

_include_dirs = [
    get_pybind_include(),  # pybind11 headers (lazy -- see class above)
    # "sili/lib",               # csr.hpp, coo.hpp, linear_sisldo.hpp, etc.
    "sili/lib/headers",  # linear_sisldo.hpp, linear_disldo.hpp, etc.
]
_extra_compile_args = [
    "-O3",
    "-std=c++20",
    "-Wall",
    "-shared",
    "-fPIC",
    "-march=native",
    "-fopenmp",
    "-ffast-math",
    # -ffast-math's -ffinite-math-only silently defeats every NaN/Inf
    # guard (found directly: a debug fprintf gated on !isfinite() had its
    # format string compiled out). Re-enable real IEEE semantics for
    # those checks, keep -ffast-math's other optimizations.
    "-fno-finite-math-only",
]
_extra_link_args = ["-lgomp"]
_library_dirs = []
_libraries = []
_runtime_library_dirs = []

if _mkl_lib_dir is not None:
    # DIDLDO/SIDLDO: MKL direct-linked against the GNU/libgomp threading
    # layer (not libmkl_rt.so + an env var), matching sili's own
    # libgomp-based kernels -- see TODO_BATCH_BLOCKING.md for why.
    _include_dirs.append(_mkl_include_dir)
    _library_dirs.append(_mkl_lib_dir)
    _runtime_library_dirs.append(_mkl_lib_dir)
    _extra_link_args += [
        f"-L{_mkl_lib_dir}",
        "-l:libmkl_intel_lp64.so.3",
        "-l:libmkl_gnu_thread.so.3",
        "-l:libmkl_core.so.3",
        f"-Wl,-rpath,{_mkl_lib_dir}",
    ]
    _extra_compile_args.append("-DSILI_HAVE_MKL=1")

cpu_ext = Extension(
    name="sili._cpu",
    sources=["sili/cpu_backend.cpp"],
    include_dirs=_include_dirs,
    library_dirs=_library_dirs,
    runtime_library_dirs=_runtime_library_dirs,
    extra_compile_args=_extra_compile_args,
    extra_link_args=_extra_link_args,
    language="c++",
)

setup(
    name="sili",
    version="0.0.3",
    description="SILi: Sparse Intelligence Library",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/simleek/SILi",
    author="SimLeek",
    author_email="simulator.leek@gmail.com",
    license="MIT License",
    packages=find_packages(exclude=["tests", "tests.*"]),
    ext_modules=[cpu_ext],
    install_requires=reqs,
    # `pip install sili[mkl]` enables DIDLDO/SIDLDO; reliable path is
    # `pip install mkl mkl-include` first, then `pip install .`.
    extras_require={"mkl": ["mkl", "mkl-include"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
    ],
)
