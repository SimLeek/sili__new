from setuptools import setup, find_packages, Extension

reqs   = [
'numpy',
'pybind11'
]
readme = open("README.md").read()


class get_pybind_include:
    """
    Lazily evaluated pybind11 include path (the pattern pybind11's own
    official example projects use, e.g. pybind11/python_example). setuptools
    calls str() on each include_dirs entry only when actually invoking the
    compiler, during build_ext -- well after build-system dependencies
    (declared in pyproject.toml) have been installed. This means setup.py
    itself no longer needs pybind11 importable just to be evaluated, which
    matters even with pyproject.toml's build-system.requires in place: this
    is the second, independent layer of defense for anyone bypassing build
    isolation (e.g. `pip install --no-build-isolation .`, or a very old pip
    that doesn't honor pyproject.toml's [build-system] section).
    """
    def __str__(self):
        import pybind11
        return pybind11.get_include()


cpu_ext = Extension(
    name="sili._cpu",
    sources=["sili/cpu_backend.cpp"],
    include_dirs=[
        get_pybind_include(),   # pybind11 headers (lazy -- see class above)
        #"sili/lib",               # csr.hpp, coo.hpp, linear_sisldo.hpp, etc.
        "sili/lib/headers",  # linear_sisldo.hpp, linear_disldo.hpp, etc.
    ],
    extra_compile_args=[
        "-O3",
        "-std=c++20",
        "-Wall",
        "-shared",
        "-fPIC",
        "-march=native",
        "-fopenmp",
        "-ffast-math",
    ],
    extra_link_args=["-lgomp"],
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
    include_package_data=True,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
    ],
)