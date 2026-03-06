from setuptools import setup, find_packages, Extension
import pybind11

reqs   = open("requirements.txt").read().splitlines()
readme = open("README.md").read()

cpu_ext = Extension(
    name="sili._cpu",
    sources=["sili/cpu_backend.cpp"],
    include_dirs=[
        pybind11.get_include(),   # pybind11 headers
        "sili/lib",               # csr.hpp, coo.hpp, linear_sisldo.hpp, etc.
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
    url="https://github.com/simleek/SILi",
    author="SimLeek",
    author_email="simulator.leek@gmail.com",
    license="MIT License",
    packages=find_packages(exclude=["test", "test.*"]),
    ext_modules=[cpu_ext],
    install_requires=reqs,
    include_package_data=True,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
    ],
)