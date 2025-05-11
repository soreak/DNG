from setuptools import setup, Extension
import os
import pybind11
import platform

module_name = "dng"
source_files = ["src/dng.cpp"]
include_dirs = ["src", pybind11.get_include()]

#compile_args = ["/std:c++17", "/arch:AVX2", "/fp:fast"]
compile_args = ["-std=c++17", "-mavx2", "-ffast-math"]

dng_module = Extension(
    module_name,
    sources=source_files,
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    language="c++"
)

setup(
    name="dng",
    version="1.0.0",
    description="A C++ library for graph-based nearest neighbor search",
    ext_modules=[dng_module],
    classifiers=[
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)

