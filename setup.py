from setuptools import setup, Extension
import os
import pybind11

module_name = "dng_graph"
source_files = [
    "src/dng.cpp"
]

include_dirs = [
    "src",
    pybind11.get_include()
]

dng_graph_module = Extension(
    module_name,
    sources=source_files,
    include_dirs=include_dirs,
    extra_compile_args=["-std=c++17", "-mavx2", "-ffast-math"],  # 启用 FMA 支持  AVX2 指令集
)

setup(
    name="dng_graph",
    version="1.0.0",
    description="A C++ library for graph-based nearest neighbor search",
    ext_modules=[dng_graph_module],
    classifiers=[
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)