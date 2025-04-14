from setuptools import setup, Extension
import pybind11
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        if ct == 'msvc':
            for ext in self.extensions:
                ext.extra_compile_args = ["/std:c++17", "/arch:AVX2", "/fp:fast"]
        else:
            for ext in self.extensions:
                ext.extra_compile_args = ["-std=c++17", "-mavx2", "-ffast-math"]
        build_ext.build_extensions(self)

module_name = "dng_graph"
source_files = ["src/dng.cpp"]
include_dirs = ["src", pybind11.get_include()]

dng_graph_module = Extension(
    module_name,
    sources=source_files,
    include_dirs=include_dirs,
    language="c++"
)

setup(
    name="dng_graph",
    version="1.0.0",
    description="A C++ library for graph-based nearest neighbor search",
    ext_modules=[dng_graph_module],
    cmdclass={"build_ext": BuildExt},
    classifiers=[
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
