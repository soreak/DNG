cmake_minimum_required(VERSION 3.5)
project(DNG)

set(PYTHON_EXECUTABLE "C:/Users/caizehua/AppData/Local/Programs/Python/Python310")
set(PYTHON_INCLUDE_DIRECTORY "C:/Users/caizehua/AppData/Local/Programs/Python/Python310/include")

add_subdirectory(extern/pybind11)
pybind11_add_module(dng_graph src/dng.cpp)