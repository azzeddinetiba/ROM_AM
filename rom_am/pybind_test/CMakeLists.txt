cmake_minimum_required(VERSION 2.8.12)
project(ex2 LANGUAGES CXX)

#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#include_directories( ${EIGEN3_INCLUDE_DIRS} )
add_subdirectory(pybind11)
#pybind11_add_module(example example.cpp)

#Add source files
add_executable(ex2 example2.cpp)
target_link_libraries(ex2 PRIVATE pybind11::embed)
#include_directories( "./pybind11/include/" )
