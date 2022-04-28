cmake_minimum_required(VERSION 2.8.12)
project(ex4 LANGUAGES CXX)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories( ${EIGEN3_INCLUDE_DIRS} )
#add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)
#pybind11_add_module(ex4m example4.cpp)

#Add source files
add_executable(ex4 example4.cpp)
target_link_libraries(ex4 pybind11::embed)
#include_directories( "./pybind11/include/" )
