cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(Automatic-Tag-Generation-for-Text-Article)

set(CMAKE_CXX_STANDARD 14)

# Torch related
#find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(include)
file(GLOB SOURCES "include/*.cpp")

# Pybind related
#find_package(pybind11 REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(pybind11)
pybind11_add_module(_matrix SHARED ${SOURCES})