cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MatrixAddO)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(matrix_add matadd.cpp)
