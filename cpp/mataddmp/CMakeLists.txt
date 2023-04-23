cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MatrixAddOpenMP)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add the executable
add_executable(matrix_add main.cpp)

# Link OpenMP to the executable
if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix_add PUBLIC OpenMP::OpenMP_CXX)
endif()
