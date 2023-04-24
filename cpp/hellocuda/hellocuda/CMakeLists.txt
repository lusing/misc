cmake_minimum_required(VERSION 3.10)

# set project name
project(hellocuda LANGUAGES CXX CUDA)

# find CUDA package
find_package(CUDA REQUIRED)

# set CUDA architecture, if not set, it will use default value
# for example, for CUDA 10.0 or later, use "7.0" for "Kepler" architecture
# refer to CUDA documentation for more details
set(CUDA_ARCH "12.1" CACHE STRING "CUDA architecture version (default is 12.1)")

# set CUDA compilation flags
set(CUDA_NVCC_FLAGS "-arch=sm_${CUDA_ARCH};--use_fast_math" CACHE STRING "NVCC flags")
set(CMAKE_CUDA_FLAGS "-arch=sm_${CUDA_ARCH};--use_fast_math" CACHE STRING "CUDA flags")

# specify source files
set(SOURCE_FILES kernel.cu)

# add executable target
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# set CUDA include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})

# link CUDA libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
