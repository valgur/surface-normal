cmake_minimum_required(VERSION 3.11)
project(surface_normal LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)
link_libraries(${OpenCV_LIBS})

list(APPEND CUDA_NVCC_FLAGS -O3 -D_FORCE_INLINES)
#list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-v)
list(APPEND CUDA_NVCC_FLAGS --expt-relaxed-constexpr)
list(APPEND CUDA_NVCC_FLAGS --compiler-options=-fPIC,-DPIC,-Wall,-Wextra,-Wno-cpp)

set(CUDA_ARCH_LIST Auto CACHE STRING
  "List of CUDA architectures (Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection.")
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_compile_options(-O3 -Wall -Wpedantic -Wno-sign-compare)

cuda_add_library(${PROJECT_NAME}_gpu src/surface_normal.cu)
target_link_libraries(${PROJECT_NAME}_gpu Eigen3::Eigen)

include(FetchContent)
FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11)
FetchContent_MakeAvailable(pybind11)
pybind11_add_module(${PROJECT_NAME} src/python.cpp src/surface_normal.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_gpu)
