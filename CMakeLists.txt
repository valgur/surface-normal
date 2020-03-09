cmake_minimum_required(VERSION 3.10)
project(surface_normal CXX)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wpedantic -Wno-sign-compare)

include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

add_executable(${PROJECT_NAME}_messy tool.cpp tool.h)

add_executable(${PROJECT_NAME} clean.cpp)
