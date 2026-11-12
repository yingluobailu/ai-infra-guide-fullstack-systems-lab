cmake_minimum_required(VERSION 3.18)

project(day07_cpp_build LANGUAGES CXX)

add_executable(
    day07_demo
    src/main.cpp
    src/vector_ops.cpp
)

target_include_directories(
    day07_demo
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(
    day07_demo
    PRIVATE
    cxx_std_17
)

target_compile_options(
    day07_demo
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)