cmake_minimum_required(VERSION 3.16)
project(IBKRTradingEngine VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Find required packages
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Include directories
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    src/inference_engine.cpp
    src/feature_pipeline.cpp
    src/market_data_parser.cpp
    src/linear_algebra_engine.cpp
    src/memory_pool.cpp
    src/simd_utils.cpp
    src/model_loader.cpp
    src/ensemble_balancer.cpp
)

# Header files
set(HEADERS
    include/inference_engine.h
    include/feature_pipeline.h
    include/market_data_parser.h
    include/linear_algebra_engine.h
    include/memory_pool.h
    include/simd_utils.h
    include/model_loader.h
    include/ensemble_balancer.h
    include/common.h
)

# Create shared library
add_library(ibkr_trading_engine SHARED ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(ibkr_trading_engine 
    OpenMP::OpenMP
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ibkr_trading_engine PRIVATE -mavx2 -mfma)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ibkr_trading_engine PRIVATE -mavx2 -mfma)
endif()

# Python bindings (Cython)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(ibkr_trading_engine_py 
    python_bindings/inference_engine_bindings.cpp
    python_bindings/feature_pipeline_bindings.cpp
    python_bindings/market_data_bindings.cpp
    python_bindings/linear_algebra_bindings.cpp
)

target_link_libraries(ibkr_trading_engine_py PRIVATE ibkr_trading_engine)

# Benchmarks
add_executable(benchmark_inference benchmarks/benchmark_inference.cpp)
target_link_libraries(benchmark_inference ibkr_trading_engine)

add_executable(benchmark_features benchmarks/benchmark_features.cpp)
target_link_libraries(benchmark_features ibkr_trading_engine)

add_executable(benchmark_market_data benchmarks/benchmark_market_data.cpp)
target_link_libraries(benchmark_market_data ibkr_trading_engine)

# Tests
add_executable(test_inference tests/test_inference.cpp)
target_link_libraries(test_inference ibkr_trading_engine)

add_executable(test_features tests/test_features.cpp)
target_link_libraries(test_features ibkr_trading_engine)

add_executable(test_market_data tests/test_market_data.cpp)
target_link_libraries(test_market_data ibkr_trading_engine)

# Installation
install(TARGETS ibkr_trading_engine ibkr_trading_engine_py
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS} DESTINATION include/ibkr_trading_engine)
