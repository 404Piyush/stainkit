cmake_minimum_required(VERSION 3.18)
project(stainkit
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated H&E stain normalization for digital pathology"
    LANGUAGES CXX CUDA
)

# ----------------------------------------------------------------------------
# Project-wide options
# ----------------------------------------------------------------------------
option(STK_BUILD_TESTS       "Build unit tests (GoogleTest)"            ON)
option(STK_BUILD_PYTHON      "Build Python bindings (pybind11)"         OFF)
option(STK_BUILD_BENCHMARKS  "Build CPU/GPU benchmark executables"       ON)
option(STK_BUILD_DOCS        "Build documentation (Doxygen, optional)"   OFF)
option(STK_USE_NPP           "Use NVIDIA NPP primitives where possible"  OFF)
option(STK_USE_OPENSLIDE     "Use OpenSlide for whole-slide image input" OFF)
option(STK_VERBOSE_LOGGING   "Enable verbose per-kernel logging"         OFF)

set(STK_CXX_STANDARD 17)
set(STK_CUDA_STANDARD 17)
set(STK_MIN_CUDA_ARCH 7.0)  # Volta or newer (covers Colab T4)

# ----------------------------------------------------------------------------
# Standard CMake module path
# ----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/_deps")

# ----------------------------------------------------------------------------
# Output layout (matches the PascaleCourseraCourses template)
# ----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)

# ----------------------------------------------------------------------------
# Default build type
# ----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ----------------------------------------------------------------------------
# Global compile/link flags
# ----------------------------------------------------------------------------
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor)
# Apply -Xcompiler=-Wall,-Wextra,-Wpedantic to CUDA translation units only.
# The previous global add_compile_options(-Xcompiler=...) leaked into
# C++ TUs and made the host c++ driver reject the flag. The
# $<COMPILE_LANGUAGE:CUDA> generator expression scopes the option to
# .cu files only.
if(CMAKE_CUDA_COMPILER)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>")
endif()

# ----------------------------------------------------------------------------
# Threads / OpenMP (used by the CPU reference implementations)
# ----------------------------------------------------------------------------
find_package(Threads REQUIRED)
find_package(OpenMP)

# Locate the CUDA toolkit so the imported targets (CUDA::cudart,
# CUDA::cuda_driver, ...) are populated. With `LANGUAGES CUDA` alone
# these targets are *not* registered on CMake >= 3.18.
find_package(CUDAToolkit REQUIRED)

# ----------------------------------------------------------------------------
# Stb image (single-header, vendored)
# ----------------------------------------------------------------------------
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb>
)
target_compile_definitions(stb_image INTERFACE STB_IMAGE_IMPLEMENTATION=0)
add_library(stainkit::stb_image ALIAS stb_image)

# ----------------------------------------------------------------------------
# Core stainkit library
# ----------------------------------------------------------------------------
add_library(stainkit_core
    src/types.cpp
    src/io.cpp
    src/pipeline.cpp
    src/cpu_reference.cpp
)
add_library(stainkit::core ALIAS stainkit_core)

target_include_directories(stainkit_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(stainkit_core
    PUBLIC stainkit::stb_image Threads::Threads
)

# ----------------------------------------------------------------------------
# CUDA library — kernels + device-side glue
# ----------------------------------------------------------------------------
set(STK_KERNEL_SOURCES
    src/kernels/color_deconvolution.cu
    src/kernels/stain_normalization.cu
    src/kernels/tissue_mask.cu
    src/kernels/od_conversion.cu
    src/kernels/morphology.cu
)
add_library(stainkit_cuda STATIC ${STK_KERNEL_SOURCES})
add_library(stainkit::cuda ALIAS stainkit_cuda)

target_include_directories(stainkit_cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(stainkit_cuda PROPERTIES
    CUDA_STANDARD          ${STK_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(stainkit_cuda PUBLIC cxx_std_${STK_CXX_STANDARD})

# Honour user-requested CUDA architectures, otherwise build for the host
# default plus the most common Colab GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()
message(STATUS "stainkit CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

target_link_libraries(stainkit_cuda
    PUBLIC stainkit::core CUDA::cudart
)

if(STK_VERBOSE_LOGGING)
    target_compile_definitions(stainkit_cuda PUBLIC STK_VERBOSE=1)
endif()

if(STK_USE_NPP)
    find_package(npp REQUIRED)
    target_link_libraries(stainkit_cuda PUBLIC npp::npp)
    target_compile_definitions(stainkit_cuda PUBLIC STK_HAS_NPP=1)
endif()

if(STK_USE_OPENSLIDE)
    find_package(OpenSlide REQUIRED)
    target_link_libraries(stainkit_cuda PUBLIC OpenSlide::OpenSlide)
    target_compile_definitions(stainkit_cuda PUBLIC STK_HAS_OPENSLIDE=1)
endif()

# ----------------------------------------------------------------------------
# CLI executable
# ----------------------------------------------------------------------------
add_executable(stainkit src/main.cpp)
target_link_libraries(stainkit PRIVATE stainkit::core stainkit::cuda)
set_target_properties(stainkit PROPERTIES OUTPUT_NAME stainkit)

# ----------------------------------------------------------------------------
# Benchmark executables (optional)
# ----------------------------------------------------------------------------
if(STK_BUILD_BENCHMARKS)
    add_executable(stainkit_bench src/benchmarks/bench_pipeline.cpp)
    target_link_libraries(stainkit_bench PRIVATE stainkit::core stainkit::cuda)
    set_target_properties(stainkit_bench PROPERTIES OUTPUT_NAME stainkit-bench)
endif()

# ----------------------------------------------------------------------------
# Python bindings (pybind11)
# ----------------------------------------------------------------------------
if(STK_BUILD_PYTHON)
    # pybind11_add_module is only available when Python has been found
    # *before* the pybind11 CMake helpers are pulled in. We pre-locate
    # the interpreter so the add_subdirectory() call inside
    # FetchContent_MakeAvailable() registers the pybind11 functions.
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

    include(FetchContent)
    set(STK_PYBIND11_TAG v2.11.1)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        ${STK_PYBIND11_TAG}
        GIT_SHALLOW    TRUE
    )
    set(PYBIND11_NOPYTHON OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(gpustain_py src/bindings/bindings.cpp
    )
    target_link_libraries(gpustain_py PRIVATE stainkit::core stainkit::cuda)
    target_compile_definitions(gpustain_py PRIVATE STK_BUILDING_PYMODULE=1)
    set_target_properties(gpustain_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/gpustain
        OUTPUT_NAME gpustain
    )
endif()

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(STK_BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    set(STK_GTEST_TAG v1.14.0)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        ${STK_GTEST_TAG}
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------------
# Install rules
# ----------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS stainkit stainkit_cuda stainkit_core stb_image
    EXPORT stainkitTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT stainkitTargets
    FILE stainkitTargets.cmake
    NAMESPACE stainkit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stainkit
)

# ----------------------------------------------------------------------------
# Status summary
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==== stainkit configuration ====")
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard      : ${STK_CXX_STANDARD}")
message(STATUS "  CUDA standard     : ${STK_CUDA_STANDARD}")
message(STATUS "  CUDA archs        : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build tests       : ${STK_BUILD_TESTS}")
message(STATUS "  Build python      : ${STK_BUILD_PYTHON}")
message(STATUS "  Build benchmarks  : ${STK_BUILD_BENCHMARKS}")
message(STATUS "  Use NPP           : ${STK_USE_NPP}")
message(STATUS "  Use OpenSlide     : ${STK_USE_OPENSLIDE}")
message(STATUS "  Install prefix    : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===============================")
message(STATUS "")
