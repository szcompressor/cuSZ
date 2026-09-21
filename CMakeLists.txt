# select backend
set(PSZ_BACKEND "CUDA" CACHE STRING "Select the desired backend: CUDA/cuda and ONEAPI/1api.")

if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  cmake_minimum_required(VERSION 3.18)
  # 75 -> code=[compute_75,sm_75]
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED ENV{CUDAARCHS})
      set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}" CACHE STRING "CUDA architecture")
    else()
      set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architecture (sm_75: Turing and up; Volta and Pascal *may* or *may not* work with the current CUDA SDK)")
    endif()
  endif()
  project(CUSZ LANGUAGES CUDA CXX C VERSION 0.19.0)
elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
  cmake_minimum_required(VERSION 3.23)
  project(DPSZ LANGUAGES CXX C VERSION 0.6.0)
endif()

# general cmake setup
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

# conditionally enable functionality
option(PSZ_ACTIVATE_LC "build LC as a submodule" ON)

# example and testing
option(PSZ_BUILD_EXAMPLES "build example codes" ON)

# python binding via nanobind (py_ext/); replaces the stripped SWIG implementation
option(PSZ_BUILD_PYBINDING "enable Python binding for specific backend" OFF)

if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  message("[psz::info] CUDA backend has been selected.")
  include(cmake/cuda.cmake)
elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
  message("[psz::info] oneAPI backend has been selected.")
  include(cmake/sycl.cmake)
else()
  message(FATAL_ERROR "Unsupported PSZ_BACKEND selected: ${PSZ_BACKEND}")
endif()

# probe cmake configs regarding compilers
include(cmake/probe.cmake)
