# select backend
set(PSZ_BACKEND "CUDA" CACHE STRING "Select the desired backend: CUDA/cuda, HIP/hip, ONEAPI/1api.")

if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  cmake_minimum_required(VERSION 3.18)
  project(CUSZ LANGUAGES CUDA CXX C VERSION 0.10.0)
elseif(PSZ_BACKEND STREQUAL "HIP" OR PSZ_BACKEND STREQUAL "hip")
  cmake_minimum_required(VERSION 3.21)
  project(HIPSZ LANGUAGES HIP CXX C VERSION 0.10.0)
elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
  cmake_minimum_required(VERSION 3.23)
  project(DPSZ LANGUAGES CXX C VERSION 0.6.0)
endif()

# general cmake setup
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

# conditionally enable functionality
# The bundled LC-framework (third_party/lc) is an external CUDA codebase that is
# not yet ported to HIP; default it OFF on the HIP backend so the core
# Lorenzo/spline + Huffman/HFR + FZG pipeline builds. It can be force-enabled
# once LC-framework gains HIP support.
if(PSZ_BACKEND STREQUAL "HIP" OR PSZ_BACKEND STREQUAL "hip")
  option(PSZ_ACTIVATE_LC "build LC as a submodule" OFF)
else()
  option(PSZ_ACTIVATE_LC "build LC as a submodule" ON)
endif()

# example and testing
option(PSZ_BUILD_EXAMPLES "build example codes" ON)

# python binding (option reserved for future use; SWIG implementation stripped)
option(PSZ_BUILD_PYBINDING "enable Python binding for specific backend" OFF)

if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  message("[psz::info] CUDA backend has been selected.")
  include(cmake/cuda.cmake)
elseif(PSZ_BACKEND STREQUAL "HIP" OR PSZ_BACKEND STREQUAL "hip")
  message("[psz::info] HIP (ROCm) backend has been selected.")
  include(cmake/hip.cmake)
elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
  message("[psz::info] oneAPI backend has been selected.")
  include(cmake/sycl.cmake)
else()
  message(FATAL_ERROR "Unsupported PSZ_BACKEND selected: ${PSZ_BACKEND}")
endif()

# probe cmake configs regarding compilers
include(cmake/probe.cmake)
