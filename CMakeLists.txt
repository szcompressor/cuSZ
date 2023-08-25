# select backend
set(PSZ_BACKEND "CUDA" CACHE STRING "Select the desired backend: CUDA/cuda, HIP/hip.")

if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  cmake_minimum_required(VERSION 3.17)
  project(CUSZ LANGUAGES CUDA CXX C VERSION 0.6.0)
elseif(PSZ_BACKEND STREQUAL "HIP" OR PSZ_BACKEND STREQUAL "hip")
  cmake_minimum_required(VERSION 3.21)
  project(HIPSZ LANGUAGES HIP CXX C VERSION 0.6.0)
endif()

# general cmake setup
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

# conditionaly enable functionality
option(PSZ_RESEARCH_HUFFBK_CUDA "(CUDA only) build research artifacts: create Huffman codebook on GPU" OFF)
option(PSZ_REACTIVATE_THRUSTGPU "build previously thrust implemented functions" OFF)

# example and testing
option(PSZ_BUILD_EXAMPLES "build example codes" OFF)

if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  include(cmake/cusz.cmake)
elseif(PSZ_BACKEND STREQUAL "HIP" OR PSZ_BACKEND STREQUAL "hip")
  include(cmake/hipsz.cmake)
else()
  message(FATAL_ERROR "Unsupported PSZ_BACKEND selected: ${PSZ_BACKEND}")
endif()
