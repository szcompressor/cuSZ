# select backend
set(PSZ_BACKEND "CUDA" CACHE STRING "Select the desired backend: CUDA/cuda, HIP/hip and ONEAPI/1api.")

if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  cmake_minimum_required(VERSION 3.18)
  project(CUSZ LANGUAGES CUDA CXX C VERSION 0.6.0)
elseif(PSZ_BACKEND STREQUAL "HIP" OR PSZ_BACKEND STREQUAL "hip")
  cmake_minimum_required(VERSION 3.21)
  project(HIPSZ LANGUAGES HIP CXX C VERSION 0.6.0)
elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
  cmake_minimum_required(VERSION 3.23)
  project(DPSZ LANGUAGES CXX C VERSION 0.6.0)
endif()

# general cmake setup
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

# log: debug and sanitize
option(PSZ_LOG_DBG_MODE "enable debugging printing" OFF)
option(PSZ_LOG_SANITIZE_MODE "enable sanitizing functions" OFF)
option(PSZLOG_ENABLE_ALL "enable all (debugging, sanitizing)" OFF)
include(cmake/logging.cmake)

# conditionally enable functionality
option(PSZ_RESEARCH_HUFFBK_CUDA "(CUDA only) build research artifacts: create Huffman codebook on GPU" OFF)
option(PSZ_REACTIVATE_THRUSTGPU "build previously thrust implemented functions" OFF)

# example and testing
option(PSZ_BUILD_EXAMPLES "build example codes" OFF)


if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  message("[psz::info] CUDA backend has been selected.")
  include(cmake/cusz.cmake)
elseif(PSZ_BACKEND STREQUAL "HIP" OR PSZ_BACKEND STREQUAL "hip")
  message("[psz::info] HIP-ROCm backend has been selected.")
  include(cmake/hipsz.cmake)
elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
  message("[psz::info] oneAPI backend has been selected.")
  include(cmake/dpsz.cmake)
else()
  message(FATAL_ERROR "Unsupported PSZ_BACKEND selected: ${PSZ_BACKEND}")
endif()

# probe cmake configs regarding compilers
include(cmake/probe.cmake)