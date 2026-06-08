
cmake_minimum_required(VERSION 3.18)

project(EVAL
  VERSION 0.1.0
  LANGUAGES CXX
)

# ------------------------------------------------------------------------------
# Standalone vs subdir 
# ------------------------------------------------------------------------------

# Check if this is being built as a standalone project
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(EVAL_STANDALONE TRUE)
  message("[eval::info] Building EVAL as standalone project")
else()
  set(EVAL_STANDALONE FALSE)
  message("[eval::info] Building EVAL as subdirectory")
endif()

# Set default backend if not specified
if(NOT PSZ_BACKEND)
  set(PSZ_BACKEND "CUDA" CACHE STRING "Select backend: CUDA/cuda, ONEAPI/oneapi")
  message("[eval::info] Using default backend: CUDA")
endif()

set_property(CACHE PSZ_BACKEND PROPERTY STRINGS CUDA HIP ONEAPI)
string(TOUPPER "${PSZ_BACKEND}" PSZ_BACKEND_UPPER)
if(PSZ_BACKEND_UPPER STREQUAL "1API")
  set(PSZ_BACKEND_UPPER "ONEAPI")
endif()

# Set default build directories for standalone builds
if(EVAL_STANDALONE)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  option(BUILD_SHARED_LIBS "prefer shared libraries" ON)
  option(EVAL "build CUDA eval implementations (requires full psz tree)" OFF)
else()
  option(EVAL "build CUDA eval implementations (requires full psz tree)" ON)
endif()

# Find CUDA if needed
if(PSZ_BACKEND_UPPER STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
endif()

# Find portable headers
find_package(PORTABLE QUIET)
if(NOT PORTABLE_FOUND AND EVAL_STANDALONE)
  # For standalone builds, expect portable headers in ../portable/include
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../portable/include/c_type.h")
    message(FATAL_ERROR "portable headers not found. Expected in ../portable/include")
  endif()
endif()

# Create compile settings for eval
if(EVAL_STANDALONE)
  add_library(eval_compile_settings INTERFACE)
  
  target_compile_features(eval_compile_settings
    INTERFACE
      cxx_std_17
  )
  
  if(PSZ_BACKEND_UPPER STREQUAL "CUDA")
    target_compile_definitions(eval_compile_settings
      INTERFACE
        PSZ_USE_CUDA
        _PORTABLE_USE_CUDA
    )
    target_compile_options(eval_compile_settings
      INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
    )
  elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
    target_compile_definitions(eval_compile_settings
      INTERFACE
        PSZ_USE_1API
    )
  endif()
else()
  # Use psz_cu_compile_settings from parent build
  set(eval_compile_settings "psz_cu_compile_settings")
endif()

# ------------------------------------------------------------------------------
# Backend selection
# ------------------------------------------------------------------------------

if(PSZ_BACKEND_UPPER STREQUAL "CUDA")

  enable_language(CUDA)
  set(EVAL_BACKEND "CUDA")

elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")

  set(EVAL_BACKEND "SYCL")

endif()

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------

# Common header interface
add_library(utils_headers INTERFACE)

target_compile_features(utils_headers
  INTERFACE
    cxx_std_17
)

target_include_directories(utils_headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../portable/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../psz/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../codec/hf/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# CPU/Sequential implementation (all backends)
add_library(eval_seq
  src/compare.stl.cc
  src/vis_stat.cc
)

if(EVAL_BACKEND STREQUAL "CUDA")
  # CUDA backend
  if(EVAL)
    add_library(eval_cu
      src/identical.cu
      src/compile/extrema-f4.cu
      src/compile/extrema-f8.cu
      src/compile/calc_err-f4.cu
      src/compile/calc_err-f8.cu
      src/compile/assess-f4.cu
      src/compile/assess-f8.cu
      src/compile/maxerr-all.cu
    )
    
    target_link_libraries(eval_cu
      PUBLIC
        utils_headers
        ${eval_compile_settings}
        CUDA::cudart
    )
    
    # Ensure CUDA compile options are set
    target_compile_options(eval_cu
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
    )
    
    # Namespace alias for consistency (EVAL::stat_cu following PHF::phf_cu convention)
    add_library(EVAL::stat_cu ALIAS eval_cu)
    
    # Viewer with CUDA implementation
    add_library(eval_viewer_cu
      src/viewer.cc
      src/compile/viewer-all.cu
    )
    target_link_libraries(eval_viewer_cu
      PUBLIC
        utils_headers
        eval_seq
        eval_cu
        ${eval_compile_settings}
        CUDA::cudart
    )
    add_library(EVAL::viewer_cu ALIAS eval_viewer_cu)
  else()
    message("[eval::info] EVAL is OFF: CUDA eval implementations skipped (require full psz tree).")
  endif()
  
elseif(EVAL_BACKEND STREQUAL "SYCL")
  # SYCL/DPC++ backend - TODO: add SYCL implementations  
  # add_library(eval_dp ...)
  
endif()

target_link_libraries(eval_seq
  PUBLIC
    utils_headers
)

# Namespace alias (EVAL::stat_seq following EVAL::stat_cu convention)
add_library(EVAL::stat_seq ALIAS eval_seq)

# Alias for compatibility with existing code (SYCL)
if(EVAL_BACKEND STREQUAL "SYCL")
  add_library(pszstat_seq ALIAS eval_seq)
endif()

# ------------------------------------------------------------------------------
# Installation + package config
# ------------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS
  utils_headers
  eval_seq
  EXPORT EVALTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TARGET eval_dp)
  install(TARGETS
    eval_dp
    EXPORT EVALTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

install(DIRECTORY
  include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT EVALTargets
  FILE EVALTargets.cmake
  NAMESPACE EVAL::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EVAL
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/EVALConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/EVALConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/EVALConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/EVAL"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/EVALConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/EVALConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/EVAL"
)
