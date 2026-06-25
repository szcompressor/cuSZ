
cmake_minimum_required(VERSION 3.18)

if(NOT PSZ_BACKEND)
  set(PSZ_BACKEND "CUDA")
endif()
string(TOUPPER "${PSZ_BACKEND}" FZG_BACKEND_UPPER)

if(FZG_BACKEND_UPPER STREQUAL "HIP")
  project(FZG VERSION 0.1.0 LANGUAGES CXX HIP)
  find_package(hip REQUIRED)
  set(_FZG_GPU_RT hip::device)
else()
  project(FZG VERSION 0.1.0 LANGUAGES CXX CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(_FZG_GPU_RT CUDA::cudart)
endif()

# ------------------------------------------------------------------------------
# Dependency: PORTABLE (installed or fallback)
# ------------------------------------------------------------------------------

find_package(PORTABLE QUIET)

if(NOT TARGET PORTABLE AND NOT TARGET PORTABLE::PORTABLE)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../portable")
    add_subdirectory(
      "${CMAKE_CURRENT_SOURCE_DIR}/../../portable"
      "${CMAKE_CURRENT_BINARY_DIR}/portable"
    )
  else()
    message(FATAL_ERROR
      "PORTABLE not found and ../../portable does not exist"
    )
  endif()
endif()

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------

add_library(fzg_cu
  src/fzg_hl.cc
  src/fzg_kernel.cu
  src/fzg_buf.cc
)

add_library(FZG::fzg_cu ALIAS fzg_cu)

if(FZG_BACKEND_UPPER STREQUAL "HIP")
  set_source_files_properties(src/fzg_kernel.cu PROPERTIES LANGUAGE HIP)
  target_compile_definitions(fzg_cu PRIVATE FZG_USE_HIP _PORTABLE_USE_HIP)
  # Host .cc TUs (fzg_hl.cc, fzg_buf.cc) and the HIP kernel TU all need the
  # CUDA->HIP translation prelude and the cuda_runtime.h shim.
  target_include_directories(fzg_cu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/hip-compat)
  target_compile_options(fzg_cu PRIVATE
    -include ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/hip-compat/psz_hip_compat.h)
endif()

target_include_directories(fzg_cu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(fzg_cu
  PUBLIC
    ${_FZG_GPU_RT}
    PORTABLE
)

set_target_properties(fzg_cu
  PROPERTIES
    LINKER_LANGUAGE CXX
)

# ------------------------------------------------------------------------------
# Installation + package config
# ------------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS fzg_cu
  EXPORT FZGTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY
  include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT FZGTargets
  FILE FZGTargets.cmake
  NAMESPACE FZG::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZG
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/FZGConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/FZGConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/FZGConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZG
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/FZGConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/FZGConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FZG
)