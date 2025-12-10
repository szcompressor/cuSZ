
cmake_minimum_required(VERSION 3.18)

project(FZG
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

find_package(CUDAToolkit REQUIRED)

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

target_include_directories(fzg_cu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(fzg_cu
  PUBLIC
    CUDA::cudart
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