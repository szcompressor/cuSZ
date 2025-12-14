
cmake_minimum_required(VERSION 3.18)

project(PORTABLE
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------
# Header-only interface target
# ------------------------------------------------------------------------------

add_library(PORTABLE INTERFACE)

target_compile_features(PORTABLE
  INTERFACE
    cxx_std_17
)

target_compile_definitions(PORTABLE
  INTERFACE
    _PORTABLE_USE_CUDA
)

target_include_directories(PORTABLE
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ------------------------------------------------------------------------------
# Installation + package config
# ------------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS PORTABLE
  EXPORT PORTABLETargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY
  include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT PORTABLETargets
  FILE PORTABLETargets.cmake
  NAMESPACE PORTABLE::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PORTABLE
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/PORTABLEConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/PORTABLEConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/PORTABLEConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/PORTABLE"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/PORTABLEConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/PORTABLEConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/PORTABLE"
)