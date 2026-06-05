
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
# Compiled test-data utilities (PORTABLE::testutils): rand_array_* (cuRAND) +
# the Synth generator. Separate from the header-only PORTABLE target.
# ------------------------------------------------------------------------------

add_library(portable_testutils
  src/utils/rand.seq.cc
  src/utils/rand.cu.cc
  src/utils/synth.cc
)
add_library(PORTABLE::testutils ALIAS portable_testutils)
target_link_libraries(portable_testutils
  PUBLIC
    PORTABLE
    CUDA::cudart
    CUDA::curand
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

# ------------------------------------------------------------------------------
# Tests (pure-ctest, no gtest dependency).
# Each test_<unit> binary uses assert() and exits 0 on pass, non-zero on fail.
# Enabled when consumer enables BUILD_TESTING (CTest convention).
# ------------------------------------------------------------------------------

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
