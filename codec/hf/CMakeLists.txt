
cmake_minimum_required(VERSION 3.18)

project(PHF
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

find_package(CUDAToolkit REQUIRED)

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

# PORTABLE is header-only; extend its interface for PHF
if(TARGET PORTABLE)
  set(PHF_PORTABLE_TARGET PORTABLE)
elseif(TARGET PORTABLE::PORTABLE)
  set(PHF_PORTABLE_TARGET PORTABLE::PORTABLE)
endif()

if(NOT PHF_PORTABLE_TARGET)
  message(FATAL_ERROR "PORTABLE target not available after setup")
endif()

target_compile_definitions(${PHF_PORTABLE_TARGET}
  INTERFACE
    PHF_USE_CUDA
)

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------

add_library(phf_cu
  src/hf_bk.seq.cc
  src/hf_bk_impl1.seq.cc
  src/hf_bk_impl2.seq.cc
  src/hf_bk_internal.seq.cc
  src/hf_buf.cc
  src/hf_buf_dummy.cu
  src/hf_canon.seq.cc
  src/hf_est.cc
  src/hf_hl.cc
  src/compile/hf_kernels.cu
  src/libphf.cc
  src/compile/hfr_r1.cu
  src/compile/hfr_r2.cu
  src/compile/hfr_r3.cu
  src/compile/hfr_r4.cu
  src/compile/hfr_concat.cu
  src/compile/hfr_encode_r2.cu
  src/compile/hfr_encode_r3.cu
  src/compile/hfr_encode_r4.cu
  src/compile/hfr-pbkc_r2.cu
  src/compile/hfr-pbkc_r3.cu
  src/compile/hfr-pbkc_r4.cu
  src/compile/hfr-pbkgo_r2.cu
  src/compile/hfr-pbkgo_r3.cu
  src/compile/hfr-pbkgo_r4.cu
  src/hfr-pbkc_x.cc
  src/hfr-pbkgo_x.cc
  src/hfr-pbk_decoder.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../psz/src/book/pbk25_r128_d.cu
)
add_library(PHF::phf_cu ALIAS phf_cu)

target_include_directories(phf_cu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../psz/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../psz/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(phf_cu
  PUBLIC
    CUDA::cudart
    ${PHF_PORTABLE_TARGET}
)

set_target_properties(phf_cu
  PROPERTIES
    LINKER_LANGUAGE CXX
)

# ------------------------------------------------------------------------------
# Installation + package config
# ------------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS phf_cu
  EXPORT PHFTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY
  include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT PHFTargets
  FILE PHFTargets.cmake
  NAMESPACE PHF::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PHF
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/PHFConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/PHFConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/PHFConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PHF
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/PHFConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/PHFConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PHF
)
