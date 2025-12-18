# ------------------------------------------------------------------------------
# Source code switches 
# ------------------------------------------------------------------------------

add_compile_definitions(
  PSZ_USE_CUDA
  _PORTABLE_USE_CUDA
)

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)
include(CTest)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/psz/src/cusz_version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h"
  @ONLY
)

# ------------------------------------------------------------------------------
# Common compile settings (interface target)
# ------------------------------------------------------------------------------

add_library(psz_cu_compile_settings INTERFACE)

target_compile_features(psz_cu_compile_settings
  INTERFACE
    cxx_std_17
    cuda_std_17
)

target_compile_definitions(psz_cu_compile_settings
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>
)

target_compile_options(psz_cu_compile_settings
  INTERFACE
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda;--expt-relaxed-constexpr;-Wno-deprecated-declarations>
)

target_include_directories(psz_cu_compile_settings
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>
)

# ------------------------------------------------------------------------------
# Dependencies (installed or fallback)
# ------------------------------------------------------------------------------

find_package(PORTABLE QUIET)
if(NOT TARGET PORTABLE::PORTABLE AND NOT TARGET PORTABLE)
  add_subdirectory(portable)
endif()

# Normalize PORTABLE target name
set(_PORTABLE_TARGET "")
if(TARGET PORTABLE::PORTABLE)
  set(_PORTABLE_TARGET PORTABLE::PORTABLE)
elseif(TARGET PORTABLE)
  set(_PORTABLE_TARGET PORTABLE)
else()
  message(FATAL_ERROR
    "PORTABLE target not available. Provide PORTABLE or add the portable subdirectory."
  )
endif()

# Back-compat alias used throughout this project
if(NOT TARGET DEPS::deps)
  add_library(DEPS::deps ALIAS "${_PORTABLE_TARGET}")
endif()

target_link_libraries(psz_cu_compile_settings
  INTERFACE
    DEPS::deps
)

find_package(FZG QUIET)
if(NOT TARGET FZG::fzg_cu AND NOT FZG_FOUND)
  add_subdirectory(codec/fzg)
endif()

find_package(PHF QUIET)
if(NOT TARGET PHF::phf_cu AND NOT PHF_FOUND)
  add_subdirectory(codec/hf)
endif()

# ------------------------------------------------------------------------------
# Options (DEPRECATED options kept for back-compat)
# ------------------------------------------------------------------------------

# option(PSZ_RESEARCH_HUFFBK_CUDA "build research artifacts: create Huffman codebook on GPU" OFF)
option(PSZ_REACTIVATE_THRUSTGPU "build previously thrust implemented functions" OFF)

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------

if(PSZ_REACTIVATE_THRUSTGPU)
  add_compile_definitions(REACTIVATE_THRUSTGPU)

  add_library(psz_cu_stat
    psz/src/stat/compare.stl.cc
    psz/src/stat/identical/all.thrust.cu
    psz/src/stat/identical/all.cu
    psz/src/stat/extrema/f4.cu
    psz/src/stat/extrema/f8.cu
    psz/src/stat/extrema/f4.thrust.cu
    psz/src/stat/extrema/f8.thrust.cu
    psz/src/stat/calcerr/f4.cu
    psz/src/stat/calcerr/f8.cu
    psz/src/stat/assess/f4.cu
    psz/src/stat/assess/f8.cu
    psz/src/stat/assess/f4.thrust.cu
    psz/src/stat/assess/f8.thrust.cu
    psz/src/stat/maxerr/f4.thrust.cu
    psz/src/stat/maxerr/f8.thrust.cu
    psz/src/stat/maxerr/max_err.cu
  )
else()
  add_library(psz_cu_stat
    psz/src/stat/compare.stl.cc
    psz/src/stat/identical/all.cu
    psz/src/stat/extrema/f4.cu
    psz/src/stat/extrema/f8.cu
    psz/src/stat/calcerr/f4.cu
    psz/src/stat/calcerr/f8.cu
    psz/src/stat/assess/f4.cu
    psz/src/stat/assess/f8.cu
    psz/src/stat/maxerr/max_err.cu
  )
endif()

target_link_libraries(psz_cu_stat
  PUBLIC
    psz_cu_compile_settings
)

# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(psz_seq_core
  psz/src/kernel/lrz.seq.cc
  psz/src/kernel/hist_generic.seq.cc
  psz/src/kernel/histsp.seq.cc
  psz/src/kernel/spvn.seq.cc
)
target_link_libraries(psz_seq_core
  PUBLIC
    psz_cu_compile_settings
)

add_library(psz_cu_mem
  psz/src/mem/buf_comp.cc
)
target_link_libraries(psz_cu_mem
  PUBLIC
    psz_cu_compile_settings
    psz_cu_stat
    DEPS::deps
    PHF::phf_cu
    CUDA::cudart
)

add_library(psz_cu_core
  psz/src/kernel/hist_generic.cu
  psz/src/kernel/histsp.cu
  psz/src/kernel/proto_lrz_c.cu
  psz/src/kernel/proto_lrz_x.cu
  psz/src/kernel/spvn.cu
  psz/src/kernel/lrz_c.cu
  psz/src/kernel/lrz_x.cu
  psz/src/kernel/spline3.cu
)
target_link_libraries(psz_cu_core
  PUBLIC
    psz_cu_compile_settings
    psz_cu_mem
    PHF::phf_cu
    CUDA::cudart
)

add_library(psz_cu_utils
  psz/src/utils/viewer.cc
  psz/src/utils/viewer.cu
  psz/src/utils/verinfo.cc
  psz/src/utils/verinfo.cu
  psz/src/utils/verinfo_nv.cu
  psz/src/utils/vis_stat.cc
  psz/src/utils/context.cc
  psz/src/utils/header.c
)
target_link_libraries(psz_cu_utils
  PUBLIC
    psz_cu_compile_settings
    PHF::phf_cu
    CUDA::cudart
    CUDA::nvml
    CUDA::cuda_driver
)

add_library(cusz
  psz/src/compressor.cc
  psz/src/libcusz.cc
)
target_link_libraries(cusz
  PUBLIC
    psz_cu_compile_settings
    psz_cu_core
    psz_cu_stat
    psz_cu_mem
    psz_cu_utils
    PHF::phf_cu
    FZG::fzg_cu
    CUDA::cudart
)

# ------------------------------------------------------------------------------
# Executable
# ------------------------------------------------------------------------------

add_executable(cusz-bin psz/src/cli/cli.cc)
set_source_files_properties(psz/src/cli/cli.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(cusz-bin PRIVATE cusz)
set_target_properties(cusz-bin PROPERTIES OUTPUT_NAME cusz)

# ------------------------------------------------------------------------------
# Examples / Tests
# ------------------------------------------------------------------------------

if(PSZ_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# ------------------------------------------------------------------------------
# Python binding (SWIG)
# ------------------------------------------------------------------------------

if(PSZ_BUILD_PYBINDING)
  find_package(SWIG REQUIRED)
  include(${SWIG_USE_FILE})
  message("[psz::info] ${SWIG_USE_FILE}: ${SWIG_USE_FILE}")

  find_package(Python REQUIRED COMPONENTS Development)

  message("[psz::info] ${Python_FOUND}: ${Python_FOUND}")
  message("[psz::info] ${Python_VERSION}: ${Python_VERSION}")
  message("[psz::info] ${Python_INCLUDE_DIRS}: ${Python_INCLUDE_DIRS}")
  message("[psz::info] ${Python_LINK_OPTIONS}: ${Python_LINK_OPTIONS}")
  message("[psz::info] ${Python_LIBRARIES}: ${Python_LIBRARIES}")
  message("[psz::info] ${Python_LIBRARY_DIRS}: ${Python_LIBRARY_DIRS}")

  set(SWIG_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/psz/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/codec/hf/include"
    "${Python_INCLUDE_DIRS}"
  )
  include_directories(${SWIG_INCLUDE_DIRECTORIES})

  # -------------------
  # add the 1st library
  # -------------------
  swig_add_library(pycusz
    LANGUAGE python
    TYPE SHARED
    SOURCES py/pycusz.i
  )
  target_include_directories(pycusz PRIVATE ${SWIG_INCLUDE_DIRECTORIES})
  set_target_properties(pycusz PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(pycusz
    PRIVATE
      CUDA::cudart
      ${PYTHON_LIBRARIES}
      cusz
      psz_cu_core
      psz_cu_stat
      psz_cu_mem
      psz_cu_utils
      psz_cu_phf
      psz_cu_fzg
  )

  # -------------------
  # add the 2nd library
  # -------------------
  swig_add_library(pycuhf
    LANGUAGE python
    TYPE SHARED
    SOURCES py/pycuhf.i
  )
  target_include_directories(pycuhf PRIVATE ${SWIG_INCLUDE_DIRECTORIES})
  set_target_properties(pycuhf PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(pycuhf
    PRIVATE
      CUDA::cudart
      ${PYTHON_LIBRARIES}
      psz_cu_mem
      psz_cu_phf
  )
endif()

# ------------------------------------------------------------------------------
# Installation (CUSZ:: namespace, back compat)
# ------------------------------------------------------------------------------

install(TARGETS psz_cu_compile_settings EXPORT CUSZTargets)

install(TARGETS
  psz_seq_core
  psz_cu_core
  psz_cu_stat
  psz_cu_mem
  psz_cu_utils
  cusz
  EXPORT CUSZTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS
  cusz-bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(PSZ_BUILD_PYBINDING)
  install(TARGETS
    pycusz
    pycuhf
    EXPORT CUSZTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

install(
  EXPORT CUSZTargets
  NAMESPACE CUSZ::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUSZConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ
)

install(DIRECTORY
  portable/include
  psz/include
  codec/hf/include
  codec/fzg/include
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz
)