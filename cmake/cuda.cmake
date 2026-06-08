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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/include/cusz>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/utils/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz/include/cusz>
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

find_package(EVAL QUIET)
if(NOT TARGET EVAL::utils_headers AND NOT EVAL_FOUND)
  add_subdirectory(utils)
endif()

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------

find_package(PHF QUIET)
if(NOT TARGET PHF::phf_cu AND NOT PHF_FOUND)
  add_subdirectory(codec/hf)
endif()
if(TARGET phf_cu AND NOT TARGET PSZ::CUDA::phf)
  add_library(PSZ::CUDA::phf ALIAS phf_cu)
  add_library(CUSZ::phf      ALIAS phf_cu)
elseif(TARGET PHF::phf_cu AND NOT TARGET PSZ::CUDA::phf)
  add_library(PSZ::CUDA::phf ALIAS PHF::phf_cu)
  add_library(CUSZ::phf      ALIAS PHF::phf_cu)
endif()

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
  psz/src/buf_comp.cc
  psz/src/buf_comp_dummy.cu
)
target_link_libraries(psz_cu_mem
  PUBLIC
    psz_cu_compile_settings
    EVAL::stat_cu
    DEPS::deps
    PHF::phf_cu
    CUDA::cudart
)

add_library(psz_cu_core
  psz/src/compile/hist_generic.cu
  psz/src/compile/histsp.cu
  psz/src/compile/proto_lrz_c.cu
  psz/src/compile/proto_lrz_x.cu
  psz/src/compile/spvn.cu
  psz/src/compile/lrz_c.cu
  psz/src/compile/lrz_x.cu
  psz/src/compile/spl_y24_c_u1.cu
  psz/src/compile/spl_y24_c_u2.cu
  psz/src/compile/spl_y24_x_u1.cu
  psz/src/compile/spl_y24_x_u2.cu
  psz/src/compile/spl_y25_c_u1.cu
  psz/src/compile/spl_y25_c_u2.cu
  psz/src/compile/spl_y25_x_u1.cu
  psz/src/compile/spl_y25_x_u2.cu
)
target_link_libraries(psz_cu_core
  PUBLIC
    psz_cu_compile_settings
    psz_cu_mem
    PHF::phf_cu
    CUDA::cudart
)

add_library(psz_cu_utils
  psz/src/cli/verinfo.cc
  psz/src/cli/verinfo.cu
  psz/src/cli/verinfo_nv.cu
  psz/src/cli/context.cc
  psz/src/header.c
)
target_link_libraries(psz_cu_utils
  PUBLIC
    psz_cu_compile_settings
    PHF::phf_cu
    EVAL::stat_seq
    EVAL::viewer_cu
    CUDA::cudart
    CUDA::nvml
    CUDA::cuda_driver
)

if(PSZ_ACTIVATE_LC)
  add_compile_definitions(PSZ_USE_LC_FIXED)
  add_library(lc_gen
    third_party/lc_gen/lc_connector.cu
    third_party/lc_gen/comp-tcms.cu third_party/lc_gen/decomp-tcms.cu
    third_party/lc_gen/comp-bitr.cu third_party/lc_gen/decomp-bitr.cu
    third_party/lc_gen/comp-rtr.cu  third_party/lc_gen/decomp-rtr.cu
  )
  target_compile_options(lc_gen PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -mno-fma>
  )
  target_link_libraries(lc_gen PUBLIC psz_cu_compile_settings CUDA::cudart)
endif()


add_library(cusz
  psz/src/compressor.cc
  psz/src/libcusz.cc
)
target_link_libraries(cusz
  PUBLIC
    psz_cu_compile_settings
    psz_cu_core
    psz_cu_mem
    psz_cu_utils
    EVAL::stat_cu
    PHF::phf_cu
    FZG::fzg_cu
    CUDA::cudart
)
if(PSZ_ACTIVATE_LC)
  target_link_libraries(cusz PUBLIC lc_gen)
endif()

# ------------------------------------------------------------------------------
# Executable
# ------------------------------------------------------------------------------

add_executable(cusz-bin psz/src/cli/cli.cc psz/src/cli/executor.cc)
set_source_files_properties(psz/src/cli/cli.cc psz/src/cli/executor.cc PROPERTIES LANGUAGE CUDA)
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

  # `ninja check` (or `make check`) runs the full ctest suite in parallel.
  # GPU tests carry RESOURCE_LOCK gpu (set in test/cmake/cuda-test.cmake) so
  # they serialize against each other while CPU-only tests (portable_*,
  # zigzag, lrz_seq, ...) run concurrently up to N workers.
  include(ProcessorCount)
  ProcessorCount(N)
  if(N EQUAL 0)
    set(N 8)
  endif()
  add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --parallel ${N}
    USES_TERMINAL
  )
endif()

# ------------------------------------------------------------------------------
# Installation (CUSZ:: namespace, back compat)
# ------------------------------------------------------------------------------

install(TARGETS psz_cu_compile_settings EXPORT CUSZTargets)

install(TARGETS
  psz_seq_core
  psz_cu_core
  psz_cu_mem
  psz_cu_utils
  cusz
  eval_cu
  eval_seq
  eval_viewer_cu
  EXPORT CUSZTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(PSZ_ACTIVATE_LC)
  install(TARGETS
    lc_gen
    EXPORT CUSZTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

install(TARGETS
  cusz-bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)


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
