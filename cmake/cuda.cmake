add_compile_definitions(PSZ_USE_CUDA)
add_compile_definitions(_PORTABLE_USE_CUDA)

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)
include(CTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/psz/src/cusz_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h)

add_library(psz_cu_compile_settings INTERFACE)
add_library(CUSZ::compile_settings ALIAS psz_cu_compile_settings)

target_compile_definitions(
  psz_cu_compile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(
  psz_cu_compile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda
  --expt-relaxed-constexpr -Wno-deprecated-declarations>)
target_compile_features(psz_cu_compile_settings INTERFACE cxx_std_17 cuda_std_17)

target_include_directories(
  psz_cu_compile_settings
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/portable/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/hf/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/hf/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/fzg/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/codec/fzg/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>
)

# option(PSZ_RESEARCH_HUFFBK_CUDA
# "build research artifacts: create Huffman codebook on GPU" OFF)
option(PSZ_REACTIVATE_THRUSTGPU
  "build previously thrust implemented functions" OFF)

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
add_library(PSZ::CUDA::stat ALIAS psz_cu_stat)
add_library(CUSZ::stat ALIAS psz_cu_stat)

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
  CUDA::cudart
)
add_library(PSZ::CUDA::core ALIAS psz_cu_core)
add_library(CUSZ::core ALIAS psz_cu_core)

add_library(psz_cu_mem
  portable/src/mem/memobj.f.cc
  portable/src/mem/memobj.i.cc
  portable/src/mem/memobj.u.cc
  portable/src/mem/memobj.misc.cc)
add_library(CUSZ::mem ALIAS psz_cu_mem)
add_library(PSZ::cu_mem ALIAS psz_cu_mem)
target_link_libraries(psz_cu_mem
  PUBLIC
  psz_cu_compile_settings
  psz_cu_stat
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
  CUDA::cudart CUDA::nvml
)
add_library(PSZ::CUDA::utils ALIAS psz_cu_utils)
add_library(CUSZ::utils ALIAS psz_cu_utils)

add_library(psz_cu_phf
  codec/hf/src/hf_est.cc
  codec/hf/src/hf_bk_impl1.seq.cc
  codec/hf/src/hf_bk_impl2.seq.cc
  codec/hf/src/hf_bk_internal.seq.cc
  codec/hf/src/hf_bk.seq.cc
  codec/hf/src/hf_canon.seq.cc
  codec/hf/src/hf_kernels.cu
  codec/hf/src/hf_ood.cc
  codec/hf/src/hf_hl.cc
  codec/hf/src/hf_buf.cc
  codec/hf/src/libphf.cc
)
target_link_libraries(psz_cu_phf
  PUBLIC
  psz_cu_compile_settings
  psz_cu_stat
  CUDA::cuda_driver
)
add_library(PSZ::CUDA::phf ALIAS psz_cu_phf)
add_library(CUSZ::phf ALIAS psz_cu_phf)

add_library(psz_cu_fzg
  codec/fzg/src/fzg_kernel.cu
  codec/fzg/src/fzg_class.cc
)
target_link_libraries(psz_cu_fzg
  PUBLIC
  psz_cu_compile_settings
  psz_cu_core
)
add_library(PSZ::CUDA::fzg ALIAS psz_cu_fzg)
add_library(CUSZ::fzg ALIAS psz_cu_fzg)

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
  psz_cu_phf
  psz_cu_fzg
  CUDA::cudart
)
add_library(PSZ::CUDA::cusz ALIAS cusz)
add_library(CUSZ::cusz ALIAS cusz)

# m export binary "cusz"
add_executable(cusz-bin psz/src/cli/cli.cc)
set_source_files_properties(psz/src/cli/cli.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(cusz-bin PRIVATE cusz)
set_target_properties(cusz-bin PROPERTIES OUTPUT_NAME cusz)

# enable examples and testing
if(PSZ_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

if(PSZ_BUILD_PYBINDING)
  # make python binding
  find_package(SWIG REQUIRED)
  include(${SWIG_USE_FILE})
  message("[psz::info] $\{SWIG_USE_FILE\}: " ${SWIG_USE_FILE})

  # deprecated as of 3.12; cmake --help-policy CMP0148
  # find_package(PythonLibs REQUIRED)
  # message("[psz::info] $\{PYTHON_INCLUDE_DIRS\}: " ${PYTHON_INCLUDE_DIRS})
  find_package(Python REQUIRED COMPONENTS Development)

  # include_directories(${Python_INCLUDE_DIRS})
  message("[psz::info] $\{Python_FOUND\}: " ${Python_FOUND})
  message("[psz::info] $\{Python_VERSION\}: " ${Python_VERSION})
  message("[psz::info] $\{Python_INCLUDE_DIRS\}: " ${Python_INCLUDE_DIRS})
  message("[psz::info] $\{Python_LINK_OPTIONS\}: " ${Python_LINK_OPTIONS})
  message("[psz::info] $\{Python_LIBRARIES\}: " ${Python_LIBRARIES})
  message("[psz::info] $\{Python_LIBRARY_DIRS\}: " ${Python_LIBRARY_DIRS})

  set(SWIG_INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/psz/include
    ${CMAKE_CURRENT_SOURCE_DIR}/hf/include
    ${Python_INCLUDE_DIRS}
  )
  include_directories(${SWIG_INCLUDE_DIRECTORIES})

  # -------------------
  # add the 1st library
  # -------------------
  swig_add_library(pycusz
    LANGUAGE python
    TYPE SHARED
    SOURCES py/pycusz.i)

  target_include_directories(pycusz PRIVATE ${SWIG_INCLUDE_DIRECTORIES})
  set_target_properties(pycusz PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(pycusz PRIVATE CUDA::cudart ${PYTHON_LIBRARIES}
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
    SOURCES py/pycuhf.i)

  target_include_directories(pycuhf PRIVATE ${SWIG_INCLUDE_DIRECTORIES})
  set_target_properties(pycuhf PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(pycuhf PRIVATE CUDA::cudart ${PYTHON_LIBRARIES}
    psz_cu_mem
    psz_cu_phf
  )
endif()

# ###############################################################################
# #  installation using CUSZ:: namespace (back compat) ##########################
# ###############################################################################

# install libs
install(TARGETS psz_cu_compile_settings EXPORT CUSZTargets)
install(TARGETS
  psz_seq_core
  psz_cu_core
  psz_cu_stat
  psz_cu_mem
  psz_cu_utils
  psz_cu_phf
  psz_cu_fzg
  cusz
  EXPORT CUSZTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install the executable
install(TARGETS
  cusz-bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(PSZ_BUILD_PYBINDING)
  install(TARGETS
    pycusz
    pycuhf
    EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# install the package
install(
  EXPORT CUSZTargets
  NAMESPACE CUSZ::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ/)
include(CMakePackageConfigHelpers) # generate and install package config files
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUSZConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ)

# install headers
install(DIRECTORY
  portable/include/
  psz/include/
  codec/hf/include/
  codec/fzg/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz/)
