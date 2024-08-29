add_compile_definitions(PSZ_USE_CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
message("[psz::info] $\{OpenMP_FOUND\}: " ${OpenMP_FOUND})

include(GNUInstallDirs)
include(CTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/psz/src/cusz_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h)

add_library(psz_cu_compile_settings INTERFACE)

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
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psz/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hf/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hf/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>)

# option(PSZ_RESEARCH_HUFFBK_CUDA
# "build research artifacts: create Huffman codebook on GPU" OFF)
option(PSZ_REACTIVATE_THRUSTGPU
  "build previously thrust implemented functions" OFF)

if(PSZ_REACTIVATE_THRUSTGPU)
  add_compile_definitions(REACTIVATE_THRUSTGPU)
  add_library(
    psz_cu_stat
    psz/src/stat/compare.stl.cc
    psz/src/stat/identical/all.thrust.cu
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
    psz/src/stat/maxerr/f8.thrust.cu)
else()
  add_library(
    psz_cu_stat
    psz/src/stat/compare.stl.cc
    psz/src/stat/identical/all.thrust.cu
    psz/src/stat/extrema/f4.cu
    psz/src/stat/extrema/f8.cu
    psz/src/stat/calcerr/f4.cu
    psz/src/stat/calcerr/f8.cu
    psz/src/stat/assess/f4.cu
    psz/src/stat/assess/f8.cu
    psz/src/stat/assess/f4.thrust.cu
    psz/src/stat/assess/f8.thrust.cu
    psz/src/stat/maxerr/f4.thrust.cu
    psz/src/stat/maxerr/f8.thrust.cu)
endif()

target_link_libraries(psz_cu_stat
  PUBLIC psz_cu_compile_settings
)

# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(
  psz_cu_core
  psz/src/kernel/l23.seq.cc
  psz/src/kernel/hist.seq.cc
  psz/src/kernel/histsp.seq.cc
  psz/src/kernel/spvn.seq.cc
  psz/src/kernel/dryrun.cu
  psz/src/kernel/lproto.cu
  psz/src/kernel/spvn.cu
  psz/src/kernel/l23_c.cu
  psz/src/kernel/l23_x.cu
  psz/src/kernel/spline3.cu
  psz/src/kernel/hist.cu
  psz/src/kernel/hist.seq.cc # workaround
  psz/src/kernel/histsp.cu
  psz/src/kernel/histsp.seq.cc
  psz/src/kernel/l23r.cu
  psz/src/module/lrz.cc
  psz/src/module/lrz_cxx.cu
  psz/src/module/spl_cxx.cu
  psz/src/module/hist_cxx.cu
  psz/src/module/scatter_cxx.cu
  psz/src/kernel/spv.cu # a thrust impl
)
target_link_libraries(psz_cu_core
  PUBLIC psz_cu_compile_settings CUDA::cudart
  psz_cu_mem
)

add_library(psz_cu_mem
  psz/src/mem/memobj.f.cc
  psz/src/mem/memobj.i.cc
  psz/src/mem/memobj.u.cc
  psz/src/mem/memobj.misc.cc)
target_link_libraries(psz_cu_mem
  PUBLIC psz_cu_compile_settings CUDA::cudart
  psz_cu_stat
)

add_library(psz_cu_utils
  psz/src/utils/viewer.cc
  psz/src/utils/viewer.cu
  psz/src/utils/verinfo.cc
  psz/src/utils/verinfo.cu
  psz/src/utils/verinfo_nv.cu
  psz/src/utils/vis_stat.cc
  psz/src/utils/context.cc
  psz/src/utils/timer_cpu.cc
  psz/src/utils/timer_gpu.cc
  psz/src/utils/header.c
)
target_link_libraries(psz_cu_utils
  PUBLIC psz_cu_compile_settings CUDA::cudart CUDA::nvml
)

add_library(phf_cu
  hf/src/hfclass.cc
  hf/src/hf_est.cc
  hf/src/hfbk_impl1.seq.cc
  hf/src/hfbk_impl2.seq.cc
  hf/src/hfbk_internal.seq.cc
  hf/src/hfbk.seq.cc
  hf/src/hfcanon.seq.cc
  hf/src/hfcxx_module.cu
  hf/src/libphf.cc
)
target_link_libraries(phf_cu
  PUBLIC psz_cu_compile_settings CUDA::cuda_driver
  psz_cu_stat
)

add_library(cusz
  psz/src/pipeline/compressor.cc
  psz/src/log/sanitize.cc
  psz/src/libcusz.cc
)
target_link_libraries(cusz
  PUBLIC psz_cu_compile_settings CUDA::cudart
  psz_cu_core
  psz_cu_stat
  psz_cu_mem
  psz_cu_utils
  phf_cu
)

#m export binary "cusz"
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
  phf_cu 
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
  phf_cu 
  )
endif()

# installation
install(TARGETS psz_cu_compile_settings EXPORT CUSZTargets)
install(TARGETS psz_cu_core EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_cu_stat EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_cu_mem EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_cu_utils EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS phf_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz-bin EXPORT CUSZTargets)
if(PSZ_BUILD_PYBINDING)
  install(TARGETS pycusz EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(TARGETS pycuhf EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(
  EXPORT CUSZTargets
  NAMESPACE CUSZ::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ/)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/CUSZConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/CUSZConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ)

# back compat
install(DIRECTORY psz/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz/)

# new testing
install(DIRECTORY psz/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/psz)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/psz/)
