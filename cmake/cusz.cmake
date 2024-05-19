add_compile_definitions(PSZ_USE_CUDA)

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)
include(CTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/psz/src/cusz_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h)

add_library(pszcompile_settings INTERFACE)

target_compile_definitions(
  pszcompile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(
  pszcompile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda
  --expt-relaxed-constexpr -Wno-deprecated-declarations>)
target_compile_features(pszcompile_settings INTERFACE cxx_std_17 cuda_std_17)

target_include_directories(
  pszcompile_settings
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
    pszstat_cu
    psz/src/stat/compare.stl.cc
    psz/src/stat/extrema.cu
    psz/src/stat/cmpg1_4.cu
    psz/src/stat/cmpg2.cu
    psz/src/stat/cmpg3.cu
    psz/src/stat/cmpg4_1.cu
    psz/src/stat/cmpg4_2.cu
    psz/src/stat/cmpg5_1.cu
    psz/src/stat/cmpg5_2.cu)
else()
  add_library(
    pszstat_cu
    psz/src/stat/compare.stl.cc
    psz/src/stat/extrema.cu
    psz/src/stat/cmpg2.cu
    psz/src/stat/cmpg4_1.cu
    psz/src/stat/cmpg4_2.cu
    psz/src/stat/cmpg5_1.cu
    psz/src/stat/cmpg5_2.cu)
endif()

target_link_libraries(pszstat_cu
  PUBLIC pszcompile_settings
)

# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(
  pszcore_cu
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
target_link_libraries(pszcore_cu
  PUBLIC pszcompile_settings CUDA::cudart
  pszmem_cu
)

add_library(pszmem_cu
  psz/src/mem/memobj.f.cc
  psz/src/mem/memobj.i.cc
  psz/src/mem/memobj.u.cc
  psz/src/mem/memobj.misc.cc)
target_link_libraries(pszmem_cu
  PUBLIC pszcompile_settings CUDA::cudart
  pszstat_cu
)

add_library(pszutils
  psz/src/utils/vis_stat.cc
  psz/src/context.cc
  psz/src/utils/timer_cpu.cc
  psz/src/utils/timer_gpu.cc
)
target_link_libraries(pszutils
  PUBLIC pszcompile_settings CUDA::cudart
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
)
target_link_libraries(phf_cu
  PUBLIC pszcompile_settings CUDA::cuda_driver
  pszstat_cu
)

add_library(cusz
  psz/src/pipeline/testframe.cc
  psz/src/compressor.cc
  psz/src/log/sanitize.cc
  psz/src/cusz_lib.cc
)
target_link_libraries(cusz
  PUBLIC pszcompile_settings CUDA::cudart
  pszcore_cu
  pszstat_cu
  pszmem_cu
  pszutils
  phf_cu
)

# export binary "cusz"
add_executable(cusz-bin psz/src/cli_psz.cc)
target_link_libraries(cusz-bin PRIVATE cusz)
set_target_properties(cusz-bin PROPERTIES OUTPUT_NAME cusz)

# enable examples and testing
if(PSZ_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# installation
install(TARGETS pszcompile_settings EXPORT CUSZTargets)
install(TARGETS pszcore_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszstat_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszmem_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszutils EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS phf_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz-bin EXPORT CUSZTargets)

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

# # back compat
install(DIRECTORY psz/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz/)

# # new testing
install(DIRECTORY psz/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/psz)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/psz/)
