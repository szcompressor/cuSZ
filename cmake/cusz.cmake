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

option(PSZ_RESEARCH_HUFFBK_CUDA
  "build research artifacts: create Huffman codebook on GPU" OFF)
option(PSZ_REACTIVATE_THRUSTGPU
  "build previously thrust implemented functions" OFF)

# seprate later
add_library(psztime psz/src/utils/timer_cpu.cc psz/src/utils/timer_gpu.cc)
target_link_libraries(psztime PUBLIC pszcompile_settings CUDA::cudart)

add_library(pszstat_seq psz/src/stat/compare.stl.cc)
target_link_libraries(pszstat_seq PUBLIC pszcompile_settings)

if(PSZ_REACTIVATE_THRUSTGPU)
  add_compile_definitions(REACTIVATE_THRUSTGPU)
  add_library(
    pszstat_cu
    psz/src/stat/extrema.cu
    psz/src/stat/cmpg1_4.cu

    # psz/src/stat/cmpg1_5.cu
    psz/src/stat/cmpg2.cu
    psz/src/stat/cmpg3.cu
    psz/src/stat/cmpg4_1.cu
    psz/src/stat/cmpg4_2.cu
    psz/src/stat/cmpg5_1.cu
    psz/src/stat/cmpg5_2.cu)
else()
  add_library(
    pszstat_cu psz/src/stat/extrema.cu psz/src/stat/cmpg2.cu psz/src/stat/cmpg4_1.cu
    psz/src/stat/cmpg4_2.cu psz/src/stat/cmpg5_1.cu psz/src/stat/cmpg5_2.cu)
endif()

target_link_libraries(pszstat_cu PUBLIC pszcompile_settings)

# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(pszkernel_seq psz/src/kernel/l23.seq.cc psz/src/kernel/hist.seq.cc
  psz/src/kernel/histsp.seq.cc psz/src/kernel/spvn.seq.cc)
target_link_libraries(pszkernel_seq PUBLIC pszcompile_settings)

add_library(
  pszkernel_cu
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
)
target_link_libraries(pszkernel_cu PUBLIC pszcompile_settings)

# TODO installation
add_library(
  pszmodule2401_cu
  psz/src/module/lrz.cc
  psz/src/module/lrz_cxx.cu
  psz/src/module/spl_cxx.cu
  psz/src/module/hist_cxx.cu
  psz/src/module/scatter_cxx.cu
)
target_link_libraries(pszmodule2401_cu PUBLIC pszcompile_settings
  pszkernel_cu
  CUDA::cudart
)

# TODO installation
add_library(
  phfmodule2403_cu
  hf/src/hfcxx_module.cu
)
target_link_libraries(phfmodule2403_cu PUBLIC pszcompile_settings
  CUDA::cudart
)

add_library(pszanalysis2402
  hf/src/hf_est.cc
  hf/src/hfbk_impl1.seq.cc
  hf/src/hfbk_impl2.seq.cc
  hf/src/hfbk_internal.seq.cc
)
target_link_libraries(pszanalysis2402 PUBLIC pszcompile_settings)

add_library(pszmem psz/src/mem/memseg.cc psz/src/mem/memseg_cu.cc)
target_link_libraries(pszmem PUBLIC pszcompile_settings CUDA::cudart)

add_library(pszutils_seq psz/src/utils/vis_stat.cc psz/src/context.cc)
target_link_libraries(pszutils_seq PUBLIC pszcompile_settings)

add_library(pszspv_cu psz/src/kernel/spv.cu)
target_link_libraries(pszspv_cu PUBLIC pszcompile_settings)

if(PSZ_RESEARCH_HUFFBK_CUDA)
  # define C/Cxx macro: https://stackoverflow.com/a/9017635
  add_compile_definitions(ENABLE_HUFFBK_GPU)
  add_library(pszhfbook_cu psz/src/hf/hfbk_p1.cu psz/src/hf/hfbk_p2.cu
    psz/src/hf/hfsort.thrust.cu)
  target_link_libraries(pszhfbook_cu PUBLIC pszcompile_settings
    CUDA::cuda_driver)
  set_target_properties(pszhfbook_cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  string(FIND "${CUDA_cuda_driver_LIBRARY}" "stub" CUDA_DRIVER_IS_STUB)

  if(NOT ${CUDA_DRIVER_IS_STUB} EQUAL -1)
    message(WARNING "the cuda driver is a stub!! adding --allow-shlib-undefined to fix downstream linking issues")
    target_link_options(pszhfbook_cu PUBLIC $<HOST_LINK:LINKER:--allow-shlib-undefined>)
  endif()
endif(PSZ_RESEARCH_HUFFBK_CUDA)

# cmake option on-and-off-cache: https://stackoverflow.com/a/48984477
# unset(PSZ_RESEARCH_HUFFBK_CUDA CACHE)
add_library(
  pszhfbook_seq hf/src/hfbk_impl1.seq.cc hf/src/hfbk_impl2.seq.cc
  hf/src/hfbk_internal.seq.cc hf/src/hfbk.seq.cc hf/src/hfcanon.seq.cc)
target_link_libraries(pszhfbook_seq PUBLIC pszcompile_settings)

add_library(pszhf_cu hf/src/hfclass.cc)

if(PSZ_RESEARCH_HUFFBK_CUDA)
  target_link_libraries(pszhf_cu PUBLIC pszcompile_settings pszstat_cu
    pszhfbook_cu pszhfbook_seq)
else()
  target_link_libraries(pszhf_cu PUBLIC pszcompile_settings pszstat_cu
    phfmodule2403_cu
    pszhfbook_seq CUDA::cuda_driver)
endif(PSZ_RESEARCH_HUFFBK_CUDA)

# unset(PSZ_RESEARCH_HUFFBK_CUDA CACHE)

# [TODO] maybe a standalone libpszdbg
add_library(pszcomp_cu psz/src/compressor.cc psz/src/log/sanitize.cc)
target_link_libraries(pszcomp_cu PUBLIC pszcompile_settings pszkernel_cu
  pszmodule2401_cu pszstat_cu pszhf_cu CUDA::cudart)

add_library(psztestframe_cu psz/src/pipeline/testframe.cc)
target_link_libraries(psztestframe_cu PUBLIC pszcomp_cu pszmem pszutils_seq)

add_library(cusz psz/src/cusz_lib.cc)
target_link_libraries(cusz PUBLIC pszcomp_cu pszhf_cu pszspv_cu pszstat_seq
  pszutils_seq pszmem)

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

install(TARGETS pszkernel_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszkernel_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszmodule2401_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszanalysis2402 EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszstat_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszstat_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszmem EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszutils_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psztime EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszspv_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszhfbook_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS phfmodule2403_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszhf_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszcomp_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psztestframe_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz-bin EXPORT CUSZTargets)

if(PSZ_RESEARCH_HUFFBK_CUDA)
  install(TARGETS pszhfbook_cu EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(PSZ_RESEARCH_HUFFBK_CUDA)

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

## back compat
install(DIRECTORY psz/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz/)

## new testing
install(DIRECTORY psz/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/psz)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/psz/include/cusz_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/psz/)
