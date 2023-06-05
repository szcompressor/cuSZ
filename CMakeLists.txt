cmake_minimum_required(VERSION 3.18)
project(
  CUSZ
  LANGUAGES CUDA CXX C
  VERSION 0.3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

find_package(CUDAToolkit REQUIRED)
find_package(CUB)

if(TARGET _CUB_CUB)
  install(TARGETS _CUB_CUB EXPORT CUSZTargets)

  if(TARGET _CUB_libcudacxx)
    install(TARGETS _CUB_libcudacxx EXPORT CUSZTargets)
  endif()
endif()

include(GNUInstallDirs)
include(CTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/cusz_version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/cusz_version.h)

add_library(pszcompile_settings INTERFACE)

if(CUB_FOUND)
  target_link_libraries(pszcompile_settings INTERFACE CUB::CUB)
else()
  message(
    WARNING
      "cub not found via cmake find_package, trying to find the header ${CUDAToolkit_INCLUDE_DIRS}/cub"
  )
  set(CMAKE_FIND_DEBUG_MODE TRUE)
  find_file(CUB_HEADER cub.cuh PATHS "${CUDAToolkit_INCLUDE_DIRS}/cub" REQUIRED)
  add_library(CUB INTERFACE)
  get_filename_component(CUB_INCLUDE_DIR ${CUB_HEADER} DIRECTORY)
  target_include_directories(CUB INTERFACE ${CUB_INCLUDE_DIR})
  target_link_libraries(pszcompile_settings INTERFACE CUB)
  install(TARGETS CUB EXPORT CUSZTargets)
endif()

target_compile_definitions(
  pszcompile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(
  pszcompile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda
            --expt-relaxed-constexpr -Wno-deprecated-declarations>)
target_compile_features(pszcompile_settings INTERFACE cxx_std_14 cuda_std_14)
target_include_directories(
  pszcompile_settings
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>)

# seprate later
add_library(psz_timer src/utils/timer_cpu.cc src/utils/timer_gpu.cu)
target_link_libraries(psz_timer PUBLIC pszcompile_settings)

# deprecated
add_library(psz_kernel src/kernel/lorenzo.cu src/kernel/lorenzo_var.cu
                       src/kernel/lorenzo_proto.cu)
target_link_libraries(psz_kernel PUBLIC pszcompile_settings)

add_library(psz_stat src/stat/compare_cpu.cc)
target_link_libraries(psz_stat PUBLIC pszcompile_settings)

add_library(
  psz_statg
  src/stat/cmpg1_1.cu
  src/stat/cmpg1_2.cu
  src/stat/cmpg1_3.cu
  src/stat/cmpg1_4.cu
  src/stat/cmpg1_5.cu
  src/stat/cmpg2_1.cu
  src/stat/cmpg2_2.cu
  src/stat/cmpg3.cu
  src/stat/cmpg4_1.cu
  src/stat/cmpg4_2.cu
  src/stat/cmpg5_1.cu
  src/stat/cmpg5_2.cu
  src/stat/stat_g.cu)
target_link_libraries(psz_statg PUBLIC pszcompile_settings)

# 2023 March-April fix start naming: PROJ_FUNC_BACKEND, where PROJ=psz,
# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(psz_core_serial src/kernel/lorenzo_serial.cc)
target_link_libraries(psz_core_serial PUBLIC pszcompile_settings)

add_library(
  psz_core_cuda
  src/kernel/lorenzo_proto.cu src/kernel/lorenzo.cu src/kernel2/l23r.cu
  src/kernel2/l23r_scttr.cu src/kernel2/hist_sp.cu)
target_link_libraries(psz_core_cuda PUBLIC pszcompile_settings)

# 2023 March-April fix end
add_library(psz_utilsg src/utils/print_gpu.cu)
target_link_libraries(psz_utilsg PUBLIC pszcompile_settings)

add_library(psz_argp src/context.cc)
target_link_libraries(psz_argp PUBLIC pszcompile_settings)

add_library(psz_spv src/kernel/spv_gpu.cu)
target_link_libraries(psz_spv PUBLIC pszcompile_settings)

add_library(psz_hfbookg src/hf/hf_bookg.cu)
target_link_libraries(psz_hfbookg PUBLIC pszcompile_settings CUDA::cuda_driver)
set_target_properties(psz_hfbookg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
string(FIND "${CUDA_cuda_driver_LIBRARY}" "stub" CUDA_DRIVER_IS_STUB)

if(NOT ${CUDA_DRIVER_IS_STUB} EQUAL -1)
  message(
    WARNING
      "the cuda driver is a stub!! adding --allow-shlib-undefined to fix downstream linking issues"
  )
  target_link_options(psz_hfbookg PUBLIC
                      $<HOST_LINK:LINKER:--allow-shlib-undefined>)
endif()

add_library(psz_hfg src/hf/hf.cu src/hf/hf_codecg.cu src/hf/hf2.cu)
target_link_libraries(psz_hfg PUBLIC pszcompile_settings psz_statg psz_hfbookg)
set_target_properties(psz_hfbookg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(psz_comp src/custom.cc src/compressor.cu)
target_link_libraries(psz_comp PUBLIC pszcompile_settings psz_kernel psz_statg
                                      psz_hfg CUDA::cudart)

# 2023 June fix start
add_library(psz_comp2 src/psz_lib.cc src/rt_config.cu)
target_link_libraries(psz_comp2 PUBLIC pszcompile_settings CUDA::cudart)
# 2023 June fix end

add_library(cusz src/cusz_lib.cc)
target_link_libraries(cusz PUBLIC psz_comp psz_argp psz_hfg psz_spv psz_stat
                                  psz_utilsg)

add_executable(cusz-bin src/cli_prog.cu src/cli.cu src/dryrun.cu)
target_link_libraries(cusz-bin PRIVATE cusz)
set_target_properties(cusz-bin PROPERTIES OUTPUT_NAME cusz)

option(CUSZ_BUILD_EXAMPLES "build example codes" OFF)

if(CUSZ_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

install(TARGETS pszcompile_settings EXPORT CUSZTargets)

# 2023 March-April fix start
install(TARGETS psz_core_serial EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_core_cuda EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# 2023 March-April fix end
install(TARGETS psz_kernel EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_stat EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_statg EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_utilsg EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_timer EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_spv EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_argp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_hfbookg EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_hfg EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_comp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
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

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cusz_version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz/)
