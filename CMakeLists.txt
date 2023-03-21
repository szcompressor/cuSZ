cmake_minimum_required(VERSION 3.18)
project(CUSZ LANGUAGES CUDA CXX C VERSION 0.3.0)

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

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cusz_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cusz_version.h
  )


add_library(pszcompile_settings INTERFACE)
if(CUB_FOUND)
	target_link_libraries(pszcompile_settings INTERFACE CUB::CUB)
else()
	message(WARNING "cub not found via cmake find_package, trying to find the header ${CUDAToolkit_INCLUDE_DIRS}/cub")
 set(CMAKE_FIND_DEBUG_MODE TRUE)
	find_file(CUB_HEADER cub.cuh PATHS "${CUDAToolkit_INCLUDE_DIRS}/cub" REQUIRED)
	add_library(CUB INTERFACE)
	get_filename_component(CUB_INCLUDE_DIR ${CUB_HEADER} DIRECTORY )
	target_include_directories(CUB INTERFACE ${CUB_INCLUDE_DIR})
	target_link_libraries(pszcompile_settings INTERFACE CUB)
	install(TARGETS CUB EXPORT CUSZTargets)
endif()
target_compile_definitions(pszcompile_settings INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(pszcompile_settings INTERFACE
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations>
  )
target_compile_features(pszcompile_settings INTERFACE cxx_std_14 cuda_std_14)
target_include_directories(pszcompile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>
  )

## seprate later
add_library(psztimer  src/utils/timer_cpu.cc src/utils/timer_gpu.cu)
target_link_libraries(psztimer PUBLIC pszcompile_settings)

# deprecated
add_library(pszkelo  src/kernel/lorenzo.cu src/kernel/lorenzo_var.cu src/kernel/lorenzo_proto.cu)
target_link_libraries(pszkelo PUBLIC pszcompile_settings)

add_library(pszstat  src/stat/compare_cpu.cc)
target_link_libraries(pszstat PUBLIC pszcompile_settings)

add_library(pszstat_g  
  src/stat/cmpg1_1.cu src/stat/cmpg1_2.cu src/stat/cmpg1_3.cu src/stat/cmpg1_4.cu src/stat/cmpg1_5.cu 
  src/stat/cmpg2.cu src/stat/cmpg3.cu 
  src/stat/cmpg4_1.cu src/stat/cmpg4_2.cu src/stat/cmpg4_3.cu src/stat/cmpg4_4.cu 
  src/stat/stat_g.cu)
target_link_libraries(pszstat_g PUBLIC pszcompile_settings)

# 2023 March fix start
# naming: PROJ_FUNC_BACKEND, where PROJ=psz, FUNC={core,api}, BACKEND={serial,cuda,...} 
add_library(psz_core_serial
  src/kernel/lorenzo_serial.cc
)
target_link_libraries(psz_core_serial PUBLIC pszcompile_settings)

add_library(psz_core_cuda
  src/kernel/lorenzo_proto.cu
  src/kernel/lorenzo.cu
  # TODO src/kernel/lorenzo_var.cu
)
target_link_libraries(psz_core_cuda PUBLIC pszcompile_settings)
# 2023 March fix end

add_library(pszkernel  src/kernel/claunch_cuda.cu)
target_link_libraries(pszkernel PUBLIC pszcompile_settings psztimer)

add_library(pszutils_g  src/utils/print_gpu.cu)
target_link_libraries(pszutils_g PUBLIC pszcompile_settings)

add_library(pszargp  src/context.cc)
target_link_libraries(pszargp PUBLIC pszcompile_settings)

# TODO reserve for speeding up compilation
# add_library(pszpq  src/detail/__to_delete_prediction_impl.cu)
# target_link_libraries(pszpq PUBLIC pszcompile_settings pszkernel pszkelo)

add_library(pszspv  src/kernel/spv_gpu.cu)
target_link_libraries(pszspv PUBLIC pszcompile_settings)

add_library(pszhfbook_g  src/hf/hf_bookg.cu)
target_link_libraries(pszhfbook_g PUBLIC pszcompile_settings CUDA::cuda_driver)
set_target_properties(pszhfbook_g PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
string(FIND "${CUDA_cuda_driver_LIBRARY}" "stub" CUDA_DRIVER_IS_STUB)
if(NOT ${CUDA_DRIVER_IS_STUB} EQUAL -1)
    message(WARNING "the cuda driver is a stub!! adding --allow-shlib-undefined to fix downstream linking issues")
    target_link_options(pszhfbook_g PUBLIC $<HOST_LINK:LINKER:--allow-shlib-undefined>)
endif()

add_library(pszhf_g  src/hf/hf.cc src/hf/hf_pimpl.cu src/hf/hf_codecg.cu)
target_link_libraries(pszhf_g PUBLIC pszcompile_settings pszstat_g pszhfbook_g)
set_target_properties(pszhfbook_g PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(pszcomp  src/cusz/custom.cc src/compressor.cc src/detail/compressor_impl.cu)
target_link_libraries(pszcomp PUBLIC pszcompile_settings pszkernel pszkelo pszstat_g pszhf_g CUDA::cudart)

add_library(pszcomp_v2  src/cusz/custom.cc src/pipeline/v2_compressor.cc src/pipeline/v2_compressor_impl.cu)
target_link_libraries(pszcomp_v2 PUBLIC pszcompile_settings pszkernel pszkelo pszstat_g pszhf_g CUDA::cudart)

add_library(cusz  src/cusz_lib.cc)
target_link_libraries(cusz PUBLIC pszcomp pszargp pszhf_g pszspv pszstat pszutils_g)

add_executable(cusz-bin  src/cli_bin.cu src/cli/cli.cu)
target_link_libraries(cusz-bin PRIVATE cusz)
set_target_properties(cusz-bin PROPERTIES OUTPUT_NAME cusz)

option(CUSZ_BUILD_EXAMPLES "build example codes" OFF)
if(CUSZ_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if (BUILD_TESTING)
  add_subdirectory(test)
endif()

install(TARGETS pszcompile_settings EXPORT CUSZTargets)
# 2023 March fix start
install(TARGETS psz_core_serial EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psz_core_cuda   EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# 2023 March fix end
install(TARGETS pszkernel EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszkelo EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszstat EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszstat_g EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszutils_g EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psztimer EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# install(TARGETS pszpq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszspv EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszargp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszhfbook_g EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszhf_g EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszcomp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszcomp_v2 EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cusz-bin EXPORT CUSZTargets)
install(EXPORT CUSZTargets NAMESPACE CUSZ:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CUSZ/)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/CUSZConfig.cmake.in
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

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cusz_version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cusz/)
