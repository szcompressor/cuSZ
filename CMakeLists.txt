cmake_minimum_required(VERSION 3.18)
project(CUSZ LANGUAGES CUDA CXX C VERSION 0.3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

find_package(CUDAToolkit REQUIRED)
find_package(CUB)
if(TARGET _CUB_CUB)
  install(TARGETS _CUB_CUB EXPORT CUSZTargets)
endif()

include(GNUInstallDirs)
include(CTest)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cusz_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cusz_version.h
  )


add_library(accszcompile_settings INTERFACE)
if(CUB_FOUND)
	target_link_libraries(accszcompile_settings INTERFACE CUB::CUB)
else()
	message(WARNING "cub not found via cmake find_package, trying to find the header ${CUDAToolkit_INCLUDE_DIRS}/cub")
 set(CMAKE_FIND_DEBUG_MODE TRUE)
	find_file(CUB_HEADER cub.cuh PATHS "${CUDAToolkit_INCLUDE_DIRS}/cub" REQUIRED)
	add_library(CUB INTERFACE)
	get_filename_component(CUB_INCLUDE_DIR ${CUB_HEADER} DIRECTORY )
	target_include_directories(CUB INTERFACE ${CUB_INCLUDE_DIR})
	target_link_libraries(accszcompile_settings INTERFACE CUB)
	install(TARGETS CUB EXPORT CUSZTargets)
endif()
target_compile_definitions(accszcompile_settings INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(accszcompile_settings INTERFACE
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations>
  )
target_compile_features(accszcompile_settings INTERFACE cxx_std_14 cuda_std_14)
target_include_directories(accszcompile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>
  )

## seprate later
add_library(asztimer  src/utils/timer_cpu.cc src/utils/timer_gpu.cu)
target_link_libraries(asztimer PUBLIC accszcompile_settings)

add_library(aszkelo  src/kernel/lorenzo.cu src/kernel/lorenzo_var.cu src/kernel/lorenzo_proto.cu)
target_link_libraries(aszkelo PUBLIC accszcompile_settings)

add_library(aszstat  src/stat/compare_cpu.cc)
target_link_libraries(aszstat PUBLIC accszcompile_settings)

add_library(aszstatg  src/stat/compare_gpu.cu src/stat/stat_g.cu)
target_link_libraries(aszstatg PUBLIC accszcompile_settings)

add_library(accszkernel  src/kernel/claunch_cuda.cu)
target_link_libraries(accszkernel PUBLIC accszcompile_settings asztimer)

add_library(aszutilsg  src/utils/print_gpu.cu)
target_link_libraries(aszutilsg PUBLIC accszcompile_settings)

add_library(accszargp  src/context.cc)
target_link_libraries(accszargp PUBLIC accszcompile_settings)

add_library(accszpq  src/component/prediction.cc src/detail/prediction_impl.cu)
target_link_libraries(accszpq PUBLIC accszcompile_settings accszkernel aszkelo)

add_library(accszspv  src/kernel/spv_gpu.cu src/component/spcodec_vec.cc src/detail/spvec.cu)
target_link_libraries(accszspv PUBLIC accszcompile_settings)

# add_library(accszspm  src/component/spcodec.cc src/detail/spmat.cu)
# target_link_libraries(accszspm PUBLIC accszcompile_settings CUDA::cusparse)

add_library(accszhuff  src/component/codec.cc src/kernel/huffman_parbook.cu src/detail/huffman_coarse.cu)
target_link_libraries(accszhuff PUBLIC accszcompile_settings CUDA::cuda_driver accszkernel aszstatg)
set_target_properties(accszhuff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(accszcomp  src/cusz/cc2c.cc src/cusz/custom.cc src/compressor.cc src/detail/compressor_impl.cu)
target_link_libraries(accszcomp PUBLIC accszcompile_settings aszstatg)

add_library(cusz  src/comp.cc src/cuszapi.cc)
target_link_libraries(cusz PUBLIC accszcomp accszargp accszhuff accszspv accszpq aszstat aszutilsg)

add_executable(cusz-bin  src/cli_bin.cu src/cli/cli.cu)
target_link_libraries(cusz-bin PRIVATE cusz)
set_target_properties(cusz-bin PROPERTIES OUTPUT_NAME cusz)

option(CUSZ_BUILD_EXAMPLES "build example codes" OFF)
if(CUSZ_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

option(CUSZ_BUILD_TESTS "build test codes" OFF)
if (CUSZ_BUILD_TESTS)
  add_subdirectory(test)
endif()

install(TARGETS accszcompile_settings EXPORT CUSZTargets)
install(TARGETS accszkernel EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS aszkelo EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS aszstat EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS aszstatg EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS aszutilsg EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS asztimer EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS accszpq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# install(TARGETS accszspm EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS accszspv EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS accszargp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS accszhuff EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS accszcomp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
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
