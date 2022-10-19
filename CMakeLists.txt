cmake_minimum_required(VERSION 3.18)
project(CUSZ LANGUAGES CUDA CXX VERSION 0.3.0)

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


add_library(cuszcompile_settings INTERFACE)
if(CUB_FOUND)
	target_link_libraries(cuszcompile_settings INTERFACE CUB::CUB)
else()
	message(WARNING "cub not found via cmake find_package, trying to find the header ${CUDAToolkit_INCLUDE_DIRS}/cub")
 set(CMAKE_FIND_DEBUG_MODE TRUE)
	find_file(CUB_HEADER cub.cuh PATHS "${CUDAToolkit_INCLUDE_DIRS}/cub" REQUIRED)
	add_library(CUB INTERFACE)
	get_filename_component(CUB_INCLUDE_DIR ${CUB_HEADER} DIRECTORY )
	target_include_directories(CUB INTERFACE ${CUB_INCLUDE_DIR})
	target_link_libraries(cuszcompile_settings INTERFACE CUB)
	install(TARGETS CUB EXPORT CUSZTargets)
endif()
target_compile_definitions(cuszcompile_settings INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(cuszcompile_settings INTERFACE
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations>
  )
target_compile_features(cuszcompile_settings INTERFACE cxx_std_14 cuda_std_14)
target_include_directories(cuszcompile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>
  )

add_library(cuszkernel  src/kernel/claunch_cuda.cu src/kernel/claunch_cuda_proto.cu)
target_link_libraries(cuszkernel PUBLIC cuszcompile_settings)

add_library(cuszargp  src/context.cc)
target_link_libraries(cuszargp PUBLIC cuszcompile_settings)

add_library(cuszpq  src/component/prediction.cc src/detail/prediction_impl.cu)
target_link_libraries(cuszpq PUBLIC cuszcompile_settings cuszkernel)

add_library(cuszspv  src/component/spcodec_vec.cc src/detail/spvec.cu)
target_link_libraries(cuszspv PUBLIC cuszcompile_settings)

add_library(cuszspm  src/component/spcodec.cc src/detail/spmat.cu)
target_link_libraries(cuszspm PUBLIC cuszcompile_settings CUDA::cusparse)

add_library(cuszhuff  src/component/codec.cc src/kernel/huffman_parbook.cu src/detail/huffman_coarse.cu)
target_link_libraries(cuszhuff PUBLIC cuszcompile_settings CUDA::cuda_driver cuszkernel)
set_target_properties(cuszhuff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(cuszcomp  src/cusz/cc2c.cc src/cusz/custom.cc src/compressor.cc src/detail/compressor_impl.cu)
target_link_libraries(cuszcomp PUBLIC cuszcompile_settings)

add_library(cusz  src/comp.cc src/cuszapi.cc)
target_link_libraries(cusz PUBLIC cuszcomp cuszargp cuszhuff cuszspm cuszspv cuszpq)

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

install(TARGETS cuszcompile_settings EXPORT CUSZTargets)
install(TARGETS cuszkernel EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cuszpq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cuszspm EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cuszspv EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cuszargp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cuszhuff EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS cuszcomp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
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
