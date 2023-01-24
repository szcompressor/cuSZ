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


add_library(parszcompile_settings INTERFACE)
if(CUB_FOUND)
	target_link_libraries(parszcompile_settings INTERFACE CUB::CUB)
else()
	message(WARNING "cub not found via cmake find_package, trying to find the header ${CUDAToolkit_INCLUDE_DIRS}/cub")
 set(CMAKE_FIND_DEBUG_MODE TRUE)
	find_file(CUB_HEADER cub.cuh PATHS "${CUDAToolkit_INCLUDE_DIRS}/cub" REQUIRED)
	add_library(CUB INTERFACE)
	get_filename_component(CUB_INCLUDE_DIR ${CUB_HEADER} DIRECTORY )
	target_include_directories(CUB INTERFACE ${CUB_INCLUDE_DIR})
	target_link_libraries(parszcompile_settings INTERFACE CUB)
	install(TARGETS CUB EXPORT CUSZTargets)
endif()
target_compile_definitions(parszcompile_settings INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(parszcompile_settings INTERFACE
  $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations>
  )
target_compile_features(parszcompile_settings INTERFACE cxx_std_14 cuda_std_14)
target_include_directories(parszcompile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>
  )

## seprate later
add_library(parsztimer  src/utils/timer_cpu.cc src/utils/timer_gpu.cu)
target_link_libraries(parsztimer PUBLIC parszcompile_settings)

add_library(parszkelo  src/kernel/lorenzo.cu src/kernel/lorenzo_var.cu src/kernel/lorenzo_proto.cu)
target_link_libraries(parszkelo PUBLIC parszcompile_settings)

add_library(parszstat  src/stat/compare_cpu.cc)
target_link_libraries(parszstat PUBLIC parszcompile_settings)

add_library(parszstat_g  
  src/stat/cmpg1_1.cu src/stat/cmpg1_2.cu src/stat/cmpg1_3.cu src/stat/cmpg1_4.cu src/stat/cmpg1_5.cu 
  src/stat/cmpg2.cu src/stat/cmpg3.cu 
  src/stat/cmpg4_1.cu src/stat/cmpg4_2.cu src/stat/cmpg4_3.cu src/stat/cmpg4_4.cu 
  src/stat/stat_g.cu)
target_link_libraries(parszstat_g PUBLIC parszcompile_settings)

add_library(parszkernel  src/kernel/claunch_cuda.cu)
target_link_libraries(parszkernel PUBLIC parszcompile_settings parsztimer)

add_library(parszutils_g  src/utils/print_gpu.cu)
target_link_libraries(parszutils_g PUBLIC parszcompile_settings)

add_library(parszargp  src/context.cc)
target_link_libraries(parszargp PUBLIC parszcompile_settings)

add_library(parszpq  src/component/prediction.cc src/detail/prediction_impl.cu)
target_link_libraries(parszpq PUBLIC parszcompile_settings parszkernel parszkelo)

add_library(parszspv  src/kernel/spv_gpu.cu src/component/spcodec_vec.cc src/detail/spvec.cu)
target_link_libraries(parszspv PUBLIC parszcompile_settings)

# add_library(parszspm  src/component/spcodec.cc src/detail/spmat.cu)
# target_link_libraries(parszspm PUBLIC parszcompile_settings CUDA::cusparse)

add_library(parszhfbook_g   src/hf/hf_bookg.cu)
target_link_libraries(parszhfbook_g PUBLIC parszcompile_settings CUDA::cuda_driver)
set_target_properties(parszhfbook_g PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(parszhf_g  src/hf/hf.cc src/hf/hf_pimpl.cu src/hf/hf_codecg.cu)
target_link_libraries(parszhf_g PUBLIC parszcompile_settings parszstat_g parszhfbook_g)
set_target_properties(parszhfbook_g PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(parszcomp  src/cusz/cc2c.cc src/cusz/custom.cc src/compressor.cc src/detail/compressor_impl.cu)
target_link_libraries(parszcomp PUBLIC parszcompile_settings parszstat_g parszhf_g)

add_library(cusz  src/comp.cc src/cuszapi.cc)
target_link_libraries(cusz PUBLIC parszcomp parszargp parszhf_g parszspv parszpq parszstat parszutils_g)

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

install(TARGETS parszcompile_settings EXPORT CUSZTargets)
install(TARGETS parszkernel EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszkelo EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszstat EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszstat_g EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszutils_g EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parsztimer EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszpq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# install(TARGETS parszspm EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszspv EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszargp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszhfbook_g EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszhf_g EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parszcomp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
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
