cmake_minimum_required(VERSION 3.18)
project(CUSZ LANGUAGES CUDA CXX VERSION 0.3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

find_package(CUDAToolkit)
include(GNUInstallDirs)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cusz_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cusz_version.h
  )


add_library(compile_settings INTERFACE)
target_compile_options(compile_settings INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations>
  )
target_compile_features(compile_settings INTERFACE cxx_std_14 cuda_std_14)
target_include_directories(compile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )


add_library(pq  src/component/extrap_lorenzo.cu src/component/interp_spline3.cu)
target_link_libraries(pq PUBLIC compile_settings)

add_library(argp  src/context.cc)
target_link_libraries(argp PUBLIC compile_settings)

add_library(sp  src/component/csr11.cu)
target_link_libraries(sp PUBLIC compile_settings CUDA::cusparse)

add_library(huff  src/component/huffman_parbook.cu src/component/huffman_coarse.cu)
target_link_libraries(huff PUBLIC compile_settings CUDA::cusparse CUDA::cuda_driver)
set_target_properties(huff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(compress  src/compressor.cu)
target_link_libraries(compress PUBLIC compile_settings)

add_library(cusz  src/cli.cu)
target_link_libraries(cusz PUBLIC compress argp huff sp pq)

add_executable(cusz-bin src/cusz-cli.cu)
target_link_libraries(cusz-bin PRIVATE cusz )
set_target_properties(cusz-bin PROPERTIES OUTPUT_NAME cusz)

option(CUSZ_BUILD_EXAMPLES "build example codes" OFF)
if(CUSZ_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

install(TARGETS compile_settings EXPORT CUSZTargets)
install(TARGETS pq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/cusz )
install(TARGETS sp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/cusz)
install(TARGETS argp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/cusz)
install(TARGETS huff EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/cusz)
install(TARGETS compress EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/cusz)
install(TARGETS cusz EXPORT CUSZTargets)
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
