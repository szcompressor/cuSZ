add_compile_definitions(PSZ_USE_1API)

# DPCPP 2023.2.0:
# IntelDPCPP Config will be deprecated soon.  Use IntelSYCL config instead!
find_package(IntelSYCL REQUIRED)

if(IntelSYCL_FOUND)
  # 1. SYSL instead of IntelSYCL per definition in .cmake in PATH
  # 2. ${SYCL_INCLUDE_DIR} and subdir ${SYCL_INCLUDE_DIR}/sycl are appened
  # for finding <sycl/sycl.hpp> and <CL/..> properly

  # [psz::note] SYCL is not treated as a supported language.
  # With only SYCL_FLAGS, static check won't pass.
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Do something for Debug build type
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsycl -g -Rno-debug-disables-optimization")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fsycl -g -O2")
  endif()

  set(COMPILE_FLAGS "-fsycl")
  set(LINK_FLAGS "-fsycl")

  message("[psz::info] IntelSYCL FOUND")
  message("[psz::info] $\{SYCL_LANGUAGE_VERSION\}: " ${SYCL_LANGUAGE_VERSION})
  message("[psz::info] $\{SYCL_INCLUDE_DIR\}: " ${SYCL_INCLUDE_DIR})
  message("[psz::info] $\{SYCL_IMPLEMENTATION_ID\}: " ${SYCL_IMPLEMENTATION_ID})
  message("[psz::info] $\{SYCL_FLAGS\}: " ${SYCL_FLAGS})
  include_directories(${SYCL_INCLUDE_DIR})
  include_directories(${SYCL_INCLUDE_DIR}/sycl)
  message("[psz::info] $\{SYCL_INCLUDE_DIR\} (again): " ${SYCL_INCLUDE_DIR})
endif()

include(GNUInstallDirs)
include(CTest)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cusz_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cusz_version.h)

add_library(pszcompile_settings INTERFACE)

target_compile_definitions(
  pszcompile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:__STRICT_ANSI__>)
target_compile_options(
  pszcompile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fsycl -fsycl-unnamed-lambda>)
target_compile_features(pszcompile_settings INTERFACE cxx_std_17)
target_include_directories(
  pszcompile_settings
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cusz>)

add_library(pszstat_seq src/stat/compare.stl.cc)
target_link_libraries(pszstat_seq PUBLIC pszcompile_settings)

add_library(
  pszstat_dp src/stat/extrema.dp.cpp src/stat/cmpg2.dp.cpp src/stat/cmpg4_1.dp.cpp
  src/stat/cmpg4_2.dp.cpp src/stat/cmpg5_1.dp.cpp src/stat/cmpg5_2.dp.cpp)
target_link_libraries(pszstat_dp PUBLIC pszcompile_settings)

# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(pszkernel_seq src/kernel/l23.seq.cc src/kernel/hist.seq.cc
  src/kernel/histsp.seq.cc src/kernel/spvn.seq.cc)
target_link_libraries(pszkernel_seq PUBLIC pszcompile_settings)

# add_executable(
# port_dummy
# src/kernel/port_dummy.dp.cpp
# )
# target_link_libraries(port_dummy PUBLIC pszcompile_settings)
add_library(pszkernel_dp
  src/kernel/spvn.dp.cpp
  src/kernel/l23_c.dp.cpp
  src/kernel/l23_x.dp.cpp
  src/kernel/l23r.dp.cpp

  # # src/kernel/hist.dp.cpp  ## no proper interop
  src/kernel/histsp.dp.cpp
  src/kernel/dryrun.dp.cpp
)
target_link_libraries(pszkernel_dp PUBLIC pszcompile_settings)

add_library(pszmem src/mem/memseg.cc src/mem/memseg_dp.cc)
target_link_libraries(pszmem PUBLIC pszcompile_settings)

add_library(pszutils_seq src/utils/vis_stat.cc src/context.cc)
target_link_libraries(pszutils_seq PUBLIC pszcompile_settings)

add_library(pszspv_dp src/kernel/spv.dp.cpp)
target_link_libraries(pszspv_dp PUBLIC pszcompile_settings)

add_library(
  pszhfbook_seq src/hf/hfbk_impl1.seq.cc src/hf/hfbk_impl2.seq.cc
  src/hf/hfbk_internal.seq.cc src/hf/hfbk.seq.cc src/hf/hfcanon.seq.cc)
target_link_libraries(pszhfbook_seq PUBLIC pszcompile_settings)

add_library(pszhf_dp src/hf/hfclass.dp.cpp src/hf/hfcodec.dp.cpp)
target_link_libraries(pszhf_dp PUBLIC pszcompile_settings pszstat_dp
  pszhfbook_seq)

add_library(pszcomp_dp src/compressor.cc)
target_link_libraries(pszcomp_dp PUBLIC pszcompile_settings pszkernel_dp
  pszstat_dp pszhf_dp pszkernel_seq)

add_library(psztestframe_dp src/pipeline/testframe.cc)
target_link_libraries(psztestframe_dp PUBLIC pszcomp_dp pszmem pszutils_seq)

add_library(dpsz src/cusz_lib.cc)
target_link_libraries(dpsz PUBLIC pszcomp_dp pszhf_dp pszspv_dp pszstat_seq
  pszutils_seq pszmem)

add_executable(dpsz-bin src/cli_psz.cc)
target_link_libraries(dpsz-bin PRIVATE dpsz)
set_target_properties(dpsz-bin PROPERTIES OUTPUT_NAME dpsz)

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

install(TARGETS pszkernel_dp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszstat_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS pszstat_dp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszmem EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszutils_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS pszspv_dp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszhfbook_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS pszhf_dp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszcomp_dp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psztestframe_dp EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS dpsz EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS dpsz-bin EXPORT CUSZTargets)
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
