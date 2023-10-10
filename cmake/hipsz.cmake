add_compile_definitions(PSZ_USE_HIP)

find_package(hip REQUIRED)

find_package(rocthrust REQUIRED)
if(rocthrust_FOUND)
  message("[psz::info] rocthrust FOUND")
  message("[psz::info] $\{rocthrust_INCLUDE_DIRS\}: " ${rocthrust_INCLUDE_DIRS})
  message("[psz::info] $\{rocthrust_LIBRARIES\}: " ${rocthrust_LIBRARIES})
  include_directories(${rocthrust_INCLUDE_DIRS})
endif()

find_package(rocprim REQUIRED)
if(rocprim_FOUND)
  message("[psz::info] rocprim FOUND")
  message("[psz::info] $\{rocprim_INCLUDE_DIRS\}: " ${rocprim_INCLUDE_DIRS})
  message("[psz::info] $\{rocprim_LIBRARIES\}: " ${rocprim_LIBRARIES})
  include_directories(${rocprim_INCLUDE_DIRS})
endif()

find_package(hiprand REQUIRED)
if(hiprand_FOUND)
  message("[psz::info] hiprand FOUND")
  message("[psz::info] $\{hiprand_INCLUDE_DIRS\}: " ${hiprand_INCLUDE_DIRS})
  message("[psz::info] $\{hiprand_LIBRARIES\}: " ${hiprand_LIBRARIES})
  include_directories(${hiprand_INCLUDE_DIRS})
endif()

find_package(rocrand REQUIRED)
if(hiprand_FOUND)
  message("[psz::info] rocrand FOUND")
  message("[psz::info] $\{rocrand_INCLUDE_DIRS\}: " ${rocrand_INCLUDE_DIRS})
  message("[psz::info] $\{rocrand_LIBRARIES\}: " ${rocrand_LIBRARIES})
  include_directories(${rocrand_INCLUDE_DIRS})
endif()

include(GNUInstallDirs)
include(CTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/cusz_version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/cusz_version.h)

add_library(pszcompile_settings INTERFACE)

target_compile_definitions(
  pszcompile_settings
  INTERFACE $<$<COMPILE_LANG_AND_ID:HIP,Clang>:__STRICT_ANSI__> -D__HIP_PLATFORM_AMD__)
target_compile_features(pszcompile_settings INTERFACE cxx_std_14) # [TODO] should be 17
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
  pszstat_hip src/stat/extrema.hip src/stat/cmpg2.hip src/stat/cmpg4_1.hip
              src/stat/cmpg4_2.hip src/stat/cmpg5_1.hip src/stat/cmpg5_2.hip)
target_link_libraries(pszstat_hip PUBLIC pszcompile_settings roc::rocthrust)

# FUNC={core,api}, BACKEND={serial,cuda,...}
add_library(pszkernel_seq src/kernel/l23.seq.cc src/kernel/hist.seq.cc
                          src/kernel/histsp.seq.cc src/kernel/spvn.seq.cc)
target_link_libraries(pszkernel_seq PUBLIC pszcompile_settings)

add_library(pszkernel_hip src/kernel/spvn.hip src/kernel/l23.hip src/kernel/l23r.hip
                          src/kernel/hist.hip src/kernel/histsp.hip
                          src/kernel/dryrun.hip)
target_link_libraries(pszkernel_hip PUBLIC pszcompile_settings)

add_library(pszmem src/mem/memseg.cc src/mem/memseg_hip.cc)
target_link_libraries(pszmem PUBLIC pszcompile_settings hip::host)

add_library(pszutils_seq src/utils/vis_stat.cc src/context.cc)
target_link_libraries(pszutils_seq PUBLIC pszcompile_settings)

add_library(pszspv_hip src/kernel/spv.hip)
target_link_libraries(pszspv_hip PUBLIC pszcompile_settings ${rocthrust_LIBRARIES})

add_library(
  pszhfbook_seq src/hf/hfbk_impl1.seq.cc src/hf/hfbk_impl2.seq.cc
                src/hf/hfbk_internal.seq.cc src/hf/hfbk.seq.cc src/hf/hfcanon.seq.cc)
target_link_libraries(pszhfbook_seq PUBLIC pszcompile_settings)

add_library(pszhf_hip src/hf/hfclass.hip src/hf/hf_codec.hip)
target_link_libraries(pszhf_hip PUBLIC pszcompile_settings pszstat_hip
                                       pszhfbook_seq hip::device)

add_library(pszcomp_hip src/compressor.cc)
target_link_libraries(pszcomp_hip PUBLIC pszcompile_settings pszkernel_hip
                                  pszstat_hip pszhf_hip hip::host)

add_library(psztestframe_hip src/pipeline/testframe.cc)
target_link_libraries(psztestframe_hip PUBLIC pszcomp_hip pszmem pszutils_seq)

add_library(hipsz src/cusz_lib.cc)
target_link_libraries(hipsz PUBLIC pszcomp_hip pszhf_hip pszspv_hip pszstat_seq
                                   pszutils_seq pszmem)

add_executable(hipsz-bin src/cli_psz.cc)
target_link_libraries(hipsz-bin PRIVATE hipsz)
set_target_properties(hipsz-bin PROPERTIES OUTPUT_NAME hipsz)

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
install(TARGETS pszkernel_hip EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszstat_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszstat_hip EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszmem EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszutils_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszspv_hip EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszhfbook_seq EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszhf_hip EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS pszcomp_hip EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS psztestframe_hip EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS hipsz EXPORT CUSZTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS hipsz-bin EXPORT CUSZTargets)

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
