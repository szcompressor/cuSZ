# utils for test
add_library(psztest_utils_hip src/utils/rand.seq.cc src/utils/rand.cu_hip.cc)
target_link_libraries(psztest_utils_hip hip::host hip::device ${hiprand_LIBRARIES})

# testing sp vector
add_executable(spv_hip src/test_spv.hip)
target_link_libraries(spv_hip PRIVATE pszspv_hip psztest_utils_hip)
add_test(test_spv_hip spv_hip)

add_library(psztestcompile_settings INTERFACE)
target_include_directories(
  psztestcompile_settings
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>)

# correctness, include kernel `.inl` directly ### test_typing test core
# functionality Level-0 basic typing
add_executable(zigzag src/test_zigzag_coding.cc)
target_link_libraries(zigzag PRIVATE psztestcompile_settings)
add_test(test_zigzag zigzag)

# Level-1 subroutine
add_executable(l1_scan src/test_l1_l23scan_hip.cpp)
target_link_libraries(l1_scan PRIVATE pszcompile_settings
  psztestcompile_settings)
add_test(test_l1_scan l1_scan)

add_executable(l1_compact src/test_l1_compact_hip.cpp)
target_link_libraries(l1_compact PRIVATE pszcompile_settings
  psztestcompile_settings psztest_utils_hip)
add_test(test_l1_compact l1_compact)

# Level-2 kernel (template; unit tests)
add_executable(l2_cudaproto src/test_l2_cudaproto_hip.cpp)
target_link_libraries(
  l2_cudaproto PRIVATE pszcompile_settings psztestcompile_settings pszmem
  pszstat_hip)
add_test(test_l2_cudaproto l2_cudaproto)

add_executable(histsp_hip src/test_histsp.hip)
target_link_libraries(l2_histsp PRIVATE pszcompile_settings pszmem pszstat_hip
  pszkernel_hip pszkernel_seq pszstat_seq)
add_test(test_histsp histsp_hip)

# Level-3 kernel with configuration (low-level API)
add_executable(l3_cuda_pred src/test_l3_cuda_pred.cc)
target_link_libraries(
  l3_cuda_pred PRIVATE pszkernel_hip psztest_utils_hip pszstat_seq pszstat_hip
  pszmem hip::host)
add_test(test_l3_cuda_pred l3_cuda_pred)

add_executable(lrz_seq src/test_lrz.seq.cc)
target_link_libraries(lrz_seq PRIVATE psztestcompile_dp)
add_test(test_lrz_seq lrz_seq)

add_executable(lrzsp_hip src/test_lrzsp.hip)
target_link_libraries(
  lrzsp_hip PRIVATE psztestcompile_settings
  pszkernel_hip
  psztest_utils_hip
  pszspv_hip
  pszstat_seq
  pszstat_hip
  pszmem)
add_test(test_lrzsp_hip lrzsp_hip)

add_executable(statfn src/test_statfn.cc)
target_link_libraries(statfn PRIVATE psztestcompile_settings psztest_utils_hip
  pszstat_hip pszstat_seq pszmem)
