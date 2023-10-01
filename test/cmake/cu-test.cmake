# utils for test
add_library(psztest_utils_cu src/utils/rand.seq.cc src/utils/rand.cu_hip.cc)
target_link_libraries(psztest_utils_cu CUDA::cudart CUDA::curand)

# testing sp vector
add_executable(spv_cu src/test_spv.cu)
target_link_libraries(spv_cu PRIVATE pszspv_cu psztest_utils_cu)
add_test(test_spv_cu spv_cu)

# testing timer wrapper

# add_executable(tcpu src/tcpu.c) target_link_libraries(tcpu PRIVATE psztime)
# add_test(test_tcpu tcpu)

# add_executable(tgpu src/tgpu.cu) target_link_libraries(tgpu PRIVATE psztime)
# add_test(test_tgpu tgpu)
add_library(psztestcompile_cu INTERFACE)
target_include_directories(
  psztestcompile_cu
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>)

# correctness, include kernel `.inl` directly ### test_typing test core
# functionality Level-0 basic typing
add_executable(l0_typing src/test_pncodec_func.cc)
target_link_libraries(l0_typing PRIVATE psztestcompile_cu)
add_test(test_l0_typing l0_typing)

# Level-1 subroutine
add_executable(l1_scan src/test_l1_l23scan.cu)
target_link_libraries(l1_scan PRIVATE pszcompile_settings
  psztestcompile_cu)
add_test(test_l1_scan l1_scan)

add_executable(l1_compact src/test_l1_compact.cu)
target_link_libraries(l1_compact PRIVATE pszcompile_settings
  psztestcompile_cu psztest_utils_cu)
add_test(test_l1_compact l1_compact)

# Level-2 kernel (template; unit tests)
add_executable(l2_serial src/test_l2_serial.cc)
target_link_libraries(l2_serial PRIVATE psztestcompile_cu)
add_test(test_l2_serial l2_serial)

add_executable(l2_cudaproto src/test_l2_cudaproto.cu)
target_link_libraries(
  l2_cudaproto PRIVATE pszcompile_settings psztestcompile_cu pszmem
  pszstat_cu)
add_test(test_l2_cudaproto l2_cudaproto)

add_executable(l2_histsp src/test_l2_histsp.cu)
target_link_libraries(l2_histsp PRIVATE pszcompile_settings pszmem pszstat_cu
  pszkernel_cu pszkernel_seq pszstat_seq)
add_test(test_l2_histsp l2_histsp)

# Level-3 kernel with configuration (low-level API)
add_executable(l3_cuda_pred src/test_l3_cuda_pred.cc)
target_link_libraries(
  l3_cuda_pred PRIVATE pszkernel_cu psztest_utils_cu pszstat_seq pszstat_cu pszmem
  CUDA::cudart)
add_test(test_l3_cuda_pred l3_cuda_pred)

add_executable(lrzsp_cu src/test_lrzsp.cu)
target_link_libraries(
  lrzsp_cu
  PRIVATE psztestcompile_cu
  pszkernel_cu
  psztest_utils_cu
  pszspv_cu
  pszstat_seq
  pszstat_cu
  pszmem)
add_test(test_lrzsp_cu lrzsp_cu)

if(PSZ_REACTIVATE_THRUSTGPU)
  add_compile_definitions(REACTIVATE_THRUSTGPU)
  add_executable(statfn src/test_statfn.cc)
  target_link_libraries(statfn PRIVATE psztestcompile_cu psztest_utils_cu
    pszstat_cu pszstat_seq pszmem)
else()
  add_executable(statfn src/test_statfn.cc)
  target_link_libraries(statfn PRIVATE psztestcompile_cu psztest_utils_cu
    pszstat_cu pszstat_seq pszmem)
endif()
