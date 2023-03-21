# utils for test
add_library(          psz_testutils  src/rand.cc src/rand_g.cc)
target_link_libraries(psz_testutils  CUDA::cudart CUDA::curand)
# TODO include test_l23_utils.hh

## testing the utils 
# add_executable(utils src/utils.cc)
# target_link_directories(utils PRIVATE psz_testutils pszstat pszstat_g pszutils_g CUDA::cudart)
# add_test(test_utils utils)

# add_executable(pred_hl src/spv.cu)
# target_link_libraries(pred_hl PRIVATE pszspv psz_testutils)
# add_test(test_pred_hl pred_hl)

## testing sp vector
add_executable(spv_ll src/spv.cu)
target_link_libraries(spv_ll PRIVATE pszspv psz_testutils)
add_test(test_spv_ll spv_ll)

add_executable(spv_hl src/spvo.cu)
target_link_libraries(spv_hl PRIVATE pszspv psz_testutils)
add_test(test_spv_hl spv_hl)

## testing hf 
# add_executable(hf_hl src/spv.cu)
# target_link_libraries(hf_hl PRIVATE pszspv psz_testutils)
# add_test(test_hf_hl hf_hl)

# add_executable(tcpu src/tcpu.c)
# target_link_libraries(tcpu PRIVATE psztimer)
# add_test(test_tcpu tcpu)

# add_executable(tgpu src/tgpu.cu)
# target_link_libraries(tgpu PRIVATE psztimer)
# add_test(test_tgpu tgpu)


# 2023 March fix start
add_library(psztestcompile_settings INTERFACE)
target_include_directories(psztestcompile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
  )

# correctness, include kernel `.inl` directly
#### test_typing
# test core functionality
# Level-0 basic typing
add_executable(          l0_typing  src/test_pncodec_func.cc)
target_link_libraries(   l0_typing  PRIVATE psztestcompile_settings)
add_test(test_l0_typing  l0_typing)

# Level-1 subroutine
add_executable(              l1_subroutine  src/test_l1_l23scan.cu)
target_link_libraries(       l1_subroutine  PRIVATE pszcompile_settings psztestcompile_settings)
add_test(test_l1_subroutine  l1_subroutine)

# Level-2 kernel (template; unit tests)
add_executable(          l2_serial  src/test_l2_serial.cc)
target_link_libraries(   l2_serial  PRIVATE psztestcompile_settings)
add_test(test_l1_serial  l2_serial)

add_executable(             l2_cudaproto  src/test_l2_cudaproto.cu)
target_link_libraries(      l2_cudaproto  PRIVATE pszcompile_settings psztestcompile_settings)
add_test(test_l1_cudaproto  l2_cudaproto)

# TODO cross-validation
# add_test(test_l1_cuda  l1_cuda)

# Level-3 kernel with configuration (low-level API)
add_executable(             l3_cuda_pred  src/test_l3_cuda_pred.cc)
target_link_libraries(      l3_cuda_pred  PRIVATE psz_core_cuda psz_testutils pszstat pszstat_g pszutils_g CUDA::cudart)
add_test(test_l3_cuda_pred  l3_cuda_pred)
# 2023 March fix end
