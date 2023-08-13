# utils for test
add_library(psz_testutils src/rand.cc src/rand_g.cc)
target_link_libraries(psz_testutils CUDA::cudart CUDA::curand)

# TODO include test_l23_utils.hh

# # testing the utils add_executable(utils src/utils.cc)
# target_link_directories(utils PRIVATE psz_testutils pszstat_ser pszstat_cu
# pszutils_cu CUDA::cudart) add_test(test_utils utils)

# add_executable(pred_hl src/spv.cu) target_link_libraries(pred_hl PRIVATE
# pszspv_cu psz_testutils) add_test(test_pred_hl pred_hl)

# # testing sp vector
add_executable(l3_spv src/test_l3_spv.cu)
target_link_libraries(l3_spv PRIVATE pszspv_cu psz_testutils)
add_test(test_l3_spv l3_spv)

# add_executable(l4_spv_obj src/test_l4_spv_obj.cu)
# target_link_libraries(l4_spv_obj PRIVATE pszspv_cu psz_testutils)
# add_test(test_l4_spv_obj l4_spv_obj)

# # testing hf add_executable(hf_hl src/spv.cu) target_link_libraries(hf_hl
# PRIVATE pszspv_cu psz_testutils) add_test(test_hf_hl hf_hl)

# add_executable(tcpu src/tcpu.c) target_link_libraries(tcpu PRIVATE psztime)
# add_test(test_tcpu tcpu)

# add_executable(tgpu src/tgpu.cu) target_link_libraries(tgpu PRIVATE psztime)
# add_test(test_tgpu tgpu)

add_library(psztestcompile_settings INTERFACE)
target_include_directories(
  psztestcompile_settings
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src/>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>)

# correctness, include kernel `.inl` directly ### test_typing test core
# functionality Level-0 basic typing
add_executable(l0_typing src/test_pncodec_func.cc)
target_link_libraries(l0_typing PRIVATE psztestcompile_settings)
add_test(test_l0_typing l0_typing)

# Level-1 subroutine
add_executable(l1_scan src/test_l1_l23scan.cu)
target_link_libraries(l1_scan PRIVATE pszcompile_settings
                                      psztestcompile_settings)
add_test(test_l1_scan l1_scan)

add_executable(l1_compact src/test_l1_compact.cu)
target_link_libraries(l1_compact PRIVATE pszcompile_settings
                                         psztestcompile_settings psz_testutils)
add_test(test_l1_compact l1_compact)

# Level-2 kernel (template; unit tests)
add_executable(l2_serial src/test_l2_serial.cc)
target_link_libraries(l2_serial PRIVATE psztestcompile_settings)
add_test(test_l2_serial l2_serial)

add_executable(l2_cudaproto src/test_l2_cudaproto.cu)
target_link_libraries(
  l2_cudaproto PRIVATE pszcompile_settings psztestcompile_settings pszmem
                       pszstat_cu)
add_test(test_l2_cudaproto l2_cudaproto)

# TODO cross-validation add_test(test_l1_cuda  l1_cuda) 2023 May fix
add_executable(l2_histsp src2/test_l2_histsp.cu)
target_link_libraries(l2_histsp PRIVATE pszcompile_settings pszmem pszstat_cu pszkernel_cu
                                        pszstat_ser)
add_test(test_l2_histsp l2_histsp)
# 2023 May fix end

# TODO cross-validation add_test(test_l1_cuda  l1_cuda)

# Level-3 kernel with configuration (low-level API)
add_executable(l3_cuda_pred src/test_l3_cuda_pred.cc)
target_link_libraries(
  l3_cuda_pred
  PRIVATE pszkernel_cu
          psz_testutils
          pszstat_ser
          pszstat_cu
          pszmem
          pszutils_cu
          CUDA::cudart)
add_test(test_l3_cuda_pred l3_cuda_pred)

add_executable(l3_lorenzosp src/test_l3_lorenzosp.cu)
target_link_libraries(
  l3_lorenzosp
  PRIVATE psztestcompile_settings
          pszkernel_cu
          psz_testutils
          pszspv_cu
          pszstat_ser
          pszstat_cu
          pszmem
          pszutils_cu
          # CUDA::cudart
)
add_test(test_l3_lorenzosp l3_lorenzosp)
