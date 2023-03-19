add_library(parsz_testutils  src/rand.cc src/rand_g.cc)
target_link_libraries(parsz_testutils CUDA::cudart CUDA::curand)


## testing the utils 
# add_executable(utils src/utils.cc)
# target_link_directories(utils PRIVATE parsz_testutils parszstat parszstat_g parszutils_g CUDA::cudart)
# add_test(test_utils utils)

## testing prediction
add_executable(pred_ll  src/pred_ll.cc)
target_link_libraries(pred_ll PRIVATE parsz_testutils parszkelo parszstat parszstat_g parszutils_g CUDA::cudart)
add_test(test_pred_ll pred_ll)

# add_executable(pred_hl src/spv.cu)
# target_link_libraries(pred_hl PRIVATE parszspv parsz_testutils)
# add_test(test_pred_hl pred_hl)

## testing sp vector
add_executable(spv_ll src/spv.cu)
target_link_libraries(spv_ll PRIVATE parszspv parsz_testutils)
add_test(test_spv_ll spv_ll)

add_executable(spv_hl src/spvo.cu)
target_link_libraries(spv_hl PRIVATE parszspv parsz_testutils)
add_test(test_spv_hl spv_hl)

## testing hf 
# add_executable(hf_hl src/spv.cu)
# target_link_libraries(hf_hl PRIVATE parszspv parsz_testutils)
# add_test(test_hf_hl hf_hl)


# add_executable(tcpu src/tcpu.c)
# target_link_libraries(tcpu PRIVATE parsztimer)
# add_test(test_tcpu tcpu)

# add_executable(tgpu src/tgpu.cu)
# target_link_libraries(tgpu PRIVATE parsztimer)
# add_test(test_tgpu tgpu)


# 2023 March fix start
add_library(psztestcompile_settings INTERFACE)
target_include_directories(psztestcompile_settings INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
  )

# correctness, include kernel `.inl` directly
add_executable(core_serial_using_templates  src/test_core_serial.cc)
target_link_libraries(core_serial_using_templates PRIVATE psztestcompile_settings)
add_test(test_core_serial core_serial_using_templates)

add_executable(core_cudaproto_using_templates  src/test_core_cudaproto.cu)
target_link_libraries(core_cudaproto_using_templates PRIVATE parszcompile_settings psztestcompile_settings)
add_test(test_core_cudaproto core_cudaproto_using_templates)
# 2023 March fix end
