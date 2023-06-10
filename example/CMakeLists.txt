add_executable(demo_capi_nvcc src/demo_capi_nvcc.cu)
target_link_libraries(demo_capi_nvcc PRIVATE cusz)
set_target_properties(demo_capi_nvcc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(demo_cxx_link src/demo_cxx_link.cc)
target_link_libraries(demo_cxx_link PRIVATE psz_kernel_cuda CUDA::cudart)

add_executable(demo_lorenzo src/demo_lorenzo.cc)
target_link_libraries(demo_lorenzo PRIVATE psz_kernel_cuda psz_utilsg psz_statg
                                           CUDA::cudart)

# add_executable(demo_lorenzo_variant src/demo_lorenzo_variant.cc)
# target_link_libraries(demo_lorenzo_variant PRIVATE psz_kernel_cuda psz_utilsg
#                                                    psz_statg CUDA::cudart)

add_executable(demo_pipeline src/demo_pipeline.cu)
target_link_libraries(demo_pipeline PRIVATE psz_spv psz_kernel_cuda psz_hfg
                                            psz_utilsg psz_statg)
set_target_properties(demo_pipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bin_hf src/bin_hf.cc)
target_link_libraries(bin_hf PRIVATE cusz psz_utilsg psz_statg psz_hfg
                                     CUDA::cudart)
set_target_properties(bin_hf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(bin_spv src/bin_spv.cc)
# target_link_libraries(bin_spv PRIVATE psz_spv psz_utilsg psz_statg CUDA::cudart)
