add_executable(demo_capi_nvcc src/demo_capi_nvcc.cu)
target_link_libraries(demo_capi_nvcc PRIVATE cusz)
set_target_properties(demo_capi_nvcc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(demo_cxx_link src/demo_cxx_link.cc)
target_link_libraries(demo_cxx_link PRIVATE pszkernel_cu CUDA::cudart)

add_executable(bin_pred-hist src/bin_pred-hist.cc)
target_link_libraries(
  bin_pred-hist
  PRIVATE pszkernel_cu
          pszutils_cu
          pszmem
          pszstat_ser
          pszstat_cu
          pszkernel_ser
          CUDA::cudart)

# deprecated
add_executable(bin_spline3 src/bin_spline3.cu)
target_link_libraries(bin_spline3 PRIVATE pszkernel_cu pszutils_cu pszmem
                                          pszstat_ser pszstat_cu)

# add_executable(demo_lorenzo_variant src/demo_lorenzo_variant.cc)
# target_link_libraries(demo_lorenzo_variant PRIVATE pszkernel_cu pszutils_cu
# pszstat_cu CUDA::cudart)

# deprecated
add_executable(bin_pipeline_proto src/bin_pipeline_proto.cu)
target_link_libraries(bin_pipeline_proto PRIVATE pszspv_cu pszkernel_cu pszmem
                                                 pszhf_cu pszutils_cu pszstat_cu)
set_target_properties(bin_pipeline_proto PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                    ON)

add_executable(bin_hf src/bin_hf.cc)
target_link_libraries(bin_hf PRIVATE cusz pszutils_cu pszstat_cu pszhf_cu
                                     CUDA::cudart)
set_target_properties(bin_hf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE pszkernel_cu pszutils_cu pszmem
                                       pszstat_cu CUDA::cudart)

# add_executable(bin_spv src/bin_spv.cc) target_link_libraries(bin_spv PRIVATE
# pszspv_cu pszutils_cu pszstat_cu CUDA::cudart)

add_executable(bin_viadisk src/bin_viadisk.cc)
target_link_libraries(
  bin_viadisk
  PRIVATE pszcompile_settings
          cusz
          pszkernel_cu
          pszutils_cu
          pszstat_cu
          pszhf_cu
          CUDA::cudart)
