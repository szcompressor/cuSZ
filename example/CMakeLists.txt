add_executable(demo_capi_nvcc src/demo_capi_nvcc.cu)
target_link_libraries(demo_capi_nvcc PRIVATE cusz)
set_target_properties(demo_capi_nvcc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(demo_cxx_link src/demo_cxx_link.cc)
target_link_libraries(demo_cxx_link PRIVATE pszkernel_cu CUDA::cudart)

# add_executable(demo_lorenzo_variant src/demo_lorenzo_variant.cc)
# target_link_libraries(demo_lorenzo_variant PRIVATE pszkernel_cu pszutils_cu
# pszstat_cu CUDA::cudart)

add_library(ex_utils src/ex_utils.cu)
target_link_libraries(ex_utils PUBLIC pszcompile_settings)

add_executable(bin_pipeline src/bin_pipeline.cc)
target_link_libraries(
  bin_pipeline
  PRIVATE pszkernel_cu
          pszutils_cu
          pszmem
          pszstat_ser
          pszstat_cu
          pszhf_cu
          pszkernel_ser
          ex_utils
          CUDA::cudart)

add_executable(bin_hf src/bin_hf.cc)
target_link_libraries(bin_hf PRIVATE cusz pszutils_cu pszstat_cu pszhf_cu
                                     CUDA::cudart)
set_target_properties(bin_hf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE pszkernel_cu pszkernel_ser pszutils_cu
                                       pszmem pszstat_cu CUDA::cudart)

# add_executable(bin_spv src/bin_spv.cc) target_link_libraries(bin_spv PRIVATE
# pszspv_cu pszutils_cu pszstat_cu CUDA::cudart)

add_executable(bin_hfserial src/bin_hfserial.cc)
target_link_libraries(bin_hfserial PRIVATE pszhfbook_ser pszhf_cu pszmem
                                           CUDA::cudart)
