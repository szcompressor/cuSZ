add_executable(demo_capi_cuda_nvcc src/demo_capi_cuda.cu)
target_link_libraries(demo_capi_cuda_nvcc PRIVATE cusz)

add_executable(demo_capi_cuda src/demo_capi_cuda.cc)
target_link_libraries(demo_capi_cuda PRIVATE cusz)

add_library(ex_utils src/ex_utils.cu)
target_link_libraries(ex_utils PUBLIC pszcompile_settings)

# add_executable(bin_pipeline src/bin_pipeline.cc)
# target_link_libraries(
#   bin_pipeline
#   PRIVATE ex_utils
#           pszmem
#           pszstat_seq
#           pszkernel_seq
#           pszkernel_cu
#           pszstat_cu
#           pszspv_cu
#           pszhf_cu
#           CUDA::cudart)

add_executable(bin_hf src/bin_hf.cc)
target_link_libraries(bin_hf PRIVATE cusz pszstat_cu pszhf_cu CUDA::cudart)
set_target_properties(bin_hf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE pszkernel_cu pszkernel_seq pszmem
                                       pszstat_cu CUDA::cudart)

if(PSZ_RESEARCH_HUFFBK_CUDA)
  add_executable(bin_hfserial src/bin_hfserial.cc)
  target_link_libraries(bin_hfserial PRIVATE pszhfbook_seq pszhf_cu pszmem
                                             CUDA::cudart)
endif(PSZ_RESEARCH_HUFFBK_CUDA)
