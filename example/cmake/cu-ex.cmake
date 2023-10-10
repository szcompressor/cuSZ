add_executable(demo_capi_cu src/demo_capi.cu_hip.cc)
target_link_libraries(demo_capi_cu PRIVATE cusz)

add_library(ex_utils src/ex_utils.cu)
target_link_libraries(ex_utils PUBLIC pszcompile_settings)

add_executable(bin_pipeline_cu src/bin_pipeline.cu_hip.cc)
target_link_libraries(bin_pipeline_cu PRIVATE psztestframe_cu CUDA::cudart)

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
