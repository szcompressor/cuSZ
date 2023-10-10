# add_executable(demo_capi_dp src/demo_capi_dp.cc)
# target_link_libraries(demo_capi_cuda PRIVATE cusz)

# add_library(ex_utils src/ex_utils.dp.cpp)
# target_link_libraries(ex_utils PUBLIC pszcompile_settings)

add_executable(bin_pipeline_dp src/bin_pipeline.dp.cpp)
target_link_libraries(bin_pipeline_dp PRIVATE psztestframe_dp)

# add_executable(bin_hf src/bin_hf.cc)
# target_link_libraries(bin_hf PRIVATE cusz pszstat_cu pszhf_cu CUDA::cudart)
# set_target_properties(bin_hf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(bin_hist src/bin_hist.cc)
# target_link_libraries(bin_hist PRIVATE pszkernel_cu pszkernel_seq pszmem
#   pszstat_cu CUDA::cudart)

