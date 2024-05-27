add_executable(demo_capi_cuda src/demo_capi.cu_hip.cc)
target_link_libraries(demo_capi_cuda
  PRIVATE cusz
)

add_library(ex_utils src/ex_utils.cu)
target_link_libraries(ex_utils
  PUBLIC pszcompile_settings
)

# add_executable(bin_pipeline_cu src/bin_pipeline.cu_hip.cc)
# target_link_libraries(bin_pipeline_cu
#   PRIVATE cusz CUDA::cudart
# )

add_executable(bin_hf src/bin_hf.cc)
target_link_libraries(bin_hf
  PRIVATE CUDA::cudart
  cusz
)
set_target_properties(bin_hf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist
  PRIVATE CUDA::cudart
  cusz
)
