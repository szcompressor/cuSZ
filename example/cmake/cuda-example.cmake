add_executable(demo_capi_cuda src/demo_capi.cuda.cc)
target_link_libraries(demo_capi_cuda PRIVATE cusz)

# add_library(ex_utils src/ex_utils.cu)
# target_link_libraries(ex_utils
# PUBLIC psz_cu_compile_settings)
# add_executable(bin_pipeline_cu src/bin_pipeline.cu_hip.cc)
# target_link_libraries(bin_pipeline_cu
# PRIVATE cusz CUDA::cudart)

add_executable(prequant src/bin_prequant.cc)
target_link_libraries(prequant PRIVATE cusz CUDA::cudart)

add_executable(phf src/bin_phf.cc)
target_link_libraries(phf PRIVATE cusz CUDA::cudart)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE cusz CUDA::cudart)
