add_library(example_utils2 src/ex_utils2.cc)
target_link_libraries(example_utils2 PRIVATE cusz)

add_executable(demo_cuda_v1 src/demo_v1.cuda.cc)
target_link_libraries(demo_cuda_v1 PRIVATE cusz)

add_executable(demo_cuda_v2 src/demo_v2.cuda.cc)
target_link_libraries(demo_cuda_v2 PRIVATE cusz)

add_executable(bin_hf src/bin_phf.cc)
target_link_libraries(bin_hf PRIVATE cusz CUDA::cudart psz_cu_phf)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE cusz CUDA::cudart)

# add_executable(bin_fzgcodec src/bin_fzgcodec.cc)
# target_link_libraries(bin_fzgcodec PRIVATE cusz CUDA::cudart)

add_executable(batch_run src/batch_run.cc)
target_link_libraries(batch_run PRIVATE cusz example_utils2 CUDA::cudart)
