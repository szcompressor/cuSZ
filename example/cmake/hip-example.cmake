# HIP (ROCm) example/driver binaries. Mirrors cuda-example.cmake; the bin_hf
# binary backs the bin_hf ctest matrix. CUDA::cupti (profiling-only) is dropped
# on the HIP path. The .cc sources are reused unchanged; the cuda_runtime.h
# shim and c_cu2hip macros (carried by cusz's psz_cu_compile_settings) translate
# their CUDA API calls.
#
# bin_pred2 is omitted: pred_run.hh uses cudaFree as a std::unique_ptr deleter
# (decltype(&cudaFree)), but cudaFree is a macro on the HIP path, not a function
# symbol. bin_pred2 is not used by any HIP ctest (its tests are in
# cuda-test-bin_pred.cmake, which is CUDA-only).

add_library(example_utils2 src/ex_utils2.cc)
target_link_libraries(example_utils2 PRIVATE cusz)

add_executable(bin_pred1 src/bin_pred1.cc)
target_link_libraries(bin_pred1 PRIVATE cusz)

add_executable(bin_hf src/bin_phf.cc)
target_link_libraries(bin_hf PRIVATE cusz hip::device PORTABLE::testutils)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE cusz hip::device)

add_executable(batch_run src/batch_run.cc)
target_link_libraries(batch_run PRIVATE cusz example_utils2 hip::device)
