string(TOUPPER "${PSZ_BACKEND}" _PSZ_TEST_BACKEND_UPPER)
if(_PSZ_TEST_BACKEND_UPPER STREQUAL "CUDA")
  set(PSZ_TEST_GPU_RT CUDA::cudart)
  include(cmake/cuda-test.cmake)              # legacy unit-test binaries
  include(cmake/cuda-test-bin_pred.cmake)     # bin_pred microbench matrix (v1-only on develop)
  include(cmake/cuda-test-bin_hf.cmake)       # bin_hf-driven ctest matrix
  include(cmake/cuda-test-cusz.cmake)         # cusz CLI-driven ctest matrix
elseif(_PSZ_TEST_BACKEND_UPPER STREQUAL "HIP")
  set(PSZ_TEST_GPU_RT hip::device)
  set(PSZ_TEST_HIP ON)
  include(cmake/cuda-test.cmake)              # same sources, marked LANGUAGE HIP
  include(cmake/cuda-test-bin_hf.cmake)       # bin_hf-driven ctest matrix (binary-agnostic)
  include(cmake/cuda-test-cusz.cmake)         # cusz CLI-driven ctest matrix (binary-agnostic)
elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
  include(cmake/sycl-test.cmake)
else()
  message(FATAL_ERROR "Unsupported PSZ_BACKEND selected: ${PSZ_BACKEND}")
endif()
