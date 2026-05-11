if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  include(cmake/cuda-test.cmake)              # legacy unit-test binaries
  include(cmake/cuda-test-bin_pred.cmake)     # bin_pred microbench matrix (v1-only on develop)
elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
  include(cmake/sycl-test.cmake)
else()
  message(FATAL_ERROR "Unsupported PSZ_BACKEND selected: ${PSZ_BACKEND}")
endif()
