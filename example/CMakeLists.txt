if(PSZ_BACKEND STREQUAL "CUDA" OR PSZ_BACKEND STREQUAL "cuda")
  include(cmake/cu-ex.cmake)
elseif(PSZ_BACKEND STREQUAL "HIP" OR PSZ_BACKEND STREQUAL "hip")
  include(cmake/hip-ex.cmake)
elseif(PSZ_BACKEND STREQUAL "ONEAPI" OR PSZ_BACKEND STREQUAL "oneapi" OR PSZ_BACKEND STREQUAL "1api")
  include(cmake/dp-ex.cmake)
endif()