
add_library(psz_cu_test_compile_settings INTERFACE)
target_include_directories(
  psz_cu_test_compile_settings
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../portable/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../psz/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../psz/src/>
)

# utils for test
add_library(psz_cu_test_utils src/utils/rand.seq.cc src/utils/rand.cu_hip.cc)
target_link_libraries(psz_cu_test_utils CUDA::cudart CUDA::curand)

# functionality
add_executable(zigzag src/test_zigzag_codec.cc)
target_link_libraries(zigzag PRIVATE psz_cu_test_compile_settings)
add_test(test_zigzag zigzag)

# Level-1 subroutine
add_executable(l1_compact src/test_l1_compact.cu)
target_link_libraries(l1_compact PRIVATE psz_cu_compile_settings
  psz_cu_test_compile_settings psz_cu_test_utils)
add_test(test_l1_compact l1_compact)

# Level-2 kernel (template; unit tests)
add_executable(histsp_cu src/tune_histsp.cu)
target_link_libraries(histsp_cu
  PRIVATE psz_cu_compile_settings
  cusz
)
add_test(test_histsp_cu histsp_cu)

# Level-3 kernel with configuration (low-level API)
add_executable(lrz_seq src/test_lrz.seq.cc)
target_link_libraries(lrz_seq
  PRIVATE psz_cu_test_compile_settings)
add_test(test_lrz_seq lrz_seq)

if(PSZ_REACTIVATE_THRUSTGPU)
  add_compile_definitions(REACTIVATE_THRUSTGPU)
  add_executable(statfn src/test_statfn.cc)
  target_link_libraries(statfn
    PRIVATE psz_cu_test_compile_settings
    psz_cu_test_utils psz_cu_mem
  )
else()
  add_executable(statfn src/test_statfn.cc)
  target_link_libraries(statfn
    PRIVATE psz_cu_test_compile_settings
    psz_cu_test_utils psz_cu_mem
  )
endif()

add_executable(stat_identical src/test_identical.cc)
target_link_libraries(stat_identical
  PRIVATE
  psz_cu_test_compile_settings
  psz_cu_compile_settings
  psz_cu_test_utils
  psz_cu_stat
  CUDA::cudart
)
add_test(test_stat_identical stat_identical)

add_executable(stat_max_error src/test_max_error.cc)
target_link_libraries(stat_max_error
  PRIVATE
  psz_cu_test_compile_settings
  psz_cu_compile_settings
  psz_cu_test_utils
  psz_cu_stat
  CUDA::cudart
)
add_test(test_stat_max_error stat_max_error)