# Two test layers coexist:
#   - Below: per-kernel unit / functional tests (executable per test, asserts
#     internally). GPU-touching tests carry RESOURCE_LOCK gpu (set at end of
#     this file) so they serialize under `ctest -j N`.
#   - test/cmake/cuda-test-bin_hf.cmake: bin_hf-driven matrix that exercises
#     full codec paths on synthesized data via add_test(... COMMAND bin_hf ...).

add_library(psz_cu_test_compile_settings INTERFACE)
target_include_directories(
  psz_cu_test_compile_settings
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../portable/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../psz/include/>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../psz/src/>
)

# functionality
add_executable(zigzag src/test_zigzag_codec.cc)
target_link_libraries(zigzag PRIVATE psz_cu_test_compile_settings)
add_test(test_zigzag zigzag)

# Level-1 subroutine
add_executable(l1_compact src/test_l1_compact.cu)
target_link_libraries(l1_compact PRIVATE psz_cu_compile_settings
  psz_cu_test_compile_settings PORTABLE::testutils)
add_test(test_l1_compact l1_compact)

# Level-2 kernel (template; unit tests)
add_executable(histsp_cu src/tune_histsp.cu)
target_link_libraries(histsp_cu
  PRIVATE psz_cu_compile_settings
  psz_seq_core
  cusz
)
add_test(test_histsp_cu histsp_cu)

# Level-3 kernel with configuration (low-level API)
add_executable(lrz_seq src/test_lrz.seq.cc)
target_link_libraries(lrz_seq
  PRIVATE psz_cu_test_compile_settings psz_seq_core)
add_test(test_lrz_seq lrz_seq)

if(PSZ_REACTIVATE_THRUSTGPU)
  add_compile_definitions(REACTIVATE_THRUSTGPU)
  add_executable(statfn src/test_statfn.cc)
  target_link_libraries(statfn
    PRIVATE psz_cu_test_compile_settings
    PORTABLE::testutils psz_cu_mem
  )
else()
  add_executable(statfn src/test_statfn.cc)
  target_link_libraries(statfn
    PRIVATE psz_cu_test_compile_settings
    PORTABLE::testutils psz_cu_mem
    UTILS::stat_seq
  )
endif()

add_executable(stat_identical1 src/test_identical1.cc)
target_link_libraries(stat_identical1
  PRIVATE
  psz_cu_test_compile_settings
  psz_cu_compile_settings
  PORTABLE::testutils
  UTILS::stat_cu
  UTILS::stat_seq
  CUDA::cudart
)
add_test(test_stat_identical1 stat_identical1)

add_executable(stat_identical2 src/test_identical2.cu)
target_link_libraries(stat_identical2
  PRIVATE
  psz_cu_test_compile_settings
  psz_cu_compile_settings
  PORTABLE::testutils
  UTILS::stat_cu
  UTILS::stat_seq
  CUDA::cudart
)
add_test(test_stat_identical2 stat_identical2)

add_executable(stat_max_error src/test_max_error.cc)
target_link_libraries(stat_max_error
  PRIVATE
  psz_cu_test_compile_settings
  psz_cu_compile_settings
  PORTABLE::testutils
  UTILS::stat_cu
  UTILS::stat_seq
  CUDA::cudart
)
add_test(test_stat_max_error stat_max_error)

add_executable(mem_unique src/test_mem_unique.cu)
target_link_libraries(mem_unique
  PRIVATE
  psz_cu_compile_settings
  psz_cu_test_compile_settings
  psz_cu_mem
  CUDA::cudart
)
add_test(test_mem_unique mem_unique)

add_executable(test_hfr src/test_hfr.cc)
target_link_libraries(test_hfr
  PRIVATE
  psz_cu_test_compile_settings
  PSZ::CUDA::phf
  CUDA::cudart
)
add_test(test_hf_revisit_altcode test_hfr)

add_executable(test_hfserial src/test_hfserial.cc)
target_link_libraries(test_hfserial
  PRIVATE
  psz_cu_test_compile_settings
  PSZ::CUDA::phf
)
add_test(test_hf_cpu_serial_codebook test_hfserial)

# GPU tests serialize via a named resource lock so `ctest -j N` doesn't
# oversubscribe the device. Tests that touch only the host (zigzag, lrz_seq,
# test_hf_cpu_serial_codebook) are intentionally omitted and run in parallel.
set_tests_properties(
  test_l1_compact
  test_histsp_cu
  test_stat_identical1
  test_stat_identical2
  test_stat_max_error
  test_mem_unique
  test_hf_revisit_altcode
  PROPERTIES RESOURCE_LOCK gpu
)
