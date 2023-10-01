add_library(psztestcompile_dp INTERFACE)
target_include_directories(
psztestcompile_dp
INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src/>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>)

# [psz::note] ad hoc solution from oneMKL sample code
# [psz::note] at least change to per-target
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -std=c++17")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core")

# utils for test
add_library(psztest_utils_dp src/utils/rand.seq.cc src/utils/rand.dp.cpp)
target_link_libraries(psztest_utils_dp)

# testing sp vector
add_executable(spv_dp src/test_spv.dp.cpp)
target_link_libraries(spv_dp PRIVATE pszspv_dp psztest_utils_dp)
add_test(test_spv_dp spv_dp)

# # testing timer wrapper

# # add_executable(tcpu src/tcpu.c) target_link_libraries(tcpu PRIVATE psztime)
# # add_test(test_tcpu tcpu)

# # add_executable(tgpu src/tgpu.cu) target_link_libraries(tgpu PRIVATE psztime)
# # add_test(test_tgpu tgpu)

# # correctness, include kernel `.inl` directly ### test_typing test core
# # functionality Level-0 basic typing
# add_executable(l0_typing src/test_pncodec_func.cc)
# target_link_libraries(l0_typing PRIVATE psztestcompile_dp)
# add_test(test_l0_typing l0_typing)

# # Level-1 subroutine
# add_executable(l1_scan src/test_l1_l23scan.cu)
# target_link_libraries(l1_scan PRIVATE pszcompile_settings
# psztestcompile_dp)
# add_test(test_l1_scan l1_scan)

# add_executable(l1_compact src/test_l1_compact.cu)
# target_link_libraries(l1_compact PRIVATE pszcompile_settings
# psztestcompile_dp psz_testutils)
# add_test(test_l1_compact l1_compact)

# # Level-2 kernel (template; unit tests)
# add_executable(l2_serial src/test_l2_serial.cc)
# target_link_libraries(l2_serial PRIVATE psztestcompile_dp)
# add_test(test_l2_serial l2_serial)

# add_executable(l2_cudaproto src/test_l2_cudaproto.cu)
# target_link_libraries(
# l2_cudaproto PRIVATE pszcompile_settings psztestcompile_dp pszmem
# pszstat_cu)
# add_test(test_l2_cudaproto l2_cudaproto)

# add_executable(l2_histsp src/test_l2_histsp.cu)
# target_link_libraries(l2_histsp PRIVATE pszcompile_settings pszmem pszstat_cu
# pszkernel_cu pszkernel_seq pszstat_seq)
# add_test(test_l2_histsp l2_histsp)

# # Level-3 kernel with configuration (low-level API)
# add_executable(l3_cuda_pred src/test_l3_cuda_pred.cc)
# target_link_libraries(
# l3_cuda_pred PRIVATE pszkernel_cu psz_testutils pszstat_seq pszstat_cu pszmem
# CUDA::cudart)
# add_test(test_l3_cuda_pred l3_cuda_pred)

add_executable(lrzsp_dp src/test_lrzsp.dp.cpp)
target_link_libraries(
    lrzsp_dp
    PRIVATE psztestcompile_dp
    pszkernel_dp
    psztest_utils_dp
    pszspv_dp
    pszstat_seq
    pszstat_dp
    pszmem)
add_test(test_lrzsp_dp lrzsp_dp)

# if(PSZ_REACTIVATE_THRUSTGPU)
# add_compile_definitions(REACTIVATE_THRUSTGPU)
# add_executable(statfn src/test_statfn.cc)
# target_link_libraries(statfn PRIVATE psztestcompile_dp psz_testutils
# pszstat_cu pszstat_seq pszmem)
# else()
# add_executable(statfn src/test_statfn.cc)
# target_link_libraries(statfn PRIVATE psztestcompile_dp psz_testutils
# pszstat_cu pszstat_seq pszmem)
# endif()
