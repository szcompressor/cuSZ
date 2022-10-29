add_executable(cxxapi src/cxxapi.cu)
target_link_libraries(cxxapi PRIVATE cusz)
set_target_properties(cxxapi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(capi src/capi.cu)
target_link_libraries(capi PRIVATE cusz)
set_target_properties(capi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(hostcclink src/hostcclink.cc)
target_link_libraries(hostcclink PRIVATE accszkernel CUDA::cudart)

add_executable(ck1 src/ck1.cc)
target_link_libraries(ck1 PRIVATE accszkernel accszutilsgpu CUDA::cudart)

add_executable(ck1var src/ck1var.cc)
target_link_libraries(ck1var PRIVATE accszkernel accszutilsgpu CUDA::cudart)

add_executable(pipeline src/comp.cu)
target_link_libraries(pipeline PRIVATE accszspv accszkernel accszhuff accszutilsgpu)
set_target_properties(pipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sahuff src/sahuff.cc)
target_link_libraries(sahuff PRIVATE cusz accszutilsgpu CUDA::cudart)
set_target_properties(sahuff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sasp src/sasp.cu)
target_link_libraries(sasp PRIVATE accszspv accszutilsgpu)
