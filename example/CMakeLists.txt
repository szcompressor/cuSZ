add_executable(cxxapi src/cxxapi.cu)
target_link_libraries(cxxapi PRIVATE cusz)
set_target_properties(cxxapi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(capi src/capi.cu)
target_link_libraries(capi PRIVATE cusz)
set_target_properties(capi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(ck1 src/ck1.cu)
target_link_libraries(ck1 PRIVATE cuszkernel)

add_executable(pipeline src/comp.cu)
target_link_libraries(pipeline PRIVATE cuszspv cuszkernel cuszhuff gpuszutilsgpu)
set_target_properties(pipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sahuff src/sahuff.cu)
target_link_libraries(sahuff PRIVATE cusz gpuszutilsgpu)
set_target_properties(sahuff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sasp src/sasp.cu)
target_link_libraries(sasp PRIVATE cuszspv gpuszutilsgpu)
