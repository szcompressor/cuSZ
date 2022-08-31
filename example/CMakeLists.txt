add_executable(cxxapi src/cxxapi.cc)
target_link_libraries(cxxapi PRIVATE cusz)
#set_target_properties(cxxapi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(capi src/capi.cc)
target_link_libraries(capi PRIVATE cusz)
#set_target_properties(capi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(ck1 src/ck1.cc)
target_link_libraries(ck1 PRIVATE cuszkernel)

add_executable(sahuff src/sahuff.cc)
target_link_libraries(sahuff PRIVATE cusz)
set_target_properties(sahuff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sasp src/sasp.cc)
target_link_libraries(sasp PRIVATE cuszspv)
