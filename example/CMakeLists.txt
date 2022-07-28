add_executable(cxxapi src/cxxapi.cc)
target_link_libraries(cxxapi PRIVATE cusz)
#set_target_properties(cxxapi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(capi src/capi.cc)
target_link_libraries(capi PRIVATE cusz)
#set_target_properties(capi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
