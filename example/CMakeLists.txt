add_executable(ex-api src/ex_api_core.cu)
target_link_libraries(ex-api PRIVATE cusz)
set_target_properties(ex-api PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
