add_executable(capi  src/capi.cu)
target_link_libraries(capi PRIVATE cusz)
set_target_properties(capi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(hostcclink  src/hostcclink.cc)
target_link_libraries(hostcclink PRIVATE parszkelo CUDA::cudart)

add_executable(ck1  src/ck1.cc)
target_link_libraries(ck1 PRIVATE parszkernel parszkelo parszutils_g parszstat_g CUDA::cudart)

add_executable(ck1var  src/ck1var.cc)
target_link_libraries(ck1var PRIVATE parszkernel parszkelo parszutils_g parszstat_g CUDA::cudart)

add_executable(pipeline  src/pipeline.cu)
target_link_libraries(pipeline PRIVATE parszspv parszkernel parszkelo parszhf_g parszutils_g parszstat_g)
set_target_properties(pipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sahuff  src/sahuff.cc)
target_link_libraries(sahuff PRIVATE cusz parszutils_g parszstat_g parszhf_g CUDA::cudart)
set_target_properties(sahuff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sasp  src/sasp.cc)
target_link_libraries(sasp PRIVATE parszspv parszutils_g parszstat_g CUDA::cudart)
