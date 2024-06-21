add_executable(demo_capi_hip src/demo_capi.cu_hip.cc)
target_link_libraries(demo_capi_hip PRIVATE hipsz)

add_library(ex_utils src/ex_utils.hip)
target_link_libraries(ex_utils PUBLIC pszcompile_settings)

add_executable(bin_pipeline_hip src/bin_pipeline.cu_hip.cc)
target_link_libraries(bin_pipeline_hip PRIVATE psztestframe_hip hip::host)

add_executable(bin_hf src/bin_hf.cc)
target_link_libraries(bin_hf PRIVATE hipsz pszstat_hip pszhf_hip hip::host)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE pszkernel_hip pszkernel_seq pszmem
                                       pszstat_hip hip::host)


