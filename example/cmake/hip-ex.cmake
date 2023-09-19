add_executable(demo_capi_hip src/demo_capi_hip.cc)
target_link_libraries(demo_capi_hip PRIVATE hipsz)

# add_executable(demo_cxx_link src/demo_cxx_link.cc)
# target_link_libraries(demo_cxx_link PRIVATE pszkernel_hip hip::host)

add_library(ex_utils src/ex_utils.hip)
target_link_libraries(ex_utils PUBLIC pszcompile_settings)

add_executable(bin_pipeline src/bin_pipeline.cc)
target_link_libraries(
  bin_pipeline
  PRIVATE ex_utils
          pszmem
          pszstat_seq
          pszkernel_seq
          pszkernel_hip
          pszstat_hip
          pszspv_hip
          pszhf_hip
          hip::host)

add_executable(bin_hf src/bin_hf.cc)
target_link_libraries(bin_hf PRIVATE hipsz pszstat_hip pszhf_hip hip::host)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE pszkernel_hip pszkernel_seq pszmem
                                       pszstat_hip hip::host)

# if(PSZ_RESEARCH_HUFFBK_CUDA) add_executable(bin_hfserial src/bin_hfserial.cc)
# target_link_libraries(bin_hfserial PRIVATE pszhfbook_seq pszhf_cu pszmem
# hip::host) endif(PSZ_RESEARCH_HUFFBK_CUDA)
