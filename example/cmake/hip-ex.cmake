# TODO add_executable(demo_capi_cxx src/demo_capi_cxx.cc)
# target_link_libraries(demo_capi_cxx PRIVATE cusz)

add_executable(demo_cxx_link src/demo_cxx_link.cc)
target_link_libraries(demo_cxx_link PRIVATE pszkernel_hip hip::host)

add_library(ex_utils src/ex_utils_hip.cpp)
target_link_libraries(ex_utils PUBLIC pszcompile_settings)

add_executable(bin_pipeline src/bin_pipeline.cc)
target_link_libraries(
  bin_pipeline
  PRIVATE ex_utils
          pszmem
          pszstat_ser
          pszkernel_ser
          pszkernel_hip
          pszstat_hip
          pszhf_hip
          hip::host)

add_executable(bin_hf src/bin_hf.cc)
target_link_libraries(bin_hf PRIVATE cusz pszstat_hp pszhf_hip hip::host)

add_executable(bin_hist src/bin_hist.cc)
target_link_libraries(bin_hist PRIVATE pszkernel_cu pszkernel_ser pszmem
                                       pszstat_cu hip::host)

# if(PSZ_RESEARCH_HUFFBK_CUDA) add_executable(bin_hfserial src/bin_hfserial.cc)
# target_link_libraries(bin_hfserial PRIVATE pszhfbook_ser pszhf_cu pszmem
# hip::host) endif(PSZ_RESEARCH_HUFFBK_CUDA)
