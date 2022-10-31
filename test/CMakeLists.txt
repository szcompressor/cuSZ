add_library(test_rand src/rand.cc)

add_executable(spv src/spv.cu)
target_link_libraries(spv PRIVATE accszspv test_rand)
add_test(test_spv spv)

add_executable(spvo src/spvo.cu)
target_link_libraries(spvo PRIVATE accszspv test_rand)
add_test(test_spv_obj spvo)


add_executable(tcpu src/tcpu.c)
target_link_libraries(tcpu PRIVATE asztimer)
add_test(test_tcpu tcpu)

add_executable(tgpu src/tgpu.cu)
target_link_libraries(tgpu PRIVATE asztimer)
add_test(test_tgpu tgpu)