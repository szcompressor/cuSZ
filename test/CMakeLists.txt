add_library(test_rand src/rand.cc)

add_executable(spv src/spv.cu)
target_link_libraries(spv PRIVATE cuszspv test_rand)
add_test(test_spv spv)

add_executable(spvo src/spvo.cu)
target_link_libraries(spvo PRIVATE cuszspv test_rand)
add_test(test_spv_obj spvo)