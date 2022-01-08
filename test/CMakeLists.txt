cmake_minimum_required(VERSION 3.18)

project(cusz_asap CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

## uncomment for pre-Turing
# set(CMAKE_CUDA_ARCHITECTURES 60 62 70 72)
## uncomment for V100
# set(CMAKE_CUDA_ARCHITECTURES 70)
## uncomment for Turing
set(CMAKE_CUDA_ARCHITECTURES 75)
## uncmoment for Ampere
# set(CMAKE_CUDA_ARCHITECTURES 80 86)
## uncmoment for A100
# set(CMAKE_CUDA_ARCHITECTURES 80)

enable_language(CUDA)

include_directories(../src)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")

set(LIB_TYPE SHARED)

add_executable(dummy_cxlorenzo src/dummy_cxlorenzo.cu)
target_link_libraries(dummy_cxlorenzo 
	${CMAKE_SOURCE_DIR}/../Release/libnvgpusz.a 
	${CMAKE_SOURCE_DIR}/../Release/libsp.a 
	${CMAKE_SOURCE_DIR}/../Release/libpq.a 
	${CMAKE_SOURCE_DIR}/../Release/libhuff.a 
	-lcusparse)


add_executable(dryrun src/test_dryrun.cu)
target_link_libraries(dryrun 
	${CMAKE_SOURCE_DIR}/../Release/libnvgpusz.a 
	${CMAKE_SOURCE_DIR}/../Release/libsp.a 
	${CMAKE_SOURCE_DIR}/../Release/libpq.a 
	${CMAKE_SOURCE_DIR}/../Release/libhuff.a 
	-lcusparse)

add_executable(type_binding src/test_type_binding.cu)
target_link_libraries(type_binding -lcusparse)

# typebinding: ../src/binding.hh test_type_binding.cu
# 	clang++ test_type_binding.cu -std=c++14 --cuda-gpu-arch=sm_75 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread
# 	nvcc test_type_binding.cu

add_executable(query src/test_query.cu)
target_compile_options(query PRIVATE -DMAIN)