cmake_minimum_required(VERSION 3.14...3.18)

project(cusz_example CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

## Uncommenting to specify GPU architecture is only for testing purpose.
## uncomment for pre-Turing
# set(CMAKE_CUDA_ARCHITECTURES 60 62 70 72)
## uncomment for V100
# set(CMAKE_CUDA_ARCHITECTURES 70)
## uncomment for Turing
# set(CMAKE_CUDA_ARCHITECTURES 75)
## uncmoment for Ampere
# set(CMAKE_CUDA_ARCHITECTURES 80 86)
## uncmoment for A100
# set(CMAKE_CUDA_ARCHITECTURES 80)

enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR}/../src)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")

set(LIB_TYPE SHARED)

add_executable(ex-api src/ex_api_core.cu)
target_link_libraries(ex-api
	PUBLIC
	${CMAKE_SOURCE_DIR}/../Release/libcompress.a 
	${CMAKE_SOURCE_DIR}/../Release/libargp.a 
	${CMAKE_SOURCE_DIR}/../Release/libpq.a 
	${CMAKE_SOURCE_DIR}/../Release/libhuff.a 
	${CMAKE_SOURCE_DIR}/../Release/libsp.a 
	-lcuda
	-lcusparse)
set_target_properties(ex-api PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
