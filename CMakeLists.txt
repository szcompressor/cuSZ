cmake_minimum_required(VERSION 3.14)

project(cuSZ LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(CUDA 9.1 REQUIRED)

set(CUDA_ARCH_LIST Auto CACHE STRING
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
)

string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=compute_61")
string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=compute_70")

add_executable(cusz
	src/cusz_exe.cu 
	src/cusz_workflow.cuh 
	src/cusz_dualquant.cuh)

target_link_libraries(cusz ${CUDA_cusparse_LIBRARY} )
set_target_properties(cusz PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
