find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE _NB_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${_NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# High-level whole-pipeline compressor binding (cusz.h C API).
add_library(psz_comp_cuda STATIC src/compressor.cu)
set_target_properties(psz_comp_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(psz_comp_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(psz_comp_cuda PUBLIC cusz psz_cu_compile_settings CUDA::cudart)

nanobind_add_module(_psz_comp src/comp_bindings.cpp)
target_include_directories(_psz_comp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(_psz_comp PRIVATE psz_comp_cuda)
set_target_properties(_psz_comp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/psz
  # `cmake --install` strips RPATH otherwise; point back at the build tree directly.
  # INSTALL_RPATH_USE_LINK_PATH skips in-build-tree deps, so list those dirs by hand.
  INSTALL_RPATH "${CMAKE_BINARY_DIR};${CMAKE_BINARY_DIR}/utils;${CMAKE_BINARY_DIR}/codec/hf;${CMAKE_BINARY_DIR}/codec/fzg"
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS _psz_comp LIBRARY DESTINATION psz)
