#include "utils/verinfo.h"

#include <cstdio>

#if defined(__clang__)
const char* cxx_compiler = "clang++";
#define CXX_COMPILER_MAJOR __clang_major__
#define CXX_COMPILER_MINOR __clang_minor__
#define CXX_COMPILER_PATCH __clang_patchlevel__
#elif defined(__GNUC__)
const char* cxx_compiler = "g++";
#define CXX_COMPILER_MAJOR __GNUC__
#define CXX_COMPILER_MINOR __GNUC_MINOR__
#define CXX_COMPILER_PATCH __GNUC_PATCHLEVEL__
#else
const char* cxx_compiler = "unknown C++ compiler";
#define CXX_COMPILER_MAJOR major
#define CXX_COMPILER_MINOR minor
#define CXX_COMPILER_PATCH patch
#endif

void print_CXX_ver()
{
  printf(
      "- %s: %s.%s.%s\n", cxx_compiler,  //
      STRINGIZE_VALUE_OF(CXX_COMPILER_MAJOR),
      STRINGIZE_VALUE_OF(CXX_COMPILER_MINOR),
      STRINGIZE_VALUE_OF(CXX_COMPILER_PATCH));
}

float DDR_memory_bandwidth_GBps_base1000(
    float mem_bus_bitwidth, float clock_rate_Hz)
{
  return (mem_bus_bitwidth / 8.0f)  // bytes
         * (clock_rate_Hz) * 2      // DDR
         / (1000 * 1000 * 1000);    // base-100
}

float DDR_memory_bandwidth_GiBps_base1024(
    float mem_bus_bitwidth, float clock_rate_Hz)
{
  return (mem_bus_bitwidth / 8.0f)  // bytes
         * (clock_rate_Hz) * 2      // DDR
         / (1024 * 1024 * 1024);    // base-1024
}