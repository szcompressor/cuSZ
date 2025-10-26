#include <cuda_runtime.h>

#include <cstdio>

#include "hf.h"
#include "hf_hl.hh"
#include "hf_impl.hh"

#if defined(PSZ_USE_CUDA)
const char* PHF_BACKEND_TEXT = "cuHF";
#elif defined(PSZ_USE_HIP)
const char* PHF_BACKEND_TEXT = "hipHF";
#elif defined(PSZ_USE_1API)
const char* PHF_BACKEND_TEXT = "dpHF";
#endif

const char* PHF_VERSION_TEXT = "coarse_2024-03-20, ReVISIT_2021-(TBD)";
const int PHF_VERSION = 20240527;

const int PHF_COMPATIBILITY = 0;
constexpr int PHF_DEFLATE_CONSTANT = 4;

constexpr int BLOCK_DIM_ENCODE = 256;
constexpr int BLOCK_DIM_DEFLATE = 256;

size_t capi_phf_coarse_tune_sublen(size_t len)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  // TODO ROCm GPUs should use different constants.
  int current_dev = 0;
  cudaSetDevice(current_dev);
  cudaDeviceProp dev_prop{};
  cudaGetDeviceProperties(&dev_prop, current_dev);

  auto nSM = dev_prop.multiProcessorCount;
  auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
  auto deflate_nthread = allowed_block_dim * nSM / PHF_DEFLATE_CONSTANT;

#elif defined(PSZ_USE_1API)
  int current_dev = 0;
  /*
  DPCT1093:0: The "current_dev" device may be not the one intended for use.
  Adjust the selected device if needed.
  */
  dpct::select_device(current_dev);
  dpct::device_info dev_prop{};
  dpct::dev_mgr::instance().get_device(current_dev).get_device_info(dev_prop);

  auto nEU = dev_prop.get_max_compute_units();
  auto allowed_block_dim = dev_prop.get_max_work_group_size();
  auto deflate_nthread = allowed_block_dim * nEU / DEFLATE_CONSTANT;
  deflate_nthread /= 16;  // simple EU-SM conversion
  deflate_nthread /= 16;  // simple test
#endif

  auto optimal_sublen = div(len, deflate_nthread);
  // round up
  optimal_sublen = div(optimal_sublen, BLOCK_DIM_DEFLATE) * BLOCK_DIM_DEFLATE;

  return optimal_sublen;
};

void capi_phf_coarse_tune(size_t len, int* sublen, int* pardeg)
{
  auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
  *sublen = capi_phf_coarse_tune_sublen(len);
  *pardeg = div(len, *sublen);
}

uint32_t capi_phf_encoded_bytes(phf_header* h) { return h->entry[PHFHEADER_END]; }

void capi_phf_version() { printf("\n///  %s build: %s\n", PHF_BACKEND_TEXT, PHF_VERSION_TEXT); }

void capi_phf_versioninfo() { capi_phf_version(); }
