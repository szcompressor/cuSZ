#include <cuda_runtime.h>

#include <cstdio>

#include "hf.h"
#include "hf_hl.hh"
#include "hf_impl.hh"

#if defined(PHF_USE_CUDA)
const char* PHF_BACKEND_TEXT = "cuHF";
#elif defined(PHF_USE_HIP)
const char* PHF_BACKEND_TEXT = "hipHF";
#elif defined(PHF_USE_1API)
const char* PHF_BACKEND_TEXT = "dpHF";
#endif

const char* PHF_VERSION_TEXT = "coarse_2024-03-20, ReVISIT_2021-(TBD)";
const int PHF_VERSION = 20240527;

const int PHF_COMPATIBILITY = 0;
constexpr int PHF_DEFLATE_CONSTANT = 4;

constexpr int BLOCK_DIM_ENCODE = 256;
constexpr int BLOCK_DIM_DEFLATE = 256;

size_t phf_coarse_tune_sublen(size_t len)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

#if defined(PHF_USE_CUDA) || defined(PHF_USE_HIP)
  // TODO ROCm GPUs should use different constants.
  int current_dev = 0;
  cudaSetDevice(current_dev);
  cudaDeviceProp dev_prop{};
  cudaGetDeviceProperties(&dev_prop, current_dev);

  auto nSM = dev_prop.multiProcessorCount;
  auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
  auto deflate_nthread = allowed_block_dim * nSM / PHF_DEFLATE_CONSTANT;

#elif defined(PHF_USE_1API)
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

void phf_coarse_tune(size_t len, int* sublen, int* pardeg)
{
  auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
  *sublen = phf_coarse_tune_sublen(len);
  *pardeg = div(len, *sublen);
}

uint32_t phf_encoded_bytes(phf_header* h) { return h->entry[PHFHEADER_END]; }

void phf_print_header(const phf_header* h, const char* dtype_str)
{
  static const char* const sec_name[] = {"HEADER",    "RVBK",   "PAR_NBIT", "PAR_ENTRY",
                                         "BITSTREAM", "SP_VAL", "SP_IDX"};

  printf("(bk-len, sub-len, par-deg)=(%d, %d, %d)\n", h->bklen, h->sublen, h->pardeg);
  printf("total (-nbit, -ncell)=(%zu, %zu)\tbr-num=%u\n", h->total_nbit, h->total_ncell, h->brnum);

  auto sizeof_dtype = dtype_str ? ((strcmp(dtype_str, "u1") == 0)   ? 1
                                   : (strcmp(dtype_str, "u2") == 0) ? 2
                                   : (strcmp(dtype_str, "u4") == 0) ? 4
                                   : (strcmp(dtype_str, "u8") == 0) ? 8
                                                                    : 0)
                                : 0;
  auto comp_bytes = h->entry[PHFHEADER_END] * 1.0;
  auto original_bytes = h->original_len * sizeof_dtype * 1.0;
  auto CR = original_bytes / comp_bytes;
  printf(
      "(ori-len, dtype)=(%zu, %s)\t"
      "(ori-bytes, comp-bytes)=(%.2f, %.2f)\t"
      "CR=%.2f\n",
      h->original_len, dtype_str ? dtype_str : "unknown", original_bytes, comp_bytes, CR);

  printf("\n");
  printf("%-12s  %10s  %10s\n", "field", "offset", "bytes");
  printf("%-12s  %10s  %10s\n", "-------", "------", "-----");
  for (int i = 0; i < PHFHEADER_END; ++i) {
    uint32_t off = h->entry[i];
    uint32_t size = h->entry[i + 1] - h->entry[i];
    if (size == 0) continue;  // skip empty SP sections when brnum==0
    printf("%-12s  %10u  %10u\n", sec_name[i], off, size);
  }
  printf("%-12s  %10s  %10u  (total)\n\n", "", "", h->entry[PHFHEADER_END]);
}

void phf_version() { printf("\n///  %s build: %s\n", PHF_BACKEND_TEXT, PHF_VERSION_TEXT); }

void phf_versioninfo() { phf_version(); }
