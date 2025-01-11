#include <cuda_runtime.h>

#include <cstdio>

#include "hf.h"
#include "hf_type.h"
#include "hfclass.hh"
#include "phf_array.hh"

namespace phf {

#if defined(PSZ_USE_CUDA)
const char* BACKEND_TEXT = "cuHF";
#elif defined(PSZ_USE_HIP)
const char* BACKEND_TEXT = "hipHF";
#elif defined(PSZ_USE_1API)
const char* BACKEND_TEXT = "dpHF";
#endif

const char* VERSION_TEXT = "coarse_2024-03-20, ReVISIT_2021-(TBD)";
// const int VERSION = 20240527;

const int COMPATIBILITY = 0;

}  // namespace phf

size_t capi_phf_coarse_tune_sublen(size_t len)
{
  using phf::HuffmanHelper;
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  // TODO ROCm GPUs should use different constants.
  int current_dev = 0;
  cudaSetDevice(current_dev);
  cudaDeviceProp dev_prop{};
  cudaGetDeviceProperties(&dev_prop, current_dev);

  auto nSM = dev_prop.multiProcessorCount;
  auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
  auto deflate_nthread = allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;

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
  auto deflate_nthread = allowed_block_dim * nEU / HuffmanHelper::DEFLATE_CONSTANT;
  deflate_nthread /= 16;  // simple EU-SM conversion
  deflate_nthread /= 16;  // simple test
#endif

  auto optimal_sublen = div(len, deflate_nthread);
  // round up
  optimal_sublen =
      div(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) * HuffmanHelper::BLOCK_DIM_DEFLATE;

  return optimal_sublen;
};

void capi_phf_coarse_tune(size_t len, int* sublen, int* pardeg)
{
  auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
  *sublen = capi_phf_coarse_tune_sublen(len);
  *pardeg = div(len, *sublen);
}

uint32_t capi_phf_encoded_bytes(phf_header* h) { return h->entry[PHFHEADER_END]; }

void capi_phf_version() { printf("\n///  %s build: %s\n", phf::BACKEND_TEXT, phf::VERSION_TEXT); }

void capi_phf_versioninfo() { capi_phf_version(); }

phf_codec* capi_phf_create(size_t const inlen, phf_dtype const t, int const bklen)
{
  int sublen, pardeg;
  capi_phf_coarse_tune(inlen, &sublen, &pardeg);

  auto ret_codec = [&]() -> void* {
    if (t == HF_U1)
      return (void*)new phf::HuffmanCodec<u1>(inlen, bklen, pardeg);
    else if (t == HF_U2)
      return (void*)new phf::HuffmanCodec<u2>(inlen, bklen, pardeg);
    else if (t == HF_U4)
      return (void*)new phf::HuffmanCodec<u4>(inlen, bklen, pardeg);
    else
      return nullptr;
  };

  return new phf_codec{.codec = ret_codec(), .header = new phf_header, .data_type = t};
}

phferr capi_phf_release(phf_codec* c)
{
  delete c;
  return PHF_SUCCESS;
}

phferr capi_phf_buildbook(phf_codec* codec, uint32_t* d_hist, phf_stream_t s)
{
  if (codec->data_type == HF_U1)
    static_cast<phf::HuffmanCodec<u1>*>(codec->codec)->buildbook(d_hist, s);
  else if (codec->data_type == HF_U2)
    static_cast<phf::HuffmanCodec<u2>*>(codec->codec)->buildbook(d_hist, s);
  else if (codec->data_type == HF_U4)
    static_cast<phf::HuffmanCodec<u4>*>(codec->codec)->buildbook(d_hist, s);
  else
    return PHF_WRONG_DTYPE;

  return PHF_SUCCESS;
}

phferr capi_phf_encode(
    phf_codec* codec, void* in, size_t const inlen, uint8_t** encoded, size_t* enc_bytes,
    phf_stream_t s)
{
  if (codec->data_type == HF_U1)
    static_cast<phf::HuffmanCodec<u1>*>(codec->codec)
        ->encode((u1*)in, inlen, encoded, enc_bytes, s);
  else if (codec->data_type == HF_U2)
    static_cast<phf::HuffmanCodec<u2>*>(codec->codec)
        ->encode((u2*)in, inlen, encoded, enc_bytes, s);
  else if (codec->data_type == HF_U4)
    static_cast<phf::HuffmanCodec<u4>*>(codec->codec)
        ->encode((u4*)in, inlen, encoded, enc_bytes, s);
  else
    return PHF_WRONG_DTYPE;

  return PHF_SUCCESS;
}

phferr capi_phf_decode(phf_codec* codec, uint8_t* encoded, void* decoded, phf_stream_t s)
{
  if (codec->data_type == HF_U1)
    static_cast<phf::HuffmanCodec<u1>*>(codec->codec)->decode(encoded, (u1*)decoded, s);
  else if (codec->data_type == HF_U2)
    static_cast<phf::HuffmanCodec<u2>*>(codec->codec)->decode(encoded, (u2*)decoded, s);
  else if (codec->data_type == HF_U4)
    static_cast<phf::HuffmanCodec<u4>*>(codec->codec)->decode(encoded, (u4*)decoded, s);
  else
    return PHF_WRONG_DTYPE;

  return PHF_SUCCESS;
}