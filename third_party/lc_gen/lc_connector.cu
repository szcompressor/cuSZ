#include "lc_gen/lc_gen.h"

#include "lc_gen/lc_impl.h"

namespace LC_Connector {

namespace {

constexpr int TPB = 512;
constexpr int CS = 1024 * 16;

inline int chunk_count(size_t insize) { return (insize + CS - 1) / CS; }

template <typename ResetKernel, typename EncodeKernel>
void compress_impl(
    ResetKernel reset, EncodeKernel encode, bool need_align8, uint8_t* input, size_t insize,
    psz::LC_Buf* buf, size_t* outsize, void* stream)
{
  const auto cfg = lc_c::config(TPB);
  const int blocks = cfg.blocks;
  const int chunks = chunk_count(insize);

  auto d_encoded = buf->encoded_d();
  auto d_encsize = buf->size_d();
  auto d_fullcarry = buf->fullcarry_d();

  reset<<<1, 1, 0, (cudaStream_t)stream>>>();
  CHECK_GPU(cudaMemsetAsync(d_fullcarry, 0, chunks * sizeof(int), (cudaStream_t)stream));
  encode<<<blocks, TPB, 0, (cudaStream_t)stream>>>(
      input, (int)insize, d_encoded, d_encsize, d_fullcarry);
  CHECK_GPU(cudaMemcpyAsync(
      buf->size_h(), d_encsize, sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t)stream));
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  const auto dencsize = *buf->size_h();
  const size_t padding = need_align8 ? (8 - (dencsize % 8)) % 8 : 0;
  *outsize = (size_t)(dencsize + padding);
}

template <typename ResetKernel, typename DecodeKernel>
void decompress_impl(ResetKernel reset, DecodeKernel decode, uint8_t* input, psz::LC_Buf* buf, void* stream)
{
  CHECK_GPU(cudaMemcpyAsync(
      buf->size_h(), input, sizeof(int), cudaMemcpyDeviceToHost, (cudaStream_t)stream));
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  const auto cfg = lc_c::config(TPB);
  const int blocks = cfg.blocks;
  auto d_decoded = buf->decoded_d();
  auto d_decsize = buf->size_d();

  reset<<<1, 1, 0, (cudaStream_t)stream>>>();
  decode<<<blocks, TPB, 0, (cudaStream_t)stream>>>(input, d_decoded, d_decsize);
}

}  // namespace

void BITR_COMPRESS(uint8_t* input, size_t insize, psz::LC_Buf* buf, size_t* outsize, void* stream)
{
  compress_impl(d_reset_bitr_comp, d_encode_bitr, false, input, insize, buf, outsize, stream);
}

void TCMS_COMPRESS(uint8_t* input, size_t insize, psz::LC_Buf* buf, size_t* outsize, void* stream)
{
  compress_impl(d_reset_tcms_comp, d_encode_tcms, true, input, insize, buf, outsize, stream);
}

void RTR_COMPRESS(uint8_t* input, size_t insize, psz::LC_Buf* buf, size_t* outsize, void* stream)
{
  compress_impl(d_reset_rtr_comp, d_encode_rtr, false, input, insize, buf, outsize, stream);
}

void BITR_DECOMPRESS(uint8_t* input, psz::LC_Buf* buf, void* stream)
{
  decompress_impl(d_reset_bitr_decomp, d_decode_bitr, input, buf, stream);
}

void TCMS_DECOMPRESS(uint8_t* input, psz::LC_Buf* buf, void* stream)
{
  decompress_impl(d_reset_tcms_decomp, d_decode_tcms, input, buf, stream);
}

void RTR_DECOMPRESS(uint8_t* input, psz::LC_Buf* buf, void* stream)
{
  decompress_impl(d_reset_rtr_decomp, d_decode_rtr, input, buf, stream);
}

}  // namespace LC_Connector
