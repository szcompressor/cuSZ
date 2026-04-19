#include "hf_hl.hh"

#include "hf_impl.hh"
#include "mem/cxx_backends.h"
#include "rs_merge.hh"

using H4 = u4;
using M = PHF_METADATA;

namespace phf {

template <typename E>
using phf_module = cuhip::modules<E, H4>;

template <typename E>
int high_level<E>::build_book(phf::Buf<E>* buf, u4* h_hist, u2 const rt_bklen, HF_STREAM stream)
{
  buf->register_runtime_bklen(rt_bklen);

  phf_CPU_build_canonized_codebook_v2<E, H4>(
      h_hist, rt_bklen, buf->book_h(), buf->rvbk_h(), buf->rvbk_bytes());
  memcpy_allkinds_async<H2D>(buf->book_d(), buf->book_h(), rt_bklen, (cudaStream_t)stream);

  // TODO duplicate memory copy
  memcpy_allkinds_async<H2D>(
      buf->rvbk_d(), buf->rvbk_h(), buf->rvbk_bytes(), (cudaStream_t)stream);

  return 0;
}

template <typename E>
int high_level<E>::encode(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    HF_STREAM stream)
{
  phf_module<E>::GPU_coarse_encode(
      in, len, buf->book_d(), buf->rt_bklen(), buf->numSMs(), {buf->sublen(), buf->pardeg()},
      // internal buffers
      buf->scratch_d(), buf->par_nbit_d(), buf->par_nbit_h(), buf->par_ncell_d(),
      buf->par_ncell_h(), buf->par_entry_d(), buf->par_entry_h(), buf->bitstream_d(),
      buf->bitstream_max_len(),
      // output
      &header.total_nbit, &header.total_ncell, stream);

  sync_by_stream(stream);

  {  // make metadata
    M nbyte[PHFHEADER_END];
    buf->update_header(header);
    buf->calc_offset(header, nbyte);
  }

  buf->memcpy_merge(header, stream);  // TODO externalize/make explicit

  *out = buf->encoded_d();
  *outlen = phf_encoded_bytes(&header);

  return 0;
}

template <typename E>
int high_level<E>::encode_HFR(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    phf_stream_t stream)
{
  // Default configuration: ChunkSize=1024 (Magnitude=10), ShardSize=4 (ReduceTimes=2).
  // BlockDim = NumShards = 2^(10-2) = 256 threads.
  constexpr int Magnitude = 10;
  constexpr int ReduceTimes = 2;
  constexpr u4 ChunkSize = 1u << Magnitude;
  using Mod = phf::module::HFReVISIT_encode<E, Magnitude, ReduceTimes>;

  const size_t pardeg = buf->pardeg();

  // Build alt-code from the current codebook.
  H4 alt_code{0};
  u4 alt_bitcount{0};
  phf::make_altcode<H4>(buf->book_h(), buf->rt_bklen(), ReduceTimes, alt_code, alt_bitcount);

  // Reset sparse-buffer counter.
  cudaMemsetAsync(buf->brnum_d(), 0, sizeof(u4), (cudaStream_t)stream);

  // Kernel writes dense output at fixed stride: dn_out[ChunkSize * blockIdx.x + threadIdx.x].
  Mod::GPU_kernel(
      in, len, buf->book_d(), buf->rt_bklen(), alt_code, alt_bitcount, buf->bitstream_d(),
      buf->par_nbit_d(), buf->brval_d(), buf->bridx_d(), buf->brnum_d(), stream);

  sync_by_stream(stream);

  // D2H: get per-block bit counts and sparse counter to build compact layout.
  cudaMemcpy(buf->par_nbit_h(), buf->par_nbit_d(), pardeg * sizeof(M), cudaMemcpyDeviceToHost);
  u4 h_brnum{0};
  cudaMemcpy(&h_brnum, buf->brnum_d(), sizeof(u4), cudaMemcpyDeviceToHost);
  header.brnum = h_brnum;

  M total_ncell = 0;
  for (size_t i = 0; i < pardeg; i++) {
    buf->par_ncell_h()[i] = (buf->par_nbit_h()[i] + 31) / 32;
    buf->par_entry_h()[i] = total_ncell;
    total_ncell += buf->par_ncell_h()[i];
  }

  // Compact the fixed-stride bitstream on the host:
  //   source block i at [i*ChunkSize, i*ChunkSize + par_ncell[i])
  //   dest   block i at [par_entry[i], par_entry[i] + par_ncell[i])
  // Processing blocks in order guarantees dest <= src, so no clobber.
  cudaMemcpy(
      buf->bitstream_h(), buf->bitstream_d(), pardeg * ChunkSize * sizeof(H4),
      cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < pardeg; i++) {
    if (buf->par_ncell_h()[i] > 0)
      memmove(
          buf->bitstream_h() + buf->par_entry_h()[i], buf->bitstream_h() + i * ChunkSize,
          buf->par_ncell_h()[i] * sizeof(H4));
  }

  // H2D: upload compact bitstream and corrected per-partition metadata.
  cudaMemcpy(
      buf->bitstream_d(), buf->bitstream_h(), total_ncell * sizeof(H4), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(
      buf->par_ncell_d(), buf->par_ncell_h(), pardeg * sizeof(M), cudaMemcpyHostToDevice,
      (cudaStream_t)stream);
  cudaMemcpyAsync(
      buf->par_entry_d(), buf->par_entry_h(), pardeg * sizeof(M), cudaMemcpyHostToDevice,
      (cudaStream_t)stream);

  header.total_nbit = 0;
  for (size_t i = 0; i < pardeg; i++) header.total_nbit += buf->par_nbit_h()[i];
  header.total_ncell = total_ncell;

  {
    M nbyte[PHFHEADER_END];
    buf->update_header(header);
    buf->calc_offset(header, nbyte);
  }

  buf->memcpy_merge(header, stream);

  *out = buf->encoded_d();
  *outlen = phf_encoded_bytes(&header);

  return 0;
}

#define PHF_ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_encoded + header.entry[PHFHEADER_##SYM])

template <typename E>
int high_level<E>::decode(
    Buf<E>* buf, phf_header& header, uint8_t* in_encoded, E* out_decoded, HF_STREAM stream)
{
  phf_module<E>::GPU_coarse_decode(
      PHF_ACCESSOR(BITSTREAM, H4), PHF_ACCESSOR(RVBK, PHF_BYTE), buf->rvbk_bytes(),
      PHF_ACCESSOR(PAR_NBIT, M), PHF_ACCESSOR(PAR_ENTRY, M), header.sublen, header.pardeg,
      out_decoded, stream);

  // HFR: scatter breaking-point values back over the decoded center-symbol placeholders.
  if (header.brnum > 0)
    phf_module<E>::GPU_scatter(
        PHF_ACCESSOR(SP_VAL, E), PHF_ACCESSOR(SP_IDX, u4), header.brnum, out_decoded, stream);

  return 0;
}

template <typename E>
int high_level<E>::encode_ReVISIT_lite(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    phf_stream_t stream)
{
  return encode_HFR(buf, in, len, out, outlen, header, stream);
}

}  // namespace phf

template struct phf::high_level<u1>;
template struct phf::high_level<u2>;
template struct phf::high_level<u4>;

#undef PHF_ACCESSOR