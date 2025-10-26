#include "hf_hl.hh"

#include <iostream>

#include "hf_hl.hh"
#include "hf_impl.hh"
#include "mem/cxx_backends.h"

using std::cout;
using std::endl;

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
int high_level<E>::encode_ReVISIT_lite(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    phf_stream_t stream)
{
  // phf_module<E>::GPU_fine_encode(
  //     in, len, buf->book_d(), buf->rt_bklen, {buf->sublen, buf->pardeg}, buf->d_scratch4.get(),
  //     buf->d_par_nbit.get(), buf->h_par_nbit.get(), buf->d_par_ncell.get(),
  //     buf->h_par_ncell.get(), buf->d_par_entry.get(), buf->h_par_entry.get(),
  //     buf->d_bitstream4.get(), buf->bitstream_max_len, buf->d_brval.get(), buf->d_bridx.get(),
  //     buf->d_brnum.get(), &header.total_nbit, &header.total_ncell, stream);
  // sync_by_stream(stream);

  {  // make metadata
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

  return 0;
}

}  // namespace phf

template struct phf::high_level<u1>;
template struct phf::high_level<u2>;
template struct phf::high_level<u4>;

#undef PHF_ACCESSOR