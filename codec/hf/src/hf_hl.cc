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
      h_hist, rt_bklen, buf->h_bk4.get(), buf->h_revbk4.get(), buf->revbk4_bytes);
  memcpy_allkinds_async<H2D>(buf->d_bk4.get(), buf->h_bk4.get(), rt_bklen, (cudaStream_t)stream);
  memcpy_allkinds_async<H2D>(
      buf->d_revbk4.get(), buf->h_revbk4.get(), buf->revbk4_bytes, (cudaStream_t)stream);

  return 0;
}

template <typename E>
int high_level<E>::encode(
    HF_SPACE* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    HF_STREAM stream)
{
  auto make_metadata = [](HF_SPACE* buf, phf_header& header) {
    // header.self_bytes = sizeof(Header);
    header.bklen = buf->rt_bklen;
    header.sublen = buf->sublen;
    header.pardeg = buf->pardeg;
    header.original_len = buf->len;

    M nbyte[PHFHEADER_END];
    nbyte[PHFHEADER_HEADER] = PHFHEADER_FORCED_ALIGN;
    nbyte[PHFHEADER_REVBK] = buf->revbk4_bytes;
    nbyte[PHFHEADER_PAR_NBIT] = buf->pardeg * sizeof(M);
    nbyte[PHFHEADER_PAR_ENTRY] = buf->pardeg * sizeof(M);
    nbyte[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;

    header.entry[0] = 0;
    // *.END + 1: need to know the ending position
    for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = nbyte[i - 1];
    for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] += header.entry[i - 1];
  };

  phf_module<E>::GPU_coarse_encode(
      in, len, buf->d_bk4.get(), buf->rt_bklen, buf->numSMs, {buf->sublen, buf->pardeg},
      // internal buffers
      buf->d_scratch4.get(), buf->d_par_nbit.get(), buf->h_par_nbit.get(), buf->d_par_ncell.get(),
      buf->h_par_ncell.get(), buf->d_par_entry.get(), buf->h_par_entry.get(),
      buf->d_bitstream4.get(), buf->bitstream_max_len,
      // output
      &header.total_nbit, &header.total_ncell, stream);

  sync_by_stream(stream);

  make_metadata(buf, header);
  buf->memcpy_merge(header, stream);  // TODO externalize/make explicit

  *out = buf->d_encoded;
  *outlen = phf_encoded_bytes(&header);

  return 0;
}

template <typename E>
int high_level<E>::encode_ReVISIT_lite(
    HF_SPACE* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    phf_stream_t stream)
{
  auto make_metadata_ReVISIT_lite = [](HF_SPACE* buf, phf_header& header) {
    // header.self_bytes = sizeof(Header);
    header.bklen = buf->rt_bklen;
    header.sublen = buf->sublen;
    header.pardeg = buf->pardeg;
    header.original_len = buf->len;

    M nbyte[PHFHEADER_END];
    nbyte[PHFHEADER_HEADER] = PHFHEADER_FORCED_ALIGN;
    nbyte[PHFHEADER_REVBK] = buf->revbk4_bytes;
    nbyte[PHFHEADER_PAR_NBIT] = buf->pardeg * sizeof(M);
    nbyte[PHFHEADER_PAR_ENTRY] = buf->pardeg * sizeof(M);
    nbyte[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;

    header.entry[0] = 0;
    // *.END + 1: need to know the ending position
    for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = nbyte[i - 1];
    for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] += header.entry[i - 1];
  };

  phf_module<E>::GPU_fine_encode(
      in, len, buf->d_bk4.get(), buf->rt_bklen, {buf->sublen, buf->pardeg}, buf->d_scratch4.get(),
      buf->d_par_nbit.get(), buf->h_par_nbit.get(), buf->d_par_ncell.get(), buf->h_par_ncell.get(),
      buf->d_par_entry.get(), buf->h_par_entry.get(), buf->d_bitstream4.get(),
      buf->bitstream_max_len, buf->d_brval.get(), buf->d_bridx.get(), buf->d_brnum.get(),
      &header.total_nbit, &header.total_ncell, stream);
  sync_by_stream(stream);

  make_metadata_ReVISIT_lite(buf, header);
  buf->memcpy_merge(header, stream);

  *out = buf->d_encoded;
  *outlen = phf_encoded_bytes(&header);

  return 0;
}

#define PHF_ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_encoded + header.entry[PHFHEADER_##SYM])

template <typename E>
int high_level<E>::decode(
    HF_SPACE* buf, phf_header& header, uint8_t* in_encoded, E* out_decoded, HF_STREAM stream)
{
  phf_module<E>::GPU_coarse_decode(
      PHF_ACCESSOR(BITSTREAM, H4), PHF_ACCESSOR(REVBK, PHF_BYTE), buf->revbk4_bytes,
      PHF_ACCESSOR(PAR_NBIT, M), PHF_ACCESSOR(PAR_ENTRY, M), header.sublen, header.pardeg,
      out_decoded, stream);

  return 0;
}

}  // namespace phf

template struct phf::high_level<u1>;
template struct phf::high_level<u2>;
template struct phf::high_level<u4>;

#undef PHF_ACCESSOR