#include "hf_buf.hh"

#include <cuda.h>

#include <cstddef>

#include "hf.h"
#include "hf_ood.hh"
#include "phf_array.hh"
#include "utils/err.hh"

namespace phf {

template <typename E>
int Buf<E>::_revbk4_bytes(int bklen)
{
  return phf_reverse_book_bytes(bklen, 4, sizeof(SYM));
}

template <typename E>
int Buf<E>::_revbk8_bytes(int bklen)
{
  return phf_reverse_book_bytes(bklen, 8, sizeof(SYM));
}

template <typename E>
Buf<E>::Buf(size_t inlen, size_t _bklen, int _pardeg, bool _use_HFR, bool debug) :
    len(inlen),
    bklen(_bklen),
    bitstream_max_len(inlen / 2),
    use_HFR(_use_HFR),
    revbk4_bytes(_revbk4_bytes(_bklen))
{
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

  sublen = use_HFR ? 1024 : capi_phf_coarse_tune_sublen(inlen);
  pardeg = (inlen - 1) / sublen + 1;
  // cout << sublen << " " << pardeg << endl;

  h_scratch4 = MAKE_UNIQUE_HOST(H4, len);
  d_scratch4 = MAKE_UNIQUE_DEVICE(H4, len);
  h_bk4 = MAKE_UNIQUE_HOST(H4, bklen);
  d_bk4 = MAKE_UNIQUE_DEVICE(H4, bklen);
  h_revbk4 = MAKE_UNIQUE_HOST(PHF_BYTE, revbk4_bytes);
  d_revbk4 = MAKE_UNIQUE_DEVICE(PHF_BYTE, revbk4_bytes);
  d_bitstream4 = MAKE_UNIQUE_DEVICE(H4, bitstream_max_len);
  h_bitstream4 = MAKE_UNIQUE_HOST(H4, bitstream_max_len);
  h_par_nbit = MAKE_UNIQUE_HOST(M, pardeg);
  d_par_nbit = MAKE_UNIQUE_DEVICE(M, pardeg);
  h_par_ncell = MAKE_UNIQUE_HOST(M, pardeg);
  d_par_ncell = MAKE_UNIQUE_DEVICE(M, pardeg);
  h_par_entry = MAKE_UNIQUE_HOST(M, pardeg);
  d_par_entry = MAKE_UNIQUE_DEVICE(M, pardeg);

  // ReVISIT-lite specific
  d_brval = MAKE_UNIQUE_DEVICE(E, 100 + len / 10 + 1);  // len / 10 is a heuristic
  d_bridx = MAKE_UNIQUE_DEVICE(u4, 100 + len / 10 + 1);
  d_brnum = MAKE_UNIQUE_DEVICE(u4, 1);
  h_brnum = MAKE_UNIQUE_HOST(u4, 1);

  // repurpose scratch after several substeps
  d_encoded = (u1*)d_scratch4.get();
  h_encoded = (u1*)h_scratch4.get();
}

template <typename E>
Buf<E>::~Buf()
{
}

template <typename E>
void Buf<E>::memcpy_merge(Header& header, phf_stream_t stream)

{
  auto memcpy_start = d_encoded;
  auto memcpy_adjust_to_start = 0;

  memcpy_helper _revbk{d_revbk4.get(), revbk4_bytes, header.entry[PHFHEADER_REVBK]};
  memcpy_helper _par_nbit{d_par_nbit.get(), pardeg * sizeof(M), header.entry[PHFHEADER_PAR_NBIT]};
  memcpy_helper _par_entry{
      d_par_entry.get(), pardeg * sizeof(M), header.entry[PHFHEADER_PAR_ENTRY]};
  memcpy_helper _bitstream{
      d_bitstream4.get(), bitstream_max_len * sizeof(H4), header.entry[PHFHEADER_BITSTREAM]};

  auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);
  auto d2d_memcpy_merge = [&](memcpy_helper& var) {
    CHECK_GPU(cudaMemcpyAsync(
        start + var.dst, var.ptr, var.nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
  };

  CHECK_GPU(cudaMemcpyAsync(
      start, &header, sizeof(header), cudaMemcpyHostToDevice, (cudaStream_t)stream));

  // /* debug */ CHECK_GPU(cudaStreamSynchronize(stream));
  d2d_memcpy_merge(_revbk);
  d2d_memcpy_merge(_par_nbit);
  d2d_memcpy_merge(_par_entry);
  d2d_memcpy_merge(_bitstream);
  // /* debug */ CHECK_GPU(cudaStreamSynchronize(stream));
}

template <typename E>
void Buf<E>::clear_buffer()
{
  memset_device(d_scratch4.get(), len);
  memset_device(d_bk4.get(), bklen);
  memset_device(d_revbk4.get(), revbk4_bytes);
  memset_device(d_bitstream4.get(), bitstream_max_len);
  memset_device(d_par_nbit.get(), pardeg);
  memset_device(d_par_ncell.get(), pardeg);
  memset_device(d_par_entry.get(), pardeg);
}

}  // namespace phf

template struct phf::Buf<u1>;
template struct phf::Buf<u2>;
template struct phf::Buf<u4>;