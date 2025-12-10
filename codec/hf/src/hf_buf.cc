#include <cuda.h>

#include "hf.h"
#include "hf_hl.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"

namespace phf {

template <typename E>
struct Buf<E>::impl {
  // types
  using H4 = u4;
  using M = PHF_METADATA;
  using SYM = E;
  using Header = phf_header;

  // helper struct(s)
  typedef struct {
    void* const ptr;
    size_t const nbyte;
    size_t const dst;
  } memcpy_helper;

  // vars
  const size_t len;
  const size_t bklen;
  const size_t rvbk4_bytes;
  const size_t bitstream_max_len;
  size_t pardeg;
  size_t sublen;
  bool use_HFR;
  u2 rt_bklen;
  int numSMs;

  // internal arrays: data
  GPU_unique_dptr<H4[]> d_scratch4;
  GPU_unique_hptr<H4[]> h_scratch4;
  PHF_BYTE* d_encoded;
  PHF_BYTE* h_encoded;
  GPU_unique_dptr<H4[]> d_bitstream4;
  GPU_unique_hptr<H4[]> h_bitstream4;

  GPU_unique_dptr<H4[]> d_book4;
  GPU_unique_hptr<H4[]> h_book4;
  GPU_unique_dptr<PHF_BYTE[]> d_rvbk4;
  GPU_unique_hptr<PHF_BYTE[]> h_rvbk4;

  // internal arrays: metadata for data partitions
  GPU_unique_dptr<M[]> d_par_nbit;
  GPU_unique_hptr<M[]> h_par_nbit;
  GPU_unique_dptr<M[]> d_par_ncell;
  GPU_unique_hptr<M[]> h_par_ncell;
  GPU_unique_dptr<M[]> d_par_entry;
  GPU_unique_hptr<M[]> h_par_entry;

  // internal arrays: ReVISIT-lite specific
  GPU_unique_dptr<E[]> d_brval;
  GPU_unique_dptr<u4[]> d_bridx;
  GPU_unique_dptr<u4[]> d_brnum;
  GPU_unique_hptr<u4[]> h_brnum;

  // internal functions
  int _rvbk4_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 4, sizeof(SYM)); }
  int _rvbk8_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 8, sizeof(SYM)); }

  // constructor
  impl(size_t inlen, size_t _bklen, int _pardeg, bool _use_HFR, bool debug) :
      len(inlen),
      bklen(_bklen),
      bitstream_max_len(inlen / 2),
      use_HFR(_use_HFR),
      rvbk4_bytes(_rvbk4_bytes(_bklen))
  {
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    sublen = use_HFR ? 1024 : phf_coarse_tune_sublen(inlen);
    pardeg = (inlen - 1) / sublen + 1;
    // cout << sublen << " " << pardeg << endl;

    h_scratch4 = MAKE_UNIQUE_HOST(H4, len);
    d_scratch4 = MAKE_UNIQUE_DEVICE(H4, len);
    h_book4 = MAKE_UNIQUE_HOST(H4, bklen);
    d_book4 = MAKE_UNIQUE_DEVICE(H4, bklen);
    h_rvbk4 = MAKE_UNIQUE_HOST(PHF_BYTE, rvbk4_bytes);
    d_rvbk4 = MAKE_UNIQUE_DEVICE(PHF_BYTE, rvbk4_bytes);
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

  // destructor
  ~impl() {}

  // public functions
  void memcpy_merge(Header& header, phf_stream_t stream)

  {
    auto memcpy_start = d_encoded;
    auto memcpy_adjust_to_start = 0;

    memcpy_helper _rvbk{d_rvbk4.get(), rvbk4_bytes, header.entry[PHFHEADER_RVBK]};
    memcpy_helper _par_nbit{
        d_par_nbit.get(), pardeg * sizeof(M), header.entry[PHFHEADER_PAR_NBIT]};
    memcpy_helper _par_entry{
        d_par_entry.get(), pardeg * sizeof(M), header.entry[PHFHEADER_PAR_ENTRY]};
    memcpy_helper _bitstream{
        d_bitstream4.get(), bitstream_max_len * sizeof(H4), header.entry[PHFHEADER_BITSTREAM]};

    auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);
    auto d2d_memcpy_merge = [&](memcpy_helper& var) {
      cudaMemcpyAsync(
          start + var.dst, var.ptr, var.nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    };

    cudaMemcpyAsync(start, &header, sizeof(header), cudaMemcpyHostToDevice, (cudaStream_t)stream);

    // /* debug */ CHECK_GPU(cudaStreamSynchronize(stream));
    d2d_memcpy_merge(_rvbk);
    d2d_memcpy_merge(_par_nbit);
    d2d_memcpy_merge(_par_entry);
    d2d_memcpy_merge(_bitstream);
    // /* debug */ CHECK_GPU(cudaStreamSynchronize(stream));
  }

  void clear_buffer()
  {
    memset_device(d_scratch4.get(), len);
    memset_device(d_book4.get(), bklen);
    memset_device(d_rvbk4.get(), rvbk4_bytes);
    memset_device(d_bitstream4.get(), bitstream_max_len);
    memset_device(d_par_nbit.get(), pardeg);
    memset_device(d_par_ncell.get(), pardeg);
    memset_device(d_par_entry.get(), pardeg);
  }
};

#define PHF_BUF_DEF(RET_TYPE) \
  template <typename E>       \
  RET_TYPE phf::Buf<E>

PHF_BUF_DEF()::Buf(size_t inlen, size_t _bklen, int _pardeg, bool _use_HFR, bool debug) :
    pimpl(std::make_unique<impl>(inlen, _bklen, _pardeg, _use_HFR, debug))
{
}

PHF_BUF_DEF()::~Buf() {}

// a series of getters: variables
PHF_BUF_DEF(size_t)::rvbk_bytes() const { return pimpl->rvbk4_bytes; }
PHF_BUF_DEF(u2)::rt_bklen() const { return pimpl->rt_bklen; }
PHF_BUF_DEF(int)::numSMs() const { return pimpl->numSMs; }
PHF_BUF_DEF(size_t)::sublen() const { return pimpl->sublen; }
PHF_BUF_DEF(size_t)::pardeg() const { return pimpl->pardeg; }
PHF_BUF_DEF(size_t)::bitstream_max_len() const { return pimpl->bitstream_max_len; }

// a series of getters: arrays
PHF_BUF_DEF(H4*)::book_d() const { return pimpl->d_book4.get(); }
PHF_BUF_DEF(H4*)::book_h() const { return pimpl->h_book4.get(); }
PHF_BUF_DEF(u1*)::rvbk_d() const { return pimpl->d_rvbk4.get(); }
PHF_BUF_DEF(u1*)::rvbk_h() const { return pimpl->h_rvbk4.get(); }
PHF_BUF_DEF(H4*)::scratch_d() const { return pimpl->d_scratch4.get(); }
PHF_BUF_DEF(H4*)::scratch_h() const { return pimpl->h_scratch4.get(); }
PHF_BUF_DEF(M*)::par_nbit_d() const { return pimpl->d_par_nbit.get(); }
PHF_BUF_DEF(M*)::par_nbit_h() const { return pimpl->h_par_nbit.get(); }
PHF_BUF_DEF(M*)::par_ncell_d() const { return pimpl->d_par_ncell.get(); }
PHF_BUF_DEF(M*)::par_ncell_h() const { return pimpl->h_par_ncell.get(); }
PHF_BUF_DEF(M*)::par_entry_d() const { return pimpl->d_par_entry.get(); }
PHF_BUF_DEF(M*)::par_entry_h() const { return pimpl->h_par_entry.get(); }
PHF_BUF_DEF(H4*)::bitstream_d() const { return pimpl->d_bitstream4.get(); }
PHF_BUF_DEF(H4*)::bitstream_h() const { return pimpl->h_bitstream4.get(); }
PHF_BUF_DEF(PHF_BYTE*)::encoded_d() const { return pimpl->d_encoded; }
PHF_BUF_DEF(PHF_BYTE*)::encoded_h() const { return pimpl->h_encoded; }

// method
PHF_BUF_DEF(void)::update_header(phf_header& header)
{
  header.bklen = pimpl->rt_bklen;
  header.sublen = pimpl->sublen;
  header.pardeg = pimpl->pardeg;
  header.original_len = pimpl->len;
}

PHF_BUF_DEF(void)::calc_offset(phf_header& header, M* byte_offsets)
{
  byte_offsets[PHFHEADER_HEADER] = PHFHEADER_FORCED_ALIGN;
  byte_offsets[PHFHEADER_RVBK] = rvbk_bytes();
  byte_offsets[PHFHEADER_PAR_NBIT] = pimpl->pardeg * sizeof(M);
  byte_offsets[PHFHEADER_PAR_ENTRY] = pimpl->pardeg * sizeof(M);
  byte_offsets[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;

  header.entry[0] = 0;
  // *.END + 1: need to know the ending position
  for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = byte_offsets[i - 1];
  for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] += header.entry[i - 1];
}

// method: set internal variable
PHF_BUF_DEF(void)::register_runtime_bklen(const int _rt_bklen) { pimpl->rt_bklen = _rt_bklen; }

// method, same-name
PHF_BUF_DEF(void)::memcpy_merge(phf_header& header, phf_stream_t stream)
{
  pimpl->memcpy_merge(header, stream);
}

// method, same-name
PHF_BUF_DEF(void)::clear_buffer() { pimpl->clear_buffer(); }

}  // namespace phf

template struct phf::Buf<u1>;
template struct phf::Buf<u2>;
template struct phf::Buf<u4>;

#undef PHF_BUF_DEF