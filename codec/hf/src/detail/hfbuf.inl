#include <cuda.h>

#include <cstddef>

#include "hf.h"
#include "hfclass.hh"
#include "mem/cxx_memobj.h"
#include "phf_array.hh"
#include "utils/err.hh"

namespace phf {

template <typename T>
using memobj = _portable::memobj<T>;

template <typename E>
struct HuffmanCodec<E>::Buf {
  // helper
  typedef struct RC {
    static const int SCRATCH = 0;
    static const int FREQ = 1;
    static const int BK = 2;
    static const int REVBK = 3;
    static const int PAR_NBIT = 4;
    static const int PAR_NCELL = 5;
    static const int PAR_ENTRY = 6;
    static const int BITSTREAM = 7;
    static const int END = 8;
    // uint32_t nbyte[END];
  } RC;

  typedef struct {
    void* const ptr;
    size_t const nbyte;
    size_t const dst;
  } memcpy_helper;

  using SYM = E;
  using Header = phf_header;

  // vars
  const size_t len;
  const size_t pardeg;
  const size_t sublen;
  const size_t bklen;
  const size_t revbk4_bytes;
  const size_t bitstream_max_len;

  static const int HFR_Chunksize = 1 << HFR_Magnitude;
  const bool HFR_in_use;
  const int HFR_nchunk;

  // array
  GPU_unique_dptr<H4[]> d_scratch4;
  GPU_unique_hptr<H4[]> h_scratch4;
  PHF_BYTE* d_encoded;
  PHF_BYTE* h_encoded;
  GPU_unique_dptr<H4[]> d_bitstream4;
  GPU_unique_hptr<H4[]> h_bitstream4;

  GPU_unique_dptr<H4[]> d_bk4;
  GPU_unique_hptr<H4[]> h_bk4;
  GPU_unique_dptr<PHF_BYTE[]> d_revbk4;
  GPU_unique_hptr<PHF_BYTE[]> h_revbk4;

  // data partition/embarrassingly parallelism description
  GPU_unique_dptr<M[]> d_par_nbit;
  GPU_unique_hptr<M[]> h_par_nbit;
  GPU_unique_dptr<M[]> d_par_ncell;
  GPU_unique_hptr<M[]> h_par_ncell;
  GPU_unique_dptr<M[]> d_par_entry;
  GPU_unique_hptr<M[]> h_par_entry;

  // dense-sparse
  GPU_unique_dptr<H4[]> d_dn_bitstream;
  GPU_unique_dptr<u4[]> d_dn_bitcount;
  GPU_unique_dptr<M[]> d_dn_start_loc;
  GPU_unique_uptr<M[]> u_dn_loc_inc;

  memobj<E>* sp_val;
  memobj<M>* sp_idx;
  memobj<M>* sp_num;

 private:
  static int _revbk4_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 4, sizeof(SYM)); }
  static int _revbk8_bytes(int bklen) { return phf_reverse_book_bytes(bklen, 8, sizeof(SYM)); }

 public:
  // auxiliary
  // TODO standalone tool
  void _debug(const std::string SYM_name, void* VAR, int SYM)
  {
    CUdeviceptr pbase0{0};
    size_t psize0{0};

    cuMemGetAddressRange(&pbase0, &psize0, (CUdeviceptr)VAR);
    printf(
        "%s:\n"
        "\t(supposed) pointer : %p\n"
        "\t(queried)  pbase0  : %p\n"
        "\t(queried)  psize0  : %'9lu\n",
        SYM_name.c_str(), (void*)VAR, (void*)&pbase0, psize0);
    pbase0 = 0, psize0 = 0;
  }

  void debug_all()
  {
    setlocale(LC_NUMERIC, "");
    printf("\nHuffmanCoarse<E, H4, M>::init() debugging:\n");
    printf("CUdeviceptr nbyte: %d\n", (int)sizeof(CUdeviceptr));
    _debug("SCRATCH", d_scratch4.get(), RC::SCRATCH);
    _debug("BITSTREAM", d_bitstream4.get(), RC::BITSTREAM);
    _debug("PAR_NBIT", d_par_nbit.get(), RC::PAR_NBIT);
    _debug("PAR_NCELL", d_par_ncell.get(), RC::PAR_NCELL);
    printf("\n");
  };

  // ctor
  Buf(size_t inlen, size_t _bklen, int _pardeg, bool _use_HFR = false, bool debug = false) :
      len(inlen),
      bitstream_max_len(inlen / 2),
      pardeg(_pardeg),
      sublen((inlen - 1) / _pardeg + 1),
      bklen(_bklen),
      revbk4_bytes(_revbk4_bytes(_bklen)),
      HFR_in_use(_use_HFR),
      HFR_nchunk((inlen + HFR_Chunksize - 1) / HFR_Chunksize)
  {
    h_scratch4 = MAKE_UNIQUE_HOST(H4, len);
    d_scratch4 = MAKE_UNIQUE_DEVICE(H4, len);
    h_bk4 = MAKE_UNIQUE_HOST(H4, bklen);
    d_bk4 = MAKE_UNIQUE_DEVICE(H4, bklen);
    h_revbk4 = MAKE_UNIQUE_HOST(PHF_BYTE, revbk4_bytes);
    d_revbk4 = MAKE_UNIQUE_DEVICE(PHF_BYTE, revbk4_bytes);

    if (not HFR_in_use) {
      d_bitstream4 = MAKE_UNIQUE_DEVICE(H4, bitstream_max_len);
      h_bitstream4 = MAKE_UNIQUE_HOST(H4, bitstream_max_len);
      h_par_nbit = MAKE_UNIQUE_HOST(M, pardeg);
      d_par_nbit = MAKE_UNIQUE_DEVICE(M, pardeg);
      h_par_ncell = MAKE_UNIQUE_HOST(M, pardeg);
      d_par_ncell = MAKE_UNIQUE_DEVICE(M, pardeg);
      h_par_entry = MAKE_UNIQUE_HOST(M, pardeg);
      d_par_entry = MAKE_UNIQUE_DEVICE(M, pardeg);
    }
    else {  // HFR: dense-sparse
      d_dn_bitstream = MAKE_UNIQUE_DEVICE(H4, len / 2);
      d_dn_bitcount = MAKE_UNIQUE_DEVICE(u4, HFR_nchunk);
      d_dn_start_loc = MAKE_UNIQUE_DEVICE(M, HFR_nchunk);

      // After the kernel run, the final value is the ending loc in u4,
      // determining the CR, such that MallocHost is needed.
      u_dn_loc_inc = MAKE_UNIQUE_UNIFIED(M, 1);

      sp_val = new memobj<E>(len / 10, "hf::sp_val", {Malloc});
      sp_idx = new memobj<M>(len / 10, "hf::sp_idx", {Malloc});
      sp_num = new memobj<M>(1, "hf::sp_num", {Malloc, MallocHost});
    }

    // repurpose scratch after several substeps
    d_encoded = (u1*)d_scratch4.get();
    h_encoded = (u1*)h_scratch4.get();

    if (debug) debug_all();
  }

  ~Buf()
  {
    if (HFR_in_use) {
      delete sp_val;
      delete sp_idx;
      delete sp_num;
    }
  }

  ::_portable::compact_array1<E> sparse_space()
  {
    return {sp_val->dptr(), sp_idx->dptr(), sp_num->dptr(), sp_num->hptr(), sp_val->len()};
  }

  phf::dense<H> dense_space(size_t HFR_nchunk)
  {
    return {
        d_dn_bitstream.get(), d_dn_bitcount.get(), d_dn_start_loc.get(), u_dn_loc_inc.get(),
        HFR_nchunk};
  }

  void memcpy_merge(Header& header, phf_stream_t stream)
  {
    auto memcpy_start = d_encoded;
    auto memcpy_adjust_to_start = 0;

    memcpy_helper _revbk{d_revbk4.get(), revbk4_bytes, header.entry[PHFHEADER_REVBK]};

    auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);
    auto d2d_memcpy_merge = [&](memcpy_helper& var) {
      CHECK_GPU(cudaMemcpyAsync(
          start + var.dst, var.ptr, var.nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
    };

    CHECK_GPU(cudaMemcpyAsync(
        start, &header, sizeof(header), cudaMemcpyHostToDevice, (cudaStream_t)stream));

    // for both non-HFR and HFR
    d2d_memcpy_merge(_revbk);

    if (not HFR_in_use) {
      memcpy_helper _par_nbit{
          d_par_nbit.get(), pardeg * sizeof(M), header.entry[PHFHEADER_PAR_NBIT]};
      memcpy_helper _par_entry{
          d_par_entry.get(), pardeg * sizeof(M), header.entry[PHFHEADER_PAR_ENTRY]};
      memcpy_helper _bitstream{
          d_bitstream4.get(), bitstream_max_len * sizeof(H4), header.entry[PHFHEADER_BITSTREAM]};

      d2d_memcpy_merge(_par_nbit);
      d2d_memcpy_merge(_par_entry);
      d2d_memcpy_merge(_bitstream);
    }
    else {
      auto total_len = u_dn_loc_inc[0];
      // TODO
    }
  }

  void clear_buffer()
  {
    memset_device(d_scratch4.get(), len);
    memset_device(d_bk4.get(), bklen);
    memset_device(d_revbk4.get(), revbk4_bytes);
    memset_device(d_bitstream4.get(), bitstream_max_len);
    memset_device(d_par_nbit.get(), pardeg);
    memset_device(d_par_ncell.get(), pardeg);
    memset_device(d_par_entry.get(), pardeg);
  }
};

}  // namespace phf