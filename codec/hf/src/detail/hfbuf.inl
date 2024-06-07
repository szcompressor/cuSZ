#include <cuda.h>

#include <cstddef>

#include "hfclass.hh"
#include "hfcxx_array.hh"
#include "mem/cxx_memobj.h"
#include "utils/err.hh"

namespace phf {

template <typename T>
using memobj = _portable::memobj<T>;

template <typename E, bool TIMING>
struct HuffmanCodec<E, TIMING>::Buf {
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
  const size_t bklen;
  size_t pardeg;
  size_t sublen;

  static const int HFR_Chunksize = 1 << HFR_Magnitude;
  const bool HFR_in_use;
  const int HFR_nchunk;

  // array
  memobj<H4>* scratch4;
  memobj<PHF_BYTE>* encoded;
  memobj<H4>* bk4;
  memobj<PHF_BYTE>* revbk4;
  memobj<H4>* bitstream4;

  // data partition/embarrassingly parallelism description
  memobj<M>* par_nbit;
  memobj<M>* par_ncell;
  memobj<M>* par_entry;

  // dense-sparse
  memobj<H4>* dn_bitstream;
  memobj<u2>* dn_bitcount;
  memobj<M>* dn_start_loc;
  memobj<M>* dn_loc_inc;
  memobj<E>* sp_val;
  memobj<M>* sp_idx;
  memobj<M>* sp_num;

  static int __revbk_bytes(int bklen, int BK_UNIT_BYTES, int SYM_BYTES)
  {
    static const int CELL_BITWIDTH = BK_UNIT_BYTES * 8;
    return BK_UNIT_BYTES * (2 * CELL_BITWIDTH) + SYM_BYTES * bklen;
  }

  static int revbk4_bytes(int bklen) { return __revbk_bytes(bklen, 4, sizeof(SYM)); }
  static int revbk8_bytes(int bklen) { return __revbk_bytes(bklen, 8, sizeof(SYM)); }

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
    _debug("SCRATCH", scratch4->dptr(), RC::SCRATCH);
    _debug("BITSTREAM", bitstream4->dptr(), RC::BITSTREAM);
    _debug("PAR_NBIT", par_nbit->dptr(), RC::PAR_NBIT);
    _debug("PAR_NCELL", par_ncell->dptr(), RC::PAR_NCELL);
    printf("\n");
  };

  // ctor
  Buf(size_t inlen, size_t _booklen, int _pardeg, bool _use_HFR = false,
      bool debug = false) :
      len(inlen),
      bklen(_booklen),
      HFR_nchunk((inlen + HFR_Chunksize - 1) / HFR_Chunksize),
      HFR_in_use(_use_HFR)
  {
    pardeg = _pardeg;

    encoded = new memobj<PHF_BYTE>(len * sizeof(u4), "hf::out4B");
    scratch4 = new memobj<H4>(len, "hf::scratch4", {Malloc, MallocHost});
    bk4 = new memobj<H4>(bklen, "hf::book4", {Malloc, MallocHost});
    revbk4 = new memobj<PHF_BYTE>(
        revbk4_bytes(bklen), "hf::revbk4", {Malloc, MallocHost});

    if (not HFR_in_use) {
      bitstream4 =
          new memobj<H4>(len / 2, "hf::enc-buf", {Malloc, MallocHost});
      par_nbit = new memobj<M>(pardeg, "hf::par_nbit", {Malloc, MallocHost});
      par_ncell = new memobj<M>(pardeg, "hf::par_ncell", {Malloc, MallocHost});
      par_entry = new memobj<M>(pardeg, "hf::par_entry", {Malloc, MallocHost});
    }
    else {  // HFR: dense-sparse
      dn_bitstream = new memobj<H4>(len / 2, "hf::dn_bitstream", {Malloc});
      dn_bitcount = new memobj<u2>(HFR_nchunk, "hf::dn_bitcount", {Malloc});
      dn_start_loc = new memobj<M>(HFR_nchunk, "hf::dn_start_loc", {Malloc});
      // After the kernel run, the final value is the ending loc in u4,
      // determining the CR, such that MallocHost is needed.
      dn_loc_inc = new memobj<M>(1, "hf::dn_loc_inc", {Malloc, MallocHost});

      sp_val = new memobj<E>(len / 10, "hf::sp_val", {Malloc});
      sp_idx = new memobj<M>(len / 10, "hf::sp_idx", {Malloc});
      sp_num = new memobj<M>(1, "hf::sp_num", {Malloc, MallocHost});
    }

    // repurpose scratch after several substeps
    encoded->dptr((u1*)scratch4->dptr())->hptr((u1*)scratch4->hptr());
    // cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    sublen = (inlen - 1) / pardeg + 1;

    if (debug) debug_all();
  }

  ~Buf()
  {
    delete bk4;
    delete revbk4;
    delete encoded;
    delete scratch4;

    if (not HFR_in_use) {
      delete par_nbit;
      delete par_ncell;
      delete par_entry;
      delete bitstream4;
    }
    else {
      delete dn_bitstream;
      delete dn_bitcount;
      delete dn_start_loc;
      delete dn_loc_inc;

      delete sp_val;
      delete sp_idx;
      delete sp_num;
    }
  }

  ::_portable::compact_array1<E> sparse_space()
  {
    return {sp_val->dptr(), sp_idx->dptr(), sp_num->dptr(), sp_num->hptr(), sp_val->len()};
  }

  // TODO input arg redundant
  hfcxx_dense<H> dense_space(size_t HFR_nchunk)
  {
    return {
        dn_bitstream->dptr(), dn_bitcount->dptr(), dn_start_loc->dptr(),
        dn_loc_inc->dptr(), HFR_nchunk};
  }

  void memcpy_merge(Header& header, phf_stream_t stream)
  {
    auto memcpy_start = encoded->dptr();
    auto memcpy_adjust_to_start = 0;

    memcpy_helper _revbk{
        revbk4->dptr(), revbk4->bytes(), header.entry[PHFHEADER_REVBK]};

    auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);
    auto d2d_memcpy_merge = [&](memcpy_helper& var) {
      CHECK_GPU(cudaMemcpyAsync(
          start + var.dst, var.ptr, var.nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
    };

    CHECK_GPU(cudaMemcpyAsync(
        start, &header, sizeof(header), cudaMemcpyHostToDevice, (cudaStream_t)stream));

    if (not HFR_in_use) {
      memcpy_helper _par_nbit{
          par_nbit->dptr(), par_nbit->bytes(),
          header.entry[PHFHEADER_PAR_NBIT]};
      memcpy_helper _par_entry{
          par_entry->dptr(), par_entry->bytes(),
          header.entry[PHFHEADER_PAR_ENTRY]};
      memcpy_helper _bitstream{
          bitstream4->dptr(), bitstream4->bytes(),
          header.entry[PHFHEADER_BITSTREAM]};

      d2d_memcpy_merge(_revbk);
      d2d_memcpy_merge(_par_nbit);
      d2d_memcpy_merge(_par_entry);
      d2d_memcpy_merge(_bitstream);
    }
    else {
      auto total_len = dn_loc_inc->control({D2H})->hat(0);

      memcpy_helper _par_nbit{
          par_nbit->dptr(), par_nbit->bytes(),
          header.entry[PHFHEADER_PAR_NBIT]};
      memcpy_helper _par_entry{
          par_entry->dptr(), par_entry->bytes(),
          header.entry[PHFHEADER_PAR_ENTRY]};
      memcpy_helper _bitstream{
          dn_bitstream->dptr(), dn_bitstream->bytes(),
          header.entry[PHFHEADER_HFR_DN_BITSTREAM]};

      d2d_memcpy_merge(_revbk);
    }
  }

  void clear_buffer()
  {
    scratch4->control({ClearDevice});
    bk4->control({ClearDevice});
    revbk4->control({ClearDevice});
    bitstream4->control({ClearDevice});

    par_nbit->control({ClearDevice});
    par_ncell->control({ClearDevice});
    par_entry->control({ClearDevice});
  }
};

}  // namespace phf