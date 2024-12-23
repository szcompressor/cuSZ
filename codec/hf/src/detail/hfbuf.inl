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
  size_t len;
  size_t pardeg;
  size_t sublen;
  size_t bklen;
  bool use_HFR;

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
  memobj<M>* dn_bitcount;
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
  Buf(size_t inlen, size_t _booklen, int _pardeg, bool _use_HFR = false, bool debug = false)
  {
    pardeg = _pardeg;
    bklen = _booklen;
    len = inlen;
    use_HFR = _use_HFR;

    encoded = new memobj<PHF_BYTE>(len * sizeof(u4), "hf::out4B");
    scratch4 = new memobj<H4>(len, "hf::scratch4", {Malloc, MallocHost});
    bk4 = new memobj<H4>(bklen, "hf::book4", {Malloc, MallocHost});
    revbk4 = new memobj<PHF_BYTE>(revbk4_bytes(bklen), "hf::revbk4", {Malloc, MallocHost});
    bitstream4 = new memobj<H4>(len / 2, "hf::enc-buf", {Malloc, MallocHost});
    par_nbit = new memobj<M>(pardeg, "hf::par_nbit", {Malloc, MallocHost});
    par_ncell = new memobj<M>(pardeg, "hf::par_ncell", {Malloc, MallocHost});
    par_entry = new memobj<M>(pardeg, "hf::par_entry", {Malloc, MallocHost});

    // HFR: dense-sparse
    if (use_HFR) {
      dn_bitstream = new memobj<H4>(len / 2, "hf::dn_bitstream", {Malloc});
      // 1 << 10 results in the max number of partitions
      dn_bitcount = new memobj<H4>((len - 1) / (1 << 10) + 1, "hf::dn_bitcount", {Malloc});
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
    delete par_nbit;
    delete par_ncell;
    delete par_entry;

    delete encoded;
    delete scratch4;
    delete bitstream4;

    if (use_HFR) {
      delete sp_val;
      delete sp_idx;
      delete sp_num;
    }
  }

  ::_portable::compact_array1<E> sparse_space()
  {
    return {sp_val->dptr(), sp_idx->dptr(), sp_num->dptr(), sp_num->hptr(), sp_val->len()};
  }

  hfcxx_dense<H> dense_space(size_t n_chunk)
  {
    return {dn_bitstream->dptr(), dn_bitcount->dptr(), n_chunk};
  }

  void memcpy_merge(Header& header, phf_stream_t stream)
  {
    auto memcpy_start = encoded->dptr();
    auto memcpy_adjust_to_start = 0;

    memcpy_helper _revbk{revbk4->dptr(), revbk4->bytes(), header.entry[PHFHEADER_REVBK]};
    memcpy_helper _par_nbit{par_nbit->dptr(), par_nbit->bytes(), header.entry[PHFHEADER_PAR_NBIT]};
    memcpy_helper _par_entry{
        par_entry->dptr(), par_entry->bytes(), header.entry[PHFHEADER_PAR_ENTRY]};
    memcpy_helper _bitstream{
        bitstream4->dptr(), bitstream4->bytes(), header.entry[PHFHEADER_BITSTREAM]};

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