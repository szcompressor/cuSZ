/**
 * @file hfclass.cu_hip.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2023-06-13
 * (created) 2020-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National
 * Laboratory See LICENSE in top-level directory
 *
 */

#ifndef ABBC78E4_3E65_4633_9BEA_27823AB7C398
#define ABBC78E4_3E65_4633_9BEA_27823AB7C398

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "busyheader.hh"
#include "cusz/type.h"
#include "hf_type.h"
#include "hfbk.hh"
#include "hfbuf.inl"
#include "hfclass.hh"
#include "hfcxx_module.hh"
#include "mem/memobj.hh"
#include "port.hh"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/format.hh"
#include "utils/timer.hh"

#define PHF_TPL template <typename E, bool TIMING>
#define PHF_CLASS HuffmanCodec<E, TIMING>

namespace phf {

template <typename E, bool TIMING>
struct HuffmanCodec<E, TIMING>::impl {
  static const bool TimeBreakdown{false};

  using phf_module = phf::coarse::kernel_wrapper<E, H, TIMING>;
  using Header = phf_header;

  phf_header header;  // TODO combine psz and pszhf headers

  hires::time_point f, e, d, c, b, a;
  float _time_book{0.0}, _time_lossless{0.0};
  float time_book() const { return _time_book; }
  float time_codec() const { return _time_lossless; }
  size_t inlen() const { return len; };

  size_t pardeg, sublen;
  int numSMs;
  size_t len, bklen;
  memobj<u4>* hist;

  internal_buffer* buf;

  impl() = default;
  ~impl() { delete buf; }

  // keep ctor short
  void init(
      size_t const inlen, int const _bklen, int const _pardeg, bool debug)
  {
    GpuDeviceGetAttribute(&numSMs, GpuDevAttrMultiProcessorCount, 0);

    pardeg = _pardeg;
    bklen = _bklen;
    len = inlen;
    sublen = (inlen - 1) / pardeg + 1;

    // TODO make unique_ptr; modify ctor
    buf = new internal_buffer(inlen, _bklen, _pardeg, false, debug);
    hist = new memobj<u4>(
        bklen, "phf::hist", {MallocHost});  // dptr from external
  }

#ifdef ENABLE_HUFFBK_GPU
  // build Huffman tree on GPU
  void buildbook(uint32_t* freq, int const bklen, phf_stream_t stream)
  {
    psz::hf_buildbook<CUDA, E, H4>(
        freq, bklen, bk4->dptr(), revbk4->dptr(), revbk4_bytes(bklen),
        &_time_book, (GpuStreamT)stream);
  }
#else
  // build Huffman tree on CPU
  void buildbook(u4* freq, phf_stream_t stream)
  {
    this->hist->dptr(freq);

    psz::hf_buildbook<SEQ, E, H4>(
        hist->control({D2H})->hptr(), bklen, buf->bk4->hptr(),
        buf->revbk4->hptr(), revbk4_bytes(bklen), &_time_book,
        (GpuStreamT)stream);
    buf->bk4->control({ASYNC_H2D}, (GpuStreamT)stream);
    buf->revbk4->control({ASYNC_H2D}, (GpuStreamT)stream);
  }
#endif

  static int __revbk_bytes(int bklen, int BK_UNIT_BYTES, int SYM_BYTES)
  {
    static const int CELL_BITWIDTH = BK_UNIT_BYTES * 8;
    return BK_UNIT_BYTES * (2 * CELL_BITWIDTH) + SYM_BYTES * bklen;
  }

  static int revbk4_bytes(int bklen)
  {
    return __revbk_bytes(bklen, 4, sizeof(SYM));
  }
  static int revbk8_bytes(int bklen)
  {
    return __revbk_bytes(bklen, 8, sizeof(SYM));
  }

  void encode(
      E* in, size_t const len, uint8_t** out, size_t* outlen,
      phf_stream_t stream)
  {
    _time_lossless = 0;

    hfpar_description hfpar{sublen, pardeg};

    {
      phf_module::encode_phase1(
          {in, len}, buf->bk4->array1_d(), numSMs, buf->scratch4->array1_d(),
          &_time_lossless, stream);

      if constexpr (TimeBreakdown) { b = hires::now(); }

      phf_module::encode_phase2(
          buf->scratch4->array1_d(), hfpar,
          buf->scratch4->array1_d() /* placeholder */,
          buf->par_nbit->array1_d(), buf->par_ncell->array1_d(),
          &_time_lossless, stream);

      if constexpr (TimeBreakdown) c = hires::now();
    }

    phf_module::encode_phase3(
        buf->par_nbit->array1_d(), buf->par_ncell->array1_d(),
        buf->par_entry->array1_d(), hfpar, buf->par_nbit->array1_h(),
        buf->par_ncell->array1_h(), buf->par_entry->array1_h(),
        &header.total_nbit, &header.total_ncell, nullptr, stream);

    if constexpr (TimeBreakdown) d = hires::now();

    phf_module::encode_phase4(
        buf->scratch4->array1_d(), buf->par_entry->array1_d(),
        buf->par_ncell->array1_d(), hfpar, buf->bitstream4->array1_d(),
        &_time_lossless, stream);

    if constexpr (TimeBreakdown) e = hires::now();

    make_metadata();
    buf->memcpy_merge(header, stream);  // TODO externalize/make explicit

    if constexpr (TimeBreakdown) f = hires::now();

    if constexpr (TimeBreakdown) {
      cout << "phase1: " << static_cast<duration_t>(b - a).count() * 1e6
           << endl;
      cout << "phase2: " << static_cast<duration_t>(c - a).count() * 1e6
           << endl;
      cout << "phase3: " << static_cast<duration_t>(d - a).count() * 1e6
           << endl;
      cout << "phase4: " << static_cast<duration_t>(e - a).count() * 1e6
           << endl;
      cout << "wrapup: " << static_cast<duration_t>(f - a).count() * 1e6
           << endl;
    }

    // TODO may cooperate with upper-level; output
    *out = buf->encoded->dptr();
    *outlen = phf_encoded_bytes(&header);
  }

  void decode(
      uint8_t* in_encoded, E* out_decoded, phf_stream_t stream,
      bool header_on_device)
  {
    Header header;
    if (header_on_device)
      CHECK_GPU(GpuMemcpyAsync(
          &header, in_encoded, sizeof(header), GpuMemcpyD2H,
          (GpuStreamT)stream));

#define PHF_ACCESSOR(SYM, TYPE) \
  reinterpret_cast<TYPE*>(in_encoded + header.entry[PHFHEADER_##SYM])

    phf_module::phf_coarse_decode(
        {PHF_ACCESSOR(BITSTREAM, H4), 0},
        {PHF_ACCESSOR(REVBK, PHF_BYTE), (size_t)revbk4_bytes(header.bklen)},
        {PHF_ACCESSOR(PAR_NBIT, M), (size_t)pardeg},
        {PHF_ACCESSOR(PAR_ENTRY, M), (size_t)pardeg},
        {(size_t)header.sublen, (size_t)header.pardeg}, {out_decoded, 0},
        &_time_lossless, stream);

#undef PHF_ACCESSOR
  }

  void calculate_CR(
      memobj<E>* ectrl, memobj<u4>* freq, szt sizeof_dtype, szt overhead_bytes)
  {
    // serial part
    f8 serial_entropy = 0;
    f8 serial_avg_bits = 0;

    auto len = std::accumulate(freq->hbegin(), freq->hend(), (szt)0);
    // printf("[psz::dbg::hf] len: %zu\n", len);

    for (auto i = 0; i < bklen; i++) {
      auto hfcode = buf->bk4->hat(i);
      if (freq != 0) {
        auto p = 1.0 * freq->hat(i) / len;
        serial_entropy += -std::log2(p) * p;

        auto bits = ((HuffmanWord<4>*)(&hfcode))->bitcount;
        serial_avg_bits += bits * p;
      }
    }

    // parallel simulation
    // f8 parallel_bits = 0;
    ectrl->control({D2H});
    auto tmp_len = ectrl->len();
    for (auto p = 0; p < pardeg; p++) {
      auto start = p * sublen;

      // auto this_ncell = 0,
      auto this_nbit = 0;

      for (auto i = 0; i < sublen; i++) {
        if (i + sublen < tmp_len) {
          auto eq = ectrl->hat(start + i);
          auto c = buf->bk4->hat(eq);
          auto b = ((HuffmanWord<4>*)(&c))->bitcount;
          this_nbit += b;
        }
      }
      buf->par_nbit->hat(p) = this_nbit;
      buf->par_ncell->hat(p) = (this_nbit - 1) / 32 + 1;
    }
    auto final_len =
        std::accumulate(buf->par_ncell->hbegin(), buf->par_ncell->hend(), 0);
    auto final_bytes = 1.0 * final_len * sizeof_dtype;
    final_bytes += buf->par_entry->len() *
                   (sizeof(U4) /* for idx */ + sizeof_dtype);  // outliers
    final_bytes += 128 * 2; /* two kinds of headers */
    final_bytes += overhead_bytes;

    // print report
    // clang-format off
  printf("[phf::calc_cr] get CR from hist and par setup\n");
  printf("[phf::calc_cr] (T, H)=(f4, u4)\n");
  printf("[phf::calc_cr] serial (ref), entropy            : %lf\n", serial_entropy);
  printf("[phf::calc_cr] serial (ref), avg-bit            : %lf\n", serial_avg_bits);
  printf("[phf::calc_cr] serial (ref), entropy-implied CR : %lf\n", sizeof_dtype * 8 / serial_entropy);
  printf("[phf::calc_cr] serial (ref), avg-bit-implied    : %lf\n", sizeof_dtype * 8 / serial_avg_bits);
  printf("[phf::calc_cr] pSZ/cuSZ achievable CR (chunked) : %lf\n", tmp_len * sizeof_dtype / final_bytes);
  printf("[phf::calc_cr] analysis done, proceeding...\n");
    // clang-format on
  }

  void make_metadata()
  {
    // header.self_bytes = sizeof(Header);
    header.bklen = bklen;
    header.sublen = sublen;
    header.pardeg = pardeg;
    header.original_len = len;

    M nbyte[PHFHEADER_END];
    nbyte[PHFHEADER_HEADER] = PHFHEADER_FORCED_ALIGN;
    nbyte[PHFHEADER_REVBK] = revbk4_bytes(bklen);
    nbyte[PHFHEADER_PAR_NBIT] = buf->par_nbit->bytes();
    nbyte[PHFHEADER_PAR_ENTRY] = buf->par_ncell->bytes();
    nbyte[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;

    header.entry[0] = 0;
    // *.END + 1: need to know the ending position
    for (auto i = 1; i < PHFHEADER_END + 1; i++)
      header.entry[i] = nbyte[i - 1];
    for (auto i = 1; i < PHFHEADER_END + 1; i++)
      header.entry[i] += header.entry[i - 1];
  }

  void clear_buffer() { buf->clear_buffer(); }

};  // end of pimpl class

PHF_TPL PHF_CLASS::HuffmanCodec(
    size_t const inlen, int const bklen, int const pardeg, bool debug) :
    pimpl{std::make_unique<impl>()},
    in_dtype{
        std::is_same_v<E, u1>   ? HF_U1
        : std::is_same_v<E, u2> ? HF_U2
        : std::is_same_v<E, u4> ? HF_U4
                                : HF_INVALID}
{
  pimpl->init(inlen, bklen, pardeg, debug);
};

PHF_TPL PHF_CLASS::~HuffmanCodec(){};

// using CPU huffman
PHF_TPL PHF_CLASS* PHF_CLASS::buildbook(u4* freq, phf_stream_t stream)
{
  pimpl->buildbook(freq, stream);
  return this;
}

// using CPU huffman
PHF_TPL void PHF_CLASS::calculate_CR(
    memobj<E>* ectrl, memobj<u4>* freq, szt sizeof_dtype, szt overhead_bytes)
{
  pimpl->calculate_CR(ectrl, freq, sizeof_dtype, overhead_bytes);
}

PHF_TPL
PHF_CLASS* PHF_CLASS::encode(
    E* in, size_t const len, uint8_t** out, size_t* outlen,
    phf_stream_t stream)
{
  pimpl->encode(in, len, out, outlen, stream);
  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::decode(
    uint8_t* in_encoded, E* out_decoded, phf_stream_t stream,
    bool header_on_device)
{
  pimpl->decode(in_encoded, out_decoded, stream, header_on_device);
  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::clear_buffer()
{
  pimpl->clear_buffer();
  return this;
}

PHF_TPL float PHF_CLASS::time_book() const { return pimpl->time_book(); }
PHF_TPL float PHF_CLASS::time_lossless() const { return pimpl->time_codec(); }
PHF_TPL size_t PHF_CLASS::inlen() const { return pimpl->inlen(); }

}  // namespace phf

#undef PHF_ACCESSOR
#undef PHF_TPL
#undef PHF_CLASS

#endif /* ABBC78E4_3E65_4633_9BEA_27823AB7C398 */
