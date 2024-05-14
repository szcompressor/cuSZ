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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "busyheader.hh"
#include "cusz/type.h"
#include "hfbuf.inl"
#include "hfclass.hh"
#include "hfcxx_module.hh"
#include "hfword.hh"
#include "utils/timer.hh"

#define PHF_ACCESSOR(SYM, TYPE) \
  reinterpret_cast<TYPE*>(in_compressed + header.entry[Header::SYM])

#define PHF_TPL template <typename E, typename M, bool TIMING>
#define PHF_CLASS HuffmanCodec<E, M, TIMING>

namespace cusz {

PHF_TPL PHF_CLASS::~HuffmanCodec() { delete buf; }

PHF_TPL PHF_CLASS* PHF_CLASS::init(
    size_t const inlen, int const _booklen, int const _pardeg, bool debug)
{
  GpuDeviceGetAttribute(&numSMs, GpuDevAttrMultiProcessorCount, 0);

  pardeg = _pardeg;
  bklen = _booklen;
  len = inlen;
  sublen = (inlen - 1) / pardeg + 1;

  // TODO make unique_ptr; modify ctor
  buf = new internal_buffer(inlen, _booklen, _pardeg, false, debug);

  return this;
}

#ifdef ENABLE_HUFFBK_GPU
PHF_TPL PHF_CLASS* PHF_CLASS::build_codebook(
    uint32_t* freq, int const bklen, uninit_stream_t stream)
{
  psz::hf_buildbook<CUDA, E, H4>(
      freq, bklen, bk4->dptr(), revbk4->dptr(), revbk4_bytes(bklen),
      &_time_book, (GpuStreamT)stream);

  return this;
}
#endif

// using CPU huffman
PHF_TPL PHF_CLASS* PHF_CLASS::build_codebook(
    MemU4* freq, int const bklen, uninit_stream_t stream)
{
  psz::hf_buildbook<SEQ, E, H4>(
      freq->control({D2H})->hptr(), bklen, buf->bk4->hptr(),
      buf->revbk4->hptr(), revbk4_bytes(bklen), &_time_book,
      (GpuStreamT)stream);
  buf->bk4->control({ASYNC_H2D}, (GpuStreamT)stream);
  buf->revbk4->control({ASYNC_H2D}, (GpuStreamT)stream);

  this->hist = freq;

  return this;
}

// using CPU huffman
PHF_TPL void PHF_CLASS::calculate_CR(
    MemU4* ectrl, MemU4* freq, szt sizeof_dtype, szt overhead_bytes)
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

      auto bits = ((PackedWordByWidth<4>*)(&hfcode))->bits;
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
        auto b = ((PackedWordByWidth<4>*)(&c))->bits;
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

PHF_TPL
PHF_CLASS* PHF_CLASS::encode(
    E* in, size_t const len, uint8_t** out, size_t* outlen,
    uninit_stream_t stream)
{
  _time_lossless = 0;

  hfpar_description hfpar{sublen, pardeg};

  {
    phf_module::phf_coarse_encode_phase1(
        {in, len}, buf->bk4->array1_d(), numSMs, buf->scratch4->array1_d(),
        &_time_lossless, stream);

    if constexpr (TimeBreakdown) { b = hires::now(); }

    phf_module::phf_coarse_encode_phase2(
        buf->scratch4->array1_d(), hfpar,
        buf->scratch4->array1_d() /* placeholder */, buf->par_nbit->array1_d(),
        buf->par_ncell->array1_d(), &_time_lossless, stream);

    if constexpr (TimeBreakdown) c = hires::now();
  }

  phf_module::phf_coarse_encode_phase3(
      buf->par_nbit->array1_d(), buf->par_ncell->array1_d(),
      buf->par_entry->array1_d(), hfpar, buf->par_nbit->array1_h(),
      buf->par_ncell->array1_h(), buf->par_entry->array1_h(),
      &header.total_nbit, &header.total_ncell, nullptr, stream);

  if constexpr (TimeBreakdown) d = hires::now();

  phf_module::phf_coarse_encode_phase4(
      buf->scratch4->array1_d(), buf->par_entry->array1_d(),
      buf->par_ncell->array1_d(), hfpar, buf->bitstream4->array1_d(),
      &_time_lossless, stream);

  if constexpr (TimeBreakdown) e = hires::now();

  make_metadata();
  buf->memcpy_merge(header, stream);  // TODO externalize/make explicit

  if constexpr (TimeBreakdown) f = hires::now();

  if constexpr (TimeBreakdown) {
    cout << "phase1: " << static_cast<duration_t>(b - a).count() * 1e6 << endl;
    cout << "phase2: " << static_cast<duration_t>(c - a).count() * 1e6 << endl;
    cout << "phase3: " << static_cast<duration_t>(d - a).count() * 1e6 << endl;
    cout << "phase4: " << static_cast<duration_t>(e - a).count() * 1e6 << endl;
    cout << "wrapup: " << static_cast<duration_t>(f - a).count() * 1e6 << endl;
  }

  // TODO may cooperate with upper-level; output
  *out = buf->compressed->dptr();
  *outlen = header.compressed_size();

  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::make_metadata()
{
  // header.self_bytes = sizeof(Header);
  header.bklen = bklen;
  header.sublen = sublen;
  header.pardeg = pardeg;
  header.original_len = len;

  M nbyte[Header::END];
  nbyte[Header::HEADER] = sizeof(Header);
  nbyte[Header::REVBK] = revbk4_bytes(bklen);
  nbyte[Header::PAR_NBIT] = buf->par_nbit->bytes();
  nbyte[Header::PAR_ENTRY] = buf->par_ncell->bytes();
  nbyte[Header::BITSTREAM] = 4 * header.total_ncell;

  header.entry[0] = 0;
  // *.END + 1: need to know the ending position
  for (auto i = 1; i < Header::END + 1; i++) header.entry[i] = nbyte[i - 1];
  for (auto i = 1; i < Header::END + 1; i++)
    header.entry[i] += header.entry[i - 1];

  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::decode(
    uint8_t* in_compressed, E* out_decompressed, uninit_stream_t stream,
    bool header_on_device)
{
  Header header;
  if (header_on_device)
    CHECK_GPU(GpuMemcpyAsync(
        &header, in_compressed, sizeof(header), GpuMemcpyD2H,
        (GpuStreamT)stream));

  phf_module::phf_coarse_decode(
      {PHF_ACCESSOR(BITSTREAM, H4), 0},
      {PHF_ACCESSOR(REVBK, BYTE), (size_t)revbk4_bytes(header.bklen)},
      {PHF_ACCESSOR(PAR_NBIT, M), (size_t)pardeg},
      {PHF_ACCESSOR(PAR_ENTRY, M), (size_t)pardeg},
      {(size_t)header.sublen, (size_t)header.pardeg}, {out_decompressed, 0},
      &_time_lossless, stream);

  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::clear_buffer()
{
  buf->clear_buffer();
  return this;
}

PHF_TPL float PHF_CLASS::time_book() const { return _time_book; }
PHF_TPL float PHF_CLASS::time_lossless() const { return _time_lossless; }

PHF_TPL constexpr bool PHF_CLASS::can_overlap_input_and_firstphase_encode()
{
  return sizeof(E) == sizeof(H4);
}

}  // namespace cusz

#undef PHF_ACCESSOR
#undef PHF_TPL
#undef PHF_CLASS

#endif /* ABBC78E4_3E65_4633_9BEA_27823AB7C398 */
