/**
 * @file hfclass.cuhip.inl
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

#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "cxx_hfbk.h"
#include "hf_type.h"
#include "hfbuf.inl"
#include "hfclass.hh"
#include "hfcxx_module.hh"
#include "mem/cxx_memobj.h"
#include "mem/cxx_sp_gpu.h"
#include "rs_merge.hxx"
#include "utils/io.hh"
#include "utils/timer.hh"

#define PHF_TPL template <typename E>
#define PHF_CLASS HuffmanCodec<E>

using _portable::utils::tofile;

namespace phf {

template <typename E>
struct HuffmanCodec<E>::impl {
  static const bool TimeBreakdown{false};

  using phf_module = phf::cuhip::modules<E, H>;
  using Header = phf_header;

  phf_header header;  // TODO combine psz and pszhf headers

  hires::time_point f, e, d, c, b, a;
  float _time_book{0.0}, _time_lossless{0.0};
  float time_book() const { return _time_book; }
  float time_codec() const { return _time_lossless; }
  size_t inlen() const { return len; };

  GPU_EVENT event_start, event_end;

  size_t pardeg, sublen;
  int numSMs;
  size_t len, bklen;

  GPU_unique_hptr<u4[]> h_hist;

  Buf* buf;

  impl() = default;
  ~impl()
  {
    delete buf;
    event_destroy_pair(event_start, event_end);
  }

  impl(size_t const inlen, int const _bklen, int const _pardeg, bool use_HFR, bool debug)
  {
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    pardeg = _pardeg;
    bklen = _bklen;
    len = inlen;
    sublen = (inlen - 1) / pardeg + 1;

    header.HFR_in_use = use_HFR;

    // TODO make unique_ptr; modify ctor
    buf = new Buf(inlen, _bklen, _pardeg, use_HFR, debug);
    h_hist = MAKE_UNIQUE_HOST(u4, bklen);

    std::tie(event_start, event_end) = event_create_pair();
  }

#ifdef ENABLE_HUFFBK_GPU
  // build Huffman tree on GPU
  void buildbook(uint32_t* freq, int const bklen, phf_stream_t stream)
  {
    psz::hf_buildbook<CUDA, E, H4>(
        freq, bklen, bk4->dptr(), revbk4->dptr(), revbk4_bytes(bklen), &_time_book,
        (cudaStream_t)stream);
  }
#else
  // build Huffman tree on CPU
  void buildbook(u4* freq, phf_stream_t stream)
  {
    memcpy_allkinds<D2H>(h_hist.get(), freq, bklen);

    phf_CPU_build_canonized_codebook_v2<E, H4>(
        h_hist.get(), bklen, buf->h_bk4.get(), buf->h_revbk4.get(), buf->revbk4_bytes,
        &_time_book);
    memcpy_allkinds_async<H2D>(buf->d_bk4.get(), buf->h_bk4.get(), bklen, (cudaStream_t)stream);
    memcpy_allkinds_async<H2D>(
        buf->d_revbk4.get(), buf->h_revbk4.get(), buf->revbk4_bytes, (cudaStream_t)stream);
  }
#endif

  void encode(E* in, size_t const len, uint8_t** out, size_t* outlen, phf_stream_t stream)
  {
    _time_lossless = 0;

    phf::par_config hfpar{sublen, pardeg};

    event_recording_start(event_start, stream);

    phf_module::GPU_coarse_encode_phase1(
        {in, len}, {buf->d_bk4.get(), bklen}, numSMs, {buf->d_scratch4.get(), len}, stream);

    phf_module::GPU_coarse_encode_phase2(
        {buf->d_scratch4.get(), len}, hfpar, {buf->d_scratch4.get(), len} /* placeholder */,
        {buf->d_par_nbit.get(), buf->pardeg}, {buf->d_par_ncell.get(), buf->pardeg}, stream);

    // sync_by_stream(stream);

    phf_module::GPU_coarse_encode_phase3_sync(
        {buf->d_par_nbit.get(), buf->pardeg}, {buf->d_par_ncell.get(), buf->pardeg},
        {buf->d_par_entry.get(), buf->pardeg}, hfpar, {buf->h_par_nbit.get(), buf->pardeg},
        {buf->h_par_ncell.get(), buf->pardeg}, {buf->h_par_entry.get(), buf->pardeg},
        &header.total_nbit, &header.total_ncell, nullptr, stream);

    // sync_by_stream(stream);

    phf_module::GPU_coarse_encode_phase4(
        {buf->d_scratch4.get(), len}, {buf->d_par_entry.get(), buf->pardeg},
        {buf->d_par_ncell.get(), buf->pardeg}, hfpar,
        {buf->d_bitstream4.get(), buf->bitstream_max_len}, stream);

    event_recording_stop(event_end, stream);
    event_time_elapsed(event_start, event_end, &_time_lossless);

    sync_by_stream(stream);

    make_metadata();
    buf->memcpy_merge(header, stream);  // TODO externalize/make explicit

    if constexpr (TimeBreakdown) {
      cout << "phase1: " << static_cast<duration_t>(b - a).count() * 1e6 << endl;
      cout << "phase2: " << static_cast<duration_t>(c - a).count() * 1e6 << endl;
      cout << "phase3: " << static_cast<duration_t>(d - a).count() * 1e6 << endl;
      cout << "phase4: " << static_cast<duration_t>(e - a).count() * 1e6 << endl;
      cout << "wrapup: " << static_cast<duration_t>(f - a).count() * 1e6 << endl;
    }

    // TODO may cooperate with upper-level; output
    *out = buf->d_encoded;
    *outlen = phf_encoded_bytes(&header);
  }

  void encode_HFR(E* in, size_t const len, uint8_t** out, size_t* outlen, phf_stream_t stream)
  {
    auto dummy0 = hires::now();
    auto dummy1 = hires::now();
    auto hfr_timer0 = hires::now();

    double avg_bitwidth = 0;

    for (auto i = 0; i < buf->bklen; i++) {
      auto word = buf->h_bk4[i];
      auto a = reinterpret_cast<HuffmanWord<4>*>(&word);
      auto count = h_hist[i];
      if (count != 0) avg_bitwidth += a->bitcount * 1.0 * count / len;
    }

    printf("avg_bitwidth: %.3lf\n", avg_bitwidth);
    printf("CR based on avg_bitwidth: %.3lf\n", 32 / avg_bitwidth);

    auto chunk = 1 << HFR_Magnitude;
    auto n_chunk = (len - 1) / chunk + 1;
    H altcode{};
    u4 alt_bitcount{0};

#define HFR_ENCODE(REDUCE)                                                        \
  phf::make_altcode(buf->h_bk4.get(), buf->bklen, REDUCE, altcode, alt_bitcount); \
  printf("%u-to-1 reduction (%ux)\n", (1 << REDUCE), REDUCE);                     \
  phf::module::GPU_HFReVISIT_encode<E, HFR_Magnitude, REDUCE, false, H>(          \
      {in, len}, {buf->h_bk4.get(), (u2)buf->bklen, altcode, alt_bitcount},       \
      buf->dense_space(n_chunk), buf->sparse_space(), stream);

    // clang-format off
  if (avg_bitwidth < 2) { HFR_ENCODE(4) }
  else if (avg_bitwidth < 4) { HFR_ENCODE(3) }
  else if (avg_bitwidth < 8) { HFR_ENCODE(2) }
  else if (avg_bitwidth < 16) { HFR_ENCODE(1) }
  else {  // FIXME TODO only for development purpose
    throw std::runtime_error("avg bitwidth >= 16"); }
    // clang-format on

    auto hfr_timer1 = hires::now();
    auto delta = hfr_timer1 - hfr_timer0;
    auto delta_dummy = dummy1 - dummy0;
    cout << "HFR time: " << static_cast<duration_t>(delta - delta_dummy).count() * 1e6 << endl;
    cout << "initial test of HFR finished..." << endl;
    exit(0);

#undef HFR_ENCODE
  }

  void decode(uint8_t* in_encoded, E* out_decoded, phf_stream_t stream, bool header_on_device)
  {
    Header header;
    if (header_on_device)
      CHECK_GPU(cudaMemcpyAsync(
          &header, in_encoded, sizeof(header), cudaMemcpyDeviceToHost, (cudaStream_t)stream));

#define PHF_ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_encoded + header.entry[PHFHEADER_##SYM])

    event_recording_start(event_start, stream);

    phf_module::GPU_coarse_decode(
        PHF_ACCESSOR(BITSTREAM, H4), PHF_ACCESSOR(REVBK, PHF_BYTE), buf->revbk4_bytes,
        PHF_ACCESSOR(PAR_NBIT, M), PHF_ACCESSOR(PAR_ENTRY, M), header.sublen, header.pardeg,
        out_decoded, stream);

    event_recording_stop(event_end, stream);
    event_time_elapsed(event_start, event_end, &_time_lossless);

#undef PHF_ACCESSOR
  }

  void make_metadata()
  {
    // header.self_bytes = sizeof(Header);
    header.bklen = bklen;
    header.sublen = sublen;
    header.pardeg = pardeg;
    header.original_len = len;

    M nbyte[PHFHEADER_END];
    nbyte[PHFHEADER_HEADER] = sizeof(header);
    nbyte[PHFHEADER_REVBK] = buf->revbk4_bytes;
    nbyte[PHFHEADER_PAR_NBIT] = buf->pardeg * sizeof(M);
    nbyte[PHFHEADER_PAR_ENTRY] = buf->pardeg * sizeof(M);
    nbyte[PHFHEADER_BITSTREAM] = 4 * header.total_ncell;

    header.entry[0] = 0;
    // *.END + 1: need to know the ending position
    for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] = nbyte[i - 1];
    for (auto i = 1; i < PHFHEADER_END + 1; i++) header.entry[i] += header.entry[i - 1];
  }

  void clear_buffer() { buf->clear_buffer(); }

};  // end of pimpl class

PHF_TPL PHF_CLASS::HuffmanCodec(
    size_t const inlen, int const bklen, int const pardeg, bool use_HFR, bool debug) :
    pimpl{std::make_unique<impl>(inlen, bklen, pardeg, use_HFR, debug)},
    in_dtype{
        std::is_same_v<E, u1>   ? HF_U1
        : std::is_same_v<E, u2> ? HF_U2
        : std::is_same_v<E, u4> ? HF_U4
                                : HF_INVALID} {};

PHF_TPL PHF_CLASS::~HuffmanCodec(){};

// using CPU huffman
PHF_TPL PHF_CLASS* PHF_CLASS::buildbook(u4* freq, phf_stream_t stream)
{
  pimpl->buildbook(freq, stream);
  return this;
}

PHF_TPL
PHF_CLASS* PHF_CLASS::encode(
    bool use_HFR, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_stream_t stream)
{
  if (not use_HFR)
    pimpl->encode(in, len, out, outlen, stream);
  else
    pimpl->encode_HFR(in, len, out, outlen, stream);
  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::decode(
    uint8_t* in_encoded, E* out_decoded, phf_stream_t stream, bool header_on_device)
{
  pimpl->decode(in_encoded, out_decoded, stream, header_on_device);
  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::clear_buffer()
{
  pimpl->clear_buffer();
  return this;
}

PHF_TPL PHF_CLASS* PHF_CLASS::dump_internal_data(string field, string fname)
{
  auto ofname = fname + ".book_u4";
  if (field == "book") _portable::utils::tofile(ofname, pimpl->buf->h_bk4.get(), pimpl->bklen);
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
