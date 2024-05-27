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

#include <linux/limits.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "cusz/type.h"
#define ACCESSOR(SYM, TYPE) \
  reinterpret_cast<TYPE*>(in_compressed + header.entry[Header::SYM])

#define TPL template <typename E, typename M>
#define HF_CODEC HuffmanCodec<E, M>

namespace cusz {

TPL HF_CODEC::~HuffmanCodec()
{
  // delete scratch;
  delete bk4, delete revbk4;
  delete bk8, delete revbk8;

  delete par_nbit, delete par_ncell, delete par_entry;

  delete book_desc;
  delete chunk_desc_d, delete chunk_desc_h;
  delete bitstream_desc;

  delete compressed;
  delete __scratch, delete scratch4, delete scratch8;
  delete __bitstream, delete bitstream4, delete bitstream8;
  delete hist_view;
}

TPL HF_CODEC* HF_CODEC::init(
    size_t const inlen, int const _booklen, int const _pardeg, bool debug)
{
  auto __debug = [&]() {
    setlocale(LC_NUMERIC, "");
    printf("\nHuffmanCoarse<E, H4, M>::init() debugging:\n");
    printf("GpuDevicePtr nbyte: %d\n", (int)sizeof(GpuDevicePtr));
    hf_debug("SCRATCH", __scratch->dptr(), RC::SCRATCH);
    // TODO separate 4- and 8- books
    // hf_debug("BK", __bk->dptr(), RC::BK);
    // hf_debug("REVBK", __revbk->dptr(), RC::REVBK);
    hf_debug("BITSTREAM", __bitstream->dptr(), RC::BITSTREAM);
    hf_debug("PAR_NBIT", par_nbit->dptr(), RC::PAR_NBIT);
    hf_debug("PAR_NCELL", par_ncell->dptr(), RC::PAR_NCELL);
    printf("\n");
  };

  pardeg = _pardeg;
  bklen = _booklen;

  // for both u4 and u8 encoding

  // placeholder length
  compressed = new pszmem_cxx<BYTE>(inlen * TYPICAL, 1, 1, "hf::out4B");

  __scratch = new pszmem_cxx<RAW>(inlen * FAILSAFE, 1, 1, "hf::__scratch");
  scratch4 = new pszmem_cxx<H4>(inlen, 1, 1, "hf::scratch4");
  scratch8 = new pszmem_cxx<H8>(inlen, 1, 1, "hf::scratch8");

  bk4 = new pszmem_cxx<H4>(bklen, 1, 1, "hf::book4");
  bk8 = new pszmem_cxx<H8>(bklen, 1, 1, "hf::book8");

  revbk4 = new pszmem_cxx<BYTE>(revbk4_bytes(bklen), 1, 1, "hf::revbk4");
  revbk8 = new pszmem_cxx<BYTE>(revbk8_bytes(bklen), 1, 1, "hf::revbk8");

  // encoded buffer
  __bitstream =
      new pszmem_cxx<RAW>(inlen * FAILSAFE / 2, 1, 1, "hf::__bitstrm");
  bitstream4 = new pszmem_cxx<H4>(inlen / 2, 1, 1, "hf::bitstrm4");
  bitstream8 = new pszmem_cxx<H8>(inlen / 2, 1, 1, "hf::bitstrm8");

  par_nbit = new pszmem_cxx<M>(pardeg, 1, 1, "hf::par_nbit");
  par_ncell = new pszmem_cxx<M>(pardeg, 1, 1, "hf::par_ncell");
  par_entry = new pszmem_cxx<M>(pardeg, 1, 1, "hf::par_entry");

  // external buffer
  hist_view = new MemU4(bklen, 1, 1, "a view of external hist");

  // allocate
  __scratch->control({Malloc, MallocHost});
  scratch4->asaviewof(__scratch);
  scratch8->asaviewof(__scratch);

  bk4->control({Malloc, MallocHost});
  bk8->control({Malloc, MallocHost});

  revbk4->control({Malloc, MallocHost});
  revbk8->control({Malloc, MallocHost});

  __bitstream->control({Malloc, MallocHost});
  bitstream4->asaviewof(__bitstream);
  bitstream8->asaviewof(__bitstream);

  par_nbit->control({Malloc, MallocHost});
  par_ncell->control({Malloc, MallocHost});
  par_entry->control({Malloc, MallocHost});

  // repurpose scratch after several substeps
  compressed->dptr(__scratch->dptr())->hptr(__scratch->hptr());

  GpuDeviceGetAttribute(&numSMs, GpuDevAttrMultiProcessorCount, 0);

  {
    int sublen = (inlen - 1) / pardeg + 1;

    book_desc = new hf_book{nullptr, nullptr, bklen};  //
    chunk_desc_d =
        new hf_chunk{par_nbit->dptr(), par_ncell->dptr(), par_entry->dptr()};
    chunk_desc_h =
        new hf_chunk{par_nbit->hptr(), par_ncell->hptr(), par_entry->hptr()};
    bitstream_desc = new hf_bitstream{
        __scratch->dptr(),
        __bitstream->dptr(),
        chunk_desc_d,
        chunk_desc_h,
        sublen,
        pardeg,
        numSMs};
  }

  if (debug) __debug();

  return this;
}

#ifdef ENABLE_HUFFBK_GPU
TPL HF_CODEC* HF_CODEC::build_codebook(
    uint32_t* freq, int const bklen, uninit_stream_t stream)
{
  psz::hf_buildbook<CUDA, E, H4>(
      freq, bklen, bk4->dptr(), revbk4->dptr(), revbk4_bytes(bklen),
      &_time_book, (GpuStreamT)stream);

  return this;
}
#endif

// using CPU huffman
TPL HF_CODEC* HF_CODEC::build_codebook(
    MemU4* freq, int const bklen, uninit_stream_t stream)
{
#ifdef __WORK_IN_PROGRESS
  psz::hf_buildbook<SEQ, E, H8>(
      freq->control({D2H})->hptr(), bklen, bk8->hptr(), revbk8->hptr(),
      revbk8_bytes(bklen), &_time_book, (GpuStreamT)stream);
  bk8->control({ASYNC_H2D}, (GpuStreamT)stream);
  revbk8->control({ASYNC_H2D}, (GpuStreamT)stream);
  __encdtype = ULL;

  // [TODO] need get max bits of huffman code
#endif

  psz::hf_buildbook<SEQ, E, H4>(
      freq->control({D2H})->hptr(), bklen, bk4->hptr(), revbk4->hptr(),
      revbk4_bytes(bklen), &_time_book, (GpuStreamT)stream);
  bk4->control({ASYNC_H2D}, (GpuStreamT)stream);
  revbk4->control({ASYNC_H2D}, (GpuStreamT)stream);
  __encdtype = U4;

  book_desc->bktype = __encdtype;
  book_desc->book = __encdtype == U4 ? (void*)bk4->dptr() : (void*)bk8->dptr();

  hist_view->asaviewof(freq);  // for analysis

  return this;
}

// using CPU huffman
TPL void HF_CODEC::calculate_CR(
    MemU4* ectrl, szt sizeof_dtype, szt overhead_bytes)
{
  // serial part
  f8 serial_entropy = 0;
  f8 serial_avg_bits = 0;

  auto len = std::accumulate(hist_view->hbegin(), hist_view->hend(), (szt)0);
  // printf("[psz::dbg::hf] len: %zu\n", len);

  for (auto i = 0; i < bklen; i++) {
    auto freq = hist_view->hat(i);
    auto hfcode = bk4->hat(i);
    if (freq != 0) {
      auto p = 1.0 * freq / len;
      serial_entropy += -std::log2(p) * p;

      auto bits = ((PackedWordByWidth<4>*)(&hfcode))->bits;
      serial_avg_bits += bits * p;
    }
  }

  // parallel simulation
  // f8 parallel_bits = 0;
  ectrl->control({D2H});
  auto tmp_sublen = bitstream_desc->sublen;
  auto tmp_pardeg = bitstream_desc->pardeg;
  auto tmp_len = ectrl->len();
  for (auto p = 0; p < tmp_pardeg; p++) {
    auto start = p * tmp_sublen;

    // auto this_ncell = 0,
    auto this_nbit = 0;

    for (auto i = 0; i < tmp_sublen; i++) {
      if (i + tmp_sublen < tmp_len) {
        auto eq = ectrl->hat(start + i);
        auto c = bk4->hat(eq);
        auto b = ((PackedWordByWidth<4>*)(&c))->bits;
        this_nbit += b;
      }
    }
    par_nbit->hat(p) = this_nbit;
    par_ncell->hat(p) = (this_nbit - 1) / 32 + 1;
  }
  auto final_len = std::accumulate(par_ncell->hbegin(), par_ncell->hend(), 0);

  auto final_bytes = 1.0 * final_len * sizeof_dtype;
  final_bytes += par_entry->len() *
                 (sizeof(U4) /* for idx */ + sizeof_dtype);  // outliers
  final_bytes += 128 * 2; /* two kinds of headers */
  final_bytes += overhead_bytes;

  // print report
  // clang-format off
  printf("[psz::info::hf::calc_cr] get CR from hist and par setup\n");
  printf("[psz::info::hf::calc_cr] (T, H)=(f4, u4)\n");
  printf("[psz::info::hf::calc_cr] serial (ref), entropy            : %lf\n", serial_entropy);
  printf("[psz::info::hf::calc_cr] serial (ref), avg-bit            : %lf\n", serial_avg_bits);
  printf("[psz::info::hf::calc_cr] serial (ref), entropy-implied CR : %lf\n", sizeof_dtype * 8 / serial_entropy);
  printf("[psz::info::hf::calc_cr] serial (ref), avg-bit-implied    : %lf\n", sizeof_dtype * 8 / serial_avg_bits);
  printf("[psz::info::hf::calc_cr] pSZ/cuSZ achievable CR (chunked) : %lf\n", tmp_len * sizeof_dtype / final_bytes);
  printf("[psz::info::hf::calc_cr] analysis done, proceeding...\n");
  // clang-format on
}

TPL HF_CODEC* HF_CODEC::encode(
    E* in, size_t const inlen, uint8_t** out, size_t* outlen,
    uninit_stream_t stream)
{
  _time_lossless = 0;

  pszhf_header header;

  // So far, the enc scheme has been deteremined.
  header.encdtype = __encdtype;

  if (__encdtype == U4)
    psz::hf_encode_coarse_rev2<E, H4, M>(
        in, inlen, book_desc, bitstream_desc, &header.total_nbit,
        &header.total_ncell, &_time_lossless, stream);
  else {
    printf("[psz::dbg::hf::enc] using H8 for encoding\n");
    psz::hf_encode_coarse_rev2<E, H8, M>(
        in, inlen, book_desc, bitstream_desc, &header.total_nbit,
        &header.total_ncell, &_time_lossless, stream);
  }

  __hf_merge(
      header, inlen, book_desc->bklen, bitstream_desc->sublen,
      bitstream_desc->pardeg, stream);

  *out = compressed->dptr();
  *outlen = header.compressed_size();

  return this;
}

TPL HF_CODEC* HF_CODEC::decode(
    uint8_t* in_compressed, E* out_decompressed, uninit_stream_t stream,
    bool header_on_device)
{
  Header header;
  if (header_on_device)
    CHECK_GPU(GpuMemcpyAsync(
        &header, in_compressed, sizeof(header), GpuMemcpyD2H,
        (GpuStreamT)stream));

  if (header.encdtype == U4)
    psz::hf_decode_coarse<E, H4, M>(
        ACCESSOR(BITSTREAM, H4), ACCESSOR(REVBK, BYTE),
        revbk4_bytes(header.bklen), ACCESSOR(PAR_NBIT, M),
        ACCESSOR(PAR_ENTRY, M), header.sublen, header.pardeg, out_decompressed,
        &_time_lossless, stream);
  else
    psz::hf_decode_coarse<E, H8, M>(
        ACCESSOR(BITSTREAM, H8), ACCESSOR(REVBK, BYTE),
        revbk8_bytes(header.bklen), ACCESSOR(PAR_NBIT, M),
        ACCESSOR(PAR_ENTRY, M), header.sublen, header.pardeg, out_decompressed,
        &_time_lossless, stream);

  return this;
}

TPL HF_CODEC* HF_CODEC::dump(
    std::vector<pszmem_dump> list, char const* basename)
{
  for (auto& i : list) {
    char __[256];

    auto ofn = [&](char const* suffix) {
      strcpy(__, basename);
      strcat(__, suffix);
      return __;
    };

    // TODO check if compressed len updated
    if (i == PszHfArchive)
      compressed->control({H2D})->file(ofn(".pszhf_ar"), ToFile);
    else if (i == PszHfBook)
      bk4->control({H2D})->file(ofn(".pszhf_bk"), ToFile);
    else if (i == PszHfRevbook)
      revbk4->control({H2D})->file(ofn(".pszhf_revbk"), ToFile);
    else if (i == PszHfParNbit)
      par_nbit->control({H2D})->file(ofn(".pszhf_pbit"), ToFile);
    else if (i == PszHfParNcell)
      par_ncell->control({H2D})->file(ofn(".pszhf_pcell"), ToFile);
    else if (i == PszHfParEntry)
      par_entry->control({H2D})->file(ofn(".pszhf_pentry"), ToFile);
    else
      printf("[hf::dump] not a valid segment to dump.");
  }

  return this;
}

TPL HF_CODEC* HF_CODEC::clear_buffer()
{
  scratch4->control({ClearDevice});
  bk4->control({ClearDevice});
  revbk4->control({ClearDevice});
  bitstream4->control({ClearDevice});

  par_nbit->control({ClearDevice});
  par_ncell->control({ClearDevice});
  par_entry->control({ClearDevice});

  return this;
}

// private helper
TPL void HF_CODEC::__hf_merge(
    Header& header, size_t const original_len, int const bklen,
    int const sublen, int const pardeg, uninit_stream_t stream)
{
  auto BARRIER = [&]() {
    if (stream)
      CHECK_GPU(GpuStreamSync(stream));
    else
      CHECK_GPU(GpuDeviceSync());
  };

  constexpr auto D2D = GpuMemcpyD2D;

  // header.self_bytes = sizeof(Header);
  header.bklen = bklen;
  header.sublen = sublen;
  header.pardeg = pardeg;
  header.original_len = original_len;

  M nbyte[Header::END];
  nbyte[Header::HEADER] = sizeof(Header);
  nbyte[Header::REVBK] =
      __encdtype == U4 ? revbk4_bytes(bklen) : revbk8_bytes(bklen);
  nbyte[Header::PAR_NBIT] = par_nbit->bytes();
  nbyte[Header::PAR_ENTRY] = par_ncell->bytes();
  nbyte[Header::BITSTREAM] = (__encdtype == U4 ? 4 : 8) * header.total_ncell;

  header.entry[0] = 0;
  // *.END + 1: need to know the ending position
  for (auto i = 1; i < Header::END + 1; i++) {
    header.entry[i] = nbyte[i - 1];
  }
  for (auto i = 1; i < Header::END + 1; i++) {
    header.entry[i] += header.entry[i - 1];
  }

  CHECK_GPU(GpuMemcpyAsync(
      compressed->dptr(), &header, sizeof(header), GpuMemcpyH2D,
      (GpuStreamT)stream));

  /* debug */ BARRIER();

  {
    auto dst = compressed->dptr() + header.entry[Header::REVBK];
    auto src = __encdtype == U4 ? revbk4->dptr() : revbk8->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::REVBK], D2D, (GpuStreamT)stream));
  }
  {
    auto dst = compressed->dptr() + header.entry[Header::PAR_NBIT];
    auto src = par_nbit->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::PAR_NBIT], D2D, (GpuStreamT)stream));
  }
  {
    auto dst = compressed->dptr() + header.entry[Header::PAR_ENTRY];
    auto src = par_entry->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::PAR_ENTRY], D2D, (GpuStreamT)stream));
  }
  {
    auto dst = compressed->dptr() + header.entry[Header::BITSTREAM];
    auto src = __bitstream->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::BITSTREAM], D2D, (GpuStreamT)stream));
  }
}

TPL float HF_CODEC::time_book() const { return _time_book; }
TPL float HF_CODEC::time_lossless() const { return _time_lossless; }

TPL constexpr bool HF_CODEC::can_overlap_input_and_firstphase_encode()
{
  return sizeof(E) == sizeof(H4);
}

// auxiliary
TPL void HF_CODEC::hf_debug(const std::string SYM_name, void* VAR, int SYM)
{
  GpuDevicePtr pbase0{0};
  size_t psize0{0};

  GpuMemGetAddressRange(&pbase0, &psize0, (GpuDevicePtr)VAR);
  printf(
      "%s:\n"
      "\t(supposed) pointer : %p\n"
      // "\t(supposed) bytes   : %'9lu\n"
      "\t(queried)  pbase0  : %p\n"
      "\t(queried)  psize0  : %'9lu\n",
      SYM_name.c_str(), (void*)VAR,
      // (size_t)rc.nbyte[SYM],
      (void*)&pbase0, psize0);
  pbase0 = 0, psize0 = 0;
}

}  // namespace cusz

#undef ACCESSOR
#undef TPL
#undef HF_CODEC

#endif /* ABBC78E4_3E65_4633_9BEA_27823AB7C398 */
