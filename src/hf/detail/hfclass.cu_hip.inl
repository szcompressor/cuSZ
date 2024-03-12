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

#include "cusz/type.h"
#include "hf/hfcxx_module.hh"
#include "utils/timer.hh"

#define PHF_ACCESSOR(SYM, TYPE) \
  reinterpret_cast<TYPE*>(in_compressed + header.entry[Header::SYM])

#define PHF_TPL template <typename E, typename M, bool TIMING>
#define PHF_CLASS HuffmanCodec<E, M, TIMING>

namespace cusz {

PHF_TPL PHF_CLASS::~HuffmanCodec()
{
  // delete scratch;
  delete bk4, delete revbk4;

  delete par_nbit, delete par_ncell, delete par_entry;

  // delete book_desc;

  delete compressed;
  delete __scratch, delete scratch4;
  delete __bitstream, delete bitstream4;
  delete hist_view;
}

PHF_TPL PHF_CLASS* PHF_CLASS::init(
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
  len = inlen;

  // for both u4 and u8 encoding

  // placeholder length
  compressed = new pszmem_cxx<BYTE>(inlen * TYPICAL, 1, 1, "hf::out4B");

  __scratch = new pszmem_cxx<RAW>(inlen * FAILSAFE, 1, 1, "hf::__scratch");
  scratch4 = new pszmem_cxx<H4>(inlen, 1, 1, "hf::scratch4");

  bk4 = new pszmem_cxx<H4>(bklen, 1, 1, "hf::book4");

  revbk4 = new pszmem_cxx<BYTE>(revbk4_bytes(bklen), 1, 1, "hf::revbk4");

  // encoded buffer
  __bitstream =
      new pszmem_cxx<RAW>(inlen * FAILSAFE / 2, 1, 1, "hf::__bitstrm");
  bitstream4 = new pszmem_cxx<H4>(inlen / 2, 1, 1, "hf::bitstrm4");

  par_nbit = new pszmem_cxx<M>(pardeg, 1, 1, "hf::par_nbit");
  par_ncell = new pszmem_cxx<M>(pardeg, 1, 1, "hf::par_ncell");
  par_entry = new pszmem_cxx<M>(pardeg, 1, 1, "hf::par_entry");

  // external buffer
  hist_view = new MemU4(bklen, 1, 1, "a view of external hist");

  // allocate
  __scratch->control({Malloc, MallocHost});
  scratch4->asaviewof(__scratch);

  bk4->control({Malloc, MallocHost});

  revbk4->control({Malloc, MallocHost});

  __bitstream->control({Malloc, MallocHost});
  bitstream4->asaviewof(__bitstream);

  par_nbit->control({Malloc, MallocHost});
  par_ncell->control({Malloc, MallocHost});
  par_entry->control({Malloc, MallocHost});

  // repurpose scratch after several substeps
  compressed->dptr(__scratch->dptr())->hptr(__scratch->hptr());

  GpuDeviceGetAttribute(&numSMs, GpuDevAttrMultiProcessorCount, 0);

  sublen = (inlen - 1) / pardeg + 1;

  if (debug) __debug();

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
      freq->control({D2H})->hptr(), bklen, bk4->hptr(), revbk4->hptr(),
      revbk4_bytes(bklen), &_time_book, (GpuStreamT)stream);
  bk4->control({ASYNC_H2D}, (GpuStreamT)stream);
  revbk4->control({ASYNC_H2D}, (GpuStreamT)stream);

  hist_view->asaviewof(freq);  // for analysis

  return this;
}

// using CPU huffman
PHF_TPL void PHF_CLASS::calculate_CR(
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
  auto tmp_len = ectrl->len();
  for (auto p = 0; p < pardeg; p++) {
    auto start = p * sublen;

    // auto this_ncell = 0,
    auto this_nbit = 0;

    for (auto i = 0; i < sublen; i++) {
      if (i + sublen < tmp_len) {
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

PHF_TPL PHF_CLASS* PHF_CLASS::encode(
    E* in, size_t const len, uint8_t** out, size_t* outlen,
    uninit_stream_t stream)
{
  _time_lossless = 0;

  H* d_buffer = (H*)__scratch->dptr();
  H* d_bitstream = (H*)__bitstream->dptr();
  H* d_book = (H*)bk4->dptr();
  hfpar_description hfpar{sublen, pardeg};

  auto d_par_nbit = par_nbit->dptr();
  auto d_par_ncell = par_ncell->dptr();
  auto d_par_entry = par_entry->dptr();

  auto h_par_nbit = par_nbit->hptr();
  auto h_par_ncell = par_ncell->hptr();
  auto h_par_entry = par_entry->hptr();

  // auto f = hires::now();
  // auto e = hires::now();
  // auto d = hires::now();
  // auto c = hires::now();
  // auto b = hires::now();
  // auto a = hires::now();

  phf_module::phf_coarse_encode_phase1(
      {in, len}, {d_book, bklen}, numSMs, {d_buffer, len}, &_time_lossless,
      stream);

  // b = hires::now();

  // phf_module::phf_coarse_encode_phase1_collect_metadata(
  //     {in, len}, {d_book, bklen}, numSMs, {d_buffer, len},
  //     {d_par_nbit, pardeg}, {d_par_ncell, pardeg}, {sublen, pardeg},
  //     &_time_lossless, stream);

  phf_module::phf_coarse_encode_phase2(
      {d_buffer, len}, hfpar, {d_buffer, len /* placeholder */},
      {d_par_nbit, pardeg}, {d_par_ncell, pardeg}, &_time_lossless, stream);

  // c = hires::now();

  phf_module::phf_coarse_encode_phase3(
      {d_par_nbit, pardeg}, {d_par_ncell, pardeg}, {d_par_entry, pardeg},
      hfpar, {h_par_nbit, pardeg}, {h_par_ncell, pardeg},
      {h_par_entry, pardeg}, &header.total_nbit, &header.total_ncell, nullptr,
      stream);

  // d = hires::now();

  phf_module::phf_coarse_encode_phase4(
      {d_buffer, len}, {d_par_entry, pardeg}, {d_par_ncell, pardeg}, hfpar,
      {d_bitstream, len}, &_time_lossless, stream);

  // e = hires::now();

  make_metadata();
  phf_memcpy_merge(stream);  // TODO externalize/make explicit

  // f = hires::now();

  // cout << "phase1: " << static_cast<duration_t>(b - a).count() * 1e6 <<
  // endl; cout << "phase2: " << static_cast<duration_t>(c - a).count() * 1e6
  // << endl; cout << "phase3: " << static_cast<duration_t>(d - a).count() *
  // 1e6 << endl; cout << "phase4: " << static_cast<duration_t>(e - a).count()
  // * 1e6 << endl; cout << "wrapup: " << static_cast<duration_t>(f -
  // a).count() * 1e6 << endl;

  // TODO may cooperate with upper-level; output
  *out = compressed->dptr();
  *outlen = header.compressed_size();

  return this;
}
PHF_TPL PHF_CLASS* PHF_CLASS::phf_memcpy_merge(uninit_stream_t stream)
{
  phf_memcpy_merge(
      header, compressed->dptr(), 0,
      {revbk4->dptr(), revbk4->bytes(), header.entry[Header::REVBK]},
      {par_nbit->dptr(), par_nbit->bytes(), header.entry[Header::PAR_NBIT]},
      {par_entry->dptr(), par_entry->bytes(), header.entry[Header::PAR_ENTRY]},
      {__bitstream->dptr(), __bitstream->bytes(),
       header.entry[Header::BITSTREAM]},
      stream);
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
  nbyte[Header::PAR_NBIT] = par_nbit->bytes();
  nbyte[Header::PAR_ENTRY] = par_ncell->bytes();
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

PHF_TPL PHF_CLASS* PHF_CLASS::dump(
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

PHF_TPL PHF_CLASS* PHF_CLASS::clear_buffer()
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
PHF_TPL void PHF_CLASS::phf_memcpy_merge(
    Header& header, void* memcpy_start, size_t memcpy_adjust_to_start,
    memcpy_helper revbk, memcpy_helper par_nbit, memcpy_helper par_entry,
    memcpy_helper bitstream,  //
    uninit_stream_t stream)
{
  auto start = ((uint8_t*)memcpy_start + memcpy_adjust_to_start);
  auto d2d_memcpy_merge = [&](memcpy_helper& var) {
    CHECK_GPU(GpuMemcpyAsync(
        start + var.dst, var.ptr, var.nbyte, GpuMemcpyD2D,
        (GpuStreamT)stream));
  };

  CHECK_GPU(GpuMemcpyAsync(
      start, &header, sizeof(header), GpuMemcpyH2D, (GpuStreamT)stream));
  // /* debug */ CHECK_GPU(GpuStreamSync(stream));

  d2d_memcpy_merge(revbk);
  d2d_memcpy_merge(par_nbit);
  d2d_memcpy_merge(par_entry);
  d2d_memcpy_merge(bitstream);
  // /* debug */ CHECK_GPU(GpuStreamSync(stream));
}

PHF_TPL float PHF_CLASS::time_book() const { return _time_book; }
PHF_TPL float PHF_CLASS::time_lossless() const { return _time_lossless; }

PHF_TPL constexpr bool PHF_CLASS::can_overlap_input_and_firstphase_encode()
{
  return sizeof(E) == sizeof(H4);
}

// auxiliary
PHF_TPL void PHF_CLASS::hf_debug(
    const std::string SYM_name, void* VAR, int SYM)
{
  GpuDevicePtr pbase0{0};
  size_t psize0{0};

  GpuMemGetAddressRange(&pbase0, &psize0, (GpuDevicePtr)VAR);
  printf(
      "%s:\n"
      "\t(supposed) pointer : %p\n"
      "\t(queried)  pbase0  : %p\n"
      "\t(queried)  psize0  : %'9lu\n",
      SYM_name.c_str(), (void*)VAR, (void*)&pbase0, psize0);
  pbase0 = 0, psize0 = 0;
}

}  // namespace cusz

#undef PHF_ACCESSOR
#undef PHF_TPL
#undef PHF_CLASS

#endif /* ABBC78E4_3E65_4633_9BEA_27823AB7C398 */
