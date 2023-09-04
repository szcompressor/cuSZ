/**
 * @file huffman_coarse.cuh
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
#define ACCESSOR(SYM, TYPE) \
  reinterpret_cast<TYPE*>(in_compressed + header.entry[Header::SYM])

#define TPL template <typename E, typename M>
#define HF_CODEC HuffmanCodec<E, M>

namespace cusz {

TPL HF_CODEC::~HuffmanCodec()
{
  // delete scratch;
  delete __bk;
  delete __revbk;
  delete __bitstream;

  delete par_nbit;
  delete par_ncell;
  delete par_entry;
}

TPL HF_CODEC* HF_CODEC::init(
    size_t const inlen, int const _booklen, int const _pardeg, bool debug)
{
  auto __debug = [&]() {
    setlocale(LC_NUMERIC, "");
    printf("\nHuffmanCoarse<E, H4, M>::init() debugging:\n");
    printf("GpuDevicePtr nbyte: %d\n", (int)sizeof(GpuDevicePtr));
    hf_debug("SCRATCH", __scratch->dptr(), RC::SCRATCH);
    hf_debug("BOOK", __bk->dptr(), RC::BOOK);
    hf_debug("REVBOOK", __revbk->dptr(), RC::REVBOOK);
    hf_debug("BITSTREAM", __bitstream->dptr(), RC::BITSTREAM);
    hf_debug("PAR_NBIT", par_nbit->dptr(), RC::PAR_NBIT);
    hf_debug("PAR_NCELL", par_ncell->dptr(), RC::PAR_NCELL);
    printf("\n");
  };

  memset(rc.nbyte, 0, sizeof(uint32_t) * RC::END);

  pardeg = _pardeg;
  bklen = _booklen;

  // for both u4 and u8 encoding

  // placeholder length
  compressed4 = new pszmem_cxx<BYTE>(inlen * TYPICAL, 1, 1, "hf::out4B");
  compressed8 = new pszmem_cxx<BYTE>(inlen * FAILSAFE, 1, 1, "hf::out8B");

  __scratch = new pszmem_cxx<RAW>(inlen * FAILSAFE, 1, 1, "hf::__scratch");
  scratch4 = new pszmem_cxx<H4>(inlen, 1, 1, "hf::scratch4");
  scratch8 = new pszmem_cxx<H8>(inlen, 1, 1, "hf::scratch8");

  __bk = new pszmem_cxx<RAW>(bklen * FAILSAFE, 1, 1, "hf::__book");
  bk4 = new pszmem_cxx<H4>(bklen, 1, 1, "hf::book4");
  bk8 = new pszmem_cxx<H8>(bklen, 1, 1, "hf::book8");

  __revbk = new pszmem_cxx<RAW>(revbk8_bytes(bklen), 1, 1, "hf::__revbk");
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

  // rc.nbyte[RC::SCRATCH] = scratch->bytes();
  rc.nbyte[RC::BOOK] = bk4->bytes();
  rc.nbyte[RC::REVBOOK] = revbk4->bytes();
  rc.nbyte[RC::BITSTREAM] = bitstream4->bytes();
  rc.nbyte[RC::PAR_NBIT] = par_nbit->bytes();
  rc.nbyte[RC::PAR_NCELL] = par_ncell->bytes();
  rc.nbyte[RC::PAR_ENTRY] = par_entry->bytes();

  __scratch->control({Malloc, MallocHost});
  scratch4->asaviewof(__scratch);
  scratch8->asaviewof(__scratch);

  __bk->control({Malloc, MallocHost});
  bk4->asaviewof(__bk);
  bk8->asaviewof(__bk);

  __revbk->control({Malloc, MallocHost});
  revbk4->asaviewof(__revbk);
  revbk8->asaviewof(__revbk);

  __bitstream->control({Malloc, MallocHost});
  bitstream4->asaviewof(__bitstream);
  bitstream8->asaviewof(__bitstream);

  par_nbit->control({Malloc, MallocHost});
  par_ncell->control({Malloc, MallocHost});
  par_entry->control({Malloc, MallocHost});

  // repurpose scratch after several substeps
  compressed4->dptr(__scratch->dptr())->hptr(__scratch->hptr());
  compressed8->dptr(__scratch->dptr())->hptr(__scratch->hptr());

  GpuDeviceGetAttribute(&numSMs, GpuDevAttrMultiProcessorCount, 0);

  // #ifdef PSZ_USE_HIP
  // cout << "[psz::dbg::hf] numSMs=" << numSMs << endl;
  // #endif

  {
    int sublen = (inlen - 1) / pardeg + 1;

    book_desc = new hf_book{nullptr, bk4->dptr(), bklen};
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
    uint32_t* freq, int const bklen, void* stream)
{
  psz::hf_buildbook<CUDA, E, H4>(
      freq, bklen, bk4->dptr(), revbk4->dptr(), revbook_bytes(bklen),
      &_time_book, (GpuStreamT)stream);

  return this;
}
#endif

TPL HF_CODEC* HF_CODEC::build_codebook(
    pszmem_cxx<uint32_t>* freq, int const bklen, void* stream)
{
  // printf("using CPU huffman\n");
  psz::hf_buildbook<CPU, E, H4>(
      freq->control({D2H})->hptr(), bklen, bk4->hptr(), revbk4->hptr(),
      revbook_bytes(bklen), &_time_book, (GpuStreamT)stream);

  // for (auto i = 0; i < bklen; i++) {
  //   auto f = freq->hptr(i);
  //   if (f != 0)
  //     printf("[psz::dbg::codebook::freq(i)] (idx) %5d    (freq) %8d\n", i,
  //     f);
  // }

  bk4->control({ASYNC_H2D}, (GpuStreamT)stream);
  revbk4->control({ASYNC_H2D}, (GpuStreamT)stream);

  return this;
}

TPL HF_CODEC* HF_CODEC::encode(
    E* in, size_t const inlen, uint8_t** out, size_t* outlen, void* stream)
{
  _time_lossless = 0;

  pszhf_header header;

  psz::hf_encode_coarse_rev2<E, H4, M>(
      in, inlen, book_desc, bitstream_desc, &header.total_nbit,
      &header.total_ncell, &_time_lossless, stream);

  // update with the precise BITSTREAM nbyte
  rc.nbyte[RC::BITSTREAM] = sizeof(H4) * header.total_ncell;

  // d_revbook and revbook_nbyte is hidden; need to improve here
  hf_merge(
      header, inlen, book_desc->bklen, bitstream_desc->sublen,
      bitstream_desc->pardeg, stream);

  *out = compressed4->dptr();
  *outlen = header.compressed_size();

  return this;
}

TPL HF_CODEC* HF_CODEC::decode(
    uint8_t* in_compressed, E* out_decompressed, void* stream,
    bool header_on_device)
{
  Header header;
  if (header_on_device)
    CHECK_GPU(GpuMemcpyAsync(
        &header, in_compressed, sizeof(header), GpuMemcpyD2H,
        (GpuStreamT)stream));

  auto d_revbook = ACCESSOR(REVBOOK, uint8_t);
  auto d_par_nbit = ACCESSOR(PAR_NBIT, M);
  auto d_par_entry = ACCESSOR(PAR_ENTRY, M);
  auto d_bitstream = ACCESSOR(BITSTREAM, H4);

  auto const revbook_nbyte = revbook_bytes(header.bklen);

  // launch_coarse_grained_Huffman_decoding<E, H4, M>(
  psz::hf_decode_coarse<E, H4, M>(
      d_bitstream, d_revbook, revbook_nbyte, d_par_nbit, d_par_entry,
      header.sublen, header.pardeg, out_decompressed, &_time_lossless, stream);

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

    // TODO check if compressed4 len updated
    if (i == PszHfArchive)
      compressed4->control({H2D})->file(ofn(".pszhf_archive"), ToFile);
    else if (i == PszHfBook)
      bk4->control({H2D})->file(ofn(".pszhf_book"), ToFile);
    else if (i == PszHfRevbook)
      revbk4->control({H2D})->file(ofn(".pszhf_revbook"), ToFile);
    else if (i == PszHfParNbit)
      par_nbit->control({H2D})->file(ofn(".pszhf_parnbit"), ToFile);
    else if (i == PszHfParNcell)
      par_ncell->control({H2D})->file(ofn(".pszhf_parncell"), ToFile);
    else if (i == PszHfParEntry)
      par_entry->control({H2D})->file(ofn(".pszhf_parentry"), ToFile);
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
TPL void HF_CODEC::hf_merge(
    Header& header, size_t const original_len, int const bklen,
    int const sublen, int const pardeg, void* stream)
{
  auto BARRIER = [&]() {
    if (stream)
      CHECK_GPU(GpuStreamSync(stream));
    else
      CHECK_GPU(GpuDeviceSync());
  };

  header.self_bytes = sizeof(Header);
  header.bklen = bklen;
  header.sublen = sublen;
  header.pardeg = pardeg;
  header.original_len = original_len;

  M nbyte[Header::END];
  nbyte[Header::HEADER] = sizeof(Header);
  nbyte[Header::REVBOOK] = rc.nbyte[RC::REVBOOK];
  nbyte[Header::PAR_NBIT] = rc.nbyte[RC::PAR_NBIT];
  nbyte[Header::PAR_ENTRY] = rc.nbyte[RC::PAR_ENTRY];
  nbyte[Header::BITSTREAM] = rc.nbyte[RC::BITSTREAM];

  header.entry[0] = 0;
  // *.END + 1: need to know the ending position
  for (auto i = 1; i < Header::END + 1; i++) {
    header.entry[i] = nbyte[i - 1];
  }
  for (auto i = 1; i < Header::END + 1; i++) {
    header.entry[i] += header.entry[i - 1];
  }

  CHECK_GPU(GpuMemcpyAsync(
      compressed4->dptr(), &header, sizeof(header), GpuMemcpyH2D,
      (GpuStreamT)stream));

  /* debug */ BARRIER();

  constexpr auto D2D = GpuMemcpyD2D;
  {
    auto dst = compressed4->dptr() + header.entry[Header::REVBOOK];
    auto src = revbk4->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::REVBOOK], D2D, (GpuStreamT)stream));
  }
  {
    auto dst = compressed4->dptr() + header.entry[Header::PAR_NBIT];
    auto src = par_nbit->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::PAR_NBIT], D2D, (GpuStreamT)stream));
  }
  {
    auto dst = compressed4->dptr() + header.entry[Header::PAR_ENTRY];
    auto src = par_entry->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::PAR_ENTRY], D2D, (GpuStreamT)stream));
  }
  {
    auto dst = compressed4->dptr() + header.entry[Header::BITSTREAM];
    auto src = bitstream4->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::BITSTREAM], D2D, (GpuStreamT)stream));
  }
}

TPL float HF_CODEC::time_book() const { return _time_book; }
TPL float HF_CODEC::time_lossless() const { return _time_lossless; }

// TPL
// H4* HF_CODEC::expose_book() const { return d_book; }

// TPL
// uint8_t* HF_CODEC::expose_revbook() const { return d_revbook; }

// TPL size_t HF_CODEC::revbook_bytes(int dict_size)
// {
//   static const int CELL_BITWIDTH = sizeof(BOOK_4B) * 8;
//   return sizeof(BOOK_4B) * (2 * CELL_BITWIDTH) + sizeof(SYM) * dict_size;
// }

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
      "\t(supposed) bytes   : %'9lu\n"
      "\t(queried)  pbase0  : %p\n"
      "\t(queried)  psize0  : %'9lu\n",
      SYM_name.c_str(), (void*)VAR, (size_t)rc.nbyte[SYM], (void*)&pbase0,
      psize0);
  pbase0 = 0, psize0 = 0;
}

}  // namespace cusz

#undef ACCESSOR
#undef TPL
#undef HF_CODEC

#endif /* ABBC78E4_3E65_4633_9BEA_27823AB7C398 */
