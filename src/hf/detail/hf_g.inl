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

#define ACCESSOR(SYM, TYPE) \
  reinterpret_cast<TYPE*>(in_compressed + header.entry[Header::SYM])
#define TEMPLATE_TYPE template <typename E, typename H, typename M>

namespace cusz {

TEMPLATE_TYPE
HuffmanCodec<E, H, M>::~HuffmanCodec()
{
  delete tmp;
  delete book;
  delete revbook;
  delete par_nbit;
  delete par_ncell;
  delete par_entry;
  delete bitstream;
}

TEMPLATE_TYPE
HuffmanCodec<E, H, M>* HuffmanCodec<E, H, M>::init(
    size_t const inlen, int const booklen, int const pardeg, bool debug)
{
  auto __debug = [&]() {
    setlocale(LC_NUMERIC, "");
    printf("\nHuffmanCoarse<E, H, M>::init() debugging:\n");
    printf("GpuDevicePtr nbyte: %d\n", (int)sizeof(GpuDevicePtr));
    hf_debug("TMP", tmp->dptr(), RTE::TMP);
    hf_debug("BOOK", book->dptr(), RTE::BOOK);
    hf_debug("REVBOOK", revbook->dptr(), RTE::REVBOOK);
    hf_debug("PAR_NBIT", par_nbit->dptr(), RTE::PAR_NBIT);
    hf_debug("PAR_NCELL", par_ncell->dptr(), RTE::PAR_NCELL);
    hf_debug("BITSTREAM", bitstream->dptr(), RTE::BITSTREAM);
    printf("\n");
  };

  memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);

  // placeholder length
  compressed = new pszmem_cxx<BYTE>(inlen * 4, 1, 1, "hf::compressed");

  tmp = new pszmem_cxx<H>(inlen, 1, 1, "hf::tmp");
  book = new pszmem_cxx<H>(booklen, 1, 1, "hf::book");
  revbook = new pszmem_cxx<BYTE>(revbook_bytes(booklen), 1, 1, "hf::revbook");
  par_nbit = new pszmem_cxx<M>(pardeg, 1, 1, "hf::par_nbit");
  par_ncell = new pszmem_cxx<M>(pardeg, 1, 1, "hf::par_ncell");
  par_entry = new pszmem_cxx<M>(pardeg, 1, 1, "hf::par_entry");
  bitstream = new pszmem_cxx<H>(inlen / 2, 1, 1, "hf::bitstream");

  rte.nbyte[RTE::TMP] = tmp->bytes();
  rte.nbyte[RTE::BOOK] = book->bytes();
  rte.nbyte[RTE::REVBOOK] = revbook->bytes();
  rte.nbyte[RTE::PAR_NBIT] = par_nbit->bytes();
  rte.nbyte[RTE::PAR_NCELL] = par_ncell->bytes();
  rte.nbyte[RTE::PAR_ENTRY] = par_entry->bytes();
  rte.nbyte[RTE::BITSTREAM] = bitstream->bytes();

  tmp->control({Malloc, MallocHost});
  book->control({Malloc, MallocHost});
  revbook->control({Malloc, MallocHost});
  par_nbit->control({Malloc, MallocHost});
  par_ncell->control({Malloc, MallocHost});
  par_entry->control({Malloc, MallocHost});
  bitstream->control({Malloc, MallocHost});

  compressed->dptr((uint8_t*)tmp->dptr())->hptr((uint8_t*)tmp->hptr());

  int numSMs;
  GpuDeviceGetAttribute(&numSMs, GpuDevAttrMultiProcessorCount, 0);

  int sublen = (inlen - 1) / pardeg + 1;

  book_desc = new hf_book{nullptr, book->dptr(), booklen};
  chunk_desc_d =
      new hf_chunk{par_nbit->dptr(), par_ncell->dptr(), par_entry->dptr()};
  chunk_desc_h =
      new hf_chunk{par_nbit->hptr(), par_ncell->hptr(), par_entry->hptr()};
  bitstream_desc = new hf_bitstream{tmp->dptr(),  bitstream->dptr(),
                                    chunk_desc_d, chunk_desc_h,
                                    sublen,       pardeg,
                                    numSMs};

  if (debug) __debug();

  return this;
}

#ifdef ENABLE_HUFFBK_GPU
TEMPLATE_TYPE
HuffmanCodec<E, H, M>* HuffmanCodec<E, H, M>::build_codebook(
    uint32_t* freq, int const booklen, void* stream)
{
  psz::hf_buildbook<CUDA, E, H>(
      freq, booklen, book->dptr(), revbook->dptr(), revbook_bytes(booklen),
      &_time_book, (GpuStreamT)stream);

  return this;
}
#endif

TEMPLATE_TYPE
HuffmanCodec<E, H, M>* HuffmanCodec<E, H, M>::build_codebook(
    pszmem_cxx<uint32_t>* freq, int const booklen, void* stream)
{
  // printf("using CPU huffman\n");
  psz::hf_buildbook<CPU, E, H>(
      freq->control({D2H})->hptr(), booklen, book->hptr(), revbook->hptr(),
      revbook_bytes(booklen), &_time_book, (GpuStreamT)stream);

  book->control({ASYNC_H2D}, (GpuStreamT)stream);
  revbook->control({ASYNC_H2D}, (GpuStreamT)stream);

  return this;
}

TEMPLATE_TYPE
HuffmanCodec<E, H, M>* HuffmanCodec<E, H, M>::encode(
    E* in, size_t const inlen, uint8_t** out, size_t* outlen, void* stream)
{
  _time_lossless = 0;

  struct Header header;

  psz::hf_encode_coarse_rev2<E, H, M>(
      in, inlen, book_desc, bitstream_desc, &header.total_nbit,
      &header.total_ncell, &_time_lossless, stream);

  // update with the precise BITSTREAM nbyte
  rte.nbyte[RTE::BITSTREAM] = sizeof(H) * header.total_ncell;

  // d_revbook and revbook_nbyte is hidden; need to improve here
  hf_merge(
      header, inlen, book_desc->booklen, bitstream_desc->sublen,
      bitstream_desc->pardeg, stream);

  *out = compressed->dptr();
  *outlen = header.compressed_size();

  return this;
}

TEMPLATE_TYPE
HuffmanCodec<E, H, M>* HuffmanCodec<E, H, M>::decode(
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
  auto d_bitstream = ACCESSOR(BITSTREAM, H);

  auto const revbook_nbyte = revbook_bytes(header.booklen);

  // launch_coarse_grained_Huffman_decoding<E, H, M>(
  psz::hf_decode_coarse<E, H, M>(
      d_bitstream, d_revbook, revbook_nbyte, d_par_nbit, d_par_entry,
      header.sublen, header.pardeg, out_decompressed, &_time_lossless, stream);

  return this;
}

TEMPLATE_TYPE
HuffmanCodec<E, H, M>* HuffmanCodec<E, H, M>::dump(
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
      compressed->control({H2D})->file(ofn(".pszhf_archive"), ToFile);
    else if (i == PszHfBook)
      book->control({H2D})->file(ofn(".pszhf_book"), ToFile);
    else if (i == PszHfRevbook)
      revbook->control({H2D})->file(ofn(".pszhf_revbook"), ToFile);
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

TEMPLATE_TYPE
HuffmanCodec<E, H, M>* HuffmanCodec<E, H, M>::clear_buffer()
{
  tmp->control({ClearDevice});
  book->control({ClearDevice});
  revbook->control({ClearDevice});
  par_nbit->control({ClearDevice});
  par_ncell->control({ClearDevice});
  par_entry->control({ClearDevice});
  bitstream->control({ClearDevice});

  return this;
}

// private helper
TEMPLATE_TYPE
void HuffmanCodec<E, H, M>::hf_merge(
    Header& header, size_t const original_len, int const booklen,
    int const sublen, int const pardeg, void* stream)
{
  auto BARRIER = [&]() {
    if (stream)
      CHECK_GPU(GpuStreamSync(stream));
    else
      CHECK_GPU(GpuDeviceSync());
  };

  header.self_bytes = sizeof(Header);
  header.booklen = booklen;
  header.sublen = sublen;
  header.pardeg = pardeg;
  header.original_len = original_len;

  M nbyte[Header::END];
  nbyte[Header::HEADER] = sizeof(Header);
  nbyte[Header::REVBOOK] = rte.nbyte[RTE::REVBOOK];
  nbyte[Header::PAR_NBIT] = rte.nbyte[RTE::PAR_NBIT];
  nbyte[Header::PAR_ENTRY] = rte.nbyte[RTE::PAR_ENTRY];
  nbyte[Header::BITSTREAM] = rte.nbyte[RTE::BITSTREAM];

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

  constexpr auto D2D = GpuMemcpyD2D;
  {
    auto dst = compressed->dptr() + header.entry[Header::REVBOOK];
    auto src = revbook->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::REVBOOK], D2D, (GpuStreamT)stream));
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
    auto src = bitstream->dptr();
    CHECK_GPU(GpuMemcpyAsync(
        dst, src, nbyte[Header::BITSTREAM], D2D, (GpuStreamT)stream));
  }
}

TEMPLATE_TYPE
float HuffmanCodec<E, H, M>::time_book() const { return _time_book; }
TEMPLATE_TYPE
float HuffmanCodec<E, H, M>::time_lossless() const { return _time_lossless; }

// TEMPLATE_TYPE
// H* HuffmanCodec<E, H, M>::expose_book() const { return d_book; }

// TEMPLATE_TYPE
// uint8_t* HuffmanCodec<E, H, M>::expose_revbook() const { return d_revbook; }

TEMPLATE_TYPE
size_t HuffmanCodec<E, H, M>::revbook_bytes(int dict_size)
{
  return sizeof(BOOK) * (2 * CELL_BITWIDTH) + sizeof(SYM) * dict_size;
}

TEMPLATE_TYPE
constexpr bool HuffmanCodec<E, H, M>::can_overlap_input_and_firstphase_encode()
{
  return sizeof(E) == sizeof(H);
}

// auxiliary
TEMPLATE_TYPE
void HuffmanCodec<E, H, M>::hf_debug(
    const std::string SYM_name, void* VAR, int SYM)
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
      SYM_name.c_str(), (void*)VAR, (size_t)rte.nbyte[SYM], (void*)&pbase0,
      psize0);
  pbase0 = 0, psize0 = 0;
}

}  // namespace cusz

#undef ACCESSOR
#undef TEMPLATE_TYPE

#endif /* ABBC78E4_3E65_4633_9BEA_27823AB7C398 */
