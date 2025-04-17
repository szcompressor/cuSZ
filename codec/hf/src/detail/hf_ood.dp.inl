/**
 * @file hfclass.dp.inl
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
#include <dpct/dpct.hpp>
#include <numeric>
#include <stdexcept>
#include <sycl/sycl.hpp>
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
    printf("CUdeviceptr nbyte: %d\n", (int)sizeof(dpct::device_ptr));
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
  compressed = new memobj<BYTE>(inlen * TYPICAL, "hf::out4B");

  __scratch = new memobj<RAW>(inlen * FAILSAFE, 1, 1, "hf::__scratch");
  scratch4 = new memobj<H4>(inlen, 1, 1, "hf::scratch4");
  scratch8 = new memobj<H8>(inlen, 1, 1, "hf::scratch8");

  bk4 = new memobj<H4>(bklen, 1, 1, "hf::book4");
  bk8 = new memobj<H8>(bklen, 1, 1, "hf::book8");

  revbk4 = new memobj<BYTE>(revbk4_bytes(bklen), 1, 1, "hf::revbk4");
  revbk8 = new memobj<BYTE>(revbk8_bytes(bklen), 1, 1, "hf::revbk8");

  // encoded buffer
  __bitstream = new memobj<RAW>(inlen * FAILSAFE / 2, 1, 1, "hf::__bitstrm");
  bitstream4 = new memobj<H4>(inlen / 2, 1, 1, "hf::bitstrm4");
  bitstream8 = new memobj<H8>(inlen / 2, 1, 1, "hf::bitstrm8");

  par_nbit = new memobj<M>(pardeg, 1, 1, "hf::par_nbit");
  par_ncell = new memobj<M>(pardeg, 1, 1, "hf::par_ncell");
  par_entry = new memobj<M>(pardeg, 1, 1, "hf::par_entry");

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

  numSMs = dpct::dev_mgr::instance().get_device(0).get_max_compute_units();

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
TPL HF_CODEC* HF_CODEC::buildbook(
    uint32_t* freq, int const bklen, void* stream)
{
  psz::hf_buildbook<CUDA, E, H4>(
      freq, bklen, bk4->dptr(), revbk4->dptr(), revbook_bytes(bklen),
      &_time_book, (cudaStream_t)stream);

  return this;
}
#endif

// using CPU huffman
TPL HF_CODEC* HF_CODEC::buildbook(MemU4* freq, int const bklen, void* stream)
{
  psz::hf_buildbook<SEQ, E, H4>(
      freq->control({D2H})->hptr(), bklen, bk4->hptr(), revbk4->hptr(),
      revbk4_bytes(bklen), &_time_book, (dpct::queue_ptr)stream);
  bk4->control({Async_H2D}, (dpct::queue_ptr)stream);
  revbk4->control({Async_H2D}, (dpct::queue_ptr)stream);

  book_desc->book = __encdtype == U4 ? (void*)bk4->dptr() : (void*)bk8->dptr();

  hist_view->asaviewof(freq);  // for analysis

  return this;
}

TPL HF_CODEC* HF_CODEC::encode(
    E* in, size_t const inlen, uint8_t** out, size_t* outlen, void* stream)
{
  _time_lossless = 0;

  pszhf_header header;

  // So far, the enc scheme has been deteremined.

  psz::phf_coarse_encode_rev2<E, H4, M>(
      in, inlen, book_desc, bitstream_desc, &header.total_nbit,
      &header.total_ncell, &_time_lossless, stream);

  phf_memcpy_merge(
      header, inlen, book_desc->bklen, bitstream_desc->sublen,
      bitstream_desc->pardeg, stream);

  *out = compressed->dptr();
  *outlen = header.compressed_size();

  return this;
}

TPL HF_CODEC* HF_CODEC::decode(
    uint8_t* in_compressed, E* out_decompressed, void* stream,
    bool header_on_device)
{
  auto queue = (sycl::queue*)stream;

  Header header;
  if (header_on_device)
    queue->memcpy(&header, in_compressed, sizeof(header)).wait();

  psz::phf_coarse_decode<E, H4, M>(
      ACCESSOR(BITSTREAM, H4), ACCESSOR(REVBK, BYTE),
      revbk4_bytes(header.bklen), ACCESSOR(PAR_NBIT, M),
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
TPL void HF_CODEC::phf_memcpy_merge(
    Header& header, size_t const original_len, int const bklen,
    int const sublen, int const pardeg, void* stream)
try {
  auto queue = (sycl::queue*)stream;

  auto BARRIER = [&]() {
    if (stream)
      queue->wait();
    else
      dpct::get_current_device().queues_wait_and_throw();
  };

  constexpr auto D2D = dpct::device_to_device;

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

  queue->memcpy(compressed->dptr(), &header, sizeof(header)).wait_and_throw();

  // /* debug */ BARRIER();

  {
    auto dst = compressed->dptr() + header.entry[Header::REVBK];
    auto src = __encdtype == U4 ? revbk4->dptr() : revbk8->dptr();

    queue->memcpy(dst, src, nbyte[Header::REVBK]);
  }
  {
    auto dst = compressed->dptr() + header.entry[Header::PAR_NBIT];
    auto src = par_nbit->dptr();
    queue->memcpy(dst, src, nbyte[Header::PAR_NBIT]);
  }
  {
    auto dst = compressed->dptr() + header.entry[Header::PAR_ENTRY];
    auto src = par_entry->dptr();
    queue->memcpy(dst, src, nbyte[Header::PAR_ENTRY]);
  }
  {
    auto dst = compressed->dptr() + header.entry[Header::BITSTREAM];
    auto src = __bitstream->dptr();
    queue->memcpy(dst, src, nbyte[Header::BITSTREAM]);
  }
}
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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
  dpct::device_ptr pbase0{0};
  size_t psize0{0};

  // [psz::TODO] not-impl exception
  // #warning "DPCT1007:99: Migration of cuMemGetAddressRange is not
  // supported."
  // /*
  // DPCT1007:99: Migration of cuMemGetAddressRange is not supported.
  // */
  // cuMemGetAddressRange(&pbase0, &psize0, (dpct::device_ptr)VAR);
  // printf(
  //     "%s:\n"
  //     "\t(supposed) pointer : %p\n"
  //     // "\t(supposed) bytes   : %'9lu\n"
  //     "\t(queried)  pbase0  : %p\n"
  //     "\t(queried)  psize0  : %'9lu\n",
  //     SYM_name.c_str(), (void*)VAR,
  //     // (size_t)rc.nbyte[SYM],
  //     (void*)&pbase0, psize0);
  // pbase0 = 0, psize0 = 0;
}

}  // namespace cusz

#undef ACCESSOR
#undef TPL
#undef HF_CODEC

#endif /* ABBC78E4_3E65_4633_9BEA_27823AB7C398 */
