/**
 * @file bin_pipeline_proto.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-10
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>

using std::cout;
using std::endl;

#include "cusz.h"
#include "hf/hf.hh"
#include "hf/hf_bookg.hh"
#include "hf/hf_codecg.hh"
#include "hf/hf_struct.h"
#include "kernel/l23.hh"
#include "kernel/lproto.hh"
#include "kernel/spv_gpu.hh"
#include "mem/layout_cxx.hh"
#include "stat/stat.hh"
#include "utils/cuda_err.cuh"
#include "utils/io.hh"
#include "utils/print_gpu.hh"
#include "utils/viewer.hh"

namespace alpha {

struct {
  uint32_t x, y, z;
} len3;

typedef struct config {
  double eb;
  int radius;
} config;

typedef struct header_superset {
  cusz_header header;
  int nnz;
  size_t total_nbit;
  size_t total_ncell;
  int hf_pardeg;
  int hf_sublen;
} header_superset;

struct hf_set {
  hf_book* book_desc;
  // hf_chunk*     chunk_desc_d;
  // hf_chunk*     chunk_desc_h;
  hf_bitstream* bitstream_desc;

  uint8_t* revbook;
  size_t revbook_nbyte;
  uint8_t* out;
  size_t outlen;

  uint8_t* hl_comp;  // high level encoder API
  size_t hl_complen;
};

// decpreated, use rt_config.h func instead
template <typename H>
int reversebook_nbyte(int booklen)
{
  return sizeof(H) * (2 * sizeof(H) * 8) + sizeof(H) * booklen;
}

}  // namespace alpha

template <
    typename T, typename E = uint32_t, typename FP = T, typename H = uint32_t,
    typename M = uint32_t>
cusz_error_status allocate_data(
    dim3 len3, alpha::hf_set* hf, cusz::HuffmanCodec<E, H, M>* codec,
    alpha::config* config)
{
  auto x = len3.x, y = len3.y, z = len3.z;
  size_t len = x * y * z;

  // coarse-grained
  auto sublen = 768;
  auto pardeg = (len + sublen - 1) / sublen;
  cout << "pardeg\t" << pardeg << endl;
  hf->book_desc->booklen = config->radius * 2;
  hf->bitstream_desc->sublen = sublen;
  hf->bitstream_desc->pardeg = pardeg;
  hf->revbook_nbyte = alpha::reversebook_nbyte<H>(hf->book_desc->booklen);

  codec->init(len, hf->book_desc->booklen, pardeg);

  CHECK_CUDA(cudaMalloc(
      &hf->book_desc->freq, sizeof(uint32_t) * hf->book_desc->booklen));
  CHECK_CUDA(cudaMalloc(&hf->hl_comp, 2 * len));

  // hf->d_metadata)
  return cusz_error_status::CUSZ_SUCCESS;
}

template <
    typename T, typename E = uint32_t, typename FP = T, typename H = uint32_t,
    typename M = uint32_t>
cusz_error_status deallocate_data(
    pszmempool_cxx<T, E, H>* mem, alpha::hf_set* hf,
    // cusz::HuffmanCodec<E, H, M>* codec,
    alpha::config* config)
{
  delete mem->od, delete mem->xd, delete mem;

  CHECK_CUDA(cudaFree(hf->book_desc->freq));
  CHECK_CUDA(cudaFree(hf->hl_comp));

  return cusz_error_status::CUSZ_SUCCESS;
}

template <
    typename T, typename E, typename FP, typename H = uint32_t,
    typename M = uint32_t>
cusz_error_status compressor(
    pszmempool_cxx<T, E, H>* mem, alpha::hf_set* hf,
    cusz::HuffmanCodec<E, H, M>* codec, alpha::config* config,
    alpha::header_superset* header_st, bool use_proto, cudaStream_t stream)
{
  float time_pq = 0, time_hist = 0, time_spv = 0;
  // , _time_book = 0, time_encoding = 0;

  if (not use_proto) {
    cout << "using optimized comp. kernel\n";
    psz_comp_l23<T, E, FP>(
        mem->od->dptr(), mem->od->template len3<dim3>(), config->eb,
        config->radius, mem->ectrl_lrz(), mem->outlier_space(), &time_pq,
        stream);
  }
  else {
    cout << "using prototype comp. kernel\n";
    // psz_comp_lproto<T, E>(                                              //
    //     data->oridata->dptr(), data->len3, config->eb, config->radius,  //
    //     data->errctrl->dptr(), data->outlier->dptr(), &time, stream);
    throw runtime_error("prototype is disabled");
  }

  cout << "time-eq\t" << time_pq << endl;

  // TODO better namesapce to specify this is a firewall
  psz::spv_gather<T, M>(
      mem->outlier_space(), mem->len, mem->outlier_val(), mem->outlier_idx(),
      &header_st->nnz, &time_spv, stream);

  cout << "time-spv\t" << time_spv << endl;
  cout << "nnz\t" << header_st->nnz << endl;

  psz::stat::histogram<psz_policy::CUDA, E>(
      mem->ectrl_lrz(), mem->len, hf->book_desc->freq, hf->book_desc->booklen,
      &time_hist, stream);

  cout << "time-hist\t" << time_hist << endl;

  codec->build_codebook(hf->book_desc->freq, hf->book_desc->booklen, stream);
  codec->encode(
      mem->ectrl_lrz(), mem->len, &hf->hl_comp, &hf->hl_complen, stream);

  return cusz_error_status::CUSZ_SUCCESS;
}

template <
    typename T, typename E, typename FP, typename H = uint32_t,
    typename M = uint32_t>
cusz_error_status decompressor(
    pszmempool_cxx<T, E, H>* mem, alpha::hf_set* hf,
    cusz::HuffmanCodec<E, H, M>* codec, alpha::header_superset* header_st,
    bool use_proto, cudaStream_t stream)
{
  float time_scatter = 0,
        // time_hf = 0,
      time_d_pq = 0;

  psz::spv_scatter<T, uint32_t>(
      mem->outlier_val(), mem->outlier_idx(), header_st->nnz, mem->xd->dptr(),
      &time_scatter, stream);
  cout << "decomp-time-spv\t" << time_scatter << endl;

  codec->decode(hf->hl_comp, mem->ectrl_lrz());

  if (not use_proto) {
    cout << "using optimized comp. kernel\n";
    psz_decomp_l23<T, E, FP>(
        mem->ectrl_lrz(), mem->od->template len3<dim3>(), mem->xd->dptr(),
        header_st->header.eb, header_st->header.radius, mem->xd->dptr(),
        &time_d_pq, stream);
  }
  else {
    cout << "using prototype comp. kernel\n";
    // psz_decomp_lproto<T, E, FP>(                      //
    //     data->errq, data->len3, data->outlier_, data->outlier_idx, 0,  //
    //     input header_st->header.eb, header_st->header.radius, // input
    //     (config) data->xdata, // output &time_d_pq, stream);
    throw runtime_error("prototype is disabled to later fix");
  }

  cout << "decomp-time-pq\t" << time_d_pq << endl;

  return cusz_error_status::CUSZ_SUCCESS;
}

template <typename T, typename E>
void f(
    std::string& fname, dim3 len3, alpha::hf_set* hf,
    alpha::header_superset* header_st, alpha::config* config, bool use_proto)
{
  using FP = T;
  using M = uint32_t;
  using H = uint32_t;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto mem =
      new pszmempool_cxx<T, E, H>(len3.x, config->radius, len3.y, len3.z);

  cusz::HuffmanCodec<E, uint32_t, uint32_t> codec;

  allocate_data<T, E, FP, H, M>(len3, hf, &codec, config);
  mem->od->control({Malloc, MallocHost})
      ->file(fname.c_str(), FromFile)
      ->control({H2D});
  mem->xd->control({Malloc, MallocHost});

  compressor<T, E, FP, H, M>(
      mem, hf, &codec, config, header_st, use_proto, stream);

  decompressor<T, E, FP, H, M>(mem, hf, &codec, header_st, use_proto, stream);

  /* view quality */ psz::eval_dataquality_gpu(
      mem->xd->dptr(), mem->od->dptr(), mem->len);

  deallocate_data<T, E, FP, H, M>(mem, hf, config);

  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
  if (argc < 6) {
    printf("0    1             2     3 4 5 6  [7]     [8:128]  [9:yes]\n");
    printf(
        "PROG /path/to/file DType X Y Z EB [EType] [Radius] [Use "
        "Prototype]\n");
    printf(" 2  DType: \"F\" for `float`, \"D\" for `double`\n");
    printf(
        "[7] EType: \"ui{8,16,32}\" for `uint{8,16,32}_t` as quant-code "
        "type\n");
    exit(0);
  }

  auto fname = std::string(argv[1]);
  auto dtype = std::string(argv[2]);
  auto x = atoi(argv[3]);
  auto y = atoi(argv[4]);
  auto z = atoi(argv[5]);
  auto eb = atof(argv[6]);

  std::string etype;
  if (argc > 7)
    etype = std::string(argv[7]);
  else
    etype = "ui16";

  int radius;
  if (argc > 8)
    radius = atoi(argv[8]);
  else
    radius = 128;

  bool use_proto;
  if (argc > 9)
    use_proto = std::string(argv[9]) == "yes";
  else
    use_proto = false;

  using T = float;
  using E = uint16_t;

  auto len3 = dim3(x, y, z);

  auto header_st = new alpha::header_superset;
  header_st->header.eb = eb;
  header_st->header.radius = radius;

  auto config = new alpha::config{eb, radius};

  auto hf = new alpha::hf_set;
  hf->book_desc = new hf_book;
  hf->bitstream_desc = new hf_bitstream;
  //// dispatch

  auto radius_legal = [&](int const sizeof_T) {
    size_t upper_bound = 1lu << (sizeof_T * 8);
    cout << upper_bound << endl;
    cout << radius * 2 << endl;
    if ((radius * 2) > upper_bound)
      throw std::runtime_error("Radius overflows error-quantization type.");
  };

  // 23-06-04 restricted to u4 for quantization code

  if (dtype == "F") {
    // if (etype == "ui8") {
    //     radius_legal(1);
    //     f<float, uint8_t>(fname, len3, hf, header_st, config, use_proto);
    // }
    // else if (etype == "ui16") {
    //     radius_legal(2);
    //     f<float, uint16_t>(fname, len3, hf, header_st, config, use_proto);
    // }
    // else if (etype == "ui32") {
    // }
    radius_legal(4);
    f<float, uint32_t>(fname, len3, hf, header_st, config, use_proto);
  }
  else if (dtype == "D") {
    // if (etype == "ui8") {
    //     radius_legal(1);
    //     f<double, uint8_t>(fname, len3, hf, header_st, config, use_proto);
    // }
    // else if (etype == "ui16") {
    //     radius_legal(2);
    //     f<double, uint16_t>(fname, len3, hf, header_st, config, use_proto);
    // }
    // else if (etype == "ui32") {
    // }
    radius_legal(4);
    f<double, uint32_t>(fname, len3, hf, header_st, config, use_proto);
  }
  else
    throw std::runtime_error("not a valid dtype.");

  return 0;
}
