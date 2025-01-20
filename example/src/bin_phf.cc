/**
 * @file bin_hf.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-15
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>
#include <string>

#include "hf.h"
#include "hfclass.hh"
#include "mem/cxx_backends.h"
#include "module/cxx_module.hh"
#include "stat.hh"
#include "utils/io.hh"
#include "utils/print_arr.hh"

namespace utils = _portable::utils;

using B = uint8_t;
using F = u4;

string fname;
bool dump_book;
int sublen, pardeg;
uint8_t* d_compressed;
float time_hist;
size_t outlen;
cudaStream_t stream;
float time_encode = (float)INT_MAX;
float time_decode = (float)INT_MAX;

#define PEEK_DATA                        \
  printf("peeking data, 20 elements\n"); \
  psz::peek_data<E>(h_oridata, 20), printf("\n");

#define PEEK_XDATA                                                \
  printf("peeking xdata, 20 elements\n");                         \
  memcpy_allkinds<E, D2H>(h_decomp, d_decomp, len), printf("\n"); \
  psz::peek_data<E>(h_decomp, 20), printf("\n");

#define CHECK_INTEGRITY                                                               \
  auto identical =                                                                    \
      psz::module::GPU_identical(d_decomp.get(), d_oridata.get(), sizeof(E), len, 0); \
  printf("%s\n", identical ? ">>>>  IDENTICAL" : "!!!!  ERROR: DIFFERENT");

#define MALLOC_BUFFERS                                                 \
  auto d_oridata = GPU_make_unique(malloc_d<E>(len), GPU_deleter_d()); \
  auto h_oridata = GPU_make_unique(malloc_h<E>(len), GPU_deleter_h()); \
  auto d_decomp = GPU_make_unique(malloc_d<E>(len), GPU_deleter_d());  \
  auto h_decomp = GPU_make_unique(malloc_h<E>(len), GPU_deleter_h());  \
  auto d_hist = GPU_make_unique(malloc_d<F>(bklen), GPU_deleter_d());

#define LOAD_FILE                                       \
  utils::fromfile(fname.c_str(), h_oridata.get(), len); \
  memcpy_allkinds<H2D>(d_oridata.get(), h_oridata.get(), len);

#define PREPARE   \
  MALLOC_BUFFERS; \
  LOAD_FILE;      \
  cudaStreamCreate(&stream);

#define CLEANUP cudaStreamDestroy(stream);

#define PRINT_REPORT                                    \
  print_GBps<E>(len, time_encode, "hf_encode");         \
  print_GBps<u1>(outlen, time_decode, "hf_decode");     \
  printf("Huffman in  bytes:\t%lu\n", len * sizeof(E)); \
  printf("Huffman out bytes:\t%lu\n", outlen);          \
  printf("Huffman CR (out/in):\t%.2lf\n", len * sizeof(E) * 1.0 / outlen);

namespace {
void print_tobediscarded_info(float time_in_ms, string fn_name)
{
  auto title = "[psz::info::discard::" + fn_name + "]";
  printf("%s time (ms): %.6f\n", title.c_str(), time_in_ms);
}

template <typename T>
float print_GBps(size_t len, float time_in_ms, string fn_name)
{
  auto B_to_GiB = 1.0 * 1024 * 1024 * 1024;
  auto GiBps = len * sizeof(T) * 1.0 / B_to_GiB / (time_in_ms / 1000);
  auto title = "[psz::info::res::" + fn_name + "]";
  printf("%s %.2f GiB/s at %.6f ms\n", title.c_str(), GiBps, time_in_ms);
  return GiBps;
}

}  // namespace

template <typename E, typename H = u4>
void hf_run(std::string fname, size_t const len, size_t const bklen = 1024)
{
  PREPARE;

  capi_phf_coarse_tune(len, &sublen, &pardeg);

  int hist_generic_grid_dim, hist_generic_block_dim, shmem_use, repeat;
  psz::module::GPU_histogram_generic_optimizer_on_initialization<E>(
      len, bklen, hist_generic_grid_dim, hist_generic_block_dim, shmem_use, repeat);

  psz::module::GPU_histogram_generic<E>(
      d_oridata.get(), len, d_hist.get(), bklen, hist_generic_grid_dim, hist_generic_block_dim,
      shmem_use, repeat, stream);
  phf::HuffmanCodec<E> codec(len, bklen, pardeg);

  codec.buildbook(d_hist.get(), stream);
  if (dump_book) codec.dump_internal_data("book", fname);

  for (auto i = 0; i < 10; i++) {
    codec.encode(d_oridata.get(), len, &d_compressed, &outlen, stream);
    time_encode = std::min(time_encode, codec.time_lossless());
    codec.decode(d_compressed, d_decomp.get(), stream);
    time_decode = std::min(time_decode, codec.time_lossless());
  }

  CHECK_INTEGRITY;
  PRINT_REPORT;

  CLEANUP;
}

int main(int argc, char** argv)
{
  if (argc < 6) {
    // clang-format off
    printf(
        "PROG  /path/to/data  X  Y  Z  bklen  [type: u1,u2,u4]  [dump book: true,false]\n"
        "0     1              2  3  4  5      [6:optional]      [7:optional]\n");
    // clang-format on
    exit(0);
  }
  else {
    fname = std::string(argv[1]);
    auto x = atoi(argv[2]), y = atoi(argv[3]), z = atoi(argv[4]);
    auto len = x * y * z;
    auto bklen = atoi(argv[5]);

    auto type = string("u1");
    if (argc == 7) type = std::string(argv[6]);
    if (argc == 8) dump_book = std::string(argv[7]) == "true";

    if (type == "u1") {
      printf("REVERT bklen to 256 for u1-type input.\n");
      hf_run<u1>(fname, len, 256);
    }
    else {
      if (type == "u2")
        hf_run<u2>(fname, len, bklen);
      else if (type == "u4")
        hf_run<u4>(fname, len, bklen);
      else
        hf_run<u4>(fname, len, bklen);
    }
  }

  return 0;
}
