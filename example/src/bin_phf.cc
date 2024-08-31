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
#include "mem.hh"
#include "module/cxx_module.hh"
#include "port.hh"
#include "stat.hh"
#include "utils/print_arr.hh"
#include "utils/viewer.hh"

using B = uint8_t;
using F = u4;

string fname;
bool dump_book;

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
  printf(
      "%s shortest time (ms): %.6f\thighest throughput (GiB/s): %.2f\n",
      title.c_str(), time_in_ms, GiBps);
  return GiBps;
}

}  // namespace

template <typename E, typename H = u4>
void hf_run(
    std::string fname, size_t const x, size_t const y, size_t const z,
    size_t const bklen = 1024)
{
  /* For demo, we use 3600x1800 CESM data. */
  auto len = x * y * z;

  int sublen, pardeg;
  capi_phf_coarse_tune(len, &sublen, &pardeg);

  auto od = new memobj<E>(len, "original", {Malloc, MallocHost});
  auto xd = new memobj<E>(len, "decompressed", {Malloc, MallocHost});
  auto ht = new memobj<F>(bklen, "histogram", {Malloc, MallocHost});
  od->file(fname.c_str(), FromFile)->control({H2D});

  uint8_t* d_compressed;

  /* a casual peek */
  printf("peeking data, 20 elements\n");
  psz::peek_data<E>(od->control({D2H})->hptr(), 20);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  dim3 len3 = dim3(x, y, z);

  float time_hist;

  pszcxx_compat_histogram_generic<PROPER_GPU_BACKEND, E>(
      od->dptr(), len, ht->dptr(), bklen, &time_hist, stream);

  phf::HuffmanCodec<E> codec(len, bklen, pardeg /* not optimal for perf */);

  // cudaMalloc(&d_compressed, len * sizeof(E) / 2);
  B* __out;

  // float  time;
  size_t outlen;
  codec.buildbook(ht->dptr(), stream);
  if (dump_book) codec.dump_internal_data("book", fname);

  E* d_oridup;
  cudaMalloc(&d_oridup, sizeof(E) * len);
  cudaMemcpy(d_oridup, od->dptr(), sizeof(E) * len, cudaMemcpyDeviceToDevice);

  auto time_comp_lossless = (float)INT_MAX;
  for (auto i = 0; i < 10; i++) {
    // codec.encode(od->dptr(), len, &d_compressed, &outlen, stream);
    codec.encode(d_oridup, len, &d_compressed, &outlen, stream);

    print_tobediscarded_info(codec.time_lossless(), "comp_hf_encode");
    time_comp_lossless = std::min(time_comp_lossless, codec.time_lossless());
  }
  print_GBps<f4>(len, time_comp_lossless, "comp_hf_encode");

  printf("Huffman in  len:\t%lu\n", len);
  printf("Huffman out len:\t%lu\n", outlen);
  printf(
      "\"Huffman CR = sizeof(E) * len / outlen\", where outlen is byte "
      "count:\t%.2lf\n",
      len * sizeof(E) * 1.0 / outlen);

  auto time_decomp_lossless = (float)INT_MAX;
  for (auto i = 0; i < 10; i++) {
    codec.decode(d_compressed, xd->dptr(), stream);

    print_tobediscarded_info(codec.time_lossless(), "decomp_hf_decode");
    time_decomp_lossless =
        std::min(time_decomp_lossless, codec.time_lossless());
  }
  print_GBps<f4>(len, time_decomp_lossless, "decomp_hf_decode");

  // psz::cppstl::CPU_identical(h_xd, h_d, len);
  auto identical =
      psz::thrustgpu::GPU_identical(xd->dptr(), od->dptr(), sizeof(E), len);

  if (identical)
    cout << ">>>>  IDENTICAL." << endl;
  else
    cout << "!!!!  ERROR: NOT IDENTICAL." << endl;

  cudaStreamDestroy(stream);

  /* a casual peek */
  printf("peeking xdata, 20 elements\n");
  psz::peek_data<E>(xd->control({D2H})->hptr(), 20);
  printf("\n");
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
    auto x = atoi(argv[2]);
    auto y = atoi(argv[3]);
    auto z = atoi(argv[4]);
    auto bklen = atoi(argv[5]);

    auto type = string("u1");
    if (argc == 7) type = std::string(argv[6]);
    if (argc == 8) dump_book = std::string(argv[7]) == "true";

    if (type == "u1") {
      printf("REVERT bklen to 256 for u1-type input.");
      hf_run<u1>(fname, x, y, z, 256);
    }
    else {
      if (type == "u2")
        hf_run<u2>(fname, x, y, z, bklen);
      else if (type == "u4")
        hf_run<u4>(fname, x, y, z, bklen);
      else
        hf_run<u4>(fname, x, y, z, bklen);
    }
  }

  return 0;
}
