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

#include <string>

#include "hf/hf.hh"
#include "kernel/hist.hh"
#include "mem.hh"
#include "stat.hh"
#include "port.hh"
#include "utils/print_arr.hh"
#include "utils/viewer.hh"

using B = uint8_t;
using F = u4;

namespace {

szt tune_coarse_huffman_sublen(szt len) {
  int current_dev = 0;
  GpuSetDevice(current_dev);
  GpuDeviceProp dev_prop{};
  GpuGetDeviceProperties(&dev_prop, current_dev);

  // auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto nSM = dev_prop.multiProcessorCount;
  auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
  auto deflate_nthread = allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;
  auto optimal_sublen = psz_utils::get_npart(len, deflate_nthread);
  optimal_sublen = psz_utils::get_npart(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) * HuffmanHelper::BLOCK_DIM_DEFLATE;

  return optimal_sublen;
}

void print_tobediscarded_info(float time_in_ms, string fn_name) {
  auto title = "[psz::info::discard::" + fn_name + "]";
  printf("%s time (ms): %.6f\n", title.c_str(), time_in_ms);
}

template <typename T>
float print_GBps(szt len, float time_in_ms, string fn_name) {
  auto B_to_GiB = 1.0 * 1024 * 1024 * 1024;
  auto GiBps = len * sizeof(T) * 1.0 / B_to_GiB / (time_in_ms / 1000);
  auto title = "[psz::info::res::" + fn_name + "]";
  printf("%s shortest time (ms): %.6f\thighest throughput (GiB/s): %.2f\n", 
    title.c_str(), time_in_ms, GiBps);
  return GiBps;
}


}

template <typename E, typename H = u4>
void hf_run(std::string fname, size_t const x, size_t const y, size_t const z)
{
  /* For demo, we use 3600x1800 CESM data. */
  auto len = x * y * z;

  constexpr auto booklen = 1024;

  auto sublen = tune_coarse_huffman_sublen(len);
  auto pardeg = psz_utils::get_npart(len, sublen);

  auto od = new pszmem_cxx<E>(len, 1, 1, "original");
  auto xd = new pszmem_cxx<E>(len, 1, 1, "decompressed");
  auto ht = new pszmem_cxx<F>(booklen, 1, 1, "histogram");
  uint8_t* d_compressed;

  od->control({Malloc, MallocHost})
      ->file(fname.c_str(), FromFile)
      ->control({H2D});
  xd->control({Malloc, MallocHost});
  ht->control({Malloc, MallocHost});

  /* a casual peek */
  printf("peeking data, 20 elements\n");
  psz::peek_data<E>(od->control({D2H})->hptr(), 20);

  GpuStreamT stream;
  GpuStreamCreate(&stream);

  dim3 len3 = dim3(x, y, z);

  float time_hist;

  psz::histogram<PROPER_GPU_BACKEND, E>(
      od->dptr(), len, ht->dptr(), booklen, &time_hist, stream);

  cusz::HuffmanCodec<E, u4> codec;
  codec.init(len, booklen, pardeg /* not optimal for perf */);

  // GpuMalloc(&d_compressed, len * sizeof(E) / 2);
  B* __out;

  // float  time;
  size_t outlen;
  codec.build_codebook(ht, booklen, stream);

  E* d_oridup;
  GpuMalloc(&d_oridup, sizeof(E) * len);
  GpuMemcpy(d_oridup, od->dptr(), sizeof(E) * len, GpuMemcpyD2D);

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
    time_decomp_lossless = std::min(time_decomp_lossless, codec.time_lossless());
  }
  print_GBps<f4>(len, time_decomp_lossless, "decomp_hf_decode");

  // psz::cppstl_identical(h_xd, h_d, len);
  auto identical =
      psz::thrustgpu_identical(xd->dptr(), od->dptr(), sizeof(E), len);

  if (identical)
    cout << ">>>>  IDENTICAL." << endl;
  else
    cout << "!!!!  ERROR: NOT IDENTICAL." << endl;

  GpuStreamDestroy(stream);

  /* a casual peek */
  printf("peeking xdata, 20 elements\n");
  psz::peek_data<E>(xd->control({D2H})->hptr(), 20);
  printf("\n");
}

int main(int argc, char** argv)
{
  if (argc < 6) {
    printf("PROG  /path/to/datafield  X  Y  Z  QuantType\n");
    printf("0     1                   2  3  4  5\n");
    exit(0);
  }
  else {
    auto fname = std::string(argv[1]);
    auto x = atoi(argv[2]);
    auto y = atoi(argv[3]);
    auto z = atoi(argv[4]);
    auto type = std::string(argv[5]);

    // if (type == "ui8")
    //   hf_run<uint8_t, u4>(fname, x, y, z);
    // else if (type == "ui16")
    //   hf_run<uint16_t, u4>(fname, x, y, z);
    // else if (type == "ui32")
    //   hf_run<u4, u4>(fname, x, y, z);
    // else
    //   hf_run<uint16_t, u4>(fname, x, y, z);

    // 23-06-04 restricted to u4 for quantization code
    hf_run<u4, u4>(fname, x, y, z);
  }

  return 0;
}
