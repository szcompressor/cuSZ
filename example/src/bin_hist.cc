#include <cuda_runtime.h>

#include <cstdio>
#include <string>

#include "detail/compare.hh"
#include "detail/port.hh"
#include "kernel/hist.hh"
#include "mem/cxx_backends.h"
#include "utils/io.hh"

#define BASE false
#define OPTIM true

using T = u4;

template <psz_runtime policy, typename T>
void hist(
    bool optim, T* whole_numbers, size_t const len, uint32_t* hist, size_t const bklen, float* t,
    cudaStream_t stream)
{
  int hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use, hist_generic_repeat;
  psz::module::GPU_histogram_generic<T>::init(
      len, bklen, hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use,
      hist_generic_repeat);

  if (optim)
    psz::module::GPU_histogram_Cauchy<T>::kernel(whole_numbers, len, hist, bklen, stream);
  else
    psz::module::GPU_histogram_generic<T>::kernel(
        whole_numbers, len, hist, bklen, hist_generic_grid_dim, hist_generic_block_dim,
        hist_generic_shmem_use, hist_generic_repeat, stream);
}

template <typename T>
void real_data_test(size_t len, size_t bklen, string fname)
{
  auto wn_h = MAKE_UNIQUE_HOST(T, len);
  auto wn_d = MAKE_UNIQUE_DEVICE(T, len);
  auto bs_h = MAKE_UNIQUE_HOST(u4, bklen);
  auto os_h = MAKE_UNIQUE_HOST(u4, bklen);
  auto bg_h = MAKE_UNIQUE_HOST(u4, bklen);
  auto bg_d = MAKE_UNIQUE_DEVICE(u4, bklen);
  auto og_h = MAKE_UNIQUE_HOST(u4, bklen);
  auto og_d = MAKE_UNIQUE_DEVICE(u4, bklen);

  _portable::utils::fromfile(fname, wn_h.get(), len);
  memcpy_allkinds<H2D>(wn_d.get(), wn_h.get(), len);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float tbs, tos, tbg, tog;

  hist<SEQ, T>(BASE, wn_h.get(), len, bs_h.get(), bklen, &tbs, stream);
  hist<SEQ, T>(OPTIM, wn_h.get(), len, os_h.get(), bklen, &tos, stream);

  hist<PROPER_RUNTIME, T>(BASE, wn_d.get(), len, bg_d.get(), bklen, &tbg, stream);
  memcpy_allkinds<D2H>(bg_h.get(), bg_d.get(), bklen);
  hist<PROPER_RUNTIME, T>(OPTIM, wn_d.get(), len, og_d.get(), bklen, &tog, stream);
  memcpy_allkinds<D2H>(og_h.get(), og_d.get(), bklen);

  auto GBps = [&](auto bytes, auto millisec) {
    return 1.0 * bytes / (1024 * 1024 * 1024) / (millisec / 1000);
  };
  printf(
      "CPU time, baseline: %5.2f, optim: %5.2f (speedup: %5.2fx)\n"
      "CPU GBps, baseline: %5.2f, optim: %5.2f\n",
      tbs, tos, tbs / tos,  //
      GBps(len * sizeof(T), tbs), GBps(len * sizeof(T), tos));
  printf(
      "GPU time, baseline: %5.2f, optim: %5.2f (speedup: %5.2fx)\n"
      "GPU GBps, baseline: %5.2f, optim: %5.2f\n",
      tbg, tog, tbg / tog,  //
      GBps(len * sizeof(T), tbg), GBps(len * sizeof(T), tog));
  printf("\n");

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("GPU error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  {
    printf(
        "%-10s %10s %10s %10s %10s\n",  //
        "idx ( rel)", "base-ser", "base-gpu", "optim-ser", "optim-gpu");
    for (auto i = 0; i < bklen; i++) {
      auto fbs = bs_h[i], fbg = bg_h[i];
      auto fos = os_h[i], fog = og_h[i];
      if (fbs != 0 or fos != 0 or fbg != 0 or fog != 0)
        printf(
            "%-4u(%4d) %10u %10u %10u %10u\n",  //
            i, i - (int)bklen / 2, fbs, fbg, fos, fog);
    }
  }

  cudaStreamDestroy(stream);
}

template <typename T>
void dummy_data_test()
{
  auto len = 1000000;
  auto bklen = 1024;

  auto wn_h = MAKE_UNIQUE_HOST(T, len);
  auto wn_d = MAKE_UNIQUE_DEVICE(T, len);
  auto serial_h = MAKE_UNIQUE_HOST(u4, bklen);
  auto gpu_h = MAKE_UNIQUE_HOST(u4, bklen);
  auto gpu_d = MAKE_UNIQUE_DEVICE(u4, bklen);

  for (auto i = 0; i < len; i += 1) wn_h[i] = bklen / 2;
  for (auto i = 2; i < len - 10; i += 100) {
    wn_h[i - 1] = bklen / 2 - 1;
    wn_h[i - 2] = bklen / 2 + 1;
  }
  memcpy_allkinds<H2D>(wn_d.get(), wn_h.get(), len);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float tbs, tos, tbg, tog;

  hist<SEQ, T>(OPTIM, wn_h.get(), len, serial_h.get(), bklen, &tos, stream);
  hist<PROPER_RUNTIME, T>(OPTIM, wn_d.get(), len, gpu_d.get(), bklen, &tog, stream);
  memcpy_allkinds<D2H>(gpu_h.get(), gpu_d.get(), bklen);

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("GPU error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  {
    printf("%-10s %10s %10s\n", "idx ( rel)", "sp-ser", "sp-gpu");
    for (auto i = 0; i < bklen; i++) {
      auto f1 = serial_h[i], f2 = gpu_h[i];
      if (f1 != 0 or f2 != 0) printf("%-4u(%4d) %10u %10u\n", i, i - bklen / 2, f1, f2);
    }
  }

  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
  if (argc < 5) {
    printf("PROG /path/to/datafield X Y Z \n");
    printf("0    1                  2 3 4 \n");
    exit(0);
  }
  else {
    auto fname = std::string(argv[1]);
    auto x = atoi(argv[2]);
    auto y = atoi(argv[3]);
    auto z = atoi(argv[4]);

    auto len = x * y * z;
    auto bklen = 1024;

    printf("dummy data test:\n");
    dummy_data_test<T>();

    printf("\nreal data test:\n");
    real_data_test<T>(len, bklen, fname);
  }

  return 0;
}
