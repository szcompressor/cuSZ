/**
 * @file bin_hist.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-07-25
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <string>

#include "ex_utils.hh"
#include "kernel.hh"
#include "mem.hh"
#include "stat.hh"
#include "typing.hh"

#define BASE false
#define OPTIM true

using T = u4;

template <psz_runtime policy, typename T>
void hist(
    bool optim, T* whole_numbers, size_t const len, uint32_t* hist, size_t const bklen, float* t,
    cudaStream_t stream)
{
  int hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use, hist_generic_repeat;
  psz::module::GPU_histogram_generic_optimizer_on_initialization<T>(
      len, bklen, hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use,
      hist_generic_repeat);

  if (optim)
    psz::module::GPU_histogram_Cauchy<T>(whole_numbers, len, hist, bklen, stream);
  else
    psz::module::GPU_histogram_generic<T>(
        whole_numbers, len, hist, bklen, hist_generic_grid_dim, hist_generic_block_dim,
        hist_generic_shmem_use, hist_generic_repeat, stream);
}

template <typename T>
void real_data_test(size_t len, size_t bklen, string fname)
{
  auto wn = new memobj<T>(len, "whole numbers", {Malloc, MallocHost});
  auto bs = new memobj<u4>(bklen, "base-ser", {Malloc, MallocHost});
  auto os = new memobj<u4>(bklen, "optim-ser", {Malloc, MallocHost});
  auto bg = new memobj<u4>(bklen, "base-gpu", {Malloc, MallocHost});
  auto og = new memobj<u4>(bklen, "optim-gpu", {Malloc, MallocHost});

  wn->file(fname.c_str(), FromFile)->control({H2D});

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float tbs, tos, tbg, tog;

  hist<SEQ, T>(BASE, wn->hptr(), len, bs->hptr(), bklen, &tbs, stream);
  hist<SEQ, T>(OPTIM, wn->hptr(), len, os->hptr(), bklen, &tos, stream);

  hist<PROPER_RUNTIME, T>(BASE, wn->dptr(), len, bg->dptr(), bklen, &tbg, stream),
      bg->control({D2H});
  hist<PROPER_RUNTIME, T>(OPTIM, wn->dptr(), len, og->dptr(), bklen, &tog, stream),
      og->control({D2H});

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
      auto fbs = bs->hptr(i), fbg = bg->hptr(i);
      auto fos = os->hptr(i), fog = og->hptr(i);
      if (fbs != 0 or fos != 0 or fbg != 0 or fog != 0)
        printf(
            "%-4u(%4d) %10u %10u %10u %10u\n",  //
            i, i - (int)bklen / 2, fbs, fbg, fos, fog);
    }
  }

  delete wn;
  delete bg, delete bs;
  delete og, delete os;

  cudaStreamDestroy(stream);
}

template <typename T>
void dummy_data_test()
{
  auto len = 1000000;
  auto bklen = 1024;

  auto wn = new memobj<T>(len, "whole numbers", {Malloc, MallocHost});
  auto serial = new memobj<u4>(bklen, "optim-ser", {MallocHost});
  auto gpu = new memobj<u4>(bklen, "optim-gpu", {Malloc, MallocHost});

  for (auto i = 0; i < len; i += 1) wn->hptr(i) = bklen / 2;
  for (auto i = 2; i < len - 10; i += 100) {
    wn->hptr(i - 1) = bklen / 2 - 1;
    wn->hptr(i - 2) = bklen / 2 + 1;
  }
  wn->control({H2D});

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float tbs, tos, tbg, tog;

  hist<SEQ, T>(OPTIM, wn->hptr(), len, serial->hptr(), bklen, &tos, stream);
  hist<PROPER_RUNTIME, T>(OPTIM, wn->dptr(), len, gpu->dptr(), bklen, &tog, stream);
  gpu->control({D2H});

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("GPU error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  {
    printf("%-10s %10s %10s\n", "idx ( rel)", "sp-ser", "sp-gpu");
    for (auto i = 0; i < bklen; i++) {
      auto f1 = serial->hptr(i), f2 = gpu->hptr(i);
      if (f1 != 0 or f2 != 0) printf("%-4u(%4d) %10u %10u\n", i, i - bklen / 2, f1, f2);
    }
  }

  delete wn;
  delete serial, delete gpu;

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