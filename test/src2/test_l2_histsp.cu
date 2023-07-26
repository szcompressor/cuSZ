/**
 * @file test_l2_histsp.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-05-20
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdio>
#include <iostream>
#include <random>

#include "kernel2/detail2/histsp.inl"
#include "utils2/memseg_cxx.hh"

using std::cout;
using std::endl;

using T = uint32_t;
using FQ = uint32_t;

constexpr auto R = 2;
constexpr auto K = 2 * R + 1;

float dist1[] = {0.01, 0.09, 0.8, 0.09, 0.01};
float dist2[] = {0.01, 0.04, 0.9, 0.04, 0.01};
float dist3[] = {0.005, 0.015, 0.96, 0.015, 0.005};

void gen_symetric_dist(
    T *in, size_t inlen, float dist[], int distlen = 5, int offset = 512)
{
  cout << "offset: " << offset << endl;

  auto R = (distlen - 1) / 2;

  std::random_device rd;   // a seed source for the random number engine
  std::mt19937 gen(rd());  // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(0, inlen);

  for (auto _ = 0; _ < inlen; _++) { in[_] = offset; }
  for (auto i = 0; i < distlen; i++) {
    if (i - R == 0)
      continue;
    else {
      auto N = (int)(inlen * dist[i]);
      auto sym = (i - R) + offset;
      printf("sym: %d, num: %d\n", sym, N);
      for (auto _ = 0; _ < N; _++) {
        auto loc = distrib(gen);
        in[loc] = sym;
      }
    }
  }
}

template <int OUTLEN = 1024, int CHUNK = 32768, int NWARP = 8>
int f_histsp_kernel(size_t inlen, float gen_dist[], int distlen = K)
{
  auto in = new pszmem_cxx<T>(inlen, 1, 1, "hist-in");
  auto out = new pszmem_cxx<FQ>(OUTLEN, 1, 1, "hist-out");

  in->control({Malloc, MallocHost});
  out->control({Malloc, MallocHost});

  // setup using randgen
  gen_symetric_dist(in->hptr(), inlen, gen_dist, distlen, OUTLEN / 2);

  // for (auto i = 0; i < inlen; i++) cout << hin[i] << "\t";
  // cout << endl;

  in->control({H2D});

  constexpr auto NTREAD = 32 * NWARP;

  histsp_multiwarp<T, NWARP, CHUNK, FQ>
      <<<(inlen - 1) / CHUNK + 1, NTREAD, sizeof(FQ) * OUTLEN>>>(
          in->dptr(), inlen, out->dptr(), OUTLEN, OUTLEN / 2);

  cudaDeviceSynchronize();

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  delete in;
  delete out;

  return 0;
}

int main()
{
  constexpr auto r = 64;
  constexpr auto NSYM = r * 2;

  auto inlen = 500 * 500 * 100;
  f_histsp_kernel<NSYM, 16384, 1>(inlen, dist3);
  f_histsp_kernel<NSYM, 16384, 2>(inlen, dist3);
  f_histsp_kernel<NSYM, 16384, 4>(inlen, dist3);
  f_histsp_kernel<NSYM, 16384, 8>(inlen, dist3);
  f_histsp_kernel<NSYM, 16384, 16>(inlen, dist3);
  f_histsp_kernel<NSYM, 16384, 32>(inlen, dist3);

  f_histsp_kernel<NSYM, 32768, 1>(inlen, dist3);
  f_histsp_kernel<NSYM, 32768, 2>(inlen, dist3);
  f_histsp_kernel<NSYM, 32768, 4>(inlen, dist3);
  f_histsp_kernel<NSYM, 32768, 8>(inlen, dist3);
  f_histsp_kernel<NSYM, 32768, 16>(inlen, dist3);
  f_histsp_kernel<NSYM, 32768, 32>(inlen, dist3);

  f_histsp_kernel<NSYM, 65536, 1>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536, 2>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536, 4>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536, 8>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536, 16>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536, 32>(inlen, dist3);

  f_histsp_kernel<NSYM, 65536 * 2, 1>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536 * 2, 2>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536 * 2, 4>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536 * 2, 8>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536 * 2, 16>(inlen, dist3);
  f_histsp_kernel<NSYM, 65536 * 2, 32>(inlen, dist3);

  return 0;
}