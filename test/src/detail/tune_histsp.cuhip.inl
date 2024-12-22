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

#include "busyheader.hh"
#include "kernel/detail/histsp.cuhip.inl"
#include "mem/memobj.hh"
#include "module/cxx_module.hh"

using namespace portable;

using T = uint32_t;
using FQ = uint32_t;

constexpr auto R = 2;
constexpr auto K = 2 * R + 1;

float dist1[] = {0.01, 0.09, 0.8, 0.09, 0.01};
float dist2[] = {0.01, 0.04, 0.9, 0.04, 0.01};
float dist3[] = {0.005, 0.015, 0.96, 0.015, 0.005};

bool test1_debug()
{
  auto inlen = 256;
  auto NSYM = 1024;

  auto in = new memobj<T>(inlen, "hist-in", {Malloc, MallocHost});
  auto o_gpusp = new memobj<FQ>(NSYM, "hist-o_gpusp", {Malloc, MallocHost});
  auto o_serial = new memobj<FQ>(NSYM, "hist-o_gpusp", {MallocHost});

  for (auto i = 0; i < inlen; i++) {
    in->hptr(i) = 512;
    if (i > 1 and i % 5 == 0) in->hptr(i) = 511, in->hptr(i - 1) = 513;
    if (i > 1 and i % 20 == 0) in->hptr(i) = 510, in->hptr(i - 1) = 514;
    if (i > 1 and i % 40 == 0) in->hptr(i) = 509, in->hptr(i - 1) = 515;
    if (i > 1 and i % 50 == 0) in->hptr(i) = 507, in->hptr(i - 1) = 516;
  }

  in->control({H2D});
  // float __t;

  float t_histsp_ser, t_histsp_cuda;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  psz::module::SEQ_histogram_Cauchy_v2<T>(
      in->hptr(), inlen, o_serial->hptr(), NSYM, &t_histsp_ser);

  psz::module::GPU_histogram_Cauchy<T>(
      in->dptr(), inlen, o_gpusp->dptr(), NSYM, &t_histsp_cuda, stream);

  o_gpusp->control({D2H});

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  auto all_eq = true;
  printf("\n\n");
  for (auto i = 0; i < NSYM; i++) {
    if (o_serial->hptr(i) != 0) {
      printf(
          "i: %d\t"
          "gpusp: %u\t"
          "serial: %u\n",
          i, o_gpusp->hptr(i), o_serial->hptr(i));
      all_eq = false;
    }
  }

  cudaStreamDestroy(stream);

  delete in;
  delete o_gpusp;
  delete o_serial;

  return all_eq;
}

void helper_generate_array(
    T* in, size_t inlen, float dist[], int distlen = 5, int offset = 512)
{
  // cout << "offset: " << offset << endl;

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
      // printf("sym: %d, num: %d\n", sym, N);
      for (auto _ = 0; _ < N; _++) {
        auto loc = distrib(gen);
        in[loc] = sym;
      }
    }
  }
}

template <int NSYM = 1024>
bool test2_fulllen_input(size_t inlen, float gen_dist[], int distlen = K)
{
  auto in = new memobj<T>(inlen, "hist-in", {Malloc, MallocHost});
  auto o_gpu = new memobj<FQ>(NSYM, "hist-o_gpu", {Malloc, MallocHost});
  auto o_gpusp = new memobj<FQ>(NSYM, "hist-o_gpusp", {Malloc, MallocHost});
  auto o_serial = new memobj<FQ>(NSYM, "hist-o_serial", {MallocHost});

  // setup using randgen
  helper_generate_array(in->hptr(), inlen, gen_dist, distlen, NSYM / 2);

  in->control({H2D});
  float t_hist_cuda, t_histsp_ser, t_histsp_cuda;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  psz::module::GPU_histogram_Cauchy<T>(
      in->dptr(), inlen, o_gpusp->dptr(), NSYM, &t_histsp_cuda, stream);
  psz::module::GPU_histogram_generic<T>(
      in->dptr(), inlen, o_gpu->dptr(), NSYM, &t_hist_cuda, stream);

  psz::module::SEQ_histogram_Cauchy_v2<T>(
      in->hptr(), inlen, o_serial->hptr(), NSYM, &t_histsp_ser);

  o_gpu->control({D2H});
  o_gpusp->control({D2H});

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  // check correctness
  auto all_eq = true;

  for (auto i = 0; i < NSYM; i++) {
    if (o_gpu->hptr(i) == o_gpusp->hptr(i) and
        o_gpusp->hptr(i) == o_serial->hptr(i)) {
      continue;
    }
    else {
      printf(
          "first not equal\t"
          "idx: %d\tgpu: %u\tgpusp: %u\tserial: %u\n",  //
          i, o_gpu->hptr(i), o_gpusp->hptr(i), o_serial->hptr(i));
      all_eq = false;
      break;
    }
  }
  if (all_eq) printf("full-length test: all equal\n");

  cudaStreamDestroy(stream);

  delete in;
  delete o_gpu;
  delete o_gpusp;
  delete o_serial;

  return all_eq;
}

template <int NSYM = 1024, int CHUNK = 32768, int NWARP = 8>
bool perf(
    memobj<T>* in, memobj<FQ>* o_gpusp,       // for histsp
    memobj<FQ>* o_gpu, memobj<FQ>* o_serial,  // reference
    cudaStream_t stream)
{
  constexpr auto NTREAD = 32 * NWARP;

  psz::KERNEL_CUHIP_histogram_sparse_multiwarp<T, NWARP, CHUNK, FQ>
      <<<(in->len() - 1) / CHUNK + 1, NTREAD, NSYM * sizeof(FQ), stream>>>(
          in->dptr(), in->len(), o_gpusp->dptr(), NSYM, NSYM / 2);

  cudaStreamSynchronize(stream);

  // check for error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("NSYM: %d\tCHUNK: %d\tNWARP: %d\n", NSYM, CHUNK, NWARP);
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  // check correctness
  auto all_eq = true;

  for (auto i = 0; i < NSYM; i++) {
    if (o_gpu->hptr(i) == o_gpusp->hptr(i) and
        o_gpusp->hptr(i) == o_serial->hptr(i)) {
      continue;
    }
    else {
      printf(
          "first not equal\t"
          "idx: %d\tgpu: %u\tgpusp: %u\tserial: %u\n",  //
          i, o_gpu->hptr(i), o_gpusp->hptr(i), o_serial->hptr(i));
      all_eq = false;
      break;
    }
  }
  if (all_eq) printf("perf test: all equal\n");

  return all_eq;
}

template <int NSYM = 1024>
bool test3_performance_tuning(size_t inlen, float gen_dist[], int distlen = K)
{
  auto in = new memobj<T>(inlen, "hist-in", {Malloc, MallocHost});
  auto o_gpu = new memobj<FQ>(NSYM, "hist-o_gpu", {Malloc, MallocHost});
  auto o_gpusp = new memobj<FQ>(NSYM, "hist-o_gpusp", {Malloc, MallocHost});
  auto o_serial = new memobj<FQ>(NSYM, "hist-o_serial", {MallocHost});

  // setup using randgen
  helper_generate_array(in->hptr(), inlen, gen_dist, distlen, NSYM / 2);
  in->control({H2D});

  float t_hist_gpu, t_histsp_ser;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // run CPU and GPU reference
  psz::module::GPU_histogram_generic<T>(
      in->dptr(), inlen, o_gpu->dptr(), NSYM, &t_hist_gpu, stream);

  psz::module::SEQ_histogram_Cauchy_v2<T>(
      in->hptr(), inlen, o_serial->hptr(), NSYM, &t_histsp_ser);
  cudaStreamSynchronize(stream);

// start testing & profiling
#define PERF(NSYM, CHUNK, NWARP) \
  eq = eq and perf<NSYM, CHUNK, NWARP>(in, o_gpusp, o_gpu, o_serial, stream);

  auto eq = true;
  PERF(NSYM, 16384, 1);
  PERF(NSYM, 16384, 2);
  PERF(NSYM, 16384, 4);
  PERF(NSYM, 16384, 8);
  PERF(NSYM, 16384, 16);
  PERF(NSYM, 16384, 32);

  PERF(NSYM, 32768, 1);
  PERF(NSYM, 32768, 2);
  PERF(NSYM, 32768, 4);
  PERF(NSYM, 32768, 8);
  PERF(NSYM, 32768, 16);
  PERF(NSYM, 32768, 32);

  PERF(NSYM, 65536, 1);
  PERF(NSYM, 65536, 2);
  PERF(NSYM, 65536, 4);
  PERF(NSYM, 65536, 8);
  PERF(NSYM, 65536, 16);
  PERF(NSYM, 65536, 32);

  PERF(NSYM, 65536 * 2, 1);
  PERF(NSYM, 65536 * 2, 2);
  PERF(NSYM, 65536 * 2, 4);
  PERF(NSYM, 65536 * 2, 8);
  PERF(NSYM, 65536 * 2, 16);
  PERF(NSYM, 65536 * 2, 32);

  cudaStreamDestroy(stream);
  delete in;
  delete o_gpu;
  delete o_gpusp;
  delete o_serial;

#undef PERF

  return eq;
}
