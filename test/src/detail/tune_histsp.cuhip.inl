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

#include "kernel/detail/histsp.cuhip.inl"
#include "kernel/hist.hh"
#include "utils/busyheader.hh"

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

  T *d_in, *h_in;
  FQ *d_gpusp, *h_gpusp, *h_serial;
  cudaMalloc(&d_in, inlen * sizeof(T));
  cudaMallocHost(&h_in, inlen * sizeof(T));
  cudaMalloc(&d_gpusp, NSYM * sizeof(FQ));
  cudaMallocHost(&h_gpusp, NSYM * sizeof(FQ));
  cudaMallocHost(&h_serial, NSYM * sizeof(FQ));

  for (auto i = 0; i < inlen; i++) {
    h_in[i] = 512;
    if (i > 1 and i % 5 == 0) h_in[i] = 511, h_in[i - 1] = 513;
    if (i > 1 and i % 20 == 0) h_in[i] = 510, h_in[i - 1] = 514;
    if (i > 1 and i % 40 == 0) h_in[i] = 509, h_in[i - 1] = 515;
    if (i > 1 and i % 50 == 0) h_in[i] = 507, h_in[i - 1] = 516;
  }

  cudaMemcpy(d_in, h_in, inlen * sizeof(T), cudaMemcpyHostToDevice);

  float t_histsp_ser;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  psz::module::SEQ_histogram_Cauchy_v2<T>(h_in, inlen, h_serial, NSYM, &t_histsp_ser);
  psz::module::GPU_histogram_Cauchy<T>::kernel(d_in, inlen, d_gpusp, NSYM, stream);

  cudaMemcpy(h_gpusp, d_gpusp, NSYM * sizeof(FQ), cudaMemcpyDeviceToHost);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  auto all_eq = true;
  printf("\n\n");
  for (auto i = 0; i < NSYM; i++) {
    if (h_serial[i] != 0) {
      printf("i: %d\tgpusp: %u\tserial: %u\n", i, h_gpusp[i], h_serial[i]);
      all_eq = false;
    }
  }

  cudaStreamDestroy(stream);
  cudaFree(d_in);
  cudaFreeHost(h_in);
  cudaFree(d_gpusp);
  cudaFreeHost(h_gpusp);
  cudaFreeHost(h_serial);

  return all_eq;
}

void helper_generate_array(T* in, size_t inlen, float dist[], int distlen = 5, int offset = 512)
{
  auto R = (distlen - 1) / 2;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, inlen);

  for (auto _ = 0; _ < inlen; _++) { in[_] = offset; }
  for (auto i = 0; i < distlen; i++) {
    if (i - R == 0)
      continue;
    else {
      auto N = (int)(inlen * dist[i]);
      auto sym = (i - R) + offset;
      for (auto _ = 0; _ < N; _++) { in[distrib(gen)] = sym; }
    }
  }
}

template <int NSYM = 1024>
bool test2_fulllen_input(size_t inlen, float gen_dist[], int distlen = K)
{
  T *d_in, *h_in;
  FQ *d_gpu, *h_gpu, *d_gpusp, *h_gpusp, *h_serial;

  cudaMalloc(&d_in, inlen * sizeof(T));
  cudaMallocHost(&h_in, inlen * sizeof(T));
  cudaMalloc(&d_gpu, NSYM * sizeof(FQ));
  cudaMallocHost(&h_gpu, NSYM * sizeof(FQ));
  cudaMalloc(&d_gpusp, NSYM * sizeof(FQ));
  cudaMallocHost(&h_gpusp, NSYM * sizeof(FQ));
  cudaMallocHost(&h_serial, NSYM * sizeof(FQ));

  helper_generate_array(h_in, inlen, gen_dist, distlen, NSYM / 2);
  cudaMemcpy(d_in, h_in, inlen * sizeof(T), cudaMemcpyHostToDevice);

  float t_histsp_ser;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int grid_dim, block_dim, shmem_use, r_per_block;
  psz::module::GPU_histogram_generic<T>::init(
      inlen, NSYM, grid_dim, block_dim, shmem_use, r_per_block);
  psz::module::GPU_histogram_generic<T>::kernel(
      d_in, inlen, d_gpu, NSYM, grid_dim, block_dim, shmem_use, r_per_block, stream);
  psz::module::GPU_histogram_Cauchy<T>::kernel(d_in, inlen, d_gpusp, NSYM, stream);
  psz::module::SEQ_histogram_Cauchy_v2<T>(h_in, inlen, h_serial, NSYM, &t_histsp_ser);

  cudaMemcpy(h_gpu, d_gpu, NSYM * sizeof(FQ), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gpusp, d_gpusp, NSYM * sizeof(FQ), cudaMemcpyDeviceToHost);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  auto all_eq = true;
  for (auto i = 0; i < NSYM; i++) {
    if (h_gpu[i] == h_gpusp[i] and h_gpusp[i] == h_serial[i]) { continue; }
    else {
      printf(
          "first not equal\tidx: %d\tgpu: %u\tgpusp: %u\tserial: %u\n", i, h_gpu[i], h_gpusp[i],
          h_serial[i]);
      all_eq = false;
      break;
    }
  }
  if (all_eq) printf("full-length test: all equal\n");

  cudaStreamDestroy(stream);
  cudaFree(d_in);
  cudaFreeHost(h_in);
  cudaFree(d_gpu);
  cudaFreeHost(h_gpu);
  cudaFree(d_gpusp);
  cudaFreeHost(h_gpusp);
  cudaFreeHost(h_serial);

  return all_eq;
}

template <int NSYM = 1024, int CHUNK = 32768, int NWARP = 8>
bool perf(
    T* d_in, size_t inlen, FQ* d_gpusp, FQ* h_gpusp, FQ* h_gpu, FQ* h_serial, cudaStream_t stream)
{
  constexpr auto NTREAD = 32 * NWARP;

  psz::KERNEL_CUHIP_histogram_sparse_multiwarp<T, NWARP, CHUNK, FQ>
      <<<(inlen - 1) / CHUNK + 1, NTREAD, NSYM * sizeof(FQ), stream>>>(
          d_in, inlen, d_gpusp, NSYM, NSYM / 2);

  cudaStreamSynchronize(stream);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("NSYM: %d\tCHUNK: %d\tNWARP: %d\n", NSYM, CHUNK, NWARP);
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  auto all_eq = true;
  for (auto i = 0; i < NSYM; i++) {
    if (h_gpu[i] == h_gpusp[i] and h_gpusp[i] == h_serial[i]) { continue; }
    else {
      printf(
          "first not equal\tidx: %d\tgpu: %u\tgpusp: %u\tserial: %u\n", i, h_gpu[i], h_gpusp[i],
          h_serial[i]);
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
  T *d_in, *h_in;
  FQ *d_gpu, *h_gpu, *d_gpusp, *h_gpusp, *h_serial;

  cudaMalloc(&d_in, inlen * sizeof(T));
  cudaMallocHost(&h_in, inlen * sizeof(T));
  cudaMalloc(&d_gpu, NSYM * sizeof(FQ));
  cudaMallocHost(&h_gpu, NSYM * sizeof(FQ));
  cudaMalloc(&d_gpusp, NSYM * sizeof(FQ));
  cudaMallocHost(&h_gpusp, NSYM * sizeof(FQ));
  cudaMallocHost(&h_serial, NSYM * sizeof(FQ));

  helper_generate_array(h_in, inlen, gen_dist, distlen, NSYM / 2);
  cudaMemcpy(d_in, h_in, inlen * sizeof(T), cudaMemcpyHostToDevice);

  float t_histsp_ser;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int grid_dim, block_dim, shmem_use, r_per_block;
  psz::module::GPU_histogram_generic<T>::init(
      inlen, NSYM, grid_dim, block_dim, shmem_use, r_per_block);
  psz::module::GPU_histogram_generic<T>(
      d_in, inlen, d_gpu, NSYM, grid_dim, block_dim, shmem_use, r_per_block, stream);
  psz::module::SEQ_histogram_Cauchy_v2<T>(h_in, inlen, h_serial, NSYM, &t_histsp_ser);
  cudaStreamSynchronize(stream);

#define PERF(NSYM, CHUNK, NWARP) \
  eq = eq and perf<NSYM, CHUNK, NWARP>(d_in, inlen, d_gpusp, h_gpusp, h_gpu, h_serial, stream);

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
  cudaFree(d_in);
  cudaFreeHost(h_in);
  cudaFree(d_gpu);
  cudaFreeHost(h_gpu);
  cudaFree(d_gpusp);
  cudaFreeHost(h_gpusp);
  cudaFreeHost(h_serial);

#undef PERF
  return eq;
}
