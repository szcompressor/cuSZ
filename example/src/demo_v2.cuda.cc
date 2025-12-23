/**
 * @file demo_capi.cuda.cc
 * @author Jiannan Tian
 * @brief Also see demo_capi_minimal.cc for a more concise view.
 * @version 0.10
 * @date 2022-05-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>

#include "cusz.h"
#include "cusz/type.h"
#include "cusz_rev1.h"
#include "utils/io.hh"
#include "utils/viewer.hh"

namespace utils = _portable::utils;

std::string fname;
size_t f4data_len = 3600 * 1800;
size_t f8data_len = 384 * 384 * 256;
auto mode = Rel;   // set compression mode
auto eb = 1.2e-4;  // set error bound

float *f4d_uncomp, *f4h_uncomp;
float *f4d_decomp, *f4h_decomp;
double *f8d_uncomp, *f8h_uncomp;
double *f8d_decomp, *f8h_decomp;

void f4demo_compress_v2(
    psz_predictor predictor, psz_len3 const len3, psz_header* header, uint8_t** compressed,
    size_t* compressed_len, cudaStream_t stream)
{
  uint8_t* d_internal_compressed{nullptr};
  auto m = psz_create_resource_manager(
      F4, len3, {predictor, DEFAULT_HISTOGRAM, Huffman, NullCodec}, stream);

  psz_compress_float(
      m, {mode, eb, DEFAULT_RADIUS}, f4d_uncomp, header, &d_internal_compressed, compressed_len);

  // INSTRUCTION: need to copy out becore releasing resource.
  cudaMallocManaged(compressed, *compressed_len);
  cudaMemcpy(*compressed, d_internal_compressed, *compressed_len, cudaMemcpyDeviceToDevice);

  psz_release_resource(m);
}

void f8demo_compress_v2(
    psz_predictor predictor, psz_len3 const len3, psz_header* header, uint8_t** compressed,
    size_t* compressed_len, cudaStream_t stream)
{
  uint8_t* d_internal_compressed{nullptr};
  auto m = psz_create_resource_manager(
      F8, len3, {predictor, DEFAULT_HISTOGRAM, Huffman, NULL_CODEC}, stream);

  psz_compress_double(
      m, {mode, eb, DEFAULT_RADIUS}, f8d_uncomp, header, &d_internal_compressed, compressed_len);

  // INSTRUCTION: need to copy out becore releasing resource.
  cudaMallocManaged(compressed, *compressed_len);
  cudaMemcpy(*compressed, d_internal_compressed, *compressed_len, cudaMemcpyDeviceToDevice);

  psz_release_resource(m);
}

void f4demo_decompress_v2(psz_header* header, uint8_t* compressed, cudaStream_t stream)
{
  auto m = psz_create_resource_manager_from_header(header, stream);
  psz_decompress_float(m, compressed, pszheader_compressed_bytes(header), f4d_decomp);
  psz_release_resource(m);
}

void f8demo_decompress_v2(psz_header* header, uint8_t* compressed, cudaStream_t stream)
{
  auto m = psz_create_resource_manager_from_header(header, stream);
  psz_decompress_double(m, compressed, pszheader_compressed_bytes(header), f8d_decomp);
  psz_release_resource(m);
}

void f4demo(std::string fname, psz_len3 len3, psz_predictor predictor)
{
  psz_header header;
  uint8_t* compressed;
  size_t compressed_len{0}, oribytes = sizeof(float) * f4data_len;

  cudaMalloc(&f4d_uncomp, oribytes), cudaMallocHost(&f4h_uncomp, oribytes);
  cudaMalloc(&f4d_decomp, oribytes), cudaMallocHost(&f4h_decomp, oribytes);
  utils::fromfile(fname, f4h_uncomp, f4data_len);
  cudaMemcpy(f4d_uncomp, f4h_uncomp, oribytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  f4demo_compress_v2(predictor, len3, &header, &compressed, &compressed_len, stream);
  f4demo_decompress_v2(&header, compressed, stream);

  psz::analysis::GPU_evaluate_quality_and_print(
      f4d_decomp, f4d_uncomp, f4data_len, pszheader_compressed_bytes(&header));

  // clean up
  cudaFree(compressed);
  cudaFree(f4d_uncomp), cudaFreeHost(f4h_uncomp);
  cudaFree(f4d_decomp), cudaFreeHost(f4h_decomp);
  cudaStreamDestroy(stream);
}

void f8demo(std::string fname, psz_len3 len3, psz_predictor predictor)
{
  psz_header header;
  uint8_t* compressed;
  size_t compressed_len{0}, oribytes = sizeof(double) * f8data_len;

  cudaMalloc(&f8d_uncomp, oribytes), cudaMallocHost(&f8h_uncomp, oribytes);
  cudaMalloc(&f8d_decomp, oribytes), cudaMallocHost(&f8h_decomp, oribytes);
  utils::fromfile(fname, f8h_uncomp, f8data_len);
  cudaMemcpy(f8d_uncomp, f8h_uncomp, oribytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  f8demo_compress_v2(predictor, len3, &header, &compressed, &compressed_len, stream);
  f8demo_decompress_v2(&header, compressed, stream);

  psz::analysis::GPU_evaluate_quality_and_print(
      f8d_decomp, f8d_uncomp, f8data_len, pszheader_compressed_bytes(&header));

  // clean up
  cudaFree(compressed);
  cudaFree(f8d_uncomp), cudaFreeHost(f8h_uncomp);
  cudaFree(f8d_decomp), cudaFreeHost(f8h_decomp);
  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
#define PRINT_HELP()                                           \
  printf("          0     1  2\n");                            \
  printf("[demo 1]  PROG  1  /path/to/cesm-3600x1800\n");      \
  printf("[demo 2]  PROG  2  /path/to/miranda-384x384x256\n"); \
  exit(-1);

  if (argc < 3) { PRINT_HELP(); }

  auto demo_i = atoi(argv[1]);
  auto fname = std::string(argv[2]);

  if (demo_i == 1)
    f4demo(fname, {3600, 1800, 1}, Lorenzo);
  else if (demo_i == 2)
    f8demo(fname, {384, 384, 256}, Lorenzo);
  else {
    PRINT_HELP();
  }

  return 0;
}