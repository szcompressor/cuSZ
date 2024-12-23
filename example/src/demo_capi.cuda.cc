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

// utilities for demo
#include "cusz/review.h"
#include "cusz/type.h"
#include "utils/io.hh"  // io::read_binary_to_array

namespace utils = _portable::utils;

using T = float;

std::string fname;
size_t len = 3600 * 1800;
size_t oribytes = sizeof(T) * len;
auto mode = Rel;   // set compression mode
auto eb = 1.2e-4;  // set error bound

T *d_decomp, *h_decomp;
T *d_uncomp, *h_uncomp;
void* comp_timerecord;
void* decomp_timerecord;

void demo_compress(
    psz_predtype predictor, psz_len3 const interpreted_len3,
    uint8_t** compressed, psz_header* header, cudaStream_t stream)
{
  uint8_t* p_compressed;
  size_t comp_len;

  auto* compressor = psz_create(
      /* data */ F4, interpreted_len3, predictor,
      /* quantizer radius */ 512,
      /* codec */ Huffman);

  psz_compress(
      compressor, d_uncomp, interpreted_len3, eb, mode, &p_compressed,
      &comp_len, header, comp_timerecord, stream);

  cudaMalloc(compressed, comp_len);
  cudaMemcpy(*compressed, p_compressed, comp_len, cudaMemcpyDeviceToDevice);

  psz_release(compressor);
}

void demo_decompress(
    uint8_t* compressed, psz_header* header, cudaStream_t stream)
{
  auto comp_len = pszheader_filesize(header);
  psz_len3 decomp_len = psz_len3{header->x, header->y, header->z};

  auto compressor = psz_create_from_header(header);
  psz_decompress(
      compressor, compressed, comp_len, d_decomp, decomp_len,
      decomp_timerecord, stream);

  psz_release(compressor);
}

void demo(std::string fname, psz_len3 interpreted_len3, psz_predtype predictor)
{
  psz_header header;
  uint8_t* compressed;

  cudaMalloc(&d_uncomp, oribytes), cudaMallocHost(&h_uncomp, oribytes);
  cudaMalloc(&d_decomp, oribytes), cudaMallocHost(&h_decomp, oribytes);
  utils::fromfile(fname, &h_uncomp, len);
  cudaMemcpy(d_uncomp, h_uncomp, oribytes, cudaMemcpyHostToDevice);

  comp_timerecord = psz_make_timerecord();
  decomp_timerecord = psz_make_timerecord();

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  demo_compress(predictor, interpreted_len3, &compressed, &header, stream);

  {
    auto comp_len = pszheader_filesize(&header);
    psz_review_compression(comp_timerecord, &header);
  }

  demo_decompress(compressed, &header, stream);

  {
    auto comp_len = pszheader_filesize(&header);
    psz_review_decompression(decomp_timerecord, oribytes);
    psz_review_evaluated_quality(
        THRUST_DPL, F4, d_decomp, d_uncomp, len, comp_len, true);
  }

  // clean up
  cudaFree(compressed);
  cudaFree(d_uncomp);
  cudaFree(d_decomp);
  cudaFreeHost(h_decomp);
  cudaFreeHost(h_decomp);

  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    /* For demo, we use 3600x1800 CESM data. */
    printf("PROG /path/to/cesm-3600x1800\n");
    exit(0);
  }

  auto print_dahsed_line = []() {
    printf("\e[1m\e[31m-------------------------------\e[0m\n");
  };
  auto print_dahsed_line_long = []() {
    printf("\e[1m\e[31m---------------------------------------\e[0m\n");
  };
  auto print_1d_title = []() {
    printf("\e[1m\e[31minterpret data as 1D: x=6480000\e[0m\n");
  };
  auto print_2d_title = []() {
    printf("\e[1m\e[31minterpret data as 2D: (x,y)=(3600,1800)\e[0m\n");
  };
  auto print_3d_title = []() {
    printf("\e[1m\e[31minterpret data as 3D: (x,y,z)=(360,180,100)\e[0m\n");
  };

  auto fname = std::string(argv[1]);

  print_dahsed_line(), print_1d_title(), print_dahsed_line();
  demo(fname, {6480000, 1, 1}, Lorenzo);
  printf("\n\n"), print_dahsed_line_long(), printf("\n\n\n");

  print_dahsed_line(), print_2d_title(), print_dahsed_line();
  demo(fname, {3600, 1800, 1}, Lorenzo);
  printf("\n\n"), print_dahsed_line_long(), printf("\n\n\n");

  print_dahsed_line(), print_3d_title(), print_dahsed_line();
  demo(fname, {360, 180, 100}, Lorenzo);
  printf("\n\n"), print_dahsed_line_long(), printf("\n\n\n");

  return 0;
}