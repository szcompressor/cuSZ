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
#include "utils/io.hh"  // io::read_binary_to_array
#include "utils/viewer/viewer.cu_hip.hh"
#include "utils/viewer/viewer.h"

using T = float;

std::string fname;
size_t len = 3600 * 1800;
size_t oribytes = sizeof(T) * len;
size_t x = 3600, y = 1800, z = 1;
psz_len3 uncomp_len = {3600, 1800, 1};  // x, y, z
auto mode = Rel;                        // set compression mode
auto eb = 1.2e-4;                       // set error bound

T *d_decomp, *h_decomp;
T *d_uncomp, *h_uncomp;
void* comp_timerecord;
void* decomp_timerecord;

void utility_verify_header(psz_header* h)
{
  printf("header.%-*s : %p\n", 12, "(addr)", (void*)h);
  printf("header.%-*s : %u, %u, %u\n", 12, "{x,y,z}", h->x, h->y, h->z);
  printf("header.%-*s : %lu\n", 12, "filesize", pszheader_filesize(h));
}

void demo_compress(
    uint8_t** compressed, psz_header* header, cudaStream_t stream)
{
  uint8_t* p_compressed;
  size_t comp_len;

  auto* compressor = psz_create(
      /* data */ F4, uncomp_len, /* predictor */ Lorenzo,
      /* quantizer radius */ 512,
      /* codec */ Huffman);

  psz_compress(
      compressor, d_uncomp, uncomp_len, eb, mode, &p_compressed, &comp_len,
      header, comp_timerecord, stream);

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

int main(int argc, char** argv)
{
  if (argc < 2) {
    /* For demo, we use 3600x1800 CESM data. */
    printf("PROG /path/to/cesm-3600x1800\n");
    exit(0);
  }

  psz_header header;
  uint8_t* compressed;
  auto fname = std::string(argv[1]);

  cudaMalloc(&d_uncomp, oribytes), cudaMallocHost(&h_uncomp, oribytes);
  cudaMalloc(&d_decomp, oribytes), cudaMallocHost(&h_decomp, oribytes);
  io::read_binary_to_array(fname, h_uncomp, len);
  cudaMemcpy(d_uncomp, h_uncomp, oribytes, cudaMemcpyHostToDevice);

  comp_timerecord = psz_make_timerecord();
  decomp_timerecord = psz_make_timerecord();

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  demo_compress(&compressed, &header, stream);

  {
    utility_verify_header(&header);
    auto comp_len = pszheader_filesize(&header);
    psz_review_compression(comp_timerecord, &header);
  }

  demo_decompress(compressed, &header, stream);

  {
    auto comp_len = pszheader_filesize(&header);
    psz_review_decompression(decomp_timerecord, oribytes);
    pszcxx_evaluate_quality_gpu(d_decomp, d_uncomp, len, comp_len);
  }

  // clean up
  cudaFree(compressed);
  cudaFree(d_uncomp);
  cudaFree(d_decomp);
  cudaFreeHost(h_decomp);
  cudaFreeHost(h_decomp);

  cudaStreamDestroy(stream);

  return 0;
}