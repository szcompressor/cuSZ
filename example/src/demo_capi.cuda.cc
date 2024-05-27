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
#include "utils/config.hh"  //psz_utils::filesize
#include "utils/io.hh"      // io::read_binary_to_array
#include "utils/viewer.hh"  // view_de/compression, pszcxx_evaluate_quality_gpu

void utility_verify_header(pszheader* h)
{
  printf("header.%-*s : %p\n", 12, "(addr)", (void*)h);
  printf("header.%-*s : %u, %u, %u\n", 12, "{x,y,z}", h->x, h->y, h->z);
  printf("header.%-*s : %lu\n", 12, "filesize", psz_utils::filesize(h));
}

template <typename T>
void f(std::string fname)
{
  /* For demo, we use 3600x1800 CESM data. */
  auto len = 3600 * 1800;

  uint8_t* ptr_compressed;
  size_t comp_len;

  T *d_uncomp, *h_uncomp;
  T *d_decomp, *h_decomp;

  auto oribytes = sizeof(T) * len;
  cudaMalloc(&d_uncomp, oribytes), cudaMallocHost(&h_uncomp, oribytes);
  cudaMalloc(&d_decomp, oribytes), cudaMallocHost(&h_decomp, oribytes);

  /* User handles loading from filesystem & transferring to device. */
  io::read_binary_to_array(fname, h_uncomp, len);
  cudaMemcpy(d_uncomp, h_uncomp, oribytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto mode = Rel;   // set compression mode
  auto eb = 2.4e-4;  // set error bound

  /*
  Using default:
    pszcompressor* comp = psz_create_default(F4, eb, mode);

  Alternatively, customize the compressor as follows.
  */
  pszcompressor* comp = psz_create(
      /* dtype */ F4, /* predictor */ Lorenzo, /* quantizer radius */ 512,
      /* codec */ Huffman, eb, mode);

  auto uncomp_len = psz_len3{3600, 1800, 1};  // x, y, z
  auto decomp_len = uncomp_len;

  // TODO still C++ types
  psz::TimeRecord comp_timerecord;
  psz::TimeRecord decomp_timerecord;

  /*
  This points to the on-host internal buffer of the header. A duplicate header
  exists as the first 128 byte of the compressed file on device (as exposed by
  `ptr_compressed` below).

  psz_compress makes a copy of header from the internal.
  */
  pszheader header;

  {
    psz_compress_init(comp, uncomp_len);
    psz_compress(
        comp, d_uncomp, uncomp_len, &ptr_compressed, &comp_len, &header,
        (void*)&comp_timerecord, stream);

    /* User can interpret the collected time information in other ways. */
    psz::TimeRecordViewer::view_compression(
        &comp_timerecord, oribytes, comp_len);

    utility_verify_header(&header);
  }

  /*
  Scenario 1: Persisting in another on-device buffer.

  Memcpy from `ptr_compressed`, on-device internal buffer (of compressor), to
  `ext_compressed`, another on-device buffer, via device-to-device memcpy.
  After this, the compressor can be freely released.
  */
  uint8_t* ext_compressed;
  cudaMalloc(&ext_compressed, comp_len);

  /*
  Also note that on-device ptr_compressed does not contain on-host header.
  */
  cudaMemcpy(
      ext_compressed, ptr_compressed, comp_len, cudaMemcpyDeviceToDevice);
  utility_verify_header(&header);

  /*
  Scenario 2: Saving the archive to filesystem.

  1. Memcpy from on-device `ptr_compressed` with `comp_len` to
  `host_compressed`, an on-host buffer for persisting in host memory or saving
  to filesystem.
  2. Put header to the begining of the archive.

    uint8_t* host_compressed;
    cudaMallocHost(&host_compressed, comp_len);
    cudaMemcpy(
        host_compressed, ptr_compressed, comp_len,
  cudaMemcpyDeviceToHost); memcpy(host_compressed, &header, sizeof(header));

  3. Save to filesystem.
  4. Put at the end of the API call to clean up:
    cudaFreeHost(host_compressed);
  */

  /*
  The decompression uses the exported header from psz_compress.
  */
  {
    utility_verify_header(&header);

    psz_decompress_init(comp, &header);
    psz_decompress(
        comp, ext_compressed, comp_len, d_decomp, decomp_len,
        (void*)&decomp_timerecord, stream);

    psz::TimeRecordViewer::view_decompression(&decomp_timerecord, oribytes);
  }

  /* demo: offline checking (de)compression quality. */
  pszcxx_evaluate_quality_gpu(d_decomp, d_uncomp, len, comp_len);

  /* Release pSZ compressor */
  psz_release(comp);

  /*
  Clean up CUDA stuff.
  */
  cudaFree(ext_compressed);
  cudaFree(d_uncomp), cudaFreeHost(h_uncomp);
  cudaFree(d_decomp), cudaFreeHost(h_decomp);
  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    printf("PROG /path/to/cesm-3600x1800\n");
    exit(0);
  }

  f<float>(std::string(argv[1]));
  return 0;
}