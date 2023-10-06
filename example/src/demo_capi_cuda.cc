/**
 * @file demo_cxx_link.cc
 * @author Jiannan Tian
 * @brief This demo is synchronous with v0.7.
 * @version 0.7
 * @date 2022-05-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>

#include "context.h"
#include "cusz.h"
#include "cusz/type.h"
#include "utils/io.hh"
#include "utils/viewer.hh"

template <typename T>
void f(std::string fname)
{
  /* For demo, we use 3600x1800 CESM data. */
  auto len = 3600 * 1800;

  uint8_t* internal_compressed;
  size_t compressed_len;

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

  // Using default:
  // pszframe* work = pszdefault_framework();
  // Alternatively,
  pszframe* work = new pszframe{
      .predictor = pszpredictor{.type = Lorenzo},
      .quantizer = pszquantizer{.radius = 512},
      .hfcoder = pszhfrc{.style = Coarse},
      .max_outlier_percent = 20};

  pszcompressor* comp = cusz_create(work, F4);

  pszctx* ctx = new pszctx{.mode = Rel, .eb = 2.4e-4};
  pszlen uncomp_len = pszlen{3600, 1800, 1, 1};  // x, y, z, w
  pszlen decomp_len = uncomp_len;

  // TODO still C++ types
  cusz::TimeRecord compress_timerecord;
  cusz::TimeRecord decompress_timerecord;

  /*
  This points to the on-host internal buffer of the header. A duplicate header
  exists as the first 128 byte of the compressed file on device (as exposed by
  `internal_compressed` below).
  */
  pszheader internal_header;

  {
    psz_compress_init(comp, uncomp_len, ctx);
    psz_compress(
        comp, d_uncomp, uncomp_len, &internal_compressed, &compressed_len,
        &internal_header, (void*)&compress_timerecord, stream);

    /* User can interpret the collected time information in other ways. */
    cusz::TimeRecordViewer::view_compression(
        &compress_timerecord, oribytes, compressed_len);

    /* verify header */
    printf("header.%-*s : %p\n", 12, "(addr)", (void*)&internal_header);
    printf(
        "header.%-*s : %u, %u, %u\n", 12, "{x,y,z}", internal_header.x,
        internal_header.y, internal_header.z);
    printf(
        "header.%-*s : %lu\n", 12, "filesize",
        psz_utils::filesize(&internal_header));
  }

  /*
  Scenario 1:
  Memcpy from `internal_compressed`, on-device internal buffer (which saves
  the compressed archive), to `external_compressed`, another on-device buffer
  that persists independent from `compressor`. And the transfer happens before
  `compressor` along with `internal_compressed` is destroyed.
  */

  /* This malloc can happen anywhere. */
  uint8_t* external_compressed;

  cudaMalloc(&external_compressed, compressed_len);
  cudaMemcpy(
      external_compressed, internal_compressed, compressed_len,
      cudaMemcpyDeviceToDevice);
  /*
  Scenario 1 (cont'ed):
  In this case, a header (the first 128 byte of the buffer) needs to be saved
  on host for . There are two ways to complete this.

  (1) Make a copy of the **on-host** `internal_header`, the exposed header in
  the compressor buffer, and handle it otherwise. During the decompression,
  this header copy can be used directly.

  (2) Extract the first 128 bytes and type-cast them to `pszheader` (or the
  pointer). If the compressed archive exists on device, an explicit
  device-to-host memcpy is needed.
  */

  /*
  Scenario 2:
  Memcpy from `internal_compressed`, on-device internal buffer (which saves
  the compressed archive), to `host_compressed`, an on-host buffer for
  persisting in host memory or saving to filesystem.

  The code snippet for Scenario 2 is commended for no conflict to compile this
  demo.
  */

  /*
  uint8_t* host_compressed;
  cudaMallocHost(&host_compressed, compressed_len);
  cudaMemcpy(
      host_compressed, internal_compressed, compressed_len,
      cudaMemcpyDeviceToHost);
  pszheader external_header = *((pszheader*)host_compressed);

  // Put at the end of the API call to clean up:
  cudaFreeHost(host_compressed);
  */

  /*
  The decompression presumes that the user is in Scenario 1 and use method (2)
  to make a copy of header.
  */
  {
    pszheader header;
    cudaMemcpy(
        &header, external_compressed, sizeof(pszheader),
        cudaMemcpyDeviceToHost);
    psz_decompress_init(comp, &header);
    psz_decompress(
        comp, external_compressed, compressed_len, d_decomp, decomp_len,
        (void*)&decompress_timerecord, stream);

    cusz::TimeRecordViewer::view_decompression(
        &decompress_timerecord, oribytes);
  }

  /* demo: offline checking (de)compression quality. */
  psz::eval_dataquality_gpu(d_decomp, d_uncomp, len, compressed_len);

  cusz_release(comp);

  cudaFree(external_compressed);
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