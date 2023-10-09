/**
 * @file demo_cxx_link.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
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

  pszheader header;
  uint8_t* ptr_compressed;
  uint8_t* compressed_buf;
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

  pszcompressor* comp = psz_create(work, F4);

  pszctx* ctx = new pszctx{.mode = Rel, .eb = 2.4e-4};
  pszlen uncomp_len = pszlen{3600, 1800, 1, 1};  // x, y, z, w
  pszlen decomp_len = uncomp_len;

  // TODO still C++ types
  psz::TimeRecord compress_timerecord;
  psz::TimeRecord decompress_timerecord;

  {
    psz_compress_init(comp, uncomp_len, ctx);
    psz_compress(
        comp, d_uncomp, uncomp_len, &ptr_compressed, &compressed_len, &header,
        (void*)&compress_timerecord, stream);

    /* User can interpret the collected time information in other ways. */
    psz::TimeRecordViewer::view_compression(
        &compress_timerecord, oribytes, compressed_len);

    /* verify header */
    printf("header.%-*s : %p\n", 12, "(addr)", (void*)&header);
    printf(
        "header.%-*s : %u, %u, %u\n", 12, "{x,y,z}", header.x, header.y,
        header.z);
    printf(
        "header.%-*s : %lu\n", 12, "filesize", psz_utils::filesize(&header));
  }

  /* If needed, User should perform a memcopy to transfer `ptr_compressed`
   * before `compressor` is destroyed. */
  cudaMalloc(&compressed_buf, compressed_len);
  cudaMemcpy(
      compressed_buf, ptr_compressed, compressed_len,
      cudaMemcpyDeviceToDevice);

  {
    psz_decompress_init(comp, &header);
    psz_decompress(
        comp, ptr_compressed, compressed_len, d_decomp, decomp_len,
        (void*)&decompress_timerecord, stream);

    psz::TimeRecordViewer::view_decompression(
        &decompress_timerecord, oribytes);
  }

  /* demo: offline checking (de)compression quality. */
  psz::eval_dataquality_gpu(d_decomp, d_uncomp, len, compressed_len);

  psz_release(comp);

  cudaFree(compressed_buf);
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