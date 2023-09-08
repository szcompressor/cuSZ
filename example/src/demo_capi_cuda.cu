/**
 * @file demo_capi_nvcc.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-05-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "context.h"
#include "cusz.h"
#include "utils/io.hh"
#include "utils/print_arr.hh"
#include "utils/viewer.hh"

template <typename T>
void f(std::string fname)
{
  /* For demo, we use 3600x1800 CESM data. */
  auto len = 3600 * 1800;

  cusz_header header;
  uint8_t* exposed_compressed;
  uint8_t* compressed;
  size_t compressed_len;

  T *d_uncomp, *h_uncomp;
  T *d_decomp, *h_decomp;

  auto oribytes = sizeof(T) * len;
  cudaMalloc(&d_uncomp, oribytes), cudaMallocHost(&h_uncomp, oribytes);
  cudaMalloc(&d_decomp, oribytes), cudaMallocHost(&h_decomp, oribytes);

  /* User handles loading from filesystem & transferring to device. */
  io::read_binary_to_array(fname, h_uncomp, len);
  cudaMemcpy(d_uncomp, h_uncomp, oribytes, cudaMemcpyHostToDevice);

  /* a casual peek */
  printf("peeking uncompressed data, 20 elements\n");
  psz::peek_data(h_uncomp + len / 2, 20);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Using default:
  pszframe* work = pszdefault_framework();
  // Alternatively,
  //   pszframe w = pszframe{
  //       .predictor = pszpredictor{.type = Lorenzo},
  //       .quantizer = pszquantizer{.radius = 512},
  //       .hfcoder = pszhfrc{.style = Coarse},
  //       .max_outlier_percent = 20};
  //   auto work = &w;

  // Brace initializing a struct pointer is not supported by all host compilers
  // when nvcc forwards.
  //   pszframe* work = new pszframe{
  //       .predictor = pszpredictor{.type = Lorenzo},
  //       .quantizer = pszquantizer{.radius = 512},
  //       .hfcoder = pszhfrc{.style = Coarse},
  //       .max_outlier_percent = 20};

  cusz_compressor* comp = cusz_create(work, F4);
  pszctx* ctx = new pszctx{.mode = Rel, .eb = 2.4e-4};
  pszlen uncomp_len = pszlen{3600, 1800, 1, 1};  // x, y, z, w
  pszlen decomp_len = uncomp_len;

  cusz::TimeRecord compress_timerecord;
  cusz::TimeRecord decompress_timerecord;

  {
    psz_compress_init(comp, uncomp_len, ctx);
    psz_compress(
        comp, d_uncomp, uncomp_len, &exposed_compressed, &compressed_len,
        &header, (void*)&compress_timerecord, stream);

    /* User can interpret the collected time information in other ways. */
    cusz::TimeRecordViewer::view_compression(
        &compress_timerecord, oribytes, compressed_len);

    /* verify header */
    printf("header.%-*s : %p\n", 12, "(addr)", (void*)&header);
    printf(
        "header.%-*s : %u, %u, %u\n", 12, "{x,y,z}", header.x, header.y,
        header.z);
    printf(
        "header.%-*s : %lu\n", 12, "filesize", psz_utils::filesize(&header));
  }

  /* If needed, User should perform a memcopy to transfer `exposed_compressed`
   * before `compressor` is destroyed. */
  cudaMalloc(&compressed, compressed_len);
  cudaMemcpy(
      compressed, exposed_compressed, compressed_len,
      cudaMemcpyDeviceToDevice);

  {
    psz_decompress_init(comp, &header);
    psz_decompress(
        comp, exposed_compressed, compressed_len, d_decomp, decomp_len,
        (void*)&decompress_timerecord, stream);

    cusz::TimeRecordViewer::view_decompression(
        &decompress_timerecord, oribytes);
  }

  /* a casual peek */
  printf("peeking decompressed data, 20 elements\n");
  cudaMemcpy(h_decomp, d_decomp, oribytes, cudaMemcpyDeviceToHost);
  psz::peek_data(h_decomp + len / 2, 20);

  /* demo: offline checking (de)compression quality. */
  psz::eval_dataquality_gpu(d_decomp, d_uncomp, len, compressed_len);

  cusz_release(comp);

  cudaFree(compressed);

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
