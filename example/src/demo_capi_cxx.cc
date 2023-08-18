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

  T *d_ori, *h_ori;
  T *d_reconst, *h_reconst;

  auto oribytes = sizeof(T) * len;
  cudaMalloc(&d_ori, oribytes), cudaMallocHost(&h_ori, oribytes);
  cudaMalloc(&d_reconst, oribytes), cudaMallocHost(&h_reconst, oribytes);

  /* User handles loading from filesystem & transferring to device. */
  io::read_binary_to_array(fname, h_ori, len);
  cudaMemcpy(d_ori, h_ori, oribytes, cudaMemcpyHostToDevice);

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

  pszrc* config = new pszrc{.eb = 2.4e-4, .mode = Rel};
  pszlen uncomp_len = pszlen{3600, 1800, 1, 1};  // x, y, z, w
  pszlen decomp_len = uncomp_len;

  // TODO still C++ types
  cusz::TimeRecord compress_timerecord;
  cusz::TimeRecord decompress_timerecord;

  {
    cusz_compress(
        comp, config, d_ori, uncomp_len, &ptr_compressed, &compressed_len,
        &header, (void*)&compress_timerecord, stream);

    /* User can interpret the collected time information in other ways. */
    cusz::TimeRecordViewer::view_compression(
        &compress_timerecord, len * sizeof(T), compressed_len);

    /* verify header */
    printf("header.%-*s : %x\n", 12, "(addr)", &header);
    printf(
        "header.%-*s : %lu, %lu, %lu\n", 12, "{x,y,z}", header.x, header.y,
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
    cusz_decompress(
        comp, &header, ptr_compressed, compressed_len, d_reconst, decomp_len,
        (void*)&decompress_timerecord, stream);

    cusz::TimeRecordViewer::view_decompression(&decompress_timerecord, len *
    sizeof(T));
  }

  /* demo: offline checking (de)compression quality. */
  psz::eval_dataquality_gpu(d_reconst, d_ori, len, compressed_len);

  cusz_release(comp);

  cudaFree(compressed_buf);
  cudaFree(d_ori), cudaFreeHost(h_ori);
  cudaFree(d_reconst), cudaFreeHost(h_reconst);

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