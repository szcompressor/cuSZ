#include <cuda_runtime.h>

#include "fzg_class.hh"
#include "fzg_kernel.hh"
#include "utils/io.hh"

namespace utils = _portable::utils;

int low_level_demo(char **argv)
{
  auto data_len = atoi(argv[2]);
  auto config = fzgpu::configure_fzgpu(data_len);
  auto buf = new fzgpu::Buf(data_len, true);

  printf("ori_len\t\t:\t%lu\n", buf->configuration().at("len"));
  printf("pad_len\t\t:\t%lu\n", buf->configuration().at("pad_len"));
  printf("grid_x\t\t:\t%lu\n", buf->configuration().at("grid_x"));
  printf("chunk_size\t:\t%lu\n", buf->configuration().at("chunk_size"));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  utils::fromfile<uint16_t>(argv[1], &buf->h_in_data, data_len);

  cudaMemcpy(
      buf->d_in_data, buf->h_in_data, sizeof(uint16_t) * data_len,
      cudaMemcpyHostToDevice);

  ////////////////////////////////////////////////////////////////

  fzgpu::cuhip::GPU_FZ_encode(
      buf->d_in_data, data_len, buf->d_offset_counter, buf->d_bitflag_array,
      buf->d_start_pos, buf->d_comp_out, buf->d_comp_len, stream);

  cudaStreamSynchronize(stream);

  cudaMemcpy(
      &(buf->h_offset_sum), buf->d_offset_counter, sizeof(uint32_t),
      cudaMemcpyDeviceToHost);

  auto s1 = sizeof(uint32_t) * config.at("chunk_size");  // d_bitflag_array
  auto s2 = buf->h_offset_sum * sizeof(uint32_t);  // partially d_comp_out
  auto s3 = sizeof(uint32_t) * config["grid_x"];   // start_pos
  auto s_all = (s1 + s2 + s3);
  auto ori_bytes = data_len * sizeof(uint16_t) * 1.0;

  printf("\ncompression report (bytes):\n");
  printf("bitflags\t:\t%lu\t(%.3f%%)\n", s1, 100.0 * s1 / s_all);
  printf("d_comp_out\t:\t%lu\t(%.3f%%)\n", s2, 100.0 * s2 / s_all);
  printf("start_pos\t:\t%lu\t(%.3f%%)\n", s3, 100.0 * s3 / s_all);
  printf("-----------------------------------------\n");
  printf("ori bytes\t:\t%.0f\n", ori_bytes);
  printf("comp bytes\t:\t%lu\n", s_all);
  printf("-----------------------------------------\n");
  printf("comp ratio\t:\t%.2f\n", ori_bytes / s_all);

  fzgpu::cuhip::GPU_FZ_decode(
      buf->d_comp_out, buf->d_bitflag_array, buf->d_start_pos, buf->d_out_data,
      data_len, stream);

  cudaStreamSynchronize(stream);

  ////////////////////////////////////////////////////////////////

  cudaMemcpy(
      buf->h_out_data, buf->d_out_data, sizeof(uint16_t) * data_len,
      cudaMemcpyDeviceToHost);

  auto passed = true;

  printf("\n");
  for (size_t i = 0; i < data_len; i++) {
    if (i < 10)
      printf(
          "in=%d, out=%d\n", (int)buf->h_out_data[i], (int)buf->h_in_data[i]);

    if (buf->h_out_data[i] != buf->h_in_data[i]) {
      printf("verification stops at idx=%lu\n", i);
      passed = false;
      break;
    }
  }
  printf("\n");

  printf("passed? %s\n", passed ? "true" : "false");

  cudaStreamDestroy(stream);

  return 0;
}

int high_level_demo(char **argv)
{
  auto data_len = atoi(argv[2]);
  auto config = fzgpu::configure_fzgpu(data_len);

  printf("ori_len\t\t:\t%lu\n", config.at("len"));
  printf("pad_len\t\t:\t%lu\n", config.at("pad_len"));
  printf("grid_x\t\t:\t%lu\n", config.at("grid_x"));
  printf("chunk_size\t:\t%lu\n", config.at("chunk_size"));

  uint16_t *h_in_data, *d_in_data;
  uint16_t *h_out_data, *d_out_data;
  uint8_t *d_archive;
  size_t archive_len;

  auto len = config.at("len");
  auto pad_len = config.at("pad_len");

  cudaMallocHost(&h_in_data, sizeof(uint16_t) * len);
  cudaMalloc(&d_in_data, sizeof(uint16_t) * pad_len);

  cudaMallocHost(&h_out_data, sizeof(uint16_t) * len);
  cudaMalloc(&d_out_data, sizeof(uint16_t) * pad_len);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  utils::fromfile<uint16_t>(argv[1], &h_in_data, data_len);

  cudaMemcpy(
      d_in_data, h_in_data, sizeof(uint16_t) * data_len,
      cudaMemcpyHostToDevice);

  auto codec = new psz::FzgCodec(data_len);

  ////////////////////////////////////////////////////////////////

  codec->encode(d_in_data, data_len, &d_archive, &archive_len, stream);
  cudaStreamSynchronize(stream);

  uint8_t *d_copy_out;
  cudaMalloc(&d_copy_out, archive_len);
  cudaMemcpy(d_copy_out, d_archive, archive_len, cudaMemcpyDeviceToDevice);

  codec->decode(d_copy_out, archive_len, d_out_data, data_len, stream);

  cudaStreamSynchronize(stream);

  ////////////////////////////////////////////////////////////////

  cudaMemcpy(
      h_out_data, d_out_data, sizeof(uint16_t) * data_len,
      cudaMemcpyDeviceToHost);

  auto passed = true;

  printf("\n");
  for (size_t i = 0; i < data_len; i++) {
    if (i < 10)
      printf("in=%d, out=%d\n", (int)h_out_data[i], (int)h_in_data[i]);

    if (h_out_data[i] != h_in_data[i]) {
      printf("verification stops at idx=%lu\n", i);
      passed = false;
      break;
    }
  }
  printf("\n");

  printf("passed? %s\n", passed ? "true" : "false");

  cudaStreamDestroy(stream);

  return 0;
}

int main(int argc, char **argv)
{
  if (argc < 3) {
    printf("PROG  u2-binary-fname  len\n");
    exit(0);
  }

  printf("low-level demo:\n");
  low_level_demo(argv);
  printf(
      "\n"
      "==============================="
      "\n");

  printf("high-level demo:\n");
  high_level_demo(argv);

  return 0;
}