#include <cuda_runtime.h>

#include "fzg_hl.hh"
#include "fzg_impl.hh"
#include "mem/cxx_backends.h"
#include "utils/io.hh"

namespace utils = _portable::utils;

#define CHECK_INTEGRITY_1                                                                   \
  memcpy_allkinds<D2H>(buf->h_out_data, buf->d_out_data, data_len);                         \
  auto passed = true;                                                                       \
  printf("\n");                                                                             \
  for (size_t i = 0; i < data_len; i++) {                                                   \
    if (i < 10) printf("in=%d, out=%d\n", (int)buf->h_out_data[i], (int)buf->h_in_data[i]); \
    if (buf->h_out_data[i] != buf->h_in_data[i]) {                                          \
      printf("verification stops at idx=%lu\n", i);                                         \
      passed = false;                                                                       \
      break;                                                                                \
    }                                                                                       \
  }                                                                                         \
  printf("\n");                                                                             \
  printf("passed? %s\n", passed ? "true" : "false");

#define PRINT_CONFIG_1                                               \
  printf("ori_len\t\t:\t%lu\n", buf->configuration().at("len"));     \
  printf("pad_len\t\t:\t%lu\n", buf->configuration().at("pad_len")); \
  printf("grid_x\t\t:\t%lu\n", buf->configuration().at("grid_x"));   \
  printf("chunk_size\t:\t%lu\n", buf->configuration().at("chunk_size"));

#define PRINT_REPORT_1                                                             \
  memcpy_allkinds<D2H>(&(buf->h_offset_sum), buf->d_offset_counter, 1);            \
  auto s1 = sizeof(uint32_t) * config.at("chunk_size"); /* d_bitflag_array */      \
  auto s2 = buf->h_offset_sum * sizeof(uint32_t);       /* partially d_comp_out */ \
  auto s3 = sizeof(uint32_t) * config["grid_x"];        /* start_pos */            \
  auto s_all = (s1 + s2 + s3);                                                     \
  auto ori_bytes = data_len * sizeof(uint16_t) * 1.0;                              \
  printf("\ncompression report (bytes):\n");                                       \
  printf("bitflags\t:\t%lu\t(%.3f%%)\n", s1, 100.0 * s1 / s_all);                  \
  printf("d_comp_out\t:\t%lu\t(%.3f%%)\n", s2, 100.0 * s2 / s_all);                \
  printf("start_pos\t:\t%lu\t(%.3f%%)\n", s3, 100.0 * s3 / s_all);                 \
  printf("-----------------------------------------\n");                           \
  printf("ori bytes\t:\t%.0f\n", ori_bytes);                                       \
  printf("comp bytes\t:\t%lu\n", s_all);                                           \
  printf("-----------------------------------------\n");                           \
  printf("comp ratio\t:\t%.2f\n", ori_bytes / s_all);

#define PRINT_CONFIG_2                                 \
  printf("ori_len\t\t:\t%lu\n", config.at("len"));     \
  printf("pad_len\t\t:\t%lu\n", config.at("pad_len")); \
  printf("grid_x\t\t:\t%lu\n", config.at("grid_x"));   \
  printf("chunk_size\t:\t%lu\n", config.at("chunk_size"));

#define CHECK_INTEGRITY_2                                                                     \
  memcpy_allkinds<D2H>(h_out_data.get(), d_out_data.get(), data_len);                         \
  auto passed = true;                                                                         \
  printf("\n");                                                                               \
  for (size_t i = 0; i < data_len; i++) {                                                     \
    if (i < 10) printf("in=%d, out=%d\n", (int)h_out_data.get()[i], (int)h_in_data.get()[i]); \
    if (h_out_data.get()[i] != h_in_data.get()[i]) {                                          \
      printf("verification stops at idx=%lu\n", i);                                           \
      passed = false;                                                                         \
      break;                                                                                  \
    }                                                                                         \
  }                                                                                           \
  printf("\n");                                                                               \
  printf("passed? %s\n", passed ? "true" : "false");

#define PREPARE_1                                               \
  auto data_len = atoi(argv[2]);                                \
  auto config = fzg::configure_fzgpu(data_len);                 \
  auto buf = new fzg::Buf(data_len, true);                      \
                                                                \
  PRINT_CONFIG_1;                                               \
                                                                \
  utils::fromfile<uint16_t>(argv[1], buf->h_in_data, data_len); \
  memcpy_allkinds<H2D>(buf->d_in_data, buf->h_in_data, data_len);

#define PREPARE_2                                                            \
  auto data_len = atoi(argv[2]);                                             \
  auto config = fzg::configure_fzgpu(data_len);                              \
                                                                             \
  PRINT_CONFIG_2;                                                            \
                                                                             \
  uint8_t* d_archive;                                                        \
  size_t archive_len;                                                        \
                                                                             \
  auto len = config.at("len");                                               \
  auto pad_len = config.at("pad_len");                                       \
                                                                             \
  auto h_in_data = GPU_make_unique(malloc_h<u2>(len), GPU_DELETER_H());      \
  auto d_in_data = GPU_make_unique(malloc_d<u2>(pad_len), GPU_DELETER_D());  \
  auto h_out_data = GPU_make_unique(malloc_h<u2>(len), GPU_DELETER_H());     \
  auto d_out_data = GPU_make_unique(malloc_d<u2>(pad_len), GPU_DELETER_D()); \
                                                                             \
  utils::fromfile<uint16_t>(argv[1], h_in_data.get(), data_len);             \
  memcpy_allkinds<H2D>(d_in_data.get(), h_in_data.get(), data_len);

int low_level_demo(char** argv)
{
  PREPARE_1;
  auto stream = create_stream();
  ////////////////////////////////////////////////////////////////
  fzg::module::GPU_FZ_encode(
      buf->d_in_data, data_len, buf->d_offset_counter, buf->d_bitflag_array, buf->d_start_pos,
      buf->d_comp_out, buf->d_comp_len, stream);
  sync_by_stream(stream);

  PRINT_REPORT_1;

  fzg::module::GPU_FZ_decode(
      buf->d_comp_out, buf->d_bitflag_array, buf->d_start_pos, buf->d_out_data, data_len, stream);
  sync_by_stream(stream);

  CHECK_INTEGRITY_1;
  ////////////////////////////////////////////////////////////////
  destroy_stream(stream);
  return 0;
}

int high_level_demo(char** argv)
{
  PREPARE_2;

  auto codec = new fzg::FzgCodec(data_len);
  auto stream = create_stream();
  ////////////////////////////////////////////////////////////////
  codec->encode(d_in_data.get(), data_len, &d_archive, &archive_len, stream);
  sync_by_stream(stream);

  auto d_copy_out = GPU_make_unique(malloc_device<u1>(archive_len), GPU_deleter_device());
  memcpy_allkinds<D2D>(d_copy_out.get(), d_archive, archive_len);

  codec->decode(d_copy_out.get(), archive_len, d_out_data.get(), data_len, stream);
  sync_by_stream(stream);

  CHECK_INTEGRITY_2;
  ////////////////////////////////////////////////////////////////
  destroy_stream(stream);
  return 0;
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    printf("PROG  u2-binary-fname  len\n");
    exit(0);
  }

  printf("low-level demo:\n");
  low_level_demo(argv);
  printf("\n===============================\n");

  printf("high-level demo:\n");
  high_level_demo(argv);

  return 0;
}