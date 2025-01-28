#ifndef C6EEF7E5_E411_4155_A85F_52E9EB99FCF6
#define C6EEF7E5_E411_4155_A85F_52E9EB99FCF6

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "fzg_type.h"

namespace fzgpu {

using config_map = std::unordered_map<std::string, size_t>;

config_map const configure_fzgpu(size_t const dtype_len);

class Buf {
 private:
  void init(size_t data_len, bool alloc_test_buf);

  fzgpu::config_map config;
  bool alloc_test_buf;

 public:
  using Header = fzg_header;
  using E = uint16_t;
  using InputT = uint16_t;

  uint32_t h_offset_sum;

  uint32_t* d_bitflag_array;
  uint32_t* d_offset_counter;
  uint32_t* d_start_pos;

  uint8_t* d_comp_out;
  uint32_t* d_comp_len;

  // overlap with d_comp_out to save space
  uint8_t* d_archive;

  bool* d_signum;

  uint16_t* h_in_data;
  uint16_t* d_in_data;

  uint16_t* h_out_data;
  uint16_t* d_out_data;

  size_t max_archive_bytes;

  Buf(size_t const data_len, bool alloc_test_buf);
  ~Buf();

  void clear_buffer();

  void memcpy_merge(Header& header, void* stream);

  static size_t padded_len(size_t const data_len);

  fzgpu::config_map configuration() const { return config; }
  size_t data_len() const { return config.at("pad_len") / 2; };
};

}  // namespace fzgpu

namespace fzgpu::cuhip {
int GPU_FZ_encode(
    uint16_t const* in_data, size_t const data_len, uint32_t* space_offset_counter,
    uint32_t* out_bitflag_array, uint32_t* out_start_pos, uint8_t* out_comp, uint32_t* comp_len,
    cudaStream_t stream);

int GPU_FZ_decode(
    uint8_t const* in_archive, uint32_t* in_bitflag_array, uint32_t* in_start_pos,
    uint16_t* out_decoded, size_t const decoded_len, cudaStream_t stream);
}  // namespace fzgpu::cuhip

#endif /* C6EEF7E5_E411_4155_A85F_52E9EB99FCF6 */
