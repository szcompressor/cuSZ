#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace fzgpu {

using config_map = std::unordered_map<std::string, size_t>;

config_map const configure_fzgpu(size_t const dtype_len);

class Buf {
  bool verify_on;

 public:
  fzgpu::config_map const* config;

  uint32_t h_offset_sum;

  uint32_t* d_bitflag_array;
  uint32_t* d_offset_counter;
  uint32_t* d_start_pos;

  uint16_t* d_comp_out;
  uint32_t* d_comp_len;

  bool* d_signum;

  uint16_t* h_in_data;
  uint16_t* d_in_data;

  uint16_t* h_out_data;
  uint16_t* d_out_data;

  Buf(fzgpu::config_map const* config, bool verify_on = false);
  ~Buf();

  size_t data_len() const { return config->at("pad_len") / 2; };
};

}  // namespace fzgpu

namespace fzgpu::cuhip {
int GPU_FZ_encode(
    uint16_t const* in_data, size_t const data_len,
    uint32_t* space_offset_counter, uint32_t* out_bitflag_array,
    uint32_t* out_start_pos, uint16_t* out_comp, uint32_t* comp_len,
    cudaStream_t stream);

int GPU_FZ_decode(
    uint16_t const* in_archive, uint32_t* in_bitflag_array,
    uint32_t* in_start_pos, uint16_t* out_decoded, size_t const decoded_len,
    cudaStream_t stream);
}  // namespace fzgpu::cuhip