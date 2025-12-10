#include <cstdint>
#include <string>
#include <unordered_map>

#include "detail/fzg_c.cuhip.inl"
#include "detail/fzg_x.cuhip.inl"
#include "fzg_hl.hh"
#include "fzg_impl.hh"

fzg::config_map const fzg::configure_fzgpu(size_t const data_len)
{
  constexpr auto UINT32_BIT_LEN = 32;
  constexpr auto block_size = 16;
  /* align */ auto data_bytes = data_len * 2;  // how many bytes of data
  /* align */ data_bytes = (data_bytes - 1) / 4096 + 1;
  /* align */ data_bytes *= 4096;
  auto pad_len = data_bytes / 2;
  int data_chunk_size = data_bytes % (block_size * UINT32_BIT_LEN) == 0
                            ? data_bytes / (block_size * UINT32_BIT_LEN)
                            : int(data_bytes / (block_size * UINT32_BIT_LEN)) + 1;

  return config_map{{"len", data_len},          {"bytes", data_len * sizeof(float)},
                    {"pad_len", pad_len},       {"chunk_size", data_chunk_size},
                    {"data_bytes", data_bytes}, {"grid_x", floor(data_bytes / 4096)}};
}

int fzg::module::GPU_FZ_encode(
    uint16_t const* in_data, size_t const data_len, uint32_t* space_offset_counter,
    uint32_t* out_bitflag_array, uint32_t* out_start_pos, uint8_t* out_comp, uint32_t* comp_len,
    void* stream)
{
  auto config = fzg::configure_fzgpu(data_len);
  dim3 grid = dim3(config["grid_x"]);
  dim3 block(32, 32);

  fzg::KERNEL_CUHIP_fz_fused_encode<<<grid, block, 0, (cudaStream_t)stream>>>(
      (uint32_t*)in_data, config["pad_len"] / 2, space_offset_counter, out_bitflag_array,
      out_start_pos, (uint32_t*)out_comp, comp_len);

  cudaStreamSynchronize((cudaStream_t)stream);

  return 0;
}

int fzg::module::GPU_FZ_decode(
    uint8_t const* in_archive, uint32_t* in_bitflag_array, uint32_t* in_start_pos,
    uint16_t* out_decoded, size_t const decoded_len, void* stream)
{
  auto config = fzg::configure_fzgpu(decoded_len);
  dim3 grid = dim3(config["grid_x"]);
  dim3 block(32, 32);

  fzg::KERNEL_CUHIP_fz_fused_decode<<<grid, block, 0, (cudaStream_t)stream>>>(
      (uint32_t*)in_archive, in_bitflag_array, in_start_pos, (uint32_t*)out_decoded,
      config["pad_len"] / 2);

  cudaStreamSynchronize((cudaStream_t)stream);

  return 0;
}
