#include <cstdint>
#include <string>
#include <unordered_map>

#include "detail/fzg_c.cuhip.inl"
#include "detail/fzg_x.cuhip.inl"
#include "kernel/fzg_cx.hh"
#include "utils/err/err.cuhip.hh"

fzgpu::config_map const fzgpu::configure_fzgpu(size_t const data_len)
{
  constexpr auto UINT32_BIT_LEN = 32;
  constexpr auto block_size = 16;
  /* align */ auto data_bytes = data_len * 2;  // how many bytes of data
  /* align */ data_bytes = (data_bytes - 1) / 4096 + 1;
  /* align */ data_bytes *= 4096;
  auto pad_len = data_bytes / 2;
  int data_chunk_size =
      data_bytes % (block_size * UINT32_BIT_LEN) == 0
          ? data_bytes / (block_size * UINT32_BIT_LEN)
          : int(data_bytes / (block_size * UINT32_BIT_LEN)) + 1;

  return config_map{
      {"len", data_len},          {"bytes", data_len * sizeof(float)},
      {"pad_len", pad_len},       {"chunk_size", data_chunk_size},
      {"data_bytes", data_bytes}, {"grid_x", floor(data_bytes / 4096)}};
}

fzgpu::Buf::Buf(config_map const* _config, bool _verifiy_on) :
    verify_on(_verifiy_on), config(_config)
{
  auto grid_x = config->at("grid_x");
  auto pad_len = config->at("pad_len");
  auto chunk_size = config->at("chunk_size");
  auto len = config->at("len");

  CHECK_GPU(cudaMallocHost(&h_in_data, sizeof(uint16_t) * len));
  CHECK_GPU(cudaMalloc(&d_in_data, sizeof(uint16_t) * pad_len));

  CHECK_GPU(cudaMalloc(&d_comp_out, sizeof(uint16_t) * len));
  CHECK_GPU(cudaMalloc(&d_bitflag_array, sizeof(uint32_t) * chunk_size));

  CHECK_GPU(cudaMallocHost(&h_out_data, sizeof(uint16_t) * len));
  CHECK_GPU(cudaMalloc(&d_out_data, sizeof(uint16_t) * pad_len));

  CHECK_GPU(cudaMalloc(&d_offset_counter, sizeof(uint32_t)));
  CHECK_GPU(cudaMalloc(&d_start_pos, sizeof(uint32_t) * grid_x));
  CHECK_GPU(cudaMalloc(&d_comp_len, sizeof(uint32_t) * grid_x));

  CHECK_GPU(cudaMemset(d_in_data, 0, sizeof(uint16_t) * len));
  CHECK_GPU(cudaMemset(d_out_data, 0, sizeof(uint16_t) * len));
  CHECK_GPU(cudaMemset(d_bitflag_array, 0, sizeof(uint32_t) * chunk_size));
  CHECK_GPU(cudaMemset(d_offset_counter, 0, sizeof(uint32_t)));
  CHECK_GPU(cudaMemset(d_start_pos, 0, sizeof(uint32_t) * grid_x));
  CHECK_GPU(cudaMemset(d_comp_len, 0, sizeof(uint32_t) * grid_x));
}

fzgpu::Buf::~Buf()
{
  CHECK_GPU(cudaFreeHost(h_in_data));
  CHECK_GPU(cudaFree(d_in_data));

  CHECK_GPU(cudaFree(d_out_data));
  CHECK_GPU(cudaFreeHost(h_out_data));

  CHECK_GPU(cudaFree(d_signum));

  CHECK_GPU(cudaFree(d_comp_out));
  CHECK_GPU(cudaFree(d_bitflag_array));

  CHECK_GPU(cudaFree(d_offset_counter));
  CHECK_GPU(cudaFree(d_start_pos));
  CHECK_GPU(cudaFree(d_comp_len));
}

int fzgpu::cuhip::GPU_FZ_encode(
    uint16_t const* in_data, size_t const data_len,
    uint32_t* space_offset_counter, uint32_t* out_bitflag_array,
    uint32_t* out_start_pos, uint16_t* out_comp, uint32_t* comp_len,
    cudaStream_t stream)
{
  auto config = fzgpu::configure_fzgpu(data_len);
  dim3 grid = dim3(config["grid_x"]);
  dim3 block(32, 32);

  fzgpu::KERNEL_CUHIP_fz_fused_encode<<<grid, block, 0, stream>>>(
      (uint32_t*)in_data, config["pad_len"] / 2, space_offset_counter,
      out_bitflag_array, out_start_pos, (uint32_t*)out_comp, comp_len);

  cudaStreamSynchronize(stream);

  return 0;
}

int fzgpu::cuhip::GPU_FZ_decode(
    uint16_t const* in_archive, uint32_t* in_bitflag_array,
    uint32_t* in_start_pos, uint16_t* out_decoded, size_t const decoded_len,
    cudaStream_t stream)
{
  auto config = fzgpu::configure_fzgpu(decoded_len);
  dim3 grid = dim3(config["grid_x"]);
  dim3 block(32, 32);

  fzgpu::KERNEL_CUHIP_fz_fused_decode<<<grid, block, 0, stream>>>(
      (uint32_t*)in_archive, in_bitflag_array, in_start_pos,
      (uint32_t*)out_decoded, config["pad_len"] / 2);

  cudaStreamSynchronize(stream);

  return 0;
}
