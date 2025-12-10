

#include "fzg_hl.hh"

#include <cuda_runtime.h>

#include <string>

#include "fzg_impl.hh"

using std::to_string;

int fzg::high_level::encode(
    Buf2* buf, E* in_data, std::size_t const data_len, uint8_t** out_archive,
    std::size_t* archive_len, fzg_header& header, void* stream)
{
  using M = uint32_t;

  fzg::module::GPU_FZ_encode(
      in_data, data_len, buf->offset_counter_d(), buf->bitflag_d(), buf->start_pos_d(),
      buf->comp_out_d(), buf->comp_len_d(), stream);

  cudaStreamSynchronize(static_cast<cudaStream_t>(stream));

  cudaMemcpy(
      &buf->h_offset_sum, buf->offset_counter_d(), sizeof(uint32_t), cudaMemcpyDeviceToHost);

  header.original_len = data_len;

  M nbyte[FZGHEADER_END];
  nbyte[FZGHEADER_HEADER] = sizeof(fzg_header);                       // header itself
  nbyte[FZGHEADER_BITFLAG] = sizeof(uint32_t) * buf->chunk_size();    // per-chunk flags
  nbyte[FZGHEADER_START_POS] = sizeof(uint32_t) * buf->grid_x();      // per-grid block offsets
  nbyte[FZGHEADER_BITSTREAM] = buf->h_offset_sum * sizeof(uint32_t);  // encoded bitstream

  header.entry[0] = 0;
  // *.END + 1: need to know ending position (same pattern as PHF)
  for (int i = 1; i < FZGHEADER_END + 1; ++i) header.entry[i] = nbyte[i - 1];
  for (int i = 1; i < FZGHEADER_END + 1; ++i) header.entry[i] += header.entry[i - 1];

  buf->memcpy_merge(header, stream);  // H2D copy of header to the start of d_archive

  *out_archive = buf->archive_d();
  *archive_len = static_cast<std::size_t>(header.entry[FZGHEADER_END]);

  return 0;
}

int fzg::high_level::decode(
    Buf2* /*buf*/, fzg_header& header, uint8_t* in_archive, std::size_t const /*archive_len*/,
    E* out_data, std::size_t const data_len, void* stream)
{
  // in_archive is a DEVICE pointer (d_archive)
  cudaMemcpy(&header, in_archive, sizeof(header), cudaMemcpyDeviceToHost);

#define FZG_ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_archive + header.entry[FZGHEADER_##SYM])

  fzg::module::GPU_FZ_decode(
      FZG_ACCESSOR(BITSTREAM, uint8_t),   // in_archive + entry[BITSTREAM]
      FZG_ACCESSOR(BITFLAG, uint32_t),    // in_archive + entry[BITFLAG]
      FZG_ACCESSOR(START_POS, uint32_t),  // in_archive + entry[START_POS]
      out_data, data_len, stream);

#undef FZG_ACCESSOR

  return 0;
}
