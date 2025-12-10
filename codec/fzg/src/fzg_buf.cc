
#include <cstddef>
#include <cstdint>
#include <memory>

#include "fzg_hl.hh"
#include "mem/cxx_backends.h"

struct fzg::Buf2::impl {
  using Header = fzg_header;
  using E = uint16_t;
  using InputT = uint16_t;

  // constants from original configure_fzgpu()
  static constexpr size_t UINT32_BIT_LEN = 32;
  static constexpr size_t BLOCK_SIZE = 16;
  static constexpr size_t PAGE_BYTES = 4096;

  const size_t len;         // original data length (elements)
  const size_t data_bytes;  // padded bytes
  const size_t pad_len;     // padded length in elements
  const size_t chunk_sz;    // "chunk_size"
  const size_t grid_x_;     // "grid_x"
  const size_t max_archive_bytes;

  GPU_unique_dptr<uint8_t[]> d_archive;
  uint32_t* d_bitflag_array = nullptr;
  uint32_t* d_start_pos = nullptr;
  uint8_t* d_comp_out = nullptr;

  GPU_unique_dptr<uint32_t[]> d_offset_counter;
  GPU_unique_dptr<uint32_t[]> d_comp_len;
  GPU_unique_dptr<bool[]> d_signum;  // length: pad_len (heuristic)

  static size_t align_data_bytes(size_t data_bytes)
  {
    if (data_bytes == 0) return 0;
    data_bytes = (data_bytes - 1) / PAGE_BYTES + 1;
    return data_bytes * PAGE_BYTES;
  }

  static size_t compute_chunk_size(size_t data_bytes)
  {
    size_t denom = BLOCK_SIZE * UINT32_BIT_LEN;
    return (data_bytes + denom - 1) / denom;
  }

  static size_t compute_grid_x(size_t data_bytes)
  {
    // floor(data_bytes / 4096) from original code
    return data_bytes / PAGE_BYTES;
  }

  static size_t compute_max_archive_bytes(
      size_t pad_len, size_t chunk_sz, size_t grid_x, size_t len)
  {
    return sizeof(Header)                 // field 1: header
           + sizeof(uint32_t) * chunk_sz  // field 2: bitflag
           + sizeof(uint32_t) * grid_x    // field 3: start pos
           + sizeof(InputT) * len;        // field 4: max compressed output
  }

  impl(size_t data_len) :
      len(data_len),
      data_bytes(align_data_bytes(data_len * sizeof(InputT))),
      pad_len(data_bytes / sizeof(InputT)),
      chunk_sz(compute_chunk_size(data_bytes)),
      grid_x_(compute_grid_x(data_bytes)),
      max_archive_bytes(compute_max_archive_bytes(pad_len, chunk_sz, grid_x_, len))
  {
    d_archive = MAKE_UNIQUE_DEVICE(uint8_t, max_archive_bytes);

    auto base = d_archive.get();
    d_bitflag_array = reinterpret_cast<uint32_t*>(base + sizeof(Header));
    d_start_pos = reinterpret_cast<uint32_t*>(base + sizeof(Header) + sizeof(uint32_t) * chunk_sz);
    d_comp_out = base + sizeof(Header) + sizeof(uint32_t) * chunk_sz + sizeof(uint32_t) * grid_x_;

    d_offset_counter = MAKE_UNIQUE_DEVICE(uint32_t, 1);
    d_comp_len = MAKE_UNIQUE_DEVICE(uint32_t, grid_x_);
    d_signum = MAKE_UNIQUE_DEVICE(bool, pad_len);  // heuristic; adjust if needed
  }

  ~impl() = default;

  void memcpy_merge(Header& header, void* stream)
  {
    // Layout is already:
    //   [header | bitflag | start_pos | comp_out]
    // so we only need to copy the header into the beginning of d_archive.
    cudaMemcpyAsync(
        d_archive.get(), &header, sizeof(Header), cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream));
  }

  void clear_buffer()
  {
    memset_device(d_archive.get(), max_archive_bytes);
    memset_device(d_offset_counter.get(), 1);
    memset_device(d_comp_len.get(), grid_x_);
    memset_device(d_signum.get(), pad_len);
  }
};

fzg::Buf2::Buf2(size_t data_len) : pimpl(std::make_unique<impl>(data_len)) {}

fzg::Buf2::~Buf2() = default;

size_t fzg::Buf2::len() const { return pimpl->len; }
size_t fzg::Buf2::pad_len() const { return pimpl->pad_len; }
size_t fzg::Buf2::data_bytes() const { return pimpl->data_bytes; }
size_t fzg::Buf2::chunk_size() const { return pimpl->chunk_sz; }
size_t fzg::Buf2::grid_x() const { return pimpl->grid_x_; }
size_t fzg::Buf2::archive_bytes() const { return pimpl->max_archive_bytes; }

// device pointers
uint32_t* fzg::Buf2::bitflag_d() const { return pimpl->d_bitflag_array; }
uint32_t* fzg::Buf2::start_pos_d() const { return pimpl->d_start_pos; }
uint8_t* fzg::Buf2::comp_out_d() const { return pimpl->d_comp_out; }
uint8_t* fzg::Buf2::archive_d() const { return pimpl->d_archive.get(); }
uint32_t* fzg::Buf2::comp_len_d() const { return pimpl->d_comp_len.get(); }
uint32_t* fzg::Buf2::offset_counter_d() const { return pimpl->d_offset_counter.get(); }
bool* fzg::Buf2::signum_d() const { return pimpl->d_signum.get(); }

// ops
void fzg::Buf2::clear_buffer() { pimpl->clear_buffer(); }
void fzg::Buf2::memcpy_merge(Header& header, void* stream) { pimpl->memcpy_merge(header, stream); }

// static helper: drop config_map, but keep same math
size_t fzg::Buf2::padded_len(size_t data_len)
{
  using InputT = fzg::Buf2::InputT;
  constexpr size_t PAGE_BYTES = impl::PAGE_BYTES;

  size_t data_bytes = data_len * sizeof(InputT);
  if (data_bytes == 0) return 0;

  data_bytes = (data_bytes - 1) / PAGE_BYTES + 1;
  data_bytes *= PAGE_BYTES;
  return data_bytes / sizeof(InputT);
}
