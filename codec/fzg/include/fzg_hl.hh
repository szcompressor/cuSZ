#ifndef FZG_HL_HH
#define FZG_HL_HH

#include <memory>
#include <string>
#include <unordered_map>

#include "fzg.h"

namespace fzg {
using E = u2;
using config_map = std::unordered_map<std::string, size_t>;
struct Buf2;
struct high_level;
config_map const configure_fzgpu(size_t const dtype_len);
}  // namespace fzg

struct fzg::Buf2 {
  struct impl;
  std::unique_ptr<impl> pimpl;

  using Header = fzg_header;
  using E = uint16_t;
  using InputT = uint16_t;

  Buf2(size_t data_len);
  ~Buf2();

  size_t len() const;
  size_t pad_len() const;
  size_t data_bytes() const;
  size_t chunk_size() const;
  size_t grid_x() const;
  size_t archive_bytes() const;

  uint32_t* bitflag_d() const;
  uint32_t* start_pos_d() const;
  uint8_t* comp_out_d() const;
  uint8_t* archive_d() const;
  uint32_t* comp_len_d() const;
  uint32_t* offset_counter_d() const;
  bool* signum_d() const;

  E* in_h() const;
  E* in_d() const;
  E* out_h() const;
  E* out_d() const;

  void clear_buffer();
  void memcpy_merge(Header& header, void* stream);

  static size_t padded_len(size_t data_len);

  uint32_t h_offset_sum = 0;
};

struct fzg::high_level {
  static int encode(
      Buf2* buf, E* in_data, size_t const data_len, uint8_t** out_archive, size_t* archive_len,
      fzg_header& header, void* stream);

  static int decode(
      Buf2* buf, fzg_header& header, uint8_t* in_archive,
      size_t const archive_len,  // can be unused, kept for symmetry
      E* out_data, size_t const data_len, void* stream);
};

#endif /* FZG_HL_HH */
