#ifndef FZG_IMPL_HH
#define FZG_IMPL_HH

#include <cstddef>
#include <cstdint>

namespace fzg {

struct module {
  static int GPU_FZ_encode(
      uint16_t const* in_data, size_t const data_len, uint32_t* space_offset_counter,
      uint32_t* out_bitflag_array, uint32_t* out_start_pos, uint8_t* out_comp, uint32_t* comp_len,
      void* stream);

  static int GPU_FZ_decode(
      uint8_t const* in_archive, uint32_t* in_bitflag_array, uint32_t* in_start_pos,
      uint16_t* out_decoded, size_t const decoded_len, void* stream);
};

}  // namespace fzg

#endif /* FZG_IMPL_HH */
