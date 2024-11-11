#ifndef F6AA22B8_F6B5_4C0A_AF44_573093D20EF7
#define F6AA22B8_F6B5_4C0A_AF44_573093D20EF7

#include <memory>

#include "fzg_kernel.hh"
#include "fzg_type.h"

namespace psz {

class FzgCodec {
  using E = uint16_t;
  using Buf = fzgpu::Buf;

 private:
  struct impl;
  std::unique_ptr<impl> pimpl;

 public:
  FzgCodec(size_t const data_len);
  ~FzgCodec();

  FzgCodec* encode(
      E* in_data, size_t const data_len, uint8_t** out_comp, size_t* comp_len,
      void* stream);

  FzgCodec* decode(
      uint8_t* in_comp, size_t const comp_len, E* out_data,
      size_t const data_len, void* stream);

  size_t expose_padded_input_len() const;
};

}  // namespace psz

#endif /* F6AA22B8_F6B5_4C0A_AF44_573093D20EF7 */
