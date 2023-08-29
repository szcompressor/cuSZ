// deps
#include "port.hh"
// definitions
#include "detail/hf_codecg_drv.inl"

#define HF_CODEC_INIT(E, H, M)                                             \
  template void psz::hf_encode_coarse_rev2<E, H, M>(                       \
      E*, size_t const, hf_book*, hf_bitstream*, size_t*, size_t*, float*, \
      void*);                                                              \
                                                                           \
  template void psz::hf_decode_coarse<E, H, M>(                            \
      H*, uint8_t*, int const, M*, M*, int const, int const, E*, float*,   \
      void*);

// HF_CODEC_INIT(u1, u4, u4);
// HF_CODEC_INIT(u2, u4, u4);
HF_CODEC_INIT(u4, u4, u4);

// HF_CODEC_INIT(f4, u4, u4);
// HF_CODEC_INIT(u1, ull, u4);
// HF_CODEC_INIT(u2, ull, u4);
HF_CODEC_INIT(u4, ull, u4);
// HF_CODEC_INIT(f4, ull, u4);

#undef HFBOOK_INIT
#undef HF_CODEC_INIT
