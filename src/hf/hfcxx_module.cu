#include <cstdint>

#include "hfcxx_module.cu_hip.inl"

#define INIT1(E, H)                                             \
  template void _2403::phf_coarse_encode_phase1(                 \
      hfarray_cxx<E> in, hfarray_cxx<H> book, const int numSMs, \
      hfarray_cxx<H> out, float* time_lossless, void* stream);

#define INIT2(H)                                                           \
  template void _2403::phf_coarse_encode_phase2(                            \
      hfarray_cxx<H> in, hfpar_description hfpar, hfarray_cxx<H> deflated, \
      hfarray_cxx<u4> par_nbit, hfarray_cxx<u4> par_ncell,                 \
      float* time_lossless, void* stream);                                 \
  template void _2403::phf_coarse_encode_phase4(                            \
      hfarray_cxx<H> buf, hfarray_cxx<u4> par_entry,                       \
      hfarray_cxx<u4> par_ncell, hfpar_description hfpar,                  \
      hfarray_cxx<H> bitstream, float* time_lossless, void* stream);

template void _2403::phf_coarse_encode_phase3(
    hfarray_cxx<u4> d_par_nbit, hfarray_cxx<u4> d_par_ncell,
    hfarray_cxx<u4> d_par_entry, hfpar_description hfpar,
    hfarray_cxx<u4> h_par_nbit, hfarray_cxx<u4> h_par_ncell,
    hfarray_cxx<u4> h_par_entry, size_t* outlen_nbit, size_t* outlen_ncell,
    float* time_cpu_time, void* stream);

#define INIT3(E, H)                                                      \
  template void _2403::phf_coarse_decode(                                 \
      hfarray_cxx<H> bitstream, hfarray_cxx<uint8_t> revbook,            \
      hfarray_cxx<u4> par_nbit, hfarray_cxx<u4> par_entry,               \
      hfpar_description hfpar, hfarray_cxx<E> out, float* time_lossless, \
      void* stream);

INIT1(u1, u4)
INIT1(u2, u4)
INIT1(u4, u4)
INIT1(u1, u8)
INIT1(u2, u8)
INIT1(u4, u8)
INIT1(u1, ull)
INIT1(u2, ull)
INIT1(u4, ull)

INIT2(u4)
INIT2(u8)
INIT2(ull)

INIT3(u1, u4)
INIT3(u2, u4)
INIT3(u4, u4)
INIT3(u1, u8)
INIT3(u2, u8)
INIT3(u4, u8)
INIT3(u1, ull)
INIT3(u2, ull)
INIT3(u4, ull)

#undef INIT1
#undef INIT2
#undef INIT3