__constant__ unsigned char PBK25_R128_BOOK_d[] = {
#include "pbk25_r128_book.inc"
};

__constant__ unsigned char PBK25_R128_RVBK_d[] = {
#include "pbk25_r128_rvbk.inc"
};

// Host-side accessors: resolve __constant__ addresses within this TU
// (cudaGetSymbolAddress cannot cross TU boundaries without -rdc)
extern "C" void* pbk25_r128_book_d_ptr()
{
  void* ptr = nullptr;
  cudaGetSymbolAddress(&ptr, PBK25_R128_BOOK_d);
  return ptr;
}

extern "C" void* pbk25_r128_rvbk_d_ptr()
{
  void* ptr = nullptr;
  cudaGetSymbolAddress(&ptr, PBK25_R128_RVBK_d);
  return ptr;
}

// Host mirror of the reverse book for CPU-side decode (PBKF).
static const unsigned char PBK25_R128_RVBK_h_arr[] = {
#include "pbk25_r128_rvbk.inc"
};

extern "C" void* pbk25_r128_rvbk_h_ptr() { return (void*)PBK25_R128_RVBK_h_arr; }
