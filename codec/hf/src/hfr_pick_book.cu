// HFR-v3: pick a best-fit global PBK
// reuse hfr_pbk::{probs_lookup, _pbk_argmin} 
#include "_future/hfr-pbk.cuh"  

namespace phf::module {

namespace {
using u4 = uint32_t;

// global top-1 prob (max histogram bin / len) -> nearest PBK book; copy it + emit id.
__global__ void KCU_HFR_pick_pbk(
    const u4* hist, u4 bklen, size_t len, const u4* pbk_book, u4* book_out, u4* out_encid,
    u4 book_len)
{
  __shared__ u4 s_max;
  __shared__ int s_tree_idx;
  if (threadIdx.x == 0) s_max = 0;
  __syncthreads();

  u4 local_max = 0;
  for (u4 i = threadIdx.x; i < bklen; i += blockDim.x) local_max = max(local_max, hist[i]);
  for (int s = 16; s > 0; s >>= 1)
    local_max = max(local_max, __shfl_xor_sync(0xffffffffu, local_max, s));
  if ((threadIdx.x & 31) == 0) atomicMax(&s_max, local_max);
  __syncthreads();

  if (threadIdx.x < 32) {
    float prob = (float)s_max / (float)len;
    int idx = hfr_pbk::_pbk_argmin<(int)psz::HFR_PBK_Constants::NumBooks>(prob);
    if (threadIdx.x == 0) {
      s_tree_idx = idx;
      *out_encid = (u4)idx;  // global PBK id -> patched into the archive header
    }
  }
  __syncthreads();

  const int tree_idx = s_tree_idx;
  for (u4 i = threadIdx.x; i < book_len; i += blockDim.x)
    book_out[i] = pbk_book[(size_t)tree_idx * book_len + i];
}
}  // namespace

int HFR_pick_pbk(
    uint32_t const* hist_d, uint32_t bklen, size_t len, uint32_t const* pbk_book, uint32_t* book_d,
    uint32_t* encid_d, void* stream)
{
  constexpr u4 BookLen = psz::HFR_PBK_Constants::Radius * 2;  // 256 H4 entries
  KCU_HFR_pick_pbk<<<1, 256, 0, (cudaStream_t)stream>>>(
      hist_d, bklen, len, pbk_book, book_d, encid_d, BookLen);
  return 0;
}

}  // namespace phf::module
