/**
 * @file hist_sp.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-05-18
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail2/hist_sp.inl"
#include "kernel2/hist_sp.hh"

template <typename T, typename FQ>
int histsp(
    T* in, uint32_t inlen, FQ* out, uint32_t outlen, cudaStream_t stream)
{
  constexpr auto CHUNK = 32768;
  constexpr auto NWARP = 8;
  constexpr auto NTREAD = 32 * NWARP;

  histsp_multiwarp<T, NWARP, CHUNK, FQ>
      <<<(inlen - 1) / CHUNK + 1, NTREAD, sizeof(FQ) * outlen, stream>>>(
          in, inlen, out, outlen, outlen / 2);

  return 0;
}

template int histsp<uint32_t>(
    uint32_t*, uint32_t, uint32_t*, uint32_t, cudaStream_t);
