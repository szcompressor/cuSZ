/**
 * @file dryrun.cuhip.inl
 * @author Jiannan Tian
 * @brief cuSZ dryrun mode, checking data quality from lossy compression.
 * @version 0.3
 * @date 2020-09-20
 * (create) 2020-05-14, (release) 2020-09-20, (rev1) 2021-01-25, (rev2)
 * 2021-06-21
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef A248A007_AE47_424C_BF3C_95F41AF049CA
#define A248A007_AE47_424C_BF3C_95F41AF049CA

namespace psz {

template <typename T = float, typename FP = T, int BLOCK = 256, int SEQ = 4>
[[obsolete]] __global__ void KERNLE_CUHIP_lorenzo_dryrun(
    T* in, T* out, size_t len, FP ebx2_r, FP ebx2)
{
  {
    constexpr auto NTHREAD = BLOCK / SEQ;
    __shared__ T shmem[BLOCK];
    auto id_base = blockIdx.x * BLOCK;

#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
      auto id = id_base + threadIdx.x + i * NTHREAD;
      if (id < len) {
        shmem[threadIdx.x + i * NTHREAD] = round(in[id] * ebx2_r) * ebx2;
        out[id] = shmem[threadIdx.x + i * NTHREAD];
      }
    }
  }
}

}  // namespace psz

namespace psz::cuhip {

template <typename T>
[[deprecated]] void GPU_lorenzo_dryrun(
    size_t len, T* original, T* reconst, PROPER_EB eb, void* stream)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2_r = 1 / (eb * 2);
  auto ebx2 = eb * 2;

  KERNLE_CUHIP_lorenzo_dryrun<<<div(len, 256), 256, 256 * sizeof(T), (cudaStream_t)stream>>>(
      original, reconst, len, ebx2_r, ebx2);

  // CHECK_CUDA(cudaStreamSynchronize((cudaStream_t)stream));
  cudaStreamSynchronize((cudaStream_t)stream);
}

}  // namespace psz::cuhip

#endif /* A248A007_AE47_424C_BF3C_95F41AF049CA */
