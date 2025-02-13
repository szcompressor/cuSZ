#include "criteria.gpu.hh"
#include "cusz/type.h"
#include "detail/spvn.cuhip.inl"
#include "kernel/spv.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

// TODO only 1 CUHIP
#define SPVN_GATHER(T, Criterion, M)                                            \
  template <>                                                                   \
  void psz::spv_gather_naive<CUDA, T, Criterion, M>(                            \
      T * in, size_t const in_len, int const radius, T* cval, M* cidx, int* cn, \
      Criterion criteria, f4* milliseconds, void* stream)                       \
  {                                                                             \
    auto grid_dim = (in_len - 1) / 128 + 1;                                     \
    CREATE_GPUEVENT_PAIR;                                                       \
    START_GPUEVENT_RECORDING(stream);                                           \
    psz::KERNEL_CUHIP_spvn_gather<<<grid_dim, 128, 0, (cudaStream_t)stream>>>(  \
        in, in_len, radius, cval, cidx, cn, criteria);                          \
    STOP_GPUEVENT_RECORDING(stream);                                            \
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));                     \
    TIME_ELAPSED_GPUEVENT(milliseconds);                                        \
    DESTROY_GPUEVENT_PAIR;                                                      \
  }

#define SPVN_SCATTER(T, M)                                                     \
  template <>                                                                  \
  void psz::spv_scatter_naive<CUDA, T, M>(                                     \
      T * val, M * idx, int const nnz, T* out, f4* milliseconds, void* stream) \
  {                                                                            \
    auto grid_dim = (nnz - 1) / 128 + 1;                                       \
    CREATE_GPUEVENT_PAIR;                                                      \
    START_GPUEVENT_RECORDING(stream);                                          \
    psz::KERNEL_CUHIP_spvn_scatter<T, M>                                       \
        <<<grid_dim, 128, 0, (cudaStream_t)stream>>>(val, idx, nnz, out);      \
    STOP_GPUEVENT_RECORDING(stream);                                           \
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));                    \
    if (milliseconds) TIME_ELAPSED_GPUEVENT(milliseconds);                     \
    DESTROY_GPUEVENT_PAIR;                                                     \
  }

SPVN_SCATTER(u1, u4)
SPVN_SCATTER(u2, u4)
SPVN_SCATTER(f4, u4)
SPVN_SCATTER(f8, u4)

SPVN_GATHER(f4, psz::criterion::gpu::eq<f4>, u4)
SPVN_GATHER(f4, psz::criterion::gpu::in_ball<f4>, u4)
SPVN_GATHER(f4, psz::criterion::gpu::in_ball_shifted<f4>, u4)
SPVN_GATHER(f8, psz::criterion::gpu::eq<f8>, u4)
SPVN_GATHER(f8, psz::criterion::gpu::in_ball<f8>, u4)
SPVN_GATHER(f8, psz::criterion::gpu::in_ball_shifted<f8>, u4)

#undef SPVN_SCATTER