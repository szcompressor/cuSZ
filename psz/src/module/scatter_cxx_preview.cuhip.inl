// deps
#include "exception/exception.hh"
#include "kernel/lrz.hh"
#include "mem/array_cxx.h"
#include "mem/compact.hh"
#include "module/cxx_module.hh"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/timer.hh"
//
#include "kernel/detail/spvn.cuhip.inl"

template <psz_policy P, typename T, bool TIMING>
pszerror pszcxx_scatter_naive(
    compact_array1<T> in, array3<T> out, f4* milliseconds, void* stream)
try {
  auto grid_dim = (*(in.host_num) - 1) / 128 + 1;
  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);
  psz::KERNEL_CUHIP_spvn_scatter<T, u4>
      <<<grid_dim, 128, 0, (cudaStream_t)stream>>>(
          in.val, in.idx, *(in.host_num), out.buf);
  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

template <psz_policy P, typename T, bool TIMING>
pszerror pszcxx_gather_make_metadata_host_available(
    compact_array1<T> in, void* stream)
try {
  cudaMemcpyAsync(
      in.host_num, in.num, sizeof(u4), cudaMemcpyDeviceToHost,
      (cudaStream_t)stream);
  // TODO portability issue
  cudaStreamSynchronize((cudaStream_t)stream);

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)

////////////////////////////////////////////////////////////////////////////////
#define INSTANTIATE_PSZCXX_MODULE_SCATTER(BACKEND, T, TIMING)                \
  template pszerror pszcxx_scatter_naive<BACKEND, T, TIMING>(                \
      compact_array1<T> in, array3<T> out, f4 * milliseconds, void* stream); \
  template pszerror                                                          \
  pszcxx_gather_make_metadata_host_available<BACKEND, T, TIMING>(            \
      compact_array1<T> in, void* stream);
