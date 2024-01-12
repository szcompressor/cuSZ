// deps
#include "cusz/cxx_array.hh"
#include "cusz/type.h"
#include "exception/exception.hh"
#include "kernel/lrz.hh"
#include "mem/compact.hh"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/timer.hh"
//
#include "kernel/detail/spvn.cu_hip.inl"

template <pszpolicy P, typename T>
pszerror _2401::pszcxx_scatter_naive(
    pszcompact_cxx<T> in, pszarray_cxx<T> out, f4* milliseconds, void* stream)
try {
  auto grid_dim = (*(in.host_num) - 1) / 128 + 1;
  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);
  psz::cu_hip::spvn_scatter<T, u4><<<grid_dim, 128, 0, (cudaStream_t)stream>>>(
      in.val, in.idx, *(in.host_num), out.buf);
  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(GpuStreamSync(stream));
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

template <pszpolicy P, typename T>
pszerror pszcxx_gather_make_metadata_host_available(
    pszcompact_cxx<T> in, void* stream)
try {
  cudaMemcpyAsync(in.host_num, in.num, sizeof(u4));
  // TODO portability issue
  cudaStreamSynchronize((GpuStreamT)stream);
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)