// deps
#include "mem/array_cxx.h"
#include "cusz/type.h"
#include "exception/exception.hh"
#include "module/cxx_module.hh"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/timer.hh"
// definitions
#include "kernel/detail/histsp.cu_hip.inl"

using namespace portable;

template <pszpolicy policy, typename T, typename FQ, bool TIMING>
pszerror _2401::pszcxx_histogram_cauchy(
    array3<T> in, array3<u4> out_hist, float* milliseconds,
    void* stream)
try {
  auto inlen = in.len3.x;
  auto outlen = out_hist.len3.x;

  auto chunk = 32768;
  auto num_chunks = (inlen - 1) / chunk + 1;
  auto num_workers = 256;  // n SIMD-32

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  histsp_multiwarp<T, u4>
      <<<num_chunks, num_workers, sizeof(u4) * outlen, (cudaStream_t)stream>>>(
          in.buf, inlen, chunk, out_hist.buf, outlen, outlen / 2);
  STOP_GPUEVENT_RECORDING(stream);

  cudaStreamSynchronize((cudaStream_t)stream);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)