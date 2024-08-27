// deps
#include <stdexcept>

#include "cusz/type.h"
#include "exception/exception.hh"
#include "kernel/lrz.hh"
#include "mem/array_cxx.h"
#include "mem/compact.hh"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/timer.hh"
// definitions
#include "kernel/detail/l23_x.cuhip.inl"
#include "kernel/detail/l23r.cuhip.inl"
#include "module/cxx_module.hh"

namespace _2401 {

#define LRZ_WRAPPER_TPL template <typename T, typename E, psz_timing_mode TIMING>
#define LRZ_WRAPPER_CLASS pszpred_lrz<T, E, TIMING>

LRZ_WRAPPER_TPL pszerror LRZ_WRAPPER_CLASS::pszcxx_predict_lorenzo(
    array3<T> in, psz_rc const rc, array3<E> out_errquant,
    compact_array1<T> out_outlier, f4* time_elapsed, void* stream)
try {
  auto len3 = dim3(in.len3.x, in.len3.y, in.len3.z);

  auto compact = typename CompactDram<CUDA, T>::Compact(out_outlier);
//       out_outlier.val,
//       out_outlier.idx,
//       out_outlier.num,
//   };

  pszcxx_predict_lorenzo__internal<T, E, TIMING>(
      in.buf, len3, rc.eb, rc.radius, out_errquant.buf, (void*)&compact,
      time_elapsed, stream);

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

LRZ_WRAPPER_TPL pszerror LRZ_WRAPPER_CLASS::pszcxx_reverse_predict_lorenzo(
    array3<E> in_errquant, array3<T> in_scattered_outlier, psz_rc const rc,
    array3<T> out_xdata, f4* time_elapsed, void* stream)
try {
  auto len3 = dim3(out_xdata.len3.x, out_xdata.len3.y, out_xdata.len3.z);

  pszcxx_reverse_predict_lorenzo__internal<T, E, TIMING>(
      in_errquant.buf, len3, out_xdata.buf, rc.eb, rc.radius, out_xdata.buf,
      time_elapsed, stream);

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

}  // namespace _2401
