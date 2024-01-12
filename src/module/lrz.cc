#include <cuda_runtime.h>

#include <exception>
#include <iostream>

#include "cusz.h"
#include "cusz/array.h"
#include "exception/exception.hh"
#include "pszcxx.hh"

pszerror psz_predict_lorenzo(
    pszarray in, pszrc2 const rc, pszarray out_errquant,
    pszoutlier out_outlier, void* stream)
try {
  if (out_errquant.dtype != U4)
    throw psz::exception_incorrect_type("errquant");

  float time_pred;
  auto l3 = in.len3;

  if (in.dtype == F4)
    _2401::pszcxx_predict_lorenzo<f4>(
        *((pszarray_cxx<f4>*)&in), rc, *((pszarray_cxx<u4>*)&out_errquant),
        *((pszcompact_cxx<f4>*)&out_outlier), stream);
  else if (in.dtype == F8)
    _2401::pszcxx_predict_lorenzo<f8>(
        *((pszarray_cxx<f8>*)&in), rc, *((pszarray_cxx<u4>*)&out_errquant),
        *((pszcompact_cxx<f8>*)&out_outlier), stream);
  else
    throw psz::exception_incorrect_type("input");

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

pszerror psz_reverse_predict_lorenzo(
    pszarray in_errquant, pszoutlier in_scattered_outlier, pszrc2 const rc,
    pszarray out_reconstruct, void* stream)
try {
  if (out_reconstruct.dtype == F4) {
    _2401::pszcxx_reverse_predict_lorenzo(
        *((pszarray_cxx<u4>*)&in_errquant),
        *((pszarray_cxx<f4>*)&in_scattered_outlier), rc,
        *((pszarray_cxx<f4>*)&out_reconstruct), stream);
  }
  else if (out_reconstruct.dtype == F8) {
    _2401::pszcxx_reverse_predict_lorenzo(
        *((pszarray_cxx<u4>*)&in_errquant),
        *((pszarray_cxx<f8>*)&in_scattered_outlier), rc,
        *((pszarray_cxx<f8>*)&out_reconstruct), stream);
  }
  throw psz::exception_incorrect_type("out_reconstruct");

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)