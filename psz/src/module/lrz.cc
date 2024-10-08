#include <cuda_runtime.h>

#include <exception>
#include <iostream>

#include "cusz.h"
#include "exception/exception.hh"
// #include "mem/array_c.h"
#include "pszcxx.hh"

using namespace portable;

pszerror psz_predict_lorenzo(
    psz_carray in, psz_rc const rc, psz_carray out_errquant,
    psz_outlier out_outlier, f4* t, void* stream)
try {
  if (out_errquant.dtype != U4)
    throw psz::exception_incorrect_type("errquant");

  float time_pred;
  auto l3 = in.len3;

  using DefaultEtypeRT = u2;

  if (in.dtype == F4)
    pszcxx_predict_lorenzo<f4, DefaultEtypeRT>(
        *((array3<f4>*)&in), rc, *((array3<DefaultEtypeRT>*)&out_errquant),
        *((compact_array1<f4>*)&out_outlier), t, stream);
  else if (in.dtype == F8)
    pszcxx_predict_lorenzo<f8, DefaultEtypeRT>(
        *((array3<f8>*)&in), rc, *((array3<DefaultEtypeRT>*)&out_errquant),
        *((compact_array1<f8>*)&out_outlier), t, stream);
  else
    throw psz::exception_incorrect_type("input");

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

pszerror psz_reverse_predict_lorenzo(
    psz_carray in_errquant, psz_outlier in_scattered_outlier, psz_rc const rc,
    psz_carray out_reconstruct, f4* t, void* stream)
try {
  using DefaultEtypeRT = u2;

  if (out_reconstruct.dtype == F4) {
    pszcxx_reverse_predict_lorenzo<f4, DefaultEtypeRT>(
        *((array3<DefaultEtypeRT>*)&in_errquant),
        *((array3<f4>*)&in_scattered_outlier), rc,
        *((array3<f4>*)&out_reconstruct), t, stream);
  }
  else if (out_reconstruct.dtype == F8) {
    pszcxx_reverse_predict_lorenzo<f8, DefaultEtypeRT>(
        *((array3<DefaultEtypeRT>*)&in_errquant),
        *((array3<f8>*)&in_scattered_outlier), rc,
        *((array3<f8>*)&out_reconstruct), t, stream);
  }
  throw psz::exception_incorrect_type("out_reconstruct");

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)