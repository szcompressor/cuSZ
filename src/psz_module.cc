#include <exception>
#include <iostream>

#include "cusz.h"
#include "cusz/type.h"
#include "exception/exception.hh"
#include "kernel/lrz/l23r.hh"
#include "mem/compact.hh"
#include "port.hh"

pszerror psz_predict_lorenzo(
    pszarray* in, pszrc2 const rc, pszarray* out_errquant,
    pszoutlier* out_outlier, void* stream)
try {
  if (out_errquant->dtype != U4)
    throw psz::exception_incorrect_type("errquant");

  float time_pred;
  auto l3 = in->len3;

  if (in->dtype == F4)
    psz_comp_l23r<f4, u4>(
        (f4*)in->buf, dim3(l3.x, l3.y, l3.z), rc.eb, rc.radius,
        (u4*)out_errquant->buf,
        new CompactDram<CUDA, f4>::Compact{
            (f4*)out_outlier->val, out_outlier->idx, out_outlier->num},
        &time_pred, stream);
  else if (in->dtype == F8)
    psz_comp_l23r<f8, u4>(
        (f8*)in->buf, dim3(l3.x, l3.y, l3.z), rc.eb, rc.radius,
        (u4*)out_errquant->buf,
        new CompactDram<CUDA, f8>::Compact{
            (f8*)out_outlier->val, out_outlier->idx, out_outlier->num},
        &time_pred, stream);
  else
    throw psz::exception_incorrect_type("input");

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)