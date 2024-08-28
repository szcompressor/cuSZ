#include "cusz/type.h"
#include "port.hh"
#include "stat/compare.hh"
#include "utils/verify.hh"
#include "utils/viewer.hh"

using namespace portable;

template <typename T, psz_policy P>
void pszcxx_evaluate_quality_gpu(
    T* xdata, T* odata, size_t len, size_t comp_bytes)
{
  // cross
  auto stat_x = new psz_summary;
  psz::utils::assess_quality<P, T>(stat_x, xdata, odata, len);
  psz::utils::print_metrics_cross<T>(stat_x, comp_bytes, true);

  auto stat_auto_lag1 = new psz_summary;
  psz::utils::assess_quality<P, T>(stat_auto_lag1, odata, odata + 1, len - 1);
  auto stat_auto_lag2 = new psz_summary;
  psz::utils::assess_quality<P, T>(stat_auto_lag2, odata, odata + 2, len - 2);

  psz::utils::print_metrics_auto(
      &stat_auto_lag1->score_coeff, &stat_auto_lag2->score_coeff);

  delete stat_x, delete stat_auto_lag1, delete stat_auto_lag2;
}

template <typename T, psz_policy P>
pszerror pszcxx_evaluate_quality_gpu(array3<T> xdata, array3<T> odata)
{
  pszcxx_evaluate_quality_gpu<T, P>(
      (T*)xdata.buf, (T*)odata.buf, xdata.len3.x);

  return CUSZ_SUCCESS;
}
