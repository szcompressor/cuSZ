#include "cusz/type.h"
#include "detail/compare.hh"
#include "detail/port.hh"
#include "utils/viewer.hh"

namespace psz::analysis {

template <typename T, psz_runtime P>
void GPU_evaluate_quality_and_print(T* xdata, T* odata, size_t len, size_t comp_bytes)
{
  // cross
  auto stat_x = new psz_statistics;
  psz::analysis::assess_quality<P, T>(stat_x, xdata, odata, len);
  psz::analysis::print_metrics_cross<T>(stat_x, comp_bytes, true);

  auto stat_auto_lag1 = new psz_statistics;
  psz::analysis::assess_quality<P, T>(stat_auto_lag1, odata, odata + 1, len - 1);
  auto stat_auto_lag2 = new psz_statistics;
  psz::analysis::assess_quality<P, T>(stat_auto_lag2, odata, odata + 2, len - 2);

  psz::analysis::print_metrics_auto(&stat_auto_lag1->score_coeff, &stat_auto_lag2->score_coeff);

  delete stat_x, delete stat_auto_lag1, delete stat_auto_lag2;
}

}  // namespace psz::analysis
