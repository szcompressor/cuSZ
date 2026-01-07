#ifndef EVAL_STAT_H
#define EVAL_STAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct psz_data_summary {
  double min, max, rng, std, avg;
} psz_data_summary;

typedef struct psz_statistics {
  psz_data_summary odata, xdata;
  double score_PSNR, score_MSE, score_NRMSE, score_coeff;
  double max_err_abs, max_err_rel, max_err_pwrrel;
  size_t max_err_idx;
  double autocor_lag_one, autocor_lag_two;
  double user_eb;
  size_t len;
} psz_stats;

#ifdef __cplusplus
}
#endif

#endif /* EVAL_STAT_H */
