/**
 * @file types.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2020-01-22
 * (created) 2019-06-08, (rev.1) 2020-09-20, (rev.2) 2021-01-22, (rev.3) 2021-09-18
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef TYPES_HH
#define TYPES_HH

#include <cstdint>

typedef struct Stat {
    double min_odata{}, max_odata{}, rng_odata{}, std_odata{};
    double min_xdata{}, max_xdata{}, rng_xdata{}, std_xdata{};
    double PSNR{}, MSE{}, NRMSE{};
    double coeff{};
    double user_set_eb{}, max_abserr_vs_rng{}, max_pwrrel_abserr{};

    size_t len{}, max_abserr_index{};
    double max_abserr{};

} stat_t;

#endif /* TYPES_HH */
