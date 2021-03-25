/**
 * @file types.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2020-01-22
 * (created) 2019-06-08, (rev.1) 2020-09-20, (rev.2) 2021-01-22
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef TYPES_HH
#define TYPES_HH

using namespace std;

enum class placeholder { length_unknown, alloc_in_called_func, alloc_with_caution };

typedef struct Stat {
    double minimum{}, maximum{}, range{};
    double PSNR{}, MSE{}, NRMSE{};
    double coeff{};
    double user_set_eb{}, max_abserr_vs_range{}, max_pwr_rel_abserr{};

    size_t len{}, max_abserr_index{};
    double max_abserr{};

} stat_t;

// clang-format off
typedef struct Integer1  { int _0; }              Integer1;
typedef struct Integer2  { int _0, _1; }          Integer2;
typedef struct Integer3  { int _0, _1, _2; }      Integer3;
typedef struct Integer4  { int _0, _1, _2, _3; }  Integer4;
typedef struct UInteger1 { int _0; }             UInteger1;
typedef struct UInteger2 { int _0, _1; }         UInteger2;
typedef struct UInteger3 { int _0, _1, _2; }     UInteger3;
typedef struct UInteger4 { int _0, _1, _2, _3; } UInteger4;
// clang-format on

#endif /* TYPES_HH */
