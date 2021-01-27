#ifndef CUSZ_DRYRUN_CUH
#define CUSZ_DRYRUN_CUH

/**
 * @file cusz_dryrun.cuh
 * @author Jiannan Tian
 * @brief cuSZ dryrun mode, checking data quality from lossy compression (header).
 * @version 0.2
 * @date 2020-09-20
 * (create) 2020-05-14, (release) 2020-09-20, (rev1) 2021-01-25
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <string>
#include "argparse.hh"
#include "metadata.hh"

// clang-format off
namespace cusz { namespace dryrun {
template <typename Data> __global__ void lorenzo_1d1l(lorenzo_dryrun, Data*);
template <typename Data> __global__ void lorenzo_2d1l(lorenzo_dryrun, Data*);
template <typename Data> __global__ void lorenzo_3d1l(lorenzo_dryrun, Data*);
}}
// clang-format on

#endif