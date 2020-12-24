#ifndef VERIFY_HH
#define VERIFY_HH

/**
 * @file verify.cc
 * @author Jiannan Tian
 * @brief Verification of decompressed data (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2019-09-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstddef>
#include <cstdint>
#include "types.hh"

namespace analysis {

template <typename T>
void VerifyData(stat_t* stat, T* xData, T* oData, size_t _len);

void PrintMetrics(
    stat_t* stat,
    int     type_byte,
    bool    override_eb  = false,
    double  new_eb       = 0,
    size_t  archive_byte = 0,
    size_t  bin_scale    = 1);

}  // namespace analysis

#endif
