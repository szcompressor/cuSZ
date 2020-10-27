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

namespace analysis {

template <typename T>
void VerifyData(
    T*     xData,
    T*     oData,
    size_t _len,
    bool   override_eb       = false,
    double new_eb            = 0,
    size_t archive_byte_size = 0,
    size_t binning_scale     = 1);

}  // namespace analysis

#endif
