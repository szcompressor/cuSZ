#ifndef INTERNAL_CONST_HH
#define INTERNAL_CONST_HH

/**
 * @file constants.hh
 * @author Jiannan Tian
 * @brief Internal constants (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2019-06-08
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstddef>

// dims_L16
extern const size_t DIM0;
extern const size_t DIM1;
extern const size_t DIM2;
extern const size_t DIM3;
extern const size_t nBLK0;
extern const size_t nBLK1;
extern const size_t nBLK2;
extern const size_t nBLK3;
extern const size_t nDIM;
extern const size_t LEN;
extern const size_t CAP;
extern const size_t RADIUS;

// ebs_L4
extern const size_t EB;
extern const size_t EBr;
extern const size_t EBx2;
extern const size_t EBx2_r;

// threshold to call nvcomp
extern const size_t nvcompTHLD;

#endif
