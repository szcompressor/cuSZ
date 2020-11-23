/**
 * @file ad_hoc_types.hh
 * @author Jiannan Tian
 * @brief To be superseded by type_trait.hh
 * @version 0.1.4
 * @date 2020-11-22
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_AD_HOC_TYPES_HH
#define CUSZ_AD_HOC_TYPES_HH

#include <cstdint>

// alias style 1
//typedef int8_t  int8__t;
//typedef uint8_t uint8__t;
typedef float   f32;
typedef double  f64;
// alias style 2
typedef int8_t             I1;
typedef int16_t            I2;
typedef int32_t            I4;
typedef long long          I8;
typedef uint8_t            UI1;
typedef uint16_t           UI2;
typedef uint32_t           UI4;
typedef unsigned long long UI8;
typedef float              FP4;
typedef double             FP8;
// just in case
typedef int64_t  I8_2;
typedef uint64_t UI8_2;

#endif
