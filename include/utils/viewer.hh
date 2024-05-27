/**
 * @file viewer.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-09
 * @deprecated 0.3.2
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef C6EF99AE_F0D7_485B_ADE4_8F55666CA96C
#define C6EF99AE_F0D7_485B_ADE4_8F55666CA96C

#include <algorithm>
#include <iomanip>

#include "cusz/type.h"
#include "header.h"
#include "mem/memseg_cxx.hh"
#include "port.hh"
#include "verify.hh"

// deps
#include "viewer/viewer.noarch.hh"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
#include "viewer/viewer.cu_hip.hh"
#elif defined(PSZ_USE_1API)
#include "viewer/viewer.dp.hh"
#endif

#endif /* C6EF99AE_F0D7_485B_ADE4_8F55666CA96C */
