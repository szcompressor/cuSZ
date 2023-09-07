/**
 * @file dbg.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.9
 * @date 2023-09-07
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E37962D3_F63F_4392_B670_41E4B31915ED
#define E37962D3_F63F_4392_B670_41E4B31915ED

#ifdef PSZ_DBG_ON
#define PSZDBG_VAR(...) __PSZDBG_VAR(__VA_ARGS__)
#define PSZDBG_PTR_WHERE(VAR) __PSZDBG_PTR_WHERE(VAR)
#define PSZDBG_LOG(...) __PSZDBG__INFO(__VA_ARGS__)
#else
#define PSZDBG_VAR(...)
#define PSZDBG_PTR_WHERE(...)
#define PSZDBG_LOG(...)
#endif

#include "dbg_noarch.hh"

#if defined(PSZ_USE_CUDA)

#include "dbg_cu.hh"

#elif defined(PSZ_USE_HIP)

#elif defined(PSZ_USE_1API)

#endif

#endif /* E37962D3_F63F_4392_B670_41E4B31915ED */
