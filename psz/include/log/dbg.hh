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

#include "dbg/dbg.noarch.hh"

#if defined(PSZ_USE_CUDA)
#include "dbg/dbg.cu.hh"
#elif defined(PSZ_USE_HIP)
#elif defined(PSZ_USE_1API)
#endif

#ifdef PSZ_DBG_ON

#define PSZDBG_LOG(...) __PSZDBG__INFO(__VA_ARGS__)

#define PSZDBG_VAR(...)   \
  __PSZLOG__NEWLINE;      \
  __PSZLOG__STATUS_DBG    \
  __PSZLOG__WHERE_CALLED; \
  __PSZDBG_VAR(__VA_ARGS__);

#define PSZDBG_PTR_WHERE(VAR) \
  __PSZLOG__NEWLINE;          \
  __PSZLOG__STATUS_DBG        \
  __PSZLOG__WHERE_CALLED;     \
  __PSZDBG_PTR_WHERE(VAR)

#else
#define PSZDBG_LOG(...)
#define PSZDBG_VAR(...)
#define PSZDBG_PTR_WHERE(...)
#endif

#endif /* E37962D3_F63F_4392_B670_41E4B31915ED */
