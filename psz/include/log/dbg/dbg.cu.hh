/**
 * @file dbg_cu.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.9
 * @date 2023-09-07
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B0D17E62_9CC5_4C2D_817C_F440519F4095
#define B0D17E62_9CC5_4C2D_817C_F440519F4095

#define __PSZDBG_VAR(WHAT, VAR)               \
  __PSZLOG__STATUS_DBG_IN(WHAT)               \
  printf("(var) " #VAR "=%ld", (int64_t)VAR); \
  __PSZLOG__NEWLINE;

#define __PSZDBG_PTR_WHERE(VAR)                                   \
  {                                                               \
    cudaPointerAttributes attributes;                             \
    cudaError_t err = cudaPointerGetAttributes(&attributes, VAR); \
    if (err != cudaSuccess) {                                     \
      __PSZLOG__STATUS_DBG                                        \
      cerr << "failed checking pointer attributes: "              \
           << cudaGetErrorString(err) << endl;                    \
    }                                                             \
    else {                                                        \
      __PSZLOG__STATUS_DBG                                        \
      if (attributes.type == cudaMemoryTypeDevice)                \
        printf("(var) " #VAR " is on CUDA Device.");              \
      else if (attributes.type == cudaMemoryTypeDevice)           \
        printf("(var) " #VAR " is on Host.");                     \
      else {                                                      \
        printf("(var) " #VAR " is in another universe.");         \
      }                                                           \
      __PSZLOG__NEWLINE                                           \
    }                                                             \
  }

#endif /* B0D17E62_9CC5_4C2D_817C_F440519F4095 */
