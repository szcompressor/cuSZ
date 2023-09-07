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

#ifdef PSZ_DBG_ON
#define PSZDBG_VAR(...) __PSZDBG_VAR(__VA_ARGS__)
#define PSZDBG_PTR_WHERE(VAR) __PSZDBG_PTR_WHERE(VAR)
#else
#define PSZDBG_VAR(...)
#define PSZDBG_PTR_WHERE(...)
#endif

#define __PSZDBG_VAR(LOC, VAR)                                       \
  __PSZLOG__STATUS_DBG_IN(LOC)                                       \
  printf(                                                            \
      "(var) " #VAR "=%ld\n        (func) %s at \e[31m%s:%d\e[0m\n", \
      (uint64_t)VAR, __func__, __FILE__, __LINE__);

#define __PSZDBG_PTR_WHERE(VAR)                                   \
  {                                                               \
    cudaPointerAttributes attributes;                             \
    cudaError_t err = cudaPointerGetAttributes(&attributes, VAR); \
    if (err != cudaSuccess) {                                     \
      __PSZLOG__STATUS_DBG                                        \
      cerr << "failed checking pointer attributes: "              \
           << cudaGetErrorString(err) << endl;                    \
      printf(                                                     \
          "        "                                              \
          "(func) %s at "                                         \
          "\e[31m"                                                \
          "%s:%d"                                                 \
          "\e[0m",                                                \
          __func__, __FILE__, __LINE__);                          \
      __PSZLOG__NEWLINE                                           \
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
      printf(                                                     \
          "        "                                              \
          "(func) %s at "                                         \
          "\e[31m"                                                \
          "%s:%d"                                                 \
          "\e[0m",                                                \
          __func__, __FILE__, __LINE__);                          \
      __PSZLOG__NEWLINE                                           \
    }                                                             \
  }

#endif /* B0D17E62_9CC5_4C2D_817C_F440519F4095 */
