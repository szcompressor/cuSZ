/**
 * @file dbg_noarch.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.9
 * @date 2023-09-07
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B5D125C9_82B7_4611_B1F2_2A2CE5930AF5
#define B5D125C9_82B7_4611_B1F2_2A2CE5930AF5

/**
 * @brief Below only functionality is added defined, not involving in verbosity
 * control.
 *
 */

#define __PSZLOG__P(STR) cout << STR << endl;

#define __PSZLOG__NEWLINE printf("\n");

// clang-format off
#define __PSZLOG__STATUS_INFO              printf("[psz::info] ");
#define __PSZLOG__STATUS_INFO_IN(WHAT)     printf("[psz::info::%s] ", WHAT);
#define __PSZLOG__STATUS_DBG               printf("[psz::\e[31mdbg\e[0m] ");
#define __PSZLOG__STATUS_DBG_IN(WHAT)      printf("[psz::\e[31mdbg\e[0m::%s] ", WHAT);
#define __PSZLOG__STATUS_SANITIZE          printf("[psz::\e[31mdbg::sanitize\e[0m] ");
#define __PSZLOG__STATUS_SANITIZE_IN(WHAT) printf("[psz::\e[31mdbg::sanitize::%s\e[0m] ", WHAT);
// clang-format on

#define __PSZLOG__WHERE_CALLED \
  printf("(func) %s at \e[31m%s:%d\e[0m\n", __func__, __FILE__, __LINE__);

#define __PSZDBG__INFO(STR) \
  {                         \
    __PSZLOG__STATUS_DBG;   \
    cout << STR << endl;    \
  }

#define __PSZDBG__FATAL(CONST_CHAR) \
  throw std::runtime_error("\e[31m[psz::fatal]\e[0m " + string(CONST_CHAR));

#endif /* B5D125C9_82B7_4611_B1F2_2A2CE5930AF5 */
