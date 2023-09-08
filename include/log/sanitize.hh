/**
 * @file sanitize.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.9
 * @date 2023-09-07
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef DA7EED10_5E5D_4281_86EF_EFFFD232B95B
#define DA7EED10_5E5D_4281_86EF_EFFFD232B95B

#include <string>

#include "cusz/type.h"
#include "dbg.hh"

#ifdef PSZ_SANITIZE_ON

#define PSZSANITIZE_PSZCTX(...)                         \
  __PSZLOG__NEWLINE;                                    \
  __PSZLOG__STATUS_SANITIZE_IN("pszctx");               \
  __PSZLOG__WHERE_CALLED;                               \
  psz::sanitize<T, E, H>::sanitize_pszctx(__VA_ARGS__); \
  __PSZLOG__NEWLINE;

#define PSZSANITIZE_QUANTCODE(...)                         \
  __PSZLOG__NEWLINE;                                       \
  __PSZLOG__STATUS_SANITIZE_IN("quantcode");               \
  __PSZLOG__WHERE_CALLED;                                  \
  psz::sanitize<T, E, H>::sanitize_quantcode(__VA_ARGS__); \
  __PSZLOG__NEWLINE;

#define PSZSANITIZE_HIST_OUTPUT(...)                      \
  __PSZLOG__NEWLINE;                                      \
  __PSZLOG__STATUS_SANITIZE_IN("hist_out");               \
  __PSZLOG__WHERE_CALLED;                                 \
  psz::sanitize<T, E, H>::sanitize_hist_out(__VA_ARGS__); \
  __PSZLOG__NEWLINE;

#define PSZSANITIZE_HIST_BK(...)                           \
  __PSZLOG__NEWLINE;                                       \
  __PSZLOG__STATUS_SANITIZE("hist_and_bk");                \
  __PSZLOG__WHERE_CALLED;                                  \
  psz::sanitize<T, E, H>::sanitize_hist_book(__VA_ARGS__); \
  __PSZLOG__NEWLINE;

#else
#define PSZSANITIZE_PSZCTX(...)
#define PSZSANITIZE_QUANTCODE(...)
#define PSZSANITIZE_HIST_OUTPUT(...)
#define PSZSANITIZE_HIST_BK(...)
#endif

namespace psz {
template <typename T = f4, typename E = u4, typename H = u4>
struct sanitize {
  using M = u4;
  static void sanitize_pszctx(pszctx const* ctx, std::string LOC = "");
  static void sanitize_quantcode(E const* h_ectrl, szt len, szt bklen);
  static void sanitize_hist_book(M const* h_hist, H const* h_ectrl, szt bklen);
  static void sanitize_hist_out(M const* h_hist, szt bklen);

};  // namespace sanitize

}  // namespace psz

#endif /* DA7EED10_5E5D_4281_86EF_EFFFD232B95B */
