/**
 * @file hist.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-02
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef D8B68EB9_A86B_4AEA_AD4C_3DF22827E7C3
#define D8B68EB9_A86B_4AEA_AD4C_3DF22827E7C3

#include <cstdint>

#include "cusz/type.h"

namespace psz {

template <pszpolicy Poilicy, typename T>
psz_error_status histogram(
    T* in, size_t const inlen, uint32_t* out_hist, int const outlen,
    float* milliseconds, void* stream = nullptr);

}  // namespace psz

#endif /* D8B68EB9_A86B_4AEA_AD4C_3DF22827E7C3 */
