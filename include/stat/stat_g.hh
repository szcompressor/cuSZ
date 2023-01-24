/**
 * @file stat_g.hh
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

#include <cuda_runtime.h>
#include "cusz/type.h"

namespace asz {
namespace stat {

/**
 * @brief Get frequency: a kernel wrapper
 *
 * @tparam T input type
 * @param in_data input device array
 * @param in_len input host var; len of in_data
 * @param out_freq output device array
 * @param nbin input host var; len of out_freq
 * @param milliseconds output time elapsed
 * @param stream optional stream
 */
template <typename T>
cusz_error_status histogram(
    T*           in_data,
    size_t const in_len,
    uint32_t*    out_freq,
    int const    nbin,
    float*       milliseconds,
    cudaStream_t stream = nullptr);

}  // namespace stat
}  // namespace asz

#endif /* D8B68EB9_A86B_4AEA_AD4C_3DF22827E7C3 */
