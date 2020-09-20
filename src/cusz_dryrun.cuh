/**
 * @file cusz_dryrun.cuh
 * @author Jiannan Tian
 * @brief cuSZ dryrun mode, checking data quality from lossy compression (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-05-14
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <string>

namespace cusz {

namespace dryrun {
template <typename T>
__global__ void lorenzo_1d1l(T*, size_t*, double*);

template <typename T>
__global__ void lorenzo_2d1l(T*, size_t*, double*);

template <typename T>
__global__ void lorenzo_3d1l(T*, size_t*, double*);

}  // namespace dryrun

namespace workflow {

template <typename T>
void DryRun(T*, T*, std::string, size_t*, double*);

}
}  // namespace cusz
