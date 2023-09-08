/**
 * @file base_compressor.cu
 * @author Jiannan Tian
 * @brief check quality for dual-quant
 * @version 0.3
 * @date 2021-10-05
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "cusz/type.h"
#include "dryrun.hh"
#include "kernel/detail/dryrun_cu.inl"

template void psz::cuda_hip_compat::dryrun(
    size_t len, f4* original, f4* reconst, double eb, void* stream);
template void psz::cuda_hip_compat::dryrun(
    size_t len, f8* original, f8* reconst, double eb, void* stream);