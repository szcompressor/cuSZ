/**
 * @file hist_sp.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-05-18
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef F7EA723F_5393_438E_83B5_AF3B3B6F8227
#define F7EA723F_5393_438E_83B5_AF3B3B6F8227

#include <cstdint>

template <typename T, typename FQ = uint32_t>
int histsp(
    T* in, uint32_t inlen, FQ* out, uint32_t outlen,
    cudaStream_t stream = nullptr);

#endif /* F7EA723F_5393_438E_83B5_AF3B3B6F8227 */
