/**
 * @file print_gpu.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "print_gpu.h"

namespace psz {

template <typename T>
void peek_device_data(T* d_arr, size_t num, size_t offset = 0);

}  // namespace psz

#undef PEEK_DEVICE_DATA
