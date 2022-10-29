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

namespace accsz {

template <typename T>
void peek_device_data(T* d_arr, size_t num, size_t offset = 0)
{
    if (std::is_same<T, int8_t>::value) {  //
        peek_device_data_Ti8((int8_t*)d_arr, num, offset);
    }
    else if (std::is_same<T, int16_t>::value) {
        peek_device_data_Ti16((int16_t*)d_arr, num, offset);
    }
    else if (std::is_same<T, int32_t>::value) {
        peek_device_data_Ti32((int32_t*)d_arr, num, offset);
    }
    else if (std::is_same<T, int64_t>::value) {
        peek_device_data_Ti64((int64_t*)d_arr, num, offset);
    }
    else if (std::is_same<T, uint8_t>::value) {
        peek_device_data_Tui8((uint8_t*)d_arr, num, offset);
    }
    else if (std::is_same<T, uint16_t>::value) {
        peek_device_data_Tui16((uint16_t*)d_arr, num, offset);
    }
    else if (std::is_same<T, uint32_t>::value) {
        peek_device_data_Tui32((uint32_t*)d_arr, num, offset);
    }
    else if (std::is_same<T, uint64_t>::value) {
        peek_device_data_Tui64((uint64_t*)d_arr, num, offset);
    }
    else if (std::is_same<T, float>::value) {
        peek_device_data_Tfp32((float*)d_arr, num, offset);
    }
    else if (std::is_same<T, double>::value) {
        peek_device_data_Tfp64((double*)d_arr, num, offset);
    }
    else {
        std::runtime_error("peek_device_data cannot accept this type.");
    }
}

}  // namespace accsz

#undef PEEK_DEVICE_DATA
