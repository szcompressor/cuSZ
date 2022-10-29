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
void peek_device_data(T* d_arr, size_t num, size_t offset = 0);
}

#define PEEK_DEVICE_DATA(Tliteral, T)                                     \
    template <>                                                           \
    void accsz::peek_device_data<T>(T * d_arr, size_t num, size_t offset) \
    {                                                                     \
        peek_device_data_T##Tliteral(d_arr, num, offset);                 \
    }

PEEK_DEVICE_DATA(i8, int8_t)
PEEK_DEVICE_DATA(i16, int16_t)
PEEK_DEVICE_DATA(i32, int32_t)
PEEK_DEVICE_DATA(i64, int64_t)
PEEK_DEVICE_DATA(ui8, uint8_t)
PEEK_DEVICE_DATA(ui16, uint16_t)
PEEK_DEVICE_DATA(ui32, uint32_t)
PEEK_DEVICE_DATA(ui64, uint64_t)
PEEK_DEVICE_DATA(fp32, float)
PEEK_DEVICE_DATA(fp64, double)

#undef PEEK_DEVICE_DATA
