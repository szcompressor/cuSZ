/**
 * @file stat.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-02
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef BBBB5712_FF60_4262_B927_85B113FD26BA
#define BBBB5712_FF60_4262_B927_85B113FD26BA

#include "cusz/type.h"

#define HIST_C(Tname, T)                                                                                 \
    cusz_error_status histogram_T##Tname(                                                                \
        T* in_data, size_t const in_len, uint32_t* out_freq, int const num_buckets, float* milliseconds, \
        cudaStream_t stream);

HIST_C(ui8, uint8_t)
HIST_C(ui16, uint16_t)
HIST_C(ui32, uint32_t)
HIST_C(ui64, uint64_t)

#undef HIST_C

#endif /* BBBB5712_FF60_4262_B927_85B113FD26BA */
