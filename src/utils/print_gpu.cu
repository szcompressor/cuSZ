/**
 * @file print_gpu.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-23
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

// #include "../detail/print_gpu.inl"
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include "utils/print_gpu.h"

#define PRINT_INT_LESS_THAN_64(Tliteral, T)                                                                 \
    void peek_device_data_T##Tliteral(T* d_arr, size_t num, size_t offset)                                  \
    {                                                                                                       \
        thrust::for_each(                                                                                   \
            thrust::device, d_arr, d_arr + num, [=] __device__(const T i) { printf("%d\t", (int32_t)i); }); \
        printf("\n");                                                                                       \
    }

PRINT_INT_LESS_THAN_64(i8, int8_t)
PRINT_INT_LESS_THAN_64(i16, int16_t)
PRINT_INT_LESS_THAN_64(i32, int32_t)

void peek_device_data_Ti64(int64_t* d_arr, size_t num, size_t offset)
{
    thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__(const int64_t i) { printf("%ld\t", i); });
    printf("\n");
}

#define PRINT_UINT_LESS_THAN_64(Tliteral, T)                                                                 \
    void peek_device_data_T##Tliteral(T* d_arr, size_t num, size_t offset)                                   \
    {                                                                                                        \
        thrust::for_each(                                                                                    \
            thrust::device, d_arr, d_arr + num, [=] __device__(const T i) { printf("%u\t", (uint32_t)i); }); \
        printf("\n");                                                                                        \
    }

PRINT_UINT_LESS_THAN_64(ui8, uint8_t)
PRINT_UINT_LESS_THAN_64(ui16, uint16_t)
PRINT_UINT_LESS_THAN_64(ui32, uint32_t)

void peek_device_data_Tui64(uint64_t* d_arr, size_t num, size_t offset)
{
    thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__(const uint64_t i) { printf("%lu\t", i); });
    printf("\n");
}

void peek_device_data_Tfp32(float* d_arr, size_t num, size_t offset)
{
    thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__(const float i) { printf("%.7f\t", i); });
    printf("\n");
}

void peek_device_data_Tfp64(double* d_arr, size_t num, size_t offset)
{
    thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__(const double i) { printf("%.7lf\t", i); });
    printf("\n");
}

#undef PRINT_INT_LESS_THAN_64
#undef PRINT_UINT_LESS_THAN_64
