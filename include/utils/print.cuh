/**
 * @file print.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-23
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef D79937FC_16EB_413C_AE31_27A8AACCCF46
#define D79937FC_16EB_413C_AE31_27A8AACCCF46

#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

template <typename T>
void __host__ __device__ typed_printf(const T val);

template <>
void __host__ __device__ typed_printf<uint32_t>(const uint32_t i)
{
    printf("%u\t", i);
}

template <>
void __host__ __device__ typed_printf<uint16_t>(const uint16_t i)
{
    printf("%u\t", (uint32_t)i);
}

template <>
void __host__ __device__ typed_printf<uint8_t>(const uint8_t i)
{
    printf("%u\t", (uint32_t)i);
}

template <>
void __host__ __device__ typed_printf<float>(const float i)
{
    printf("%.7f\t", i);
}

template <>
void typed_printf<double>(const double i)
{
    printf("%.7lf\t", i);
}

/* code snippet for looking at the device array easily */
template <typename T>
void __host__ __device__ peek_device_data(T* d_arr, size_t num, size_t offset = 0)
{
    thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const T i) { typed_printf<T>(i); });
    printf("\n");
};

#endif /* D79937FC_16EB_413C_AE31_27A8AACCCF46 */
