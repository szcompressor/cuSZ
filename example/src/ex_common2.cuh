/**
 * @file ex_common2.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-01-15
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef EX_COMMON2_CUH
#define EX_COMMON2_CUH

template <typename T>
struct printf_functor;

template <>
struct printf_functor<float> {
    __host__ __device__ void operator()(float x) { printf("%f ", x); }
};

template <>
struct printf_functor<double> {
    __host__ __device__ void operator()(double x) { printf("%lf ", x); }
};

template <>
struct printf_functor<int> {
    __host__ __device__ void operator()(int x) { printf("%d ", x); }
};

template <>
struct printf_functor<unsigned int> {
    __host__ __device__ void operator()(unsigned int x) { printf("%u ", x); }
};

#endif