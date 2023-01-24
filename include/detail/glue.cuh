/**
 * @file glue.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-03-01
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef WRAPPER_GLUE_CUH
#define WRAPPER_GLUE_CUH

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "spcodec.hh"

// when using nvcc, functors must be defined outside a (__host__) function
template <typename E>
struct cleanup : public thrust::unary_function<E, E> {
    int radius;
    cleanup(int radius) : radius(radius) {}
    __host__ __device__ E operator()(const E e) const { return e; }
};

template <typename E, typename Policy, typename IDX = int, bool SHIFT = true>
void split_by_radius(
    E*           in_errctrl,
    size_t       in_len,
    int const    radius,
    IDX*         out_idx,
    E*           out_val,
    int&         out_nnz,
    cudaStream_t stream = nullptr,
    Policy       policy = thrust::device)
{
    using thrust::placeholders::_1;

    thrust::cuda::par.on(stream);
    thrust::counting_iterator<IDX> zero(0);

    // find out the indices
    out_nnz = thrust::copy_if(policy, zero, zero + in_len, in_errctrl, out_idx, _1 >= 2 * radius or _1 <= 0) - out_idx;

    // fetch corresponding values
    thrust::copy(
        policy, thrust::make_permutation_iterator(in_errctrl, out_idx),
        thrust::make_permutation_iterator(in_errctrl + out_nnz, out_idx + out_nnz), out_val);

    // clear up
    cleanup<E> functor(radius);
    thrust::transform(
        policy,                                                                      //
        thrust::make_permutation_iterator(in_errctrl, out_idx),                      //
        thrust::make_permutation_iterator(in_errctrl + out_nnz, out_idx + out_nnz),  //
        thrust::make_permutation_iterator(in_errctrl, out_idx),                      //
        functor);
}

template <typename E, typename Policy, typename IDX = int>
void split_by_binary_twopass(
    E*           in_errctrl,
    size_t       in_len,
    int const    radius,
    IDX*         out_idx,
    E*           out_val,
    int&         out_nnz,
    cudaStream_t stream = nullptr,
    Policy       policy = thrust::device)
{
    using thrust::placeholders::_1;

    thrust::cuda::par.on(stream);
    thrust::counting_iterator<IDX> zero(0);

    // find out the indices
    out_nnz = thrust::copy_if(policy, zero, zero + in_len, in_errctrl, out_idx, _1 != radius) - out_idx;

    // fetch corresponding values
    thrust::copy(
        policy, thrust::make_permutation_iterator(in_errctrl, out_idx),
        thrust::make_permutation_iterator(in_errctrl + out_nnz, out_idx + out_nnz), out_val);
}

// when using nvcc, functors must be defined outside a (__host__) function
template <typename Tuple>
struct is_outlier {
    int radius;
    is_outlier(int radius) : radius(radius) {}
    __host__ __device__ bool operator()(const Tuple t) const { return thrust::get<1>(t) != radius; }
};

template <typename E, typename Policy, typename IDX = int>
void split_by_binary_onepass(
    E*           in_errctrl,
    size_t       in_len,
    int const    radius,
    IDX*         out_idx,
    E*           out_val,
    int&         out_nnz,
    cudaStream_t stream = nullptr,
    Policy       policy = thrust::device)
{
    thrust::cuda::par.on(stream);
    using Tuple = thrust::tuple<IDX, E>;
    thrust::counting_iterator<IDX> zero(0);

    auto in      = thrust::make_zip_iterator(thrust::make_tuple(zero, in_errctrl));
    auto in_last = thrust::make_zip_iterator(thrust::make_tuple(zero + in_len, in_errctrl + in_len));
    auto out     = thrust::make_zip_iterator(thrust::make_tuple(out_idx, out_val));

    is_outlier<Tuple> functor(radius);
    out_nnz = thrust::copy_if(policy, in, in_last, out, functor) - out;
}

enum class GlueMethod { SPLIT_BY_RADIUS, SPLIT_01_ONEPASS, SPLIT_01_TWOPASS };

#endif
